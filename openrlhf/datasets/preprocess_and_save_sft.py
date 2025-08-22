import os
import json
import pickle
import argparse
from typing import Dict, List, Any
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer
import torch
from tqdm import tqdm
import gc

class SFTDataPreprocessor:
    def __init__(
        self,
        tokenizer_path: str,
        max_length: int = 4096,
        input_key: str = "question",
        output_key: str = "response",
        apply_chat_template: bool = True,
        batch_size: int = 1000,
        save_format: str = "pickle"
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.max_length = max_length
        self.input_key = input_key
        self.output_key = output_key
        self.apply_chat_template = apply_chat_template
        self.batch_size = batch_size
        self.save_format = save_format
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def preprocess_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            prompt = data[self.input_key]
            response = data[self.output_key]
            
            if not prompt or not response:
                return None
            
            if self.apply_chat_template:
                prompt_message = [{"role": "user", "content": prompt}]
                response_message = [{"role": "assistant", "content": response}]
                
                prompt = self.tokenizer.apply_chat_template(
                    prompt_message, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
                response = self.tokenizer.apply_chat_template(
                    prompt_message + response_message, 
                    tokenize=False
                )[len(prompt):]
            
            prompt_tokens = self.tokenizer(
                prompt,
                max_length=self.max_length,
                padding=False,
                truncation=True,
                return_tensors="pt",
                add_special_tokens=False,
            )
            prompt_ids_len = prompt_tokens["attention_mask"].int().sum().item()
            
            if prompt_ids_len >= self.max_length - 2:
                return None
            
            text = (prompt + response).rstrip("\n")
            if not text.endswith(self.tokenizer.eos_token):
                text += " " + self.tokenizer.eos_token
            
            input_tokens = self.tokenizer(
                text,
                max_length=self.max_length,
                padding=False,
                truncation=True,
                return_tensors="pt",
                add_special_tokens=False,
            )
            
            input_ids = input_tokens["input_ids"][0]
            attention_mask = input_tokens["attention_mask"][0]
            
            loss_mask = torch.zeros_like(input_ids, dtype=torch.float32)
            loss_mask[prompt_ids_len - 1 : -1] = 1
            
            if len(input_ids) > 0:
                input_ids[-1] = self.tokenizer.eos_token_id
                attention_mask[-1] = True
            
            return {
                "input_ids": input_ids.tolist(),
                "attention_mask": attention_mask.tolist(),
                "loss_mask": loss_mask.tolist(),
                "prompt_ids_len": prompt_ids_len,
                "prompt": prompt,
                "response": response
            }
            
        except Exception as e:
            print(f"Error processing sample: {e}")
            return None
    
    def process_and_save_dataset(self, dataset: Dataset, save_dir: str):
        print(f"Processing dataset with {len(dataset)} samples...")
        
        os.makedirs(save_dir, exist_ok=True)
        
        total_batches = (len(dataset) + self.batch_size - 1) // self.batch_size
        processed_count = 0
        failed_count = 0
        
        for batch_idx in tqdm(range(total_batches), desc="Processing batches"):
            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, len(dataset))
            
            batch_data = dataset.select(range(start_idx, end_idx))
            processed_batch = []
            
            for item in batch_data:
                processed_item = self.preprocess_data(item)
                if processed_item is not None:
                    processed_batch.append(processed_item)
                else:
                    failed_count += 1
            
            if processed_batch:
                batch_filename = f"batch_{batch_idx:06d}.{self.save_format}"
                batch_path = os.path.join(save_dir, batch_filename)
                
                if self.save_format == "pickle":
                    with open(batch_path, 'wb') as f:
                        pickle.dump(processed_batch, f)
                else:
                    with open(batch_path, 'w', encoding='utf-8') as f:
                        json.dump(processed_batch, f, ensure_ascii=False, indent=2)
                
                processed_count += len(processed_batch)
                print(f"Batch {batch_idx + 1}/{total_batches} completed, processed {len(processed_batch)} samples")
            
            del batch_data, processed_batch
            gc.collect()
        
        metadata = {
            "total_samples": len(dataset),
            "processed_samples": processed_count,
            "failed_samples": failed_count,
            "batch_size": self.batch_size,
            "max_length": self.max_length,
            "input_key": self.input_key,
            "output_key": self.output_key,
            "apply_chat_template": self.apply_chat_template,
            "save_format": self.save_format,
            "total_batches": total_batches
        }
        
        metadata_path = os.path.join(save_dir, "metadata.json")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        print(f"\nProcessing completed!")
        print(f"Total samples: {len(dataset)}")
        print(f"Successfully processed: {processed_count}")
        print(f"Failed samples: {failed_count}")
        print(f"Save directory: {save_dir}")
        print(f"Total batches: {total_batches}")

def main():
    parser = argparse.ArgumentParser(description="Preprocess SFT dataset and save to local")
    parser.add_argument("--dataset_path", type=str, required=True, help="Dataset path")
    parser.add_argument("--tokenizer_path", type=str, required=True, help="Tokenizer path")
    parser.add_argument("--save_dir", type=str, required=True, help="Save directory")
    parser.add_argument("--max_length", type=int, default=4096, help="Max sequence length")
    parser.add_argument("--input_key", type=str, default="question", help="Input field name")
    parser.add_argument("--output_key", type=str, default="response", help="Output field name")
    parser.add_argument("--apply_chat_template", action="store_true", help="Apply chat template")
    parser.add_argument("--batch_size", type=int, default=1000, help="Batch size")
    parser.add_argument("--save_format", type=str, default="pickle", choices=["pickle", "json"], help="Save format")
    
    args = parser.parse_args()
    
    print(f"Loading dataset: {args.dataset_path}")
    if args.dataset_path.endswith('.json'):
        dataset = load_dataset('json', data_files=args.dataset_path)['train']
    else:
        dataset = load_dataset(args.dataset_path)['train']
    
    preprocessor = SFTDataPreprocessor(
        tokenizer_path=args.tokenizer_path,
        max_length=args.max_length,
        input_key=args.input_key,
        output_key=args.output_key,
        apply_chat_template=args.apply_chat_template,
        batch_size=args.batch_size,
        save_format=args.save_format
    )
    
    preprocessor.process_and_save_dataset(dataset, args.save_dir)

if __name__ == "__main__":
    main()