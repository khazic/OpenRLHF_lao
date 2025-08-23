import os
import tempfile
from datasets import load_dataset, concatenate_datasets, load_from_disk
from transformers import AutoTokenizer
from typing import Dict, Any, Generator
import gc
from tqdm import tqdm

class DatasetProcessor:
    def __init__(self, args):
        self.args = args
        self.tokenizer = self._setup_tokenizer()
        self.temp_dir = tempfile.mkdtemp()
    
    def _setup_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.args.tokenizer_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer
    
    def process_shard(self, shard_idx: int, num_shards: int) -> str:
        dataset = load_dataset(
            'json', 
            data_files=self.args.dataset_path,
            split=f'train[{shard_idx}%{num_shards}]'
        )
        
        processed_shard = self._process_dataset_chunk(dataset)
        
        shard_path = os.path.join(self.temp_dir, f"shard_{shard_idx}")
        processed_shard.save_to_disk(shard_path)
        
        del dataset, processed_shard
        gc.collect()
        
        return shard_path
    
    def _process_dataset_chunk(self, dataset):
        return dataset.map(
            self._preprocess_function,
            batched=True,
            batch_size=self.args.batch_size,
            num_proc=self.args.num_proc,
            remove_columns=dataset.column_names,
            desc="Processing chunk",
            load_from_cache_file=False
        )
    
    def _preprocess_function(self, examples: Dict[str, Any]) -> Dict[str, list]:
        try:
            if self.args.apply_chat_template:
                prompts = []
                responses = []
                for i in range(len(examples[self.args.input_key])):
                    prompt_message = examples[self.args.input_key][i]
                    response_message = examples[self.args.output_key][i]
                    
                    if isinstance(prompt_message, str):
                        prompt_message = [{"role": "user", "content": prompt_message}]
                        response_message = [{"role": "assistant", "content": response_message}]
                    
                    prompt = self.tokenizer.apply_chat_template(prompt_message, tokenize=False, add_generation_prompt=True)
                    full_conversation = self.tokenizer.apply_chat_template(prompt_message + response_message, tokenize=False)
                    response = full_conversation[len(prompt):]
                    
                    total_tokens = len(self.tokenizer.encode(prompt + response))
                    if total_tokens > self.args.max_length:
                        continue
                    
                    prompts.append(prompt)
                    responses.append(response)
            else:
                prompts = examples[self.args.input_key]
                responses = examples[self.args.output_key]
            
            if not prompts:
                return {"input_ids": [], "attention_mask": [], "loss_mask": []}
            
            tokenized_inputs = self.tokenizer(
                prompts,
                responses,
                max_length=self.args.max_length,
                truncation=True,
                padding=False,
                return_tensors=None
            )
            
            loss_masks = []
            for i in range(len(tokenized_inputs["input_ids"])):
                prompt_ids = self.tokenizer(prompts[i], return_tensors="pt")["input_ids"]
                prompt_len = len(prompt_ids[0])
                
                loss_mask = [0] * prompt_len + [1] * (len(tokenized_inputs["input_ids"][i]) - prompt_len)
                loss_masks.append(loss_mask)
            
            tokenized_inputs["loss_mask"] = loss_masks
            return tokenized_inputs
        except Exception as e:
            print(f"Error processing batch: {str(e)}")
            return {"input_ids": [], "attention_mask": [], "loss_mask": []}
    
    def process_and_save(self):
        try:
            num_shards = max(1, self.args.num_proc // 2)
            
            shard_paths = []
            for shard_idx in range(num_shards):
                shard_path = self.process_shard(shard_idx, num_shards)
                shard_paths.append(shard_path)
            
            print("Merging shards...")
            final_dataset = concatenate_datasets([
                load_from_disk(path) for path in shard_paths
            ])
            
            print(f"Saving final dataset to {self.args.save_path}")
            final_dataset.save_to_disk(self.args.save_path)
            
        finally:
            import shutil
            shutil.rmtree(self.temp_dir)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--tokenizer_path", type=str, required=True)
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--input_key", type=str, default="question")
    parser.add_argument("--output_key", type=str, default="response")
    parser.add_argument("--apply_chat_template", action="store_true")
    parser.add_argument("--num_proc", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=1000)
    args = parser.parse_args()
    
    print(f"Loading dataset from {args.dataset_path}")
    if args.dataset_path.endswith('.json') or args.dataset_path.endswith('.jsonl'):
        dataset = load_dataset('json', data_files=args.dataset_path)['train']
    else:
        dataset = load_dataset(args.dataset_path)['train']
    
    print(f"Loading tokenizer from {args.tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("Processing dataset...")
    processor = DatasetProcessor(args)
    processor.process_and_save()

if __name__ == "__main__":
    main()