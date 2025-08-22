import os
from datasets import load_dataset
from transformers import AutoTokenizer
from typing import Dict, Any
import torch
from tqdm import tqdm

def preprocess_function(examples: Dict[str, Any], tokenizer, max_length: int, input_key: str, output_key: str, apply_chat_template: bool):
    try:
        if apply_chat_template:
            prompts = []
            responses = []
            for i in range(len(examples[input_key])):
                prompt_message = examples[input_key][i]
                response_message = examples[output_key][i]
                
                if isinstance(prompt_message, str):
                    prompt_message = [{"role": "user", "content": prompt_message}]
                    response_message = [{"role": "assistant", "content": response_message}]
                
                prompt = tokenizer.apply_chat_template(prompt_message, tokenize=False, add_generation_prompt=True)
                full_conversation = tokenizer.apply_chat_template(prompt_message + response_message, tokenize=False)
                response = full_conversation[len(prompt):]
                
                # 检查总长度
                total_tokens = len(tokenizer.encode(prompt + response))
                if total_tokens > max_length:
                    # 跳过过长的样本
                    continue
                
                prompts.append(prompt)
                responses.append(response)
        else:
            prompts = examples[input_key]
            responses = examples[output_key]
        
        if not prompts:  # 如果所有样本都被过滤掉了
            return {"input_ids": [], "attention_mask": [], "loss_mask": []}
        
        # 对话模板处理后的tokenization
        tokenized_inputs = tokenizer(
            prompts,
            responses,
            max_length=max_length,
            truncation=True,
            padding=False,
            return_tensors=None
        )
        
        # 创建loss mask
        loss_masks = []
        for i in range(len(tokenized_inputs["input_ids"])):
            prompt_ids = tokenizer(prompts[i], return_tensors="pt")["input_ids"]
            prompt_len = len(prompt_ids[0])
            
            loss_mask = [0] * prompt_len + [1] * (len(tokenized_inputs["input_ids"][i]) - prompt_len)
            loss_masks.append(loss_mask)
        
        tokenized_inputs["loss_mask"] = loss_masks
        return tokenized_inputs
    except Exception as e:
        print(f"Error processing batch: {str(e)}")
        return {"input_ids": [], "attention_mask": [], "loss_mask": []}

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
    processed_dataset = dataset.map(
        lambda x: preprocess_function(
            x,
            tokenizer=tokenizer,
            max_length=args.max_length,
            input_key=args.input_key,
            output_key=args.output_key,
            apply_chat_template=args.apply_chat_template
        ),
        batched=True,
        batch_size=100,  # 减小batch_size
        num_proc=32,     # 减小进程数
        remove_columns=dataset.column_names,
        desc="Processing dataset",
        load_from_cache_file=False,  # 禁用缓存
        writer_batch_size=1000  # 控制写入批次大小
    )
    
    print(f"Saving processed dataset to {args.save_path}")
    processed_dataset.save_to_disk(args.save_path)
    print("Done!")

if __name__ == "__main__":
    main()