import argparse
import json
import math
import os
from datetime import timedelta

import torch
from torch import distributed as dist
from tqdm import tqdm

from openrlhf.datasets import SFTDataset
from openrlhf.datasets.utils import blending_datasets
from openrlhf.models import Actor, SFTLoss
from openrlhf.utils import get_strategy, get_tokenizer


def calculate_perplexity_per_sample(args):
    """
    计算每个样本的困惑度(Perplexity)，用于数据筛选
    """
    # 配置策略
    strategy = get_strategy(args)
    strategy.setup_distributed(timeout=timedelta(minutes=180))

    # 加载模型
    model = Actor(
        args.pretrain,
        attn_implementation=args.attn_implementation,
        bf16=args.bf16,
        load_in_4bit=args.load_in_4bit,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=args.target_modules,
        lora_dropout=args.lora_dropout,
    )

    # 配置tokenizer
    tokenizer = get_tokenizer(
        args.pretrain, model.model, "right", strategy, use_fast=not args.disable_fast_tokenizer
    )

    # 准备模型
    model = strategy.prepare(model)
    model.eval()

    # 加载原始数据集（用于保存时获取原始内容）
    raw_dataset = blending_datasets(
        args.dataset,
        args.dataset_probs,
        strategy,
        args.seed,
        max_count=args.max_samples,
        dataset_split=args.dataset_split,
    )
    raw_dataset = raw_dataset.select(range(min(args.max_samples, len(raw_dataset))))
    
    # 加载SFT数据集
    sft_dataset = SFTDataset(
        raw_dataset,
        tokenizer,
        args.max_len,
        strategy,
        pretrain_mode=args.pretrain_mode,
        input_template=args.input_template,
        multiturn=args.multiturn,
    )

    # 准备dataloader - batch_size设为1，这样可以单独计算每个样本
    dataloader = strategy.setup_dataloader(
        sft_dataset,
        1,  # 每次处理一个样本
        True,
        False,
        sft_dataset.collate_fn,
        drop_last=False,
    )

    # 初始化loss函数
    loss_fn = SFTLoss()

    # 计算每个样本的困惑度
    dist.barrier()
    
    results = []
    sample_idx = 0

    pbar = tqdm(
        dataloader,
        desc="Calculating PPL per sample",
        disable=not strategy.is_rank_0(),
    )

    with torch.no_grad():
        for inputs, attention_masks, loss_masks in pbar:
            inputs = inputs.to(torch.cuda.current_device()).squeeze(1)
            attention_mask = attention_masks.to(torch.cuda.current_device()).squeeze(1)
            loss_mask = loss_masks.to(torch.cuda.current_device()).squeeze(1)

            # 获取log probabilities
            per_token_log_probs = model(
                inputs,
                attention_mask=attention_mask,
                return_logprobs=True,
            )

            # 计算该样本的loss
            loss = loss_fn(per_token_log_probs, loss_mask[:, :-1])
            
            # 计算该样本的token数（有效token，不包括padding）
            num_tokens = loss_mask[:, :-1].sum().item()
            
            # 计算PPL
            ppl = math.exp(loss.item())
            
            # 获取原始数据
            raw_data = raw_dataset[sample_idx]
            
            # 保存结果
            result = {
                "index": sample_idx,
                "ppl": round(ppl, 4),
                "loss": round(loss.item(), 4),
                "num_tokens": int(num_tokens),
                args.input_key: raw_data[args.input_key],
                args.output_key: raw_data[args.output_key],
            }
            results.append(result)
            
            sample_idx += 1
            
            # 更新进度条
            pbar.set_postfix({
                "current_ppl": f"{ppl:.2f}",
                "avg_ppl": f"{sum(r['ppl'] for r in results) / len(results):.2f}"
            })

    # 收集所有GPU的结果
    if strategy.is_rank_0():
        # 按PPL排序
        results_sorted_by_ppl = sorted(results, key=lambda x: x['ppl'])
        
        # 计算统计信息
        ppls = [r['ppl'] for r in results]
        avg_ppl = sum(ppls) / len(ppls)
        min_ppl = min(ppls)
        max_ppl = max(ppls)
        median_ppl = sorted(ppls)[len(ppls) // 2]
        
        print("\n" + "="*70)
        print(f"数据集: {args.dataset}")
        print(f"总样本数: {len(results)}")
        print(f"平均PPL: {avg_ppl:.2f}")
        print(f"中位数PPL: {median_ppl:.2f}")
        print(f"最小PPL: {min_ppl:.2f}")
        print(f"最大PPL: {max_ppl:.2f}")
        print("="*70)
        
        # 如果设置了阈值，显示筛选信息
        if args.ppl_threshold:
            kept_samples = [r for r in results if r['ppl'] <= args.ppl_threshold]
            removed_samples = [r for r in results if r['ppl'] > args.ppl_threshold]
            
            print(f"\n📊 使用阈值 PPL <= {args.ppl_threshold} 筛选:")
            print(f"  ✅ 保留样本: {len(kept_samples)} ({len(kept_samples)/len(results)*100:.1f}%)")
            print(f"  ❌ 删除样本: {len(removed_samples)} ({len(removed_samples)/len(results)*100:.1f}%)")
            print("="*70)
        
        # 保存所有样本的PPL结果
        output_all = args.output_file.replace('.json', '_all.json')
        with open(output_all, 'w', encoding='utf-8') as f:
            json.dump(results_sorted_by_ppl, f, ensure_ascii=False, indent=2)
        print(f"\n💾 所有样本的PPL已保存到: {output_all}")
        
        # 如果设置了阈值，保存筛选后的数据
        if args.ppl_threshold:
            # 保存高质量数据（低PPL）
            output_kept = args.output_file.replace('.json', '_filtered.json')
            kept_data = [
                {args.input_key: r[args.input_key], args.output_key: r[args.output_key]} 
                for r in results if r['ppl'] <= args.ppl_threshold
            ]
            with open(output_kept, 'w', encoding='utf-8') as f:
                json.dump(kept_data, f, ensure_ascii=False, indent=2)
            print(f"💾 筛选后的高质量数据已保存到: {output_kept}")
            
            # 保存被删除的数据（高PPL）
            output_removed = args.output_file.replace('.json', '_removed.json')
            removed_data = [
                {args.input_key: r[args.input_key], args.output_key: r[args.output_key], "ppl": r['ppl']} 
                for r in results if r['ppl'] > args.ppl_threshold
            ]
            with open(output_removed, 'w', encoding='utf-8') as f:
                json.dump(removed_data, f, ensure_ascii=False, indent=2)
            print(f"💾 被删除的低质量数据已保存到: {output_removed}")
        
        # 显示PPL最高和最低的样本
        print(f"\n🔝 PPL最低的3个样本（高质量）:")
        for i, r in enumerate(results_sorted_by_ppl[:3], 1):
            question_preview = r[args.input_key][:100] + "..." if len(r[args.input_key]) > 100 else r[args.input_key]
            print(f"  {i}. PPL={r['ppl']:.2f}, Tokens={r['num_tokens']}")
            print(f"     {question_preview}\n")
        
        print(f"🔻 PPL最高的3个样本（低质量）:")
        for i, r in enumerate(results_sorted_by_ppl[-3:], 1):
            question_preview = r[args.input_key][:100] + "..." if len(r[args.input_key]) > 100 else r[args.input_key]
            print(f"  {i}. PPL={r['ppl']:.2f}, Tokens={r['num_tokens']}")
            print(f"     {question_preview}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="计算每个样本的PPL，用于数据筛选")
    
    # 模型参数
    parser.add_argument("--pretrain", type=str, required=True, help="模型路径")
    parser.add_argument("--bf16", action="store_true", default=False, help="使用bfloat16")
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="flash_attention_2",
        help="注意力实现方式",
    )
    parser.add_argument("--disable_fast_tokenizer", action="store_true", default=False)
    
    # LoRA参数
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--lora_rank", type=int, default=0)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--target_modules", type=str, nargs="*", default="all-linear")
    parser.add_argument("--lora_dropout", type=float, default=0)
    
    # 数据集参数
    parser.add_argument("--dataset", type=str, required=True, help="数据集路径")
    parser.add_argument("--dataset_probs", type=str, default=None)
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument("--max_samples", type=int, default=1000000)
    parser.add_argument("--max_len", type=int, default=2048)
    parser.add_argument("--input_key", type=str, default="input", help="输入字段名称")
    parser.add_argument("--output_key", type=str, default="output", help="输出字段名称")
    parser.add_argument("--input_template", type=str, default=None, help="输入模板")
    parser.add_argument("--pretrain_mode", action="store_true", default=False)
    parser.add_argument("--multiturn", action="store_true", default=False)
    parser.add_argument("--apply_chat_template", action="store_true", default=False)
    
    # 筛选参数
    parser.add_argument(
        "--ppl_threshold", 
        type=float, 
        default=None, 
        help="PPL阈值，高于此值的样本将被标记为低质量（例如: 10.0）"
    )
    
    # 其他参数
    parser.add_argument("--micro_batch_size", type=int, default=1)  # 固定为1
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--zero_stage", type=int, default=0)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--full_determinism", action="store_true", default=False)
    parser.add_argument("--output_file", type=str, default="ppl_per_sample.json", help="输出文件路径")
    
    args = parser.parse_args()
    
    calculate_perplexity_per_sample(args)

