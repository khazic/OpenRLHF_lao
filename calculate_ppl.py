import argparse
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


def calculate_perplexity(args):
    """
    计算SFT数据集的困惑度(Perplexity)
    PPL = exp(average_loss)
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

    # 加载数据集
    dataset = blending_datasets(
        args.dataset,
        args.dataset_probs,
        strategy,
        args.seed,
        max_count=args.max_samples,
        dataset_split=args.dataset_split,
    )
    dataset = dataset.select(range(min(args.max_samples, len(dataset))))
    
    sft_dataset = SFTDataset(
        dataset,
        tokenizer,
        args.max_len,
        strategy,
        pretrain_mode=args.pretrain_mode,
        input_template=args.input_template,
        multiturn=args.multiturn,
    )

    # 准备dataloader
    dataloader = strategy.setup_dataloader(
        sft_dataset,
        args.micro_batch_size,
        True,
        False,
        sft_dataset.collate_fn,
        drop_last=False,
    )

    # 初始化loss函数
    loss_fn = SFTLoss()

    # 计算困惑度
    dist.barrier()
    
    total_loss = 0
    total_tokens = 0
    num_batches = 0

    pbar = tqdm(
        dataloader,
        desc="Calculating Perplexity",
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

            # 计算loss
            loss = loss_fn(per_token_log_probs, loss_mask[:, :-1])
            
            # 累计loss和token数
            batch_tokens = loss_mask[:, :-1].sum().item()
            total_loss += loss.item() * num_batches if num_batches == 0 else loss.item()
            total_tokens += batch_tokens
            num_batches += 1

            # 计算当前的平均loss和PPL
            avg_loss = total_loss / num_batches
            ppl = math.exp(avg_loss)

            pbar.set_postfix({
                "loss": f"{avg_loss:.4f}",
                "ppl": f"{ppl:.2f}",
                "tokens": total_tokens
            })

    # 在所有进程间同步结果
    metrics = {
        "total_loss": total_loss,
        "num_batches": num_batches,
        "total_tokens": total_tokens,
    }
    
    # 使用all_reduce聚合所有GPU的结果
    metrics = strategy.all_reduce(metrics)
    
    # 计算最终的PPL
    final_avg_loss = metrics["total_loss"] / metrics["num_batches"]
    final_ppl = math.exp(final_avg_loss)
    
    if strategy.is_rank_0():
        print("\n" + "="*50)
        print(f"数据集: {args.dataset}")
        print(f"样本数: {len(dataset)}")
        print(f"总Token数: {int(metrics['total_tokens'])}")
        print(f"平均Loss: {final_avg_loss:.4f}")
        print(f"困惑度(PPL): {final_ppl:.2f}")
        print("="*50)
        
        # 保存结果到文件
        if args.output_file:
            with open(args.output_file, "w") as f:
                f.write(f"Dataset: {args.dataset}\n")
                f.write(f"Model: {args.pretrain}\n")
                f.write(f"Samples: {len(dataset)}\n")
                f.write(f"Total Tokens: {int(metrics['total_tokens'])}\n")
                f.write(f"Average Loss: {final_avg_loss:.4f}\n")
                f.write(f"Perplexity: {final_ppl:.2f}\n")
            print(f"\n结果已保存到: {args.output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="计算SFT数据集的困惑度(PPL)")
    
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
    parser.add_argument("--input_key", type=str, default="input", help="输入字段名称（默认: input）")
    parser.add_argument("--output_key", type=str, default="output", help="输出字段名称（默认: output）")
    parser.add_argument("--input_template", type=str, default=None, help="输入模板（例如: 'User: {}\\nAssistant: '）")
    parser.add_argument("--pretrain_mode", action="store_true", default=False)
    parser.add_argument("--multiturn", action="store_true", default=False)
    parser.add_argument(
        "--apply_chat_template", action="store_true", default=False
    )
    
    # 其他参数
    parser.add_argument("--micro_batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--zero_stage", type=int, default=0, help="DeepSpeed ZeRO stage")
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for deepspeed")
    parser.add_argument(
        "--full_determinism",
        action="store_true",
        default=False,
    )
    parser.add_argument("--output_file", type=str, default=None, help="结果输出文件路径")
    
    args = parser.parse_args()
    
    calculate_perplexity(args)

