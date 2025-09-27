#!/usr/bin/env python3
"""
Intensive GPU Keep-Alive Script
高强度占用GPU显存和计算资源，确保GPU利用率保持高水平
"""

import torch
import time
import argparse
import threading
import numpy as np
from datetime import datetime
import multiprocessing as mp

def intensive_gpu_worker(gpu_id, memory_fraction=0.8, compute_intensity=0.9):
    """
    高强度GPU工作进程，确保GPU利用率保持高水平
    """
    device = torch.device(f'cuda:{gpu_id}')
    allocated_tensors = []
    
    print(f"[{datetime.now()}] GPU {gpu_id}: 启动高强度计算...")
    
    try:
        while True:
            # 显存管理
            allocated = torch.cuda.memory_allocated(device) / 1024**3
            total = torch.cuda.get_device_properties(device).total_memory / 1024**3
            
            if allocated / total < memory_fraction:
                try:
                    # 分配大块显存
                    tensor = torch.randn(1024*1024*50, device=device, dtype=torch.float32)  # 200MB
                    allocated_tensors.append(tensor)
                except RuntimeError:
                    pass
            
            # 动态管理tensor数量
            if len(allocated_tensors) > 20:
                to_remove = allocated_tensors[:10]
                for tensor in to_remove:
                    del tensor
                allocated_tensors = allocated_tensors[10:]
                torch.cuda.empty_cache()
            
            # 高强度计算任务
            batch_size = int(512 * compute_intensity)
            seq_len = int(1024 * compute_intensity)
            hidden_size = int(2048 * compute_intensity)
            
            # 创建输入数据
            x = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=torch.float32)
            
            # 模拟Transformer计算
            for layer in range(10):
                # Self-attention
                q = torch.linear(x, torch.randn(hidden_size, hidden_size, device=device))
                k = torch.linear(x, torch.randn(hidden_size, hidden_size, device=device))
                v = torch.linear(x, torch.randn(hidden_size, hidden_size, device=device))
                
                # Attention计算
                scores = torch.matmul(q, k.transpose(-2, -1)) / (hidden_size ** 0.5)
                attn_weights = torch.softmax(scores, dim=-1)
                attn_output = torch.matmul(attn_weights, v)
                
                # Feed-forward
                ff1 = torch.relu(torch.linear(attn_output, torch.randn(hidden_size, hidden_size*4, device=device)))
                ff2 = torch.linear(ff1, torch.randn(hidden_size*4, hidden_size, device=device))
                
                # 残差连接和层归一化
                x = x + ff2
                x = torch.layer_norm(x, x.shape[-1:])
                
                # 添加一些额外的计算
                x = torch.dropout(x, 0.1, training=True)
                x = torch.gelu(x)
            
            # 最终计算
            output = torch.mean(x, dim=1)  # 全局平均池化
            logits = torch.linear(output, torch.randn(hidden_size, 1000, device=device))
            loss = torch.cross_entropy(logits, torch.randint(0, 1000, (batch_size,), device=device))
            
            # 反向传播（模拟）
            loss.backward()
            
            # 清理梯度
            for param in [q, k, v, ff1, ff2, logits]:
                if param.grad is not None:
                    param.grad.zero_()
            
            print(f"[{datetime.now()}] GPU {gpu_id}: 高强度计算完成，Loss: {loss.item():.6f}, 显存: {allocated:.1f}GB")
            
            # 短暂休息，但保持计算频率
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print(f"[{datetime.now()}] GPU {gpu_id}: 收到中断信号")
    except Exception as e:
        print(f"[{datetime.now()}] GPU {gpu_id}: 错误 - {e}")
    finally:
        # 清理显存
        for tensor in allocated_tensors:
            del tensor
        torch.cuda.empty_cache()
        print(f"[{datetime.now()}] GPU {gpu_id}: 显存已清理")

def main():
    parser = argparse.ArgumentParser(description='Intensive GPU Keep-Alive Script')
    parser.add_argument('--gpu-ids', type=str, default='0,1,2,3,4,5,6,7', 
                       help='要使用的GPU ID列表，用逗号分隔')
    parser.add_argument('--memory-fraction', type=float, default=0.8, 
                       help='显存占用比例 (0.0-1.0)')
    parser.add_argument('--compute-intensity', type=float, default=0.9, 
                       help='计算强度 (0.0-1.0)')
    
    args = parser.parse_args()
    
    if not torch.cuda.is_available():
        print("错误: CUDA不可用")
        return
    
    gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(',')]
    available_gpus = torch.cuda.device_count()
    
    invalid_gpus = [gpu_id for gpu_id in gpu_ids if gpu_id >= available_gpus]
    if invalid_gpus:
        print(f"错误: GPU {invalid_gpus} 不存在，可用GPU数量: {available_gpus}")
        return
    
    print(f"启动高强度GPU占用: {gpu_ids}")
    print(f"显存占用: {args.memory_fraction*100:.1f}%")
    print(f"计算强度: {args.compute_intensity*100:.1f}%")
    print("按 Ctrl+C 停止")
    print("-" * 60)
    
    processes = []
    
    try:
        for gpu_id in gpu_ids:
            p = mp.Process(
                target=intensive_gpu_worker,
                args=(gpu_id, args.memory_fraction, args.compute_intensity)
            )
            p.start()
            processes.append(p)
            time.sleep(0.5)
        
        print(f"[{datetime.now()}] 所有GPU进程已启动")
        
        # 监控进程状态
        while True:
            time.sleep(5)
            alive_count = sum(1 for p in processes if p.is_alive())
            print(f"[{datetime.now()}] 活跃进程数: {alive_count}/{len(processes)}")
            
            # 重启死亡的进程
            for i, p in enumerate(processes):
                if not p.is_alive():
                    gpu_id = gpu_ids[i]
                    print(f"[{datetime.now()}] 重启GPU {gpu_id}进程")
                    new_p = mp.Process(
                        target=intensive_gpu_worker,
                        args=(gpu_id, args.memory_fraction, args.compute_intensity)
                    )
                    new_p.start()
                    processes[i] = new_p
    
    except KeyboardInterrupt:
        print(f"\n[{datetime.now()}] 收到中断信号，正在停止...")
    
    finally:
        for p in processes:
            if p.is_alive():
                p.terminate()
                p.join(timeout=5)
                if p.is_alive():
                    p.kill()
        print(f"[{datetime.now()}] 所有进程已停止")

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
