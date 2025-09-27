#!/usr/bin/env python3
"""
Multi-GPU Keep-Alive Script
占用多张GPU显存和计算资源，防止进程被系统kill
支持8卡GPU
"""

import torch
import time
import argparse
import threading
import numpy as np
from datetime import datetime
import multiprocessing as mp

def gpu_worker(gpu_id, memory_fraction=0.8, compute_intensity=0.8, chunk_size=1024*1024*100):
    """
    单个GPU的工作进程
    """
    device = torch.device(f'cuda:{gpu_id}')
    allocated_tensors = []
    
    print(f"[{datetime.now()}] GPU {gpu_id}: 开始工作...")
    
    try:
        while True:
            # 获取当前显存使用情况
            allocated = torch.cuda.memory_allocated(device) / 1024**3  # GB
            cached = torch.cuda.memory_reserved(device) / 1024**3  # GB
            total = torch.cuda.get_device_properties(device).total_memory / 1024**3  # GB
            
            print(f"[{datetime.now()}] GPU {gpu_id}: 显存 {allocated:.2f}GB/{total:.2f}GB (缓存: {cached:.2f}GB)")
            
            # 显存管理
            if allocated / total < memory_fraction:
                try:
                    tensor = torch.randn(chunk_size, device=device, dtype=torch.float32)
                    allocated_tensors.append(tensor)
                    print(f"[{datetime.now()}] GPU {gpu_id}: 分配 {chunk_size/1024**2:.1f}MB 显存")
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"[{datetime.now()}] GPU {gpu_id}: 显存不足，停止分配")
                    else:
                        raise e
            
            # 动态管理tensor数量
            if len(allocated_tensors) > 8:
                to_remove = allocated_tensors[:len(allocated_tensors)//2]
                for tensor in to_remove:
                    del tensor
                allocated_tensors = allocated_tensors[len(allocated_tensors)//2:]
                torch.cuda.empty_cache()
            
            # 计算任务
            size = int(1024 * compute_intensity)
            a = torch.randn(size, size, device=device, dtype=torch.float32)
            b = torch.randn(size, size, device=device, dtype=torch.float32)
            
            for _ in range(5):
                c = torch.matmul(a, b)
                c = torch.relu(c)
                c = torch.softmax(c, dim=1)
                a = c
            
            result = torch.sum(c)
            print(f"[{datetime.now()}] GPU {gpu_id}: 计算完成，结果: {result.item():.6f}")
            
            time.sleep(3)  # 每3秒一个周期
            
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
    parser = argparse.ArgumentParser(description='Multi-GPU Keep-Alive Script')
    parser.add_argument('--gpu-ids', type=str, default='0,1,2,3,4,5,6,7', 
                       help='要使用的GPU ID列表，用逗号分隔 (例如: 0,1,2,3)')
    parser.add_argument('--memory-fraction', type=float, default=0.8, 
                       help='显存占用比例 (0.0-1.0)')
    parser.add_argument('--compute-intensity', type=float, default=0.8, 
                       help='计算强度 (0.0-1.0)')
    parser.add_argument('--duration', type=int, default=0, 
                       help='运行时长(秒)，0表示无限运行')
    
    args = parser.parse_args()
    
    # 检查CUDA是否可用
    if not torch.cuda.is_available():
        print("错误: CUDA不可用")
        return
    
    # 解析GPU ID列表
    gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(',')]
    available_gpus = torch.cuda.device_count()
    
    # 验证GPU ID
    invalid_gpus = [gpu_id for gpu_id in gpu_ids if gpu_id >= available_gpus]
    if invalid_gpus:
        print(f"错误: GPU {invalid_gpus} 不存在，可用GPU数量: {available_gpus}")
        return
    
    print(f"开始占用 {len(gpu_ids)} 张GPU: {gpu_ids}")
    print(f"显存占用比例: {args.memory_fraction*100:.1f}%")
    print(f"计算强度: {args.compute_intensity*100:.1f}%")
    print(f"运行时长: {'无限' if args.duration == 0 else f'{args.duration}秒'}")
    print("按 Ctrl+C 停止")
    print("-" * 60)
    
    # 创建进程列表
    processes = []
    
    try:
        # 为每个GPU创建进程
        for gpu_id in gpu_ids:
            p = mp.Process(
                target=gpu_worker,
                args=(gpu_id, args.memory_fraction, args.compute_intensity)
            )
            p.start()
            processes.append(p)
            time.sleep(1)  # 错开启动时间
        
        print(f"[{datetime.now()}] 所有GPU进程已启动")
        
        # 监控进程状态
        if args.duration > 0:
            time.sleep(args.duration)
        else:
            # 无限运行，定期检查进程状态
            while True:
                time.sleep(10)
                alive_count = sum(1 for p in processes if p.is_alive())
                print(f"[{datetime.now()}] 活跃进程数: {alive_count}/{len(processes)}")
                
                # 如果有进程死亡，重新启动
                for i, p in enumerate(processes):
                    if not p.is_alive():
                        gpu_id = gpu_ids[i]
                        print(f"[{datetime.now()}] 重启GPU {gpu_id}进程")
                        new_p = mp.Process(
                            target=gpu_worker,
                            args=(gpu_id, args.memory_fraction, args.compute_intensity)
                        )
                        new_p.start()
                        processes[i] = new_p
    
    except KeyboardInterrupt:
        print(f"\n[{datetime.now()}] 收到中断信号，正在停止所有进程...")
    
    finally:
        # 终止所有进程
        for p in processes:
            if p.is_alive():
                p.terminate()
                p.join(timeout=5)
                if p.is_alive():
                    p.kill()
        
        print(f"[{datetime.now()}] 所有GPU进程已停止")

if __name__ == "__main__":
    # 设置多进程启动方法
    mp.set_start_method('spawn', force=True)
    main()
