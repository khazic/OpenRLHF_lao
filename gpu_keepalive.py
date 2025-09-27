#!/usr/bin/env python3
"""
GPU Keep-Alive Script
占用GPU显存和计算资源，防止进程被系统kill
"""

import torch
import time
import argparse
import threading
import numpy as np
from datetime import datetime

def gpu_memory_allocator(device_id, memory_fraction=0.8, chunk_size=1024*1024*100):
    """
    持续分配和释放GPU显存，保持显存占用
    """
    device = torch.device(f'cuda:{device_id}')
    allocated_tensors = []
    
    print(f"[{datetime.now()}] 开始在GPU {device_id}上分配显存...")
    
    try:
        while True:
            # 获取当前显存使用情况
            allocated = torch.cuda.memory_allocated(device) / 1024**3  # GB
            cached = torch.cuda.memory_reserved(device) / 1024**3  # GB
            total = torch.cuda.get_device_properties(device).total_memory / 1024**3  # GB
            
            print(f"[{datetime.now()}] GPU {device_id} 显存使用: {allocated:.2f}GB / {total:.2f}GB (缓存: {cached:.2f}GB)")
            
            # 如果显存使用率低于目标，分配更多
            if allocated / total < memory_fraction:
                try:
                    # 分配大块显存
                    tensor = torch.randn(chunk_size, device=device, dtype=torch.float32)
                    allocated_tensors.append(tensor)
                    print(f"[{datetime.now()}] 分配了 {chunk_size/1024**2:.1f}MB 显存")
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"[{datetime.now()}] 显存不足，停止分配")
                        break
                    else:
                        raise e
            
            # 保持一些tensor，释放一些tensor，模拟动态使用
            if len(allocated_tensors) > 10:
                # 释放一半的tensor
                to_remove = allocated_tensors[:len(allocated_tensors)//2]
                for tensor in to_remove:
                    del tensor
                allocated_tensors = allocated_tensors[len(allocated_tensors)//2:]
                torch.cuda.empty_cache()
                print(f"[{datetime.now()}] 释放了部分显存，当前保持 {len(allocated_tensors)} 个tensor")
            
            time.sleep(5)  # 每5秒检查一次
            
    except KeyboardInterrupt:
        print(f"[{datetime.now()}] 收到中断信号，清理显存...")
    finally:
        # 清理所有分配的tensor
        for tensor in allocated_tensors:
            del tensor
        torch.cuda.empty_cache()
        print(f"[{datetime.now()}] GPU {device_id} 显存已清理")

def gpu_compute_worker(device_id, compute_intensity=0.8):
    """
    持续进行GPU计算，保持计算资源占用
    """
    device = torch.device(f'cuda:{device_id}')
    
    print(f"[{datetime.now()}] 开始在GPU {device_id}上进行计算...")
    
    try:
        while True:
            # 创建随机矩阵进行计算
            size = int(2048 * compute_intensity)
            a = torch.randn(size, size, device=device, dtype=torch.float32)
            b = torch.randn(size, size, device=device, dtype=torch.float32)
            
            # 进行矩阵运算
            for _ in range(10):
                c = torch.matmul(a, b)
                c = torch.relu(c)
                c = torch.softmax(c, dim=1)
                a = c  # 链式计算
            
            # 计算一些统计信息
            result = torch.sum(c)
            result = torch.log(result + 1e-8)
            
            print(f"[{datetime.now()}] GPU {device_id} 计算完成，结果: {result.item():.6f}")
            time.sleep(1)  # 短暂休息
            
    except KeyboardInterrupt:
        print(f"[{datetime.now()}] 计算线程收到中断信号")

def main():
    parser = argparse.ArgumentParser(description='GPU Keep-Alive Script')
    parser.add_argument('--gpu-id', type=int, default=0, help='GPU设备ID')
    parser.add_argument('--memory-fraction', type=float, default=0.8, help='显存占用比例 (0.0-1.0)')
    parser.add_argument('--compute-intensity', type=float, default=0.8, help='计算强度 (0.0-1.0)')
    parser.add_argument('--duration', type=int, default=0, help='运行时长(秒)，0表示无限运行')
    
    args = parser.parse_args()
    
    # 检查GPU是否可用
    if not torch.cuda.is_available():
        print("错误: CUDA不可用")
        return
    
    if args.gpu_id >= torch.cuda.device_count():
        print(f"错误: GPU {args.gpu_id} 不存在，可用GPU数量: {torch.cuda.device_count()}")
        return
    
    print(f"开始占用GPU {args.gpu_id}")
    print(f"显存占用比例: {args.memory_fraction*100:.1f}%")
    print(f"计算强度: {args.compute_intensity*100:.1f}%")
    print(f"运行时长: {'无限' if args.duration == 0 else f'{args.duration}秒'}")
    print("按 Ctrl+C 停止")
    print("-" * 50)
    
    # 启动显存分配线程
    memory_thread = threading.Thread(
        target=gpu_memory_allocator, 
        args=(args.gpu_id, args.memory_fraction)
    )
    memory_thread.daemon = True
    memory_thread.start()
    
    # 启动计算线程
    compute_thread = threading.Thread(
        target=gpu_compute_worker, 
        args=(args.gpu_id, args.compute_intensity)
    )
    compute_thread.daemon = True
    compute_thread.start()
    
    try:
        if args.duration > 0:
            time.sleep(args.duration)
        else:
            # 无限运行
            while True:
                time.sleep(1)
    except KeyboardInterrupt:
        print(f"\n[{datetime.now()}] 收到中断信号，正在停止...")
    
    print(f"[{datetime.now()}] 程序结束")

if __name__ == "__main__":
    main()
