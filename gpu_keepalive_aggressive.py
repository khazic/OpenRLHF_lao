#!/usr/bin/env python3
"""
Aggressive GPU Keep-Alive Script
激进GPU占用脚本，快速达到目标显存占用
"""

import torch
import time
import argparse
import multiprocessing as mp
from datetime import datetime

def aggressive_gpu_worker(gpu_id, memory_fraction=0.8, compute_intensity=0.9):
    """
    激进GPU工作进程，快速达到目标显存占用
    """
    device = torch.device(f'cuda:{gpu_id}')
    allocated_tensors = []
    
    print(f"[{datetime.now()}] GPU {gpu_id}: 启动激进显存占用...")
    
    try:
        # 快速分配显存到目标比例
        total_memory = torch.cuda.get_device_properties(device).total_memory
        target_memory = int(total_memory * memory_fraction)
        
        print(f"[{datetime.now()}] GPU {gpu_id}: 目标显存 {target_memory/1024**3:.1f}GB")
        
        # 快速分配大块显存
        chunk_size = 1024*1024*1000  # 4GB per chunk
        while True:
            allocated = torch.cuda.memory_allocated(device)
            
            if allocated < target_memory:
                try:
                    # 分配4GB大块
                    tensor = torch.randn(chunk_size, device=device, dtype=torch.float32)
                    allocated_tensors.append(tensor)
                    print(f"[{datetime.now()}] GPU {gpu_id}: 已分配 {allocated/1024**3:.1f}GB / {target_memory/1024**3:.1f}GB")
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"[{datetime.now()}] GPU {gpu_id}: 显存分配完成，开始计算")
                        break
                    else:
                        raise e
            else:
                print(f"[{datetime.now()}] GPU {gpu_id}: 显存分配完成，开始计算")
                break
        
        # 持续高强度计算
        while True:
            # 显存管理
            allocated = torch.cuda.memory_allocated(device) / 1024**3
            total = torch.cuda.get_device_properties(device).total_memory / 1024**3
            
            # 如果显存不足，重新分配
            if allocated / total < memory_fraction * 0.9:  # 90%阈值
                try:
                    tensor = torch.randn(chunk_size, device=device, dtype=torch.float32)
                    allocated_tensors.append(tensor)
                except RuntimeError:
                    pass
            
            # 动态管理tensor数量
            if len(allocated_tensors) > 20:
                to_remove = allocated_tensors[:5]
                for tensor in to_remove:
                    del tensor
                allocated_tensors = allocated_tensors[5:]
                torch.cuda.empty_cache()
            
            # 高强度计算
            size = int(2048 * compute_intensity)
            
            # 创建大矩阵
            a = torch.randn(size, size, device=device, dtype=torch.float32)
            b = torch.randn(size, size, device=device, dtype=torch.float32)
            
            # 持续进行矩阵运算
            for _ in range(100):  # 增加计算轮数
                c = torch.matmul(a, b)
                c = torch.relu(c)
                c = torch.softmax(c, dim=1)
                
                # 添加更多计算
                c = torch.sin(c) + torch.cos(c)
                c = torch.tanh(c)
                c = torch.sigmoid(c)
                
                # 更新矩阵用于下一轮
                a = c
                b = torch.randn(size, size, device=device, dtype=torch.float32)
            
            # 计算一些统计信息
            result = torch.sum(c)
            result = torch.log(torch.abs(result) + 1e-8)
            
            print(f"[{datetime.now()}] GPU {gpu_id}: 计算完成，结果: {result.item():.6f}, 显存: {allocated:.1f}GB")
            
            # 很短的休息时间
            time.sleep(0.02)
            
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
    parser = argparse.ArgumentParser(description='Aggressive GPU Keep-Alive Script')
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
    
    print(f"启动激进GPU占用: {gpu_ids}")
    print(f"显存占用: {args.memory_fraction*100:.1f}%")
    print(f"计算强度: {args.compute_intensity*100:.1f}%")
    print("按 Ctrl+C 停止")
    print("-" * 60)
    
    processes = []
    
    try:
        for gpu_id in gpu_ids:
            p = mp.Process(
                target=aggressive_gpu_worker,
                args=(gpu_id, args.memory_fraction, args.compute_intensity)
            )
            p.start()
            processes.append(p)
            time.sleep(1)  # 错开启动时间
        
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
                        target=aggressive_gpu_worker,
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
