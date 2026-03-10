#!/usr/bin/env python3
"""
Aggressive GPU Keep-Alive Script
"""

import torch
import time
import argparse
import multiprocessing as mp
from datetime import datetime

def aggressive_gpu_worker(gpu_id, memory_fraction=0.8, compute_intensity=0.9):
    device = torch.device(f'cuda:{gpu_id}')
    allocated_tensors = []
    
    print(f"[{datetime.now()}] GPU {gpu_id}: Starting aggressive memory allocation...")
    
    try:
        total_memory = torch.cuda.get_device_properties(device).total_memory
        target_memory = int(total_memory * memory_fraction)
        
        print(f"[{datetime.now()}] GPU {gpu_id}: Target memory {target_memory/1024**3:.1f}GB")
        
        chunk_size = 1024*1024*1000
        while True:
            allocated = torch.cuda.memory_allocated(device)
            
            if allocated < target_memory:
                try:
                    tensor = torch.randn(chunk_size, device=device, dtype=torch.float32)
                    allocated_tensors.append(tensor)
                    print(f"[{datetime.now()}] GPU {gpu_id}: Allocated {allocated/1024**3:.1f}GB / {target_memory/1024**3:.1f}GB")
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"[{datetime.now()}] GPU {gpu_id}: Memory allocation completed, starting computation")
                        break
                    else:
                        raise e
            else:
                print(f"[{datetime.now()}] GPU {gpu_id}: Memory allocation completed, starting computation")
                break
        
        while True:
            allocated = torch.cuda.memory_allocated(device) / 1024**3
            total = torch.cuda.get_device_properties(device).total_memory / 1024**3
            
            if allocated / total < memory_fraction * 0.9:
                try:
                    tensor = torch.randn(chunk_size, device=device, dtype=torch.float32)
                    allocated_tensors.append(tensor)
                except RuntimeError:
                    pass
            
            if len(allocated_tensors) > 20:
                to_remove = allocated_tensors[:5]
                for tensor in to_remove:
                    del tensor
                allocated_tensors = allocated_tensors[5:]
                torch.cuda.empty_cache()
            
            size = int(2048 * compute_intensity)
            
            a = torch.randn(size, size, device=device, dtype=torch.float32)
            b = torch.randn(size, size, device=device, dtype=torch.float32)
            
            for _ in range(100):
                c = torch.matmul(a, b)
                c = torch.relu(c)
                c = torch.softmax(c, dim=1)
                
                c = torch.sin(c) + torch.cos(c)
                c = torch.tanh(c)
                c = torch.sigmoid(c)
                
                a = c
                b = torch.randn(size, size, device=device, dtype=torch.float32)
            
            result = torch.sum(c)
            result = torch.log(torch.abs(result) + 1e-8)
            
            print(f"[{datetime.now()}] GPU {gpu_id}: Computation completed, result: {result.item():.6f}, memory: {allocated:.1f}GB")
            
            time.sleep(0.02)
            
    except KeyboardInterrupt:
        print(f"[{datetime.now()}] GPU {gpu_id}: Received interrupt signal")
    except Exception as e:
        print(f"[{datetime.now()}] GPU {gpu_id}: Error - {e}")
    finally:
        for tensor in allocated_tensors:
            del tensor
        torch.cuda.empty_cache()
        print(f"[{datetime.now()}] GPU {gpu_id}: Memory cleared")

def main():
    parser = argparse.ArgumentParser(description='Aggressive GPU Keep-Alive Script')
    parser.add_argument('--gpu-ids', type=str, default='0,1,2,3,4,5,6,7', 
                       help='Comma-separated list of GPU IDs to use')
    parser.add_argument('--memory-fraction', type=float, default=0.8, 
                       help='Memory usage fraction (0.0-1.0)')
    parser.add_argument('--compute-intensity', type=float, default=0.9, 
                       help='Compute intensity (0.0-1.0)')
    
    args = parser.parse_args()
    
    if not torch.cuda.is_available():
        print("Error: CUDA is not available")
        return
    
    gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(',')]
    available_gpus = torch.cuda.device_count()
    
    invalid_gpus = [gpu_id for gpu_id in gpu_ids if gpu_id >= available_gpus]
    if invalid_gpus:
        print(f"Error: GPU {invalid_gpus} does not exist, available GPU count: {available_gpus}")
        return
    
    print(f"Starting aggressive GPU allocation: {gpu_ids}")
    print(f"Memory usage: {args.memory_fraction*100:.1f}%")
    print(f"Compute intensity: {args.compute_intensity*100:.1f}%")
    print("Press Ctrl+C to stop")
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
            time.sleep(1)
        
        print(f"[{datetime.now()}] All GPU processes started")
        
        while True:
            time.sleep(5)
            alive_count = sum(1 for p in processes if p.is_alive())
            print(f"[{datetime.now()}] Active processes: {alive_count}/{len(processes)}")
            
            for i, p in enumerate(processes):
                if not p.is_alive():
                    gpu_id = gpu_ids[i]
                    print(f"[{datetime.now()}] Restarting GPU {gpu_id} process")
                    new_p = mp.Process(
                        target=aggressive_gpu_worker,
                        args=(gpu_id, args.memory_fraction, args.compute_intensity)
                    )
                    new_p.start()
                    processes[i] = new_p
    
    except KeyboardInterrupt:
        print(f"\n[{datetime.now()}] Received interrupt signal, stopping...")
    
    finally:
        for p in processes:
            if p.is_alive():
                p.terminate()
                p.join(timeout=5)
                if p.is_alive():
                    p.kill()
        print(f"[{datetime.now()}] All processes stopped")

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()