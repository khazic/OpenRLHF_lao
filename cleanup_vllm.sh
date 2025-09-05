#!/bin/bash

echo "正在清理 vLLM 和相关进程..."

# 1. 杀死所有 vLLM 相关进程
echo "1. 正在杀死 vLLM 进程..."
pkill -f "vllm" || echo "没有找到 vLLM 进程"

# 2. 杀死所有 Ray 进程
echo "2. 正在杀死 Ray 进程..."
pkill -f "ray::" || echo "没有找到 Ray worker 进程"
pkill -f "raylet" || echo "没有找到 raylet 进程"

# 3. 杀死所有 OpenRLHF 训练进程
echo "3. 正在杀死 OpenRLHF 训练进程..."
pkill -f "train_ppo_ray" || echo "没有找到训练进程"
pkill -f "python.*openrlhf" || echo "没有找到 OpenRLHF 进程"

# 4. 杀死所有 Python 进程（如果包含相关关键词）
echo "4. 正在杀死相关 Python 进程..."
ps aux | grep -E "(LLMRayActor|EngineCore|WorkerWrap)" | grep -v grep | awk '{print $2}' | xargs -r kill -9

# 5. 强制杀死所有可能残留的 CUDA 进程
echo "5. 正在清理 CUDA 进程..."
nvidia-smi --gpu-reset-ecc=0,1,2,3,4,5,6,7 2>/dev/null || echo "无法重置 GPU ECC"

# 6. 清理共享内存
echo "6. 正在清理共享内存..."
ipcs -m | awk '/0x/ {print $2}' | xargs -r ipcrm -m 2>/dev/null || echo "共享内存清理完成"

# 7. 停止 Ray 集群（如果存在）
echo "7. 正在停止 Ray 集群..."
ray stop --force 2>/dev/null || echo "Ray 集群已停止或不存在"

# 8. 清理临时文件
echo "8. 正在清理临时文件..."
rm -rf /tmp/ray* 2>/dev/null || echo "临时文件清理完成"
rm -rf /dev/shm/ray* 2>/dev/null || echo "共享内存文件清理完成"

echo "清理完成！"
echo "等待 5 秒钟让系统稳定..."
sleep 5

echo "检查是否还有残留进程："
ps aux | grep -E "(vllm|ray|LLMRayActor|EngineCore)" | grep -v grep || echo "没有发现残留进程"

echo "GPU 状态："
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null || echo "无法查询 GPU 状态"
