#!/bin/bash

# 8卡GPU占用脚本
# 快速启动，占用所有8张GPU

echo "=========================================="
echo "启动8卡GPU占用脚本"
echo "=========================================="

# 检查GPU数量
GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
echo "检测到 $GPU_COUNT 张GPU"

if [ $GPU_COUNT -lt 8 ]; then
    echo "警告: 检测到的GPU数量少于8张"
    echo "将使用所有可用的GPU: 0到$((GPU_COUNT-1))"
    GPU_IDS=$(seq -s, 0 $((GPU_COUNT-1)))
else
    GPU_IDS="0,1,2,3,4,5,6,7"
fi

echo "使用的GPU: $GPU_IDS"
echo "显存占用: 80%"
echo "计算强度: 80%"
echo "按 Ctrl+C 停止"
echo "=========================================="

# 后台运行并输出到日志
nohup python3 gpu_keepalive_multi.py \
    --gpu-ids "$GPU_IDS" \
    --memory-fraction 0.8 \
    --compute-intensity 0.8 > gpu_keepalive_8gpu.log 2>&1 &

PID=$!
echo "进程ID: $PID"
echo "日志文件: gpu_keepalive_8gpu.log"
echo ""
echo "查看日志: tail -f gpu_keepalive_8gpu.log"
echo "停止进程: kill $PID"
echo "查看进程: ps aux | grep gpu_keepalive"

# 等待一下确保进程启动
sleep 3

# 显示初始状态
echo ""
echo "初始GPU状态:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
