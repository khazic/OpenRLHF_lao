#!/bin/bash

# 高强度GPU占用脚本
# 确保GPU利用率保持高水平

echo "=========================================="
echo "启动高强度8卡GPU占用脚本"
echo "=========================================="

# 检查GPU数量
GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
echo "检测到 $GPU_COUNT 张GPU"

if [ $GPU_COUNT -lt 8 ]; then
    echo "警告: 检测到的GPU数量少于8张"
    GPU_IDS=$(seq -s, 0 $((GPU_COUNT-1)))
else
    GPU_IDS="0,1,2,3,4,5,6,7"
fi

echo "使用的GPU: $GPU_IDS"
echo "显存占用: 80%"
echo "计算强度: 90% (高强度)"
echo "按 Ctrl+C 停止"
echo "=========================================="

# 后台运行
nohup python3 gpu_keepalive_intensive.py \
    --gpu-ids "$GPU_IDS" \
    --memory-fraction 0.8 \
    --compute-intensity 0.9 > gpu_intensive.log 2>&1 &

PID=$!
echo "进程ID: $PID"
echo "日志文件: gpu_intensive.log"
echo ""
echo "查看日志: tail -f gpu_intensive.log"
echo "停止进程: kill $PID"
echo "查看GPU状态: watch -n 1 nvidia-smi"

# 等待启动
sleep 3

# 显示初始状态
echo ""
echo "初始GPU状态:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits

echo ""
echo "监控GPU利用率变化..."
