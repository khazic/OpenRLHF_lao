#!/bin/bash

#   ./keep_gpu_busy.sh [GPU_IDS] [MEMORY_FRACTION] [COMPUTE_INTENSITY]
#   例如: ./keep_gpu_busy.sh "0,1,2,3,4,5,6,7,8" 0.3 0.3

GPU_IDS=${1:-"0,1,2,3,4,5,6,7"}
MEMORY_FRACTION=${2:-0.5}
COMPUTE_INTENSITY=${3:-0.5}

echo "启动高强度GPU占用脚本..."
echo "GPU IDs: $GPU_IDS"
echo "显存占用: $(echo "$MEMORY_FRACTION * 100" | bc)%"
echo "计算强度: $(echo "$COMPUTE_INTENSITY * 100" | bc)%"
echo "按 Ctrl+C 停止"
echo "----------------------------------------"

python3 gpu_keepalive_aggressive.py \
    --gpu-ids "$GPU_IDS" \
    --memory-fraction $MEMORY_FRACTION \
    --compute-intensity $COMPUTE_INTENSITY
