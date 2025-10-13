#!/bin/bash

# GPU Keep-Alive Script
# Usage: ./keep_gpu_busy.sh [GPU_IDS] [MEMORY_FRACTION] [COMPUTE_INTENSITY]
# Example: ./keep_gpu_busy.sh "0,1,2,3,4,5,6,7" 0.3 0.5
#
# Parameters:
#   GPU_IDS           - Comma-separated list of GPU device IDs to occupy
#                       Default: "0,1,2,3,4,5,6,7"
#                       Example: "0,1,2,3" or "0,2,4,6"
#
#   MEMORY_FRACTION   - Fraction of GPU memory to allocate per device (0.0-1.0)
#                       Default: 0.5 (50% of available memory)
#                       Lower values: Less memory usage, suitable for sharing GPUs
#                       Higher values: More memory usage, prevents other processes
#
#   COMPUTE_INTENSITY - Computational workload intensity (0.0-1.0)
#                       Default: 0.5 (moderate computation)
#                       Lower values: Light computation, low power consumption
#                       Higher values: Heavy computation, high GPU utilization
#                       Note: Higher intensity may cause GPU throttling/heating

GPU_IDS=${1:-"0,1,2,3,4,5,6,7"}
MEMORY_FRACTION=${2:-0.5}
COMPUTE_INTENSITY=${3:-0.5}

echo "Starting GPU Keep-Alive Script..."
echo "GPU IDs: $GPU_IDS"
echo "Memory Fraction: $(echo "$MEMORY_FRACTION * 100" | bc)%"
echo "Compute Intensity: $(echo "$COMPUTE_INTENSITY * 100" | bc)%"
echo "Press Ctrl+C to stop"
echo "----------------------------------------"

python3 gpu_keepalive_aggressive.py \
    --gpu-ids "$GPU_IDS" \
    --memory-fraction $MEMORY_FRACTION \
    --compute-intensity $COMPUTE_INTENSITY
