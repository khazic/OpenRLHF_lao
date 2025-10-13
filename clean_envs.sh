#!/bin/bash

echo "Cleaning up vLLM and related processes..."

# 1. Kill all vLLM related processes
echo "1. Killing vLLM processes..."
pkill -f "vllm" || echo "No vLLM processes found"

# 2. Kill all Ray processes
echo "2. Killing Ray processes..."
pkill -f "ray::" || echo "No Ray worker processes found"
pkill -f "raylet" || echo "No raylet processes found"

# 3. Kill all OpenRLHF training processes
echo "3. Killing OpenRLHF training processes..."
pkill -f "train_ppo_ray" || echo "No training processes found"
pkill -f "python.*openrlhf" || echo "No OpenRLHF processes found"

# 4. Kill all Python processes (if containing related keywords)
echo "4. Killing related Python processes..."
ps aux | grep -E "(LLMRayActor|EngineCore|WorkerWrap)" | grep -v grep | awk '{print $2}' | xargs -r kill -9

# 5. Force kill all possible residual CUDA processes
echo "5. Cleaning CUDA processes..."
nvidia-smi --gpu-reset-ecc=0,1,2,3,4,5,6,7 2>/dev/null || echo "Unable to reset GPU ECC"

# 6. Clean shared memory
echo "6. Cleaning shared memory..."
ipcs -m | awk '/0x/ {print $2}' | xargs -r ipcrm -m 2>/dev/null || echo "Shared memory cleanup completed"

# 7. Stop Ray cluster (if exists)
echo "7. Stopping Ray cluster..."
ray stop --force 2>/dev/null || echo "Ray cluster stopped or does not exist"

# 8. Clean temporary files
echo "8. Cleaning temporary files..."
rm -rf /tmp/ray* 2>/dev/null || echo "Temporary files cleanup completed"
rm -rf /dev/shm/ray* 2>/dev/null || echo "Shared memory files cleanup completed"

echo "Cleanup completed!"
echo "Waiting 5 seconds for system stabilization..."
sleep 5

echo "Checking for residual processes:"
ps aux | grep -E "(vllm|ray|LLMRayActor|EngineCore)" | grep -v grep || echo "No residual processes found"

echo "GPU status:"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null || echo "Unable to query GPU status"
