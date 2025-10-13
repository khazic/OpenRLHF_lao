#!/bin/bash

scripts=(
    "paper_rloo_nokl.sh"
    "paper_rplus_nokl.sh"
    "paper_xpo_nokl_eps.sh"
    "paper_xpo_nokl_eps_entropy.sh"
    "paper_xpo_nokl_eps_overlong_entropy.sh"
)

cleanup() {
    echo "清理 GPU 内存和训练进程..."
    
    pkill -f train_ppo_ray 2>/dev/null || echo "没有训练进程"
    pkill -f vllm 2>/dev/null || echo "没有 vLLM 进程"
    
    ps aux | grep -E "(python.*openrlhf|LLMRayActor|EngineCore)" | grep -v grep | awk '{print $2}' | xargs -r kill -9 2>/dev/null || echo "清理训练进程完成"
    
    sleep 3
    
    python3 -c "
import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print('PyTorch GPU 缓存已清理')
else:
    print('没有可用的 CUDA')
" 2>/dev/null || echo "无法清理 PyTorch 缓存"
    
    echo "GPU 内存清理完成，Ray 集群保持运行"
    nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits
}

for script in "${scripts[@]}"; do
    echo "========================================"
    echo "运行: $script"
    echo "时间: $(date)"
    echo "========================================"
    
    cd examples/scripts
    bash "$script"
    cd ../..
    
    echo "完成: $script"
    cleanup
    
    echo "等待 10 秒后继续下一个实验..."
    sleep 10
done

echo "所有实验完成!"
