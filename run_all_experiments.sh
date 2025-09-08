#!/bin/bash

# 自动运行多个 PPO 训练实验的脚本
# 每个实验完成后会清理显存，然后继续下一个

set -e  # 遇到错误时退出

# 定义要运行的脚本列表
SCRIPTS=(
    "paper_dapo_nokl.sh"
    "paper_grpo_nokl.sh"
    "paper_rloo_nokl.sh"
    "paper_rplus_nokl.sh"
    "paper_xpo_nokl_eps.sh"
    "paper_xpo_nokl_eps_entropy.sh"
    "paper_xpo_nokl_eps_overlong_entropy.sh"
)

# 日志目录
LOG_DIR="./experiment_logs"
mkdir -p $LOG_DIR

# 清理显存和训练进程的函数（保留 Ray 集群）
cleanup_gpu_memory() {
    echo "========================================"
    echo "正在清理 GPU 内存和训练进程（保留 Ray 集群）..."
    echo "========================================"
    
    # 只杀死训练相关的进程，保留 Ray 集群
    pkill -f train_ppo_ray 2>/dev/null || echo "没有找到训练进程"
    pkill -f vllm 2>/dev/null || echo "没有找到 vLLM 进程"
    
    # 杀死 OpenRLHF 相关的 Python 进程，但保留 Ray 核心进程
    ps aux | grep -E "(python.*openrlhf|LLMRayActor|EngineCore|WorkerWrap)" | grep -v grep | awk '{print $2}' | xargs -r kill -9 2>/dev/null || echo "清理训练进程完成"
    
    # 等待进程完全退出
    echo "等待训练进程完全退出..."
    sleep 5
    
    # 强制清理 GPU 内存
    python3 -c "
import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print('PyTorch GPU 缓存已清理')
    # 显示每个 GPU 的内存使用情况
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.memory_allocated(i)/1024**3:.2f}GB allocated, {torch.cuda.memory_reserved(i)/1024**3:.2f}GB reserved')
else:
    print('没有可用的 CUDA')
" 2>/dev/null || echo "无法清理 PyTorch 缓存"
    
    # 检查 GPU 状态
    echo "GPU 状态检查："
    nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits || echo "无法查询 GPU 状态"
    
    # 检查 Ray 集群状态
    echo "Ray 集群状态："
    ray status 2>/dev/null || echo "Ray 集群未运行或无法查询状态"
    
    echo "内存清理完成，Ray 集群保持运行！"
    echo "========================================"
}

# 运行单个实验的函数
run_experiment() {
    local script_name=$1
    local script_path="examples/scripts/$script_name"
    local log_file="$LOG_DIR/${script_name%.sh}_$(date +%Y%m%d_%H%M%S).log"
    
    echo "========================================"
    echo "开始运行实验: $script_name"
    echo "时间: $(date)"
    echo "日志文件: $log_file"
    echo "========================================"
    
    # 检查脚本是否存在
    if [ ! -f "$script_path" ]; then
        echo "错误: 脚本 $script_path 不存在!"
        return 1
    fi
    
    # 运行脚本并记录日志
    cd examples/scripts
    if timeout 7200 bash "$script_name" 2>&1 | tee "../../$log_file"; then
        echo "实验 $script_name 成功完成!"
        cd ../..
        return 0
    else
        echo "实验 $script_name 失败或超时!"
        cd ../..
        return 1
    fi
}

# 主函数
main() {
    echo "========================================"
    echo "开始自动化实验流程"
    echo "总共 ${#SCRIPTS[@]} 个实验"
    echo "开始时间: $(date)"
    echo "========================================"
    
    # 初始清理
    cleanup_gpu_memory
    
    local success_count=0
    local failed_experiments=()
    
    # 遍历运行所有实验
    for i in "${!SCRIPTS[@]}"; do
        local script="${SCRIPTS[$i]}"
        local experiment_num=$((i + 1))
        
        echo ""
        echo "========================================"
        echo "实验进度: $experiment_num/${#SCRIPTS[@]}"
        echo "当前实验: $script"
        echo "========================================"
        
        # 运行实验
        if run_experiment "$script"; then
            success_count=$((success_count + 1))
            echo "✅ 实验 $script 成功完成"
        else
            failed_experiments+=("$script")
            echo "❌ 实验 $script 失败"
        fi
        
        # 清理内存（除了最后一个实验）
        if [ $experiment_num -lt ${#SCRIPTS[@]} ]; then
            cleanup_gpu_memory
        fi
        
        echo "已完成 $success_count/${#SCRIPTS[@]} 个实验"
    done
    
    # 最终总结
    echo ""
    echo "========================================"
    echo "所有实验完成!"
    echo "结束时间: $(date)"
    echo "========================================"
    echo "成功完成: $success_count/${#SCRIPTS[@]} 个实验"
    
    if [ ${#failed_experiments[@]} -gt 0 ]; then
        echo "失败的实验:"
        for failed in "${failed_experiments[@]}"; do
            echo "  - $failed"
        done
    else
        echo "🎉 所有实验都成功完成!"
    fi
    
    echo "日志文件保存在: $LOG_DIR/"
    echo "========================================"
}

# 信号处理函数
cleanup_on_exit() {
    echo ""
    echo "收到退出信号，正在清理..."
    cleanup_gpu_memory
    exit 1
}

# 设置信号处理
trap cleanup_on_exit SIGINT SIGTERM

# 运行主函数
main "$@"
