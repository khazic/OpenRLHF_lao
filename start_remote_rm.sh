#!/bin/bash
echo "Starting Remote Reward Model Server..."

if [ "$CONDA_DEFAULT_ENV" != "openrlhf" ]; then
    echo "Activating openrlhf environment..."
    source /xfr_ceph_sh/liuchonghan/envs/etc/profile.d/conda.sh
    conda activate openrlhf
fi

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TOKENIZERS_PARALLELISM=false

# 启动服务器
# 启动多个实例（可选）
python3 remote_reward_server.py --port 8000 &
python3 remote_reward_server.py --port 8001 &

# 等待所有后台进程
wait

echo "Remote Reward Model Server stopped."
