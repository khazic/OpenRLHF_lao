#!/bin/bash
set -x

# 基础环境设置
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"  # 明确指定GPU
export CUDA_LAUNCH_BLOCKING=1  # 同步执行

# 网络配置 - 更保守的设置
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=eth0
export NCCL_MIN_NCHANNELS=1
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_SOCKET_NTHREADS=1
export NCCL_NSOCKS_PERTHREAD=1
export NCCL_BUFFSIZE=16777216
export NCCL_NET_GDR_LEVEL=0

# 超时设置 - 更长的超时时间
export NCCL_TIMEOUT=3600
export DS_COMM_TIMEOUT=3600
export DS_HEARTBEAT_TIMEOUT=3600
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export TORCH_CPP_LOG_LEVEL=INFO
export TORCH_DISTRIBUTED_DETAIL=DEBUG

# SSH配置 - 更稳定的连接
export DS_SSH_TIMEOUT=3600
export DS_SSH_PORT_TIMEOUT=3600
export DS_SSH_CONNECT_TIMEOUT=60
export DS_SSH_PASSWORD=1
export DS_SSH_PASSWORD_AUTH=true
export DS_SSH_OPTS="-o ServerAliveInterval=15 -o ServerAliveCountMax=10 -o ConnectTimeout=60 -o ConnectionAttempts=5 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"

# 主节点配置
export MASTER_PORT=29501
export MASTER_ADDR=$(hostname -i)  # 使用IP而不是主机名
export WORLD_SIZE=16

# 检查网络连接
echo "🚀 Testing network connectivity..."
for host in $(awk '{print $1}' hostfile.txt); do
    echo "Testing connection to $host..."
    if ! ping -c 1 -W 5 $host &>/dev/null; then
        echo "ERROR: Cannot ping $host"
        exit 1
    fi
    if ! ssh -o ConnectTimeout=5 $host "echo Connected to $host" &>/dev/null; then
        echo "ERROR: Cannot SSH to $host"
        exit 1
    fi
done

# 训练命令
read -r -d '' training_commands <<EOF
openrlhf.cli.train_sft \
   --max_len 4096 \
   --dataset /xfr_ceph_sh/liuchonghan/test_dataset \
   --input_key question \
   --output_key response \
   --train_batch_size 8192 \
   --micro_train_batch_size 2 \
   --max_samples 10000000 \
   --pretrain /llm-align/duyimin/duyimin/open_modle/Qwen2.5-7B-8Langs-CPT-250819 \
   --save_path ./checkpoint/Qwen2_5_sft_0823 \
   --save_steps 3000 \
   --logging_steps 3 \
   --eval_steps 1000 \
   --max_epochs 1 \
   --bf16 \
   --attn_implementation flash_attention_2 \
   --learning_rate 5e-6 \
   --gradient_checkpointing \
   --packing_samples \
   --apply_chat_template \
   --wandb_project 360_Repo \
   --wandb_run_name Qwen2_5_sft_0823 \
   --use_wandb 9c69c18b00c7dac67189f39e261a257ebd476cda
EOF

# 启动训练
deepspeed --hostfile hostfile.txt \
          --master_addr $MASTER_ADDR \
          --master_port $MASTER_PORT \
          --no_local_rank \
          --force_multi \
          --no_ssh \
          --module $training_commands 

