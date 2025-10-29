# -*- coding: utf-8 -*-
set -x

export WANDB_MODE=offline
export HF_DATASETS_DISABLE_MULTIPROCESSING=0

export TRANSFORMERS_CACHE="/mnt/data/liuchonghan/hf_cache"
export HF_HOME="/mnt/data/liuchonghan/hf_home"
export TRITON_CACHE_DIR="/mnt/data/liuchonghan/triton_cache"
export TORCH_HOME="/mnt/data/liuchonghan/torch_home"
OPENRLHF_PREFIX="/mnt/data/liuchonghan/env/openrlhf"
OPENRLHF_SITE="$OPENRLHF_PREFIX/lib/python3.10/site-packages"

if [[ ":$PATH:" != *":$OPENRLHF_PREFIX/bin:"* ]]; then
  export PATH=/mnt/data/liuchonghan/env/openrlhf/bin:$PATH
fi
if [[ -d "$OPENRLHF_SITE" && ":$PYTHONPATH:" != *":$OPENRLHF_SITE:"* ]]; then
  export PYTHONPATH=/mnt/data/liuchonghan/OpenRLHF_lao:$PYTHONPATH
fi
export CONDA_PREFIX="$OPENRLHF_PREFIX"

mkdir -p /mnt/data/liuchonghan/hf_cache
mkdir -p /mnt/data/liuchonghan/hf_home
mkdir -p /mnt/data/liuchonghan/triton_cache
mkdir -p /mnt/data/liuchonghan/torch_home

export CUDA_LAUNCH_BLOCKING=0
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1

export NCCL_DEBUG=ERROR  
export NCCL_DEBUG_SUBSYS=NONE  

export NCCL_SOCKET_TIMEOUT=3600000
export NCCL_HEARTBEAT_TIMEOUT_SEC=300
export NCCL_CONNECT_TIMEOUT=600
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_IB_TIMEOUT=22
export NCCL_IB_RETRY_CNT=7

export DS_TIMEOUT=1800
export DS_ELASTIC_TIMEOUT=1800

export MASTER_ADDR=22.25.243.26
export MASTER_PORT=29501
export WORLD_SIZE=64
export LOCAL_RANK=0

echo "🚀   Master node IP: $MASTER_ADDR"
echo "🚀   Total nodes: 8"
echo "🚀   Total GPUs: 64 (8 per node)"

read -r -d '' training_commands <<EOF
openrlhf.cli.train_sft \
   --max_len 8192 \
   --dataset /mnt/data/liuchonghan/72b_dataset_arrow \
   --train_batch_size 192 \
   --input_key question \
   --output_key response \
   --micro_train_batch_size 4 \
   --max_samples 50000000 \
   --pretrain /mnt/data/liuchonghan/Qwen72b_cpt \
   --save_path ./checkpoint/RLer_Qwen72ckpt \
   --save_steps -1 \
   --logging_steps 1 \
   --eval_steps -1 \
   --packing_samples \
   --max_epochs 1 \
   --sft_loss encouraging \
   --bf16 \
   --attn_implementation flash_attention_2 \
   --learning_rate 7e-6 \
   --gradient_checkpointing \
   --apply_chat_template \
   --lr_warmup_ratio 0.1 \
   --ds_tensor_parallel_size 4 \
   --zero_stage 2 \
   --zpg 8 \
   --overlap_comm 
EOF

export DS_SSH_PASSWORD=1
export DS_SSH_PASSWORD_AUTH=true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOSTFILE="${SCRIPT_DIR}/hostfile_4nodes.txt"
DEEPSPEED_BIN="${OPENRLHF_PREFIX}/bin/deepspeed"

"$DEEPSPEED_BIN" --hostfile "$HOSTFILE" \
                 --master_addr $MASTER_ADDR \
                 --master_port $MASTER_PORT \
                 --module $training_commands