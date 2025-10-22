# -*- coding: utf-8 -*-
set -x

export WANDB_MODE=offline
export HF_DATASETS_DISABLE_MULTIPROCESSING=1

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

export MASTER_ADDR=22.25.243.26
export MASTER_PORT=29501
export WORLD_SIZE=64
export LOCAL_RANK=0

echo "🚀  Master node IP: $MASTER_ADDR"
echo "🚀  Total nodes: 8"
echo "🚀  Total GPUs: 64 (8 per node)"

read -r -d '' training_commands <<EOF
openrlhf.cli.train_sft \
   --max_len 30000 \
   --dataset /mnt/data/liuchonghan/longcontext_dataset \
   --train_batch_size 512 \
   --input_key question \
   --output_key response \
   --micro_train_batch_size 4 \
   --max_samples 90000000 \
   --pretrain /mnt/data/liuchonghan/Qwen_cpt \
   --save_path ./checkpoint/RLer_1021_elloss_long \
   --save_steps 3000 \
   --logging_steps 3 \
   --eval_steps 500 \
   --max_epochs 1 \
   --sft_loss encouraging \
   --bf16 \
   --attn_implementation flash_attention_2 \
   --learning_rate 5e-6 \
   --gradient_checkpointing \
   --packing_samples \
   --apply_chat_template \
   --cache_dataset_to_disk \
   --dataset_cache_dir /mnt/data/liuchonghan/dataset_cache \
   --use_wandb 9c69c18b00c7dac67189f39e261a257ebd476cda \
   --wandb_org "khazzz1c" \
   --wandb_project SFTLongContext \
   --wandb_run_name RLer_1021_elloss_long
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