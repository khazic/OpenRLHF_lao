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

export MASTER_ADDR=22.25.242.58
export MASTER_PORT=29501
TOTAL_NODES=${TOTAL_NODES:-1}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
export WORLD_SIZE=$((TOTAL_NODES * GPUS_PER_NODE))
export LOCAL_RANK=${LOCAL_RANK:-0}
MICRO_TRAIN_BATCH_SIZE=${MICRO_TRAIN_BATCH_SIZE:-8}
GRADIENT_ACCUM_STEPS=${GRADIENT_ACCUM_STEPS:-1}
TRAIN_BATCH_SIZE=$((MICRO_TRAIN_BATCH_SIZE * GRADIENT_ACCUM_STEPS * WORLD_SIZE))

echo "🚀  Master node IP: $MASTER_ADDR"
echo "🚀  Total nodes:${TOTAL_NODES}"
echo "🚀   Total GPUs:$WORLD_SIZE ($GPUS_PER_NODE per node)"
echo "🚀  Micro batch:$MICRO_TRAIN_BATCH_SIZE  Grad accum:$GRADIENT_ACCUM_STEPS  Global batch:$TRAIN_BATCH_SIZE"

read -r -d '' training_commands <<EOF
openrlhf.cli.train_sft \
   --max_len 8192 \
   --dataset /mnt/data/liuchonghan/1215_arrow \
   --train_batch_size $TRAIN_BATCH_SIZE \
   --save_hf_ckpt \
   --input_key question \
   --output_key response \
   --micro_train_batch_size $MICRO_TRAIN_BATCH_SIZE \
   --max_samples 90000000 \
   --pretrain /mnt/data/liuchonghan/Qwen3-8b-20b \
   --save_path ./checkpoint/RLer_Qwen7b_ckpt20b_1215_2e_elloss \
   --ckpt_path ./ckpt/RLer_Qwen7b_ckpt20b_1215_2e_elloss \
   --save_steps 1500 \
   --logging_steps 1 \
   --eval_steps -1 \
   --max_epochs 2 \
   --bf16 \
   --max_ckpt_num 2 \
   --sft_loss encouraging \
   --attn_implementation flash_attention_2 \
   --learning_rate 5e-6 \
   --lr_warmup_ratio 0.05 \
   --gradient_checkpointing \
   --packing_samples \
   --apply_chat_template
EOF

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOSTFILE="${SCRIPT_DIR}/hostfile_1nodes.txt"
DEEPSPEED_BIN="${OPENRLHF_PREFIX}/bin/deepspeed"

if [[ "$TOTAL_NODES" -gt 1 ]]; then
  if [[ ! -f "$HOSTFILE" ]]; then
    echo "Hostfile $HOSTFILE not found but multi-node training requested" >&2
    exit 1
  fi
  HOSTFILE_ARGS=(--hostfile "$HOSTFILE")
else
  echo "🚦  Single node detected, skipping hostfile and SSH checks"
  HOSTFILE_ARGS=()
fi

"$DEEPSPEED_BIN" "${HOSTFILE_ARGS[@]}" \
                 --master_addr $MASTER_ADDR \
                 --master_port $MASTER_PORT \
                 --module $training_commands
