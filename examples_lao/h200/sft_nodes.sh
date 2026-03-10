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

export MASTER_ADDR=22.25.250.89
export MASTER_PORT=29501
TOTAL_NODES=${TOTAL_NODES:-12}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
export WORLD_SIZE=$((TOTAL_NODES * GPUS_PER_NODE))
export LOCAL_RANK=${LOCAL_RANK:-0}
MICRO_TRAIN_BATCH_SIZE=${MICRO_TRAIN_BATCH_SIZE:-16}
GRADIENT_ACCUM_STEPS=${GRADIENT_ACCUM_STEPS:-1}
TRAIN_BATCH_SIZE=$((MICRO_TRAIN_BATCH_SIZE * GRADIENT_ACCUM_STEPS * WORLD_SIZE))

echo "🚀  Master node IP: $MASTER_ADDR"
echo "🚀  Total nodes:${TOTAL_NODES}"
echo "🚀  Total GPUs:$WORLD_SIZE ($GPUS_PER_NODE per node)"
echo "🚀  Micro batch:$MICRO_TRAIN_BATCH_SIZE  Grad accum:$GRADIENT_ACCUM_STEPS  Global batch:$TRAIN_BATCH_SIZE"

read -r -d '' training_commands <<EOF
openrlhf.cli.train_sft \
   --max_len 8192 \
   --dataset /mnt/data/liuchonghan/8b_arrow \
   --train_batch_size $TRAIN_BATCH_SIZE \
   --input_key question \
   --output_key response \
   --micro_train_batch_size $MICRO_TRAIN_BATCH_SIZE \
   --max_samples 5000000000 \
   --pretrain /mnt/data/liuchonghan/8b_0202 \
   --save_path ./checkpoint/Qwen8bckpt_ckpt0202_standardloss_2ep \
   --ckpt_path ./ckpt/Qwen8bckpt_ckpt0202_standardloss_2ep \
   --save_hf_ckpt \
   --save_steps -1 \
   --logging_steps 2 \
   --sft_loss encouraging \
   --eval_steps -1 \
   --packing_samples  \
   --max_epochs 1 \
   --bf16 \
   --attn_implementation flash_attention_2 \
   --learning_rate 5e-6 \
   --apply_chat_template \
   --lr_warmup_ratio 0.05 \
   --gradient_checkpointing
EOF
export DS_SSH_PASSWORD=1
export DS_SSH_PASSWORD_AUTH=true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOSTFILE="${SCRIPT_DIR}/hostfile_12nodes.txt"
DEEPSPEED_BIN="${OPENRLHF_PREFIX}/bin/deepspeed"

"$DEEPSPEED_BIN" --hostfile "$HOSTFILE" \
                 --master_addr $MASTER_ADDR \
                 --master_port $MASTER_PORT \
                 --module $training_commands
