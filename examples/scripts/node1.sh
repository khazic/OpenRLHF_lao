# -*- coding: utf-8 -*-
set -x

export TRANSFORMERS_CACHE="/xfr_ceph_sh/liuchonghan/hf_cache"  # HuggingFace缓存
export HF_HOME="/xfr_ceph_sh/liuchonghan/hf_home"             # HuggingFace主目录
export TRITON_CACHE_DIR="/xfr_ceph_sh/liuchonghan/triton_cache"  # Triton缓存
export TORCH_HOME="/xfr_ceph_sh/liuchonghan/torch_home"       # PyTorch缓存
mkdir -p /xfr_ceph_sh/liuchonghan/hf_cache
mkdir -p /xfr_ceph_sh/liuchonghan/hf_home
mkdir -p /xfr_ceph_sh/liuchonghan/triton_cache
mkdir -p /xfr_ceph_sh/liuchonghan/torch_home

export CUDA_LAUNCH_BLOCKING=1
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export WANDB_API_KEY="9c69c18b00c7dac67189f39e261a257ebd476cda" 

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=eth0
export NCCL_TIMEOUT=1800
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=0

export MASTER_ADDR="11.131.241.226"  # 主节点IP
export MASTER_PORT=29500
export NNODES=2
export NODE_RANK=0

read -r -d '' training_commands <<EOF
openrlhf.cli.train_sft \
   --max_len 4096 \
   --dataset /xfr_ceph_sh/liuchonghan/OpenRLHF/domain_sft \
   --input_key question \
   --output_key response \
   --train_batch_size 2048 \
   --micro_train_batch_size 1 \
   --max_samples 8000000 \
   --pretrain /llm-align/duyimin/multi_lang/Qwen2.5-7B-8Langs-CPT-250724 \
   --save_path ./checkpoint/qwen2_5_sft_domain \
   --save_steps 2000 \
   --logging_steps 3 \
   --eval_steps 1000 \
   --max_epochs 1 \
   --bf16 \
   --flash_attn \
   --learning_rate 4e-6 \
   --load_checkpoint \
   --use_wandb True \
   --gradient_checkpointing \
   --packing_samples \
   --apply_chat_template
EOF

if [[ ${1} != "slurm" ]]; then
    deepspeed --master_addr $MASTER_ADDR \
              --master_port $MASTER_PORT \
              --num_nodes $NNODES \
              --node_rank $NODE_RANK \
              --num_gpus 8 \
              --module $training_commands
fi
