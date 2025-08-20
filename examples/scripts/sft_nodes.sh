# -*- coding: utf-8 -*-
set -x

export TRANSFORMERS_CACHE="/xfr_ceph_sh/liuchonghan/hf_cache"
export HF_HOME="/xfr_ceph_sh/liuchonghan/hf_home"
export TRITON_CACHE_DIR="/xfr_ceph_sh/liuchonghan/triton_cache"
export TORCH_HOME="/xfr_ceph_sh/liuchonghan/torch_home"
mkdir -p /xfr_ceph_sh/liuchonghan/hf_cache
mkdir -p /xfr_ceph_sh/liuchonghan/hf_home
mkdir -p /xfr_ceph_sh/liuchonghan/triton_cache
mkdir -p /xfr_ceph_sh/liuchonghan/torch_home

export CUDA_LAUNCH_BLOCKING=1
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export WANDB_API_KEY="9c69c18b00c7dac67189f39e261a257ebd476cda"
export WANDB_MODE=offline

export NCCL_DEBUG=INFO
export TOKENIZERS_PARALLELISM=true
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=eth0
export NCCL_TIMEOUT=1800
export NCCL_SHM_DISABLE=0
export NCCL_SOCKET_FAMILY=IPv4
export GLOO_SOCKET_IFNAME=eth0
export MASTER_PORT=29500
export WORLD_SIZE=16  
export LOCAL_RANK=0   

export MASTER_ADDR=$(head -n 1 hostfile.txt | awk '{print $1}')
echo "🚀 主节点IP (从hostfile.txt自动读取): $MASTER_ADDR"

deepspeed --hostfile hostfile.txt \
          --master_addr $MASTER_ADDR \
          --master_port 29500 \
          --ssh_args="-i /xfr_ceph_sh/liuchonghan/.ssh/id_rsa -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null" \
          --module openrlhf.cli.train_sft \
          --max_len 4096 \
          --dataset /xfr_ceph_sh/liuchonghan/sft_dataset \
          --input_key question \
          --output_key response \
          --train_batch_size 4096 \
          --micro_train_batch_size 1 \
          --max_samples 9000000 \
          --pretrain /llm-align/duyimin/duyimin/open_modle/Qwen2.5-7B-8Langs-CPT-250819/ \
          --save_path ./checkpoint/Qwen2_5_sft_0820 \
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
          --wandb_project "360_Repo" \
          --wandb_run_name "Qwen2_5_sft_0820"
