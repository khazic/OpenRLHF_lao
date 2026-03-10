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

export CUDA_LAUNCH_BLOCKING=0
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1

export NCCL_DEBUG=ERROR  
export NCCL_DEBUG_SUBSYS=NONE  

export MASTER_ADDR=$(head -n 1 /xfr_ceph_sh/liuchonghan/OpenRLHF_lao/examples/scripts/hostfile.txt | awk '{print $1}')
export MASTER_PORT=29501
export WORLD_SIZE=24
export LOCAL_RANK=0

echo "🚀  Master node IP (auto-read from hostfile.txt): $MASTER_ADDR"

read -r -d '' training_commands <<EOF
openrlhf.cli.train_sft \
   --max_len 4096 \
   --dataset /xfr_ceph_sh/liuchonghan/sft_translate_dataset/SFT_M_QA.json,/xfr_ceph_sh/liuchonghan/sft_translate_dataset/SFT_M_QA_2.json,/xfr_ceph_sh/liuchonghan/sft_translate_dataset/SFT_M_Translate.json,/xfr_ceph_sh/liuchonghan/sft_translate_dataset/SFT_M_Translate_2.json \
   --train_batch_size 7680 \
   --input_key question \
   --output_key response \
   --micro_train_batch_size 16 \
   --max_samples 90000000 \
   --pretrain /llm-align/duyimin/duyimin/open_modle/Qwen2.5-7B-8Langs-CPT-250819 \
   --save_path ./checkpoint/RLer_0924 \
   --save_steps 3000 \
   --logging_steps 3 \
   --eval_steps 100000 \
   --max_epochs 1 \
   --bf16 \
   --attn_implementation flash_attention_2 \
   --learning_rate 5e-6 \
   --gradient_checkpointing \
   --packing_samples \
   --apply_chat_template \
   --wandb_project SFT_360_Repo \
   --wandb_run_name RLer_SFT_0924 \
   --use_wandb 9c69c18b00c7dac67189f39e261a257ebd476cda
EOF

export DS_SSH_PASSWORD=1
export DS_SSH_PASSWORD_AUTH=true
export DS_SSH_OPTS="-o PasswordAuthentication=yes -o PubkeyAuthentication=no"

deepspeed --hostfile /xfr_ceph_sh/liuchonghan/OpenRLHF_lao/examples/scripts/hostfile.txt \
          --master_addr $MASTER_ADDR \
          --master_port $MASTER_PORT \
          --module $training_commands 
