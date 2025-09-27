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

export https_proxy=http://lidongming:YqN2VZBHtkYe3aNA@proxy.aidataset.qihoo.net:8000/
export http_proxy=http://lidongming:YqN2VZBHtkYe3aNA@proxy.aidataset.qihoo.net:8000/
export WANDB_PROXY=http://lidongming:YqN2VZBHtkYe3aNA@proxy.aidataset.qihoo.net:8000/
export WANDB_HTTP_PROXY=http://lidongming:YqN2VZBHtkYe3aNA@proxy.aidataset.qihoo.net:8000/
export WANDB_HTTPS_PROXY=http://lidongming:YqN2VZBHtkYe3aNA@proxy.aidataset.qihoo.net:8000/

read -r -d '' training_commands <<EOF
openrlhf.cli.train_sft \
   --max_len 4096 \
   --dataset /xfr_ceph_sh/liuchonghan/sft_dataset\
   --train_batch_size 32 \
   --input_key question \
   --output_key response \
   --micro_train_batch_size 2 \
   --max_samples 90000000 \
   --pretrain /llm-align/duyimin/duyimin/open_modle/Qwen2.5-7B-8Langs-CPT-250819 \
   --save_path ./checkpoint/RLer_0926_Universal \
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
   --wandb_run_name RLer_SFT_0926_Universal \
   --use_wandb 9c69c18b00c7dac67189f39e261a257ebd476cda
EOF

deepspeed --include localhost:0,1,2,3,4,5,6,7 \
          --module $training_commands 

