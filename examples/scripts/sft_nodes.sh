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


export MASTER_ADDR=$(head -n 1 /xfr_ceph_sh/liuchonghan/OpenRLHF_lao/examples/scripts/hostfile.txt | awk '{print $1}')
export MASTER_PORT=29501
export WORLD_SIZE=16
export LOCAL_RANK=0

echo "🚀 主节点IP (从hostfile.txt自动读取): $MASTER_ADDR"


read -r -d '' training_commands <<EOF
openrlhf.cli.train_sft \
   --max_len 4096 \
   --dataset /xfr_ceph_sh/liuchonghan/sft_dataset \
   --input_key question \
   --output_key response \
   --train_batch_size 4096 \
   --micro_train_batch_size 1 \
   --max_samples 9000000 \
   --pretrain /llm-align/duyimin/duyimin/open_modle/Qwen2.5-7B-8Langs-CPT-250819 \
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
   --wandb_project 360_Repo \
   --wandb_run_name Qwen2_5_sft_0820 \
   --use_wandb 9c69c18b00c7dac67189f39e261a257ebd476cda
EOF

deepspeed --hostfile /xfr_ceph_sh/liuchonghan/OpenRLHF_lao/examples/scripts/hostfile.txt \
          --master_addr $MASTER_ADDR \
          --master_port $MASTER_PORT \
          --export https_proxy,http_proxy,WANDB_PROXY,WANDB_HTTP_PROXY,WANDB_HTTPS_PROXY \
          --module $training_commands 

