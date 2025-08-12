set -x

export TRANSFORMERS_CACHE="/llm-align/liuchonghan/hf_cache"  
export HF_HOME="/llm-align/liuchonghan/hf_home"             
export TRITON_CACHE_DIR="/llm-align/liuchonghan/triton_cache"  
export TORCH_HOME="/llm-align/liuchonghan/torch_home"       
mkdir -p /llm-align/liuchonghan/hf_cache
mkdir -p /llm-align/liuchonghan/hf_home
mkdir -p /llm-align/liuchonghan/triton_cache
mkdir -p /llm-align/liuchonghan/torch_home
export CUDA_LAUNCH_BLOCKING=1
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export WANDB_API_KEY="9c69c18b00c7dac67189f39e261a257ebd476cda"
#export NCCL_DEBUG=INFO
#export NCCL_DEBUG_SUBSYS=ALL
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=eth0 
export NCCL_TIMEOUT=1800
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=0

read -r -d '' training_commands <<EOF
openrlhf.cli.train_rm \
   --save_path ./checkpoint/qwen2_5-7b-rm-domain-0812 \
   --save_steps -1 \
   --logging_steps 1 \
   --eval_steps -1 \
   --train_batch_size 128 \
   --micro_train_batch_size 16 \
   --pretrain /llm-align/duyimin/multi_lang/Qwen2.5-7B-8Langs-CPT-250724 \
   --bf16 \
   --max_epochs 1 \
   --max_len 4096 \
   --learning_rate 9e-6 \
   --dataset /llm-align/liuchonghan/OpenRLHF/0807_data \
   --apply_chat_template \
   --chosen_key chosen \
   --rejected_key rejected \
   --load_checkpoint \
   --packing_samples \
   --gradient_checkpointing \
   --use_wandb False \
   --value_head_prefix score \
   --no_shuffle
EOF

if [[ ${1} != "slurm" ]]; then
    deepspeed --master_addr localhost --master_port 29500 \
              --include localhost:0,1,2,3,4,5,6,7 \
              --module $training_commands
fi