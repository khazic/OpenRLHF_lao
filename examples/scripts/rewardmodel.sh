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
   --save_path ./checkpoint/RewardModel_Qwen2.5_7b \
   --save_steps -1 \
   --logging_steps 2 \
   --eval_steps 20 \
   --train_batch_size 256 \
   --micro_train_batch_size 16 \
   --pretrain /xfr_ceph_sh/liuchonghan/checkpoints_set/qwen2_5_sft_domain \
   --bf16 \
   --max_epochs 1 \
   --max_len 4096 \
   --learning_rate 1e-6 \
   --dataset /xfr_ceph_sh/liuchonghan/reward_dataset \
   --apply_chat_template \
   --chosen_key chosen \
   --rejected_key rejected \
   --load_checkpoint \
   --packing_samples \
   --gradient_checkpointing \
   --value_head_prefix score \
   --no_shuffle \
   --use_wandb 9c69c18b00c7dac67189f39e261a257ebd476cda \
   --wandb_project 360_Repo \
   --wandb_run_name rm-1
EOF

if [[ ${1} != "slurm" ]]; then
    deepspeed --master_addr localhost --master_port 29500 \
              --include localhost:0,1,2,3,4,5,6,7 \
              --module $training_commands
fi