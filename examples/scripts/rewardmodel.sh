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
export NCCL_DEBUG=ERROR  
export NCCL_DEBUG_SUBSYS=NONE 

read -r -d '' training_commands <<EOF
openrlhf.cli.train_rm \
   --save_path ./checkpoint/RewardModel_0904_translate_2 \
   --save_steps -1 \
   --logging_steps 2 \
   --eval_steps 200 \
   --train_batch_size 16 \
   --micro_train_batch_size 2 \
   --pretrain /xfr_ceph_sh/liuchonghan/OpenRLHF_lao/examples/scripts/checkpoint/SFTmodel_0823 \
   --bf16 \
   --max_epochs 1 \
   --max_len 8192 \
   --learning_rate 1e-6 \
   --dataset /xfr_ceph_sh/liuchonghan/tranlate_datset \
   --chosen_key chosen \
   --rejected_key rejected \
   --max_samples 10000000 \
   --prompt_key prompt \
   --attn_implementation flash_attention_2 \
   --packing_samples \
   --gradient_checkpointing \
   --value_head_prefix score \
   --use_wandb 9c69c18b00c7dac67189f39e261a257ebd476cda \
   --wandb_project 360_Repo \
   --wandb_run_name reward_model_translate_0904_2
EOF

if [[ ${1} != "slurm" ]]; then
    NCCL_DEBUG=ERROR deepspeed --master_addr localhost --master_port 29500 \
              --include localhost:0,1,2,3,4,5,6,7 \
              --module $training_commands
fi