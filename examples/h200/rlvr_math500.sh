#!/usr/bin/env bash
set -x

REPO_ROOT="/mnt/data/liuchonghan/OpenRLHF_lao"

if [[ "${CONDA_DEFAULT_ENV:-}" != "openrlhf" ]]; then
    echo "Warning: conda environment is not openrlhf, current environment: ${CONDA_DEFAULT_ENV:-none}"
    source ~/.bashrc
fi

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=eth0

python3 -m openrlhf.cli.train_ppo_ray \
   --ref_num_nodes 16 \
   --ref_num_gpus_per_node 8 \
   --reward_num_nodes 0 \
   --reward_num_gpus_per_node 0 \
   --critic_num_nodes 0 \
   --critic_num_gpus_per_node 0 \
   --actor_num_nodes 16 \
   --actor_num_gpus_per_node 8 \
   --vllm_num_engines 16 \
   --vllm_tensor_parallel_size 8 \
   --colocate_all_models \
   --vllm_gpu_memory_utilization 0.7 \
   --advantage_estimator reinforce_baseline \
   --pretrain /mnt/data/liuchonghan/Qwen2.5_72b_ckpt70b_0275/RLer_qwen72b_ckpt70b_200w_0277_70w_2e \
   --remote_rm_url ${REPO_ROOT}/examples/python/math_rlvr_reward_func.py \
   --save_path ./checkpoint/Qwen2.5_72b_ckpt70b_0275_rlvr_math500 \
   --ckpt_path ./ckpt/Qwen2.5_72b_ckpt70b_0275_rlvr_math500_ckpt \
   --save_hf_ckpt \
   --rollout_batch_size 512 \
   --n_samples_per_prompt 16 \
   --train_batch_size 1024 \
   --micro_train_batch_size 8 \
   --micro_rollout_batch_size 8 \
   --max_epochs 2 \
   --prompt_max_len 2048 \
   --max_samples 50000000 \
   --generate_max_len 8192 \
   --zero_stage 3 \
   --bf16 \
   --actor_learning_rate 5e-7 \
   --init_kl_coef 0.01 \
   --prompt_data /mnt/data/liuchonghan/rlvr_dataset/RL_openmath.json \
   --input_key question \
   --label_key response \
   --apply_chat_template \
   --normalize_reward \
   --gradient_checkpointing \
   --packing_samples \
   --vllm_sync_backend nccl \
   --enforce_eager \
   --vllm_enable_sleep \
   --deepspeed_enable_sleep

# --dynamic_filtering \
# --dynamic_filtering_reward_range 0.1 0.9 \