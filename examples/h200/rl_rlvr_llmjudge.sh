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
   --ref_num_nodes 10 \
   --ref_num_gpus_per_node 8 \
   --reward_num_nodes 0 \
   --reward_num_gpus_per_node 0 \
   --critic_num_nodes 10 \
   --critic_num_gpus_per_node 8 \
   --actor_num_nodes 10 \
   --actor_num_gpus_per_node 8 \
   --vllm_num_engines 10 \
   --vllm_tensor_parallel_size 8 \
   --colocate_all_models \
   --vllm_gpu_memory_utilization 0.8 \
   --advantage_estimator gae \
   --pretrain /mnt/data/liuchonghan/checkpoint_model/RLer_policy_model \
   --remote_rm_url http://10.181.107.66:5000/get_reward \
   --save_path ./checkpoint/RL_llmjudge \
   --ckpt_path ./checkpoint/RL_llmjudge_ckpt \
   --save_hf_ckpt \
   --rollout_batch_size 80 \
   --n_samples_per_prompt 16 \
   --train_batch_size 80 \
   --micro_train_batch_size 2 \
   --micro_rollout_batch_size 1 \
   --max_epochs 1 \
   --prompt_max_len 2048 \
   --max_samples 900000 \
   --generate_max_len 8192 \
   --zero_stage 3 \
   --bf16 \
   --actor_learning_rate 5e-7 \
   --critic_learning_rate 5e-7 \
   --init_kl_coef 0.01 \
   --prompt_data /mnt/data/liuchonghan/prompt_benchmark_arrow \
   --input_key transed_prompt \
   --label_key answer \
   --apply_chat_template \
   --normalize_reward \
   --gradient_checkpointing \
   --packing_samples \
   --vllm_sync_backend nccl \
   --enforce_eager \
   --vllm_enable_sleep \
   --deepspeed_enable_sleep