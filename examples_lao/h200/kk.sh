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
export NCCL_SOCKET_IFNAME=net0

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
   --vllm_gpu_memory_utilization 0.8 \
   --colocate_all_models \
   --advantage_estimator group_norm \
  --pretrain /mnt/data/liuchonghan/75_0129_ckpt3000 \
  --remote_rm_url ${REPO_ROOT}/examples/python/mc_rlvr_reward_func.py \
  --save_path ./checkpoint/Vi-VMLU_main_0202 \
  --ckpt_path ./ckpt/Vi-VMLU_main_0202 \
   --save_hf_ckpt \
   --rollout_batch_size 32 \
   --n_samples_per_prompt 16 \
   --train_batch_size 512 \
   --micro_train_batch_size 4 \
   --micro_rollout_batch_size 4 \
   --max_epochs 1 \
   --prompt_max_len 1024 \
   --max_samples 50000000 \
   --generate_max_len 8192 \
   --zero_stage 3 \
   --bf16 \
   --actor_learning_rate 1e-6 \
   --init_kl_coef 0.01 \
  --prompt_data /mnt/data/liuchonghan/vmlu_dataset/all_data_merged.json \
   --input_key question \
   --label_key response \
   --apply_chat_template \
   --normalize_reward \
   --gradient_checkpointing \
   --packing_samples \
   --vllm_sync_backend nccl \
   --enforce_eager \
   --vllm_enable_sleep \
   --deepspeed_enable_sleep \
   --enable_vllm_is_correction \
   --enable_ema \
   --num_episodes 3 \
   --save_steps 200

# --dynamic_filtering \
# --dynamic_filtering_reward_range 0.1 0.9 \
#    --disable_ds_ckpt \