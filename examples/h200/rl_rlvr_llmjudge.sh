#!/usr/bin/env bash
set -x

REPO_ROOT="/mnt/data/liuchonghan/OpenRLHF_lao"
REMOTE_RM_URL="${REPO_ROOT}/examples/python/reward_func.py"

if [[ "${CONDA_DEFAULT_ENV:-}" != "openrlhf" ]]; then
    echo "Warning: conda environment is not openrlhf, current environment: ${CONDA_DEFAULT_ENV:-none}"
    source /mnt/data/liuchonghan/env/etc/profile.d/conda.sh
    conda activate openrlhf
fi

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

if ! python3 -c "import ray" 2>/dev/null; then
    echo "Installing Ray..."
    pip install ray
fi

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
   --vllm_gpu_memory_utilization 0.6 \
   --advantage_estimator gae \
   --pretrain /mnt/data/liuchonghan/Qwen7b_cpt \
   --remote_rm_url http://11.131.209.97:8000/reward \
   --save_path ./checkpoint/RL_rlvr_llmjudge \
   --ckpt_path ./checkpoint/RL_rlvr_llmjudge_ckpt \
   --save_hf_ckpt \
   --rollout_batch_size 128 \
   --n_samples_per_prompt 16 \
   --train_batch_size 256 \
   --micro_train_batch_size 16 \
   --micro_rollout_batch_size 8 \
   --max_epochs 1 \
   --prompt_max_len 4096 \
   --max_samples 900000 \
   --generate_max_len 8192 \
   --zero_stage 3 \
   --bf16 \
   --actor_learning_rate 5e-7 \
   --critic_learning_rate 5e-7 \
   --init_kl_coef 0.01 \
   --prompt_data /mnt/data/liuchonghan/prompt_dataset \
   --input_key context_messages \
   --apply_chat_template \
   --normalize_reward \
   --gradient_checkpointing \
   --packing_samples \
   --vllm_sync_backend nccl \
   --enforce_eager \
   --vllm_enable_sleep \
   --deepspeed_enable_sleep 

