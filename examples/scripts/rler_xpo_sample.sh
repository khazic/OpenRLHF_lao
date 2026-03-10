set -x

if [ "$CONDA_DEFAULT_ENV" != "openrlhf" ]; then
    echo "Warning: conda environment is not openrlhf, current environment: $CONDA_DEFAULT_ENV"
    source /xfr_ceph_sh/liuchonghan/envs/etc/profile.d/conda.sh
    conda activate openrlhf
fi

export PYTHONPATH=/xfr_ceph_sh/liuchonghan/OpenRLHF:$PYTHONPATH

if ! python3 -c "import ray" 2>/dev/null; then
    echo "Installing Ray..."
fi

# xPO training script - similar to GRPO but with different reward normalization
# 
# Key modifications from GRPO:
# 1. Reward normalization strategy: Changed from group std to batch std
#    - GRPO: (rewards - group_mean) / group_std
#    - xPO:  (rewards - group_mean) / batch_std
#    Using batch-level variance for normalization provides more stable training
#
# 2. KL estimator: Changed from k3 to k2
#    - k3 typically has higher variance, k2 is more stable
#    - Used with use_kl_loss for more stable KL control
#
# 3. KL coefficient adjustment: Increased from 1e-3 to 5e-3
#    - Makes policy updates more aggressive, reduces over-conservatism
#    - Allows larger policy exploration while maintaining stability
#
# 4. Entropy reward enabled: Set entropy_loss_coef to 0.02
#    - Builds larger reward signals for exploration
#    - Prevents premature policy convergence, enhances policy diversity
#    - Balances the stability brought by xPO's batch variance normalization
#
# 5. PPO clip range: Added custom clip range
#    - Set eps_clip_low_high to 0.15 0.25 for more controlled policy updates
#    - Provides asymmetric clipping for better exploration-exploitation balance
#
# 6. Length penalty mechanism: Added overlong sequence penalty
#    - overlong_buffer_len: 256 tokens as length threshold
#    - overlong_penalty_factor: 1.0 penalty per excess token
#    - Encourages concise and focused responses


python3 -m openrlhf.cli.train_ppo_ray \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node 8 \
   --reward_num_nodes 1 \
   --reward_num_gpus_per_node 8 \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node 8 \
   --vllm_num_engines 2 \
   --vllm_tensor_parallel_size 4 \
   --vllm_gpu_memory_utilization 0.6 \
   --init_kl_coef 5e-3 \
   --gamma 1.0 \
   --colocate_all_models \
   --eps_clip_low_high 0.15 0.25 \
   --use_kl_loss \
   --kl_estimator k2 \
   --advantage_estimator xpo \
   --overlong_penalty_factor 1 \
   --overlong_buffer_len 256 \
   --entropy_loss_coef 0.02 \
   --pretrain /xfr_ceph_sh/liuchonghan/OpenRLHF_lao/examples/scripts/checkpoint/SFTmodel_0823 \
   --reward_pretrain /xfr_ceph_sh/liuchonghan/OpenRLHF_lao/examples/scripts/checkpoint/RewardModel_Qwen_0825_translate \
   --save_path ./checkpoint/RLer_xpo_0826 \
   --ckpt_path ./checkpoint/RLer_xpo_0826_ckpt \
   --save_hf_ckpt \
   --rollout_batch_size 32 \
   --n_samples_per_prompt 4 \
   --train_batch_size 32 \
   --micro_train_batch_size 4 \
   --micro_rollout_batch_size 4 \
   --max_epochs 1 \
   --prompt_max_len 8192 \
   --max_samples 500000 \
   --generate_max_len 8192 \
   --zero_stage 3 \
   --bf16 \
   --actor_learning_rate 5e-7 \
   --prompt_data /xfr_ceph_sh/liuchonghan/prompt_dataset \
   --input_key context_messages \
   --apply_chat_template \
   --normalize_reward \
   --gradient_checkpointing \
   --packing_samples \
   --vllm_sync_backend nccl \
   --enforce_eager \
   --vllm_enable_sleep \
   --deepspeed_enable_sleep \
   --use_wandb 9c69c18b00c7dac67189f39e261a257ebd476cda \
   --wandb_project 360_Repo \
   --wandb_run_name RLer_xpo_0826