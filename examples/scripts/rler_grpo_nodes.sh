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
python3 -m openrlhf.cli.train_ppo_ray \
   --ref_num_nodes 2 \
   --ref_num_gpus_per_node 8 \
   --reward_num_nodes 2 \
   --reward_num_gpus_per_node 8 \
   --actor_num_nodes 2 \
   --actor_num_gpus_per_node 8 \
   --vllm_num_engines 4 \
   --vllm_tensor_parallel_size 4 \
   --vllm_gpu_memory_utilization 0.6 \
   --init_kl_coef 1e-3 \
   --gamma 1.0 \
   --colocate_all_models \
   --use_kl_loss \
   --kl_estimator k2 \
   --advantage_estimator group_norm \
   --pretrain /xfr_ceph_sh/liuchonghan/checkpoints_set/qwen2_5_sft_domain \
   --reward_pretrain /xfr_ceph_sh/liuchonghan/checkpoints_set/rm-1 \
   --save_path ./checkpoint/qwen2_5_rler_grpo_main \
   --ckpt_path ./checkpoint/qwen2_5_rler_grpo_ckpt \
   --save_hf_ckpt \
   --rollout_batch_size 128 \
   --n_samples_per_prompt 8 \
   --train_batch_size 128 \
   --micro_train_batch_size 8 \
   --micro_rollout_batch_size 16 \
   --max_epochs 1 \
   --prompt_max_len 4096 \
   --max_samples 100000 \
   --generate_max_len 4096 \
   --zero_stage 3 \
   --bf16 \
   --actor_learning_rate 5e-7 \
   --prompt_data /xfr_ceph_sh/liuchonghan/grpo_dataset \
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
   --wandb_run_name qwen2_5_rler_grpo_main

# ============================================================================
# Parameter relationships that must hold (Critical Parameter Constraints)
# ============================================================================

# [GPU resource constraints]
# 1. Total GPU count calculation:
#    - Actor GPUs: ref_num_nodes × ref_num_gpus_per_node = 2 × 8 = 16 GPUs
#    - Reference GPUs: actor_num_nodes × actor_num_gpus_per_node = 2 × 8 = 16 GPUs  
#    - Reward GPUs: reward_num_nodes × reward_num_gpus_per_node = 2 × 8 = 16 GPUs
#    - VLLM GPUs: vllm_num_engines × vllm_tensor_parallel_size = 4 × 4 = 16 GPUs
#    Total requirement: 64 GPUs (if colocate_all_models=True, only 16 GPUs are needed in practice)

# 2. VLLM engine constraints:
#    - vllm_tensor_parallel_size must evenly divide the GPU count
#    - vllm_num_engines × vllm_tensor_parallel_size ≤ total available GPUs
#    - Recommended: vllm_tensor_parallel_size = power of two (1,2,4,8)

# [Batch size constraints]
# 3. Training batch relationship:
#    - train_batch_size must be divisible by micro_train_batch_size
#    - Current: 128 ÷ 8 = 16 (each training step needs 16 GPU micro-batches)
#    - train_batch_size must be ≤ (actor_num_nodes × actor_num_gpus_per_node × micro_train_batch_size)
#    - Current check: 128 ≤ (2 × 8 × 8) = 128 ✓

# 4. Rollout batch relationship:
#    - rollout_batch_size must be divisible by micro_rollout_batch_size  
#    - Current: 128 ÷ 16 = 8 (each rollout step needs 8 GPU micro-batches)
#    - rollout_batch_size must be ≤ (vllm_num_engines × micro_rollout_batch_size × concurrency factor)

# [Memory and sequence length constraints]
# 5. Sequence length limits:
#    - prompt_max_len + generate_max_len ≤ model maximum sequence length
#    - Current: 4096 + 4096 = 8192 tokens (requires the model to support an 8K context)
#    - Memory usage ∝ batch_size × sequence_length × hidden_size

# 6. Memory estimation (rough):
#    - Memory per sample ≈ sequence_length × hidden_size × 4 bytes (bf16)
#    - Training peak: train_batch_size × total_sequence_length × model_size × 3 (gradients + optimizer states)
#    - Rollout peak: rollout_batch_size × n_samples_per_prompt × total_sequence_length

# [Data-flow constraints]
# 7. Data processing flow:
#    - Data processed per epoch: min(max_samples, actual dataset size)
#    - Samples generated per rollout batch: rollout_batch_size × n_samples_per_prompt
#    - Training steps per rollout batch: (rollout_batch_size × n_samples_per_prompt) ÷ train_batch_size
#    - Current: (128 × 8) ÷ 128 = 8 training steps

# 8. Total training step calculation:
#    - Total number of batches = ceil(actual data volume ÷ rollout_batch_size)
#    - Total training steps = total batches × steps per batch × max_epochs
#    - Current estimate: ceil(5000 ÷ 128) × 8 × 1 = 40 × 8 = 320 steps

# [Learning rate and convergence constraints]
# 9. Learning rate settings:
#    - actor_learning_rate is typically 1–2 orders of magnitude smaller than the pretraining LR
#    - Current: 5e-7 is suitable for fine-tuning an SFT model
#    - Suggested range: 1e-7 to 1e-5

# [KL divergence control constraints]
# 10. KL coefficient settings:
#     - init_kl_coef controls how far we diverge from the reference model
#     - Too large: model will not explore; too small: model may drift away from baseline ability
#     - Current: 1e-3 is a common value

# [Hardware compatibility checklist]
# 11. Mandatory checks:
#     - Ensure every node has the same GPU count and model
#     - Ensure network bandwidth can support large-batch all-reduce communication
#     - Ensure each node has enough memory to load the model (typically > model size × 2.5)

# ============================================================================


# 
