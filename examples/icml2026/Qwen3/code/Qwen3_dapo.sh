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
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node 8 \
   --reward_num_nodes 1 \
   --reward_num_gpus_per_node 8 \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node 8 \
   --vllm_num_engines 2 \
   --vllm_tensor_parallel_size 4 \
   --vllm_gpu_memory_utilization 0.7 \
   --init_kl_coef 0.0 \
   --gamma 1.0 \
   --colocate_all_models \
   --use_kl_loss \
   --advantage_estimator group_norm \
   --eps_clip_low_high 0.2 0.28 \
   --pretrain /xfr_ceph_sh/liuchonghan/Qwen3-8B \
   --reward_pretrain /xfr_ceph_sh/liuchonghan/Skywork-Reward-V2-Qwen3-8B \
   --save_path ./paper_checkpoint/ICML2026_Qwen3_dapo \
   --ckpt_path ./paper_checkpoint/ICML2026_Qwen3_dapo_ckpt \
   --save_hf_ckpt \
   --rollout_batch_size 64 \
   --n_samples_per_prompt 8 \
   --train_batch_size 64 \
   --micro_train_batch_size 4 \
   --micro_rollout_batch_size 8 \
   --max_epochs 1 \
   --prompt_max_len 4096 \
   --max_samples 50000 \
   --generate_max_len 4096 \
   --zero_stage 3 \
   --bf16 \
   --apply_chat_template \
   --disable_thinking \
   --enable_new_token_monitoring \
   --actor_learning_rate 5e-7 \
   --prompt_data /xfr_ceph_sh/liuchonghan/prompt_dataset/code_cleaned.jsonl \
   --input_key context_messages \
   --normalize_reward \
   --gradient_checkpointing \
   --packing_samples \
   --vllm_sync_backend nccl \
   --enforce_eager \
   --vllm_enable_sleep \
   --deepspeed_enable_sleep \
   --use_wandb 9c69c18b00c7dac67189f39e261a257ebd476cda \
   --wandb_org "khazzz1c" \
   --wandb_project ICML2026 \
   --wandb_run_name ICML2026_Code_Qwen3_dapo