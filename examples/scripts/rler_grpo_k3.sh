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
   --vllm_gpu_memory_utilization 0.6 \
   --init_kl_coef 1e-3 \
   --gamma 1.0 \
   --colocate_all_models \
   --use_kl_loss \
   --kl_estimator k3 \
   --advantage_estimator group_norm \
   --pretrain /xfr_ceph_sh/liuchonghan/OpenRLHF_lao/examples/scripts/checkpoint/SFTmodel_0823 \
   --reward_pretrain /xfr_ceph_sh/liuchonghan/OpenRLHF_lao/examples/scripts/checkpoint/RewardModel_Qwen_0825_translate \
   --save_path ./checkpoint/RLer_grpo_sample_k3 \
   --ckpt_path ./checkpoint/RLer_grpo_sample_k3_ckpt \
   --save_hf_ckpt \
   --rollout_batch_size 64 \
   --n_samples_per_prompt 4 \
   --train_batch_size 64 \
   --micro_train_batch_size 8 \
   --micro_rollout_batch_size 8 \
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
   --wandb_run_name RLer_grpo_sample_k3

