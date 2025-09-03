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
   --eps_clip_low_high 0.2 0.28 \
   --pretrain /xfr_ceph_sh/liuchonghan/checkpoints_set/S0825 \
   --reward_pretrain /xfr_ceph_sh/liuchonghan/OpenRLHF_lao/examples/scripts/checkpoint/RewardModel_0902_translate \
   --save_path ./paper_checkpoint/paper_dapo_main_k2 \
   --ckpt_path ./paper_checkpoint/paper_dapo_main_k2_ckpt \
   --save_hf_ckpt \
   --rollout_batch_size 128 \
   --n_samples_per_prompt 16 \
   --train_batch_size 128 \
   --micro_train_batch_size 8 \
   --micro_rollout_batch_size 16 \
   --max_epochs 1 \
   --prompt_max_len 4096 \
   --max_samples 500000 \
   --generate_max_len 4096 \
   --zero_stage 3 \
   --bf16 \
   --actor_learning_rate 5e-7 \
   --prompt_data /xfr_ceph_sh/liuchonghan/prompt_dataset \
   --input_key context_messages \
   --normalize_reward \
   --gradient_checkpointing \
   --packing_samples \
   --vllm_sync_backend nccl \
   --enforce_eager \
   --vllm_enable_sleep \
   --deepspeed_enable_sleep \
   --use_wandb 9c69c18b00c7dac67189f39e261a257ebd476cda \
   --wandb_project 360_Repo_paper \
   --wandb_run_name paper_dapo_main_k2