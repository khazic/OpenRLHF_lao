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
   --reward_num_nodes 0 \
   --reward_num_gpus_per_node 0 \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node 8 \
   --vllm_num_engines 2 \
   --vllm_tensor_parallel_size 4 \
   --vllm_gpu_memory_utilization 0.7 \
   --init_kl_coef 1e-3 \
   --gamma 1.0 \
   --colocate_all_models \
   --use_kl_loss \
   --kl_estimator k2 \
   --advantage_estimator rloo \
   --pretrain /xfr_ceph_sh/liuchonghan/OpenRLHF_lao/examples/scripts/checkpoint/SFTmodel_0823 \
   --remote_rm_url http://11.131.209.97:8000/reward \
   --save_path ./paper_checkpoint/paper_rloo_main_k2 \
   --ckpt_path ./paper_checkpoint/paper_rloo_main_k2_ckpt \
   --save_hf_ckpt \
   --rollout_batch_size 64 \
   --n_samples_per_prompt 8 \
   --train_batch_size 64 \
   --micro_train_batch_size 4 \
   --micro_rollout_batch_size 8 \
   --max_epochs 1 \
   --prompt_max_len 4096 \
   --max_samples 10000 \
   --generate_max_len 4096 \
   --zero_stage 3 \
   --bf16 \
   --apply_chat_template \
   --enable_new_token_monitoring \
   --tokenizer_config_path /xfr_ceph_sh/liuchonghan/OpenRLHF_lao/tokenizer_config_added.json \
   --auto_detect_original_vocab \
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
   --wandb_run_name paper_rloo_main_k2

# --colocate_all_models with --async_train only merge the deepspeed models, not the vllm engines

# You could also try
#   --use_kl_loss \
#   --kl_estimator k3 | k2 \

# also supports --advantage_estimator rloo | reinforce_baseline
