set -x

export WANDB_MODE=offline
export PATH=/mnt/data/liuchonghan/env/openrlhf/bin:$PATH

export PYTHONPATH=/mnt/data/liuchonghan/OpenRLHF_lao:$PYTHONPATH

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
   --advantage_estimator reinforce_baseline \
   --pretrain /mnt/data/liuchonghan/Qwen3-8B \
   --reward_pretrain /mnt/data/liuchonghan/Skywork-Reward-V2-Qwen3-8B \
   --save_path ./paper_checkpoint/ICML2026_Qwen3_rplus \
   --ckpt_path ./paper_checkpoint/ICML2026_Qwen3_rplus_ckpt \
   --save_hf_ckpt \
   --rollout_batch_size 64 \
   --n_samples_per_prompt 8 \
   --train_batch_size 64 \
   --micro_train_batch_size 4 \
   --micro_rollout_batch_size 8 \
   --max_epochs 5 \
   --prompt_max_len 4096 \
   --max_samples 50000 \
   --generate_max_len 4096 \
   --zero_stage 3 \
   --bf16 \
   --apply_chat_template \
   --disable_thinking \
   --enable_new_token_monitoring \
   --actor_learning_rate 5e-7 \
   --prompt_data /mnt/data/liuchonghan/prompt_dataset/chat_cleaned.jsonl \
   --input_key context_messages \
   --normalize_reward \
   --gradient_checkpointing \
   --packing_samples \
   --vllm_sync_backend nccl \
   --enforce_eager \
   --vllm_enable_sleep \
   --deepspeed_enable_sleep
