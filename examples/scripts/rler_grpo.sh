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
# 必须遵守的数量关系约束 (Critical Parameter Constraints)
# ============================================================================

# 【GPU资源约束】
# 1. 总GPU数量计算:
#    - Actor GPUs: ref_num_nodes × ref_num_gpus_per_node = 2 × 8 = 16 GPUs
#    - Reference GPUs: actor_num_nodes × actor_num_gpus_per_node = 2 × 8 = 16 GPUs  
#    - Reward GPUs: reward_num_nodes × reward_num_gpus_per_node = 2 × 8 = 16 GPUs
#    - VLLM GPUs: vllm_num_engines × vllm_tensor_parallel_size = 4 × 4 = 16 GPUs
#    总需求: 64 GPUs (如果colocate_all_models=True，实际只需16 GPUs)

# 2. VLLM引擎约束:
#    - vllm_tensor_parallel_size 必须能被GPU数量整除
#    - vllm_num_engines × vllm_tensor_parallel_size ≤ 可用GPU总数
#    - 推荐: vllm_tensor_parallel_size = 2的幂次 (1,2,4,8)

# 【批次大小约束】
# 3. 训练批次关系:
#    - train_batch_size 必须能被 micro_train_batch_size 整除
#    - 当前: 128 ÷ 8 = 16 (每个训练步骤需要16个GPU微批次)
#    - train_batch_size 必须 ≤ (actor_num_nodes × actor_num_gpus_per_node × micro_train_batch_size)
#    - 当前检查: 128 ≤ (2 × 8 × 8) = 128 ✓

# 4. Rollout批次关系:
#    - rollout_batch_size 必须能被 micro_rollout_batch_size 整除  
#    - 当前: 128 ÷ 16 = 8 (每个rollout步骤需要8个GPU微批次)
#    - rollout_batch_size 必须 ≤ (vllm_num_engines × micro_rollout_batch_size × 并发倍数)

# 【内存和序列长度约束】
# 5. 序列长度限制:
#    - prompt_max_len + generate_max_len ≤ 模型最大序列长度
#    - 当前: 4096 + 4096 = 8192 tokens (需要模型支持8K上下文)
#    - 显存需求 ∝ batch_size × sequence_length × hidden_size

# 6. 显存估算 (粗略):
#    - 每个样本显存需求 ≈ sequence_length × hidden_size × 4 bytes (bf16)
#    - 训练时峰值: train_batch_size × total_sequence_length × model_size × 3倍(梯度+优化器)
#    - Rollout时峰值: rollout_batch_size × n_samples_per_prompt × total_sequence_length

# 【数据流约束】
# 7. 数据处理流程:
#    - 每个epoch处理数据量: min(max_samples, 实际数据集大小)
#    - 每批rollout生成: rollout_batch_size × n_samples_per_prompt 个样本
#    - 每批rollout需要训练步数: (rollout_batch_size × n_samples_per_prompt) ÷ train_batch_size
#    - 当前: (128 × 8) ÷ 128 = 8 个训练步骤

# 8. 总训练步骤计算:
#    - 总批次数 = ceil(实际数据量 ÷ rollout_batch_size)
#    - 总训练步骤 = 总批次数 × 每批训练步数 × max_epochs
#    - 当前估算: ceil(5000 ÷ 128) × 8 × 1 = 40 × 8 = 320 步

# 【学习率和收敛约束】
# 9. 学习率设置:
#    - actor_learning_rate 通常比预训练小1-2个数量级
#    - 当前: 5e-7 适合从已训练的SFT模型微调
#    - 建议范围: 1e-7 到 1e-5

# 【KL散度控制约束】
# 10. KL系数设置:
#     - init_kl_coef 控制与reference模型的偏离程度
#     - 过大: 模型不敢探索，过小: 可能偏离原始能力
#     - 当前: 1e-3 是常用值

# 【硬件兼容性检查】
# 11. 必要检查项:
#     - 确保所有节点都有相同的GPU数量和型号
#     - 确保网络带宽足够支持大批次的all-reduce通信
#     - 确保每个节点有足够内存加载模型 (通常需要 > 模型大小 × 2.5)

# ============================================================================


# 