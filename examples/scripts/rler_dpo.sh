set -x

# Check conda environment
if [ "$CONDA_DEFAULT_ENV" != "openrlhf" ]; then
    echo "Warning: conda environment is not openrlhf, current environment: $CONDA_DEFAULT_ENV"
    source /xfr_ceph_sh/liuchonghan/envs/etc/profile.d/conda.sh
    conda activate openrlhf
fi

# Set Python path
export PYTHONPATH=/xfr_ceph_sh/liuchonghan/OpenRLHF:$PYTHONPATH

# Validate paths
PRETRAIN_PATH="/xfr_ceph_sh/liuchonghan/checkpoints_set/S0825"
DATASET_PATH="/xfr_ceph_sh/liuchonghan/tranlate_datset"

if [ ! -d "$PRETRAIN_PATH" ]; then
    echo "Error: Pretrained model path does not exist: $PRETRAIN_PATH"
    exit 1
fi

if [ ! -d "$DATASET_PATH" ]; then
    echo "Error: Dataset path does not exist: $DATASET_PATH"
    exit 1
fi

echo "✅ All paths validated successfully"

if ! python3 -c "import ray" 2>/dev/null; then
    echo "Installing Ray..."
fi
read -r -d '' training_commands <<EOF
openrlhf.cli.train_dpo \
   --save_path ./checkpoint/rler_dpo \
   --save_steps -1 \
   --logging_steps 2 \
   --eval_steps 1000 \
   --train_batch_size 512 \
   --micro_train_batch_size 4 \
   --pretrain /xfr_ceph_sh/liuchonghan/checkpoints_set/S0825 \
   --bf16 \
   --max_epochs 1 \
   --max_len 8192 \
   --max_samples 10000000 \
   --zero_stage 3 \
   --learning_rate 3e-6 \
   --beta 0.1 \
   --dataset /xfr_ceph_sh/liuchonghan/tranlate_datset \
   --chosen_key chosen \
   --rejected_key rejected \
   --attn_implementation flash_attention_2 \
   --packing_samples \
   --gradient_checkpointing \
   --use_wandb 9c69c18b00c7dac67189f39e261a257ebd476cda \
   --wandb_project 360_Repo \
   --wandb_run_name rler_dpo_0902
EOF
    # --use_wandb [WANDB_TOKENS] or True (use wandb login command)
    # --ipo [for IPO]
    # --label_smoothing 0.1 [for cDPO]
    # --ref_offload
    # --packing_samples
    # --nll_loss_coef (Regularization with NLL loss)
#    --apply_chat_template \


if [[ ${1} != "slurm" ]]; then
    NCCL_DEBUG=ERROR deepspeed --master_addr localhost --master_port 29500 \
              --include localhost:0,1,2,3,4,5,6,7 \
              --module $training_commands
fi
