#!/bin/bash

MODEL_PATH="/xfr_ceph_sh/liuchonghan/checkpoints_set/RLer_MtPO_allenai_025"

DATASET_PATH="/xfr_ceph_sh/liuchonghan/sft_dataset/SFT_EN_QA.json"

# python calculate_ppl.py \
#     --pretrain ${MODEL_PATH} \
#     --dataset ${DATASET_PATH} \
#     --input_key question \
#     --output_key response \
#     --max_len 2048 \
#     --micro_batch_size 8 \
#     --bf16 \
#     --output_file ppl_results.txt

# 如果使用LoRA模型
# python calculate_ppl.py \
#     --pretrain ${MODEL_PATH} \
#     --dataset ${DATASET_PATH} \
#     --input_key question \
#     --output_key response \
#     --max_len 2048 \
#     --micro_batch_size 8 \
#     --bf16 \
#     --lora_rank 64 \
#     --lora_alpha 16 \
#     --output_file ppl_results.txt

# 如果需要使用chat template
python calculate_ppl.py \
    --pretrain ${MODEL_PATH} \
    --dataset ${DATASET_PATH} \
    --input_key question \
    --output_key response \
    --max_len 2048 \
    --micro_batch_size 8 \
    --bf16 \
    --apply_chat_template \
    --output_file ppl_results.txt

# 如果使用多GPU（DeepSpeed）
# deepspeed --num_gpus 8 calculate_ppl.py \
#     --pretrain ${MODEL_PATH} \
#     --dataset ${DATASET_PATH} \
#     --input_key question \
#     --output_key response \
#     --max_len 2048 \
#     --micro_batch_size 8 \
#     --bf16 \
#     --zero_stage 0 \
#     --output_file ppl_results.txt

