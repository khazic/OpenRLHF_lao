#!/bin/bash
# 基于PPL筛选数据集 - 删除低质量样本

MODEL_PATH="/xfr_ceph_sh/liuchonghan/checkpoints_set/RLer_MtPO_allenai_025"
DATASET_PATH="/xfr_ceph_sh/liuchonghan/sft_dataset"

# PPL阈值设置：高于此值的样本将被删除
# 建议值：
#   - 严格筛选: 5.0  (删除更多样本，保留最高质量)
#   - 中等筛选: 10.0 (平衡质量和数量)
#   - 宽松筛选: 20.0 (只删除明显的异常样本)
PPL_THRESHOLD=1.0

deepspeed --num_gpus 8 calculate_ppl_per_sample.py \
    --pretrain ${MODEL_PATH} \
    --dataset ${DATASET_PATH} \
    --input_key question \
    --output_key response \
    --max_len 4096 \
    --bf16 \
    --zero_stage 0 \
    --ppl_threshold ${PPL_THRESHOLD} \
    --output_file ppl_filter_results.json

# 运行完成后会生成3个文件：
# 1. ppl_filter_results_all.json     - 所有样本及其PPL（按PPL排序）
# 2. ppl_filter_results_filtered.json - 筛选后的高质量数据（PPL <= 阈值）
# 3. ppl_filter_results_removed.json  - 被删除的低质量数据（PPL > 阈值）

echo ""
echo "✅ 筛选完成！"
echo "📊 查看详细统计："
echo "   所有样本PPL: ppl_filter_results_all.json"
echo "   高质量数据: ppl_filter_results_filtered.json"
echo "   低质量数据: ppl_filter_results_removed.json"

