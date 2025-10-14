#!/bin/bash
# 批量处理文件夹下所有JSON数据集

MODEL_PATH="/xfr_ceph_sh/liuchonghan/checkpoints_set/RLer_MtPO_allenai_025"
DATASET_DIR="/xfr_ceph_sh/liuchonghan/sft_dataset"
OUTPUT_DIR="/xfr_ceph_sh/liuchonghan/sft_dataset"

# PPL阈值设置
PPL_THRESHOLD=1.0

# 输入字段配置（根据你的数据集格式）
INPUT_KEY="question"
OUTPUT_KEY="response"

# 创建汇总日志文件
SUMMARY_LOG="${OUTPUT_DIR}/batch_ppl_filter_summary_$(date +%Y%m%d_%H%M%S).txt"

echo "========================================================================"
echo "🚀 批量PPL筛选开始"
echo "========================================================================"
echo "模型路径: ${MODEL_PATH}"
echo "数据集目录: ${DATASET_DIR}"
echo "PPL阈值: ${PPL_THRESHOLD}"
echo "========================================================================"
echo ""

# 先统计有多少JSON文件
JSON_FILES=($(find ${DATASET_DIR} -maxdepth 1 -name "*.json" -type f))
TOTAL_FILES=${#JSON_FILES[@]}

if [ ${TOTAL_FILES} -eq 0 ]; then
    echo "❌ 没有找到JSON文件！"
    exit 1
fi

echo "📂 找到 ${TOTAL_FILES} 个JSON文件"
echo ""

# 初始化汇总日志
cat > ${SUMMARY_LOG} << EOF
========================================================================
批量PPL筛选汇总报告
========================================================================
生成时间: $(date)
模型路径: ${MODEL_PATH}
数据集目录: ${DATASET_DIR}
PPL阈值: ${PPL_THRESHOLD}
总文件数: ${TOTAL_FILES}
========================================================================

EOF

# 计数器
PROCESSED=0
SUCCESS=0
FAILED=0

# 遍历所有JSON文件
for JSON_FILE in "${JSON_FILES[@]}"; do
    PROCESSED=$((PROCESSED + 1))
    FILENAME=$(basename ${JSON_FILE})
    DATASET_NAME="${FILENAME%.json}"
    
    echo "========================================================================"
    echo "[$PROCESSED/$TOTAL_FILES] 处理: ${FILENAME}"
    echo "========================================================================"
    
    # 跳过已经处理过的文件（带_ppl, _filtered, _removed, _all后缀的）
    if [[ ${FILENAME} =~ _ppl[0-9.]+.*\.json$ ]] || \
       [[ ${FILENAME} =~ _filtered\.json$ ]] || \
       [[ ${FILENAME} =~ _removed\.json$ ]] || \
       [[ ${FILENAME} =~ _all\.json$ ]]; then
        echo "⏭️  跳过（已处理的文件）"
        echo ""
        continue
    fi
    
    # 先统计原始样本数
    echo "📊 统计原始样本数..."
    ORIGINAL_COUNT=$(python3 -c "import json; data=json.load(open('${JSON_FILE}', 'r', encoding='utf-8')); print(len(data) if isinstance(data, list) else 'ERROR')")
    
    if [ "${ORIGINAL_COUNT}" == "ERROR" ]; then
        echo "❌ 无法读取文件，跳过"
        FAILED=$((FAILED + 1))
        echo "" >> ${SUMMARY_LOG}
        echo "❌ ${FILENAME}: 读取失败" >> ${SUMMARY_LOG}
        echo ""
        continue
    fi
    
    echo "   原始样本数: ${ORIGINAL_COUNT}"
    echo ""
    
    # 运行PPL筛选
    echo "🔄 开始PPL计算..."
    if deepspeed --num_gpus 8 calculate_ppl_per_sample.py \
        --pretrain ${MODEL_PATH} \
        --dataset ${JSON_FILE} \
        --input_key ${INPUT_KEY} \
        --output_key ${OUTPUT_KEY} \
        --max_len 4096 \
        --bf16 \
        --zero_stage 0 \
        --ppl_threshold ${PPL_THRESHOLD} \
        --output_file ${OUTPUT_DIR}/${DATASET_NAME}_ppl${PPL_THRESHOLD}.json; then
        
        SUCCESS=$((SUCCESS + 1))
        
        # 统计处理后的样本数
        FILTERED_FILE="${OUTPUT_DIR}/${DATASET_NAME}_ppl${PPL_THRESHOLD}_filtered.json"
        REMOVED_FILE="${OUTPUT_DIR}/${DATASET_NAME}_ppl${PPL_THRESHOLD}_removed.json"
        
        if [ -f "${FILTERED_FILE}" ]; then
            FILTERED_COUNT=$(python3 -c "import json; print(len(json.load(open('${FILTERED_FILE}', 'r', encoding='utf-8'))))")
            REMOVED_COUNT=$(python3 -c "import json; print(len(json.load(open('${REMOVED_FILE}', 'r', encoding='utf-8'))))")
            KEEP_RATE=$(python3 -c "print(f'{${FILTERED_COUNT}/${ORIGINAL_COUNT}*100:.1f}')")
            
            echo ""
            echo "✅ ${FILENAME} 处理完成"
            echo "   原始: ${ORIGINAL_COUNT} | 保留: ${FILTERED_COUNT} (${KEEP_RATE}%) | 删除: ${REMOVED_COUNT}"
            
            # 写入汇总日志
            cat >> ${SUMMARY_LOG} << EOF

------------------------------------------------------------------------
✅ ${FILENAME}
------------------------------------------------------------------------
原始样本数: ${ORIGINAL_COUNT}
保留样本数: ${FILTERED_COUNT} (${KEEP_RATE}%)
删除样本数: ${REMOVED_COUNT}
输出文件: ${DATASET_NAME}_ppl${PPL_THRESHOLD}_filtered.json
EOF
        fi
    else
        echo "❌ 处理失败"
        FAILED=$((FAILED + 1))
        echo "" >> ${SUMMARY_LOG}
        echo "❌ ${FILENAME}: 处理失败" >> ${SUMMARY_LOG}
    fi
    
    echo ""
done

# 写入最终汇总
cat >> ${SUMMARY_LOG} << EOF

========================================================================
批处理完成统计
========================================================================
总文件数: ${TOTAL_FILES}
成功处理: ${SUCCESS}
失败: ${FAILED}
========================================================================

所有筛选后的文件都在目录: ${OUTPUT_DIR}/
文件命名格式: *_ppl${PPL_THRESHOLD}_filtered.json

EOF

echo "========================================================================"
echo "🎉 批量处理完成！"
echo "========================================================================"
echo "总文件数: ${TOTAL_FILES}"
echo "✅ 成功: ${SUCCESS}"
echo "❌ 失败: ${FAILED}"
echo ""
echo "📋 详细汇总报告: ${SUMMARY_LOG}"
echo "========================================================================"
echo ""
echo "查看汇总报告:"
echo "  cat ${SUMMARY_LOG}"
echo ""
echo "查看所有筛选后的文件:"
echo "  ls -lh ${OUTPUT_DIR}/*_ppl${PPL_THRESHOLD}_filtered.json"
echo "========================================================================"

