# PPL数据筛选工具 - 使用说明

## 📋 功能概述

这套工具可以帮助你：
1. 计算数据集中每个样本的PPL（困惑度）
2. 基于PPL阈值筛选高质量数据
3. 批量处理多个数据集
4. 生成详细的统计报告

---

## 🚀 快速开始

### 步骤1：查看所有数据集的样本数

```bash
python count_all_samples.py /xfr_ceph_sh/liuchonghan/sft_dataset
```

**输出示例：**
```
📊 数据集样本数统计
================================================================================
找到 5 个JSON文件
================================================================================

序号   文件名                                              样本数      备注
--------------------------------------------------------------------------------
1      SFT_EN_QA.json                                       850       📁 原始数据
2      SFT_Math.json                                       1200       📁 原始数据
3      SFT_Code.json                                        650       📁 原始数据
...

📈 汇总统计
原始数据集总样本数: 2,700
```

---

### 步骤2：批量处理所有数据集

```bash
bash filter_all_datasets.sh
```

**这个脚本会：**
- 自动找到所有JSON文件
- 对每个文件计算PPL
- 根据阈值筛选数据
- 生成详细报告

**输出文件（每个数据集会生成4个文件）：**
```
SFT_EN_QA_ppl1.0_all.json        - 所有样本及其PPL（按PPL排序）
SFT_EN_QA_ppl1.0_filtered.json   - ⭐ 筛选后的高质量数据（用于训练）
SFT_EN_QA_ppl1.0_removed.json    - 被删除的低质量数据
SFT_EN_QA_ppl1.0_summary.txt     - 📋 详细汇总报告
```

---

### 步骤3：查看筛选结果

#### 查看批处理汇总：
```bash
cat /xfr_ceph_sh/liuchonghan/sft_dataset/batch_ppl_filter_summary_*.txt
```

#### 查看单个数据集的详细报告：
```bash
cat /xfr_ceph_sh/liuchonghan/sft_dataset/SFT_EN_QA_ppl1.0_summary.txt
```

#### 再次统计样本数（查看筛选效果）：
```bash
python count_all_samples.py /xfr_ceph_sh/liuchonghan/sft_dataset
```

---

## ⚙️ 配置说明

### 修改PPL阈值

编辑 `filter_all_datasets.sh`，修改第7行：

```bash
PPL_THRESHOLD=1.0  # 改成你想要的阈值
```

**推荐阈值：**
- `1.0` - 极严格（只保留模型最熟悉的数据）
- `5.0` - 严格（保留高质量数据）
- `10.0` - 中等（平衡质量和数量）
- `20.0` - 宽松（只删除明显的异常样本）

### 修改字段名

如果你的数据集使用不同的字段名，编辑第10-11行：

```bash
INPUT_KEY="question"   # 改成你的输入字段名
OUTPUT_KEY="response"  # 改成你的输出字段名
```

---

## 📊 输出文件说明

### 1. `*_filtered.json` - 筛选后的高质量数据 ⭐
**这是你要用于训练的文件！**

格式：
```json
[
  {
    "question": "...",
    "response": "..."
  },
  ...
]
```

### 2. `*_all.json` - 所有样本的PPL详情
包含每个样本的PPL值，按PPL从低到高排序

格式：
```json
[
  {
    "index": 0,
    "ppl": 1.05,
    "loss": 0.0488,
    "num_tokens": 234,
    "question": "...",
    "response": "..."
  },
  ...
]
```

### 3. `*_removed.json` - 被删除的低质量数据
包含PPL超过阈值的样本

### 4. `*_summary.txt` - 详细统计报告
包含：
- 样本数量统计
- PPL分布信息
- 保留/删除比例
- 文件路径

---

## 🎯 常见使用场景

### 场景1：只处理单个数据集

编辑 `filter_dataset_by_ppl.sh`，设置数据集路径：

```bash
DATASET_PATH="/xfr_ceph_sh/liuchonghan/sft_dataset/SFT_EN_QA.json"
```

运行：
```bash
bash filter_dataset_by_ppl.sh
```

### 场景2：使用不同的阈值测试

```bash
# 测试阈值 5.0
sed -i 's/PPL_THRESHOLD=.*/PPL_THRESHOLD=5.0/' filter_all_datasets.sh
bash filter_all_datasets.sh

# 测试阈值 10.0
sed -i 's/PPL_THRESHOLD=.*/PPL_THRESHOLD=10.0/' filter_all_datasets.sh
bash filter_all_datasets.sh
```

然后对比两个阈值的结果，选择最合适的。

### 场景3：查看PPL分布决定阈值

先不设置阈值，只分析PPL分布：

修改 `calculate_ppl_per_sample.py` 的调用，不传 `--ppl_threshold`：

```bash
deepspeed --num_gpus 8 calculate_ppl_per_sample.py \
    --pretrain ${MODEL_PATH} \
    --dataset ${DATASET_PATH} \
    --input_key question \
    --output_key response \
    --max_len 4096 \
    --bf16 \
    --zero_stage 0 \
    --output_file analysis.json
```

查看 `analysis_all.json`，根据PPL分布决定合适的阈值。

---

## 📈 PPL值解读

| PPL范围 | 含义 | 建议 |
|---------|------|------|
| 1-5 | 模型非常熟悉 | 优质数据，保留 |
| 5-10 | 模型理解良好 | 高质量数据，保留 |
| 10-20 | 模型基本理解 | 可接受的数据 |
| 20-50 | 模型理解困难 | 考虑删除 |
| >50 | 模型几乎不理解 | 建议删除 |

---

## 🔧 故障排除

### 问题1：显存不足
**错误信息：** `CUDA out of memory`

**解决方法：**
编辑脚本，降低 batch size（虽然已经是1了，但可以减少GPU数量）：
```bash
deepspeed --num_gpus 4 calculate_ppl_per_sample.py ...  # 从8改成4
```

### 问题2：序列太长
**解决方法：**
降低 `--max_len`：
```bash
--max_len 2048  # 从4096改成2048
```

### 问题3：字段名不匹配
**错误信息：** `KeyError: 'question'`

**解决方法：**
检查你的JSON数据格式，修改 `INPUT_KEY` 和 `OUTPUT_KEY`。

---

## 📁 文件清单

| 文件名 | 用途 |
|--------|------|
| `calculate_ppl_per_sample.py` | 核心脚本：计算每个样本的PPL |
| `filter_all_datasets.sh` | 批量处理多个数据集 ⭐ |
| `filter_dataset_by_ppl.sh` | 处理单个数据集 |
| `count_all_samples.py` | 统计所有JSON文件的样本数 |
| `PPL筛选使用说明.md` | 本文档 |

---

## ✅ 完整工作流程示例

```bash
# 1. 查看现有数据集
python count_all_samples.py /xfr_ceph_sh/liuchonghan/sft_dataset

# 输出：
# 原始数据集: 5个文件，共2,700样本

# 2. 批量筛选
bash filter_all_datasets.sh

# 运行中...
# [1/5] 处理: SFT_EN_QA.json
#   原始样本数: 850
#   ✅ 保留: 723 (85.1%) | 删除: 127
# ...

# 3. 查看结果
python count_all_samples.py /xfr_ceph_sh/liuchonghan/sft_dataset

# 输出：
# 原始数据集: 2,700样本
# 筛选后数据集: 2,301样本 (保留率85.2%)

# 4. 查看详细报告
cat /xfr_ceph_sh/liuchonghan/sft_dataset/batch_ppl_filter_summary_*.txt

# 5. 使用筛选后的数据训练
# 所有 *_filtered.json 文件就是高质量数据！
```

---

## 💡 最佳实践

1. **先统计，后筛选**：运行 `count_all_samples.py` 了解数据规模
2. **选择合适的阈值**：根据数据集大小和质量要求选择
3. **保留中间文件**：`*_all.json` 可以帮助你分析PPL分布
4. **备份原始数据**：筛选不会修改原文件，但建议备份
5. **查看汇总报告**：每次处理后检查 `*_summary.txt`

---

## 🎉 完成后

筛选完成后，你的目录结构：

```
/xfr_ceph_sh/liuchonghan/sft_dataset/
├── SFT_EN_QA.json                          # 原始文件
├── SFT_EN_QA_ppl1.0_filtered.json         # ⭐ 用这个训练！
├── SFT_EN_QA_ppl1.0_all.json
├── SFT_EN_QA_ppl1.0_removed.json
├── SFT_EN_QA_ppl1.0_summary.txt
├── SFT_Math.json
├── SFT_Math_ppl1.0_filtered.json          # ⭐ 用这个训练！
├── ...
└── batch_ppl_filter_summary_xxx.txt       # 📋 总汇总
```

**在训练脚本中使用筛选后的数据：**
```bash
--dataset /xfr_ceph_sh/liuchonghan/sft_dataset/SFT_EN_QA_ppl1.0_filtered.json
```

