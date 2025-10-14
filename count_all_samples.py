#!/usr/bin/env python3
"""
统计文件夹下所有JSON文件的样本数
用法: python count_all_samples.py /path/to/dataset/folder
"""

import json
import os
import sys
from pathlib import Path


def count_samples_in_json(json_file):
    """统计单个JSON文件的样本数"""
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            return len(data), None
        elif isinstance(data, dict):
            return len(data), "字典格式（非列表）"
        else:
            return 0, "未知格式"
    except Exception as e:
        return 0, f"错误: {str(e)}"


def main(dataset_dir):
    """统计目录下所有JSON文件"""
    
    if not os.path.exists(dataset_dir):
        print(f"❌ 目录不存在: {dataset_dir}")
        sys.exit(1)
    
    # 查找所有JSON文件
    json_files = sorted(Path(dataset_dir).glob("*.json"))
    
    if not json_files:
        print(f"❌ 在 {dataset_dir} 中没有找到JSON文件")
        sys.exit(1)
    
    print("\n" + "="*100)
    print("📊 数据集样本数统计")
    print("="*100)
    print(f"目录: {dataset_dir}")
    print(f"找到 {len(json_files)} 个JSON文件")
    print("="*100)
    print()
    
    # 分类统计
    original_files = []  # 原始数据文件
    filtered_files = []  # 筛选后的文件
    other_files = []     # 其他文件（all, removed等）
    
    total_original_samples = 0
    total_filtered_samples = 0
    
    # 表头
    print(f"{'序号':<6} {'文件名':<50} {'样本数':<12} {'备注'}")
    print("-"*100)
    
    for idx, json_file in enumerate(json_files, 1):
        filename = json_file.name
        count, error = count_samples_in_json(json_file)
        
        # 分类
        if '_filtered.json' in filename:
            filtered_files.append((filename, count))
            total_filtered_samples += count
            category = "✅ 筛选后"
        elif any(x in filename for x in ['_ppl', '_all.json', '_removed.json', '_summary']):
            other_files.append((filename, count))
            category = "📊 中间文件"
        else:
            original_files.append((filename, count))
            total_original_samples += count
            category = "📁 原始数据"
        
        status = error if error else category
        print(f"{idx:<6} {filename:<50} {count:>10}   {status}")
    
    print("-"*100)
    
    # 汇总统计
    print("\n" + "="*100)
    print("📈 汇总统计")
    print("="*100)
    
    if original_files:
        print(f"\n📁 原始数据集 ({len(original_files)} 个文件):")
        print(f"   总样本数: {total_original_samples:,}")
        for filename, count in sorted(original_files, key=lambda x: x[1], reverse=True):
            print(f"      {count:>8,}  {filename}")
    
    if filtered_files:
        print(f"\n✅ 筛选后数据集 ({len(filtered_files)} 个文件):")
        print(f"   总样本数: {total_filtered_samples:,}")
        if total_original_samples > 0:
            retention_rate = total_filtered_samples / total_original_samples * 100
            removed_samples = total_original_samples - total_filtered_samples
            print(f"   保留率: {retention_rate:.1f}%")
            print(f"   删除样本: {removed_samples:,}")
        for filename, count in sorted(filtered_files, key=lambda x: x[1], reverse=True):
            print(f"      {count:>8,}  {filename}")
    
    if other_files:
        print(f"\n📊 其他文件 ({len(other_files)} 个):")
        for filename, count in other_files[:5]:  # 只显示前5个
            print(f"      {count:>8,}  {filename}")
        if len(other_files) > 5:
            print(f"      ... 还有 {len(other_files)-5} 个文件")
    
    print("\n" + "="*100)
    
    # 生成使用建议
    if original_files and not filtered_files:
        print("\n💡 提示:")
        print(f"   发现 {len(original_files)} 个原始数据集，尚未进行PPL筛选")
        print(f"   运行以下命令进行批量筛选:")
        print(f"   bash filter_all_datasets.sh")
    elif filtered_files:
        print("\n💡 筛选后的数据集可用于训练:")
        for filename, count in filtered_files:
            print(f"   {filename}")
    
    print("="*100 + "\n")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        dataset_dir = sys.argv[1]
    else:
        dataset_dir = "/xfr_ceph_sh/liuchonghan/sft_dataset"
    
    main(dataset_dir)

