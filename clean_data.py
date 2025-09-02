#!/usr/bin/env python3
import json
from pathlib import Path

def clean_data(input_file, output_file):
    """清理数据文件，移除无效样本"""
    print(f"清理文件: {input_file}")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    original_count = len(data)
    cleaned_data = []
    
    for sample in data:
        # 检查必需字段
        required_fields = ['prompt', 'chosen', 'rejected']
        if not all(field in sample for field in required_fields):
            continue
        
        # 检查字段是否为空
        if any(not sample[field] or sample[field].strip() == '' for field in required_fields):
            continue
        
        # 检查chosen和rejected是否相同
        if sample['chosen'] == sample['rejected']:
            continue
        
        cleaned_data.append(sample)
    
    # 保存清理后的数据
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(cleaned_data, f, ensure_ascii=False, indent=2)
    
    print(f"  原始样本: {original_count}")
    print(f"  清理后样本: {len(cleaned_data)}")
    print(f"  保留率: {len(cleaned_data)/original_count*100:.2f}%")
    print(f"  保存到: {output_file}")

def main():
    data_dir = Path("/Users/liuyibo/Downloads/translate_rm/processed")
    output_dir = Path("/Users/liuyibo/Downloads/translate_rm/processed_clean")
    output_dir.mkdir(exist_ok=True)
    
    # 清理有问题的文件
    files_to_clean = [
        "processed_ckpt_with_rejected_20250826_032814.json",
        "processed_translate_with_rejected_20250827_003851.json"
    ]
    
    for file_name in files_to_clean:
        input_file = data_dir / file_name
        output_file = output_dir / f"cleaned_{file_name}"
        
        if input_file.exists():
            clean_data(input_file, output_file)
        else:
            print(f"文件不存在: {input_file}")

if __name__ == "__main__":
    main()
