#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将数据集转换为 JSON 数组格式
输入: JSONL 格式，每行一个 JSON 对象
输出: JSON 数组格式，包含 question 和 response 字段
"""

import json
import re
from pathlib import Path

def extract_nothink(output_text):
    """从 output 中提取非 <think> 部分"""
    cleaned = re.sub(r'<think>.*?</think>', '', output_text, flags=re.DOTALL)
    return cleaned.strip()

def convert_to_array(input_file, output_file):
    """将 JSONL 转换为 JSON 数组"""
    
    data_array = []
    processed_count = 0
    error_count = 0
    
    print(f"📖 正在读取: {input_file}")
    
    with open(input_file, 'r', encoding='utf-8') as f_in:
        for line_num, line in enumerate(f_in, 1):
            try:
                data = json.loads(line.strip())
                
                # 提取 question 字段（从 input）
                question = data.get('input', '')
                
                # 提取 response 字段（优先使用 gemini_nothink，否则从 output 提取）
                if 'gemini_nothink' in data:
                    response = data['gemini_nothink']
                elif 'output' in data:
                    response = extract_nothink(data['output'])
                elif 'response' in data:
                    response = data['response']
                else:
                    print(f"⚠️  第 {line_num} 行：找不到 response 字段，跳过")
                    error_count += 1
                    continue
                
                if not question or not response:
                    print(f"⚠️  第 {line_num} 行：question 或 response 为空，跳过")
                    error_count += 1
                    continue
                
                # 添加到数组
                data_array.append({
                    "question": question,
                    "response": response
                })
                
                processed_count += 1
                
                if processed_count % 1000 == 0:
                    print(f"✅ 已处理 {processed_count} 条...")
                    
            except json.JSONDecodeError as e:
                print(f"❌ 第 {line_num} 行 JSON 解析错误: {e}")
                error_count += 1
            except Exception as e:
                print(f"❌ 第 {line_num} 行处理错误: {e}")
                error_count += 1
    
    # 写入 JSON 数组
    print(f"\n💾 正在写入: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f_out:
        json.dump(data_array, f_out, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*50}")
    print(f"✅ 转换完成！")
    print(f"   成功: {processed_count} 条")
    print(f"   失败: {error_count} 条")
    print(f"   输出文件: {output_file}")
    print(f"   文件大小: {Path(output_file).stat().st_size / (1024*1024):.2f} MB")

if __name__ == "__main__":
    import sys
    
    # 支持命令行参数
    if len(sys.argv) >= 3:
        INPUT_FILE = sys.argv[1]
        OUTPUT_FILE = sys.argv[2]
    else:
        # 默认配置
        INPUT_FILE = "hard_dataset.json"
        OUTPUT_FILE = "output_array.json"
    
    # 检查输入文件
    if not Path(INPUT_FILE).exists():
        print(f"❌ 错误：输入文件不存在: {INPUT_FILE}")
        print("\n使用方法:")
        print(f"  python3 {sys.argv[0]} <输入文件> <输出文件>")
        print(f"\n示例:")
        print(f"  python3 {sys.argv[0]} hard_dataset.json output_array.json")
        exit(1)
    
    # 执行转换
    convert_to_array(INPUT_FILE, OUTPUT_FILE)

