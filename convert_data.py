#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据格式转换脚本
从原始数据提取 input 和 gemini_nothink 字段，转换为 question-response 格式
"""

import json
import re
from pathlib import Path

def extract_nothink(output_text):
    """从 output 中提取非 <think> 部分"""
    # 移除 <think>...</think> 标签及其内容
    cleaned = re.sub(r'<think>.*?</think>', '', output_text, flags=re.DOTALL)
    return cleaned.strip()

def process_data(input_file, output_file):
    """处理数据文件"""
    
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        
        processed_count = 0
        error_count = 0
        
        for line_num, line in enumerate(f_in, 1):
            try:
                data = json.loads(line.strip())
                
                # 提取 input 字段
                input_text = data.get('input', '')
                
                # 提取或生成 gemini_nothink
                if 'gemini_nothink' in data:
                    response_text = data['gemini_nothink']
                elif 'output' in data:
                    # 从 output 中去掉 <think> 部分
                    response_text = extract_nothink(data['output'])
                else:
                    print(f"⚠️  第 {line_num} 行：缺少 gemini_nothink 或 output 字段，跳过")
                    error_count += 1
                    continue
                
                if not input_text or not response_text:
                    print(f"⚠️  第 {line_num} 行：input 或 response 为空，跳过")
                    error_count += 1
                    continue
                
                # 生成 question-response 格式
                output_data = {
                    "question": input_text,
                    "response": response_text
                }
                
                # 写入文件
                f_out.write(json.dumps(output_data, ensure_ascii=False) + '\n')
                
                processed_count += 1
                
                if processed_count % 1000 == 0:
                    print(f"✅ 已处理 {processed_count} 条数据...")
                    
            except json.JSONDecodeError as e:
                print(f"❌ 第 {line_num} 行 JSON 解析错误: {e}")
                error_count += 1
            except Exception as e:
                print(f"❌ 第 {line_num} 行处理错误: {e}")
                error_count += 1
        
        print(f"\n{'='*50}")
        print(f"✅ 处理完成！")
        print(f"   成功: {processed_count} 条")
        print(f"   失败: {error_count} 条")
        print(f"   输出文件: {output_file}")

if __name__ == "__main__":
    import sys
    
    # 支持命令行参数
    if len(sys.argv) >= 3:
        INPUT_FILE = sys.argv[1]
        OUTPUT_FILE = sys.argv[2]
    else:
        # 默认配置
        INPUT_FILE = "input_data.jsonl"
        OUTPUT_FILE = "output_formatted.jsonl"
    
    # 检查输入文件是否存在
    if not Path(INPUT_FILE).exists():
        print(f"❌ 错误：输入文件不存在: {INPUT_FILE}")
        print("\n使用方法:")
        print(f"  python3 {sys.argv[0]} <输入文件> <输出文件>")
        print(f"\n示例:")
        print(f"  python3 {sys.argv[0]} input_data.jsonl output_formatted.jsonl")
        exit(1)
    
    # 处理数据
    process_data(INPUT_FILE, OUTPUT_FILE)

