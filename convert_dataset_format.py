#!/usr/bin/env python3
"""
数据集格式转换脚本
将 context_messages 列表格式转换为字符串格式，使其能被 OpenRLHF 直接读取
"""

import json
import os
from pathlib import Path

def convert_jsonl_file(input_file, output_file):
    """
    转换单个 JSONL 文件
    
    Args:
        input_file: 输入文件路径
        output_file: 输出文件路径
    """
    converted_count = 0
    error_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        for line_num, line in enumerate(infile, 1):
            try:
                # 解析 JSON 行
                data = json.loads(line.strip())
                
                # 提取 context_messages 中的 content
                if 'context_messages' in data and isinstance(data['context_messages'], list):
                    if len(data['context_messages']) > 0 and 'content' in data['context_messages'][0]:
                        # 提取第一个消息的 content 作为 prompt
                        prompt = data['context_messages'][0]['content']
                        
                        # 创建新的数据格式
                        new_data = {
                            'context_messages': prompt,  # 转换为字符串
                            'metadata': data.get('metadata', {}),
                            'datasource': data.get('datasource', 'default')
                        }
                        
                        # 写入转换后的数据
                        outfile.write(json.dumps(new_data, ensure_ascii=False) + '\n')
                        converted_count += 1
                    else:
                        print(f"警告: 第 {line_num} 行的 context_messages 格式异常，跳过")
                        error_count += 1
                else:
                    print(f"警告: 第 {line_num} 行缺少 context_messages 字段，跳过")
                    error_count += 1
                    
            except json.JSONDecodeError as e:
                print(f"错误: 第 {line_num} 行 JSON 解析失败: {e}")
                error_count += 1
            except Exception as e:
                print(f"错误: 第 {line_num} 行处理失败: {e}")
                error_count += 1
    
    return converted_count, error_count

def main():
    """主函数"""
    input_dir = Path("/Users/liuyibo/Downloads/prompt_dataset")
    output_dir = Path("/Users/liuyibo/Downloads/prompt_dataset_converted")
    
    # 创建输出目录
    output_dir.mkdir(exist_ok=True)
    
    # 获取所有 .jsonl 文件
    jsonl_files = list(input_dir.glob("*.jsonl"))
    
    if not jsonl_files:
        print("未找到任何 .jsonl 文件")
        return
    
    print(f"找到 {len(jsonl_files)} 个 .jsonl 文件")
    
    total_converted = 0
    total_errors = 0
    
    for jsonl_file in jsonl_files:
        print(f"\n处理文件: {jsonl_file.name}")
        
        output_file = output_dir / jsonl_file.name
        converted, errors = convert_jsonl_file(jsonl_file, output_file)
        
        total_converted += converted
        total_errors += errors
        
        print(f"  转换成功: {converted} 条")
        print(f"  错误/跳过: {errors} 条")
        print(f"  输出文件: {output_file}")
    
    print(f"\n=== 转换完成 ===")
    print(f"总计转换: {total_converted} 条数据")
    print(f"总计错误: {total_errors} 条数据")
    print(f"输出目录: {output_dir}")
    
    # 显示转换前后的对比示例
    if jsonl_files:
        print(f"\n=== 转换示例 ===")
        sample_file = jsonl_files[0]
        with open(sample_file, 'r', encoding='utf-8') as f:
            original = json.loads(f.readline().strip())
        
        output_sample = output_dir / sample_file.name
        with open(output_sample, 'r', encoding='utf-8') as f:
            converted = json.loads(f.readline().strip())
        
        print("转换前:")
        print(json.dumps(original, ensure_ascii=False, indent=2))
        print("\n转换后:")
        print(json.dumps(converted, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
