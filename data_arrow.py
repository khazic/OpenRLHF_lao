# data_arrow.py (修改版本)
import json
import os
import shutil
import argparse
import glob
from datasets import Dataset, concatenate_datasets
from tqdm import tqdm

def get_directory_size(path):
    """计算目录大小"""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            try:
                total_size += os.path.getsize(filepath)
            except (OSError, IOError):
                pass
    return total_size

def load_data_from_path(input_path):
    """
    从路径加载数据，支持文件和目录
    """
    all_data = []
    
    if os.path.isfile(input_path):
        # 单个文件
        print(f"处理单个文件: {input_path}")
        data = load_single_file(input_path)
        all_data.extend(data)
    
    elif os.path.isdir(input_path):
        # 目录 - 查找所有支持的文件
        print(f"处理目录: {input_path}")
        
        # 支持的文件扩展名
        patterns = ['*.json', '*.jsonl']
        
        files_found = []
        for pattern in patterns:
            files = glob.glob(os.path.join(input_path, pattern))
            files_found.extend(files)
        
        # 递归查找子目录
        for pattern in patterns:
            files = glob.glob(os.path.join(input_path, '**', pattern), recursive=True)
            files_found.extend(files)
        
        # 去重并排序
        files_found = sorted(list(set(files_found)))
        
        if not files_found:
            print(f"错误：在目录 {input_path} 中未找到 .json 或 .jsonl 文件")
            return []
        
        print(f"找到 {len(files_found)} 个数据文件:")
        for file in files_found[:10]:  # 只显示前10个
            file_size = os.path.getsize(file) / 1024 / 1024  # MB
            print(f"  - {os.path.basename(file)} ({file_size:.2f} MB)")
        if len(files_found) > 10:
            print(f"  ... 还有 {len(files_found) - 10} 个文件")
        
        # 逐个处理文件
        for file_path in tqdm(files_found, desc="处理文件"):
            try:
                data = load_single_file(file_path)
                all_data.extend(data)
                print(f"  ✅ {os.path.basename(file_path)}: {len(data)} 条数据")
            except Exception as e:
                print(f"  ❌ {os.path.basename(file_path)}: 处理失败 - {e}")
                continue
    
    else:
        print(f"错误：路径不存在或无法识别: {input_path}")
        return []
    
    return all_data

def load_single_file(file_path):
    """加载单个文件"""
    data = []
    
    try:
        if file_path.endswith('.jsonl'):
            # JSONL格式
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f):
                    try:
                        item = json.loads(line.strip())
                        if item:  # 跳过空行
                            data.append(item)
                    except json.JSONDecodeError as e:
                        if line_num < 5:  # 只显示前5个错误
                            print(f"    跳过第{line_num+1}行，JSON解析错误: {e}")
                        continue
        
        elif file_path.endswith('.json'):
            # JSON格式
            with open(file_path, 'r', encoding='utf-8') as f:
                content = json.load(f)
                if isinstance(content, list):
                    data = content
                elif isinstance(content, dict):
                    data = [content]
                else:
                    print(f"    警告：未知的JSON格式: {type(content)}")
        
        else:
            print(f"    跳过不支持的文件格式: {file_path}")
    
    except Exception as e:
        print(f"    读取文件失败 {file_path}: {e}")
    
    return data

def preprocess_gemini_dataset(input_path, output_path):
    """
    预处理数据集
    
    Args:
        input_path: 输入数据集路径（文件或目录）
        output_path: 输出数据集路径
    """
    
    # 检查输入路径是否存在
    if not os.path.exists(input_path):
        print(f"错误：输入路径不存在: {input_path}")
        return False
    
    print(f"输入数据集: {input_path}")
    print(f"输出路径: {output_path}")
    
    # 如果输出目录已存在，删除
    if os.path.exists(output_path):
        print(f"删除已存在的输出目录: {output_path}")
        shutil.rmtree(output_path)
    
    print("开始预处理数据集...")
    
    try:
        # 加载数据
        data = load_data_from_path(input_path)
        
        if not data:
            print("错误：没有加载到任何数据！")
            return False
        
        print(f"原始数据量: {len(data)}")
        
        # 验证和清理数据
        valid_data = []
        invalid_count = 0
        
        # 检测数据格式
        if len(data) > 0:
            sample_item = data[0]
            print(f"数据样例字段: {list(sample_item.keys()) if isinstance(sample_item, dict) else 'unknown'}")
        
        for item in tqdm(data, desc="验证数据"):
            if isinstance(item, dict):
                question_fields = ['question', 'input', 'prompt', 'instruction', 'query']
                response_fields = ['response', 'output', 'answer', 'completion', 'target']
                
                question = None
                response = None
                
                for field in question_fields:
                    if field in item and item[field] and str(item[field]).strip():
                        question = str(item[field]).strip()
                        break
                
                for field in response_fields:
                    if field in item and item[field] and str(item[field]).strip():
                        response = str(item[field]).strip()
                        break
                
                if question and response:
                    # 标准化字段名
                    standardized_item = {
                        'question': question,
                        'response': response
                    }
                    # 保留其他有用字段
                    for key in ['datasource', 'id', 'metadata', 'source']:
                        if key in item:
                            standardized_item[key] = item[key]
                    
                    valid_data.append(standardized_item)
                else:
                    invalid_count += 1
                    if invalid_count <= 3:  # 只显示前3个无效样例
                        print(f"无效数据样例 {invalid_count}: {list(item.keys()) if isinstance(item, dict) else item}")
            else:
                invalid_count += 1
        
        print(f"有效数据量: {len(valid_data)}")
        print(f"无效数据量: {invalid_count}")
        
        if len(valid_data) == 0:
            print("错误：没有有效数据！")
            print("请检查数据格式。数据应该包含以下字段组合：")
            print("  问题字段: question, input, prompt, instruction, query")
            print("  回答字段: response, output, answer, completion, target")
            return False
        
        # 显示示例数据
        print("\n示例数据:")
        example = json.dumps(valid_data[0], ensure_ascii=False, indent=2)
        print(example[:300] + ("..." if len(example) > 300 else ""))
        
        # 转换为Dataset并保存为Arrow格式
        print("转换为Dataset格式...")
        dataset = Dataset.from_list(valid_data)
        
        print("保存到磁盘...")
        dataset.save_to_disk(output_path)
        
        # 计算文件大小
        dir_size = get_directory_size(output_path)
        dir_size_mb = dir_size / 1024 / 1024
        
        print(f"\n✅ 预处理完成！")
        print(f"📁 数据保存到: {output_path}")
        print(f"💾 文件大小: {dir_size_mb:.2f} MB")
        print(f"📊 数据条数: {len(dataset)}")
        
        # 验证保存的数据
        print("🔍 验证保存的数据...")
        try:
            from datasets import load_from_disk
            loaded_dataset = load_from_disk(output_path)
            print(f"✅ 验证成功！加载了 {len(loaded_dataset)} 条数据")
            print("📝 示例数据:", loaded_dataset[0])
        except Exception as e:
            print(f"❌ 验证失败: {e}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ 预处理过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(description='预处理数据集为Arrow格式')
    parser.add_argument('--input', '-i', required=True, 
                       help='输入数据集路径 (支持文件或目录，自动查找 .json 和 .jsonl 文件)')
    parser.add_argument('--output', '-o', required=True,
                       help='输出数据集目录路径')
    parser.add_argument('--force', '-f', action='store_true',
                       help='强制覆盖输出目录（不询问）')
    
    args = parser.parse_args()
    
    # 执行预处理
    success = preprocess_gemini_dataset(args.input, args.output)
    
    if success:
        print("\n🎉 数据预处理成功完成！")
        print(f"现在可以在训练脚本中使用: --dataset {args.output}")
    else:
        print("\n💥 数据预处理失败！")
        exit(1)

if __name__ == "__main__":
    main()