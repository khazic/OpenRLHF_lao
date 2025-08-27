import os
import json
import tempfile
from datasets import interleave_datasets, load_dataset, load_from_disk


def exist_and_not_none(d, key):
    return key in d and not d[key] is None


def load_json_with_error_handling(file_path, strategy=None):
    """
    加载JSON文件，跳过损坏的样本
    """
    valid_data = []
    error_count = 0
    total_lines = 0
    
    strategy.print(f"使用错误处理加载JSON文件: {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            
            # 尝试作为完整JSON文件加载
            if content.startswith('[') and content.endswith(']'):
                try:
                    data_list = json.loads(content)
                    for i, item in enumerate(data_list):
                        total_lines += 1
                        if isinstance(item, dict):
                            valid_data.append(item)
                        else:
                            error_count += 1
                            strategy.print(f"跳过第{i+1}个项目: 不是有效的字典对象")
                except json.JSONDecodeError as e:
                    strategy.print(f"完整JSON解析失败: {str(e)}，尝试逐行解析")
                    # 如果完整解析失败，回退到逐行解析
                    valid_data, error_count, total_lines = _parse_json_lines(content, strategy)
            else:
                # 作为JSONL格式逐行解析
                valid_data, error_count, total_lines = _parse_json_lines(content, strategy)
                
    except Exception as e:
        strategy.print(f"读取文件失败 {file_path}: {str(e)}")
        raise e
    
    strategy.print(f"成功加载 {len(valid_data)} 个有效样本，跳过 {error_count} 个损坏样本，总共 {total_lines} 行")
    
    if not valid_data:
        raise ValueError(f"文件中没有找到有效数据: {file_path}")
    
    # 创建临时文件保存清理后的数据
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False, encoding='utf-8')
    try:
        for item in valid_data:
            temp_file.write(json.dumps(item, ensure_ascii=False) + '\n')
        temp_file.close()
        
        # 使用datasets加载清理后的数据
        dataset = load_dataset('json', data_files=temp_file.name)['train']
        return dataset
    finally:
        # 清理临时文件
        try:
            os.unlink(temp_file.name)
        except:
            pass


def _parse_json_lines(content, strategy):
    """逐行解析JSON内容"""
    valid_data = []
    error_count = 0
    total_lines = 0
    
    lines = content.split('\n')
    for line_num, line in enumerate(lines, 1):
        line = line.strip()
        if not line:
            continue
            
        total_lines += 1
        try:
            data = json.loads(line)
            if isinstance(data, dict):
                # 基本验证数据结构
                if _validate_reward_data_structure(data):
                    valid_data.append(data)
                else:
                    error_count += 1
                    strategy.print(f"跳过第{line_num}行: 数据结构不符合奖励模型要求")
            else:
                error_count += 1
                strategy.print(f"跳过第{line_num}行: 不是有效的字典对象")
        except json.JSONDecodeError as e:
            error_count += 1
            strategy.print(f"跳过第{line_num}行: JSON解析错误 - {str(e)}")
        except Exception as e:
            error_count += 1
            strategy.print(f"跳过第{line_num}行: {str(e)}")
    
    return valid_data, error_count, total_lines


def _validate_reward_data_structure(data):
    """验证奖励数据的基本结构"""
    # 检查是否包含必要的字段
    required_fields = ['chosen', 'rejected']
    optional_fields = ['prompt', 'question', 'instruction', 'input']
    
    # 至少要有chosen和rejected字段
    if not all(field in data for field in required_fields):
        return False
    
    # 检查chosen和rejected是否为字符串或列表
    for field in required_fields:
        if not isinstance(data[field], (str, list)):
            return False
    
    # 检查是否至少有一个prompt相关字段
    has_prompt_field = any(field in data for field in optional_fields)
    
    return True  # 允许没有prompt字段的数据


def load_directory_with_error_handling(directory_path, strategy):
    """
    加载目录中的数据集，对JSON文件使用错误处理
    """
    strategy.print(f"扫描目录中的数据文件: {directory_path}")
    
    # 查找目录中的数据文件
    json_files = []
    other_files = []
    
    for file_name in os.listdir(directory_path):
        file_path = os.path.join(directory_path, file_name)
        if os.path.isfile(file_path):
            ext = os.path.splitext(file_name)[-1].lower()
            if ext in ['.json', '.jsonl']:
                json_files.append(file_path)
            elif ext in ['.csv', '.parquet', '.arrow']:
                other_files.append(file_path)
    
    datasets_to_concat = []
    
    # 处理JSON文件
    for json_file in json_files:
        try:
            strategy.print(f"处理JSON文件: {json_file}")
            data = load_json_with_error_handling(json_file, strategy)
            datasets_to_concat.append(data)
        except Exception as e:
            strategy.print(f"跳过损坏的JSON文件 {json_file}: {str(e)}")
            continue
    
    # 处理其他格式文件
    for other_file in other_files:
        try:
            ext = os.path.splitext(other_file)[-1].lower().strip('.')
            strategy.print(f"处理{ext.upper()}文件: {other_file}")
            data = load_dataset(ext, data_files=other_file)['train']
            datasets_to_concat.append(data)
        except Exception as e:
            strategy.print(f"跳过损坏的文件 {other_file}: {str(e)}")
            continue
    
    if not datasets_to_concat:
        raise ValueError(f"目录 {directory_path} 中没有找到可用的数据文件")
    
    # 合并所有数据集
    if len(datasets_to_concat) == 1:
        return datasets_to_concat[0]
    else:
        from datasets import concatenate_datasets
        return concatenate_datasets(datasets_to_concat)


def blending_datasets(
    datasets,
    probabilities=None,
    strategy=None,
    seed=42,
    max_count=1e8,
    stopping_strategy="all_exhausted",
    dataset_split="train",
):
    """Blend multiple datasets with optional probability sampling.

    Args:
        datasets (str): Comma-separated list of dataset paths
        probabilities (str, optional): Comma-separated list of probabilities for sampling.
            If None, datasets will be concatenated without probability sampling.
        strategy: Training strategy object
        seed (int): Random seed
        max_count (int): Maximum number of samples per dataset
    """
    datasets = datasets.split(",")
    if probabilities is not None:
        probabilities = list(map(float, probabilities.split(",")))
        assert len(probabilities) == len(datasets)

    data_list = []
    for i, dataset in enumerate(datasets):
        dataset = dataset.strip()
        strategy.print(f"dataset: {dataset}")

        data_dir = dataset.split("@")[1].strip() if "@" in dataset else None
        dataset = dataset.split("@")[0].strip()
        dataset_basename = os.path.basename(dataset)

        ext = os.path.splitext(dataset)[-1]
        # local python script
        if ext == ".py" or (
            os.path.isdir(dataset) and os.path.exists(os.path.join(dataset, f"{dataset_basename}.py"))
        ):
            data = load_dataset(dataset, trust_remote_code=True)
            strategy.print(f"loaded {dataset} with python script")
        # local text file
        elif ext in [".json", ".jsonl", ".csv", ".parquet", ".arrow"]:
            ext = ext.lower().strip(".")
            if ext == "jsonl":
                ext = "json"
            
            # 对JSON文件使用错误处理加载
            if ext == "json":
                try:
                    data = load_json_with_error_handling(dataset, strategy)
                    strategy.print(f"使用错误处理成功加载 {dataset}")
                except Exception as e:
                    strategy.print(f"错误处理加载失败 {dataset}: {str(e)}，尝试标准加载方法")
                    try:
                        data = load_dataset(ext, data_files=dataset)
                        strategy.print(f"使用标准方法加载 {dataset}")
                    except Exception as e2:
                        strategy.print(f"标准加载方法也失败 {dataset}: {str(e2)}")
                        # 跳过这个数据集
                        continue
            else:
                data = load_dataset(ext, data_files=dataset)
                strategy.print(f"loaded {dataset} with data_files={dataset}")
        # local dataset saved with `datasets.Dataset.save_to_disk`
        elif os.path.isdir(dataset):
            try:
                data = load_from_disk(dataset)
                strategy.print(f"loaded {dataset} from disk")
            except Exception as e:
                strategy.print(f"failed to load {dataset} from disk: {e}")
                # 尝试使用带错误处理的目录加载
                try:
                    data = load_directory_with_error_handling(dataset, strategy)
                    strategy.print(f"使用错误处理成功加载目录 {dataset}")
                except Exception as e2:
                    strategy.print(f"错误处理加载目录失败 {dataset}: {str(e2)}，尝试标准方法")
                    try:
                        data = load_dataset(dataset, data_dir=data_dir)
                        strategy.print(f"loaded {dataset} from files")
                    except Exception as e3:
                        strategy.print(f"所有加载方法都失败，跳过数据集 {dataset}: {str(e3)}")
                        continue
        # remote/local folder or common file
        elif strategy.args.use_ms:
            from modelscope.msdatasets import MsDataset

            namespace, dataset = dataset.split("/")
            data = MsDataset.load(dataset, namespace=namespace)
        else:
            data = load_dataset(dataset, data_dir=data_dir)
            strategy.print(f"loaded {dataset} from files")

        # Select dataset
        if dataset_split and dataset_split in data:
            data = data[dataset_split]
        data = data.select(range(min(max_count, len(data))))
        data_list.append(data)

    # merge datasets
    if strategy.is_rank_0():
        print(data_list)

    # If probabilities is None, concatenate datasets directly
    if probabilities is None:
        from datasets import concatenate_datasets

        dataset = concatenate_datasets(data_list)
    else:
        dataset = interleave_datasets(
            data_list,
            probabilities=probabilities,
            seed=seed,
            stopping_strategy=stopping_strategy,
        )

    return dataset
