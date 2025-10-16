import glob
import hashlib
import json
import os
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path

from datasets import concatenate_datasets, interleave_datasets, load_dataset, load_from_disk

try:
    from filelock import FileLock
except ImportError:  # pragma: no cover - filelock is an optional dependency in some envs
    FileLock = None


def exist_and_not_none(d, key):
    return key in d and not d[key] is None


def _strategy_print(strategy, message: str):
    if strategy is None:
        return
    try:
        strategy.print(message)
    except Exception:
        pass


def _should_cache_dataset(strategy) -> bool:
    if strategy is None or not hasattr(strategy, "args"):
        return False
    return getattr(strategy.args, "cache_dataset_to_disk", False)


def _dataset_cache_base_dir(strategy) -> str:
    if strategy is not None and hasattr(strategy, "args"):
        cache_dir = getattr(strategy.args, "dataset_cache_dir", None)
        if cache_dir:
            return os.path.abspath(cache_dir)
    return os.path.join(Path.home(), ".cache", "openrlhf", "datasets")


@contextmanager
def _dataset_cache_lock(path: str):
    """Provide a shared lock around cache operations."""
    if FileLock is None:
        yield
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lock = FileLock(path)
    with lock:
        yield


def _build_cache_key(dataset_path: str, dataset_split: str = None, data_dir: str = None) -> str:
    parts = [dataset_path]
    if data_dir:
        parts.append(f"data_dir={data_dir}")
    if dataset_split:
        parts.append(f"split={dataset_split}")
    # Include modification time for local assets to help invalidation
    candidate = dataset_path if dataset_path else ""
    try:
        if candidate and os.path.exists(candidate):
            parts.append(f"mtime={os.path.getmtime(candidate)}")
    except OSError:
        pass
    return "||".join(parts)


def _maybe_cache_dataset(dataset_obj, dataset_path: str, dataset_split: str, data_dir: str, strategy):
    """Persist dataset to Arrow files on disk for faster subsequent loads."""
    if not _should_cache_dataset(strategy):
        return dataset_obj

    base_dir = _dataset_cache_base_dir(strategy)
    os.makedirs(base_dir, exist_ok=True)

    cache_key = _build_cache_key(dataset_path, dataset_split, data_dir)
    cache_hash = hashlib.md5(cache_key.encode("utf-8")).hexdigest()
    cache_path = os.path.join(base_dir, cache_hash)
    lock_path = f"{cache_path}.lock"

    with _dataset_cache_lock(lock_path):
        if os.path.isdir(cache_path):
            try:
                cached_dataset = load_from_disk(cache_path)
                _strategy_print(strategy, f"Loaded dataset cache from {cache_path}")
                return cached_dataset
            except Exception as exc:
                _strategy_print(strategy, f"Failed to load dataset cache at {cache_path}: {exc}. Rebuilding cache.")
                shutil.rmtree(cache_path, ignore_errors=True)
                if os.path.exists(cache_path):
                    try:
                        os.remove(cache_path)
                    except OSError:
                        pass

        if os.path.exists(cache_path) and not os.path.isdir(cache_path):
            try:
                os.remove(cache_path)
            except OSError:
                pass

        dataset_obj.save_to_disk(cache_path)
        _strategy_print(strategy, f"Cached dataset to {cache_path}")
        return dataset_obj


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
        dataset_entry = dataset.strip()
        _strategy_print(strategy, f"dataset: {dataset_entry}")

        data_dir = dataset_entry.split("@")[1].strip() if "@" in dataset_entry else None
        dataset_path = dataset_entry.split("@")[0].strip()
        dataset_basename = os.path.basename(dataset_path)

        ext = os.path.splitext(dataset)[-1]
        # local python script
        if ext == ".py" or (
            os.path.isdir(dataset_path) and os.path.exists(os.path.join(dataset_path, f"{dataset_basename}.py"))
        ):
            data = load_dataset(dataset_path, trust_remote_code=True)
            _strategy_print(strategy, f"loaded {dataset_path} with python script")
        # local text file
        elif ext in [".json", ".jsonl", ".csv", ".parquet", ".arrow"]:
            ext = ext.lower().strip(".")
            if ext == "jsonl":
                ext = "json"
            if ext == "json":
                # Use our error handling for JSON files with auto-detection
                data = load_json_with_error_handling(dataset_path, strategy, "auto")
            else:
                data = load_dataset(ext, data_files=dataset_path)
            _strategy_print(strategy, f"loaded {dataset_path} with data_files={dataset_path}")
        # local dataset saved with `datasets.Dataset.save_to_disk`
        elif os.path.isdir(dataset_path):
            try:
                data = load_from_disk(dataset_path)
                _strategy_print(strategy, f"loaded {dataset_path} from disk")
            except Exception as e:
                _strategy_print(strategy, f"failed to load {dataset_path} from disk: {e}")
                try:
                    data = load_directory_with_error_handling(dataset_path, strategy, "auto")
                    _strategy_print(strategy, f"loaded {dataset_path} with error handling")
                except Exception as e2:
                    data = load_dataset(dataset_path, data_dir=data_dir)
                    _strategy_print(strategy, f"loaded {dataset_path} from files")
        # remote/local folder or common file
        elif strategy.args.use_ms:
            from modelscope.msdatasets import MsDataset

            namespace, dataset_name = dataset_path.split("/")
            data = MsDataset.load(dataset_name, namespace=namespace)
        else:
            data = load_dataset(dataset_path, data_dir=data_dir)
            _strategy_print(strategy, f"loaded {dataset_path} from files")

        # Select dataset
        if dataset_split and dataset_split in data:
            data = data[dataset_split]
        data = _maybe_cache_dataset(data, dataset_path, dataset_split, data_dir, strategy)
        data = data.select(range(min(max_count, len(data))))
        data_list.append(data)

    # merge datasets
    if strategy.is_rank_0():
        print(data_list)

    # If probabilities is None, concatenate datasets directly
    if probabilities is None:
        dataset = concatenate_datasets(data_list)
    else:
        dataset = interleave_datasets(
            data_list,
            probabilities=probabilities,
            seed=seed,
            stopping_strategy=stopping_strategy,
        )

    return dataset


def _validate_reward_data_structure(data):
    """Validate if data contains correct reward model fields"""
    if not isinstance(data, dict):
        return False
    
    # Check multiple variants of chosen field
    chosen_keys = ["chosen", "response_chosen", "good", "better", "win"]
    rejected_keys = ["rejected", "response_rejected", "bad", "worse", "lose"]
    
    chosen_key = None
    rejected_key = None
    
    for key in chosen_keys:
        if key in data and data[key]:
            chosen_key = key
            break
    
    for key in rejected_keys:
        if key in data and data[key]:
            rejected_key = key
            break
    
    return chosen_key is not None and rejected_key is not None


def _validate_sft_data_structure(data):
    """Validate if data contains correct SFT fields"""
    if not isinstance(data, dict):
        return False
    
    # Check for common SFT data formats
    # Format 1: input/output keys
    input_keys = ["input", "question", "prompt", "instruction", "query", "context_messages"]
    output_keys = ["output", "response", "answer", "target", "completion"]
    
    # Format 2: conversation format
    conversation_keys = ["conversation", "messages", "dialogue"]
    
    # Check if it has input/output structure
    has_input = any(key in data and data[key] for key in input_keys)
    has_output = any(key in data and data[key] for key in output_keys)
    
    # Check if it has conversation structure
    has_conversation = any(key in data and data[key] for key in conversation_keys)
    
    return has_input or has_conversation


def load_json_with_error_handling(file_path, strategy=None, data_type='auto'):
    """Load JSON file, skip problematic samples
    
    Args:
        file_path: Path to JSON file
        strategy: Training strategy object
        data_type: 'sft', 'reward', or 'auto' for automatic detection
    """
    valid_data = []
    total_count = 0
    valid_count = 0
    validation_failed = 0
    identical_samples = 0
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        
        # Try to parse as complete JSON array
        try:
            data = json.loads(content)
            if isinstance(data, list):
                # Auto-detect data type from first few samples
                if data_type == 'auto' and len(data) > 0:
                    sample_data = data[0] if len(data) == 1 else data[:min(10, len(data))]
                    sft_count = sum(1 for item in (sample_data if isinstance(sample_data, list) else [sample_data]) if _validate_sft_data_structure(item))
                    reward_count = sum(1 for item in (sample_data if isinstance(sample_data, list) else [sample_data]) if _validate_reward_data_structure(item))
                    
                    # Determine data type based on which validation passes more samples
                    if sft_count >= reward_count:
                        data_type = 'sft'
                        if strategy:
                            strategy.print(f"Auto-detected SFT data format in {os.path.basename(file_path)}")
                    else:
                        data_type = 'reward'
                        if strategy:
                            strategy.print(f"Auto-detected Reward data format in {os.path.basename(file_path)}")
                
                for item in data:
                    total_count += 1
                    # Choose validation function based on data type
                    if data_type == 'sft':
                        if _validate_sft_data_structure(item):
                            # Keep original data structure for SFT
                            valid_data.append(item)
                            valid_count += 1
                        else:
                            validation_failed += 1
                    else:  # reward data
                        if _validate_reward_data_structure(item):
                            # Only keep necessary three columns: prompt, chosen, rejected
                            chosen = item.get('chosen', item.get('response_chosen', item.get('good', '')))
                            rejected = item.get('rejected', item.get('response_rejected', item.get('bad', '')))
                            
                            # Skip samples where chosen and rejected are identical
                            if chosen == rejected:
                                identical_samples += 1
                                continue
                                
                            filtered_item = {
                                'prompt': item.get('prompt', ''),
                                'chosen': chosen,
                                'rejected': rejected
                            }
                            valid_data.append(filtered_item)
                            valid_count += 1
                        else:
                            validation_failed += 1
            else:
                # Single object
                total_count = 1
                if data_type == 'auto':
                    if _validate_sft_data_structure(data):
                        data_type = 'sft'
                        if strategy:
                            strategy.print(f"Auto-detected SFT data format in {os.path.basename(file_path)}")
                    elif _validate_reward_data_structure(data):
                        data_type = 'reward'
                        if strategy:
                            strategy.print(f"Auto-detected Reward data format in {os.path.basename(file_path)}")
                
                if data_type == 'sft':
                    if _validate_sft_data_structure(data):
                        valid_data.append(data)
                        valid_count = 1
                else:  # reward data
                    if _validate_reward_data_structure(data):
                        chosen = data.get('chosen', data.get('response_chosen', data.get('good', '')))
                        rejected = data.get('rejected', data.get('response_rejected', data.get('bad', '')))
                        
                        # Skip samples where chosen and rejected are identical
                        if chosen != rejected:
                            filtered_item = {
                                'prompt': data.get('prompt', ''),
                                'chosen': chosen,
                                'rejected': rejected
                            }
                            valid_data.append(filtered_item)
                        valid_count = 1
        except json.JSONDecodeError:
            # Try to parse line by line as JSONL
            valid_data, total_count, valid_count = _parse_json_lines(content, strategy, data_type)
    
    except Exception as e:
        if strategy:
            strategy.print(f"Failed to read file {file_path}: {e}")
        return load_dataset("json", data_files=[])
    
    # Print debug info
    if strategy:
        strategy.print(f"File: {os.path.basename(file_path)}")
        strategy.print(f"  Total samples: {total_count}")
        strategy.print(f"  Validation failed: {validation_failed}")
        strategy.print(f"  Identical chosen/rejected: {identical_samples}")
        strategy.print(f"  Valid samples: {valid_count}")
    
    if not valid_data:
        return load_dataset("json", data_files=[])
    
    # Write valid data to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False, encoding='utf-8') as temp_file:
        for item in valid_data:
            json.dump(item, temp_file, ensure_ascii=False)
            temp_file.write('\n')
        temp_file_path = temp_file.name
    
    try:
        dataset = load_dataset("json", data_files=temp_file_path)
        os.unlink(temp_file_path)  # Clean up temporary file
        return dataset
    except Exception as e:
        os.unlink(temp_file_path)  # Clean up temporary file
        if strategy:
            strategy.print(f"Failed to load processed data: {e}")
        return load_dataset("json", data_files=[])


def _parse_json_lines(content, strategy, data_type='auto'):
    """Parse JSONL format content"""
    valid_data = []
    total_count = 0
    valid_count = 0
    
    lines = content.split('\n')
    
    # Auto-detect data type from first few valid lines
    if data_type == 'auto':
        sample_lines = []
        for line in lines[:10]:  # Check first 10 lines
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                sample_lines.append(data)
                if len(sample_lines) >= 5:  # Enough samples for detection
                    break
            except json.JSONDecodeError:
                continue
        
        if sample_lines:
            sft_count = sum(1 for item in sample_lines if _validate_sft_data_structure(item))
            reward_count = sum(1 for item in sample_lines if _validate_reward_data_structure(item))
            
            if sft_count >= reward_count:
                data_type = 'sft'
                if strategy:
                    strategy.print(f"Auto-detected SFT data format in JSONL")
            else:
                data_type = 'reward'
                if strategy:
                    strategy.print(f"Auto-detected Reward data format in JSONL")
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        total_count += 1
        try:
            data = json.loads(line)
            if data_type == 'sft':
                if _validate_sft_data_structure(data):
                    # Keep original data structure for SFT
                    valid_data.append(data)
                    valid_count += 1
            else:  # reward data
                if _validate_reward_data_structure(data):
                    # Only keep necessary three columns: prompt, chosen, rejected
                    chosen = data.get('chosen', data.get('response_chosen', data.get('good', '')))
                    rejected = data.get('rejected', data.get('response_rejected', data.get('bad', '')))
                    
                    # Skip samples where chosen and rejected are identical
                    if chosen != rejected:
                        filtered_item = {
                            'prompt': data.get('prompt', ''),
                            'chosen': chosen,
                            'rejected': rejected
                        }
                        valid_data.append(filtered_item)
                    valid_count += 1
        except json.JSONDecodeError:
            continue  # Skip invalid lines
    
    return valid_data, total_count, valid_count


def load_directory_with_error_handling(directory_path, strategy, data_type='auto'):
    """Scan directory for data files and load with error handling"""
    datasets_to_concat = []
    
    # Supported file formats
    supported_extensions = ['.json', '.jsonl', '.csv', '.parquet', '.arrow']
    
    for ext in supported_extensions:
        pattern = os.path.join(directory_path, f"*{ext}")
        files = glob.glob(pattern)
        
        for file_path in files:
            try:
                if ext in ['.json', '.jsonl']:
                    # Load JSON files with error handling
                    dataset = load_json_with_error_handling(file_path, strategy, data_type)
                else:
                    # Other formats use standard loading
                    if ext == '.csv':
                        dataset = load_dataset('csv', data_files=file_path)
                    elif ext == '.parquet':
                        dataset = load_dataset('parquet', data_files=file_path)
                    elif ext == '.arrow':
                        dataset = load_dataset('arrow', data_files=file_path)
                
                if len(dataset['train']) > 0:
                    datasets_to_concat.append(dataset['train'])
                    if strategy:
                        strategy.print(f"Successfully loaded file: {file_path}")
            except Exception as e:
                if strategy:
                    strategy.print(f"Failed to load file {file_path}: {e}")
                continue
    
    if not datasets_to_concat:
        # If no data was successfully loaded, return empty dataset
        return load_dataset("json", data_files=[])
    
    # Merge all datasets
    final_dataset = concatenate_datasets(datasets_to_concat)
    return {"train": final_dataset}
