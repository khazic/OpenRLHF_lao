import json
import os
import shutil
import argparse
import glob
from datasets import Dataset, concatenate_datasets
from tqdm import tqdm

def get_directory_size(path):
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
    all_data = []
    
    if os.path.isfile(input_path):
        print(f"Processing single file: {input_path}")
        data = load_single_file(input_path)
        all_data.extend(data)
    
    elif os.path.isdir(input_path):
        print(f"Processing directory: {input_path}")
        
        patterns = ['*.json', '*.jsonl']
        
        files_found = []
        for pattern in patterns:
            files = glob.glob(os.path.join(input_path, pattern))
            files_found.extend(files)
        
        for pattern in patterns:
            files = glob.glob(os.path.join(input_path, '**', pattern), recursive=True)
            files_found.extend(files)
        
        files_found = sorted(list(set(files_found)))
        
        if not files_found:
            print(f"Error: No .json or .jsonl files found in directory {input_path}")
            return []
        
        print(f"Found {len(files_found)} data files:")
        for file in files_found[:10]:
            file_size = os.path.getsize(file) / 1024 / 1024
            print(f"  - {os.path.basename(file)} ({file_size:.2f} MB)")
        if len(files_found) > 10:
            print(f"  ... and {len(files_found) - 10} more files")
        
        for file_path in tqdm(files_found, desc="Processing files"):
            try:
                data = load_single_file(file_path)
                all_data.extend(data)
                print(f"  ✅ {os.path.basename(file_path)}: {len(data)} records")
            except Exception as e:
                print(f"  ❌ {os.path.basename(file_path)}: Processing failed - {e}")
                continue
    
    else:
        print(f"Error: Path does not exist or cannot be recognized: {input_path}")
        return []
    
    return all_data

def cleanup_dataset_caches(output_path, *datasets_to_cleanup):
    """Remove Hugging Face cache artifacts and temporary files after verification."""
    for dataset_obj in datasets_to_cleanup:
        if dataset_obj is None:
            continue
        try:
            dataset_obj.cleanup_cache_files()
        except Exception as exc:
            print(f"    Warning: Failed to cleanup dataset cache via API: {exc}")
    
    removed_files = 0
    for pattern in ('cache-*', 'tmp*'):
        pattern_path = os.path.join(output_path, pattern)
        for cache_path in glob.glob(pattern_path):
            try:
                if os.path.isdir(cache_path):
                    shutil.rmtree(cache_path)
                else:
                    os.remove(cache_path)
                removed_files += 1
            except OSError as exc:
                print(f"    Warning: Failed to delete temporary file {os.path.basename(cache_path)}: {exc}")
    
    if removed_files:
        print(f"🧹 Removed {removed_files} leftover cache files")

def load_single_file(file_path):
    data = []
    
    try:
        if file_path.endswith('.jsonl'):
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f):
                    try:
                        item = json.loads(line.strip())
                        if item:
                            data.append(item)
                    except json.JSONDecodeError as e:
                        if line_num < 5:
                            print(f"    Skipping line {line_num+1}, JSON parsing error: {e}")
                        continue
        
        elif file_path.endswith('.json'):
            with open(file_path, 'r', encoding='utf-8') as f:
                content = json.load(f)
                if isinstance(content, list):
                    data = content
                elif isinstance(content, dict):
                    data = [content]
                else:
                    print(f"    Warning: Unknown JSON format: {type(content)}")
        
        else:
            print(f"    Skipping unsupported file format: {file_path}")
    
    except Exception as e:
        print(f"    Failed to read file {file_path}: {e}")
    
    return data

def preprocess_gemini_dataset(input_path, output_path):
    if not os.path.exists(input_path):
        print(f"Error: Input path does not exist: {input_path}")
        return False
    
    print(f"Input dataset: {input_path}")
    print(f"Output path: {output_path}")
    
    if os.path.exists(output_path):
        print(f"Removing existing output directory: {output_path}")
        shutil.rmtree(output_path)
    
    print("Starting dataset preprocessing...")
    
    try:
        data = load_data_from_path(input_path)
        
        if not data:
            print("Error: No data loaded!")
            return False
        
        print(f"Original data count: {len(data)}")
        
        valid_data = []
        invalid_count = 0
        
        if len(data) > 0:
            sample_item = data[0]
            print(f"Sample data fields: {list(sample_item.keys()) if isinstance(sample_item, dict) else 'unknown'}")
        
        for item in tqdm(data, desc="Validating data"):
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
                    standardized_item = {
                        'question': question,
                        'response': response
                    }
                    for key in ['datasource', 'id', 'metadata', 'source']:
                        if key in item:
                            standardized_item[key] = item[key]
                    
                    valid_data.append(standardized_item)
                else:
                    invalid_count += 1
                    if invalid_count <= 3:
                        print(f"Invalid data sample {invalid_count}: {list(item.keys()) if isinstance(item, dict) else item}")
            else:
                invalid_count += 1
        
        print(f"Valid data count: {len(valid_data)}")
        print(f"Invalid data count: {invalid_count}")
        
        if len(valid_data) == 0:
            print("Error: No valid data!")
            print("Please check the data format. Data should contain the following field combinations:")
            print("  Question fields: question, input, prompt, instruction, query")
            print("  Response fields: response, output, answer, completion, target")
            return False
        
        print("\nSample data:")
        example = json.dumps(valid_data[0], ensure_ascii=False, indent=2)
        print(example[:300] + ("..." if len(example) > 300 else ""))
        
        print("Converting to Dataset format...")
        dataset = Dataset.from_list(valid_data)
        
        print("Saving to disk...")
        dataset.save_to_disk(output_path)
        
        dir_size = get_directory_size(output_path)
        dir_size_mb = dir_size / 1024 / 1024
        
        print(f"\n✅ Preprocessing completed!")
        print(f"📁 Data saved to: {output_path}")
        print(f"💾 File size: {dir_size_mb:.2f} MB")
        print(f"📊 Data count: {len(dataset)}")
        
        print("🔍 Verifying saved data...")
        try:
            from datasets import load_from_disk
            loaded_dataset = load_from_disk(output_path)
            print(f"✅ Verification successful! Loaded {len(loaded_dataset)} records")
            print("📝 Sample data:", loaded_dataset[0])
            print("🧹 Cleaning up temporary cache files...")
            cleanup_dataset_caches(output_path, dataset, loaded_dataset)
        except Exception as e:
            print(f"❌ Verification failed: {e}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Error occurred during preprocessing: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(description='Preprocess dataset to Arrow format')
    parser.add_argument('--input', '-i', required=True, 
                       help='Input dataset path (supports file or directory, automatically finds .json and .jsonl files)')
    parser.add_argument('--output', '-o', required=True,
                       help='Output dataset directory path')
    parser.add_argument('--force', '-f', action='store_true',
                       help='Force overwrite output directory (no prompt)')
    
    args = parser.parse_args()
    
    success = preprocess_gemini_dataset(args.input, args.output)
    
    if success:
        print("\n🎉 Data preprocessing completed successfully!")
        print(f"Now you can use in training script: --dataset {args.output}")
    else:
        print("\n💥 Data preprocessing failed!")
        exit(1)

if __name__ == "__main__":
    main()
