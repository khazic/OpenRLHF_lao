import json
import glob
import os

def clean_string(s):
    if not isinstance(s, str):
        return None
    # 确保字符串的引号使用是正确的
    return s.strip()

def process_file(file_path):
    print(f"\n处理文件: {file_path}")
    
    # 读取原始数据
    with open(file_path, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"错误: 文件 {file_path} 不是有效的JSON格式: {e}")
            return
    
    if not isinstance(data, list):
        print(f"错误: 文件 {file_path} 的内容不是列表格式")
        return
    
    original_count = len(data)
    cleaned_data = []
    
    for item in data:
        if not isinstance(item, dict):
            continue
            
        # 只保留需要的字段
        cleaned_item = {}
        valid = True
        
        for field in ['prompt', 'chosen', 'rejected']:
            value = item.get(field)
            if value is None:
                valid = False
                break
                
            cleaned_value = clean_string(value)
            if cleaned_value is None:
                valid = False
                break
                
            cleaned_item[field] = cleaned_value
        
        if valid:
            cleaned_data.append(cleaned_item)
    
    # 写回文件
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(cleaned_data, f, ensure_ascii=False, indent=2)
    
    removed_count = original_count - len(cleaned_data)
    print(f"原始数据条数: {original_count}")
    print(f"清理后数据条数: {len(cleaned_data)}")
    print(f"删除的数据条数: {removed_count}")

def main():
    # 获取所有merged开头的json文件
    files = glob.glob(os.path.join(os.getcwd(), "merged*.json"))
    
    if not files:
        print("没有找到merged开头的JSON文件")
        return
        
    print(f"找到 {len(files)} 个文件需要处理")
    
    for file_path in files:
        process_file(file_path)

if __name__ == "__main__":
    main()