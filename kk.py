import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import requests
import time
import threading

def get_model_name(port, max_retries=3):
    """获取指定端口的可用模型名"""
    url = f"http://localhost:{port}/v1/models"
    
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            result = response.json()
            if 'data' in result and len(result['data']) > 0:
                model_name = result['data'][0]['id']
                print(f"Port {port} model: {model_name}")
                return model_name
            else:
                return "auto"  # 如果获取不到，使用auto
                
        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:
                print(f"Port {port} - Failed to get model name: {str(e)}")
                return "auto"
            time.sleep(1)
    
    return "auto"

def try_api(messages, port, model_name, max_retries=3):
    """连接到指定端口的vLLM OpenAI API服务"""
    url = f"http://localhost:{port}/v1/chat/completions"
    
    payload = {
        "model": model_name,
        "messages": messages,
        "max_tokens": 4096,
        "temperature": 0,
        "top_p": 1.0,
        "top_k": -1,
        "stop": ["<|im_end|>", "<|endoftext|>"] ,
        "stream": False
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=60)
            response.raise_for_status()
            
            result = response.json()
            content = result['choices'][0]['message']['content']
            return True, content
            
        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:
                return False, f"Port {port} - Request failed after {max_retries} attempts: {str(e)}"
            time.sleep(2)  # 等待2秒后重试
    
    return False, f"Port {port} - Unknown error"

def process_single_request(line_data, port, model_name):
    """处理单个请求"""
    line, line_number = line_data
    
    try:
        data = json.loads(line.strip())
        messages = []
        
        # 检查是否有system字段
        if "system" in data and len(data["system"]) > 0:
            messages.append({"role": "system", "content": data["system"]})
        else:
            messages.append({"role": "system", "content": "You are a helpful assistant."})
        
        # 添加用户prompt
        question = data["prompt"]
        messages.append({"role": "user", "content": question})
        
        success, response = try_api(messages, port, model_name)
        
        if success:
            # 保留原始数据，添加响应
            result = data.copy()
            result["response"] = response
            result["port"] = port
            result["model"] = model_name
            return line_number, json.dumps(result, ensure_ascii=False)
        else:
            print(f"Port {port} - Error processing line {line_number}: {response}", file=sys.stderr, flush=True)
            return line_number, None
            
    except Exception as e:
        print(f"Port {port} - Error processing line {line_number}: {e}", file=sys.stderr, flush=True)
        return line_number, None

def process_port(lines, port, output_file):
    """处理指定端口的所有请求"""
    print(f"Starting processing for port {port}...")
    
    # 首先获取模型名
    model_name = get_model_name(port)
    
    # 用于存储结果的字典，保证输出顺序
    results = {}
    
    # 设置线程数，可以根据需要调整
    max_workers = min(5, len(lines))  # 每个端口最多5个并发
    
    # 使用ThreadPoolExecutor进行并行处理
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_line = {executor.submit(process_single_request, line_data, port, model_name): line_data[1] 
                         for line_data in lines}
        
        # 使用tqdm显示进度
        desc = f"Port {port}"
        for future in tqdm(as_completed(future_to_line), total=len(lines), desc=desc):
            line_num = future_to_line[future]
            try:
                result_line_num, result = future.result()
                if result is not None:
                    results[result_line_num] = result
            except Exception as exc:
                print(f'Port {port} - Line {line_num} generated an exception: {exc}', file=sys.stderr, flush=True)
    
    # 按原始顺序输出结果到文件
    with open(output_file, 'w', encoding='utf-8') as f:
        for i in range(len(lines)):
            if i in results:
                f.write(results[i] + '\n')
                f.flush()  # 实时写入
    
    print(f"Port {port} processing completed. Results saved to {output_file}")

def main():
    # 读取所有输入行
    lines = []
    line_number = 0
    
    print("Reading input data...")
    for line in sys.stdin:
        lines.append((line, line_number))
        line_number += 1
    
    if not lines:
        print("No input data found!")
        return
    
    print(f"Read {len(lines)} lines of data")
    
    # 定义端口和对应的输出文件
    port_configs = [
        (8000, "results_port_8000.json"),
        (8001, "results_port_8001.json"), 
        (8002, "results_port_8002.json")
    ]
    
    # 创建线程来并行处理不同端口
    threads = []
    
    for port, output_file in port_configs:
        thread = threading.Thread(
            target=process_port, 
            args=(lines.copy(), port, output_file)
        )
        threads.append(thread)
        thread.start()
    
    # 等待所有线程完成
    for thread in threads:
        thread.join()
    
    print("All ports processing completed!")
    print("Output files:")
    for port, output_file in port_configs:
        print(f"  Port {port}: {output_file}")

if __name__ == "__main__":
    main()

