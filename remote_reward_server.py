#!/usr/bin/env python3
"""
Remote Reward Model Server
为 OpenRLHF 提供 HTTP API 接口的 Reward Model 服务
"""

from flask import Flask, request, jsonify
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# 全局变量
model = None
tokenizer = None

def load_model():
    """加载 Reward Model"""
    global model, tokenizer
    
    model_path = "/xfr_ceph_sh/liuchonghan/Qwen_rm_72b/merged_rm8.52_gptpro-2model"
    
    print(f"Loading reward model from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.eval()
    print("Reward model loaded successfully!")

@app.route('/reward', methods=['POST'])
def get_rewards():
    """
    处理 reward 请求
    
    输入格式：
    {
        "query": ["text1", "text2", ...],
        "prompts": ["prompt1", "prompt2", ...],
        "labels": ["label1", "label2", ...]
    }
    
    输出格式：
    {
        "rewards": [0.85, 0.72, ...],
        "status": "success"
    }
    """
    try:
        data = request.get_json()
        
        # 获取输入数据
        queries = data.get('query', [])
        prompts = data.get('prompts', [])
        labels = data.get('labels', [])
        
        print(f"Received {len(queries)} queries for reward calculation")
        
        texts = queries
        
        rewards = []
        
        batch_size = 8  # 根据GPU内存调整
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize
            inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=4096,  # 根据模型支持的长度调整
                return_tensors="pt"
            )
            
            # 移动到GPU（如果可用）
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            # 推理
            with torch.no_grad():
                outputs = model(**inputs)
                # 获取reward分数
                scores = outputs.logits.squeeze(-1)  # [batch_size]
                rewards.extend(scores.cpu().tolist())
        
        print(f"Calculated rewards: min={min(rewards):.3f}, max={max(rewards):.3f}, mean={sum(rewards)/len(rewards):.3f}")
        
        return jsonify({
            "rewards": rewards,
            "status": "success"
        })
        
    except Exception as e:
        print(f"Error processing reward request: {e}")
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """健康检查接口"""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None
    })

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8000, help='Port to run the server on')
    args = parser.parse_args()
    
    load_model()
    
    # 启动服务器
    app.run(
        host='0.0.0.0',  # 监听所有网络接口
        port=args.port,  # 端口号
        debug=False,     # 生产环境关闭debug
        threaded=True    # 支持多线程
    )
