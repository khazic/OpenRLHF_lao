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

model = None
tokenizer = None

def load_model():
    """加载 Reward Model"""
    global model, tokenizer
    
    model_path = "/xfr_ceph_sh/liuchonghan/OpenRLHF_lao/examples/scripts/checkpoint/RewardModel_0829_tongyong"
    
    print(f"Loading reward model from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=None,  # 不使用自动设备分配
        num_labels=1,  # reward模型通常只有1个输出
        ignore_mismatched_sizes=True  # 忽略大小不匹配的层
    )
    # 手动移动到GPU
    if torch.cuda.is_available():
        model = model.to("cuda:0")
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
        
        queries = data.get('query', [])
        prompts = data.get('prompts', [])
        labels = data.get('labels', [])
        
        print(f"Received {len(queries)} queries for reward calculation")
        
        # 组合 prompt 和 query 形成完整对话
        full_conversations = []
        for prompt, query in zip(prompts, queries):
            if prompt and query:
                # 组合成完整对话格式
                full_conversation = f"[Human]: {prompt}\n[Assistant]: {query}"
            elif query:
                # 如果只有 query，直接使用
                full_conversation = query
            else:
                full_conversation = prompt
            full_conversations.append(full_conversation)
        
        print(f"Full conversation[0]: {full_conversations[0]}")
        
        rewards = []
        
        batch_size = 8  
        for i in range(0, len(full_conversations), batch_size):
            batch_texts = full_conversations[i:i + batch_size]
            
            # Tokenize
            inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=4096, 
                return_tensors="pt"
            )
            
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)
                scores = outputs.logits.squeeze(-1)
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
    
    app.run(
        host='0.0.0.0',
        port=args.port,
        debug=False,
        threaded=True
    )
