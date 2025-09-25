#!/usr/bin/env python3
"""
离线奖励模型服务器
为 OpenRLHF 提供离线部署的 Reward Model 服务
支持完整对话打分
"""

import argparse
import json
import logging
from typing import List, Dict, Any

import torch
import uvicorn
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Offline Reward Model Server", version="1.0.0")

# 全局变量
model = None
tokenizer = None
device = None


def load_reward_model(model_path: str, device_name: str = "cuda:0"):
    """加载奖励模型"""
    global model, tokenizer, device
    
    device = device_name if torch.cuda.is_available() else "cpu"
    logger.info(f"Loading reward model from: {model_path}")
    logger.info(f"Using device: {device}")
    
    try:
        # 加载tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # 加载模型
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16 if device != "cpu" else torch.float32,
            device_map=None,  # 不使用自动设备分配
            num_labels=1,
            ignore_mismatched_sizes=True
        )
        
        # 手动移动到指定设备
        model = model.to(device)
        model.eval()
        
        logger.info("Reward model loaded successfully!")
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise e


def format_conversation(prompt: str, query: str, label: str = None) -> str:
    """格式化完整对话"""
    if prompt and query:
        conversation = f"[Human]: {prompt}\n[Assistant]: {query}"
        if label:
            conversation += f"\n[Label]: {label}"
    elif query:
        conversation = query
    else:
        conversation = prompt
    
    return conversation


def calculate_rewards(conversations: List[str], batch_size: int = 8) -> List[float]:
    """计算奖励分数"""
    global model, tokenizer, device
    
    rewards = []
    
    with torch.no_grad():
        for i in range(0, len(conversations), batch_size):
            batch_conversations = conversations[i:i + batch_size]
            
            # Tokenize
            inputs = tokenizer(
                batch_conversations,
                padding=True,
                truncation=True,
                max_length=4096,
                return_tensors="pt"
            )
            
            # 移动到设备
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # 前向传播
            outputs = model(**inputs)
            scores = outputs.logits.squeeze(-1)
            
            # 转换为列表
            if scores.dim() == 0:
                scores = [scores.item()]
            else:
                scores = scores.cpu().tolist()
            
            rewards.extend(scores)
    
    return rewards


@app.post("/get_reward")
async def get_reward(request: Request):
    """
    获取奖励分数
    
    输入格式:
    {
        "query": ["回答1", "回答2", ...],
        "prompts": ["问题1", "问题2", ...],
        "labels": ["标签1", "标签2", ...]  // 可选
    }
    
    输出格式:
    {
        "rewards": [0.85, 0.72, ...],
        "scores": [0.85, 0.72, ...],
        "extra_logs": {
            "dummy_scores": [0.85, 0.72, ...]
        },
        "status": "success"
    }
    """
    try:
        data = await request.json()
        
        queries = data.get("query", [])
        prompts = data.get("prompts", [])
        labels = data.get("labels", [])
        
        # 验证输入
        if not queries:
            raise HTTPException(status_code=400, detail="query field is required")
        
        if prompts and len(queries) != len(prompts):
            raise HTTPException(status_code=400, detail="query and prompts must have the same length")
        
        if labels and len(queries) != len(labels):
            raise HTTPException(status_code=400, detail="query and labels must have the same length")
        
        logger.info(f"Processing {len(queries)} conversations for reward calculation")
        
        # 格式化完整对话
        conversations = []
        for i, query in enumerate(queries):
            prompt = prompts[i] if i < len(prompts) else ""
            label = labels[i] if i < len(labels) else None
            conversation = format_conversation(prompt, query, label)
            conversations.append(conversation)
        
        logger.info(f"Formatted conversation[0]: {conversations[0]}")
        
        # 计算奖励分数
        rewards = calculate_rewards(conversations)
        
        logger.info(f"Calculated rewards: min={min(rewards):.3f}, max={max(rewards):.3f}, mean={sum(rewards)/len(rewards):.3f}")
        
        # 返回结果
        result = {
            "rewards": rewards,
            "scores": rewards,
            "extra_logs": {
                "dummy_scores": rewards
            },
            "status": "success"
        }
        
        return JSONResponse(result)
        
    except Exception as e:
        logger.error(f"Error processing reward request: {e}")
        return JSONResponse(
            {"error": str(e), "status": "error"},
            status_code=500
        )


@app.get("/health")
async def health_check():
    """健康检查"""
    return JSONResponse({
        "status": "healthy",
        "model_loaded": model is not None,
        "device": device
    })


@app.get("/info")
async def get_info():
    """获取服务信息"""
    return JSONResponse({
        "service": "Offline Reward Model Server",
        "version": "1.0.0",
        "model_loaded": model is not None,
        "device": device,
        "endpoints": {
            "get_reward": "POST /get_reward",
            "health": "GET /health",
            "info": "GET /info"
        }
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Offline Reward Model Server")
    
    # 模型参数
    parser.add_argument("--model_path", type=str, required=True, help="Path to the reward model")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to run the model on")
    
    # 服务器参数
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind the server to")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the server on")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for inference")
    
    args = parser.parse_args()
    
    # 加载模型
    load_reward_model(args.model_path, args.device)
    
    # 启动服务器
    logger.info(f"Starting server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
