#!/usr/bin/env python3
"""
测试离线奖励模型服务器
"""

import requests
import json
import time

def test_reward_server(base_url="http://localhost:8000"):
    """测试奖励模型服务器"""
    
    # 测试数据
    test_data = {
        "query": [
            "你好，我是一个AI助手，很高兴为您服务！",
            "这个问题比较复杂，让我仔细分析一下。",
            "根据我的理解，这个问题的答案是..."
        ],
        "prompts": [
            "请介绍一下自己",
            "能帮我解决这个数学题吗？",
            "请解释一下机器学习的基本概念"
        ],
        "labels": [
            "自我介绍",
            "数学问题",
            "概念解释"
        ]
    }
    
    print("🚀 开始测试离线奖励模型服务器...")
    print(f"服务器地址: {base_url}")
    print(f"测试数据: {len(test_data['query'])} 个对话")
    print()
    
    # 1. 健康检查
    print("1️⃣ 健康检查...")
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            health_data = response.json()
            print(f"✅ 服务健康: {health_data}")
        else:
            print(f"❌ 健康检查失败: {response.status_code}")
            return
    except Exception as e:
        print(f"❌ 无法连接到服务器: {e}")
        return
    
    print()
    
    # 2. 获取服务信息
    print("2️⃣ 获取服务信息...")
    try:
        response = requests.get(f"{base_url}/info")
        if response.status_code == 200:
            info_data = response.json()
            print(f"✅ 服务信息: {json.dumps(info_data, indent=2, ensure_ascii=False)}")
        else:
            print(f"❌ 获取服务信息失败: {response.status_code}")
    except Exception as e:
        print(f"❌ 获取服务信息失败: {e}")
    
    print()
    
    # 3. 测试奖励计算
    print("3️⃣ 测试奖励计算...")
    try:
        start_time = time.time()
        response = requests.post(f"{base_url}/get_reward", json=test_data)
        end_time = time.time()
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ 奖励计算成功!")
            print(f"⏱️  耗时: {end_time - start_time:.2f} 秒")
            print(f"📊 奖励分数: {result['rewards']}")
            print(f"📈 分数统计:")
            print(f"   - 最小值: {min(result['rewards']):.3f}")
            print(f"   - 最大值: {max(result['rewards']):.3f}")
            print(f"   - 平均值: {sum(result['rewards'])/len(result['rewards']):.3f}")
        else:
            print(f"❌ 奖励计算失败: {response.status_code}")
            print(f"错误信息: {response.text}")
    except Exception as e:
        print(f"❌ 奖励计算失败: {e}")
    
    print()
    
    # 4. 测试完整对话格式
    print("4️⃣ 测试完整对话格式...")
    try:
        # 只测试一个对话
        single_data = {
            "query": ["你好，我是一个AI助手"],
            "prompts": ["请介绍一下自己"],
            "labels": ["自我介绍"]
        }
        
        response = requests.post(f"{base_url}/get_reward", json=single_data)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ 单对话测试成功!")
            print(f"📊 奖励分数: {result['rewards'][0]:.3f}")
        else:
            print(f"❌ 单对话测试失败: {response.status_code}")
    except Exception as e:
        print(f"❌ 单对话测试失败: {e}")
    
    print()
    print("🎉 测试完成!")

if __name__ == "__main__":
    test_reward_server()
