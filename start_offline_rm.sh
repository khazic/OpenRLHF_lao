#!/bin/bash
set -x

# 离线奖励模型服务器启动脚本

# 模型路径
MODEL_PATH="/xfr_ceph_sh/liuchonghan/OpenRLHF_lao/examples/scripts/checkpoint/RewardModel_0829_tongyong"

# 服务器配置
HOST="0.0.0.0"
PORT=8000
DEVICE="cuda:0"
BATCH_SIZE=8

# 启动服务器
/xfr_ceph_sh/liuchonghan/envs/envs/openrlhf/bin/python offline_reward_server.py \
    --model_path "$MODEL_PATH" \
    --host "$HOST" \
    --port "$PORT" \
    --device "$DEVICE" \
    --batch_size "$BATCH_SIZE"
