#!/bin/bash
set -e

HEAD_NODE="10.181.107.66"
RAY_PORT="6379"
NUM_CPUS=128
NUM_GPUS=8

OPENRLHF_PREFIX="/mnt/data/liuchonghan/env/openrlhf"
SETUP_SCRIPT="/mnt/data/liuchonghan/setup_openrlhf_safe.sh"

echo "🚀 开始配置并加入 Ray 集群..."
echo "主节点: ${HEAD_NODE}:${RAY_PORT}"
echo ""

if [ -f "$SETUP_SCRIPT" ]; then
    echo "📦 配置 OpenRLHF 环境..."
    cd /mnt/data/liuchonghan
    bash "$SETUP_SCRIPT"
    source ~/.bashrc
    echo ""
fi

echo "🔧 加载环境变量..."
export OPENRLHF_PREFIX="/mnt/data/liuchonghan/env/openrlhf"
export OPENRLHF_BIN="$OPENRLHF_PREFIX/bin"
export OPENRLHF_SITE="$OPENRLHF_PREFIX/lib/python3.10/site-packages"
export PATH="$OPENRLHF_BIN:$PATH"

if [ -d "$OPENRLHF_SITE" ]; then
    if [ -z "$PYTHONPATH" ]; then
        export PYTHONPATH="$OPENRLHF_SITE"
    else
        export PYTHONPATH="$OPENRLHF_SITE:$PYTHONPATH"
    fi
fi

export CONDA_PREFIX="$OPENRLHF_PREFIX"

echo "🌐 配置网络环境..."
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=eth0

echo "🛑 停止旧的 Ray 进程..."
ray stop --force 2>/dev/null || true
sleep 3

echo "🔗 加入 Ray 集群..."
ray start \
  --address="${HEAD_NODE}:${RAY_PORT}" \
  --num-cpus=${NUM_CPUS} \
  --num-gpus=${NUM_GPUS}

echo ""
echo "✅ 成功加入 Ray 集群！"
echo "节点 IP: $(hostname -I | awk '{print $1}')"
echo "主节点: ${HEAD_NODE}:${RAY_PORT}"
echo ""
echo "在主节点运行 'ray status' 查看集群状态"