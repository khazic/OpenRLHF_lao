#!/bin/bash
echo "Starting Remote Reward Model Server..."

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TOKENIZERS_PARALLELISM=false

/xfr_ceph_sh/liuchonghan/envs/envs/openrlhf/bin/python remote_reward_server.py --port 8000 &

wait

echo "Remote Reward Model Server stopped."
