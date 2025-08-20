#!/bin/bash
set -x
cd /xfr_ceph_sh/liuchonghan/OpenRLHF_lao
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=eth0
export NCCL_DEBUG_SUBSYS=ALL

export RAY_HEAD_IP="11.131.250.118"
WORKER_IP="11.131.244.162"

mkdir -p /tmp/ray

ray stop --force
sleep 5

ray start --head \
    --port=6379 \
    --num-cpus=96 \
    --num-gpus=8 \
    --temp-dir=/tmp/ray \
    --dashboard-host=0.0.0.0

sleep 10

sshpass -p '1' ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null root@${WORKER_IP} "
source /xfr_ceph_sh/liuchonghan/envs/etc/profile.d/conda.sh
conda activate openrlhf
cd /xfr_ceph_sh/liuchonghan/OpenRLHF_lao

export https_proxy=http://lidongming:YqN2VZBHtkYe3aNA@proxy.aidataset.qihoo.net:8000/
export http_proxy=http://lidongming:YqN2VZBHtkYe3aNA@proxy.aidataset.qihoo.net:8000/

export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=eth0
export NCCL_DEBUG_SUBSYS=ALL

mkdir -p /tmp/ray
ray stop  --force
sleep 5
ray start --address=${RAY_HEAD_IP}:6379 --num-cpus=96 --num-gpus=8
"

sleep 10

bash /xfr_ceph_sh/liuchonghan/OpenRLHF_lao/examples/scripts/rler_grpo.sh
