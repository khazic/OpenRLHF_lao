#!/bin/bash
set -x
cd /xfr_ceph_sh/liuchonghan/OpenRLHF_lao

export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=eth0
export NCCL_DEBUG_SUBSYS=ALL

export MASTER_ADDR="11.131.250.118"
WORKER_IP="11.131.244.162"

export https_proxy=http://lidongming:YqN2VZBHtkYe3aNA@proxy.aidataset.qihoo.net:8000/
export http_proxy=http://lidongming:YqN2VZBHtkYe3aNA@proxy.aidataset.qihoo.net:8000/

export TRANSFORMERS_CACHE="/xfr_ceph_sh/liuchonghan/hf_cache"
export HF_HOME="/xfr_ceph_sh/liuchonghan/hf_home"
export TRITON_CACHE_DIR="/xfr_ceph_sh/liuchonghan/triton_cache"
export TORCH_HOME="/xfr_ceph_sh/liuchonghan/torch_home"

mkdir -p /xfr_ceph_sh/liuchonghan/hf_cache
mkdir -p /xfr_ceph_sh/liuchonghan/hf_home
mkdir -p /xfr_ceph_sh/liuchonghan/triton_cache
mkdir -p /xfr_ceph_sh/liuchonghan/torch_home

echo "🚀 开始在工作节点设置代理..."

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

echo \"✅ 工作节点代理设置完成\"
"

echo "✅ 所有节点环境设置完成，开始执行SFT训练..."

sleep 5

bash examples/scripts/sft_nodes.sh