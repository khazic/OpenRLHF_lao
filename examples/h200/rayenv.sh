cd /mnt/data/liuchonghan/OpenRLHF_lao
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=eth0
export NCCL_DEBUG_SUBSYS=ALL

ray stop --force
sleep 5
ray start --head --port=6379 --num-cpus=96 --num-gpus=8 --temp-dir=/tmp/ray --dashboard-host=0.0.0.0



cd /mnt/data/liuchonghan/OpenRLHF_lao
source /mnt/data/liuchonghan/env/etc/profile.d/conda.sh
conda activate openrlhf
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=eth0
export NCCL_DEBUG_SUBSYS=ALL
ray stop --force
sleep 5
ray start --address=11.131.250.118:6379 --num-cpus=96 --num-gpus=8




ray status
bash examples/h200/rl_rlvr_llmjudge.sh


ray status
ray list tasks --detail
ray list actors --detail

