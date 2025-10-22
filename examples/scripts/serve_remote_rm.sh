set -x

/xfr_ceph_sh/liuchonghan/envs/envs/openrlhf/bin/python - -m openrlhf.cli.serve_rm \
    --reward_pretrain /xfr_ceph_sh/liuchonghan/OpenRLHF_lao/examples/scripts/checkpoint/RewardModel_0829_tongyong \
    --port 8000 \
    --bf16 \
    --attn_implementation flash_attention_2 \
    --normalize_reward \
    --max_len 8192 \
    --batch_size 16


# curl -X POST "http://localhost:8000/get_reward" \
#      -H "Content-Type: application/json" \
#      -d '{"query": ["Hello world"], "prompts": ["Hi there"]}'
