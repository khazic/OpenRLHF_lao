set -x

python -m openrlhf.cli.serve_rm \
    --reward_pretrain /xfr_ceph_sh/liuchonghan/Qwen_rm_72b/merged_rm8.52_gptpro-2model \
    --port 8000 \
    --bf16 \
    --attn_implementation flash_attention_2 \
    --normalize_reward \
    --max_len 8192 \
    --batch_size 16