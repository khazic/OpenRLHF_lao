set -x

python -m openrlhf.cli.serve_rm_new \
    --reward_pretrain /llm-align/liuchonghan/RMmodel \
    --port 8000 \
    --bf16 \
    --attn_implementation flash_attention_2 \
    --normalize_reward \
    --max_len 8192 \
    --batch_size 16 \
    --device cuda:0 &

python -m openrlhf.cli.serve_rm_new \
    --reward_pretrain /llm-align/liuchonghan/RMmodel \
    --port 8001 \
    --bf16 \
    --attn_implementation flash_attention_2 \
    --normalize_reward \
    --max_len 8192 \
    --batch_size 16 \
    --device cuda:1 &

python -m openrlhf.cli.serve_rm_new \
    --reward_pretrain /llm-align/liuchonghan/RMmodel \
    --port 8002 \
    --bf16 \
    --attn_implementation flash_attention_2 \
    --normalize_reward \
    --max_len 8192 \
    --batch_size 16 \
    --device cuda:2 &

python -m openrlhf.cli.serve_rm_new \
    --reward_pretrain /llm-align/liuchonghan/RMmodel \
    --port 8003 \
    --bf16 \
    --attn_implementation flash_attention_2 \
    --normalize_reward \
    --max_len 8192 \
    --batch_size 16 \
    --device cuda:3 &

wait