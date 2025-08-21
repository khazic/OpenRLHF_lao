import torch
from openrlhf.models import get_llm_for_sequence_regression
from openrlhf.utils import get_tokenizer

def interactive_test():
    reward_model = get_llm_for_sequence_regression(
        model_name_or_path="/xfr_ceph_sh/liuchonghan/checkpoints_set/qwen2_5-7b-rm-domain",
        model_type="reward",
        normalize_reward=True,
        bf16=True,
        device_map="auto",
    )
    reward_model.eval()
    
    tokenizer = get_tokenizer(
        model_name_or_path="/xfr_ceph_sh/liuchonghan/checkpoints_set/qwen2_5-7b-rm-domain",
        model=reward_model,
        padding_side="left",
        truncation_side="right",
        use_fast=True,
    )
    
    
    while True:
        text = input("input text: ").strip()
        if text.lower() == 'quit':
            break
            
        if not text:
            continue
            
        with torch.no_grad():
            inputs = tokenizer(
                [text],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=4096,
            )
            
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
                reward_model = reward_model.cuda()
            
            rewards = reward_model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"]
            )
            
            reward_score = rewards[0].item()
            print(f"🎯 reward score: {reward_score:.4f}")

if __name__ == "__main__":
    interactive_test()