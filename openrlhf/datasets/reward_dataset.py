from typing import Callable

from torch.utils.data import Dataset

from openrlhf.datasets.utils import exist_and_not_none
from openrlhf.utils.utils import zero_pad_sequences


def preprocess_data(
    data,
    input_template=None,
    prompt_key=None,
    chosen_key="chosen",
    rejected_key="rejected",
    apply_chat_template=None,
    is_dpo=False,
) -> str:
    if apply_chat_template:
        if prompt_key:
            prompt = apply_chat_template(data[prompt_key], tokenize=False, add_generation_prompt=True)
            chosen = apply_chat_template(data[prompt_key] + data[chosen_key], tokenize=False)[len(prompt) :]
            rejected = apply_chat_template(data[prompt_key] + data[rejected_key], tokenize=False)[len(prompt) :]
        else:
            prompt = ""
            chosen = apply_chat_template(data[chosen_key], tokenize=False)
            rejected = apply_chat_template(data[rejected_key], tokenize=False)

            if is_dpo:
                prompt = apply_chat_template(data[chosen_key][:-1], tokenize=False, add_generation_prompt=True)
                chosen = chosen[len(prompt) :]
                rejected = rejected[len(prompt) :]
    else:
        if prompt_key:
            prompt = data[prompt_key]
            if input_template:
                prompt = input_template.format(prompt)
        else:
            prompt = ""
        chosen = data[chosen_key]
        rejected = data[rejected_key]

    # margin loss
    margin = data["margin"] if exist_and_not_none(data, "margin") else 0

    return prompt, chosen, rejected, margin


class RewardDataset(Dataset):
    """
    Dataset for reward model

    Args:
        dataset: dataset for reward model
        self.tokenizer: self.tokenizer for reward model
        self.max_length: max length of input
    """

    def __init__(
        self,
        dataset,
        tokenizer: Callable,
        max_length: int,
        strategy,
        input_template=None,
        is_dpo=False,
        num_processors=8,
    ) -> None:
        super().__init__()
        self.is_dpo = is_dpo
        self.tokenizer = tokenizer
        self.strategy = strategy
        self.max_length = max_length

        # chat_template
        self.input_template = input_template
        self.prompt_key = getattr(self.strategy.args, "prompt_key", None)
        self.chosen_key = getattr(self.strategy.args, "chosen_key", None)
        self.rejected_key = getattr(self.strategy.args, "rejected_key", None)
        self.apply_chat_template = getattr(self.strategy.args, "apply_chat_template", False)

        if self.apply_chat_template:
            self.apply_chat_template = self.tokenizer.apply_chat_template
            tokenizer_chat_template = getattr(self.strategy.args, "tokenizer_chat_template", None)
            if tokenizer_chat_template:
                self.tokenizer.chat_template = tokenizer_chat_template

        # Parallel loading datasets with error handling
        processed_dataset = dataset.map(
            self.process_data_with_error_handling, 
            remove_columns=dataset.column_names, 
            num_proc=num_processors
        )

        # Filter out None values and invalid data
        processed_dataset = processed_dataset.filter(lambda x: x["prompt"] is not None and x["chosen"] is not None and x["reject"] is not None)

        # Store the processed data in class attributes
        self.prompts = processed_dataset["prompt"]
        self.chosens = processed_dataset["chosen"]
        self.rejects = processed_dataset["reject"]
        self.extras = processed_dataset["extra"]

    def process_data_with_error_handling(self, data):
        """带错误处理的数据处理函数，静默跳过不符合的样本"""
        try:
            return self.process_data(data)
        except Exception:
            # 静默跳过错误样本，不输出任何信息
            return {
                "prompt": None,
                "chosen": None,
                "reject": None,
                "extra": None,
            }

    def process_data(self, data):
        # 检查必要字段是否存在，支持多种字段名变体
        chosen_key = self.chosen_key or "chosen"
        rejected_key = self.rejected_key or "rejected"
        
        # 尝试寻找chosen字段的变体
        chosen_variants = [chosen_key, 'chosen', 'response_chosen', 'answer_chosen', 'good', 'preferred']
        actual_chosen_key = None
        for variant in chosen_variants:
            if variant in data:
                actual_chosen_key = variant
                break
        
        # 尝试寻找rejected字段的变体  
        rejected_variants = [rejected_key, 'rejected', 'response_rejected', 'answer_rejected', 'bad', 'dispreferred']
        actual_rejected_key = None
        for variant in rejected_variants:
            if variant in data:
                actual_rejected_key = variant
                break
        
        # 静默跳过缺少必要字段的数据
        if not actual_chosen_key or not actual_rejected_key:
            raise ValueError("Missing required fields")
        
        # 检查字段值是否为空，静默跳过
        chosen_value = data[actual_chosen_key]
        rejected_value = data[actual_rejected_key]
        
        if not chosen_value or chosen_value is None or not rejected_value or rejected_value is None:
            raise ValueError("Empty field values")
        
        # 临时更新键名以供后续处理使用
        if actual_chosen_key != chosen_key:
            data[chosen_key] = data[actual_chosen_key]
        if actual_rejected_key != rejected_key:
            data[rejected_key] = data[actual_rejected_key]
        
        prompt, chosen, reject, margin = preprocess_data(
            data,
            self.input_template,
            self.prompt_key,
            self.chosen_key,
            self.rejected_key,
            self.apply_chat_template,
            self.is_dpo,
        )

        if self.is_dpo:
            prompt_token = self.tokenizer(
                prompt,
                max_length=self.max_length,
                padding=False,
                truncation=True,
                return_tensors="pt",
                add_special_tokens=False,
            )
            prompt_ids_len = prompt_token["attention_mask"].int().sum().item()

            # Filter the sample whose length is greater than max_length (2 for answer length)
            if prompt_ids_len >= self.max_length - 2:
                prompt = None

        return {
            "prompt": prompt,
            "chosen": chosen,
            "reject": reject,
            "extra": prompt_ids_len if self.is_dpo else margin,
        }

    def __len__(self):
        length = len(self.chosens)
        return length

    def __getitem__(self, idx):
        prompt, chosen, reject, extra = self.prompts[idx], self.chosens[idx], self.rejects[idx], self.extras[idx]

        # 调试：检查chosen和rejected是否相同
        if idx < 5:  # 只打印前5个样本避免刷屏
            print(f"样本 {idx}:")
            print(f"  prompt: {prompt[:50]}...")
            print(f"  chosen: {chosen[:50]}...")
            print(f"  reject: {reject[:50]}...")
            print(f"  chosen==reject: {chosen == reject}")
            print("-" * 50)

        chosen = (prompt + chosen).rstrip("\n")
        if not chosen.endswith(self.tokenizer.eos_token):
            chosen += " " + self.tokenizer.eos_token
        chosen_token = self.tokenizer(
            chosen,
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=False,
        )

        reject = (prompt + reject).rstrip("\n")
        if not reject.endswith(self.tokenizer.eos_token):
            reject += " " + self.tokenizer.eos_token
        reject_token = self.tokenizer(
            reject,
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=False,
        )

        # to avoid EOS_token truncation
        chosen_token["input_ids"][0][-1] = self.tokenizer.eos_token_id
        reject_token["input_ids"][0][-1] = self.tokenizer.eos_token_id
        chosen_token["attention_mask"][0][-1] = True
        reject_token["attention_mask"][0][-1] = True

        return (
            chosen_token["input_ids"],
            chosen_token["attention_mask"],
            reject_token["input_ids"],
            reject_token["attention_mask"],
            extra,
        )

    def collate_fn(self, item_list):
        chosen_ids = []
        chosen_masks = []
        reject_ids = []
        rejects_masks = []
        extras = []
        for chosen_id, chosen_mask, reject_id, rejects_mask, extra in item_list:
            chosen_ids.append(chosen_id)
            chosen_masks.append(chosen_mask)
            reject_ids.append(reject_id)
            rejects_masks.append(rejects_mask)
            extras.append(extra)

        if self.is_dpo:
            padding_side = "right"
        else:
            padding_side = "left"
        chosen_ids = zero_pad_sequences(chosen_ids, side=padding_side, value=self.tokenizer.pad_token_id)
        chosen_masks = zero_pad_sequences(chosen_masks, side=padding_side)
        reject_ids = zero_pad_sequences(reject_ids, side=padding_side, value=self.tokenizer.pad_token_id)
        rejects_masks = zero_pad_sequences(rejects_masks, side=padding_side)
        return chosen_ids, chosen_masks, reject_ids, rejects_masks, extras
