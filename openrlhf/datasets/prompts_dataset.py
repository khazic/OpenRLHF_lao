from torch.utils.data import Dataset
from tqdm import tqdm


def preprocess_data(data, input_template=None, input_key="input", label_key=None, apply_chat_template=None, disable_thinking=False) -> str:
    if apply_chat_template:
        chat = data[input_key]
        if isinstance(chat, str):
            chat = [{"role": "user", "content": chat}]        
        if disable_thinking:
            content = chat[0]["content"] if len(chat) > 0 and "content" in chat[0] else str(chat)
            prompt = f"<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n"
        else:
            # 正常使用tokenizer的chat template
            try:
                prompt = apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
            except Exception as e:
                # 回退到简化格式
                content = chat[0]["content"] if len(chat) > 0 and "content" in chat[0] else str(chat)
                prompt = f"<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n"
    else:
        prompt = data[input_key]
        if input_template:
            prompt = input_template.format(prompt)

    # for Reinforced Fine-tuning
    label = "" if label_key is None else data[label_key]
    return prompt, label


class PromptDataset(Dataset):
    """
    Dataset for PPO model

    Args:
        dataset: dataset for PPO model
        tokenizer: tokenizer for PPO model
        max_length: max length of input
    """

    def __init__(
        self,
        dataset,
        tokenizer,
        strategy,
        input_template=None,
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.tokenizer = tokenizer

        # chat_template
        self.input_template = input_template
        input_key = getattr(self.strategy.args, "input_key", None)
        label_key = getattr(self.strategy.args, "label_key", None)
        apply_chat_template = getattr(self.strategy.args, "apply_chat_template", False)
        disable_thinking = getattr(self.strategy.args, "disable_thinking", False)

        if apply_chat_template:
            apply_chat_template = self.tokenizer.apply_chat_template

        self.prompts = []
        self.labels = []
        self.datasources = []
        for data in tqdm(dataset, desc="Preprocessing data", disable=not self.strategy.is_rank_0()):
            prompt, label = preprocess_data(data, input_template, input_key, label_key, apply_chat_template, disable_thinking)
            self.prompts.append(prompt)
            self.labels.append(label)
            self.datasources.append(data.get("datasource", "default"))

    def __len__(self):
        length = len(self.prompts)
        return length

    def __getitem__(self, idx):
        return self.datasources[idx], self.prompts[idx], self.labels[idx]
