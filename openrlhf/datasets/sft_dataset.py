from typing import Callable, List

import torch
from torch.utils.data import Dataset, Sampler

from openrlhf.utils.utils import zero_pad_sequences


def preprocess_data(
    data, input_template=None, input_key="input", output_key=None, apply_chat_template=None, multiturn=False
):
    if apply_chat_template:
        if output_key:
            prompt_message = data[input_key]
            response_message = data[output_key]

            if isinstance(prompt_message, str) and isinstance(response_message, str):
                prompt_message = [{"role": "user", "content": prompt_message}]
                response_message = [{"role": "assistant", "content": response_message}]

            prompt = apply_chat_template(prompt_message, tokenize=False, add_generation_prompt=True)
            response = apply_chat_template(prompt_message + response_message, tokenize=False)[len(prompt) :]
        else:
            prompt = apply_chat_template(data[input_key][:-1], tokenize=False, add_generation_prompt=True)
            response = apply_chat_template(data[input_key], tokenize=False)[len(prompt) :]
    else:
        prompt = data[input_key]
        if input_template:
            prompt = input_template.format(prompt)
        # output_key is None for continue pretrain
        response = data[output_key] if output_key else ""
    return prompt, response


class SFTDataset(Dataset):
    """
    Dataset for SFT model

    Args:
        dataset: dataset for SFT model
        tokenizer: tokenizer for SFT model
        max_length: max length of input
    """

    def __init__(
        self,
        dataset,
        tokenizer: Callable,
        max_length: int,
        strategy,
        input_template=None,
        pretrain_mode=False,
        num_processors=8,  # Specify the number of processors you want to use
        multiturn=False,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.strategy = strategy
        self.pretrain_mode = pretrain_mode
        self.max_length = max_length
        self.multiturn = multiturn

        # chat template
        self.input_template = input_template
        self.input_key = getattr(self.strategy.args, "input_key", None)
        self.output_key = getattr(self.strategy.args, "output_key", None)
        self.apply_chat_template = getattr(self.strategy.args, "apply_chat_template", False)

        if self.apply_chat_template:
            self.apply_chat_template = self.tokenizer.apply_chat_template
            tokenizer_chat_template = getattr(self.strategy.args, "tokenizer_chat_template", None)
            if tokenizer_chat_template:
                self.tokenizer.chat_template = tokenizer_chat_template

        # Parallel loading datasets
        processed_dataset = dataset.map(
            self.process_data,
            remove_columns=dataset.column_names,
            num_proc=num_processors,
        )
        processed_dataset = processed_dataset.filter(lambda x: x["prompt"] is not None)

        # Store the processed data in class attributes
        self.prompts = processed_dataset["prompt"]
        self.responses = processed_dataset["response"]
        self.prompt_ids_lens = processed_dataset["prompt_ids_len"]
        self.response_ranges = processed_dataset["response_ranges"] if self.multiturn else None

    def process_data(self, data):
        if self.multiturn and self.output_key:
            data[self.input_key].append(data[self.output_key])
            data[self.output_key] = None

        if self.multiturn:
            assert (
                not self.output_key or not data[self.output_key]
            ), "You should put the whole trajectory into data[input_key] and do not set output_key"
            input_key = self.input_key
            apply_chat_template = self.apply_chat_template
            response_ranges = []
            for idx, message in enumerate(data[input_key]):
                if message["role"] == "assistant":
                    prompt = apply_chat_template(data[input_key][:idx], tokenize=False, add_generation_prompt=True)
                    response = apply_chat_template(data[input_key][: idx + 1], tokenize=False)[len(prompt) :]

                    start_idx = (
                        self.tokenizer(
                            prompt,
                            max_length=self.max_length,
                            padding=False,
                            truncation=True,
                            return_tensors="pt",
                            add_special_tokens=False,
                        )["attention_mask"]
                        .int()
                        .sum()
                        .item()
                    )

                    end_idx = (
                        start_idx
                        + self.tokenizer(
                            response,
                            max_length=self.max_length,
                            padding=False,
                            truncation=True,
                            return_tensors="pt",
                            add_special_tokens=False,
                        )["attention_mask"]
                        .int()
                        .sum()
                        .item()
                        - 1
                    )
                    response_ranges.append((start_idx, end_idx))  # left close right close

        prompt, response = preprocess_data(
            data,
            None if self.pretrain_mode else self.input_template,
            self.input_key,
            self.output_key,
            apply_chat_template=None if self.pretrain_mode else self.apply_chat_template,
            multiturn=self.multiturn,
        )

        if not self.pretrain_mode:
            prompt_token = self.tokenizer(
                prompt,
                max_length=self.max_length,
                padding=False,
                truncation=True,
                return_tensors="pt",
                add_special_tokens=False,
            )
            prompt_ids_len = prompt_token["attention_mask"].int().sum().item()
            # filter the sample whose length is greater than max_length (2 for answer length)
            if not prompt or not response or prompt_ids_len >= self.max_length - 2:
                prompt = None
        else:
            prompt_ids_len = 0

        return {
            "prompt": prompt,
            "response": response,
            "prompt_ids_len": prompt_ids_len,
            "response_ranges": response_ranges if self.multiturn else None,
        }

    def __len__(self):
        length = len(self.prompts)
        return length

    def __getitem__(self, idx):
        prompt = self.prompts[idx]
        response = self.responses[idx]

        if not self.pretrain_mode:
            text = (prompt + response).rstrip("\n")
            if not text.endswith(self.tokenizer.eos_token):
                text += " " + self.tokenizer.eos_token
        else:
            text = prompt

        input_token = self.tokenizer(
            text,
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=False,
        )
        input_ids = input_token["input_ids"]
        attention_mask = input_token["attention_mask"]
        loss_mask = self.get_loss_mask(input_ids, idx)

        if not self.pretrain_mode:
            # to avoid EOS_token truncation
            input_ids[0][-1] = self.tokenizer.eos_token_id
            attention_mask[0][-1] = True
        return input_ids, attention_mask, loss_mask

    def get_loss_mask(self, input_ids, idx):
        if self.pretrain_mode:
            return torch.ones_like(input_ids, dtype=torch.float32)  # shape:[1, seq_len]

        loss_mask = torch.zeros_like(input_ids, dtype=torch.float32)
        if not self.multiturn:
            prompt_ids_len = self.prompt_ids_lens[idx]
            loss_mask[0, prompt_ids_len - 1 : -1] = 1
        else:
            response_ranges = self.response_ranges[idx]
            for start_idx, end_idx in response_ranges:
                loss_mask[0, start_idx - 1 : end_idx] = 1
        return loss_mask

    def collate_fn(self, item_list):
        input_ids = []
        attention_masks = []
        loss_masks = []

        for input_id, attention_mask, loss_mask in item_list:
            input_ids.append(input_id)
            attention_masks.append(attention_mask)
            loss_masks.append(loss_mask)

        input_ids = zero_pad_sequences(input_ids, "right", self.tokenizer.pad_token_id)
        attention_masks = zero_pad_sequences(attention_masks, "right")
        loss_masks = zero_pad_sequences(loss_masks, "right")
        return input_ids, attention_masks, loss_masks


class DynamicBatchSampler(Sampler):
    """
    Dynamic batch sampler that groups samples to ensure total tokens per batch <= max_tokens.
    
    This sampler uses First Fit Decreasing algorithm to pack samples into batches,
    ensuring each batch's total token count doesn't exceed max_tokens_per_gpu.
    
    可以和 --packing_samples 组合使用：
    - 只用 --use_dynamic_batch：动态 batch size，普通 padding
    - 只用 --packing_samples：固定 batch size，unpad packing（原有行为）
    - 两者都用：动态 batch size + unpad packing（最安全高效）
    
    Example:
        max_tokens_per_gpu = 8000
        Sample lengths: [4000, 3500, 3800, 4096]
        
        Result batches:
        - Batch 1: [sample_0, sample_1] = 7500 tokens
        - Batch 2: [sample_2, sample_3] = 7896 tokens
    """
    
    def __init__(
        self,
        dataset: SFTDataset,
        max_tokens_per_gpu: int,
        shuffle: bool = True,
        seed: int = 42,
        num_replicas: int = 1,
        rank: int = 0,
    ):
        self.dataset = dataset
        self.max_tokens_per_gpu = max_tokens_per_gpu
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self.num_replicas = num_replicas  # 分布式训练的总进程数
        self.rank = rank  # 当前进程的 rank
        
        # 预先计算每个样本的长度（复用已有的 tokenize 结果）
        self.sample_lengths = self._compute_sample_lengths()
        # 计算 batch 分组
        self.batches = self._create_batches()
    
    def _compute_sample_lengths(self) -> List[int]:
        """计算每个样本的 token 长度（使用近似估计，避免重复 tokenize）"""
        lengths = []
        for idx in range(len(self.dataset)):
            prompt = self.dataset.prompts[idx]
            response = self.dataset.responses[idx]
            
            # 使用字符长度的近似估计（平均 4 字符 = 1 token）
            # 或者直接用 prompt_ids_len + response 估计
            if not self.dataset.pretrain_mode:
                prompt_len = self.dataset.prompt_ids_lens[idx]
                # 估计 response 长度（可以用字符数 / 4 来近似）
                response_len = len(response) // 4 + 1
                total_len = min(prompt_len + response_len, self.dataset.max_length)
            else:
                total_len = min(len(prompt) // 4 + 1, self.dataset.max_length)
            
            lengths.append(total_len)
        return lengths
    
    def _create_batches(self) -> List[List[int]]:
        """使用 First Fit Decreasing 算法创建 batch"""
        import random
        
        # 获取所有样本索引
        indices = list(range(len(self.sample_lengths)))
        
        if self.shuffle:
            random.seed(self.seed + self.epoch)
            random.shuffle(indices)
        
        # 按长度排序（降序），有助于更好地 packing
        sorted_indices = sorted(indices, key=lambda i: self.sample_lengths[i], reverse=True)
        
        batches = []  # List of (batch_indices, total_tokens)
        
        for idx in sorted_indices:
            length = self.sample_lengths[idx]
            
            # 尝试找到一个可以容纳当前样本的 batch
            placed = False
            for i, (batch_indices, total_tokens) in enumerate(batches):
                if total_tokens + length <= self.max_tokens_per_gpu:
                    batches[i] = (batch_indices + [idx], total_tokens + length)
                    placed = True
                    break
            
            if not placed:
                # 创建新的 batch
                batches.append(([idx], length))
        
        # 只返回 indices
        all_batches = [batch_indices for batch_indices, _ in batches]
        
        # 分布式训练：每个 rank 只取自己的 batch
        if self.num_replicas > 1:
            # 确保每个 rank 的 batch 数量相同（向上取整）
            num_batches = len(all_batches)
            num_batches_per_rank = (num_batches + self.num_replicas - 1) // self.num_replicas
            
            # 填充 batch 使其可以均匀分配
            while len(all_batches) < num_batches_per_rank * self.num_replicas:
                all_batches.append(all_batches[len(all_batches) % num_batches])
            
            # 每个 rank 取自己的部分
            all_batches = all_batches[self.rank::self.num_replicas]
        
        return all_batches
    
    def __iter__(self):
        # 每个 epoch 重新创建 batches（如果 shuffle）
        if self.shuffle:
            self.batches = self._create_batches()
        
        for batch_indices in self.batches:
            yield batch_indices
    
    def __len__(self):
        return len(self.batches)
    
    def set_epoch(self, epoch: int):
        """设置 epoch，用于 shuffle"""
        self.epoch = epoch
        if self.shuffle:
            self.batches = self._create_batches()