#!/usr/bin/env python3
"""Multiple-choice RLVR reward function."""
from __future__ import annotations

import contextlib
import json
import os
import re
from typing import Iterable, List

import torch

DEFAULT_CHOICES = ("A", "B", "C", "D", "E")
BOXED_PATTERN = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")
CHOICE_PATTERN = re.compile(
    r"(?:answer|option|choice)?\s*[:=]?\s*([A-Za-z])\b", re.IGNORECASE
)


def _extract_boxed_answer(text: str) -> str:
    matches = BOXED_PATTERN.findall(text)
    return matches[-1] if matches else ""


def _normalize_choice(text: str, valid_choices=DEFAULT_CHOICES) -> str:
    text = (text or "").strip().upper()
    for char in text:
        if char in valid_choices:
            return char
    return ""


def extract_choice(text: str, valid_choices=DEFAULT_CHOICES) -> str:
    """
    Extract a single-letter choice, preferring \\boxed{} values but falling back
    to phrases like "Answer: C" or the first standalone letter.
    """
    text = str(text or "")
    candidate = _normalize_choice(_extract_boxed_answer(text), valid_choices)
    if candidate:
        return candidate
    match = CHOICE_PATTERN.search(text)
    if match:
        candidate = _normalize_choice(match.group(1), valid_choices)
        if candidate:
            return candidate
    return _normalize_choice(text, valid_choices)


def _response_only_texts(queries: Iterable[str], prompts: Iterable[str]) -> List[str]:
    """Strip prompt prefixes from query strings."""
    texts: List[str] = []
    for query, prompt in zip(queries, prompts):
        query = str(query)
        prompt = str(prompt)
        texts.append(query[len(prompt) :] if query.startswith(prompt) else query)
    return texts


def _truncate_text(text: str, limit: int) -> str:
    if limit <= 0 or len(text) <= limit:
        return text
    truncated = text[:limit]
    return f"{truncated}... <truncated {len(text) - limit} chars>"


def compute_score(model_output: str, ground_truth: str) -> float:
    model_choice = extract_choice(model_output)
    gold_choice = extract_choice(ground_truth)
    return 1.0 if model_choice and gold_choice and model_choice == gold_choice else 0.0


def reward_func(queries, prompts, labels, **kwargs):
    if not queries:
        raise ValueError("`queries` is empty, cannot compute rewards.")

    queries = [str(query) for query in queries]
    prompts = [str(prompt) for prompt in prompts]
    responses = _response_only_texts(queries, prompts)
    ground_truths = [str(label) for label in labels]

    scores_list = []
    zero_score_details = []
    for idx, (response, gt) in enumerate(zip(responses, ground_truths)):
        score = compute_score(response, gt)
        scores_list.append(score)
        if score <= 0.0:
            zero_score_details.append(
                {
                    "index": idx,
                    "prompt": str(prompts[idx]),
                    "response": response,
                    "ground_truth": gt,
                    "model_choice": extract_choice(response),
                    "gold_choice": extract_choice(gt),
                }
            )

    reward_tensor = torch.tensor(scores_list, dtype=torch.float32)

    print_limit = int(os.getenv("ZERO_SCORE_PRINT_LIMIT", "0"))
    print_chars = int(os.getenv("ZERO_SCORE_PRINT_CHARS", "0"))
    if print_limit > 0 and zero_score_details:
        print(f"[mc_rlvr_reward_func] zero-score samples (showing up to {print_limit}):")
        for rank, detail in enumerate(zero_score_details[:print_limit], start=1):
            print(f"--- Zero-score #{rank} | sample_index={detail['index']}")
            print(f"  Ground Truth : {detail['ground_truth']} (choice={detail['gold_choice']})")
            print(f"  Model Choice : {detail['model_choice']}")
            print(f"  Prompt       : {_truncate_text(detail['prompt'], print_chars)}")
            print(f"  Response     : {_truncate_text(detail['response'], print_chars)}")
            print("---")

    log_path = os.getenv("ZERO_SCORE_LOG_PATH")
    if log_path and zero_score_details:
        with contextlib.suppress(OSError):
            with open(log_path, "a", encoding="utf-8") as log_file:
                for detail in zero_score_details:
                    log_file.write(json.dumps(detail, ensure_ascii=False))
                    log_file.write("\n")

    return {
        "rewards": reward_tensor,
        "scores": reward_tensor,
        "extra_logs": {
            "mc_rlvr_score": reward_tensor,
        },
    }
