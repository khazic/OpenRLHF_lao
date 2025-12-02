# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Math RLVR (Reinforcement Learning with Verifiable Rewards) reward function.

This module uses math-verify library to verify mathematical answers.
Install dependency: pip install math-verify

This module is meant to be passed to `--remote_rm_url` in OpenRLHF launch scripts.
"""

from __future__ import annotations

import contextlib
import re
from typing import Iterable, List

import torch

try:
    from math_verify.metric import math_metric
    from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig
except ImportError:
    print("To use Math-Verify, please install it first by running `pip install math-verify`.")
    raise


def compute_score(model_output: str, ground_truth: str) -> float:
    """
    Compute score for a single model output against ground truth.

    Args:
        model_output: The model's response containing the answer
        ground_truth: The expected correct answer

    Returns:
        Score of 1.0 if correct, 0.0 otherwise
    """
    verify_func = math_metric(
        gold_extraction_target=(LatexExtractionConfig(),),
        pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
    )
    ret_score = 0.0
    # Wrap the ground truth in \boxed{} format for verification
    ground_truth_boxed = "\\boxed{" + ground_truth + "}"
    with contextlib.suppress(Exception):
        ret_score, _ = verify_func([ground_truth_boxed], [model_output])
    return float(ret_score)


def extract_boxed_answer(text: str) -> str:
    """
    Extract the answer from \\boxed{...} format in the model's response.

    Args:
        text: The model's full response

    Returns:
        The content inside \\boxed{} or empty string if not found
    """
    # Match \boxed{...} pattern, handling nested braces
    pattern = r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}"
    matches = re.findall(pattern, text)
    if matches:
        return matches[-1]  # Return the last match (final answer)
    return ""


def _response_only_texts(queries: Iterable[str], prompts: Iterable[str]) -> List[str]:
    """
    Extract only the response part from full query strings.

    Args:
        queries: Full prompt+response strings
        prompts: Original prompts

    Returns:
        List of response-only strings
    """
    texts: List[str] = []
    for query, prompt in zip(queries, prompts):
        query = str(query)
        prompt = str(prompt)
        if query.startswith(prompt):
            texts.append(query[len(prompt) :])
        else:
            texts.append(query)
    return texts


def reward_func(queries, prompts, labels, **kwargs):
    """
    Reward function called by OpenRLHF for Math RLVR.

    Uses math-verify library to verify if the model's mathematical answer
    is equivalent to the ground truth, supporting various mathematical formats.

    Args:
        queries: list of full prompt+response strings coming from the actor.
        prompts: list of prompt strings (same length as queries).
        labels: list of reference answers (ground-truth answers).

    Returns:
        Dictionary containing:
            - rewards: tensor of rewards for each sample
            - scores: tensor of scores (same as rewards for RLVR)
            - extra_logs: additional logging information
    """
    if not queries:
        raise ValueError("`queries` is empty, cannot compute rewards.")

    # Extract response-only texts (remove prompt prefix)
    responses = _response_only_texts(queries, prompts)
    ground_truths = [str(label) for label in labels]

    # Compute scores for each response
    scores_list = []
    for response, gt in zip(responses, ground_truths):
        score = compute_score(response, gt)
        scores_list.append(score)

    # Convert to tensors
    reward_tensor = torch.tensor(scores_list, dtype=torch.float32)

    return {
        "rewards": reward_tensor,
        "scores": reward_tensor,
        "extra_logs": {
            "math_rlvr_score": reward_tensor,
            "num_correct": (reward_tensor > 0.5).sum().item(),
            "accuracy": (reward_tensor > 0.5).float().mean().item(),
        },
    }

