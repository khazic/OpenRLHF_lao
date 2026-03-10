"""
Example reward function that uses an external LLM judge (GRM) to score answers.

Fill in DEFAULT_BASE_URL / DEFAULT_API_KEY / DEFAULT_MODEL_NAME with the endpoint
of your critic model (any OpenAI-compatible server works).

This module is meant to be passed to `--remote_rm_url` in OpenRLHF launch scripts.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Iterable, List, Sequence

import torch

try:
    from openai import AsyncOpenAI
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError("`openai` package is required for the LLM judge reward.") from exc

CRITIC_PROMPT_TEMPLATE = """\
Act as an impartial judge to evaluate the answer quality.
You receive one user question and two candidate answers (Assistant A/B).
Decide which answer better follows the instructions considering helpfulness,
relevance, accuracy, depth, creativity, and harmlessness. Ignore answer order
or length biases.

Return your final verdict strictly as:
  [[A]] if Assistant A is better
  [[B]] if Assistant B is better
  [[Tie]] if they are equivalent

[User Question]
{question}

[Assistant A]
{candidate}

[Assistant B]
{reference}
""".strip()

DEFAULT_BASE_URL = "http://192.168.4.136:8009/v1"
DEFAULT_API_KEY = "empty"
DEFAULT_MODEL_NAME = "qwq"


@dataclass
class AsyncLLMJudge:
    """Thin wrapper around an OpenAI-compatible endpoint working as a judge."""

    base_url: str = DEFAULT_BASE_URL
    api_key: str = DEFAULT_API_KEY
    model_name: str = DEFAULT_MODEL_NAME
    max_concurrent_requests: int = 32

    def __post_init__(self) -> None:
        self._client = AsyncOpenAI(base_url=self.base_url, api_key=self.api_key)
        self._semaphore = asyncio.Semaphore(self.max_concurrent_requests)

    async def score_pair(self, question: str, candidate: str, reference: str, **kwargs) -> float:
        """Ask the judge to compare candidate vs. reference answer."""
        prompt = CRITIC_PROMPT_TEMPLATE.format(
            question=question.strip(),
            candidate=candidate.strip() or "<empty>",
            reference=reference.strip() or "<empty>",
        )
        messages = [{"role": "user", "content": prompt}]
        async with self._semaphore:
            completion = await self._client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.0,
                **kwargs,
            )
        verdict = completion.choices[0].message.content.strip()
        return self._verdict_to_reward(verdict)

    async def score_batch(
        self,
        questions: Sequence[str],
        candidates: Sequence[str],
        references: Sequence[str],
        **kwargs,
    ) -> List[float]:
        tasks = [
            self.score_pair(q, c, r, **kwargs)
            for q, c, r in zip(questions, candidates, references)
        ]
        return await asyncio.gather(*tasks)

    @staticmethod
    def _verdict_to_reward(verdict: str) -> float:
        verdict_upper = verdict.upper()
        if "[[A]]" in verdict_upper or verdict_upper.strip() == "A":
            return 1.0
        if "[[B]]" in verdict_upper or verdict_upper.strip() == "B":
            return -1.0
        return 0.0


_judge = AsyncLLMJudge()


def _response_only_texts(queries: Iterable[str], prompts: Iterable[str]) -> List[str]:
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
    Reward function called by OpenRLHF.

    Args:
        queries: list of full prompt+response strings coming from the actor.
        prompts: list of prompt strings (same length as queries).
        labels: list of reference answers (e.g., ground-truth responses).
    """

    if not queries:
        raise ValueError("`queries` is empty, cannot call reward model.")

    questions = [str(p) for p in prompts]
    candidates = _response_only_texts(queries, prompts)
    references = [str(label) for label in labels]

    rewards = asyncio.run(
        _judge.score_batch(
            questions=questions,
            candidates=candidates,
            references=references,
            **kwargs,
        )
    )

    reward_tensor = torch.tensor(rewards, dtype=torch.float32)
    score_tensor = (reward_tensor + 1.0) / 2.0  # map {-1,0,1} -> {0,0.5,1}

    return {
        "rewards": reward_tensor,
        "scores": score_tensor,
        "extra_logs": {
            "llm_judge_raw_reward": reward_tensor,
            "llm_judge_score": score_tensor,
        },
    }
