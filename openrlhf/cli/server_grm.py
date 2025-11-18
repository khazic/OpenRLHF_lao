import argparse
import asyncio
import json
import re
from typing import List, Optional

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio

from openrlhf.utils.logging_utils import init_logger

logger = init_logger(__name__)

CRITIC_PROMPT_TEMPLATE = """
请你充当客观、严格的评审员，根据参考答案对助手的回答进行评分。

【用户问题】
{transed_prompt}

【参考答案】
{answer}

【待评答案】
{response}

【待评答案语种】
{lang}

评分要求：
- 如果待评答案翻译成英文后与参考答案意思完全不一致：0分
- 如果待评答案翻译成英文后与参考答案意思完全一致，且语言与用户问题的语种相同：5分
- 如果待评答案翻译成英文后与参考答案意思部分一致，且语言与用户问题的语种相同，视待评答案的正确程度给2-4分
- 如果待评答案翻译成英文后与参考答案意思部分一致，且语言与用户问题的语种不一致：2分
- 其他情况根据待评答案的正确程度和语言一致性程度给1-2分
请只输出分数（0-5之间的数字），不要输出任何其他内容。
""".strip()

SCORE_PATTERN = re.compile(r"\[\[\s*score\s*=\s*([0-5](?:\.\d+)?)\s*\]\]", re.IGNORECASE)


def extract_score(text: str) -> Optional[float]:
    match = SCORE_PATTERN.search(text)
    if match:
        return float(match.group(1))
    fallback = re.search(r"([0-5](?:\.\d+)?)", text)
    return float(fallback.group(1)) if fallback else None


class AsyncGRMProxy:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        concurrency: int = 30,
        eos_str: str = "<|im_end|>",
        timeout: int = 3600,
        score_min: float = 0.0,
        score_max: float = 5.0,
        default_score: float = 0.0,
        require_eos: bool = True,
        strip_think: bool = True,
    ):
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
        self.model = model
        self.semaphore = asyncio.Semaphore(concurrency)

        self.eos_str = eos_str
        self.score_min = score_min
        self.score_max = score_max
        self.default_score = default_score
        self.require_eos = require_eos
        self.strip_think = strip_think

    async def _score_with_chat(self, prepared_input: str) -> str:
        try:
            resp = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prepared_input}],
            )
            content = resp.choices[0].message.content or ""
            return content.strip()
        except Exception as e:
            logger.debug(f"ChatCompletions failed: {e}")
        return ""

    async def _score_one(self, transed_prompt: str, response: str, lang: str, answer: str) -> float:
        text = response.strip()
        if self.require_eos:
            if not text.endswith(self.eos_str):
                return self.default_score
            text = text[: -len(self.eos_str)].strip()

        if self.strip_think:
            think_pos = text.rfind("`</think>`")
            if think_pos != -1:
                text = text[think_pos + len("`</think>`") :].strip()

        prompt = CRITIC_PROMPT_TEMPLATE.format(
            transed_prompt=transed_prompt.strip(),
            answer=answer.strip(),
            response=text,
            lang=lang.strip()
        )

        async with self.semaphore:
            critic_text = await self._score_with_chat(prompt)
            score = extract_score(critic_text)
            if score is None:
                logger.debug(f"Failed to parse score: {critic_text}")
                return self.default_score
            return float(
                max(self.score_min, min(self.score_max, score))
            )

    async def score_batch(
        self,
        transed_prompts: List[str],
        responses: List[str],
        langs: List[str],
        answers: List[str],
    ) -> List[float]:

        tasks = []
        for transed_prompt, response, lang, answer in zip(transed_prompts, responses, langs, answers):
            tasks.append(self._score_one(transed_prompt, response, lang, answer))

        scores = await tqdm_asyncio.gather(*tasks, desc="Gathering rewards")
        return scores


def build_app(proxy: AsyncGRMProxy) -> FastAPI:
    app = FastAPI()

    @app.get("/health")
    async def health():
        return {"status": "ok", "model": proxy.model}

    @app.post("/get_reward")
    async def get_reward(request: Request):
        data = await request.json()
        transed_prompts = data.get("transed_prompts", [])
        responses = data.get("responses", [])
        langs = data.get("langs", [])
        answers = data.get("answers", [])
        
        if not transed_prompts and "queries" in data:
            queries = data.get("query", [])
            prompts = data.get("prompts", [])
            labels = data.get("labels", [])
            transed_prompts = prompts
            responses = [q[len(p):] if len(q) > len(p) else q for q, p in zip(queries, prompts)]
            answers = labels
            langs = [""] * len(transed_prompts)

        logger.info(f"Received /get_reward with {len(transed_prompts)} items")
        rewards = await proxy.score_batch(transed_prompts, responses, langs, answers)

        result = {
            "rewards": rewards,
            "scores": rewards,
            "extra_logs": {"dummy_scores": rewards},
        }
        logger.info(f"Sending {len(rewards)} scores")
        return JSONResponse(result)

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("OpenAI-compatible GRM Server")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)

    parser.add_argument("--base_url", type=str, default="http://192.168.4.142:8009/v1")
    parser.add_argument("--api_key", type=str, default="empty")
    parser.add_argument(
        "--model", type=str, default="qwen3", help="OpenAI-compatible model name"
    )
    parser.add_argument("--concurrency", type=int, default=30)
    parser.add_argument("--timeout", type=int, default=3600)

    parser.add_argument("--eos_str", type=str, default="<|im_end|>")
    parser.add_argument("--require_eos", action="store_true", default=False)
    parser.add_argument("--strip_think", action="store_true", default=True)

    parser.add_argument("--score_min", type=float, default=0.0)
    parser.add_argument("--score_max", type=float, default=5.0)
    parser.add_argument("--default_score", type=float, default=0.0)

    return parser.parse_args()


def main():
    args = parse_args()

    proxy = AsyncGRMProxy(
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        concurrency=args.concurrency,
        eos_str=args.eos_str,
        timeout=args.timeout,
        score_min=args.score_min,
        score_max=args.score_max,
        default_score=args.default_score,
        require_eos=args.require_eos,
        strip_think=args.strip_think,
    )

    app = build_app(proxy)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
