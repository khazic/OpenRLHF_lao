import re
import json
import asyncio
import argparse
from typing import List
from tqdm.asyncio import tqdm_asyncio

import uvicorn
from openai import AsyncOpenAI
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from openrlhf.utils.logging_utils import init_logger

logger = init_logger(__name__)


CRITIC_PROMPT_TEMPLATE = """请对以下答案进行打分（0或1分）：

问题：{question}

标准答案：{label}

待评分答案：{response}

评分标准：
- 如果答案所用语言和问题所用语言一致且答案完全正确：1分
- 其他情况根据答案的正确程度和语言一致性程度给0分

请只输出分数（0或1），不要输出任何其他内容。
""".strip()


def extract_verdict(response: str) -> str:
    match = re.search(r'\b([01])\b', response)
    return match.group(1) if match else None

# =========================
# Core: Async GRM Proxy
# =========================
class AsyncGRMProxy:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        concurrency: int = 30,
        # eos_str="<｜end▁of▁sentence｜>",
        eos_str="<|im_end|>",
        correct_score=1.0,
        format_score=0.0,
        error_score=-0.1,
        timeout=3600,
    ):
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
        self.model = model

        self.semaphore = asyncio.Semaphore(concurrency)

        self.eos_str = eos_str
        self.score_dict = {
            "correct_score": correct_score,
            "format_score": format_score,
            "error_score": error_score,
        }

    async def _score_with_chat(self, prepared_input: str) -> str:
        try:
            resp = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prepared_input}],
            )
            content = resp.choices[0].message.content or ""
            return content.strip()
        except Exception as e:
            logger.debug(f"ChatCompletions fallback failed: {e}")
        return ""

    async def _score_one(self, question: str, response: str, label: str) -> float:
        if not response.endswith(self.eos_str):
            return self.score_dict["error_score"]
        response = response[:-len(self.eos_str)].strip()

        think_pos = response.rfind("</think>")
        if think_pos == -1:
            return self.score_dict["error_score"]
        response = response[think_pos + len("</think>"):].strip() # think after

        prompt = CRITIC_PROMPT_TEMPLATE.format(question=question, response=response, label=label)
        async with self.semaphore:
            critic_text = await self._score_with_chat(prompt)
            verdict = extract_verdict(critic_text)
            score_map = {
                "1": self.score_dict["correct_score"],
                "0": self.score_dict["format_score"],
            }
            score = score_map.get(verdict, self.score_dict["format_score"])
            return float(score)

    async def score_batch(
        self,
        queries: List[str],
        prompts: List[str],
        labels: List[str],
    ) -> List[float]:

        tasks = []
        for query, prompt, label in zip(queries, prompts, labels):
            question = prompt
            response = query[len(prompt):]
            tasks.append(self._score_one(question, response, label))

        # scores = await asyncio.gather(*tasks)
        scores = await tqdm_asyncio.gather(*tasks, desc="Gathering rewards")

        # for query, prompt, label, score in zip(queries, prompts, labels, scores):
        #     question = prompt
        #     response = query[len(prompt):]

            # print("="*50)
            # print(f"question: {question}")
            # # print("-"*50)
            # # print(f"response: {response}")
            # # print("-"*50)
            # # print(f"label: {label}")
            # print("-"*50)
            # print(f"score: {score}")
            # print("="*50)

        return scores


# =========================
# FastAPI Server
# =========================
def build_app(proxy: AsyncGRMProxy) -> FastAPI:
    app = FastAPI()

    @app.get("/health")
    async def health():
        return {"status": "ok", "model": proxy.model}

    @app.post("/get_reward")
    async def get_reward(request: Request):
        data = await request.json()
        queries = data.get("query")
        prompts = data.get("prompts")
        labels = data.get("labels")

        logger.info(f"Received /get_reward with {len(queries)} items")
        rewards = await proxy.score_batch(queries, prompts, labels)

        # import random
        # rewards = [random.randint(0, 1) for _ in range(len(queries))]
        result = {
            "rewards": rewards,
            "scores": rewards,
            "extra_logs": {
                "dummy_scores": rewards
            },
        }
        logger.info(f"Sending {len(rewards)} scores")
        return JSONResponse(result)

    return app


# =========================
# Entrypoint
# =========================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("OpenAI-compatible GRM Server")

    # Server
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)

    # OpenAI
    parser.add_argument("--base_url", type=str, default="http://192.168.4.142:8009/v1")
    parser.add_argument("--api_key", type=str, default="empty")
    parser.add_argument("--model", type=str, default="qwen3", help="Model name, e.g., gpt-4.1-mini or a local OpenAI-compatible model")
    parser.add_argument("--concurrency", type=int, default=30, help="Client-side concurrency limit")

    # Inference / batching
    parser.add_argument("--batch_size", type=int, default=None, help="Batching on our side (task fan-out), not server batch")
    parser.add_argument("--timeout", type=int, default=120, help="Client request timeout (seconds)")

    # Scoring range clipping (optional)
    parser.add_argument("--score_min", type=float, default=None)
    parser.add_argument("--score_max", type=float, default=None)

    # System instruction override
    parser.add_argument("--system_instructions", type=str, default=None)

    return parser.parse_args()


def main():
    args = parse_args()

    proxy = AsyncGRMProxy(
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        concurrency=args.concurrency,
    )

    app = build_app(proxy)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
