import time
import ray
import requests

from openrlhf.utils.logging_utils import init_logger

logger = init_logger(__name__)


def request_api_wrapper(url, data, try_max_times=5):
    """Synchronous request API wrapper"""
    headers = {
        "Content-Type": "application/json",
    }
    for _ in range(try_max_times):
        try:
            response = requests.post(url=url, json=data, headers=headers, timeout=180)
            response.raise_for_status()  # Raise an HTTPError for bad responses
            response = response.json()
            return response
        except requests.RequestException as e:
            logger.info(f"Request error, please check: {e}")
        except Exception as e:
            logger.info(f"Unexpected error, please check: {e}")
        time.sleep(1)

    raise Exception(f"Request error for {try_max_times} times, returning None. Please check the API server.")


@ray.remote
def remote_rm_fn_ray(api_url, queries, prompts, labels, transed_prompts, langs, answers):
    # 新格式：发送完整的字段给 GRM
    data = {
        "transed_prompts": transed_prompts,
        "responses": [q[len(p):] if q.startswith(p) else q for q, p in zip(queries, prompts)],
        "langs": langs,
        "answers": answers,
    }
    # 如果没有新字段，使用旧格式作为fallback
    if not transed_prompts or not langs:
        data = {"query": queries, "prompts": prompts, "labels": labels}
    return request_api_wrapper(api_url, data)


@ray.remote
class RemoteRewardModel:
    def __init__(self, args, remote_rm_url):
        self.args = args
        self.remote_rm_url = [remote_rm_url] if isinstance(remote_rm_url, str) else remote_rm_url
        self.custom_reward_func = None

        if self.remote_rm_url and self.remote_rm_url[0].endswith(".py"):
            print(f"Loading custom `reward_func(queries, prompts, labels)` from {self.remote_rm_url[0]}")
            import importlib.util

            spec = importlib.util.spec_from_file_location("reward_func", self.remote_rm_url[0])
            reward_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(reward_module)
            self.custom_reward_func = ray.remote(reward_module.reward_func)

    def get_rewards(self, queries_list, prompts_list, labels_list, metadata=None):
        # 从 metadata 中提取额外字段
        if metadata:
            transed_prompts_list = [m["transed_prompt"] for m in metadata]
            langs_list = [m["lang"] for m in metadata]
            answers_list = labels_list  # answers 就是 labels
        else:
            transed_prompts_list = []
            langs_list = []
            answers_list = []
        
        if self.custom_reward_func:
            # Let Ray automatically distribute the workload across available resources
            batch_size = self.args.micro_rollout_batch_size
            num_chunks = (len(queries_list) + batch_size - 1) // batch_size
            r_refs = []
            for i in range(num_chunks):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(queries_list))
                r = self.custom_reward_func.remote(
                    queries_list[start_idx:end_idx],
                    prompts_list[start_idx:end_idx],
                    labels_list[start_idx:end_idx],
                )
                r_refs.append(r)
        else:
            # Distribute data across different remote reward function servers
            num_servers = len(self.remote_rm_url)
            batch_size = (len(queries_list) + num_servers - 1) // num_servers
            r_refs = []
            for i in range(num_servers):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(queries_list))
                rm = self.remote_rm_url[i]
                
                # 准备分片数据
                queries_chunk = queries_list[start_idx:end_idx]
                prompts_chunk = prompts_list[start_idx:end_idx]
                labels_chunk = labels_list[start_idx:end_idx]
                transed_chunk = transed_prompts_list[start_idx:end_idx] if transed_prompts_list else []
                langs_chunk = langs_list[start_idx:end_idx] if langs_list else []
                answers_chunk = answers_list[start_idx:end_idx] if answers_list else []
                
                r = remote_rm_fn_ray.remote(
                    rm,
                    queries=queries_chunk,
                    prompts=prompts_chunk,
                    labels=labels_chunk,
                    transed_prompts=transed_chunk,
                    langs=langs_chunk,
                    answers=answers_chunk,
                )
                r_refs.append(r)

        return ray.get(r_refs)
