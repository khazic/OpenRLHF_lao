import os
import time
from abc import ABC
from datetime import timedelta

import ray
import torch
from tqdm import tqdm

from openrlhf.datasets import PromptDataset
from openrlhf.datasets.utils import blending_datasets
from openrlhf.trainer.ppo_utils import AdaptiveKLController, FixedKLController
from openrlhf.trainer.ppo_utils.experience_maker import RemoteExperienceMaker
from openrlhf.trainer.ppo_utils.replay_buffer import balance_experiences
from openrlhf.trainer.ray.launcher import RayActorGroup
from openrlhf.utils.deepspeed import DeepspeedStrategy
from openrlhf.utils.logging_utils import init_logger
from openrlhf.utils.utils import get_tokenizer

logger = init_logger(__name__)


class BasePPOTrainer(ABC):
    def __init__(
        self,
        pretrain: str,
        strategy: DeepspeedStrategy,
        actor_model_group: RayActorGroup,
        critic_model_group: RayActorGroup,
        reward_model_group: RayActorGroup,
        reference_model_group: RayActorGroup,
        vllm_engines=None,
        prompt_max_len: int = 120,
        dataloader_pin_memory: bool = True,
        prompt_split: str = "train",
        eval_split: str = "test",
        **generate_kwargs,
    ) -> None:
        super().__init__()

        self.strategy = strategy
        self.args = strategy.args

        self.tokenizer = get_tokenizer(pretrain, None, "left", strategy, use_fast=not self.args.disable_fast_tokenizer)
        self.actor_model_group = actor_model_group
        self.critic_model_group = critic_model_group
        self.reward_model_group = reward_model_group
        self.reference_model_group = reference_model_group
        self.dataloader_pin_memory = dataloader_pin_memory
        self.vllm_engines = vllm_engines

        self.prompt_split = prompt_split
        self.eval_split = eval_split

        self.prompt_max_len = prompt_max_len
        self.generate_kwargs = generate_kwargs

        self.max_epochs = self.args.max_epochs
        self.remote_rm_url = self.args.remote_rm_url
        self.init_kl_coef = self.args.init_kl_coef
        self.kl_target = self.args.kl_target
        self.kl_horizon = self.args.kl_horizon

        self.freezing_actor_steps = getattr(self.args, "freezing_actor_steps", -1)

        # Init dummy variables
        self.prompts_dataloader = None
        self.eval_dataloader = None
        self.max_steps = None

        self.samples_generator = None
        self.experience_maker = None
        self.remote_reward_model = None

        if self.args.agent_func_path:
            from openrlhf.trainer.ppo_utils.experience_maker_async import SamplesGeneratorAsync as SamplesGenerator
        else:
            from openrlhf.trainer.ppo_utils.experience_maker import SamplesGenerator

        self.generator_cls = SamplesGenerator

    def _init_wandb(self):
        # wandb/tensorboard setting
        self._wandb = None
        self._tensorboard = None
        self.generated_samples_table = None
        if self.strategy.args.use_wandb:
            import wandb

            self._wandb = wandb
            if not wandb.api.api_key:
                wandb.login(key=self.strategy.args.use_wandb)
            wandb.init(
                entity=self.strategy.args.wandb_org,
                project=self.strategy.args.wandb_project,
                group=self.strategy.args.wandb_group,
                name=self.strategy.args.wandb_run_name,
                config=self.strategy.args.__dict__,
                reinit=True,
            )

            wandb.define_metric("train/global_step")
            wandb.define_metric("train/*", step_metric="train/global_step", step_sync=True)
            wandb.define_metric("eval/epoch")
            wandb.define_metric("eval/*", step_metric="eval/epoch", step_sync=True)
            self.generated_samples_table = wandb.Table(columns=["global_step", "text", "reward"])

        # Initialize TensorBoard writer if wandb is not available
        if self.strategy.args.use_tensorboard and self._wandb is None:
            from torch.utils.tensorboard import SummaryWriter

            os.makedirs(self.strategy.args.use_tensorboard, exist_ok=True)
            log_dir = os.path.join(self.strategy.args.use_tensorboard, self.strategy.args.wandb_run_name)
            self._tensorboard = SummaryWriter(log_dir=log_dir)

    def fit(self):
        raise NotImplementedError("fit method is not implemented")

    def ppo_train(self, global_steps):
        status = {}

        # triger remote critic model training
        if self.critic_model_group is not None:
            # sync for deepspeed_enable_sleep
            if self.strategy.args.deepspeed_enable_sleep:
                ray.get(self.critic_model_group.async_run_method(method_name="reload_states"))

            critic_status_ref = self.critic_model_group.async_run_method(method_name="fit")

            if self.strategy.args.colocate_all_models or self.strategy.args.deepspeed_enable_sleep:
                status.update(ray.get(critic_status_ref)[0])
            if self.strategy.args.deepspeed_enable_sleep:
                ray.get(self.critic_model_group.async_run_method(method_name="offload_states"))

        # actor model training
        if global_steps > self.freezing_actor_steps:
            if self.strategy.args.deepspeed_enable_sleep:
                ray.get(self.actor_model_group.async_run_method(method_name="reload_states"))

            actor_status_ref = self.actor_model_group.async_run_method(method_name="fit", kl_ctl=self.kl_ctl.value)
            status.update(ray.get(actor_status_ref)[0])

            if self.strategy.args.deepspeed_enable_sleep:
                ray.get(self.actor_model_group.async_run_method(method_name="offload_states"))

            # 4. broadcast weights to vllm engines
            if self.vllm_engines is not None:
                self._broadcast_to_vllm()

        # 5. wait remote critic model training done
        if self.critic_model_group and not self.strategy.args.colocate_all_models:
            status.update(ray.get(critic_status_ref)[0])

        return status

    def _broadcast_to_vllm(self):
        if self.strategy.args.vllm_enable_sleep:
            from openrlhf.trainer.ray.vllm_engine import batch_vllm_engine_call

            batch_vllm_engine_call(self.vllm_engines, "wake_up")

        ray.get(self.actor_model_group.async_run_method(method_name="broadcast_to_vllm"))

        if self.strategy.args.vllm_enable_sleep:
            batch_vllm_engine_call(self.vllm_engines, "sleep")

    def _prepare_train_logs(self, logs_dict: dict) -> dict:
        """Normalize and regroup training metrics before logging."""
        alias_map = {
            "policy_loss": "actor/policy_loss",
            "ppl": "actor/ppl",
            "grad_norm": "actor/grad_norm",
            "actor_lr": "optimizer/actor_lr",
            "entropy_loss": "actor/entropy_loss",
            "token_entropy": "actor/token_entropy",
            "sequence_entropy": "actor/sequence_entropy",
            "policy_entropy": "actor/policy_entropy",
            "entropy_var": "actor/entropy_variance",
            "entropy_var_loss": "actor/entropy_var_loss",
            "kl_coefficient": "actor/kl_coef",
            "ppo_clip_ratio": "actor/clip_fraction",
            "ppo_kl": "actor/approx_kl",
            "vllm_kl": "actor/vllm_kl",
            "critic_loss": "critic/loss",
            "critic_total_loss": "critic/total_loss",
            "critic_aux_loss": "critic/aux_loss",
            "critic_value_mean": "critic/value_mean",
            "critic_value_std": "critic/value_std",
            "critic_target_mean": "critic/target_mean",
            "critic_target_std": "critic/target_std",
            "critic_lr": "optimizer/critic_lr",
            "reward": "rollout/reward",
            "reward_mean": "rollout/reward_mean",
            "reward_std": "rollout/reward_std",
            "return": "rollout/return_sum",
            "return_mean": "rollout/return_mean",
            "return_std": "rollout/return_std",
            "return_sum_mean": "rollout/return_sum_mean",
            "return_sum_std": "rollout/return_sum_std",
            "advantage_mean": "rollout/advantage_mean",
            "advantage_std": "rollout/advantage_std",
            "value_baseline_mean": "rollout/value_baseline_mean",
            "value_baseline_std": "rollout/value_baseline_std",
            "ratio_mean": "rollout/ratio_mean",
            "kl": "rollout/kl_mean",
            "tis_ratio": "rollout/truncated_is_ratio",
            "response_length": "rollout/response_length",
            "total_length": "rollout/total_length",
            "response_clip_ratio": "rollout/response_truncation_rate",
            "dynamic_filtering_pass_rate": "rollout/dynamic_filter_pass_rate",
        }

        processed_logs = {}
        for key, value in logs_dict.items():
            alias = alias_map.get(key, key)
            metric_value = value
            if isinstance(metric_value, torch.Tensor):
                metric_value = metric_value.detach()
                metric_value = metric_value.mean().item() if metric_value.numel() > 1 else metric_value.item()
            processed_logs[alias] = metric_value
        return processed_logs

    def save_logs_and_checkpoints(self, args, global_step, step_bar, logs_dict={}, client_states={}):
        logs_dict = dict(logs_dict)
        if global_step % args.logging_steps == 0:
            generated_samples = logs_dict.pop("generated_samples", None)
            processed_logs = self._prepare_train_logs(logs_dict)

            # wandb
            if self._wandb is not None:
                # Add generated samples to wandb using Table
                if generated_samples is not None:
                    # https://github.com/wandb/wandb/issues/2981#issuecomment-1997445737
                    new_table = self._wandb.Table(
                        columns=self.generated_samples_table.columns, data=self.generated_samples_table.data
                    )
                    new_table.add_data(global_step, *generated_samples)
                    self.generated_samples_table = new_table
                    self._wandb.log({"train/generated_samples": new_table})
                logs = {
                    "train/%s" % k: v
                    for k, v in {
                        **processed_logs,
                        "global_step": global_step,
                    }.items()
                }
                self._wandb.log(logs)
            # TensorBoard
            elif self._tensorboard is not None:
                if generated_samples is not None:
                    text, reward = generated_samples
                    formatted_text = f"Sample:\n{text}\n\nReward: {reward:.4f}"
                    self._tensorboard.add_text("train/generated_samples", formatted_text, global_step)

                for k, v in processed_logs.items():
                    self._tensorboard.add_scalar(f"train/{k}", v, global_step)

        # TODO: Add evaluation mechanism for PPO
        if global_step % args.eval_steps == 0 and self.eval_dataloader and len(self.eval_dataloader) > 0:
            self.evaluate(self.eval_dataloader, global_step, args.eval_temperature, args.eval_n_samples_per_prompt)
        # save ckpt
        # TODO: save best model on dev, use loss/perplexity/others on whole dev dataset as metric
        if global_step % args.save_steps == 0:
            tag = f"global_step{global_step}"
            ref = self.actor_model_group.async_run_method(
                method_name="save_checkpoint", tag=tag, client_states=client_states
            )
            if self.critic_model_group is not None:
                ref.extend(self.critic_model_group.async_run_method(method_name="save_checkpoint", tag=tag))
            ray.get(ref)

    def evaluate(self, eval_dataloader, global_step, temperature=0.6, n_samples_per_prompt=1):
        """Evaluate model performance on eval dataset.

        Args:
            eval_dataloader: DataLoader containing evaluation prompts, labels and data sources
            global_step: Current training step for logging
            n_samples_per_prompt: Number of samples to generate per prompt for pass@k calculation
        """
        start_time = time.time()
        logger.info(f"⏰ Evaluation start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

        # vLLM wakeup when vllm_enable_sleep
        if self.strategy.args.vllm_enable_sleep:
            from openrlhf.trainer.ray.vllm_engine import batch_vllm_engine_call

            batch_vllm_engine_call(self.vllm_engines, "wake_up")

        with torch.no_grad():
            # First collect all prompts and labels
            all_prompts = []
            all_labels = []
            all_metadata = []
            prompt_to_datasource = {}  # Dictionary to store mapping between prompts and their data sources

            for batch_data in eval_dataloader:
                if len(batch_data) == 4:
                    datasources, prompts, labels, metadata = batch_data
                    all_metadata.extend(metadata)
                else:
                    datasources, prompts, labels = batch_data
                all_prompts.extend(prompts)
                all_labels.extend(labels)
                # Create mapping for each prompt to its corresponding data source
                for prompt, datasource in zip(prompts, datasources):
                    prompt_to_datasource[prompt] = datasource

            # Generate samples and calculate rewards
            generate_kwargs = self.generate_kwargs.copy()
            generate_kwargs["temperature"] = temperature
            generate_kwargs["n_samples_per_prompt"] = n_samples_per_prompt
            samples_list = self.samples_generator.generate_samples(
                all_prompts, all_labels, metadata=all_metadata if all_metadata else None, 
                remote_reward_model=self.remote_reward_model, **generate_kwargs
            )

            # duplicate prompts and labels for each sample
            all_prompts = sum([s.prompts for s in samples_list], [])
            all_labels = sum([s.labels for s in samples_list], [])

            # Get rewards from samples, such as agent rewards or remote reward models
            rewards_list = []
            for samples in samples_list:
                rewards_list.append(samples.rewards)
            # Reshape rewards to (num_prompts, n_samples_per_prompt)
            rewards = torch.tensor(rewards_list).reshape(-1, n_samples_per_prompt)

            # Collect local statistics for each data source
            global_metrics = {}  # {datasource: {"pass{n_samples_per_prompt}": 0, "pass1": 0, "count": 0}}

            # Process rewards in chunks of n_samples_per_prompt
            num_prompts = len(all_prompts) // n_samples_per_prompt
            for i in range(num_prompts):
                # Get the original prompt (first one in the chunk)
                original_prompt = all_prompts[i * n_samples_per_prompt]
                datasource = prompt_to_datasource[original_prompt]  # Get corresponding data source using the mapping

                if datasource not in global_metrics:
                    global_metrics[datasource] = {f"pass{n_samples_per_prompt}": 0, "pass1": 0, "count": 0}

                # Get rewards for this chunk
                chunk_rewards = rewards[i]

                # Calculate pass@k and pass@1
                if n_samples_per_prompt > 1:
                    global_metrics[datasource][f"pass{n_samples_per_prompt}"] += chunk_rewards.max().float().item()
                global_metrics[datasource]["pass1"] += chunk_rewards.mean().float().item()
                global_metrics[datasource]["count"] += 1

            # Calculate global averages
            logs = {}
            for datasource, metrics in global_metrics.items():
                logs[f"eval_{datasource}_pass{n_samples_per_prompt}"] = (
                    metrics[f"pass{n_samples_per_prompt}"] / metrics["count"]
                )
                logs[f"eval_{datasource}_pass1"] = metrics["pass1"] / metrics["count"]

            # Log to wandb/tensorboard
            if self._wandb is not None:
                logs = {"eval/%s" % k: v for k, v in {**logs, "global_step": global_step}.items()}
                self._wandb.log(logs)
            elif self._tensorboard is not None:
                for k, v in logs.items():
                    self._tensorboard.add_scalar(f"eval/{k}", v, global_step)

        if self.strategy.args.vllm_enable_sleep:
            batch_vllm_engine_call(self.vllm_engines, "sleep")

        end_time = time.time()
        duration = end_time - start_time
        time_str = str(timedelta(seconds=duration)).split(".")[0]
        logger.info(f"✨ Evaluation completed in {time_str}, global_step {global_step}, eval_metrics: {logs}")

    def prepare_datasets(self):
        args = self.args
        strategy = self.strategy

        # prepare datasets
        train_data = blending_datasets(
            args.prompt_data,
            args.prompt_data_probs,
            strategy,
            args.seed,
            max_count=args.max_samples,
            dataset_split=self.prompt_split,
        )

        # Create train dataset
        train_data = train_data.select(range(min(args.max_samples, len(train_data))))
        prompts_dataset = PromptDataset(train_data, self.tokenizer, strategy, input_template=args.input_template)
        prompts_dataloader = strategy.setup_dataloader(
            prompts_dataset,
            args.vllm_generate_batch_size,
            True,
            True,
        )

        # Create eval dataset if eval data exists
        if getattr(args, "eval_dataset", None):
            eval_data = blending_datasets(
                args.eval_dataset,
                None,  # No probability sampling for eval datasets
                strategy,
                dataset_split=self.eval_split,
            )
            eval_data = eval_data.select(range(min(args.max_samples, len(eval_data))))
            eval_dataset = PromptDataset(eval_data, self.tokenizer, strategy, input_template=args.input_template)
            eval_dataloader = strategy.setup_dataloader(eval_dataset, 1, True, False)
        else:
            eval_dataloader = None

        self.prompts_dataloader = prompts_dataloader
        self.eval_dataloader = eval_dataloader
        self.max_steps = (
            len(prompts_dataset)
            * args.n_samples_per_prompt
            // args.train_batch_size
            * args.num_episodes
            * args.max_epochs
        )

    def get_max_steps(self):
        return self.max_steps


@ray.remote
class PPOTrainer(BasePPOTrainer):
    """
    Trainer for Proximal Policy Optimization (PPO) / REINFORCE++ / GRPO / RLOO / xPO and their variants.
    Single Controller with Multiple ActorGroups
    """

    def __init__(
        self,
        pretrain: str,
        strategy: DeepspeedStrategy,
        actor_model_group: RayActorGroup,
        critic_model_group: RayActorGroup,
        reward_model_group: RayActorGroup,
        reference_model_group: RayActorGroup,
        vllm_engines=None,
        prompt_max_len: int = 120,
        dataloader_pin_memory: bool = True,
        prompt_split: str = "train",
        eval_split: str = "test",
        **generate_kwargs,
    ) -> None:
        super().__init__(
            pretrain,
            strategy,
            actor_model_group,
            critic_model_group,
            reward_model_group,
            reference_model_group,
            vllm_engines,
            prompt_max_len,
            dataloader_pin_memory,
            prompt_split,
            eval_split,
            **generate_kwargs,
        )

        if self.kl_target:
            self.kl_ctl = AdaptiveKLController(self.init_kl_coef, self.kl_target, self.kl_horizon)
        else:
            self.kl_ctl = FixedKLController(self.init_kl_coef)

        if self.args.remote_rm_url and not self.args.remote_rm_url[0] == "agent":
            from openrlhf.utils.remote_rm_utils import RemoteRewardModel

            self.remote_reward_model = RemoteRewardModel.remote(self.args, self.remote_rm_url)

        self.samples_generator = self.generator_cls(
            self.vllm_engines,
            self.strategy,
            self.tokenizer,
            self.prompt_max_len,
        )

        self.experience_maker = RemoteExperienceMaker(
            self.actor_model_group,
            self.critic_model_group,
            self.reward_model_group,
            self.reference_model_group,
            self.kl_ctl,
            self.strategy,
            self.tokenizer,
            remote_reward_model=self.remote_reward_model,
        )

        self.prepare_datasets()
        self._init_wandb()

        # get eval and save steps
        if self.args.eval_steps == -1:
            self.args.eval_steps = float("inf")  # do not evaluate
        if self.args.save_steps == -1:
            self.args.save_steps = float("inf")  # do not save ckpt

    def fit(
        self,
    ) -> None:
        args = self.args

        # broadcast init checkpoint to vllm
        ckpt_path = os.path.join(args.ckpt_path, "_actor")
        if args.load_checkpoint and os.path.exists(ckpt_path):
            checkpoint_states = ray.get(self.actor_model_group.async_run_method(method_name="get_checkpoint_states"))[
                0
            ]
            logger.info(f"checkpoint_states: {checkpoint_states}")
            self._broadcast_to_vllm()
        else:
            checkpoint_states = {"global_step": 0, "episode": 0, "data_loader_state_dict": {}}

        # Restore step and start_epoch
        steps = checkpoint_states["global_step"] + 1
        episode = checkpoint_states["episode"]
        data_loader_state_dict = checkpoint_states["data_loader_state_dict"]
        if data_loader_state_dict:
            self.prompts_dataloader.load_state_dict(data_loader_state_dict)

        for episode in range(episode, args.num_episodes):
            pbar = tqdm(
                range(self.prompts_dataloader.__len__()),
                desc=f"Episode [{episode + 1}/{args.num_episodes}]",
                disable=False,
                initial=steps,
            )

            filtered_samples = []
            number_of_samples = 0
            for batch_data in self.prompts_dataloader:
                if len(batch_data) == 4:
                    _, rand_prompts, labels, metadata = batch_data
                else:
                    _, rand_prompts, labels = batch_data
                    metadata = None
                remote_reward_model = self.remote_reward_model if self.args.dynamic_filtering else None
                rollout_samples = self.samples_generator.generate_samples(
                    rand_prompts, labels, metadata=metadata, remote_reward_model=remote_reward_model, **self.generate_kwargs
                )
                pbar.update()

                # dynamic filtering
                pass_rate = None
                if self.args.dynamic_filtering:
                    number_of_samples += len(rollout_samples)
                    # Group individual samples into batches of n_samples size
                    for i in range(0, len(rollout_samples), self.args.n_samples_per_prompt):
                        batch_samples = rollout_samples[i : i + self.args.n_samples_per_prompt]
                        if len(batch_samples) < self.args.n_samples_per_prompt:
                            continue

                        # Calculate average reward for this batch of samples
                        avg_reward = sum(sample.scores[0].item() for sample in batch_samples) / len(batch_samples)

                        # Check if average reward is within the specified range
                        min_reward, max_reward = self.args.dynamic_filtering_reward_range
                        if min_reward + 1e-6 < avg_reward < max_reward - 1e-6:
                            filtered_samples.extend(batch_samples)

                    # Continue sampling if filtered samples are insufficient
                    if len(filtered_samples) / self.args.n_samples_per_prompt < self.args.rollout_batch_size:
                        logger.info(
                            f"filtered_samples {len(filtered_samples) / self.args.n_samples_per_prompt} < rollout_batch_size {self.args.rollout_batch_size}, continue sampling"
                        )
                        continue

                    pass_rate = len(filtered_samples) / number_of_samples * 100
                    logger.info(
                        f"Dynamic filtering pass rate: {pass_rate:.2f}% ({len(filtered_samples)}/{number_of_samples})"
                    )
                    rollout_samples = filtered_samples[: self.args.rollout_batch_size * self.args.n_samples_per_prompt]
                    filtered_samples = []
                    number_of_samples = 0

                experiences = self.experience_maker.make_experience_batch(rollout_samples)
                sample0 = self.tokenizer.batch_decode(
                    experiences[0].sequences[0].unsqueeze(0), skip_special_tokens=True
                )
                
                # 打印训练批次的样本和评分统计
                print(f"\n{'='*100}")
                print(f"🎓 Training Step {steps} - Batch Summary")
                print(f"{'='*100}")
                
                # 计算并打印评分统计
                all_rewards = []
                for exp in experiences:
                    if "reward" in exp.info:
                        all_rewards.extend(exp.info["reward"].tolist())
                
                if all_rewards:
                    avg_reward = sum(all_rewards) / len(all_rewards)
                    min_reward = min(all_rewards)
                    max_reward = max(all_rewards)
                    print(f"📊 Rewards: avg={avg_reward:.3f}, min={min_reward:.3f}, max={max_reward:.3f}, count={len(all_rewards)}")
                
                # 打印前3个样本的详细信息
                print(f"\n🔍 Sample Details (showing first 3):")
                for i, exp in enumerate(experiences[:3]):
                    decoded = self.tokenizer.batch_decode(exp.sequences, skip_special_tokens=True)
                    reward = exp.info.get("reward", [0])[0]
                    reward_val = reward.item() if hasattr(reward, 'item') else reward
                    
                    print(f"\n  Sample {i+1}:")
                    print(f"    Text: {decoded[0][:150]}..." if len(decoded[0]) > 150 else f"    Text: {decoded[0]}")
                    print(f"    💯 Reward: {reward_val:.3f}")
                
                print(f"{'='*100}\n")

                # balance experiences across dp
                if args.use_dynamic_batch:
                    experiences = balance_experiences(experiences, args)

                refs = self.actor_model_group.async_run_method_batch(method_name="append", experience=experiences)
                if self.critic_model_group is not None:
                    refs.extend(
                        self.critic_model_group.async_run_method_batch(method_name="append", experience=experiences)
                    )
                ray.get(refs)

                status = self.ppo_train(steps)

                if "kl" in status:
                    self.kl_ctl.update(status["kl"], args.rollout_batch_size * args.n_samples_per_prompt)

                # Add generated samples to status dictionary
                if self.args.dynamic_filtering:
                    status["dynamic_filtering_pass_rate"] = pass_rate
                logger.info(f"✨ Global step {steps}: {status}")
                status["generated_samples"] = [sample0[0], experiences[0].info["reward"][0]]

                # logs/checkpoints
                client_states = {
                    "global_step": steps,
                    "episode": episode,
                    "data_loader_state_dict": self.prompts_dataloader.state_dict(),
                }
                self.save_logs_and_checkpoints(args, steps, pbar, status, client_states)

                steps = steps + 1

        if self._wandb is not None:
            self._wandb.finish()
        if self._tensorboard is not None:
            self._tensorboard.close()
