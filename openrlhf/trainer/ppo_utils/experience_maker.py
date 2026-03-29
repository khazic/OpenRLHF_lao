import itertools
import time
from datetime import timedelta
from typing import List, Tuple

import ray
import torch

from openrlhf.models.utils import compute_approx_kl, compute_reward, masked_mean
from openrlhf.trainer.ppo_utils.experience import Experience
from openrlhf.trainer.ppo_utils.length_penalty import apply_length_penalties
from openrlhf.trainer.ray.launcher import RayActorGroup
from openrlhf.utils.logging_utils import init_logger
from openrlhf.utils.seqlen_balancing import get_minimum_num_micro_batch_size, get_seqlen_balanced_partitions

logger = init_logger(__name__)


class RemoteExperienceMaker:
    def __init__(
        self,
        actor_model_group: RayActorGroup,
        critic_model_group: RayActorGroup,
        reward_model_group: RayActorGroup,
        initial_model_group: RayActorGroup,
        kl_controller,
        strategy,
        tokenizer,
        **kwargs,
    ):
        super().__init__()

        self.strategy = strategy
        self.args = strategy.args
        self.advantage_estimator = strategy.args.advantage_estimator

        self.actor_model_group = actor_model_group
        self.critic_model_group = critic_model_group
        self.reward_model_group = reward_model_group
        self.initial_model_group = initial_model_group
        self.tokenizer = tokenizer
        self.kl_ctl = kl_controller

    def split_rollout_samples(self, rollout_samples):
        for i, sample in enumerate(rollout_samples):
            sample.index = [i]

        samples_list = []
        if self.args.use_dynamic_batch:
            total_lengths = [int(s.total_length.item()) for s in rollout_samples]
            effective_actor_num = (
                self.args.actor_num_nodes
                * self.args.actor_num_gpus_per_node
                // self.args.ring_attn_size
                // self.args.ds_tensor_parallel_size
            )
            minimum_batch_num = get_minimum_num_micro_batch_size(
                total_lengths,
                self.args.rollout_max_tokens_per_gpu,
                self.args.ring_attn_size,
                self.args.ds_tensor_parallel_size,
            )
            minimum_batch_num = minimum_batch_num // effective_actor_num * effective_actor_num
            num_batch = max(minimum_batch_num, effective_actor_num)
            batch_indexes = get_seqlen_balanced_partitions(total_lengths, num_batch, False)
            for micro_index in batch_indexes:
                micro_batch = [rollout_samples[idx] for idx in micro_index]
                concat_samples = Experience.concat_experiences(micro_batch, self.tokenizer.pad_token_id)
                samples_list.append(concat_samples)
        else:
            batch_size = self.args.micro_rollout_batch_size
            for i in range(0, len(rollout_samples), batch_size):
                concat_samples = Experience.concat_experiences(
                    rollout_samples[i : i + batch_size], self.tokenizer.pad_token_id
                )
                samples_list.append(concat_samples)
        return samples_list

    @torch.no_grad()
    def make_experience_batch(self, rollout_samples) -> List[Experience]:
        """
        Make a list of experience with the micro_rollout_batch_size.

        This method will first calculate the response sequences and rewards for the given prompts.
        Then, if we need certain processing for the rewards or do certain filtering, we can process the rollout as a whole.
        After that, we will calculate the advantages and returns for each experience.
        """
        # Each batch of samples will be scheduled to a effective Ray Actor (i.e, a DP rank)
        samples_list = self.split_rollout_samples(rollout_samples)

        # Make experiences (models forward: logprobs, values, rewards, and kl divergence)
        experiences = self.make_experience(samples_list)

        # Process experiences (reward shaping, etc.)
        experiences = self.compute_advantages_and_returns(experiences)
        return experiences

    # ── Remote model dispatch helpers ──

    def _dispatch_forward(self, group, sync_condition, **kwargs):
        """Dispatch a batched forward call and optionally sync + empty cache."""
        ref = group.async_run_method_batch(method_name="forward", **kwargs)
        if sync_condition:
            ray.get(ref)
            ray.get(group.async_run_method(method_name="empty_cache"))
        return ref

    def _flatten_results(self, refs, duplicate_factor):
        """Gather ray refs and flatten results, deduplicating ring_attn/tp copies."""
        return list(itertools.chain.from_iterable(ray.get(refs)[::duplicate_factor]))

    @torch.no_grad()
    def make_experience(self, samples_list: List[Experience]) -> List[Experience]:
        """Turn samples into experience by calculating logprobs, values, rewards, and kl divergence."""
        start_time = time.time()
        logger.info(f"🚀 Starting experience making with {sum([len(s.sequences) for s in samples_list])} samples")

        args = self.strategy.args
        device = "cpu"
        duplicate_factor = args.ring_attn_size * args.ds_tensor_parallel_size
        dummy_ref = ray.put([[None]] * (len(samples_list) * duplicate_factor))

        # Extract tensors for batch processing
        sequences_list = [s.sequences for s in samples_list]
        attention_mask_list = [s.attention_mask for s in samples_list]
        action_mask_list = [s.action_mask for s in samples_list]
        forward_kwargs = dict(
            sequences=sequences_list, action_mask=action_mask_list, attention_mask=attention_mask_list
        )

        # ── Dispatch all model forward calls ──

        # Reward model
        use_reward_model = samples_list[0].rewards is None
        if use_reward_model:
            if self.reward_model_group is None:
                raise ValueError("reward_model_group is required when rewards are not precomputed")
            r_refs = self._dispatch_forward(
                self.reward_model_group,
                args.colocate_all_models,
                sequences=sequences_list,
                attention_mask=attention_mask_list,
                pad_sequence=[True] * len(samples_list),
            )
        else:
            r_refs = None

        # Actor model
        action_log_probs_ref = self._dispatch_forward(
            self.actor_model_group,
            args.colocate_all_models or args.colocate_actor_ref,
            **forward_kwargs,
        )

        # Critic model
        if self.critic_model_group is not None:
            if args.colocate_critic_reward and r_refs is not None:
                ray.get(r_refs)
                ray.get(self.reward_model_group.async_run_method(method_name="empty_cache"))

            value_ref = self._dispatch_forward(
                self.critic_model_group,
                args.colocate_all_models or args.colocate_critic_reward,
                **forward_kwargs,
            )
        else:
            value_ref = dummy_ref

        # Reference model
        if self.initial_model_group is not None:
            base_action_log_probs_ref = self._dispatch_forward(
                self.initial_model_group,
                args.colocate_all_models or args.colocate_actor_ref,
                **forward_kwargs,
            )
        else:
            base_action_log_probs_ref = dummy_ref

        # ── Gather and flatten results ──

        action_log_probs_list = self._flatten_results(action_log_probs_ref, duplicate_factor)
        base_action_log_probs_list = self._flatten_results(base_action_log_probs_ref, duplicate_factor)
        value_list = self._flatten_results(value_ref, duplicate_factor)

        if use_reward_model:
            rewards_list = self._flatten_results(r_refs, duplicate_factor)
            for i, samples in enumerate(samples_list):
                samples.rewards = rewards_list[i]
                samples.info["reward"] = rewards_list[i]

        assert (
            len(samples_list) == len(action_log_probs_list) == len(base_action_log_probs_list) == len(value_list)
        ), f"len(samples_list): {len(samples_list)}, len(action_log_probs_list): {len(action_log_probs_list)}, len(base_action_log_probs_list): {len(base_action_log_probs_list)}, len(value_list): {len(value_list)}"

        # ── Compute KL and attach results to experiences ──

        for i, (samples, action_log_probs, base_action_log_probs, value) in enumerate(
            zip(samples_list, action_log_probs_list, base_action_log_probs_list, value_list)
        ):
            if (self.initial_model_group is not None) and (not args.use_kl_loss):
                kl = compute_approx_kl(
                    action_log_probs,
                    base_action_log_probs,
                    kl_estimator=self.strategy.args.kl_estimator,
                )
                logprobs_diff = action_log_probs.float() - base_action_log_probs.float()
            else:
                kl = torch.zeros_like(action_log_probs, dtype=action_log_probs.dtype, device=device)
                logprobs_diff = torch.zeros_like(action_log_probs, dtype=action_log_probs.dtype, device=device)
            kl_mean = masked_mean(kl, samples.action_mask, dim=-1)
            logprobs_diff_mean = masked_mean(logprobs_diff, samples.action_mask, dim=-1)

            if not args.use_kl_loss:
                base_action_log_probs = None

            # Update experience with new information
            samples.action_log_probs = action_log_probs
            samples.base_action_log_probs = base_action_log_probs
            samples.values = value
            samples.kl = kl
            samples.info["kl"] = kl_mean
            samples.info["logprobs_diff"] = logprobs_diff_mean

        end_time = time.time()
        duration = end_time - start_time
        time_str = str(timedelta(seconds=duration)).split(".")[0]
        logger.info(f"✨ Experience making completed in {time_str}")
        return samples_list

    # ── Advantage and return computation ──

    @torch.no_grad()
    def compute_advantages_and_returns(self, experiences: List[Experience]) -> List[Experience]:
        """Compute shaped rewards, advantages, and returns for a batch of experiences.

        Pipeline:
        1. Apply length penalties (DAPO / ProRL)
        2. Reward shaping (RLOO / GRPO baseline subtraction)
        3. Advantage calculation (GAE / REINFORCE) + optional normalization
        """
        args = self.strategy.args

        # Step 1: Apply length penalties (DAPO overlong / ProRL stop properly)
        apply_length_penalties(experiences, args)

        # Step 2: Reward shaping
        rewards = self._apply_reward_shaping(experiences, args)

        # Step 3: Compute advantages and returns
        self._compute_per_experience_advantages(experiences, rewards, args)

        # Step 4: Normalize advantages across all experiences (for applicable estimators)
        self._normalize_advantages(experiences, args)

        return experiences

    def _apply_reward_shaping(self, experiences: List[Experience], args) -> List[torch.Tensor]:
        """Apply reward baselines (RLOO, GRPO, group_norm) and return per-experience reward tensors."""
        exp_len = [len(experience.index) for experience in experiences]
        # indices is an identity mapping when not using dynamic batch;
        # otherwise, it maps back to the original indices after rearranging samples
        indices = torch.tensor(list(itertools.chain.from_iterable(experience.index for experience in experiences)))
        raw_rewards = torch.cat([experience.rewards for experience in experiences], dim=0)
        rewards = torch.empty_like(raw_rewards)
        rewards[indices] = raw_rewards  # sorted

        rewards = rewards.reshape(-1, args.n_samples_per_prompt)

        # Log group reward std
        if args.n_samples_per_prompt > 1:
            group_reward_stds = (
                rewards.std(-1, keepdim=True).repeat(1, args.n_samples_per_prompt).reshape(-1)[indices].split(exp_len)
            )
            for experience, group_reward_std in zip(experiences, group_reward_stds):
                experience.info["group_reward_std"] = group_reward_std

        # Baseline subtraction
        if args.advantage_estimator == "rloo":
            baseline = (rewards.sum(-1, keepdim=True) - rewards) / (args.n_samples_per_prompt - 1)
            rewards = rewards - baseline
        elif args.advantage_estimator in ["reinforce_baseline", "dr_grpo"]:
            rewards = rewards - rewards.mean(-1, keepdim=True)
        elif args.advantage_estimator == "group_norm":
            rewards = (rewards - rewards.mean(-1, keepdim=True)) / (rewards.std(-1, keepdim=True) + 1e-9)

        return rewards.reshape(-1)[indices].split(exp_len)

    def _compute_per_experience_advantages(
        self, experiences: List[Experience], rewards: List[torch.Tensor], args
    ) -> None:
        """Compute advantages and returns for each experience in-place."""
        for experience, reward in zip(experiences, rewards):
            reward = compute_reward(
                reward,
                self.kl_ctl.value,
                experience.kl,
                action_mask=experience.action_mask,
                reward_clip_range=args.reward_clip_range,
            )

            if self.advantage_estimator == "gae":
                experience.advantages, experience.returns = self.get_advantages_and_returns(
                    experience.values,
                    reward,
                    experience.action_mask,
                    args.gamma,
                    args.lambd,
                )
            elif self.advantage_estimator in ["reinforce", "rloo", "reinforce_baseline", "group_norm", "dr_grpo"]:
                if args.gamma != 1.0 and self.advantage_estimator in [
                    "rloo",
                    "reinforce_baseline",
                    "group_norm",
                    "dr_grpo",
                ]:
                    logger.warning("gamma is set to 1.0 for rloo, reinforce_baseline, and group_norm")
                    args.gamma = 1.0

                experience.returns = self.get_cumulative_returns(
                    reward,
                    experience.action_mask,
                    args.gamma,
                )
                experience.advantages = experience.returns.clone()
            else:
                raise Exception(f"Unkown advantage_estimator {self.advantage_estimator}")

            # Calculate the return info
            return_sums = reward.sum(dim=-1)
            experience.info["return"] = return_sums
            # Remove unnecessary info
            experience.kl = None

    def _normalize_advantages(self, experiences: List[Experience], args) -> None:
        """Normalize advantages across all experiences for applicable estimators."""
        if args.advantage_estimator not in ["gae", "reinforce", "reinforce_baseline"]:
            return

        all_advantages = []
        all_action_masks = []
        for exp in experiences:
            all_advantages.append(exp.advantages.flatten())
            all_action_masks.append(exp.action_mask.flatten())

        advantages_vector = torch.cat(all_advantages, dim=0).float()
        action_masks_vector = torch.cat(all_action_masks, dim=0)
        num_actions = action_masks_vector.sum()

        mean = (advantages_vector * action_masks_vector).sum() / num_actions
        if not args.no_advantage_std_norm:
            var = ((advantages_vector - mean).pow(2) * action_masks_vector).sum() / num_actions
            rstd = var.clamp(min=1e-8).rsqrt()
        else:
            rstd = 1

        for exp in experiences:
            exp.advantages = (exp.advantages - mean) * rstd

    @torch.no_grad()
    def get_advantages_and_returns(
        self,
        values: torch.Tensor,
        rewards: torch.Tensor,
        action_mask: torch.Tensor,
        gamma: float,
        lambd: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Function that computes advantages and returns from rewards and values.
        Calculated as in the original PPO paper: https://arxiv.org/abs/1707.06347
        Note that rewards may include a KL divergence loss term.

        Advantages looks like this:
        Adv1 =  R1 + γ * λ * R2     + γ^2 * λ^2 * R3       + ...
              - V1 + γ * (1 - λ) V2 + γ^2 * λ * (1 - λ) V3 + ...

        Returns looks like this:
        Ret1 =  R1 + γ * λ * R2     + γ^2 * λ^2 * R3       + ...
                   + γ * (1 - λ) V2 + γ^2 * λ * (1 - λ) V3 + ...

        Input:
        - values: Tensor of shape (batch_size, response_size)
        - rewards: Tensor of shape (batch_size, response_size)

        Output:
        - advantages: Tensor of shape (batch_size, response_size)
        - returns: Tensor of shape (batch_size, response_size)
        """
        lastgaelam = 0
        advantages_reversed = []
        response_length = rewards.size(1)

        # Mask invalid responses
        if action_mask is not None:
            values = action_mask * values
            rewards = action_mask * rewards

        for t in reversed(range(response_length)):
            nextvalues = values[:, t + 1] if t < response_length - 1 else 0.0
            delta = rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam = delta + gamma * lambd * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        returns = advantages + values
        return advantages.detach(), returns

    @torch.no_grad()
    def get_cumulative_returns(
        self,
        rewards: torch.Tensor,
        action_mask: torch.Tensor,
        gamma: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Function that computes advantages and returns from rewards using REINFORCE.
        REINFORCE uses cumulative returns without the GAE (Generalized Advantage Estimation).

        Input:
        - rewards: Tensor of shape (batch_size, response_size)
        - action_mask: Tensor of shape (batch_size, response_size), binary mask
        - gamma: discount factor

        Output:
        - returns: Tensor of shape (batch_size, response_size)
        """
        response_length = rewards.size(1)
        returns = torch.zeros_like(rewards)
        cumulative_return = torch.zeros(rewards.size(0), device=rewards.device)

        # Mask invalid responses if action_mask is provided
        if action_mask is not None:
            rewards = action_mask * rewards

        # Calculate returns by accumulating discounted rewards
        for t in reversed(range(response_length)):
            cumulative_return = rewards[:, t] + gamma * cumulative_return
            returns[:, t] = cumulative_return

        return returns
