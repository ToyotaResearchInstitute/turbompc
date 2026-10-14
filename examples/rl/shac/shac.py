"""Short-Horizon Actor-Critic for differentiable Torch environments."""

import math
from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import torch
from torch.distributions import Normal

from ..env.runtime import Runtime
from ..util.training import Metrics, explained_variance, zero_nonfinite_gradient
from .returns import (
    DiscountedReturnNormalizer,
    SHACRollout,
    _clip_grad_norm_stable,
    _ExpandedStandardDeviation,
    td_lambda_targets,
)


@dataclass(frozen=True)
class SHACConfig:
    seed: int = 0
    num_envs: int = 64
    training_horizon: int = 32
    total_updates: int = 100
    gamma: float = 0.99
    td_lambda: float = 0.95
    actor_learning_rate: float = 5e-4
    critic_learning_rate: float = 1e-3
    anneal_lr: bool = True
    final_learning_rate: float = 5e-4
    critic_final_learning_rate: float | None = None
    critic_iterations: int = 16
    critic_minibatches: int = 4
    target_critic_alpha: float = 0.995
    adam_betas: tuple = (0.7, 0.99)
    max_gradient_norm: float = 1.0
    reward_scale: float = 1.0
    normalize_returns: bool = True
    reward_clip: float = 10.0
    shuffle_critic: bool = False

    def __post_init__(self):
        if (
            min(
                self.num_envs,
                self.training_horizon,
                self.total_updates,
                self.critic_iterations,
                self.critic_minibatches,
            )
            < 1
        ):
            raise ValueError("SHAC rollout and update sizes must be positive")
        if not all(
            0 <= value <= 1
            for value in (self.gamma, self.td_lambda, self.target_critic_alpha)
        ):
            raise ValueError(
                "discount, lambda, and target coefficient must lie in [0, 1]"
            )
        if (
            min(
                self.actor_learning_rate,
                self.critic_learning_rate,
                self.max_gradient_norm,
            )
            <= 0
        ):
            raise ValueError("learning rates and gradient limit must be positive")


class SHAC:
    """Train policy parameters and Gaussian exploration with SHAC."""

    evaluation_seed_offset = 1_000_000
    backward_scale = 2.0**-64

    def __init__(self, runtime: Runtime, actor, critic, config: SHACConfig):
        self.runtime, self.actor, self.critic = runtime, actor, critic
        self.hyperparameters = hp = config
        self.device, self.dtype = runtime.device, runtime.dtype
        self.target_critic = deepcopy(critic).requires_grad_(False)
        self.actor_optimizer = torch.optim.Adam(
            actor.parameters(), lr=hp.actor_learning_rate, betas=hp.adam_betas
        )
        self.critic_optimizer = torch.optim.Adam(
            critic.parameters(), lr=hp.critic_learning_rate, betas=hp.adam_betas
        )
        self.normalizer = (
            DiscountedReturnNormalizer(
                hp.num_envs,
                hp.gamma,
                hp.reward_clip,
                device=self.device,
                dtype=self.dtype,
            )
            if hp.normalize_returns
            else None
        )
        self.update_index = 0
        self.reset_training_state()

    def reset_training_state(self, *, update_statistics=False):
        """Reset episodes while preserving networks and normalizers."""
        hp = self.hyperparameters
        self._reset_counter = 1
        self.state = self.runtime.reset(hp.num_envs, hp.seed)
        self.episode_returns = torch.zeros(
            hp.num_envs, device=self.device, dtype=self.dtype
        )
        self.episode_lengths = torch.zeros(
            hp.num_envs, device=self.device, dtype=torch.int64
        )
        if self.normalizer is not None:
            self.normalizer.discounted = torch.zeros_like(self.episode_returns)

    def policy_parameters(self, observation):
        return self.actor(observation)

    @staticmethod
    def _guard(value):
        if value.requires_grad:
            value.register_hook(zero_nonfinite_gradient)
        return value

    def _observation(self, state):
        return self._guard(
            self.runtime.observe(state).to(device=self.device, dtype=self.dtype)
        )

    def collect(self):
        hp = self.hyperparameters
        self.actor.train()
        self.target_critic.eval()
        self.state = self.runtime.detach(self.state)
        observation = self._observation(self.state)
        frozen_scale = (
            self.normalizer.begin_segment(observation)
            if self.normalizer
            else observation.new_ones(())
        )
        observations, rewards, dones, next_values, raw_rewards = [], [], [], [], []
        completed_returns, completed_lengths = [], []
        metrics = Metrics()
        valid_count = observation.new_zeros((), dtype=torch.int64)
        actor_loss = observation.new_zeros(())
        segment_return = observation.new_zeros(hp.num_envs)
        segment_discount = observation.new_ones(hp.num_envs)
        state = self.state
        for index in range(hp.training_horizon):
            observations.append(observation.detach())
            parameters = self.actor(observation)
            policy = self.runtime.policy(parameters, state, retain=True)
            mean = self._guard(policy.mean.to(dtype=self.dtype))
            standard_deviation = self._guard(
                _ExpandedStandardDeviation.apply(self.actor.log_std, mean)
            )
            latent = Normal(mean, standard_deviation, validate_args=False).rsample()
            action = self._guard(self.runtime.squash_action(latent))
            step = self.runtime.step(state, action, policy)
            reward = step.reward.to(dtype=self.dtype)
            valid, done = step.valid, step.done
            metrics.add(step.metrics)
            scaled_reward = hp.reward_scale * reward
            normalized = (
                self.normalizer.normalize(scaled_reward, done, frozen_scale)
                if self.normalizer
                else scaled_reward
            )
            safe_observation = torch.where(valid[:, None], step.observation, 0.0)
            value = self.target_critic(safe_observation)
            value = torch.where(~done | step.timeout, value, 0.0)
            segment_return = segment_return + segment_discount * normalized
            next_discount = segment_discount * hp.gamma
            boundary = (
                done if index < hp.training_horizon - 1 else torch.ones_like(done)
            )
            actor_loss = (
                actor_loss - (segment_return + next_discount * value)[boundary].sum()
            )
            with torch.no_grad():
                self.episode_returns.add_(reward.detach())
                self.episode_lengths.add_(1)
                completed_returns.append(
                    torch.where(done, self.episode_returns, torch.nan)
                )
                completed_lengths.append(torch.where(done, self.episode_lengths, 0))
                self.episode_returns.masked_fill_(done, 0.0)
                self.episode_lengths.masked_fill_(done, 0)
            rewards.append(normalized.detach())
            dones.append(done.detach())
            next_values.append(value.detach())
            raw_rewards.append(reward.detach())
            valid_count += valid.detach().sum()
            segment_return = torch.where(
                done, torch.zeros_like(segment_return), segment_return
            )
            segment_discount = torch.where(
                done, torch.ones_like(next_discount), next_discount
            )
            state = self.runtime.reset_done(
                step.state, done.detach(), hp.seed + self._reset_counter
            )
            self._reset_counter += 1
            observation = self._observation(state)
        self.state = state
        actor_loss = actor_loss / float(hp.training_horizon * hp.num_envs)
        if self.normalizer:
            actor_loss = actor_loss * frozen_scale
        if not actor_loss.requires_grad:
            raise RuntimeError(
                "SHAC requires a differentiable policy and plant transition"
            )
        episode_returns = torch.cat(completed_returns)
        episode_returns = episode_returns[torch.isfinite(episode_returns)]
        episode_lengths = torch.cat(completed_lengths)
        episode_lengths = episode_lengths[episode_lengths > 0]
        summary = metrics.finish("rollout")
        summary.update(
            {
                "rollout/reward_mean": float(torch.stack(raw_rewards).mean()),
                "rollout/valid_fraction": float(valid_count) / (
                    hp.training_horizon * hp.num_envs
                ),
                "rollout/episodes_completed": episode_lengths.numel(),
                "rollout/episode_return_mean": (
                    float(episode_returns.mean())
                    if episode_returns.numel()
                    else float("nan")
                ),
            }
        )
        return SHACRollout(
            actor_loss,
            torch.stack(observations),
            torch.stack(rewards),
            torch.stack(dones),
            torch.stack(next_values),
            summary,
        )

    def update(self, rollout):
        hp = self.hyperparameters
        fraction = (
            min(self.update_index / max(hp.total_updates - 1, 1), 1.0)
            if hp.anneal_lr
            else 0.0
        )
        actor_rate = hp.actor_learning_rate + fraction * (
            hp.final_learning_rate - hp.actor_learning_rate
        )
        critic_rate = (
            hp.critic_learning_rate
            if hp.critic_final_learning_rate is None
            else hp.critic_learning_rate
            + fraction * (hp.critic_final_learning_rate - hp.critic_learning_rate)
        )
        self.actor_optimizer.param_groups[0]["lr"] = actor_rate
        self.critic_optimizer.param_groups[0]["lr"] = critic_rate
        self.actor_optimizer.zero_grad(set_to_none=True)
        if not torch.isfinite(rollout.actor_loss):
            raise RuntimeError("SHAC actor loss is non-finite")
        rollout.actor_loss.backward(
            gradient=rollout.actor_loss.new_tensor(self.backward_scale)
        )
        gradient_norm = _clip_grad_norm_stable(
            self.actor.parameters(),
            hp.max_gradient_norm,
            gradient_scale=self.backward_scale,
        )
        if not torch.isfinite(gradient_norm):
            raise RuntimeError("SHAC actor gradient is non-finite")
        self.actor_optimizer.step()
        targets = td_lambda_targets(
            rollout.rewards, rollout.next_values, rollout.dones, hp.gamma, hp.td_lambda
        ).reshape(-1)
        observations = rollout.observations.reshape(-1, rollout.observations.shape[-1])
        minibatch = max(1, math.ceil(targets.numel() / hp.critic_minibatches))
        losses, norms = [], []
        predictions = torch.empty_like(targets)
        self.critic.train()
        for iteration in range(hp.critic_iterations):
            indices = (
                torch.randperm(targets.numel(), device=self.device)
                if hp.shuffle_critic
                else torch.arange(targets.numel(), device=self.device)
            )
            for start in range(0, targets.numel(), minibatch):
                rows = indices[start : start + minibatch]
                prediction = self.critic(observations.index_select(0, rows))
                loss = (prediction - targets.index_select(0, rows)).square().mean()
                if iteration == hp.critic_iterations - 1:
                    predictions.index_copy_(0, rows, prediction.detach())
                self.critic_optimizer.zero_grad(set_to_none=True)
                loss.backward()
                norm = _clip_grad_norm_stable(
                    self.critic.parameters(), hp.max_gradient_norm
                )
                if not torch.isfinite(norm):
                    raise RuntimeError("SHAC critic gradient is non-finite")
                self.critic_optimizer.step()
                losses.append(float(loss.detach()))
                norms.append(float(norm))
        with torch.no_grad():
            for target, current in zip(
                self.target_critic.parameters(), self.critic.parameters()
            ):
                target.mul_(hp.target_critic_alpha).add_(
                    current, alpha=1.0 - hp.target_critic_alpha
                )
        self.update_index += 1
        return {
            "train/actor_loss": float(rollout.actor_loss.detach()),
            "train/value_loss": float(np.mean(losses)),
            "train/actor_gradient_norm": float(gradient_norm),
            "train/critic_gradient_norm": float(np.mean(norms)),
            "train/actor_learning_rate": actor_rate,
            "train/critic_learning_rate": critic_rate,
            "critic/explained_variance": explained_variance(predictions, targets),
        }
