"""PPO for direct and MPC policies."""

from dataclasses import dataclass

import torch
from torch import nn
from torch.distributions import Normal

from ..env.runtime import Runtime
from ..util.training import Metrics, explained_variance


@dataclass(frozen=True)
class PPOConfig:
    seed: int = 0
    num_envs: int = 64
    total_updates: int = 100
    num_steps: int = 128
    num_minibatches: int = 32
    update_epochs: int = 10
    actor_learning_rate: float = 3e-4
    critic_learning_rate: float = 1e-3
    # None fits actor/critic together. An integer fits the critic separately.
    critic_iterations: int | None = None
    critic_max_grad_norm: float | None = None
    adam_betas: tuple = (0.9, 0.999)
    adam_epsilon: float = 1e-5
    anneal_lr: bool = True
    final_actor_learning_rate: float = 0.0
    final_critic_learning_rate: float | None = None
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    norm_adv: bool = True
    clip_vloss: bool = True
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float | None = None
    normalize_obs: bool = True
    normalize_reward: bool = True
    obs_clip: float = 10.0
    reward_clip: float = 10.0

    def __post_init__(self):
        if (
            min(
                self.num_envs,
                self.num_steps,
                self.num_minibatches,
                self.update_epochs,
                self.total_updates,
            )
            < 1
        ):
            raise ValueError("PPO batch sizes and update counts must be positive")
        if self.num_steps * self.num_envs % self.num_minibatches:
            raise ValueError("PPO rollout size must be divisible by num_minibatches")
        if self.critic_iterations is not None and self.critic_iterations < 1:
            raise ValueError("critic_iterations must be positive")
        if not all(0 <= value <= 1 for value in (self.gamma, self.gae_lambda)):
            raise ValueError("PPO discounts must lie in [0, 1]")


class RunningMeanStd:
    def __init__(self, shape, device, dtype):
        self.mean = torch.zeros(shape, device=device, dtype=dtype)
        self.var = torch.ones_like(self.mean)
        self.count = 1e-4

    @torch.no_grad()
    def update(self, batch):
        batch_mean, batch_var, count = (
            batch.mean(0),
            batch.var(0, unbiased=False),
            batch.shape[0],
        )
        delta = batch_mean - self.mean
        total = self.count + count
        m2 = (
            self.var * self.count
            + batch_var * count
            + delta.square() * self.count * count / total
        )
        self.mean = self.mean + delta * count / total
        self.var, self.count = m2 / total, total


@dataclass
class Rollout:
    observations: torch.Tensor
    inputs: dict[str, torch.Tensor]
    actions: torch.Tensor
    logprobs: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    values: torch.Tensor
    valid: torch.Tensor
    metrics: dict[str, float]


def gae(rewards, values, dones, next_value, next_done, gamma, lam):
    advantages = torch.zeros_like(rewards)
    running = torch.zeros_like(next_value)
    for index in reversed(range(rewards.shape[0])):
        continuation = 1.0 - (
            next_done.to(rewards.dtype)
            if index == rewards.shape[0] - 1
            else dones[index + 1]
        )
        value = next_value if index == rewards.shape[0] - 1 else values[index + 1]
        delta = rewards[index] + gamma * value * continuation - values[index]
        running = delta + gamma * lam * continuation * running
        advantages[index] = running
    return advantages, advantages + values


class PPO:
    evaluation_seed_offset = 2_000_000

    def __init__(self, runtime: Runtime, actor, critic, config: PPOConfig):
        self.runtime, self.actor, self.critic = runtime, actor, critic
        self.hyperparameters = hp = config
        self.device, self.dtype = runtime.device, runtime.dtype
        self.actor_parameters = list(actor.parameters())
        self.critic_parameters = list(critic.parameters())
        self.optimizer = torch.optim.Adam(
            [
                {"params": self.actor_parameters, "lr": hp.actor_learning_rate},
                {"params": self.critic_parameters, "lr": hp.critic_learning_rate},
            ],
            betas=hp.adam_betas,
            eps=hp.adam_epsilon,
        )
        self.obs_rms = (
            RunningMeanStd((runtime.observation_dim,), self.device, self.dtype)
            if hp.normalize_obs
            else None
        )
        self.reward_rms = (
            RunningMeanStd((), self.device, self.dtype) if hp.normalize_reward else None
        )
        self.update_index = 0
        self.reset_training_state(update_statistics=True)

    def reset_training_state(self, *, update_statistics=False):
        """Reset episodes while preserving networks and normalizers."""
        hp = self.hyperparameters
        self.returns = torch.zeros(hp.num_envs, device=self.device, dtype=self.dtype)
        self.episode_returns = torch.zeros_like(self.returns)
        self.episode_lengths = torch.zeros_like(self.returns)
        self._reset_counter = 1
        self.state = self.runtime.reset(hp.num_envs, hp.seed)
        self.next_obs = self.normalize_obs(
            self.runtime.observe(self.state), update=update_statistics
        )
        self.next_done = torch.zeros(hp.num_envs, dtype=torch.bool, device=self.device)

    def normalize_obs(self, observation, *, update=False):
        observation = observation.to(device=self.device, dtype=self.dtype)
        if self.obs_rms is None:
            return observation
        if update:
            self.obs_rms.update(observation)
        scaled = (observation - self.obs_rms.mean) / torch.sqrt(self.obs_rms.var + 1e-8)
        return scaled.clamp(
            -self.hyperparameters.obs_clip, self.hyperparameters.obs_clip
        )

    def policy_parameters(self, observation):
        return self.actor(self.normalize_obs(observation))

    def distribution(self, mean):
        return Normal(
            mean, self.actor.log_std.exp().expand_as(mean), validate_args=False
        )

    def normalize_reward(self, reward, done):
        if self.reward_rms is None:
            return reward
        hp = self.hyperparameters
        self.returns = self.returns * hp.gamma * (~done).to(self.dtype) + reward
        self.reward_rms.update(self.returns)
        return (reward / torch.sqrt(self.reward_rms.var + 1e-8)).clamp(
            -hp.reward_clip, hp.reward_clip
        )

    @torch.no_grad()
    def collect(self):
        hp = self.hyperparameters
        observations, actions, logprobs, rewards, dones, values, valid_rows = (
            [] for _ in range(7)
        )
        contexts = {}
        metrics = Metrics()
        raw_reward_sum, completed_returns, completed_lengths = 0.0, [], []
        state = self.state
        for _ in range(hp.num_steps):
            observations.append(self.next_obs)
            dones.append(self.next_done.to(self.dtype))
            # Snapshot before the solve; replay never borrows a mutable warm start.
            inputs = {
                key: value.detach().clone()
                for key, value in self.runtime.replay_inputs(state).items()
            }
            for key, value in inputs.items():
                contexts.setdefault(key, []).append(value)
            parameters = self.actor(self.next_obs)
            policy = self.runtime.policy(parameters, state, retain=False, inputs=inputs)
            distribution = self.distribution(policy.mean)
            latent = distribution.sample()
            actions.append(latent)
            logprobs.append(distribution.log_prob(latent).sum(-1))
            values.append(self.critic(self.next_obs))
            transition = self.runtime.step(
                state, self.runtime.squash_action(latent), policy
            )
            reward, done = transition.reward, transition.done
            valid_rows.append(transition.valid & policy.valid)
            metrics.add(transition.metrics)
            raw_reward_sum += float(reward.sum())
            self.episode_returns += reward
            self.episode_lengths += 1.0
            if bool(done.any()):
                completed_returns.extend(self.episode_returns[done].tolist())
                completed_lengths.extend(self.episode_lengths[done].tolist())
                self.episode_returns = torch.where(
                    done, torch.zeros_like(self.episode_returns), self.episode_returns
                )
                self.episode_lengths = torch.where(
                    done, torch.zeros_like(self.episode_lengths), self.episode_lengths
                )
            normalized_reward = self.normalize_reward(reward, done)
            if bool(transition.timeout.any()):
                terminal_value = self.critic(self.normalize_obs(transition.observation))
                normalized_reward = (
                    normalized_reward
                    + hp.gamma * terminal_value * transition.timeout.to(self.dtype)
                )
            rewards.append(normalized_reward)
            state = self.runtime.reset_done(
                transition.state, done, hp.seed + self._reset_counter
            )
            self._reset_counter += 1
            self.next_obs = self.normalize_obs(self.runtime.observe(state), update=True)
            self.next_done = done
        self.state = self.runtime.detach(state)
        summary = metrics.finish("rollout")
        summary.update(
            {
                "rollout/reward_mean": raw_reward_sum / (hp.num_steps * hp.num_envs),
                "rollout/valid_fraction": float(torch.stack(valid_rows).float().mean()),
                "rollout/episodes_completed": len(completed_returns),
            }
        )
        if completed_returns:
            summary["rollout/episode_return_mean"] = sum(completed_returns) / len(
                completed_returns
            )
        return Rollout(
            torch.stack(observations),
            {key: torch.stack(value) for key, value in contexts.items()},
            torch.stack(actions),
            torch.stack(logprobs),
            torch.stack(rewards),
            torch.stack(dones),
            torch.stack(values),
            torch.stack(valid_rows),
            summary,
        )

    def _policy_loss(
        self, observations, inputs, actions, old_logprob, advantages, valid
    ):
        hp = self.hyperparameters
        mean, replay_valid = self.runtime.replay(
            self.actor(observations), inputs, retain=True
        )
        valid = (
            valid
            & replay_valid
            & torch.isfinite(actions).all(-1)
            & torch.isfinite(old_logprob)
        )
        mean = torch.where(valid[:, None], mean, 0.0)
        distribution = self.distribution(mean)
        new_logprob = distribution.log_prob(
            torch.where(valid[:, None], actions, 0.0)
        ).sum(-1)
        entropy = distribution.entropy().sum(-1)
        weight = valid.to(self.dtype)
        count, denominator = weight.sum(), weight.sum().clamp_min(1)
        log_ratio = new_logprob - torch.where(valid, old_logprob, 0.0)
        ratio = log_ratio.exp()
        if hp.norm_adv and advantages.numel() > 1:
            if hp.critic_iterations is None:
                advantages = (advantages - advantages.mean()) / (
                    advantages.std() + 1e-8
                )
            else:
                centered = advantages - (advantages * weight).sum() / denominator
                variance = (centered.square() * weight).sum() / (count - 1).clamp_min(1)
                advantages = torch.where(
                    count > 1, centered / (variance.sqrt() + 1e-8), advantages
                )
        surrogate = torch.maximum(
            -advantages * ratio,
            -advantages * ratio.clamp(1.0 - hp.clip_coef, 1.0 + hp.clip_coef),
        )
        pg_loss = (surrogate * weight).sum() / denominator
        entropy_loss = (entropy * weight).sum() / denominator
        with torch.no_grad():
            kl = (((ratio - 1) - log_ratio) * weight).sum() / denominator
            clipfrac = (
                ((ratio - 1).abs() > hp.clip_coef).to(self.dtype) * weight
            ).sum() / denominator
        return pg_loss - hp.ent_coef * entropy_loss, pg_loss, entropy_loss, kl, clipfrac

    def _value_loss(self, observations, old_values, returns):
        hp = self.hyperparameters
        value = self.critic(observations)
        error = (value - returns).square()
        if hp.clip_vloss:
            clipped = old_values + (value - old_values).clamp(
                -hp.clip_coef, hp.clip_coef
            )
            error = torch.maximum(error, (clipped - returns).square())
        return 0.5 * error.mean()

    def _optimize(self, loss, parameters, limit):
        self.optimizer.zero_grad(set_to_none=True)
        if not torch.isfinite(loss):
            raise RuntimeError("PPO loss is non-finite")
        loss.backward()
        norm = nn.utils.clip_grad_norm_(parameters, limit, error_if_nonfinite=True)
        self.optimizer.step()
        return norm.detach().to(torch.float64)

    def _minibatches(self, size):
        permutation = torch.randperm(size, device=self.device)
        batch = size // self.hyperparameters.num_minibatches
        return permutation.split(batch)

    def update(self, rollout):
        hp = self.hyperparameters
        self.update_index += 1
        if hp.anneal_lr:
            fraction = 1 - (self.update_index - 1) / hp.total_updates
            self.optimizer.param_groups[0]["lr"] = (
                hp.final_actor_learning_rate
                + fraction * (hp.actor_learning_rate - hp.final_actor_learning_rate)
            )
            if hp.final_critic_learning_rate is not None:
                self.optimizer.param_groups[1]["lr"] = (
                    hp.final_critic_learning_rate
                    + fraction
                    * (hp.critic_learning_rate - hp.final_critic_learning_rate)
                )
        with torch.no_grad():
            advantages, returns = gae(
                rollout.rewards,
                rollout.values,
                rollout.dones,
                self.critic(self.next_obs),
                self.next_done,
                hp.gamma,
                hp.gae_lambda,
            )
        size = hp.num_steps * hp.num_envs
        observations = rollout.observations.flatten(0, 1)
        inputs = {key: value.flatten(0, 1) for key, value in rollout.inputs.items()}
        actions = rollout.actions.flatten(0, 1)
        old_logprobs, advantages, returns = (
            rollout.logprobs.flatten(),
            advantages.flatten(),
            returns.flatten(),
        )
        old_values, valid = rollout.values.flatten(), rollout.valid.flatten()
        joint = hp.critic_iterations is None
        norms, critic_norms, clipfracs, value_losses = [], [], [], []
        for _ in range(hp.update_epochs):
            for index in self._minibatches(size):
                loss, policy_loss, entropy, kl, clipfrac = self._policy_loss(
                    observations[index],
                    {key: value[index] for key, value in inputs.items()},
                    actions[index],
                    old_logprobs[index],
                    advantages[index],
                    valid[index],
                )
                clipfracs.append(clipfrac)
                parameters = self.actor_parameters
                if joint:
                    value_loss = self._value_loss(
                        observations[index], old_values[index], returns[index]
                    )
                    loss = loss + hp.vf_coef * value_loss
                    value_losses.append(value_loss.detach())
                    parameters = parameters + self.critic_parameters
                norms.append(self._optimize(loss, parameters, hp.max_grad_norm))
            if hp.target_kl is not None and kl > hp.target_kl:
                break
        if not joint:
            for _ in range(hp.critic_iterations):
                for index in self._minibatches(size):
                    value_loss = self._value_loss(
                        observations[index], old_values[index], returns[index]
                    )
                    value_losses.append(value_loss.detach())
                    critic_norms.append(
                        self._optimize(
                            hp.vf_coef * value_loss,
                            self.critic_parameters,
                            hp.critic_max_grad_norm or hp.max_grad_norm,
                        )
                    )
        with torch.no_grad():
            predictions = self.critic(observations)
        result = {
            "train/policy_loss": float(policy_loss.detach()),
            "train/value_loss": (
                float(value_loss.detach())
                if joint
                else float(0.5 * (predictions - returns).square().mean())
            ),
            "train/value_loss_optimization_mean": float(
                torch.stack(value_losses).mean()
            ),
            "train/entropy": float(entropy.detach()),
            "train/approx_kl": float(kl),
            "train/clipfrac": float(torch.stack(clipfracs).mean()),
            "train/gradient_norm": float(torch.stack(norms).mean()),
            "train/explained_variance": explained_variance(predictions, returns),
            "train/actor_learning_rate": self.optimizer.param_groups[0]["lr"],
            "train/critic_learning_rate": self.optimizer.param_groups[1]["lr"],
        }
        if critic_norms:
            result["train/critic_gradient_norm"] = float(
                torch.stack(critic_norms).mean()
            )
        return result
