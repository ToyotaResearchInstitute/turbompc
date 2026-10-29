"""RL training, evaluation, checkpoints, and metrics."""

import json
import random
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
import yaml

from ..env.runtime import EvaluationProtocol


def training_config(config):
    return config["training"]


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def zero_nonfinite_gradient(gradient):
    return torch.nan_to_num(gradient, nan=0.0, posinf=0.0, neginf=0.0)


class Metrics:
    """Aggregate scalar metrics across environment batches."""

    def __init__(self):
        self.values = {}

    def add(self, values, mask=None):
        for key, value in values.items():
            if not isinstance(value, torch.Tensor) or value.ndim != 1:
                continue
            value = value.detach().to(torch.float64)
            if mask is not None:
                value = value[mask]
            self.values.setdefault(key, []).append(value)

    def finish(self, prefix):
        result = {}
        for key, values in self.values.items():
            values = torch.cat(values)
            values = values[torch.isfinite(values)]
            if values.numel():
                result[f"{prefix}/{key}_mean"] = float(values.mean())
                result[f"{prefix}/{key}_max"] = float(values.max())
        return result


def explained_variance(predictions, targets):
    variance = targets.var(unbiased=False)
    return (
        float(1.0 - (targets - predictions).var(unbiased=False) / variance)
        if variance > 0
        else float("nan")
    )


@torch.no_grad()
def evaluate(algorithm, *, steps, num_envs=1, record=False, fixed_parameters=None):
    """Evaluate learned or fixed policy parameters."""
    runtime = algorithm.runtime
    protocol = (
        runtime.evaluation_protocol()
        if hasattr(runtime, "evaluation_protocol")
        else EvaluationProtocol()
    )
    seed = algorithm.hyperparameters.seed + algorithm.evaluation_seed_offset
    if not protocol.fixed_seed:
        seed += algorithm.update_index
    state = runtime.reset(num_envs, seed, **protocol.reset_kwargs)
    active = torch.ones(num_envs, dtype=torch.bool, device=runtime.device)
    lengths = torch.zeros(num_envs, dtype=torch.int64, device=runtime.device)
    rewards = torch.zeros(num_envs, dtype=runtime.dtype, device=runtime.device)
    valid_count = torch.zeros((), device=runtime.device)
    total_count = torch.zeros_like(valid_count)
    metrics = Metrics()
    trace = []
    sums = {}
    fixed = (
        None
        if fixed_parameters is None
        else torch.as_tensor(
            fixed_parameters, device=runtime.device, dtype=runtime.dtype
        ).expand(num_envs, -1)
    )
    if fixed is None:
        actor_training = algorithm.actor.training
        algorithm.actor.eval()
    try:
        for index in range(steps):
            if protocol.retire and not bool(active.any()):
                break
            mask = active if protocol.retire else torch.ones_like(active)
            parameters = (
                algorithm.policy_parameters(runtime.observe(state))
                if fixed is None
                else fixed
            )
            kwargs = {"active": mask} if protocol.retire else {}
            policy = runtime.policy(parameters, state, retain=False, **kwargs)
            transition = runtime.step(state, policy.mean, policy, **kwargs)
            rewards += torch.where(mask, transition.reward, 0.0)
            lengths += mask
            valid_count += (mask & transition.valid).sum()
            total_count += mask.sum()
            metrics.add(transition.metrics, mask)
            for key in getattr(runtime, "evaluation_sum_metrics", ()):
                value = torch.where(mask, transition.metrics[key], 0.0)
                sums[key] = sums.get(key, torch.zeros_like(value)) + value
            if record:
                trace.append(runtime.trace_frame(state, transition, policy))
            state = transition.state
            if protocol.retire:
                active = active & ~transition.done
            else:
                state = runtime.reset_done(
                    state,
                    transition.done,
                    algorithm.hyperparameters.seed + 2_000_000 + index,
                    **protocol.reset_done_kwargs,
                )
    finally:
        if fixed is None:
            algorithm.actor.train(actor_training)
    result = metrics.finish("eval")
    result.update(
        {
            "eval/reward_mean": float((rewards / lengths.clamp_min(1)).mean()),
            "eval/return_mean": float(rewards.mean()),
            "eval/valid_fraction": float(valid_count / total_count.clamp_min(1)),
            "eval/episodic_length_mean": float(lengths.float().mean()),
        }
    )
    result.update(
        {f"eval/{key}_total_mean": float(value.mean()) for key, value in sums.items()}
    )
    return (result, trace) if record else result


def save_checkpoint(algorithm, path):
    """Save networks and normalizers for resuming from fresh episodes."""
    payload = {
        "update": algorithm.update_index,
        "hyperparameters": asdict(algorithm.hyperparameters),
    }
    for name in (
        "actor",
        "critic",
        "target_critic",
        "optimizer",
        "actor_optimizer",
        "critic_optimizer",
    ):
        value = getattr(algorithm, name, None)
        if value is not None:
            payload[name] = value.state_dict()
    for name in ("normalizer", "obs_rms", "reward_rms"):
        value = getattr(algorithm, name, None)
        if value is not None:
            payload[name] = vars(value).copy()
    torch.save(payload, path)


def load_checkpoint(algorithm, path):
    payload = torch.load(path, map_location=algorithm.device, weights_only=False)
    for name in (
        "actor",
        "critic",
        "target_critic",
        "optimizer",
        "actor_optimizer",
        "critic_optimizer",
    ):
        if name in payload:
            getattr(algorithm, name).load_state_dict(payload[name])
    for name in ("normalizer", "obs_rms", "reward_rms"):
        if name in payload:
            vars(getattr(algorithm, name)).update(payload[name])
    algorithm.update_index = int(payload["update"])
    algorithm.reset_training_state()
    return algorithm.update_index


def _json_value(value):
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    return None if isinstance(value, float) and not np.isfinite(value) else value


def train(algorithm, config, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "metrics.jsonl").exists():
        raise FileExistsError(f"choose a fresh run directory: {output}")
    (output / "effective_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False)
    )
    run_config = config["training"][config["algorithm"]]
    evaluation = config["evaluation"]
    every = int(run_config.get("eval_every", 10))
    checkpoint_every = int(run_config.get("checkpoint_every", 10))
    with (output / "metrics.jsonl").open("w") as stream:
        for update in range(
            algorithm.update_index + 1, algorithm.hyperparameters.total_updates + 1
        ):
            start = perf_counter()
            rollout = algorithm.collect()
            collected = perf_counter()
            metrics = {**rollout.metrics, **algorithm.update(rollout)}
            trained = perf_counter()
            if checkpoint_every > 0 and update % checkpoint_every == 0:
                save_checkpoint(algorithm, output / f"checkpoint_{update:04d}.pt")
            eval_start = perf_counter()
            if every > 0 and update % every == 0:
                metrics.update(
                    evaluate(
                        algorithm,
                        steps=evaluation["steps"],
                        num_envs=evaluation["num_envs"],
                    )
                )
            metrics.update(
                update=update,
                **{
                    "time/collect_seconds": collected - start,
                    "time/train_seconds": trained - collected,
                    "time/eval_seconds": (
                        perf_counter() - eval_start
                        if every > 0 and update % every == 0
                        else 0.0
                    ),
                    "time/update_seconds": perf_counter() - start,
                },
            )
            stream.write(json.dumps(_json_value(metrics), sort_keys=True) + "\n")
            stream.flush()
            summary = " ".join(
                f"{key}={value:.5g}"
                for key, value in metrics.items()
                if key
                in {
                    "rollout/reward_mean",
                    "rollout/valid_fraction",
                    "rollout/admm_iterations_mean",
                    "train/actor_loss",
                    "train/policy_loss",
                    "train/value_loss",
                    "train/actor_gradient_norm",
                    "eval/reward_mean",
                    "eval/valid_fraction",
                    "time/collect_seconds",
                    "time/train_seconds",
                }
            )
            print(
                (
                    f"update {update:04d}/{algorithm.hyperparameters.total_updates}:"
                    f" {summary}"
                ),
                flush=True,
            )
    save_checkpoint(algorithm, output / "policy_final.pt")
    return output
