"""Benchmark racing iteration time for MPC-SHAC, MPC-PPO, and MPC-TD3."""

import argparse
import json
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
import yaml
from torch import nn

from ..racing_training_experiment import build, configure_jax


class Ledger:
    def __init__(self, device):
        self.device = device
        self.reset()

    def sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def reset(self):
        self.seconds = defaultdict(float)
        self.components = {}
        self.phase = None
        self.started = None

    def switch(self, phase):
        self.sync()
        now = perf_counter()
        if self.phase is not None:
            self.seconds[self.phase] += now - self.started
        self.phase, self.started = phase, now

    def instrument(self, backend, batch_size):
        for name, component in (
            ("solve_device", "forward"),
            ("vjp_plan_device", "backward"),
        ):
            original = getattr(backend, name)

            def timed(*args, _original=original, _component=component, **kwargs):
                rows = (
                    args[0].shape[0] if _component == "forward" else args[0].batch_size
                )
                if rows != batch_size:
                    raise AssertionError(
                        f"MPC {_component}: expected {batch_size} rows, got {rows}"
                    )
                self.sync()
                start = perf_counter()
                result = _original(*args, **kwargs)
                self.sync()
                key = f"{self.phase}/{_component}"
                entry = self.components.setdefault(
                    key, {"seconds": 0.0, "calls": 0, "rows": 0}
                )
                entry["seconds"] += perf_counter() - start
                entry["calls"] += 1
                entry["rows"] += int(rows)
                return result

            setattr(backend, name, timed)

    def result(self):
        self.switch(None)
        return {
            **{
                f"{key}_seconds": self.seconds[key]
                for key in ("collection", "actor", "critic")
            },
            "total_seconds": sum(self.seconds.values()),
            "mpc": deepcopy(self.components),
        }


class ReplayBuffer:
    """Store transitions and MPC replay inputs."""

    def __init__(self, capacity, device):
        self.capacity, self.device = capacity, device
        self.storage = {}
        self.position = self.size = 0

    def add(self, values):
        rows = values["observation"].shape[0]
        if rows > self.capacity:
            raise ValueError("replay capacity is smaller than a collection batch")
        indices = (
            torch.arange(rows, device=self.device) + self.position
        ) % self.capacity
        for name, value in values.items():
            if name not in self.storage:
                self.storage[name] = value.new_empty((self.capacity, *value.shape[1:]))
            self.storage[name].index_copy_(0, indices, value.detach())
        self.position = (self.position + rows) % self.capacity
        self.size = min(self.size + rows, self.capacity)

    def sample(self, size):
        index = torch.randint(self.size, (size,), device=self.device)
        return {key: value[index] for key, value in self.storage.items()}


class QNetwork(nn.Module):
    def __init__(self, nx, nu):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(nx + nu, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, observation, action):
        return self.net(torch.cat((observation, action), -1)).squeeze(-1)


class TD3Profile:
    """TD3 with an MPC actor and twin Q critics."""

    def __init__(
        self, runtime, actor, ledger, *, batch_size, steps, gradient_steps, seed=0
    ):
        self.runtime, self.actor, self.ledger = runtime, actor, ledger
        self.batch_size, self.steps, self.gradient_steps = (
            batch_size,
            steps,
            gradient_steps,
        )
        self.seed, self.reset_counter, self.gradient_step = seed, 0, 0
        self.target_actor = deepcopy(actor).requires_grad_(False)
        self.q1, self.q2 = (
            QNetwork(runtime.observation_dim, runtime.action_dim).to(
                device=runtime.device, dtype=runtime.dtype
            )
            for _ in range(2)
        )
        self.target_q1, self.target_q2 = (
            deepcopy(q).requires_grad_(False) for q in (self.q1, self.q2)
        )
        self.actor_optimizer = torch.optim.Adam(actor.parameters(), lr=0.0)
        self.q_optimizer = torch.optim.Adam(
            (*self.q1.parameters(), *self.q2.parameters()), lr=0.0
        )
        self.buffer = ReplayBuffer(batch_size * steps, runtime.device)
        self.state = runtime.reset(batch_size, seed)
        self.scale = runtime.action_high - runtime.action_low

    @torch.no_grad()
    def collect(self):
        valid = []
        for _ in range(self.steps):
            observation = self.runtime.observe(self.state)
            inputs = self.runtime.replay_inputs(self.state)
            policy = self.runtime.policy(
                self.actor(observation), self.state, retain=False, inputs=inputs
            )
            action = self.runtime.squash_action(
                policy.mean + 0.1 * self.scale * torch.randn_like(policy.mean)
            )
            transition = self.runtime.step(self.state, action, policy)
            # Timeouts bootstrap the physical pre-reset successor, never the
            # new episode's initial state. True terminal rows do not bootstrap.
            successor = self.runtime.replay_inputs(transition.state)
            self.buffer.add(
                {
                    "observation": observation,
                    "action": action,
                    "reward": transition.reward,
                    "next_observation": transition.observation,
                    "terminal": transition.done & ~transition.timeout,
                    "valid": transition.valid & policy.valid,
                    **{f"input/{key}": value for key, value in inputs.items()},
                    **{f"next_input/{key}": value for key, value in successor.items()},
                }
            )
            self.state = self.runtime.reset_done(
                transition.state, transition.done, self.seed + self.reset_counter
            )
            self.reset_counter += 1
            valid.append(transition.valid)
        return float(torch.stack(valid).float().mean())

    def update(self):
        actor_updates = 0
        for _ in range(self.gradient_steps):
            self.ledger.switch("critic")
            batch = self.buffer.sample(self.batch_size)
            inputs = {
                key.removeprefix("input/"): value
                for key, value in batch.items()
                if key.startswith("input/")
            }
            successor = {
                key.removeprefix("next_input/"): value
                for key, value in batch.items()
                if key.startswith("next_input/")
            }
            with torch.no_grad():
                mean, target_valid = self.runtime.replay(
                    self.target_actor(batch["next_observation"]),
                    successor,
                    retain=False,
                )
                noise = (torch.randn_like(mean) * 0.2 * self.scale).clamp(
                    -0.1 * self.scale, 0.1 * self.scale
                )
                next_action = self.runtime.squash_action(mean + noise)
                target_q = torch.minimum(
                    self.target_q1(batch["next_observation"], next_action),
                    self.target_q2(batch["next_observation"], next_action),
                )
                target = (
                    batch["reward"]
                    + 0.99 * (~batch["terminal"] & target_valid) * target_q
                )
            weight = batch["valid"].to(self.runtime.dtype)
            loss = sum(
                (
                    (q(batch["observation"], batch["action"]) - target).square()
                    * weight
                ).sum()
                for q in (self.q1, self.q2)
            ) / weight.sum().clamp_min(1)
            if not torch.isfinite(loss):
                raise RuntimeError("TD3 critic loss is non-finite")
            self.q_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.q_optimizer.step()
            self.gradient_step += 1
            if self.gradient_step % 2 == 0:
                self.ledger.switch("actor")
                mean, valid = self.runtime.replay(
                    self.actor(batch["observation"]), inputs, retain=True
                )
                weight = (valid & batch["valid"]).to(self.runtime.dtype)
                actor_loss = -(
                    self.q1(batch["observation"], mean) * weight
                ).sum() / weight.sum().clamp_min(1)
                self.actor_optimizer.zero_grad(set_to_none=True)
                actor_loss.backward()
                if not torch.isfinite(actor_loss):
                    raise RuntimeError("TD3 actor loss is non-finite")
                self.actor_optimizer.step()
                with torch.no_grad():
                    for source, target_network in (
                        (self.actor, self.target_actor),
                        (self.q1, self.target_q1),
                        (self.q2, self.target_q2),
                    ):
                        for value, target in zip(
                            source.parameters(), target_network.parameters()
                        ):
                            target.lerp_(value, 0.005)
                actor_updates += 1
        return actor_updates


def freeze_optimizer(optimizer, after=None):
    original = optimizer.step

    def step(*args, **kwargs):
        # SHAC/PPO may reset learning rates at the beginning of each update.
        for group in optimizer.param_groups:
            group["lr"] = 0.0
        result = original(*args, **kwargs)
        if after:
            after()
        return result

    optimizer.step = step


def make_config(name, batch_size, steps, mpc_horizon, trials, seed):
    algorithm = "shac" if name == "shac" else "ppo"
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / f"configs/{algorithm}_mpc.yaml").read_text())
    config["env"].update(horizon=mpc_horizon, max_episode_steps=steps)
    config["training"].update(batch_size=batch_size, device="cuda")
    hp = config["training"][algorithm]
    hp.update(seed=seed, total_updates=trials + 2, anneal_lr=False)
    if algorithm == "shac":
        hp.update(training_horizon=steps)
    else:
        hp.update(
            num_steps=steps,
            num_minibatches=steps,
            update_epochs=10,
            critic_iterations=10,
            target_kl=None,
        )
    return config


def run_case(args):
    configure_jax()
    import jax

    if not torch.cuda.is_available() or not all(
        d.platform == "gpu" for d in jax.devices()
    ):
        raise RuntimeError(
            "Racing iteration benchmarks require both PyTorch and JAX on CUDA"
        )
    if (
        min(args.batch_size, args.steps, args.mpc_horizon, args.trials) < 1
        or args.warmups < 1
        or args.td3_utd <= 0
    ):
        raise ValueError(
            "positive workload sizes and at least one warm-up are required"
        )
    config = make_config(
        args.algorithm,
        args.batch_size,
        args.steps,
        args.mpc_horizon,
        args.trials,
        args.seed,
    )
    learner = build(config)
    ledger = Ledger(learner.runtime.device)
    ledger.instrument(learner.runtime.backend, args.batch_size)
    gradient_steps = max(1, round(args.td3_utd * args.batch_size * args.steps))
    if args.algorithm == "td3":
        learner = TD3Profile(
            learner.runtime,
            learner.actor,
            ledger,
            batch_size=args.batch_size,
            steps=args.steps,
            gradient_steps=gradient_steps,
            seed=args.seed,
        )
    elif args.algorithm == "shac":
        freeze_optimizer(learner.actor_optimizer, lambda: ledger.switch("critic"))
        freeze_optimizer(learner.critic_optimizer)
    else:
        freeze_optimizer(learner.optimizer)
        original = learner._value_loss

        def critic_loss(*values):
            if ledger.phase != "critic":
                ledger.switch("critic")
            return original(*values)

        learner._value_loss = critic_loss

    def iteration():
        ledger.reset()
        ledger.switch("collection")
        rollout = learner.collect()
        ledger.switch("actor")
        if args.algorithm == "td3":
            actor_updates = learner.update()
            valid_fraction = rollout
        else:
            learner.update(rollout)
            actor_updates = 1 if args.algorithm == "shac" else args.steps * 10
            valid_fraction = rollout.metrics["rollout/valid_fraction"]
        result = ledger.result()
        result.update(actor_updates=actor_updates, valid_fraction=valid_fraction)
        return result

    warmups = [iteration() for _ in range(args.warmups)]
    print(
        f"{args.algorithm}: warmed in {sum(r['total_seconds'] for r in warmups):.2f}s",
        flush=True,
    )
    measurements = []
    for i in range(args.trials):
        row = iteration()
        measurements.append(row)
        print(f"trial {i + 1}/{args.trials}: {row['total_seconds']:.3f}s", flush=True)
    fields = ["collection_seconds", "actor_seconds", "critic_seconds", "total_seconds"]
    result = {
        "schema_version": 1,
        "algorithm": args.algorithm,
        "config": config,
        "batch_size": args.batch_size,
        "rollout_steps": args.steps,
        "mpc_horizon": args.mpc_horizon,
        "warmups": warmups,
        "measurements": measurements,
        "mean": {
            key: float(np.mean([row[key] for row in measurements])) for key in fields
        },
        "std": {
            key: (
                float(np.std([row[key] for row in measurements], ddof=1))
                if len(measurements) > 1
                else 0.0
            )
            for key in fields
        },
        "td3_gradient_steps": gradient_steps if args.algorithm == "td3" else None,
        "td3_utd": (
            gradient_steps / (args.batch_size * args.steps)
            if args.algorithm == "td3"
            else None
        ),
        "learning_rates_frozen": True,
        "device": "cuda",
        "gpu_name": torch.cuda.get_device_name(),
        "torch_version": torch.__version__,
        "jax_version": jax.__version__,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", choices=("shac", "ppo", "td3"), required=True)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--mpc-horizon", type=int, default=32)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--td3-utd", type=float, default=1 / 50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    result = run_case(parser.parse_args())
    print("RESULT_JSON=" + json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
