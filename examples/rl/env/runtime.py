"""Torch environment interface for SHAC and PPO."""

from dataclasses import dataclass, field
from typing import Any, Protocol

import torch


@dataclass
class Transition:
    state: Any
    observation: torch.Tensor
    reward: torch.Tensor
    done: torch.Tensor
    timeout: torch.Tensor  # Done rows eligible for a pre-reset value bootstrap.
    valid: torch.Tensor
    metrics: dict[str, torch.Tensor] = field(default_factory=dict)


@dataclass(frozen=True)
class EvaluationProtocol:
    reset_kwargs: dict = field(default_factory=lambda: {"evaluation": True})
    reset_done_kwargs: dict = field(default_factory=lambda: {"evaluation": True})
    retire: bool = False
    fixed_seed: bool = False


class Runtime(Protocol):
    device: torch.device
    dtype: torch.dtype
    observation_dim: int
    action_dim: int

    def reset(self, batch_size: int, seed: int, **kwargs) -> Any:
        """Start a seeded batch of episodes."""
        ...

    def observe(self, state: Any) -> torch.Tensor:
        """Return differentiable observations with shape [batch, observation]."""
        ...

    def policy(
        self, parameters: torch.Tensor, state: Any, *, retain: bool, inputs=None
    ) -> Any:
        """Return the action mean, validity, and controller context."""
        ...

    def squash_action(self, action: torch.Tensor) -> torch.Tensor:
        """Map sampled actions to valid plant inputs."""
        ...

    def step(self, state: Any, action: torch.Tensor, policy: Any) -> Transition:
        """Advance the plant and return the transition before reset."""
        ...

    def reset_done(self, state: Any, done: torch.Tensor, seed: int, **kwargs) -> Any:
        """Reset finished episodes and cut their gradient history."""
        ...

    def detach(self, state: Any) -> Any:
        """Cut the graph between rollout segments."""
        ...

    # PPO stores snapshots before collection and replays them independently.
    def replay_inputs(self, state: Any) -> dict[str, torch.Tensor]:
        """Snapshot policy inputs before advancing the state."""
        ...

    def replay(self, parameters: torch.Tensor, inputs: dict, *, retain: bool) -> tuple:
        """Recompute the action mean and validity from saved inputs."""
        ...
