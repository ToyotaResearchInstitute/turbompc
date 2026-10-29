"""Numerical operations used by SHAC."""

from dataclasses import dataclass

import numpy as np
import torch


class _ExpandedStandardDeviation(torch.autograd.Function):
    """Broadcast exploration scales with a float64 batch reduction."""

    @staticmethod
    def forward(ctx, log_std: torch.Tensor, reference: torch.Tensor):
        standard_deviation = log_std.to(dtype=reference.dtype).exp()
        ctx.save_for_backward(standard_deviation)
        ctx.log_std_dtype = log_std.dtype
        return standard_deviation.expand_as(reference)

    @staticmethod
    def backward(ctx, gradient: torch.Tensor):
        (standard_deviation,) = ctx.saved_tensors
        reduced = gradient.to(torch.float64).sum(dim=0)
        log_std_gradient = reduced * standard_deviation.to(torch.float64)
        return log_std_gradient.to(dtype=ctx.log_std_dtype), None


@torch.no_grad()
def _clip_grad_norm_stable(
    parameters,
    max_norm: float,
    *,
    gradient_scale: float = 1.0,
) -> torch.Tensor:
    """Unscale and clip by global L2 norm without float32 overflow."""
    if not np.isfinite(gradient_scale) or gradient_scale <= 0.0:
        raise ValueError("gradient_scale must be finite and positive")
    parameters = [parameter for parameter in parameters if parameter.grad is not None]
    if not parameters:
        return torch.zeros((), dtype=torch.float64)
    norms = []
    for parameter in parameters:
        gradient = parameter.grad.detach()
        if not torch.all(torch.isfinite(gradient)):
            return torch.full(
                (), float("inf"), dtype=torch.float64, device=gradient.device
            )
        norms.append(torch.linalg.vector_norm(gradient.to(torch.float64)))
    scaled_norm = torch.linalg.vector_norm(torch.stack(norms))
    if torch.isfinite(scaled_norm):
        scale = torch.as_tensor(
            gradient_scale, dtype=torch.float64, device=scaled_norm.device
        )
        coefficient = torch.minimum(
            scale.reciprocal(),
            torch.as_tensor(max_norm, dtype=torch.float64, device=scaled_norm.device)
            / (scaled_norm + scale * 1.0e-12),
        )
        for parameter in parameters:
            parameter.grad.mul_(coefficient.to(parameter.grad.dtype))
    return scaled_norm / gradient_scale


class DiscountedReturnNormalizer:
    """Normalize returns with a fixed scale per rollout segment."""

    def __init__(
        self,
        batch_size: int,
        gamma: float,
        clip: float,
        *,
        device: torch.device,
        dtype: torch.dtype = torch.float64,
        epsilon: float = 1.0e-4,
    ):
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("return normalizer dtype must be float32 or float64")
        self.dtype = dtype
        self.gamma = float(gamma)
        self.clip = float(clip)
        self.discounted = torch.zeros(batch_size, dtype=dtype, device=device)
        self.mean = torch.zeros((), dtype=dtype, device=device)
        self.variance = torch.ones((), dtype=dtype, device=device)
        self.count = torch.as_tensor(epsilon, dtype=dtype, device=device)

    @torch.no_grad()
    def begin_segment(self, like: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(self.variance + 1.0e-6).to(
            device=like.device, dtype=like.dtype
        )

    @torch.no_grad()
    def _update(self, values: torch.Tensor) -> None:
        values = values.detach().to(dtype=self.dtype)
        batch_count = torch.as_tensor(
            values.numel(), dtype=self.dtype, device=values.device
        )
        batch_mean = values.mean()
        batch_variance = values.var(unbiased=False)
        delta = batch_mean - self.mean
        total = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total
        m_a = self.variance * self.count
        m_b = batch_variance * batch_count
        m2 = m_a + m_b + delta.square() * self.count * batch_count / total
        self.mean = new_mean
        self.variance = m2 / total
        self.count = total

    def normalize(
        self,
        reward: torch.Tensor,
        done: torch.Tensor,
        frozen_scale: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            self.discounted.mul_(self.gamma).add_(reward.detach().to(dtype=self.dtype))
            self._update(self.discounted)
            self.discounted.masked_fill_(done, 0.0)
        normalized = reward / frozen_scale
        return torch.clamp(normalized, -self.clip, self.clip)


@dataclass
class SHACRollout:
    actor_loss: torch.Tensor
    observations: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    next_values: torch.Tensor
    metrics: dict[str, float]


def td_lambda_targets(
    rewards: torch.Tensor,
    next_values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
    td_lambda: float,
) -> torch.Tensor:
    """Compute TD(lambda) targets with rollout and timeout bootstraps.

    Set ``next_values`` to zero for true terminal states.
    """
    if rewards.shape != next_values.shape or rewards.shape != dones.shape:
        raise ValueError("TD(lambda) inputs must have identical [time, env] shapes")
    boundary = dones.to(dtype=torch.bool).clone()
    boundary[-1] = True
    targets = torch.empty_like(rewards)
    continuation = rewards[-1] + gamma * next_values[-1]
    targets[-1] = continuation
    for step in range(rewards.shape[0] - 2, -1, -1):
        one_step = rewards[step] + gamma * next_values[step]
        lambda_step = rewards[step] + gamma * (
            (1.0 - td_lambda) * next_values[step] + td_lambda * targets[step + 1]
        )
        targets[step] = torch.where(boundary[step], one_step, lambda_step)
    return targets
