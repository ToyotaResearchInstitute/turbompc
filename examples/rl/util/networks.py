"""Actor and critic networks for SHAC and PPO."""

from itertools import pairwise

import numpy as np
import torch
from torch import nn


def mlp(
    input_dim,
    hidden_sizes,
    output_dim,
    activation="tanh",
    output_gain=1.0,
    *,
    dtype=None,
):
    activation_type = {"relu": nn.ReLU, "tanh": nn.Tanh, "elu": nn.ELU}[activation]
    sizes = [input_dim, *hidden_sizes, output_dim]
    layers = []
    for index, (before, after) in enumerate(pairwise(sizes)):
        layer = nn.Linear(before, after, dtype=dtype)
        gain = output_gain if index == len(sizes) - 2 else np.sqrt(2.0)
        if gain:
            nn.init.orthogonal_(layer.weight, gain)
        else:
            nn.init.zeros_(layer.weight)
        nn.init.zeros_(layer.bias)
        layers.append(layer)
        if index < len(sizes) - 2:
            layers.append(activation_type())
    return nn.Sequential(*layers)


class Actor(nn.Module):
    """MLP policy parameters with Gaussian exploration."""

    def __init__(
        self,
        input_dim,
        output_dim,
        action_dim,
        config,
        *,
        dtype,
        bounded=False,
        initial_sigma=None,
        output_indices=None,
    ):
        super().__init__()
        self.network = mlp(
            input_dim,
            config["hidden_sizes"],
            output_dim,
            config["activation"],
            config["output_gain"],
        ).to(dtype=dtype)
        sigma = np.asarray(
            initial_sigma if initial_sigma is not None else config["initial_sigma"],
            dtype=np.float64,
        )
        if sigma.shape != (action_dim,) or not np.all(np.isfinite(sigma) & (sigma > 0)):
            raise ValueError("initial_sigma must contain one positive value per action")
        self.log_std = nn.Parameter(
            torch.as_tensor(np.log(sigma), dtype=dtype),
            requires_grad=bool(config.get("learn_sigma", True)),
        )
        self.bounded = bounded
        self.register_buffer(
            "output_indices",
            None if output_indices is None else torch.as_tensor(output_indices),
            persistent=False,
        )

    def forward(self, observation):
        value = self.network(observation)
        if self.output_indices is not None:
            value = value.index_select(-1, self.output_indices)
        return torch.tanh(value) if self.bounded else value

    @torch.no_grad()
    def recenter(self, observation, initial_parameters):
        reference = self.network[-1].weight
        observation = torch.as_tensor(
            np.array(observation, copy=True),
            device=reference.device,
            dtype=reference.dtype,
        ).reshape(1, -1)
        target = torch.as_tensor(
            np.array(initial_parameters, copy=True),
            device=reference.device,
            dtype=reference.dtype,
        )
        if self.output_indices is not None:
            target = reference.new_zeros(reference.shape[0]).index_copy(
                0, self.output_indices, target
            )
        if self.bounded:
            if bool((target.abs() >= 1).any()):
                raise ValueError(
                    "initial bounded policy output must lie inside (-1, 1)"
                )
            target = torch.atanh(target)
        hidden = self.network[:-1](observation)
        self.network[-1].bias.copy_(target - hidden[0] @ self.network[-1].weight.T)


class Critic(nn.Module):
    def __init__(self, input_dim, config, *, dtype):
        super().__init__()
        self.network = mlp(
            input_dim,
            config["hidden_sizes"],
            1,
            config["activation"],
            config.get("output_gain", 1.0),
        ).to(dtype=dtype)

    def forward(self, observation):
        return self.network(observation).squeeze(-1)
