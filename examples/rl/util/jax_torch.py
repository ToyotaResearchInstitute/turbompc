"""Torch/JAX tensor copies and MPC and simulation autograd bridges."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import torch
from torch import nn


def _ready(tree: Any) -> Any:
    """Wait for all arrays in a JAX pytree."""
    return jax.tree_util.tree_map(
        lambda value: (
            value.block_until_ready() if hasattr(value, "block_until_ready") else value
        ),
        tree,
    )


def torch_to_jax(value: Any) -> Any:
    """Copy a tensor or tensor pytree into JAX-owned memory."""
    tensors = jax.tree.map(lambda tensor: tensor.detach().contiguous(), value)

    def copy(tensor):
        try:
            array = jax.dlpack.from_dlpack(tensor)
        except (AssertionError, BufferError, RuntimeError, TypeError):
            array = jnp.asarray(tensor.cpu().numpy())
        return jnp.array(array, copy=True)

    # Keep every source alive until all copies finish.
    return _ready(jax.tree.map(copy, tensors))


def jax_to_torch(
    value: Any,
    *,
    like: torch.Tensor,
) -> Any:
    """Copy a JAX array or pytree, synchronizing the Torch copies together."""
    value = _ready(value)

    def copy(array):
        try:
            platforms = (
                {device.platform for device in array.devices()}
                if hasattr(array, "devices")
                else set()
            )
            matching_device = (
                not platforms
                or ("cpu" in platforms and like.device.type == "cpu")
                or (bool(platforms & {"gpu", "cuda"}) and like.device.type == "cuda")
            )
            if not matching_device:
                raise BufferError("JAX and Torch arrays use different devices")
            tensor = torch.utils.dlpack.from_dlpack(array)
        except (AssertionError, BufferError, RuntimeError, TypeError):
            tensor = torch.as_tensor(np.array(array, copy=True))
        if tensor.dtype != like.dtype or tensor.device != like.device:
            tensor = tensor.to(device=like.device, dtype=like.dtype)
        return tensor.clone()

    result = jax.tree.map(copy, value)
    # The JAX owners remain live until every Torch copy completes.
    if like.device.type == "cuda":
        torch.cuda.synchronize(like.device)
    return result


def _finite_torch_gradient(
    value: Any,
    *,
    like: torch.Tensor,
    callback: Any = None,
    name: str = "",
) -> torch.Tensor:
    """Report raw gradients, then replace non-finite entries with zero."""
    gradient = jax_to_torch(value, like=like)
    if callback is not None:
        callback(name, gradient)
    return torch.nan_to_num(gradient, nan=0.0, posinf=0.0, neginf=0.0)


class MPCFunction(torch.autograd.Function):
    """Differentiate MPC controls with respect to costs and context."""

    @staticmethod
    def forward(
        ctx,
        costs: torch.Tensor,
        context: torch.Tensor,
        bridge: MPC,
        seed: Any,
        history_seed: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mean, _, controls = _retain_mpc_forward(
            ctx, costs, context, bridge, seed, history_seed
        )
        return jax_to_torch((mean, controls), like=context)

    @staticmethod
    def backward(ctx, grad_mean, grad_controls):
        return _mpc_backward(ctx, grad_mean, grad_controls)


class MPCMeanFunction(torch.autograd.Function):
    """Expose the first MPC control with a retained adjoint."""

    @staticmethod
    def forward(
        ctx,
        costs: torch.Tensor,
        context: torch.Tensor,
        bridge: MPC,
        seed: Any,
        history_seed: Any,
    ) -> torch.Tensor:
        mean, _, _ = _retain_mpc_forward(
            ctx, costs, context, bridge, seed, history_seed
        )
        return jax_to_torch(mean, like=context)

    @staticmethod
    def backward(ctx, grad_mean):
        return _mpc_backward(ctx, grad_mean, None)


def _retain_mpc_forward(
    ctx,
    costs: torch.Tensor,
    context: torch.Tensor,
    bridge: MPC,
    seed: Any,
    history_seed: Any,
) -> tuple[Any, Any, Any]:
    """Solve MPC and retain its solution on the JAX device."""
    solve_kwargs = {"retain": True}
    if history_seed is not None:
        solve_kwargs["history_seed"] = history_seed
    costs_jax, context_jax = torch_to_jax((costs, context))
    result = bridge.backend.solve_device(
        costs_jax,
        context_jax,
        seed,
        **solve_kwargs,
    )
    bridge._last_result = result
    valid = jnp.asarray(result.valid, dtype=bool)
    mean = jnp.asarray(result.mean)
    states = jnp.asarray(result.states)
    controls = jnp.asarray(result.controls)
    ctx.backend = bridge.backend
    ctx.cache = result.cache
    ctx.valid = valid
    ctx.costs_like = costs
    ctx.context_like = context
    ctx.output_templates = (mean, states, controls)
    ctx.grad_callback = bridge._grad_callback
    return mean, states, controls


def _mpc_backward(ctx, grad_mean, grad_controls):
    """Apply the retained MPC adjoint to control gradients."""
    valid = jnp.asarray(ctx.valid, dtype=bool)
    mean_template, state_template, control_template = ctx.output_templates
    dmean = jnp.where(
        valid[:, None],
        (
            jnp.zeros_like(mean_template)
            if grad_mean is None
            else torch_to_jax(grad_mean)
        ),
        0.0,
    )
    dstates = jnp.zeros_like(state_template)
    dcontrols = jnp.where(
        valid[:, None, None],
        (
            jnp.zeros_like(control_template)
            if grad_controls is None
            else torch_to_jax(grad_controls)
        ),
        0.0,
    )
    # The solver's policy mean is the plan's first control node.
    dcontrols = dcontrols.at[:, 0].add(dmean)
    gradients = ctx.backend.vjp_plan_device(ctx.cache, dstates, dcontrols)
    dcosts = jnp.asarray(gradients["costs"])
    dcontext = jnp.asarray(gradients["context"])
    return (
        _finite_torch_gradient(
            dcosts,
            like=ctx.costs_like,
            callback=ctx.grad_callback,
            name="mpc_cost_adjoint",
        ),
        _finite_torch_gradient(
            dcontext,
            like=ctx.context_like,
            callback=ctx.grad_callback,
            name="mpc_context_adjoint",
        ),
        None,
        None,
        None,
    )


class MPC(nn.Module):
    """Torch MPC bridge with retained implicit gradients."""

    def __init__(self, backend: Any):
        super().__init__()
        if not bool(getattr(backend, "supports_plan_adjoint", False)):
            raise ValueError(
                "selected MPC backend does not expose the full-plan/context adjoint"
            )
        self.backend = backend
        self._last_result: Any | None = None
        self._grad_callback: Any = None

    def forward(
        self,
        costs,
        context,
        seed,
        *,
        history_seed=None,
        grad_callback=None,
    ):
        self._last_result = None
        self._grad_callback = grad_callback
        try:
            return MPCFunction.apply(
                costs,
                context,
                self,
                seed,
                history_seed,
            )
        finally:
            self._grad_callback = None

    def mean_only(
        self,
        costs,
        context,
        seed,
        *,
        history_seed=None,
        grad_callback=None,
    ):
        """Return the first MPC control and retain its adjoint."""
        self._last_result = None
        self._grad_callback = grad_callback
        try:
            return MPCMeanFunction.apply(
                costs,
                context,
                self,
                seed,
                history_seed,
            )
        finally:
            self._grad_callback = None

    def pop_result(self) -> Any:
        result = self._last_result
        if result is None:
            raise RuntimeError("no MPC solve result is available")
        self._last_result = None
        return result


class PlantStepFunction(torch.autograd.Function):
    """Differentiate a JAX plant transition from Torch."""

    @staticmethod
    def forward(
        ctx,
        state: torch.Tensor,
        controls: torch.Tensor,
        physics: Any,
        bridge: PlantStep,
        grad_callback: Any,
    ):
        state_jax, controls_jax = torch_to_jax((state, controls))
        outputs, discrete, applied_next = bridge.forward_device(
            state_jax, controls_jax, physics
        )
        ctx.bridge = bridge
        ctx.physics = physics
        ctx.save_for_backward(state, controls)
        ctx.output_specs = tuple(
            jax.ShapeDtypeStruct(value.shape, value.dtype) for value in outputs
        )
        ctx.grad_callback = grad_callback
        outputs, discrete, applied_next = jax_to_torch(
            (outputs, discrete, applied_next), like=state
        )
        return (
            *outputs,
            *(value.bool() for value in discrete),
            applied_next,
        )

    @staticmethod
    def backward(
        ctx,
        grad_state,
        grad_reward,
        grad_bound_violation_squared,
        *_discrete_and_applied_gradients,
    ):
        specs = ctx.output_specs

        def cotangent(value, spec):
            result = (
                jnp.zeros(spec.shape, dtype=spec.dtype)
                if value is None
                else jnp.asarray(value, dtype=spec.dtype)
            )
            return jnp.where(jnp.isfinite(result), result, 0.0)

        gradients = torch_to_jax(
            (grad_state, grad_reward, grad_bound_violation_squared)
        )
        cotangents = tuple(
            cotangent(value, spec) for value, spec in zip(gradients, specs)
        )
        state, controls = ctx.saved_tensors
        state_jax, controls_jax = torch_to_jax((state, controls))
        dstate, dcontrols = ctx.bridge.vjp_device(
            state_jax,
            controls_jax,
            cotangents,
            ctx.physics,
        )
        return (
            _finite_torch_gradient(
                dstate,
                like=state,
                callback=ctx.grad_callback,
                name="plant_state_adjoint",
            ),
            _finite_torch_gradient(
                dcontrols,
                like=controls,
                callback=ctx.grad_callback,
                name="plant_control_adjoint",
            ),
            None,
            None,
            None,
        )


class PlantStep(nn.Module):
    """Torch bridge for JAX plant transitions."""

    def __init__(self, transition: Callable):
        super().__init__()
        self.forward_device = jax.jit(transition)

        def vjp(state, controls, cotangents, physics=None):
            def differentiable_outputs(x, plan):
                return transition(x, plan, physics)[0]

            _, pullback = jax.vjp(differentiable_outputs, state, controls)
            return pullback(cotangents)

        self.vjp_device = jax.jit(vjp)

    def forward(
        self,
        state,
        controls,
        *,
        physics=None,
        grad_callback=None,
    ):
        return PlantStepFunction.apply(state, controls, physics, self, grad_callback)
