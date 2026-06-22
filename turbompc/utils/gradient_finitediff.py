"""Helper function to validate gradient computation via finite-differences."""
from typing import Any, Callable, Dict, Tuple

import jax.numpy as jnp
import numpy as np


def gradient_finite_diff(
    differentiable_function: Callable[..., jnp.ndarray | Tuple[jnp.ndarray, Any]],
    *const_args: Any,
    weights: Dict[str, jnp.ndarray],
    eps: float = 1e-5,
) -> Dict[str, np.ndarray]:
    """
    Central-difference gradient estimate w.r.t. `weights`.

    Args:
        differentiable_function: function evaluated as
            `differentiable_function(*const_args, weights)`
            Returns either:
                - values: array-like
                - (values, aux): tuple where values is array-like
            The scalar objective used for finite differences is `values.sum()`.
        const_args: constant arguments passed to `differentiable_function`
            Any PyTrees; held fixed (not perturbed).
        weights: parameters to perturb
            (key=string, value=jnp.ndarray)
        eps: finite-difference step size
            (float)

    Returns:
            grad: finite-difference gradients for each weight entry
            (key=string, value=np.ndarray) with the same shape as `weights[key]`.
    """
    grad: Dict[str, np.ndarray] = {}
    for k in weights:
        arr = np.array(weights[k])
        grad_k = np.zeros_like(arr)
        for i in range(grad_k.size):
            idx = np.unravel_index(i, grad_k.shape)
            arr_plus = arr.copy()
            arr_plus[idx] += eps
            arr_minus = arr.copy()
            arr_minus[idx] -= eps
            out_plus = differentiable_function(
                *const_args, {**weights, k: jnp.array(arr_plus)}
            )
            out_minus = differentiable_function(
                *const_args, {**weights, k: jnp.array(arr_minus)}
            )
            val_plus = (
                (out_plus[0] if isinstance(out_plus, tuple) else out_plus).sum().item()
            )
            val_minus = (
                (out_minus[0] if isinstance(out_minus, tuple) else out_minus)
                .sum()
                .item()
            )
            grad_k[idx] = (val_plus - val_minus) / (2 * eps)
        grad[k] = grad_k
    return grad
