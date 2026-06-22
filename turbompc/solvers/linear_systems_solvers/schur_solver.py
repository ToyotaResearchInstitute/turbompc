"""Schur complement linear system solvers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import jax.numpy as jnp
from turbompc.solvers.linear_systems_solvers.backends import SchurSolverBackend
from turbompc.solvers.linear_systems_solvers.linear_solve import (
    solve_block_tridi_system,
)
from turbompc.solvers.linear_systems_solvers.pcg_primal import (
    PCGDebugOutput,
    PCGPrimalOptimalControl,
    SchurComplementMatrices,
)


@dataclass(frozen=True)
class SchurSystemSolver:
    """Base class for solvers of block-tridiagonal Schur systems."""

    horizon: int
    num_states: int
    num_controls: int
    backend: SchurSolverBackend
    name: str

    def solve(
        self,
        schur: SchurComplementMatrices,
        gammas: jnp.ndarray,
        zs_guess: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, PCGDebugOutput]:
        raise NotImplementedError


@dataclass(frozen=True)
class PcgIterativeSchurSolver(SchurSystemSolver):
    """Pure-JAX iterative PCG."""

    _pcg: PCGPrimalOptimalControl

    def solve(self, schur, gammas, zs_guess):
        return self._pcg.solve_linear_system(schur, gammas, zs_guess)


@dataclass(frozen=True)
class PcgFfiSchurSolver(SchurSystemSolver):
    """CUDA FFI PCG solve."""

    pcg_params: Dict[str, Any]

    def solve(self, schur, gammas, zs_guess):
        from turbompc.solvers.linear_systems_solvers.pcg_ffi_backend import (
            pcg_ffi_solve,
        )

        S = schur.S
        Phiinv = schur.preconditioner_Phiinv
        pcg_max_iter = int(self.pcg_params["max_iter"])
        pcg_epsilon = float(self.pcg_params["tol_epsilon"])
        sol, iters = pcg_ffi_solve(
            S, Phiinv, gammas, zs_guess, eps=pcg_epsilon, max_iters=pcg_max_iter
        )
        return sol, PCGDebugOutput(num_iterations=iters, convergence_eta=0.0)


@dataclass(frozen=True)
class CudssFfiSchurSolver(SchurSystemSolver):
    """CUDA FFI direct solve (cuDSS)."""

    def solve(self, schur, gammas, zs_guess):
        from turbompc.solvers.linear_systems_solvers.cudss_ffi_backend import (
            cudss_ffi_solve,
        )

        sol = cudss_ffi_solve(schur.S, gammas)
        return sol, PCGDebugOutput(num_iterations=jnp.array(0), convergence_eta=0.0)


@dataclass(frozen=True)
class JaxDenseSchurSolver(SchurSystemSolver):
    """Dense direct solve using `jnp.linalg.solve`."""

    def solve(self, schur, gammas, zs_guess):
        zs = solve_block_tridi_system(
            schur.S, gammas, backend=SchurSolverBackend.JAX_DENSE
        )
        zs = jnp.asarray(zs)
        return zs, PCGDebugOutput(num_iterations=jnp.array(0), convergence_eta=0.0)


def make_schur_solver(
    backend: SchurSolverBackend,
    horizon: int,
    num_states: int,
    num_controls: int,
    *,
    pcg_params: Dict[str, Any],
) -> SchurSystemSolver:
    """Factory returning a backend-specific Schur system solver."""
    if backend == SchurSolverBackend.IGNORED:
        raise ValueError(
            "SchurSolverBackend.IGNORED cannot be used to construct a solver."
        )
    if backend == SchurSolverBackend.PCG:
        pcg = PCGPrimalOptimalControl(horizon, num_states, num_controls, pcg_params)
        return PcgIterativeSchurSolver(
            horizon=horizon,
            num_states=num_states,
            num_controls=num_controls,
            backend=backend,
            name="PcgIterativeSchurSolver",
            _pcg=pcg,
        )
    if backend == SchurSolverBackend.PCG_FFI:
        return PcgFfiSchurSolver(
            horizon=horizon,
            num_states=num_states,
            num_controls=num_controls,
            backend=backend,
            name="PcgFfiSchurSolver",
            pcg_params=pcg_params,
        )
    if backend == SchurSolverBackend.CUDSS_FFI:
        return CudssFfiSchurSolver(
            horizon=horizon,
            num_states=num_states,
            num_controls=num_controls,
            backend=backend,
            name="CudssFfiSchurSolver",
        )
    if backend == SchurSolverBackend.JAX_DENSE:
        return JaxDenseSchurSolver(
            horizon=horizon,
            num_states=num_states,
            num_controls=num_controls,
            backend=backend,
            name="JaxDenseSchurSolver",
        )
    raise ValueError(f"Unsupported Schur solver backend: {backend}")
