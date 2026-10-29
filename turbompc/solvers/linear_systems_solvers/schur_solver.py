"""Schur complement linear system solvers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

import jax.numpy as jnp
from jax import vmap

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

    @property
    def requires_preconditioner(self) -> bool:
        return self.backend in {
            SchurSolverBackend.PCG,
            SchurSolverBackend.PCG_FFI,
        }

    def solve(
        self,
        schur: SchurComplementMatrices,
        gammas: jnp.ndarray,
        zs_guess: jnp.ndarray,
        *,
        factorize=True,
    ) -> Tuple[jnp.ndarray, PCGDebugOutput]:
        """Solve one or a batch of Schur systems.

        ``schur.S`` has shape ``(T, nz, 3*nz)`` or ``(B, T, nz, 3*nz)``.
        ``gammas`` and ``zs_guess`` have shape ``(T, nz)`` or ``(B, T, nz)``.
        """
        del factorize
        if schur.S.ndim == 3:
            return self._solve_single(schur, gammas, zs_guess)
        if schur.S.ndim == 4:
            return vmap(self._solve_single)(schur, gammas, zs_guess)
        raise ValueError("Schur blocks must have rank 3 or 4")

    def _solve_single(
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

    def _solve_single(self, schur, gammas, zs_guess):
        return self._pcg.solve_linear_system(schur, gammas, zs_guess)


@dataclass(frozen=True)
class PcgFfiSchurSolver(SchurSystemSolver):
    """CUDA FFI PCG solve."""

    pcg_params: Dict[str, Any]

    def _solve_single(self, schur, gammas, zs_guess):
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

    _workspace: Any = field(default=None, init=False, repr=False, compare=False)

    def solve(self, schur, gammas, zs_guess, *, factorize=True):
        from turbompc.solvers.linear_systems_solvers.cudss_ffi_backend import (
            CudssBlockTridiWorkspace,
        )

        del zs_guess
        batch_size = 1 if schur.S.ndim == 3 else schur.S.shape[0]
        workspace = self._workspace
        if workspace is None:
            workspace = CudssBlockTridiWorkspace()
            object.__setattr__(self, "_workspace", workspace)
        sol = workspace.solve(schur.S, gammas, factorize=factorize)
        debug_shape = () if batch_size == 1 else (batch_size,)
        return sol, PCGDebugOutput(
            num_iterations=jnp.zeros(debug_shape, dtype=jnp.int32),
            convergence_eta=0.0,
        )

    def close(self) -> None:
        workspace = self._workspace
        object.__setattr__(self, "_workspace", None)
        if workspace is not None:
            workspace.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


@dataclass(frozen=True)
class JaxDenseSchurSolver(SchurSystemSolver):
    """Dense direct solve using `jnp.linalg.solve`."""

    def _solve_single(self, schur, gammas, zs_guess):
        zs = solve_block_tridi_system(
            schur.S, gammas, backend=SchurSolverBackend.JAX_DENSE
        )
        zs = jnp.asarray(zs)
        return zs, PCGDebugOutput(num_iterations=jnp.array(0), convergence_eta=0.0)


@dataclass(frozen=True)
class EigenSchurSolver(SchurSystemSolver):
    """Direct solve using Eigen block LDLT factorization."""

    _workspace: Any = field(default=None, init=False, repr=False, compare=False)

    def solve(self, schur, gammas, zs_guess, *, factorize=True):
        from turbompc.solvers.linear_systems_solvers.eigen_backend import (
            EigenBlockTridiWorkspace,
        )

        del zs_guess
        batch_size = 1 if schur.S.ndim == 3 else schur.S.shape[0]
        workspace = self._workspace
        if workspace is None:
            workspace = EigenBlockTridiWorkspace()
            object.__setattr__(self, "_workspace", workspace)
        zs = workspace.solve(schur.S, gammas, factorize=factorize)
        debug_shape = () if batch_size == 1 else (batch_size,)
        zeros = jnp.zeros(debug_shape, dtype=jnp.int32)
        return zs, PCGDebugOutput(num_iterations=zeros, convergence_eta=0.0)

    def close(self) -> None:
        """Release the native Eigen workspace owned by this solver."""
        workspace = self._workspace
        object.__setattr__(self, "_workspace", None)
        if workspace is not None:
            workspace.close()

    def __del__(self) -> None:
        """Release owned Eigen resources when Python destroys this solver."""
        try:
            self.close()
        except Exception:
            pass


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
    if backend == SchurSolverBackend.EIGEN:
        return EigenSchurSolver(
            horizon=horizon,
            num_states=num_states,
            num_controls=num_controls,
            backend=backend,
            name="EigenSchurSolver",
        )
    raise ValueError(f"Unsupported Schur solver backend: {backend}")
