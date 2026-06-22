from __future__ import annotations

import enum


class SchurSolverBackend(enum.IntEnum):
    """Backend for solving the block-tridiagonal Schur system."""

    IGNORED = -1
    PCG = 0
    PCG_FFI = 1
    CUDSS_FFI = 2
    JAX_DENSE = 3


class AdmmBackend(enum.IntEnum):
    """Backend for executing the ADMM loop."""

    JAX_LOOP = 0
    FUSED_PCG = 1
    FUSED_CUDSS = 2
