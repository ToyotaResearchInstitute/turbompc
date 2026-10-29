"""Solver termination statuses."""

import enum


class ADMMStatus(enum.IntEnum):
    """Termination for ADMM QP solver."""

    NOT_RUN = -1
    SOLVED = 0
    MAX_ITER = 1
    PRIMAL_INFEASIBLE = 2
    DUAL_INFEASIBLE = 3
    NONFINITE = 4


class SQPStatus(enum.IntEnum):
    """Termination for SQP solver."""

    NOT_RUN = -1
    SOLVED = 0
    MAX_ITER = 1
    QP_FAILED = 2
    NONFINITE = 3
