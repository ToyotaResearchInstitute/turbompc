"""Naming helpers for benchmark result directories and files."""

from __future__ import annotations


def _constraint_tag(constrained: bool) -> str:
    return "c" if constrained else "u"


def _init_tag(warm_start: bool) -> str:
    return "ws" if warm_start else "cs"


def turbompc_dirname(
    batch_size: int,
    horizon: int,
    dimensions: int,
    num_repeats: int,
    *,
    constrained: bool,
    warm_start: bool,
    pcg_eps: float,
    alpha: float,
    umax: float,
    tol: float,
    admm_max_iter: int,
    sim_steps: int = 1,
) -> str:
    """Keep a stable, concise dirname compatible with existing plot scripts."""
    return (
        f"turbompc_{batch_size}_{horizon}_{dimensions}_{num_repeats}_"
        f"{_constraint_tag(constrained)}_{_init_tag(warm_start)}_"
        f"pcg={pcg_eps}_alpha={alpha}_umax={umax}_"
        f"tol={tol}_admm={admm_max_iter}_steps={sim_steps}"
    )


def acados_dirname(
    batch_size: int,
    horizon: int,
    dimensions: int,
    num_repeats: int,
    *,
    constrained: bool,
    umax: float,
    tol: float,
    sim_steps: int = 1,
) -> str:
    return (
        f"acados_{batch_size}_{horizon}_{dimensions}_{num_repeats}_"
        f"{_constraint_tag(constrained)}_umax={umax}_tol={tol}_steps={sim_steps}"
    )


def mpcpytorch_dirname(
    batch_size: int,
    horizon: int,
    dimensions: int,
    num_repeats: int,
    *,
    constrained: bool,
    umax: float,
    tol: float,
) -> str:
    return (
        f"mpcpytorch_{batch_size}_{horizon}_{dimensions}_{num_repeats}_"
        f"{_constraint_tag(constrained)}_umax={umax}_tol={tol}"
    )


def device_name_for_file(device: object) -> str:
    """Normalize device labels used in saved numpy file names."""
    return str(device)


def acados_drone_dirname(
    batch_size: int,
    horizon: int,
    num_repeats: int,
    *,
    umax: float,
    init_mode: str = "warm",
    dt: float,
    n_substeps: int,
    nlp_iter: int,
    tol: float = 1e-3,
    seed: int,
) -> str:
    return (
        f"acados_drone_{batch_size}_{horizon}_{num_repeats}_c_umax={umax}_"
        f"{init_mode}_dt{dt}_subs{n_substeps}_nlp{nlp_iter}_tol{tol}_s{seed}"
    )


def turbompc_drone_dirname(
    batch_size: int,
    horizon: int,
    num_repeats: int,
    *,
    warm_start: bool,
    pcg_eps: float,
    alpha: float,
    umax: float,
    use_slack: bool,
    init_mode: str = "line",
    dt: float,
    scheme: str,
    sqp_iter: int,
    admm_max_iter: int,
    rd_weight: float = 0.01,
    admm_tol: float = 1e-3,
    seed: int,
) -> str:
    slack_tag = "slack" if use_slack else "noslack"
    return (
        f"turbompc_drone_{batch_size}_{horizon}_{num_repeats}_c_"
        f"{_init_tag(warm_start)}_{slack_tag}_"
        f"pcg={pcg_eps}_alpha={alpha}_umax={umax}_"
        f"{init_mode}_{scheme}_dt{dt}_sqp{sqp_iter}_admm{admm_max_iter}_"
        f"rd{rd_weight}_tol{admm_tol}_s{seed}"
    )
