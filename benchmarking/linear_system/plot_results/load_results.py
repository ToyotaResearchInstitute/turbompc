"""Load benchmark timing results into a tidy pandas DataFrame.

Expected directory layout:
    timing_results/
        acados/<config>/acados_<device>_{fwd,bwd}.npy
        fwd=<fb>_bwd=<bb>/<config>/turbompc_<device>_{fwd,bwd}.npy
        mpcpytorch_<config>/mpcpytorch_<device>_{fwd,bwd}.npy
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd


def _arch(device: str) -> str:
    return "cpu" if ("cpu" in device.lower() or "tfrt" in device.lower()) else "gpu"


def _parse_backend_dir(name: str) -> tuple[str, str] | None:
    if not name.startswith("fwd="):
        return None
    parts = name.split("_bwd=", 1)
    return (parts[0][4:], parts[1]) if len(parts) == 2 else None


def _parse_config_dir(dirname: str) -> dict | None:
    m = re.match(r"^(turbompc|acados|mpcpytorch)_", dirname)
    if not m:
        return None
    solver = m.group(1)

    umax_m = re.search(r"_umax=([0-9.e+\-]+)", dirname)
    if not umax_m:
        return None

    def _float(pat):
        x = re.search(pat, dirname)
        return float(x.group(1)) if x else None

    def _int(pat):
        x = re.search(pat, dirname)
        return int(x.group(1)) if x else None

    backbone = re.sub(r"_umax=[0-9.e+\-]+", "", dirname)
    backbone = re.sub(r"_alpha=[0-9.e+\-]+", "", backbone)
    backbone = re.sub(r"_pcg=[0-9.e+\-]+", "", backbone)
    parts = backbone.split("_")
    try:
        idx = 1
        batch_size = int(parts[idx])
        idx += 1
        horizon = int(parts[idx])
        idx += 1
        dimensions = int(parts[idx])
        idx += 1
        num_repeats = int(parts[idx])
        idx += 1
        constrained = parts[idx] == "c"
        idx += 1
    except (ValueError, IndexError):
        return None

    warm_start = None
    if idx < len(parts) and parts[idx] in ("ws", "cs"):
        warm_start = parts[idx] == "ws"

    return dict(
        solver=solver,
        batch_size=batch_size,
        horizon=horizon,
        dimensions=dimensions,
        num_repeats=num_repeats,
        constrained=constrained,
        warm_start=warm_start,
        alpha=_float(r"_alpha=([0-9.e+\-]+)"),
        pcg_eps=_float(r"_pcg=([0-9.e+\-]+)"),
        umax=float(umax_m.group(1)),
        tol=_float(r"_tol=([0-9.e+\-]+)"),
        admm_max_iter=_int(r"_admm=([0-9]+)"),
    )


def _entries(root, acados_root, mpcpytorch_root, turbompc_root):  # noqa: C901
    entries = []
    if root.exists():
        for d in root.iterdir():
            if not d.is_dir():
                continue
            if d.name == "acados" and acados_root is None:
                for sub in d.iterdir():
                    if sub.is_dir():
                        entries.append((None, None, sub))
            elif d.name.startswith("fwd=") and turbompc_root is None:
                parsed = _parse_backend_dir(d.name)
                if parsed:
                    fwd_b, bwd_b = parsed
                    for sub in d.iterdir():
                        if sub.is_dir():
                            entries.append((fwd_b, bwd_b, sub))
            elif d.name.startswith("mpcpytorch_") and mpcpytorch_root is None:
                entries.append((None, None, d))
    if acados_root is not None:
        for sub in acados_root.iterdir():
            if sub.is_dir():
                entries.append((None, None, sub))
    if mpcpytorch_root is not None:
        for sub in mpcpytorch_root.iterdir():
            if sub.is_dir() and sub.name.startswith("mpcpytorch_"):
                entries.append((None, None, sub))
    if turbompc_root is not None:
        fwd_b, bwd_b = None, None
        parsed = _parse_backend_dir(turbompc_root.name) or _parse_backend_dir(
            turbompc_root.parent.name
        )
        if parsed:
            fwd_b, bwd_b = parsed
        for sub in turbompc_root.iterdir():
            if sub.is_dir():
                entries.append((fwd_b, bwd_b, sub))
    return entries


def load_dataframe(
    results_root: str = "timing_results",
    acados_root: str | None = None,
    mpcpytorch_root: str | None = None,
    turbompc_root: str | None = None,
) -> pd.DataFrame:
    root = Path(results_root)
    acados_path = Path(acados_root) if acados_root else None
    mpcpytorch_path = Path(mpcpytorch_root) if mpcpytorch_root else None
    turbompc_path = Path(turbompc_root) if turbompc_root else None

    rows = []
    for fwd_backend, bwd_backend, problem_dir in _entries(
        root, acados_path, mpcpytorch_path, turbompc_path
    ):
        cfg = _parse_config_dir(problem_dir.name)
        if cfg is None:
            continue
        solver = cfg["solver"]
        for npy in problem_dir.glob("*.npy"):
            stem = npy.stem
            if not (stem.endswith("_fwd") or stem.endswith("_bwd")):
                continue
            pass_type = "fwd" if stem.endswith("_fwd") else "bwd"
            device_str = stem[len(solver) + 1 : -(len(pass_type) + 1)]
            try:
                data = np.load(npy)
            except Exception:
                continue
            for i, t in enumerate(data.ravel()):
                row = dict(cfg)
                row.update(
                    architecture=_arch(device_str),
                    device=device_str,
                    pass_type=pass_type,
                    time_s=float(t),
                    fwd_backend=fwd_backend,
                    bwd_backend=bwd_backend,
                    repeat_idx=i,
                )
                rows.append(row)

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["backend"] = df.apply(
        lambda r: (
            f"fwd={r.fwd_backend}_bwd={r.bwd_backend}"
            if pd.notna(r.fwd_backend)
            else None
        ),
        axis=1,
    )
    return df
