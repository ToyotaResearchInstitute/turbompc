"""Compare GPU racing iteration times for MPC-SHAC, MPC-PPO, and MPC-TD3.

This code uses synthetic data and parameters, and does not represent any manufacturer's product.
"""

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORKER = "examples.rl.benchmarks.racing_iteration"
WORKLOAD = {
    "batch_size": 1024,
    "steps": 8,
    "mpc_horizon": 32,
    "warmups": 2,
    "trials": 5,
    "seed": 0,
}
METHODS = {"shac": "MPC + SHAC (FORL)", "ppo": "MPC + PPO", "td3": "MPC + TD3"}
PHASES = (
    ("collection_seconds", "Collection", "#94a3b8"),
    ("actor_seconds", "Actor update", "#087e8b"),
    ("critic_seconds", "Critic update", "#e9a03b"),
)


def measure(algorithm, output, rerun):
    """Reuse matching measurements or run a benchmark worker."""
    path = output / f"racing_{algorithm}.json"
    log = path.with_suffix(".log")
    expected = {
        "schema_version": 1,
        "algorithm": algorithm,
        "batch_size": WORKLOAD["batch_size"],
        "rollout_steps": WORKLOAD["steps"],
        "mpc_horizon": WORKLOAD["mpc_horizon"],
        "device": "cuda",
        "learning_rates_frozen": True,
    }
    if path.exists() and not rerun:
        result = json.loads(path.read_text())
        settings = result.get("config", {}).get("training", {})
        seed = settings.get("shac" if algorithm == "shac" else "ppo", {}).get("seed")
        if (
            all(result.get(key) == value for key, value in expected.items())
            and len(result.get("measurements", [])) == WORKLOAD["trials"]
            and len(result.get("warmups", [])) == WORKLOAD["warmups"]
            and seed == WORKLOAD["seed"]
        ):
            print(f"Reusing recorded result: {algorithm}")
            return result

    command = [sys.executable, "-u", "-m", WORKER, "--algorithm", algorithm]
    for key, value in WORKLOAD.items():
        command += ["--" + key.replace("_", "-"), str(value)]
    command += ["--output", str(path)]
    environment = {
        **os.environ,
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        "JAX_PLATFORMS": "cuda",
    }
    print(f"Running {algorithm}; log: {log}", flush=True)
    with log.open("w") as stream:
        process = subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if process.returncode:
        raise RuntimeError(f"{algorithm} failed; inspect {log}")
    result = json.loads(path.read_text())
    source = ROOT / (WORKER.replace(".", "/") + ".py")
    result.update(
        worker_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        command=command,
        log_path=str(log),
    )
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def report(results, output):
    """Export timing tables and a stacked phase plot."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    headers = [
        "Method",
        "Collection (s)",
        "Actor (s)",
        "Critic (s)",
        "Total (s)",
        "MPC forward rows",
        "MPC backward rows",
    ]
    rows = []
    for result in results:
        calls = result["measurements"][0]["mpc"]
        counts = [
            sum(v["rows"] for key, v in calls.items() if key.endswith("/" + direction))
            for direction in ("forward", "backward")
        ]
        rows.append(
            [
                METHODS[result["algorithm"]],
                *[f"{result['mean'][key]:.3f}" for key, _, _ in PHASES],
                (
                    f"{result['mean']['total_seconds']:.3f} ±"
                    f" {result['std']['total_seconds']:.3f}"
                ),
                *counts,
            ]
        )
    with (output / "racing_iteration_time.csv").open("w", newline="") as stream:
        csv.writer(stream).writerows([headers, *rows])
    print(f"GPU: {results[0]['gpu_name']}")
    for row in [headers, *rows]:
        print(" | ".join(map(str, row)))

    fig, ax = plt.subplots(figsize=(10, 3.8), layout="constrained")
    y = np.arange(len(results))
    left = np.zeros(len(results))
    for key, label, color in PHASES:
        values = np.array([row["mean"][key] for row in results])
        ax.barh(y, values, left=left, label=label, color=color)
        left += values
    std = np.array([row["std"]["total_seconds"] for row in results])
    ax.errorbar(left, y, xerr=std, fmt="none", color="black", capsize=4)
    for i, total in enumerate(left):
        ax.text(
            total + std[i] + max(left) * 0.025,
            i,
            f"{total:.3f} ± {std[i]:.3f} s ({total / left[0]:.1f}×)",
            va="center",
        )
    ax.set(
        yticks=y,
        yticklabels=[METHODS[r["algorithm"]] for r in results],
        xlabel="Wall-clock time per training iteration (s)",
        xlim=(0, max(left) * 1.45),
    )
    ax.invert_yaxis()
    ax.legend(loc="upper right")
    ax.grid(axis="x", alpha=0.2)
    fig.savefig(output / "racing_iteration_time.png", dpi=180)
    fig.savefig(output / "racing_iteration_time.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rerun", action="store_true", help="measure all methods again"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "results/rl/racing_training_benchmarks"
    )
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    results = [measure(method, output, args.rerun) for method in METHODS]
    report(results, output)
    print(f"Results: {output}")


if __name__ == "__main__":
    main()
