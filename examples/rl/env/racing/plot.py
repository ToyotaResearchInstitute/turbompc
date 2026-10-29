"""Plot racing training curves and rollouts."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_training(output):
    rows = [
        json.loads(line)
        for line in (Path(output) / "metrics.jsonl").read_text().splitlines()
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), layout="constrained")
    for axis, key, label in zip(
        axes,
        ("rollout/reward_mean", "train/value_loss", "time/train_seconds"),
        ("Mean rollout reward", "Value loss", "Learning update (s)"),
    ):
        axis.plot([r["update"] for r in rows], [r.get(key) for r in rows])
        axis.set(xlabel="Update", ylabel=label)
        axis.grid(alpha=0.2)
    fig.savefig(Path(output) / "training.png", dpi=150)
    plt.close(fig)


def plot_rollout(runtime, trace, output):
    if not trace:
        return
    arrays = {key: np.stack([frame[key] for frame in trace]) for key in trace[0]}
    np.savez_compressed(Path(output) / "rollout.npz", **arrays)
    state = arrays["state"][:, 0]
    time = np.arange(state.shape[0]) * runtime.env.dt
    reference = np.asarray(runtime.env.track.table["V"])
    path = np.asarray(runtime.env.track.table["s"])
    position = state[:, -1]
    wrapped = np.where(
        position < runtime.lap_start + runtime.lap_length,
        position,
        runtime.lap_start
        + np.remainder(position - runtime.lap_start, runtime.lap_length),
    )
    speed_error = state[:, runtime.speed_index] - np.interp(wrapped, path, reference)
    fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True, layout="constrained")
    axes[0].plot(time, state[:, runtime.lateral_index])
    axes[0].set_ylabel("Lateral deviation (m)")
    axes[1].plot(time, speed_error)
    axes[1].set_ylabel("Speed error (m/s)")
    axes[2].plot(time, arrays["reward"][:, 0])
    axes[2].set(xlabel="Time (s)", ylabel="Reward")
    for axis in axes:
        axis.grid(alpha=0.2)
    fig.savefig(Path(output) / "rollout.png", dpi=150)
    plt.close(fig)
