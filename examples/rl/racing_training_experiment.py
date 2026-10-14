"""Compare MPC and direct policies trained with SHAC and PPO.

This code uses synthetic data and parameters, and does not represent any manufacturer's product.
"""

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
from copy import deepcopy
from dataclasses import fields
from itertools import product
from pathlib import Path


def configure_jax():
    """Configure precision, caching, and GPU settings before JAX initializes."""
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    flags = os.environ.get("XLA_FLAGS", "").split()
    names = {flag.partition("=")[0] for flag in flags}
    for flag in (
        "--xla_gpu_autotune_level=0",
        "--xla_gpu_deterministic_ops=true",
        "--xla_gpu_enable_command_buffer=",
    ):
        if flag.partition("=")[0] not in names:
            flags.append(flag)
    os.environ["XLA_FLAGS"] = " ".join(flags)
    import jax

    jax.config.update("jax_enable_x64", True)
    jax.config.update(
        "jax_compilation_cache_dir",
        os.environ.get(
            "JAX_COMPILATION_CACHE_DIR",
            str(Path.home() / ".cache/turbompc/rl" / jax.__version__),
        ),
    )


def smoke_config(config):
    config = deepcopy(config)
    config["env"]["horizon"] = 3
    config["training"]["batch_size"] = 2
    settings = config["training"][config["algorithm"]]
    settings.update(total_updates=1, eval_every=1, checkpoint_every=1)
    if config["algorithm"] == "shac":
        settings.update(training_horizon=2, critic_iterations=1, critic_minibatches=1)
    else:
        settings.update(num_steps=2, num_minibatches=1, update_epochs=1)
        if "critic_iterations" in settings:
            settings["critic_iterations"] = 1
    config["evaluation"] = {"num_envs": 1, "steps": 2}
    if config["policy"] == "mpc":
        config["solver"]["admm"]["max_iter"] = 50
    return config


def build(config):
    configure_jax()
    import numpy as np
    import torch

    from .env.racing.env import RacingEnv
    from .env.racing.runtime import RacingRuntime
    from .ppo.ppo import PPO, PPOConfig
    from .shac.shac import SHAC, SHACConfig
    from .util.networks import Actor, Critic
    from .util.training import seed_everything

    name, policy = config["algorithm"], config["policy"]
    if name not in {"shac", "ppo"} or policy not in {"direct", "mpc"}:
        raise ValueError("choose algorithm shac/ppo and policy direct/mpc")
    training = config["training"]
    values = dict(training[name])
    values.pop("eval_every", None)
    values.pop("checkpoint_every", None)
    values["num_envs"] = training["batch_size"]
    config_type, algorithm_type = (
        (SHACConfig, SHAC) if name == "shac" else (PPOConfig, PPO)
    )
    unknown = values.keys() - {field.name for field in fields(config_type)}
    if unknown:
        raise ValueError(f"unknown {name} configuration keys: {sorted(unknown)}")
    hyperparameters = config_type(**values)
    env = RacingEnv(config)
    problem = backend = None
    if policy == "mpc":
        from .env.racing.dynamics import VEHICLE_PARAMETERS
        from .env.racing.problem import RacingProblem
        from .util.mpc import MPCBackend

        problem = RacingProblem(
            config, track=env.track, state_dot_params=VEHICLE_PARAMETERS
        )
        backend = MPCBackend(config, problem)
    runtime = RacingRuntime(config, env, problem, backend)
    seed_everything(hyperparameters.seed)
    sigma = training["actor"]["initial_sigma"]
    if policy == "direct":
        # Direct PPO historically begins with FP32 physical exploration values.
        sigma = np.asarray(sigma, dtype=np.float32 if name == "ppo" else np.float64)
        sigma = runtime.policy_sigma_from_physical(sigma)
    paper_initialization = policy == "mpc" and training.get(
        "paper_initialization", False
    )
    actor = Actor(
        runtime.observation_dim,
        34 if paper_initialization else runtime.policy_parameter_dim,
        runtime.action_dim,
        training["actor"],
        dtype=runtime.dtype,
        bounded=policy == "direct" and name == "shac",
        initial_sigma=sigma,
        # Keep the paper's complete head so its initialization and matrix
        # products are unchanged; only quadratic costs enter the current OCP.
        output_indices=(
            [*range(7), *range(14, 17), *range(20, 27)]
            if paper_initialization
            else None
        ),
    )
    if name == "shac" or policy == "mpc":
        actor.recenter(
            runtime.actor_anchor_observation(), runtime.initial_policy_parameters()
        )
    critic = Critic(runtime.observation_dim, training["critic"], dtype=runtime.dtype)
    actor, critic = actor.to(runtime.device), critic.to(runtime.device)
    algorithm = algorithm_type(runtime, actor, critic, hyperparameters)
    if name == "ppo" and policy == "direct":
        with torch.no_grad():
            actor.log_std.copy_(
                torch.as_tensor(sigma, device=runtime.device, dtype=runtime.dtype).log()
            )
        actor.recenter(
            algorithm.next_obs[0].detach().cpu().numpy(),
            runtime.initial_policy_parameters(),
        )
    return algorithm


METHODS = {
    "shac_mpc": ("MPC SHAC", "#009e73"),
    "ppo_mpc": ("MPC PPO", "#0072b2"),
    "shac_direct": ("Direct SHAC", "#e69f00"),
    "ppo_direct": ("Direct PPO", "#d55e00"),
}
ROOT = Path(__file__).resolve().parents[2]


def experiment_config(method, seed, *, device="cuda", smoke=False, updates=None):
    import yaml

    path = Path(__file__).parent / "configs" / f"{method}.yaml"
    config = yaml.safe_load(path.read_text())
    # Experiment B allows five off-track steps before terminating an episode.
    config["env"]["off_track_grace_steps"] = 5
    if smoke:
        config = smoke_config(config)
    config["training"]["device"] = device
    if config["policy"] == "mpc":
        config["training"]["paper_initialization"] = True
    settings = config["training"][config["algorithm"]]
    settings["seed"] = seed
    if updates is not None:
        if updates < 1:
            raise ValueError("updates must be positive")
        settings["total_updates"] = updates
    if device == "cpu" and config["policy"] == "mpc":
        config["solver"].update(
            forward_backend="admm_jax_loop_jax_dense",
            backward_backend="direct_jax_dense",
        )
    return config


def config_digest(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def records(directory):
    return [
        json.loads(line)
        for line in (directory / "metrics.jsonl").read_text().splitlines()
        if line.strip()
    ]


def complete_evaluation(metrics, steps):
    return (
        metrics.get("eval/return_mean") is not None
        and metrics.get("eval/valid_fraction") == 1.0
        and metrics.get("eval/episodic_length_mean", 0) >= steps
    )


def save_tracking_errors(runtime, trace, path):
    """Save the per-step errors used in the lower panels of Fig. 5."""
    import numpy as np

    states = np.stack([frame["state"] for frame in trace])
    reference_speed = np.asarray(runtime.env.track.interpolate(states[..., -1], "V"))
    np.savez_compressed(
        path,
        lateral=np.abs(states[..., 4]).ravel(),
        speed=np.abs(states[..., 1] - reference_speed).ravel(),
    )


def run_training(method, seed, config, directory):
    """Train a policy and evaluate its final and selected checkpoints."""
    from .env.racing.plot import plot_rollout, plot_training
    from .util.training import evaluate, load_checkpoint, save_checkpoint, train

    learner = build(config)
    import jax
    import torch

    directory.mkdir(parents=True, exist_ok=True)
    save_checkpoint(learner, directory / "policy_initial.pt")
    initial_metrics = evaluate(learner, **config["evaluation"])
    train(learner, config, directory)
    plot_training(directory)
    history = records(directory)
    final_metrics, final_trace = evaluate(learner, **config["evaluation"], record=True)
    save_tracking_errors(
        learner.runtime, final_trace, directory / "final_tracking_errors.npz"
    )
    nominal_metrics = None
    if method == "shac_mpc":
        nominal_metrics, nominal_trace = evaluate(
            learner,
            **config["evaluation"],
            record=True,
            fixed_parameters=learner.runtime.initial_policy_parameters(),
        )
        save_tracking_errors(
            learner.runtime, nominal_trace, directory / "nominal_tracking_errors.npz"
        )
    candidates = [
        row
        for row in history
        if complete_evaluation(row, config["evaluation"]["steps"])
    ]
    if complete_evaluation(final_metrics, config["evaluation"]["steps"]):
        candidates.append({"update": learner.update_index, **final_metrics})
    selected = (
        max(candidates, key=lambda row: row["eval/return_mean"])
        if candidates
        else {"update": learner.update_index}
    )
    update = int(selected["update"])
    checkpoint = directory / (
        "policy_final.pt"
        if update == learner.update_index
        else f"checkpoint_{update:04d}.pt"
    )
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Selected evaluation has no saved checkpoint: {checkpoint}"
        )
    if update == learner.update_index:
        selected_metrics, trace = final_metrics, final_trace
    else:
        load_checkpoint(learner, checkpoint)
        selected_metrics, trace = evaluate(learner, **config["evaluation"], record=True)
    shutil.copy2(checkpoint, directory / "selected_policy.pt")
    plot_rollout(learner.runtime, trace, directory)
    settings = config["training"][config["algorithm"]]
    samples_per_update = config["training"]["batch_size"] * settings.get(
        "training_horizon", settings.get("num_steps")
    )
    timed = history[1:]  # Match the reference's exclusion of update 1/JIT warm-up.
    training_seconds = sum(
        row["time/collect_seconds"] + row["time/train_seconds"] for row in timed
    )
    summary = {
        "method": method,
        "seed": seed,
        "config_digest": config_digest(config),
        "updates": len(history),
        "transitions_per_update": samples_per_update,
        "transitions": len(history) * samples_per_update,
        "training_seconds_excluding_update1_and_evaluation": training_seconds,
        "seconds_per_100k_transitions": (
            training_seconds * 100000 / (len(timed) * samples_per_update)
            if timed
            else None
        ),
        "selected_update": update,
        "selected_complete_evaluation": bool(candidates),
        "selection": (
            "best_complete_evaluation" if candidates else "final_no_complete_evaluation"
        ),
        "final_evaluation": final_metrics,
        "initial_evaluation": initial_metrics,
        "nominal_evaluation": nominal_metrics,
        "final_complete_evaluation": complete_evaluation(
            final_metrics, config["evaluation"]["steps"]
        ),
        "selected_evaluation": selected_metrics,
        "device": str(learner.device),
        "gpu_name": (
            torch.cuda.get_device_name(learner.device)
            if learner.device.type == "cuda"
            else None
        ),
        "torch_version": torch.__version__,
        "jax_version": jax.__version__,
        "experiment_source_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
    }
    (directory / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False)
    )
    return summary


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def learning_curve(history, summary, baseline_reward):
    """Use mean step reward, matching the paper's return normalization."""
    elapsed, curve = 0.0, {}
    if summary.get("initial_evaluation") is not None:
        curve[0] = dict(
            method=summary["method"],
            seed=summary["seed"],
            update=0,
            transitions=0,
            training_seconds=0.0,
            **{"return": baseline_reward},
        )
    final_update = summary["updates"]
    for row in history:
        update = row["update"]
        if update > 1:
            elapsed += row["time/collect_seconds"] + row["time/train_seconds"]
        reward = row.get("eval/reward_mean")
        if update == final_update and reward is None:
            reward = summary["final_evaluation"]["eval/reward_mean"]
        if reward is not None:
            curve[update] = dict(
                method=summary["method"],
                seed=summary["seed"],
                update=update,
                transitions=update * summary["transitions_per_update"],
                training_seconds=elapsed,
                **{"return": reward},
            )
    return curve


def paper_tracking_errors(directory, *, nominal=False):
    """Recover pre-control errors from the saved selected-policy rollout."""
    import numpy as np
    import yaml

    summary = json.loads((directory / "summary.json").read_text())
    if not nominal and not summary["selected_complete_evaluation"]:
        raise ValueError(f"No complete valid checkpoint at {directory}")
    config = yaml.safe_load((directory / "effective_config.yaml").read_text())
    with np.load(directory / "rollout.npz") as saved:
        states = saved["state"][:, 0]
    expected = config["evaluation"]["steps"]
    if states.shape != (expected, 7) or not np.isfinite(states).all():
        raise ValueError(f"Expected {expected} finite states at {directory}")
    table = np.genfromtxt(
        ROOT / "examples/racing_trajectory.csv", delimiter=",", names=True
    )
    if nominal:
        with np.load(directory / "nominal_tracking_errors.npz") as saved:
            errors = {key: saved[key] for key in ("lateral", "speed")}
    else:
        errors = {
            "lateral": np.abs(states[:, 4]),
            "speed": np.abs(
                states[:, 1] - np.interp(states[:, -1], table["s"], table["V"])
            ),
        }
    # Recorded states are x_1,...,x_T; the paper plots x_0,...,x_{T-1}.
    start = config["env"]["initial_s"]
    initial_s = np.asarray(start, dtype=states.dtype)
    initial_v = np.asarray(np.interp(start, table["s"], table["V"]), dtype=states.dtype)
    initial_error = {
        "lateral": 0.0,
        "speed": abs(initial_v - np.interp(initial_s, table["s"], table["V"])),
    }
    for key, values in errors.items():
        if values.shape != (expected,) or not np.isfinite(values).all():
            raise ValueError(f"Incomplete {key} errors at {directory}")
        errors[key] = np.concatenate(([initial_error[key]], values[:-1]))
    name = "paper_nominal_errors.npz" if nominal else "paper_selected_errors.npz"
    np.savez_compressed(directory / name, **errors)
    return errors


def comparison_row(summary):
    selected = summary["selected_evaluation"]
    keys = (
        "method",
        "seed",
        "gpu_name",
        "updates",
        "transitions",
        "training_seconds_excluding_update1_and_evaluation",
        "seconds_per_100k_transitions",
        "selected_update",
        "selection",
    )
    return {
        **{key: summary[key] for key in keys},
        "final_return": summary["final_evaluation"]["eval/return_mean"],
        "selected_return": selected["eval/return_mean"],
        "complete_evaluation": summary["selected_complete_evaluation"],
        "final_complete_evaluation": summary["final_complete_evaluation"],
        "evaluation_steps": selected["eval/episodic_length_mean"],
        "mean_abs_lateral_m": selected.get("eval/abs_lateral_deviation_mean"),
        "mean_abs_speed_m_s": selected.get("eval/abs_velocity_deviation_mean"),
    }


def summarize_experiment(output, methods, seeds):
    """Plot the paper's normalized rewards and selected-checkpoint errors."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    summaries, curves = [], {method: [] for method in methods}
    for method in methods:
        for seed in seeds:
            directory = output / method / f"seed_{seed}"
            summary = json.loads((directory / "summary.json").read_text())
            summaries.append(summary)
    # The paper uses one shared zero-transition baseline per policy class.
    baselines = {}
    for summary in sorted(
        summaries, key=lambda row: (not row["method"].startswith("shac"), row["seed"])
    ):
        baselines.setdefault(
            summary["method"].split("_")[1],
            summary["initial_evaluation"]["eval/reward_mean"],
        )
    for summary in summaries:
        method, seed = summary["method"], summary["seed"]
        curves[method].append(
            learning_curve(
                records(output / method / f"seed_{seed}"),
                summary,
                baselines[method.split("_")[1]],
            )
        )
    rows = [
        row for group in curves.values() for curve in group for row in curve.values()
    ]
    minimum = min(row["return"] for row in rows)
    for row in rows:
        row["normalized_return"] = (
            (row["return"] - minimum) / -minimum if minimum < 0 else 1.0
        )
    write_csv(output / "training_curves.csv", rows)

    figure, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    fig5 = plt.figure(figsize=(10, 7), layout="constrained")
    grid = fig5.add_gridspec(2, 2)
    return_axis = fig5.add_subplot(grid[0, :])
    error_axes = [fig5.add_subplot(grid[1, column]) for column in range(2)]
    rng = np.random.default_rng(0)
    for method, group in curves.items():
        updates = sorted(group[0])
        if any(sorted(curve) != updates for curve in group[1:]):
            raise ValueError(f"Evaluation updates differ across {method} seeds")
        values = np.array(
            [[curve[u]["normalized_return"] for u in updates] for curve in group]
        )
        # Resample whole seed curves to retain their temporal dependence.
        # Five seeds have only 5^5=3125 resamples: enumerate them exactly.
        draws = (
            np.asarray(list(product(range(len(group)), repeat=len(group))))
            if len(group) <= 5
            else rng.integers(len(group), size=(2000, len(group)))
        )
        low, high = np.quantile(values[draws].mean(axis=1), [0.025, 0.975], axis=0)
        label, color = METHODS[method]
        for axis, key, scale in (
            (axes[0], "transitions", 1e6),
            (axes[1], "training_seconds", 60),
            (return_axis, "transitions", 1e6),
        ):
            x = (
                np.mean([[curve[u][key] for u in updates] for curve in group], axis=0)
                / scale
            )
            axis.plot(
                x,
                values.mean(axis=0),
                label=label,
                color=color,
                marker="o",
                markersize=3,
            )
            if len(group) > 1:
                axis.fill_between(x, low, high, color=color, alpha=0.18)
    labels = (
        "Environment transitions (millions)",
        "Mean training time (minutes)\n(update 1 and evaluation excluded)",
    )
    for axis, label in zip(axes, labels):
        axis.set(
            xlabel=label, ylabel="Normalized evaluation return", ylim=(-0.03, 1.03)
        )
        axis.grid(alpha=0.2)
        axis.legend()
    for extension in ("png", "pdf"):
        figure.savefig(output / f"training_comparison.{extension}", dpi=180)
    plt.close(figure)
    return_axis.set(
        xlabel=labels[0], ylabel="Normalized evaluation return", ylim=(-0.03, 1.03)
    )
    return_axis.grid(alpha=0.2)
    return_axis.legend(ncol=2)
    groups = [
        (
            METHODS[method][0],
            METHODS[method][1],
            [
                (seed, paper_tracking_errors(output / method / f"seed_{seed}"))
                for seed in seeds
            ],
        )
        for method in ("shac_mpc", "shac_direct", "ppo_direct", "ppo_mpc")
        if method in methods
    ]
    if "shac_mpc" in methods:
        groups.append(
            (
                "MPC Nominal",
                "#999999",
                [
                    (
                        seeds[0],
                        paper_tracking_errors(
                            output / "shac_mpc" / f"seed_{seeds[0]}", nominal=True
                        ),
                    )
                ],
            )
        )
    error_rows = []
    for axis, key, label in zip(
        error_axes,
        ("lateral", "speed"),
        ("Absolute lateral error (m)", "Absolute speed error (m/s)"),
    ):
        for position, (name, color, trials) in enumerate(groups, 1):
            values = []
            for seed, errors in trials:
                sample = errors[key]
                values.append(sample)
                error_rows.append(
                    dict(
                        method=name,
                        seed=seed,
                        error=key,
                        samples=len(sample),
                        mean=float(sample.mean()),
                        q25=float(np.quantile(sample, 0.25)),
                        median=float(np.median(sample)),
                        q75=float(np.quantile(sample, 0.75)),
                    )
                )
            values = np.concatenate(values)
            parts = axis.violinplot(
                [values],
                positions=[position],
                showextrema=False,
                quantiles=[[0.25, 0.5, 0.75]],
            )
            parts["bodies"][0].set(facecolor=color, edgecolor="black", alpha=0.65)
            parts["cquantiles"].set(color="black", linewidth=1)
            axis.scatter(
                position,
                values.mean(),
                marker="D",
                facecolor="white",
                edgecolor="black",
                zorder=3,
            )
        axis.set(
            xticks=range(1, len(groups) + 1),
            xticklabels=[name.replace(" ", "\n") for name, _, _ in groups],
            ylabel=label,
            ylim=(0, None),
        )
        axis.grid(axis="y", alpha=0.2)
    for extension in ("png", "pdf"):
        fig5.savefig(output / f"fig5.{extension}", dpi=180)
    plt.close(fig5)
    write_csv(output / "tracking_errors.csv", error_rows)
    comparisons = [comparison_row(summary) for summary in summaries]
    write_csv(output / "comparison.csv", comparisons)
    report = {
        "methods": methods,
        "seeds": seeds,
        "normalization": {
            "metric": "eval/reward_mean",
            "minimum_observed_return": minimum,
            "ideal_return": 0.0,
        },
        "zero_transition_baselines": baselines,
        "confidence_interval": (
            "95% percentile bootstrap of seed curves; exact enumeration for up to five"
            " seeds"
        ),
        "time_axis": (
            "Mean cumulative training time at matched evaluation updates within each"
            " method"
        ),
        "tracking_errors_checkpoint": (
            "best complete evaluation; pre-control states pooled across seeds"
        ),
        "nominal_policy": "fixed MPC cost weights, without a neural cost map",
        "postprocessing_source_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "runs": comparisons,
    }
    (output / "comparison.json").write_text(
        json.dumps(report, indent=2, allow_nan=False)
    )


def run_trial(args, method, seed):
    """Reuse a completed run or start a training process."""
    directory = args.output / method / f"seed_{seed}"
    config = experiment_config(
        method, seed, device=args.device, smoke=args.smoke, updates=args.updates
    )
    if args.worker:
        run_training(method, seed, config, directory)
        return
    summary = directory / "summary.json"
    if summary.exists():
        if json.loads(summary.read_text())["config_digest"] != config_digest(config):
            raise ValueError(f"Existing run uses different settings: {directory}")
        print(f"Reusing {method}, seed {seed}", flush=True)
        return
    if (directory / "metrics.jsonl").exists():
        raise FileExistsError(
            f"Incomplete run at {directory}; choose a fresh --output directory"
        )
    directory.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-u",
        "-m",
        "examples.rl.racing_training_experiment",
        "--worker",
        "--methods",
        method,
        "--seeds",
        str(seed),
        "--device",
        args.device,
        "--output",
        str(args.output),
    ]
    if args.smoke:
        command.append("--smoke")
    if args.updates is not None:
        command += ["--updates", str(args.updates)]
    environment = {
        **os.environ,
        "JAX_PLATFORMS": args.device,
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
    }
    print(
        f"Training {METHODS[method][0]}, seed {seed}: {directory / 'training.log'}",
        flush=True,
    )
    with (directory / "training.log").open("w") as stream:
        subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=True,
        )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--methods", nargs="+", choices=tuple(METHODS), default=list(METHODS)
    )
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument(
        "--output", type=Path, default=Path("results/rl/racing_training_experiment")
    )
    parser.add_argument(
        "--updates", type=int, help="Override all methods' preset budgets"
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--plots-only", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    methods = list(dict.fromkeys(args.methods))
    seeds = list(
        dict.fromkeys(
            args.seeds if args.seeds is not None else ([0] if args.smoke else range(5))
        )
    )
    if any(seed < 0 for seed in seeds):
        parser.error("seeds must be nonnegative")
    if args.worker and (len(methods) != 1 or len(seeds) != 1):
        parser.error("a worker runs exactly one method and seed")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    if not args.plots_only:
        for method in methods:
            for seed in seeds:
                run_trial(args, method, seed)
    if not args.worker:
        summarize_experiment(args.output, methods, seeds)
        print(f"Experiment B artifacts: {args.output}")


if __name__ == "__main__":
    main()
