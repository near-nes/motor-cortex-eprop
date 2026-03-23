"""
Sweep M1 training parameters and evaluate performance.

Usage:
    python sweep_delay.py --delays 0 50 100 200 --learning-windows 500 550
    python sweep_delay.py --delays 0 50 100       # sweep delay only
    python sweep_delay.py --learning-windows 300 500 550  # sweep learning window only (delay=0)
"""

import argparse
import itertools
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import nest
import numpy as np
import structlog
from motor_controller_model.config_schema import MotorControllerConfig, TrainingTimings
from motor_controller_model.m1_factory import get_m1_or_train
from motor_controller_model.signals import generate_training_signals
from motor_controller_model.utils import install_nestml_module

_log = structlog.get_logger("sweep")


def build_targets(config):
    """Build target signals from config for inference MSE computation."""
    timings = TrainingTimings.from_config(config)
    training_cfg = config.training
    step_ms = config.simulation.step
    n_seq_steps = timings.n_timesteps_per_sequence

    targets = {"pos": [], "neg": []}
    for spec in training_cfg.trajectories:
        sig = generate_training_signals(spec, training_cfg, step_ms, config.task.input_shift_ms)
        for key, arr in [("pos", sig.target_rates_pos), ("neg", sig.target_rates_neg)]:
            targets[key].append(arr[:n_seq_steps])
    return targets


def run_inference_test(config, network, artifacts_dir, nest_module):
    """Run standalone inference, compute MSE against target, save plot."""
    nest.ResetKernel()
    nest.SetKernelStatus({"resolution": config.simulation.step})
    install_nestml_module(nest_module)

    training_cfg = config.training
    timings = TrainingTimings.from_config(config)
    step_ms = timings.step_ms
    n_trajectories = timings.n_samples
    n_steps_per_seq = timings.n_timesteps_per_sequence
    sim_time_ms = n_steps_per_seq * n_trajectories * step_ms

    network.build_network(simulation_time_ms=sim_time_ms)

    # Generate planner input signals
    all_signals = [
        generate_training_signals(spec, training_cfg, step_ms, config.task.input_shift_ms)
        for spec in training_cfg.trajectories
    ]

    full_traj = np.concatenate([sig.input_trajectory[:n_steps_per_seq] for sig in all_signals])
    sim_steps = len(full_traj)

    n_input = training_cfg.n_input_neurons

    planner_pos = nest.Create("tracking_neuron_nestml", n_input, {
        "kp": training_cfg.planner_kp, "base_rate": training_cfg.planner_base_rate,
    })
    nest.SetStatus(planner_pos, {
        "pos": True,
        "traj": full_traj.tolist(),
        "simulation_steps": sim_steps,
    })

    planner_neg = nest.Create("tracking_neuron_nestml", n_input, {
        "kp": training_cfg.planner_kp, "base_rate": training_cfg.planner_base_rate,
    })
    nest.SetStatus(planner_neg, {
        "pos": False,
        "traj": full_traj.tolist(),
        "simulation_steps": sim_steps,
    })

    network.connect(planner_pos + planner_neg)

    out_pos, out_neg = network.get_output_pops()

    mm_out = nest.Create("multimeter", {
        "record_from": ["readout_signal"],
        "interval": step_ms,
        "start": step_ms,
        "stop": sim_time_ms,
    })
    nest.Connect(mm_out, out_pos + out_neg)
    nest.Simulate(sim_time_ms)

    # Extract readout signals
    events = mm_out.get("events")
    idc_pos = events["senders"] == out_pos.tolist()[0]
    idc_neg = events["senders"] == out_neg.tolist()[0]
    readout_pos = events["readout_signal"][idc_pos]
    readout_neg = events["readout_signal"][idc_neg]

    # Compute MSE against shifted target
    targets = build_targets(config)
    mse_values = []
    for traj_idx in range(n_trajectories):
        t_start = traj_idx * n_steps_per_seq
        t_end = t_start + n_steps_per_seq
        for readout, key in [(readout_pos, "pos"), (readout_neg, "neg")]:
            r = readout[t_start:t_end]
            t = targets[key][traj_idx]
            n = min(len(r), len(t))
            mse_values.append(np.mean((r[:n] - t[:n]) ** 2))
    inference_mse = float(np.mean(mse_values))

    # Plot
    delay = config.task.input_shift_ms
    lw = config.task.learning_window_ms
    plt.figure(figsize=(10, 4))
    plt.plot(events["times"][idc_pos], readout_pos, label="Pos Readout", color="blue")
    plt.plot(events["times"][idc_neg], readout_neg, label="Neg Readout", color="red")
    total_seq_ms = n_steps_per_seq * step_ms
    for i in range(1, n_trajectories):
        plt.axvline(
            x=i * total_seq_ms, color="gray", linestyle="--", alpha=0.5,
            label="Trajectory Boundary" if i == 1 else "",
        )
    plt.title(
        f"Inference — delay={int(delay)}ms, learning_window={int(lw)}ms  (MSE={inference_mse:.1f})"
    )
    plt.xlabel("Time (ms)")
    plt.ylabel("Rate Signal")
    plt.legend()
    plt.tight_layout()
    plt.savefig(artifacts_dir / "standalone_inference_test.png")
    plt.close()

    return inference_mse


def compute_final_training_loss(run_dir, n_samples):
    """Average loss over the last training iteration (= last n_samples entries)."""
    loss = np.load(run_dir / "training_loss.npy")
    return float(np.mean(loss[-n_samples:]))


def build_run_configs(delays, learning_windows, n_iter):
    """Build (label, config) pairs for the cartesian product of sweep parameters."""
    runs = []
    for delay, lw in itertools.product(delays, learning_windows):
        config = MotorControllerConfig()
        config.task.input_shift_ms = delay
        config.task.learning_window_ms = lw
        config.task.n_iter = n_iter
        label = f"delay_{int(delay)}ms_lw_{int(lw)}ms"
        runs.append((label, config))
    return runs


def plot_summary(results, output_dir):
    """Plot sweep results on both axes: vs delay (grouped by ls) and vs ls (grouped by delay)."""
    metrics = [
        ("final_training_loss", "Final Training Loss"),
        ("inference_mse", "Inference MSE"),
    ]
    views = [
        ("input_shift_ms", "input_shift_ms (ms)", "learning_window_ms", "lw={v}ms"),
        ("learning_window_ms", "learning_window_ms (ms)", "input_shift_ms", "delay={v}ms"),
    ]

    fig, axes = plt.subplots(len(views), len(metrics), figsize=(7 * len(metrics), 5 * len(views)))

    for row, (x_key, x_label, group_key, group_fmt) in enumerate(views):
        groups = sorted(set(r[group_key] for r in results))
        for group_val in groups:
            subset = sorted(
                [r for r in results if r[group_key] == group_val],
                key=lambda r: r[x_key],
            )
            xs = [r[x_key] for r in subset]
            for col, (metric, ylabel) in enumerate(metrics):
                ax = axes[row, col]
                label = group_fmt.format(v=int(group_val)) if len(groups) > 1 else None
                ax.plot(xs, [r[metric] for r in subset], "o-", label=label)

        for col, (metric, ylabel) in enumerate(metrics):
            ax = axes[row, col]
            ax.set_xlabel(x_label)
            ax.set_ylabel(ylabel)
            ax.set_title(f"{ylabel} vs {x_label}")
            if len(groups) > 1:
                ax.legend()

    fig.tight_layout()
    fig.savefig(output_dir / "sweep_summary.png")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Sweep M1 training parameters")
    parser.add_argument(
        "--delays", type=float, nargs="+", default=[0, 50, 100, 150, 200, 300],
        help="input_shift_ms values to test (ms)",
    )
    parser.add_argument(
        "--learning-windows", type=float, nargs="+", default=[550],
        help="learning_window_ms values to test (ms)",
    )
    parser.add_argument(
        "--n-iter", type=int, default=100, help="Training iterations per run",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path(__file__).resolve().parent / "results_sweep",
        help="Base output directory",
    )
    parser.add_argument(
        "--nest-module", type=str, default="motor_neuron_module",
        help="NEST module name",
    )
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir / timestamp
    runs = build_run_configs(args.delays, args.learning_windows, args.n_iter)
    results = []

    for label, config in runs:
        run_dir = output_dir / label
        run_dir.mkdir(parents=True, exist_ok=True)
        n_samples = len(config.training.trajectories)

        _log.info("starting run", label=label, output=str(run_dir))

        network = get_m1_or_train(
            config,
            artifacts_dir=run_dir, force_retrain=True,
            nest_module=args.nest_module,
        )

        final_loss = compute_final_training_loss(run_dir, n_samples)

        _log.info("running inference test", label=label)
        inference_mse = run_inference_test(
            config, network, run_dir, args.nest_module
        )

        entry = {
            "input_shift_ms": config.task.input_shift_ms,
            "learning_window_ms": config.task.learning_window_ms,
            "final_training_loss": final_loss,
            "inference_mse": inference_mse,
        }
        results.append(entry)
        _log.info("finished run", **entry)

    # Save summary
    summary_path = output_dir / "sweep_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)

    plot_summary(results, output_dir)
    _log.info("sweep complete", summary=str(summary_path))


if __name__ == "__main__":
    main()
