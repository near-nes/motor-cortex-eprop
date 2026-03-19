"""
Sweep M1 training parameters and evaluate performance.

Usage:
    python sweep_delay.py --delays 0 50 100 200 --learning-starts 0 650
    python sweep_delay.py --delays 0 50 100       # sweep delay only (learning_start=650)
    python sweep_delay.py --learning-starts 0 300 650  # sweep learning_start only (delay=0)
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
from motor_controller_model.config_schema import MotorControllerConfig
from motor_controller_model.m1_factory import get_m1_or_train
from motor_controller_model.utils import install_nestml_module, load_spike_data

_log = structlog.get_logger("sweep")

BASE_DATA_DIR = (
    Path(__file__).resolve().parent
    / "motor_controller_model"
    / "dataset_motor_training"
    / "input_ouput_data"
)

TRAINING_DATA = [
    {
        "input": (
            str(BASE_DATA_DIR / "N200_9020_planner_p.dat"),
            str(BASE_DATA_DIR / "N200_9020_planner_n.dat"),
        ),
        "output": (
            str(BASE_DATA_DIR / "N200_9020_mc_m1_p.dat"),
            str(BASE_DATA_DIR / "N200_9020_mc_m1_n.dat"),
        ),
    },
    {
        "input": (
            str(BASE_DATA_DIR / "N200_90140_planner_p.dat"),
            str(BASE_DATA_DIR / "N200_90140_planner_n.dat"),
        ),
        "output": (
            str(BASE_DATA_DIR / "N200_90140_mc_m1_p.dat"),
            str(BASE_DATA_DIR / "N200_90140_mc_m1_n.dat"),
        ),
    },
]


def build_targets(training_data, config):
    """Build target signals from output spike data, shifted to match training alignment."""
    step_ms = config.simulation.step
    sequence_ms = config.task.sequence
    silent_ms = config.task.silent_period
    input_shift_ms = config.task.input_shift_ms
    shift_steps = int(input_shift_ms / step_ms) if input_shift_ms > 0 else 0
    n_timesteps = int(round(sequence_ms / step_ms))

    targets = {"pos": [], "neg": []}
    for sample in training_data:
        out_pos = load_spike_data(sample["output"][0])
        out_neg = load_spike_data(sample["output"][1])
        for key, data in zip(["pos", "neg"], [out_pos, out_neg]):
            hist = np.histogram(data[:, 1], bins=n_timesteps, range=(0, sequence_ms))[0]
            smoothed = np.convolve(hist, np.ones(50) / 10, mode="same")
            if silent_ms > 0:
                silent_steps = int(silent_ms / step_ms)
                smoothed = np.concatenate((np.zeros(silent_steps), smoothed))
            if shift_steps > 0:
                smoothed = np.concatenate((np.zeros(shift_steps), smoothed))
            targets[key].append(smoothed)
    return targets


def run_inference_test(config, network, training_data, artifacts_dir, nest_module):
    """Run standalone inference, compute MSE against target, save plot."""
    nest.ResetKernel()
    nest.SetKernelStatus({"resolution": config.simulation.step})
    install_nestml_module(nest_module)

    step_ms = config.simulation.step
    sequence_ms = config.task.sequence
    silent_ms = config.task.silent_period
    total_seq_ms = sequence_ms + silent_ms
    n_trajectories = len(training_data)

    n_steps_per_seq = int(round(total_seq_ms / step_ms))
    sim_time_ms = n_steps_per_seq * n_trajectories * step_ms + step_ms
    network.build_network(simulation_time_ms=sim_time_ms)

    # Collect unique input neuron IDs
    all_senders = set()
    for traj in training_data:
        for f in traj["input"]:
            all_senders.update(load_spike_data(f)[:, 0].astype(int))

    n_input = len(all_senders)
    sender_to_idx = {s: i for i, s in enumerate(sorted(all_senders))}
    spike_times_per_neuron = [[] for _ in range(n_input)]

    for traj_idx, traj in enumerate(training_data):
        offset = traj_idx * total_seq_ms
        for f in traj["input"]:
            for s_id, t in load_spike_data(f):
                spike_times_per_neuron[sender_to_idx[int(s_id)]].append(
                    t + offset + silent_ms
                )

    mock_planner = nest.Create("spike_generator", n_input)
    for i, times in enumerate(spike_times_per_neuron):
        nest.SetStatus(mock_planner[i : i + 1], {"spike_times": sorted(times)})

    network.connect(mock_planner)
    out_pos, out_neg = network.get_output_pops()

    sim_time = n_trajectories * total_seq_ms + step_ms
    mm_out = nest.Create(
        "multimeter",
        {
            "record_from": ["readout_signal"],
            "interval": step_ms,
            "start": step_ms,
            "stop": sim_time,
        },
    )
    nest.Connect(mm_out, out_pos + out_neg)
    nest.Simulate(sim_time)

    # Extract readout signals
    events = mm_out.get("events")
    idc_pos = events["senders"] == out_pos.tolist()[0]
    idc_neg = events["senders"] == out_neg.tolist()[0]
    readout_pos = events["readout_signal"][idc_pos]
    readout_neg = events["readout_signal"][idc_neg]

    # Compute MSE against shifted target
    targets = build_targets(training_data, config)
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
    ls = config.task.learning_start
    plt.figure(figsize=(10, 4))
    plt.plot(events["times"][idc_pos], readout_pos, label="Pos Readout", color="blue")
    plt.plot(events["times"][idc_neg], readout_neg, label="Neg Readout", color="red")
    for i in range(1, n_trajectories):
        plt.axvline(
            x=i * total_seq_ms, color="gray", linestyle="--", alpha=0.5,
            label="Trajectory Boundary" if i == 1 else "",
        )
    plt.title(
        f"Inference — delay={int(delay)}ms, learning_start={int(ls)}ms  (MSE={inference_mse:.1f})"
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


def build_run_configs(delays, learning_starts, n_iter):
    """Build (label, config) pairs for the cartesian product of sweep parameters."""
    runs = []
    for delay, ls in itertools.product(delays, learning_starts):
        config = MotorControllerConfig()
        config.task.input_shift_ms = delay
        config.task.learning_start = ls
        config.task.n_iter = n_iter
        label = f"delay_{int(delay)}ms_ls_{int(ls)}ms"
        runs.append((label, config))
    return runs


def plot_summary(results, output_dir):
    """Plot sweep results on both axes: vs delay (grouped by ls) and vs ls (grouped by delay)."""
    metrics = [
        ("final_training_loss", "Final Training Loss"),
        ("inference_mse", "Inference MSE"),
    ]
    views = [
        ("input_shift_ms", "input_shift_ms (ms)", "learning_start", "ls={v}ms"),
        ("learning_start", "learning_start (ms)", "input_shift_ms", "delay={v}ms"),
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
        "--learning-starts", type=float, nargs="+", default=[650],
        help="learning_start values to test (ms)",
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
    n_samples = len(TRAINING_DATA)
    runs = build_run_configs(args.delays, args.learning_starts, args.n_iter)
    results = []

    for label, config in runs:
        run_dir = output_dir / label
        run_dir.mkdir(parents=True, exist_ok=True)

        _log.info("starting run", label=label, output=str(run_dir))

        network = get_m1_or_train(
            config, TRAINING_DATA,
            artifacts_dir=run_dir, force_retrain=True,
            nest_module=args.nest_module,
        )

        final_loss = compute_final_training_loss(run_dir, n_samples)

        _log.info("running inference test", label=label)
        inference_mse = run_inference_test(
            config, network, TRAINING_DATA, run_dir, args.nest_module
        )

        entry = {
            "input_shift_ms": config.task.input_shift_ms,
            "learning_start": config.task.learning_start,
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
