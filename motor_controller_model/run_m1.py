"""
run_m1.py: Script to run M1 training/loading and standalone inference test.
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import nest
import numpy as np

from .m1_network import M1Network

# Ensure package is in path if running as script
sys.path.append(str(Path(__file__).resolve().parent.parent))

import structlog

from .config_schema import MotorControllerConfig, TrainingTimings
from .m1_factory import get_m1_or_train
from .signals import generate_training_signals
from .utils import install_nestml_module

_log = structlog.get_logger("m1_train")


def run_inference_test(
    config: MotorControllerConfig,
    network: M1Network,
    artifacts_dir: Path,
    nest_module: str,
):
    """Run standalone inference using tracking_neuron_nestml as planner input."""
    timings = TrainingTimings.from_config(config)

    nest.ResetKernel()
    nest.SetKernelStatus(
        {
            "resolution": config.simulation.step,
            "total_num_virtual_procs": config.simulation.total_num_virtual_procs,
        }
    )
    # Keep kernel-level e-prop settings explicit so inference is independent
    # of whether training was run earlier in the same process.
    # Note: we set the learning window to the full sequence duration for the inference test,
    # which avoid any zeroing of readout/error/target signals that would occur if the window was shorter than the sequence.
    nest.set(
        eprop_learning_window=timings.sequence_ms,  # Full sequence for inference test
        eprop_reset_neurons_on_update=False,
        eprop_update_interval=timings.sequence_ms,
    )
    install_nestml_module(nest_module)

    training_cfg = config.training
    step_ms = timings.step_ms
    n_trajectories = timings.n_samples
    # Run one configured sequence per trajectory.
    n_steps_per_seq = timings.n_timesteps_per_sequence
    sim_time_ms = n_steps_per_seq * n_trajectories * step_ms

    network.build_network(simulation_time_ms=sim_time_ms)

    # Build planner trajectory from configured training trajectories.
    all_signals = [
        generate_training_signals(
            spec, training_cfg, step_ms, config.task.input_shift_ms
        )
        for spec in training_cfg.trajectories
    ]

    full_traj = np.concatenate([sig.input_trajectory for sig in all_signals])
    # Pad trajectory to guarantee it covers the full simulation
    n_sim_steps = int(sim_time_ms / step_ms) + 1
    if len(full_traj) < n_sim_steps:
        full_traj = np.pad(
            full_traj, (0, n_sim_steps - len(full_traj)), constant_values=full_traj[-1]
        )
    sim_steps = len(full_traj)

    n_input = training_cfg.n_input_neurons
    planner_pos = nest.Create("tracking_neuron_nestml", n_input)
    nest.SetStatus(
        planner_pos,
        {
            "kp": training_cfg.planner_kp,
            "base_rate": training_cfg.planner_base_rate,
            "pos": True,
            "traj": full_traj.tolist(),
            "simulation_steps": sim_steps,
        },
    )

    planner_neg = nest.Create("tracking_neuron_nestml", n_input)
    nest.SetStatus(
        planner_neg,
        {
            "kp": training_cfg.planner_kp,
            "base_rate": training_cfg.planner_base_rate,
            "pos": False,
            "traj": full_traj.tolist(),
            "simulation_steps": sim_steps,
        },
    )

    network.connect(planner_pos)
    # network.connect(planner_neg)

    out_pos, out_neg = network.get_output_pops()

    mm_out = nest.Create(
        "multimeter",
        {
            "record_from": ["readout_signal"],
            "interval": step_ms,
            "start": step_ms,
            "stop": sim_time_ms,
        },
    )
    nest.Connect(mm_out, out_pos + out_neg)

    sr_rb = nest.Create("spike_recorder", {"start": step_ms, "stop": sim_time_ms})
    nest.Connect(network.nrns_rb, sr_rb)

    sr_rec = nest.Create("spike_recorder", {"start": step_ms, "stop": sim_time_ms})
    nest.Connect(network.nrns_rec, sr_rec)

    _log.debug("simulating inference", sim_time_ms=sim_time_ms)
    nest.Simulate(sim_time_ms)

    # Plot
    events = mm_out.get("events")
    idc_pos = events["senders"] == out_pos.tolist()[0]
    idc_neg = events["senders"] == out_neg.tolist()[0]
    events_rb = sr_rb.get("events")
    events_rec = sr_rec.get("events")

    fig, axs = plt.subplots(4, 1, sharex=True, figsize=(10, 10), dpi=300)

    # Row 0: planner input trajectory
    one_iter = np.concatenate([sig.input_trajectory for sig in all_signals])
    t_traj = np.arange(len(one_iter)) * step_ms
    axs[0].plot(t_traj, np.rad2deg(one_iter), lw=1.5, color="#1f77b4")
    axs[0].set_ylabel("planner (deg)")
    axs[0].grid(True, linestyle="--", alpha=0.3)

    # Row 1: RBF spike raster
    rb_ids = network.nrns_rb.tolist()
    rb_mask = np.isin(events_rb["senders"], rb_ids)
    if np.any(rb_mask):
        axs[1].scatter(
            events_rb["times"][rb_mask],
            events_rb["senders"][rb_mask],
            s=2,
            color="black",
            alpha=0.7,
        )
    axs[1].set_ylabel(r"$z_{rb}$")
    axs[1].grid(True, linestyle="--", alpha=0.3)

    # Row 2: M1 recurrent spike raster
    rec_ids = network.nrns_rec.tolist()
    rec_mask = np.isin(events_rec["senders"], rec_ids)
    if np.any(rec_mask):
        axs[2].scatter(
            events_rec["times"][rec_mask],
            events_rec["senders"][rec_mask],
            s=2,
            color="black",
            alpha=0.7,
        )
    axs[2].set_ylabel(r"$z_{rec}$")
    axs[2].grid(True, linestyle="--", alpha=0.3)

    # Row 3: readout signals
    axs[3].plot(
        events["times"][idc_pos],
        events["readout_signal"][idc_pos],
        label="Pos Readout",
        color="blue",
    )
    axs[3].plot(
        events["times"][idc_neg],
        events["readout_signal"][idc_neg],
        label="Neg Readout",
        color="red",
    )
    axs[3].set_ylabel("Rate Signal")
    axs[3].set_xlabel("Time (ms)")
    axs[3].legend()
    axs[3].grid(True, linestyle="--", alpha=0.3)

    # Trajectory boundaries
    total_seq_ms = n_steps_per_seq * step_ms
    for i in range(1, n_trajectories):
        for ax in axs:
            ax.axvline(x=i * total_seq_ms, color="gray", linestyle="--", alpha=0.5)

    fig.suptitle(
        f"Standalone Inference Test ({n_trajectories} Trajectories)",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout()

    plot_path = artifacts_dir / "standalone_inference_test.png"
    fig.savefig(plot_path)
    plt.close(fig)
    _log.debug(f"inference test complete {plot_path}")

    # Return readout arrays for downstream use
    return events["readout_signal"][idc_pos], events["readout_signal"][idc_neg]


def main():
    parser = argparse.ArgumentParser(
        description="Run M1 Network Training + Inference Test"
    )
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help="Force retraining even if cache exists",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to experiment YAML config (default: built-in MotorControllerConfig defaults)",
    )
    default_artifacts = Path(__file__).resolve().parent.parent / "results"
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_artifacts,
        help=f"Directory to save artifacts (default: {default_artifacts})",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional run subdirectory name created inside --output-dir",
    )
    parser.add_argument(
        "--nest-module",
        type=str,
        default="motor_neuron_module",
        help="NEST module (default: motor_neuron_module; use custom_stdp_module in controller)",
    )
    args = parser.parse_args()

    if args.config is not None:
        config = MotorControllerConfig.from_yaml(args.config)
    else:
        config = MotorControllerConfig()

    artifacts_dir = args.output_dir / args.run_name if args.run_name else args.output_dir
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    network = get_m1_or_train(
        config,
        artifacts_dir=artifacts_dir,
        force_retrain=args.force_retrain,
        nest_module=args.nest_module,
    )

    run_inference_test(config, network, artifacts_dir, args.nest_module)


if __name__ == "__main__":
    main()
