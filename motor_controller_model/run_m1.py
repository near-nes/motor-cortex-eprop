"""
run_m1.py: Script to run M1 training/loading and standalone inference test.
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import nest
import numpy as np
from motor_cortex_eprop.motor_controller_model.m1_network import M1Network

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
    nest.ResetKernel()
    nest.SetKernelStatus(
        {
            "resolution": config.simulation.step,
            "total_num_virtual_procs": config.simulation.total_num_virtual_procs,
        }
    )
    install_nestml_module(nest_module)

    training_cfg = config.training
    timings = TrainingTimings.from_config(config)
    step_ms = timings.step_ms
    n_trajectories = timings.n_samples
    # For inference: no input_shift, just sequence_ms per trajectory
    n_steps_per_seq = timings.n_timesteps_per_sequence
    sim_time_ms = n_steps_per_seq * n_trajectories * step_ms

    network.build_network(simulation_time_ms=sim_time_ms)

    # Generate signals and build planner trajectory for inference
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

    _log.debug("simulating inference", sim_time_ms=sim_time_ms)
    nest.Simulate(sim_time_ms)

    # Plot
    events = mm_out.get("events")
    idc_pos = events["senders"] == out_pos.tolist()[0]
    idc_neg = events["senders"] == out_neg.tolist()[0]

    plt.figure(figsize=(10, 4))
    plt.plot(
        events["times"][idc_pos],
        events["readout_signal"][idc_pos],
        label="Pos Readout",
        color="blue",
    )
    plt.plot(
        events["times"][idc_neg],
        events["readout_signal"][idc_neg],
        label="Neg Readout",
        color="red",
    )

    total_seq_ms = n_steps_per_seq * step_ms
    for i in range(1, n_trajectories):
        plt.axvline(
            x=i * total_seq_ms,
            color="gray",
            linestyle="--",
            alpha=0.5,
            label="Trajectory Boundary" if i == 1 else "",
        )

    plt.title(f"Standalone Inference Test Output ({n_trajectories} Trajectories)")
    plt.xlabel("Time (ms)")
    plt.ylabel("Rate Signal")
    plt.legend()
    plt.tight_layout()

    plot_path = artifacts_dir / "standalone_inference_test.png"
    plt.savefig(plot_path)
    plt.close()
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
    default_artifacts = Path(__file__).resolve().parent.parent / "results"
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_artifacts,
        help=f"Directory to save artifacts (default: {default_artifacts})",
    )
    parser.add_argument(
        "--nest-module",
        type=str,
        default="motor_neuron_module",
        help="NEST module (default: motor_neuron_module; use custom_stdp_module in controller)",
    )
    args = parser.parse_args()

    config = MotorControllerConfig()
    artifacts_dir = args.output_dir
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
