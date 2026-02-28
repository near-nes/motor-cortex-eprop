"""
run_m1.py: Script to run M1 training/loading using the new factory structure.
"""

import argparse
from pathlib import Path
import sys
import numpy as np
import nest
import matplotlib.pyplot as plt

# Ensure package is in path if running as script
sys.path.append(str(Path(__file__).resolve().parent.parent))

from motor_controller_model.config_schema import MotorControllerConfig
from motor_controller_model.m1_factory import get_trained_m1
from motor_controller_model.utils import load_spike_data


def main():
    parser = argparse.ArgumentParser(description="Run M1 Network (Spike Input Mode)")
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help="Force retraining even if cache exists",
    )
    args = parser.parse_args()

    # 1. Define Configuration
    config = MotorControllerConfig()

    # 2. Define Training Data
    base_data_dir = (
        Path(__file__).resolve().parent / "dataset_motor_training" / "input_ouput_data"
    )

    if not base_data_dir.exists():
        print(f"Error: Dataset directory not found at {base_data_dir}")
        return

    traj1 = {
        "input": (
            str(base_data_dir / "N200_9020_planner_p.dat"),
            str(base_data_dir / "N200_9020_planner_n.dat"),
        ),
        "output": (
            str(base_data_dir / "N200_9020_mc_m1_p.dat"),
            str(base_data_dir / "N200_9020_mc_m1_n.dat"),
        ),
    }

    traj2 = {
        "input": (
            str(base_data_dir / "N200_90140_planner_p.dat"),
            str(base_data_dir / "N200_90140_planner_n.dat"),
        ),
        "output": (
            str(base_data_dir / "N200_90140_mc_m1_p.dat"),
            str(base_data_dir / "N200_90140_mc_m1_n.dat"),
        ),
    }
    training_data = [traj1, traj2]

    # 3. Use Factory to get Model (Returns a clean inference SNN)
    print(f"Requesting M1 model (Artifacts: sim_results/m1_artifacts)...")
    network = get_trained_m1(config, training_data, args.force_retrain)
    print("M1 Model ready.")

    # =====================================================================
    # 4. STANDALONE INFERENCE TEST (Evaluating All Trajectories)
    # =====================================================================
    print(
        f"\n--- Running Standalone Inference Test on {len(training_data)} Trajectories ---"
    )

    step_ms = config.simulation.step
    sequence_ms = config.task.sequence
    silent_ms = config.task.silent_period
    total_seq_ms = sequence_ms + silent_ms
    n_trajectories = len(training_data)

    # A. Figure out how many unique input neurons we have across ALL data
    all_senders = set()
    for traj in training_data:
        pos_data = load_spike_data(traj["input"][0])
        neg_data = load_spike_data(traj["input"][1])
        all_senders.update(pos_data[:, 0].astype(int))
        all_senders.update(neg_data[:, 0].astype(int))

    n_input = len(all_senders)
    sender_to_idx = {s: i for i, s in enumerate(sorted(all_senders))}
    spike_times_per_neuron = [[] for _ in range(n_input)]

    # B. Load and offset spikes for each trajectory sequentially
    for traj_idx, traj in enumerate(training_data):
        time_offset = traj_idx * total_seq_ms

        pos_data = load_spike_data(traj["input"][0])
        neg_data = load_spike_data(traj["input"][1])

        for data in [pos_data, neg_data]:
            for s_id, t in data:
                # Add the trajectory offset + the silent period padding
                adjusted_time = t + time_offset + silent_ms
                spike_times_per_neuron[sender_to_idx[int(s_id)]].append(adjusted_time)

    # C. Create mock Planner neurons
    mock_planner = nest.Create("spike_generator", n_input)
    for i, times in enumerate(spike_times_per_neuron):
        nest.SetStatus(mock_planner[i : i + 1], {"spike_times": sorted(times)})

    # D. Connect to the M1 Network Interface
    network.connect(mock_planner)
    out_pos, out_neg = network.get_output_pops()

    sim_time = (n_trajectories * total_seq_ms) + step_ms

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

    # E. Run Inference
    print(f"Simulating inference pass for {sim_time} ms...")
    nest.Simulate(sim_time)

    # F. Plotting
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

    # Add vertical dividers to show where each trajectory starts
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

    plot_path = network.artifacts_dir / "standalone_inference_test.png"
    plt.savefig(plot_path)
    print(f"Test complete! Output plotted to: {plot_path}")


if __name__ == "__main__":
    main()
