"""
m1_training: Standalone training function for the M1 e-prop network.
"""

from pathlib import Path
from typing import Dict, List, Tuple

import nest
import numpy as np
import structlog

from .config_schema import MotorControllerConfig
from .m1_network import M1Network, get_weights
from .plot_results import (
    plot_spikes_and_dynamics,
    plot_training_error,
    plot_weight_matrices,
)
from .utils import load_spike_data

_log = structlog.get_logger("m1_train")


def setup_nest_kernel(
    config: MotorControllerConfig, duration_dict: dict, nest_module: str
):
    """Set kernel parameters for M1 training."""
    nest.ResetKernel()
    nest.Install(nest_module)
    nest.set(
        eprop_learning_window=duration_dict["learning_window"],
        eprop_reset_neurons_on_update=False,
        eprop_update_interval=duration_dict["total_sequence_with_silence"],
        print_time=config.simulation.print_time,
        resolution=config.simulation.step,
        total_num_virtual_procs=config.simulation.total_num_virtual_procs,
        rng_seed=config.simulation.rng_seed,
    )


def train_m1(
    config: MotorControllerConfig,
    training_data: List[Dict[str, Tuple[str, str]]],
    artifacts_dir: Path,
    nest_module: str = None,
) -> M1Network:
    """
    Train the M1 network using e-prop and return the trained M1Network.

    Args:
        config: Motor controller configuration.
        training_data: List of dicts, each containing:
            {'input': (pos_file, neg_file), 'output': (pos_file, neg_file)}
        artifacts_dir: Directory for saving weights, plots, etc.

    Returns:
        Trained M1Network instance with weights saved to disk and in memory.
    """
    _log.debug(f"Initializing M1 training with {len(training_data)} trajectories...")

    # 1. Timing and Duration Setup
    step_ms = config.simulation.step
    sequence_ms = config.task.sequence
    silent_ms = config.task.silent_period
    total_seq_ms = sequence_ms + silent_ms
    n_iter = config.task.n_iter
    n_samples = len(training_data)

    learning_start = config.task.learning_start
    learning_end = config.task.learning_end
    learning_window = max(
        0.0, min(sequence_ms, learning_end) - max(0.0, learning_start)
    )

    duration = {
        "step": step_ms,
        "sequence": sequence_ms,
        "silent_period": silent_ms,
        "total_sequence_with_silence": total_seq_ms,
        "learning_window": learning_window,
        "task": int(round(total_seq_ms / step_ms)) * n_samples * n_iter * step_ms,
        "n_trajectories": n_samples,
    }
    duration["sim"] = duration["task"] + step_ms

    setup_nest_kernel(config, duration, nest_module)

    # 2. Data Loading and Preparation
    input_spikes_list = []
    desired_targets_list = {"pos": [], "neg": []}
    n_timesteps_per_stimulus = int(round(sequence_ms / step_ms))

    for sample in training_data:
        in_pos = load_spike_data(sample["input"][0])
        in_neg = load_spike_data(sample["input"][1])
        input_spikes_list.append((in_pos, in_neg))

        out_pos = load_spike_data(sample["output"][0])
        out_neg = load_spike_data(sample["output"][1])

        for key, data in zip(["pos", "neg"], [out_pos, out_neg]):
            hist = np.histogram(
                data[:, 1], bins=n_timesteps_per_stimulus, range=(0, sequence_ms)
            )[0]
            smoothed = np.convolve(hist, np.ones(50) / 10, mode="same")

            if silent_ms > 0:
                silent_steps = int(silent_ms / step_ms)
                smoothed = np.concatenate((np.zeros(silent_steps), smoothed))

            if config.task.input_shift_ms > 0:
                shift_steps = int(config.task.input_shift_ms / step_ms)
                shifted = np.roll(smoothed, shift_steps)
                shifted[:shift_steps] = 0.0
                smoothed = shifted

            desired_targets_list[key].append(smoothed)

    # 3. Build network in training mode
    network = M1Network(config)
    network.build_network(simulation_time_ms=duration["sim"], train=True)

    # 4. Create training-only NEST objects

    # Input Spike Generators & Parrot Neurons
    all_senders = set()
    for pos, neg in input_spikes_list:
        all_senders.update(pos[:, 0].astype(int))
        all_senders.update(neg[:, 0].astype(int))
    sender_to_idx = {s: i for i, s in enumerate(sorted(all_senders))}
    n_input_total = len(all_senders)

    spike_times_per_neuron = [[] for _ in range(n_input_total)]

    for iter_num in range(n_iter):
        for traj_idx, (pos, neg) in enumerate(input_spikes_list):
            offset = (traj_idx + iter_num * n_samples) * total_seq_ms
            for data in [pos, neg]:
                for s_id, t in data:
                    idx = sender_to_idx[int(s_id)]
                    spike_times_per_neuron[idx].append(t + offset + silent_ms)

    network.nrns_parrot = nest.Create("parrot_neuron", n_input_total)
    spike_gens = nest.Create("spike_generator", n_input_total)
    for i, times in enumerate(spike_times_per_neuron):
        nest.SetStatus(spike_gens[i : i + 1], {"spike_times": sorted(times)})
    nest.Connect(spike_gens, network.nrns_parrot, "one_to_one")
    nest.Connect(
        network.nrns_parrot,
        network.nrns_rb,
        "all_to_all",
        {
            "synapse_model": "static_synapse",
            "delay": config.synapses.static_delay,
            "weight": 1.0,
        },
    )

    # Target Rate Generators
    syn_cfg = config.synapses
    n_out = config.neurons.n_out
    gen_rate_target = nest.Create("step_rate_generator", n_out)
    target_amp_times = (
        np.arange(len(desired_targets_list["pos"][0]) * n_samples * n_iter) * step_ms
        + step_ms
    )
    concat_targets = {
        k: np.tile(np.concatenate(v), n_iter) for k, v in desired_targets_list.items()
    }

    nest.SetStatus(
        gen_rate_target[0],
        {
            "amplitude_times": target_amp_times,
            "amplitude_values": concat_targets["pos"],
        },
    )
    nest.SetStatus(
        gen_rate_target[1],
        {
            "amplitude_times": target_amp_times,
            "amplitude_values": concat_targets["neg"],
        },
    )

    nest.Connect(
        gen_rate_target[0],
        network.nrns_out_p,
        "one_to_one",
        {
            "synapse_model": "rate_connection_delayed",
            "delay": syn_cfg.rate_target_delay,
            "receptor_type": syn_cfg.receptor_type,
        },
    )
    nest.Connect(
        gen_rate_target[1],
        network.nrns_out_n,
        "one_to_one",
        {
            "synapse_model": "rate_connection_delayed",
            "delay": syn_cfg.rate_target_delay,
            "receptor_type": syn_cfg.receptor_type,
        },
    )

    # Recorders
    rec_cfg = config.recording

    mm_out = nest.Create(
        "multimeter",
        {
            **rec_cfg.mm_out.model_dump(),
            "interval": step_ms,
            "start": step_ms,
            "stop": duration["task"],
        },
    )
    nrns_out = network.nrns_out_p + network.nrns_out_n
    nest.Connect(mm_out, nrns_out)

    mm_rec = nest.Create(
        "multimeter",
        {
            **rec_cfg.mm_rec.model_dump(),
            "interval": step_ms,
            "start": step_ms,
            "stop": duration["task"],
        },
    )
    nrns_rec_record = network.nrns_rec[: rec_cfg.n_record]
    nest.Connect(mm_rec, nrns_rec_record)

    spike_recorder = nest.Create(
        "spike_recorder", {"start": step_ms, "stop": duration["task"]}
    )
    nest.Connect(network.nrns_rec, spike_recorder)

    # Force final update
    gen_final = nest.Create(
        "spike_generator", 1, {"spike_times": [duration["task"] + step_ms]}
    )
    nest.Connect(gen_final, network.nrns_rec, "all_to_all", {"weight": 1000.0})

    # Capture pre-training weights
    weights_pre_train = {
        "rec_rec": get_weights(network.nrns_rec, network.nrns_rec),
        "rec_out": get_weights(network.nrns_rec, nrns_out),
    }

    # 5. Simulation
    _log.debug(f"Simulating for {duration['sim']} ms...")
    nest.Simulate(duration["sim"])
    network.trained = True
    weights_path = artifacts_dir / "trained_weights.npz"
    network.save_weights(weights_path)

    # Capture post-training weights
    weights_post_train = {
        "rec_rec": get_weights(network.nrns_rec, network.nrns_rec),
        "rec_out": get_weights(network.nrns_rec, nrns_out),
        "rb_rec": get_weights(network.nrns_rb, network.nrns_rec),
    }

    # 6. Loss Calculation & Plotting
    events_mm_out = mm_out.get("events")
    readout_signal = events_mm_out["readout_signal"]
    target_signal = events_mm_out["target_signal"]
    senders = events_mm_out["senders"]

    loss_list = []
    for sender in set(senders):
        idc = senders == sender
        error = (readout_signal[idc] - target_signal[idc]) ** 2
        task_steps = int(duration["task"] / step_ms)
        seq_with_silence_steps = int(duration["total_sequence_with_silence"] / step_ms)
        loss_list.append(
            0.5
            * np.add.reduceat(error, np.arange(0, task_steps, seq_with_silence_steps))
        )
    loss = np.sum(loss_list, axis=0)

    if config.plotting.do_plotting:
        _log.debug("Generating plots...")
        colors = {"blue": "#1f77b4", "red": "#d62728", "white": "#ffffff"}

        plot_training_error(loss, artifacts_dir / "training_error.png")

        events_sr = spike_recorder.get("events")
        events_mm_rec = mm_rec.get("events")
        plot_spikes_and_dynamics(
            events_sr,
            events_mm_rec,
            events_mm_out,
            network.nrns_rec,
            rec_cfg.n_record,
            duration,
            colors,
            artifacts_dir / "spikes_and_dynamics.png",
            task_cfg=config.task.model_dump(),
        )

        plot_weight_matrices(
            weights_pre_train,
            weights_post_train,
            colors,
            artifacts_dir / "weight_matrices.png",
        )
        _log.debug(f"Plots saved to {artifacts_dir}")

    return network
