"""
m1_training: Standalone training function for the M1 e-prop network.
"""

import json
from pathlib import Path
from typing import List

import nest
import numpy as np
import structlog

from .config_schema import MotorControllerConfig, TrainingTimings
from .convergence import TrainingDidNotConverge, check_firing_rate
from .m1_network import M1Network, get_weights
from .plot_results import (
    plot_spikes_and_dynamics,
    plot_training_error,
    plot_weight_matrices,
)
from .signals import TrainingSignals, generate_training_signals
from .utils import install_nestml_module

_log = structlog.get_logger("m1_train")


def setup_nest_kernel(
    config: MotorControllerConfig, timings: TrainingTimings, nest_module: str
):
    """Reset NEST and configure kernel for M1 training."""
    nest.ResetKernel()
    install_nestml_module(nest_module)
    np.random.seed(config.simulation.rng_seed)
    nest.set(
        print_time=config.simulation.print_time,
        resolution=config.simulation.step,
        total_num_virtual_procs=config.simulation.total_num_virtual_procs,
        rng_seed=config.simulation.rng_seed,
    )


# ---------------------------------------------------------------------------
# Training sub-steps
# ---------------------------------------------------------------------------


def _generate_all_signals(
    config: MotorControllerConfig,
) -> List[TrainingSignals]:
    """Generate input/target signals for every trajectory in the config."""
    return [
        generate_training_signals(
            spec, config.training, config.simulation.step, config.task.input_shift_ms
        )
        for spec in config.training.trajectories
    ]


def _create_planner_neurons(
    network: M1Network,
    all_signals: List[TrainingSignals],
    timings: TrainingTimings,
    config: MotorControllerConfig,
):
    """Create tracking_neuron_nestml populations as planner input.

    For each channel (pos/neg), creates N tracking neurons whose ``traj``
    is the raw trajectory (radians).  The neuron applies kp/base_rate internally.
    Connects directly to the RBF layer.
    """
    n_input = config.training.n_input_neurons
    tcfg = config.training

    full_traj = np.tile(
        np.concatenate([sig.input_trajectory for sig in all_signals]),
        timings.n_iter,
    )
    sim_steps = len(full_traj)

    planner_pos = nest.Create("tracking_neuron_nestml", n_input)
    planner_pos.set({
            "kp": tcfg.planner_kp,
            "base_rate": tcfg.planner_base_rate,
            "pos": True,
            "traj": full_traj.tolist(),
            "simulation_steps": sim_steps,
        })

    # planner_neg = nest.Create("tracking_neuron_nestml", n_input)
    # #     planner_neg.set({#     {
    #         "kp": tcfg.planner_kp,
    #         "base_rate": tcfg.planner_base_rate,
    #         "pos": False,
    #         "traj": full_traj.tolist(),
    #         "simulation_steps": sim_steps,
    #     }: #})

    network.connect(planner_pos)
    # network.connect(planner_neg)


def _build_learning_window_schedule(
    timings: TrainingTimings,
    learning_start_ms: float,
) -> tuple[list[float], list[float]]:
    """Build piecewise-constant learning-window signal for readout receptor 1.

    The signal is 0 before ``learning_start_ms`` within each sequence and 1 from
    ``learning_start_ms`` until sequence end.
    """
    step_ms = timings.step_ms
    seq_starts = np.arange(0.0, timings.task_ms, timings.sequence_ms)

    times = [step_ms]
    values = [0.0]

    for seq_start in seq_starts:
        start_time = seq_start + learning_start_ms + step_ms
        end_time = seq_start + timings.sequence_ms + step_ms

        if start_time < end_time:
            times.extend([start_time, end_time])
            values.extend([1.0, 0.0])

    return times, values


def _create_target_generators(
    network: M1Network,
    all_signals: List[TrainingSignals],
    timings: TrainingTimings,
    config: MotorControllerConfig,
):
    """Create step_rate_generators that feed target signals to output neurons."""
    step_ms = timings.step_ms
    syn_cfg = config.synapses

    concat_pos = np.tile(
        np.concatenate([sig.target_rates_pos for sig in all_signals]),
        timings.n_iter,
    )
    concat_neg = np.tile(
        np.concatenate([sig.target_rates_neg for sig in all_signals]),
        timings.n_iter,
    )

    amp_times = np.arange(len(concat_pos)) * step_ms + step_ms

    # Create generator nodes first.
    gen_rate_target = nest.Create("step_rate_generator", 2)
    gen_learning_window = nest.Create("step_rate_generator", 1)

    # Configure target signals.
    gen_rate_target[0].set({
            "amplitude_times": amp_times,
            "amplitude_values": concat_pos,
        })
    gen_rate_target[1].set({
            "amplitude_times": amp_times,
            "amplitude_values": concat_neg,
        })

    # Learning-window gate for eprop_readout (receptor 1),
    # separate from target input on receptor 2.
    lw_times, lw_values = _build_learning_window_schedule(
        timings,
        float(config.task.learning_start_ms),
    )

    # Configure learning-window signal.
    gen_learning_window[0].set({
            "amplitude_times": lw_times,
            "amplitude_values": lw_values,
        })

    # Connect generators after all node creation/configuration.
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
    nest.Connect(
        gen_learning_window,
        network.nrns_out_p + network.nrns_out_n,
        "all_to_all",
        {
            "synapse_model": "rate_connection_delayed",
            "delay": syn_cfg.rate_target_delay,
            "receptor_type": 1,
        },
    )


def _create_recorders(network, timings, config):
    """Create multimeters and spike recorder; return (mm_out, mm_rec, spike_recorder)."""
    step_ms = timings.step_ms
    rec_cfg = config.recording

    mm_out = nest.Create(
        "multimeter",
        {
            **rec_cfg.mm_out.model_dump(),
            "interval": step_ms,
            "start": step_ms,
            "stop": timings.task_ms,
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
            "stop": timings.task_ms,
        },
    )
    nrns_rec_record = network.nrns_rec[: rec_cfg.n_record]
    nest.Connect(mm_rec, nrns_rec_record)

    spike_recorder = nest.Create(
        "spike_recorder", {"start": step_ms, "stop": timings.task_ms}
    )
    nest.Connect(network.nrns_rec, spike_recorder)

    spike_recorder_rb = nest.Create(
        "spike_recorder", {"start": step_ms, "stop": timings.task_ms}
    )
    nest.Connect(network.nrns_rb, spike_recorder_rb)

    return mm_out, mm_rec, spike_recorder, spike_recorder_rb


def _compute_loss(events_mm_out, timings) -> np.ndarray:
    """Compute per-sequence MSE from multimeter output events."""
    readout = events_mm_out["readout_signal"]
    target = events_mm_out["target_signal"]
    senders = events_mm_out["senders"]

    loss_list = []
    task_steps = int(timings.task_ms / timings.step_ms)
    seq_steps = timings.n_timesteps_per_sequence
    for sender in set(senders):
        mask = senders == sender
        error = (readout[mask] - target[mask]) ** 2
        loss_list.append(
            0.5 * np.add.reduceat(error, np.arange(0, task_steps, seq_steps))
        )
    return np.sum(loss_list, axis=0)


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------


def train_m1(
    config: MotorControllerConfig,
    artifacts_dir: Path,
    nest_module: str = None,
) -> M1Network:
    """Train the M1 network using e-prop and return the trained M1Network.

    Training data is fully specified via ``config.training`` — no external
    spike files are needed.
    """
    timings = TrainingTimings.from_config(config)
    _log.debug(
        "starting M1 training",
        n_trajectories=timings.n_samples,
        n_iter=timings.n_iter,
        sim_ms=timings.task_ms,
    )

    setup_nest_kernel(config, timings, nest_module)
    all_signals = _generate_all_signals(config)

    # Build network in training mode
    network = M1Network(config)
    network.build_network(simulation_time_ms=timings.task_ms, train=True)

    # Wire up training-specific NEST objects
    _create_planner_neurons(network, all_signals, timings, config)
    _create_target_generators(network, all_signals, timings, config)
    mm_out, mm_rec, spike_recorder, spike_recorder_rb = _create_recorders(
        network, timings, config
    )

    # Capture pre-training weights
    nrns_out = network.nrns_out_p + network.nrns_out_n
    weights_pre = {
        "rec_rec": get_weights(network.nrns_rec, network.nrns_rec),
        "rb_rec": get_weights(network.nrns_rb, network.nrns_rec),
        "rec_out": get_weights(network.nrns_rec, nrns_out),
    }

    # Run simulation
    _log.debug("simulating", sim_ms=timings.task_ms)
    nest.Simulate(timings.task_ms)
    network.trained = True
    network.save_weights(artifacts_dir / "trained_weights.npz")

    weights_post = {
        "rec_rec": get_weights(network.nrns_rec, network.nrns_rec),
        "rec_out": get_weights(network.nrns_rec, nrns_out),
        "rb_rec": get_weights(network.nrns_rb, network.nrns_rec),
    }

    # Loss calculation
    events_mm_out = mm_out.get("events")
    loss = _compute_loss(events_mm_out, timings)
    np.save(artifacts_dir / "training_loss.npy", loss)

    # Calculate the duration of a single training iteration (all trajectories combined)
    iter_duration_ms = (
        timings.n_timesteps_per_sequence * timings.n_samples * timings.step_ms
    )
    last_iter_start_ms = timings.task_ms - iter_duration_ms

    events_rec = spike_recorder.get("events")
    spike_times = events_rec["times"]
    spike_senders = events_rec["senders"]

    # Filter spikes that occurred only in the last iteration.
    mask = spike_times >= last_iter_start_ms
    spike_times_last_iter = spike_times[mask]
    spike_senders_last_iter = spike_senders[mask]

    n_neurons = len(network.nrns_rec)
    iter_duration_s = iter_duration_ms / 1000.0

    if n_neurons > 0 and iter_duration_s > 0:
        n_spikes_last_iter = len(spike_times_last_iter)
        mean_firing_rate_hz = n_spikes_last_iter / (n_neurons * iter_duration_s)

        rec_ids = np.asarray([nrn.global_id for nrn in network.nrns_rec])
        spike_counts = np.zeros(n_neurons, dtype=float)
        unique_senders, sender_counts = np.unique(
            spike_senders_last_iter, return_counts=True
        )
        sender_to_index = {int(gid): idx for idx, gid in enumerate(rec_ids)}
        for sender, count in zip(unique_senders, sender_counts):
            sender_gid = int(getattr(sender, "global_id", sender))
            neuron_index = sender_to_index.get(sender_gid)
            if neuron_index is not None:
                spike_counts[neuron_index] = float(count)

        rate_per_neuron_hz = spike_counts / iter_duration_s
        rate_mean = float(np.mean(rate_per_neuron_hz))
        rate_std = float(np.std(rate_per_neuron_hz))
        spike_rate_cv = float(rate_std / (abs(rate_mean) + 1e-12))
    else:
        mean_firing_rate_hz = 0.0
        spike_rate_cv = 0.0

    with open(artifacts_dir / "mean_firing_rate.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "mean_firing_rate_hz": float(mean_firing_rate_hz),
                "spike_rate_cv": float(spike_rate_cv),
            },
            f,
        )

    # Plotting
    if config.plotting.do_plotting:
        _log.debug("generating plots")

        plot_training_error(loss, artifacts_dir / "training_error.png")
        last_10pct_start = max(0, int(np.floor(len(loss) * 0.9)))
        last_10pct_x = np.arange(last_10pct_start + 1, len(loss) + 1)
        plot_training_error(
            loss[last_10pct_start:],
            artifacts_dir / "training_error_last_10pct.png",
            x=last_10pct_x,
        )
        plot_spikes_and_dynamics(
            spike_recorder.get("events"),
            mm_rec.get("events"),
            events_mm_out,
            network.nrns_rec,
            config.recording.n_record,
            timings,
            artifacts_dir / "spikes_and_dynamics.png",
            input_signals=all_signals,
            events_sr_rb=spike_recorder_rb.get("events"),
            nrns_rb=network.nrns_rb,
        )
        weight_colors = {"blue": "#1f77b4", "red": "#d62728", "white": "#ffffff"}
        n_exc = config.neurons.n_exc
        plot_weight_matrices(
            weights_pre,
            weights_post,
            weight_colors,
            artifacts_dir / "weight_matrices.png",
            n_exc=n_exc,
        )

    if config.convergence.enabled:
        result = check_firing_rate(float(mean_firing_rate_hz), config.convergence)
        if not result.ok:
            failure_path = artifacts_dir / "convergence_failure.json"
            with open(failure_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "reason": result.reason,
                        "detail": result.detail,
                        "mean_firing_rate_hz": result.mean_firing_rate_hz,
                        "min_firing_rate_hz": config.convergence.min_firing_rate_hz,
                        "max_firing_rate_hz": config.convergence.max_firing_rate_hz,
                    },
                    f,
                    indent=2,
                )
            _log.error(
                "training did not converge",
                reason=result.reason,
                detail=result.detail,
            )
            raise TrainingDidNotConverge(result.detail)

    return network
