"""
m1_training: Standalone training function for the M1 e-prop network.
"""

from pathlib import Path
from typing import List

import nest
import numpy as np
import structlog

from .config_schema import MotorControllerConfig, TrainingTimings
from .m1_network import M1Network, get_weights
from .plot_results import (
    plot_spikes_and_dynamics,
    plot_training_error,
    plot_weight_matrices,
)
from .signals import TrainingSignals, generate_training_signals

_log = structlog.get_logger("m1_train")


def setup_nest_kernel(
    config: MotorControllerConfig, timings: TrainingTimings, nest_module: str
):
    """Reset NEST and configure kernel for M1 training."""
    nest.ResetKernel()
    nest.Install(nest_module)
    nest.set(
        eprop_learning_window=timings.learning_window,
        eprop_reset_neurons_on_update=True,
        eprop_update_interval=timings.sequence_ms,
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
    n_seq_steps = timings.n_timesteps_per_sequence
    tcfg = config.training

    full_traj = np.tile(
        np.concatenate([sig.input_trajectory for sig in all_signals]),
        timings.n_iter,
    )
    sim_steps = len(full_traj)

    planner_pos = nest.Create("tracking_neuron_nestml", n_input)
    nest.SetStatus(
        planner_pos,
        {
            "kp": tcfg.planner_kp,
            "base_rate": tcfg.planner_base_rate,
            "pos": True,
            "traj": full_traj.tolist(),
            "simulation_steps": sim_steps,
        },
    )

    planner_neg = nest.Create("tracking_neuron_nestml", n_input)
    nest.SetStatus(
        planner_neg,
        {
            "kp": tcfg.planner_kp,
            "base_rate": tcfg.planner_base_rate,
            "pos": False,
            "traj": full_traj.tolist(),
            "simulation_steps": sim_steps,
        },
    )

    network.connect(planner_pos)


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

    gen_rate_target = nest.Create("step_rate_generator", 2)
    nest.SetStatus(
        gen_rate_target[0],
        {
            "amplitude_times": amp_times,
            "amplitude_values": concat_pos,
        },
    )
    nest.SetStatus(
        gen_rate_target[1],
        {
            "amplitude_times": amp_times,
            "amplitude_values": concat_neg,
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
        sim_ms=timings.sim_ms,
    )

    setup_nest_kernel(config, timings, nest_module)
    all_signals = _generate_all_signals(config)

    # Build network in training mode
    network = M1Network(config)
    network.build_network(simulation_time_ms=timings.sim_ms, train=True)

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
        "rec_out": get_weights(network.nrns_rec, nrns_out),
    }

    # Run simulation
    _log.debug("simulating", sim_ms=timings.sim_ms)
    nest.Simulate(timings.sim_ms)
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

    # Plotting
    if config.plotting.do_plotting:
        _log.debug("generating plots")

        plot_training_error(loss, artifacts_dir / "training_error.png")
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
        plot_weight_matrices(
            weights_pre,
            weights_post,
            weight_colors,
            artifacts_dir / "weight_matrices.png",
        )

    return network
