"""Training signal generation for M1 training.

Uses the shared minjerk_dynamics package for trajectory and motor command
generation — no dependency on complete_control.
"""

from dataclasses import dataclass

import numpy as np
from minjerk_dynamics import generate_motor_commands, generate_trajectory

from .config_schema import TrainingSignalConfig, TrajectorySpec


def signal_to_pos_neg_rates(
    signal: np.ndarray, kp: float, base_rate: float
) -> tuple[np.ndarray, np.ndarray]:
    """Split *signal* into positive/negative channel firing rates.

    Mirrors the logic of ``tracking_neuron_nestml``:
    - pos channel: rate = kp * max(signal, 0) + base_rate
    - neg channel: rate = kp * max(-signal, 0) + base_rate
    """
    pos_rates = kp * np.maximum(signal, 0.0) + base_rate
    neg_rates = kp * np.maximum(-signal, 0.0) + base_rate
    return pos_rates, neg_rates


@dataclass
class TrainingSignals:
    """All signals for one trajectory, as 1-D arrays (one value per timestep)."""

    input_trajectory: np.ndarray
    target_rates_pos: np.ndarray
    target_rates_neg: np.ndarray


def sample_trajectory_indices(
    n_pool: int,
    n_iter: int,
    strategy: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Sample trajectory indices from a pool of training trajectories.

    Parameters
    ----------
    n_pool : int
        Number of available trajectories in the pool.
    n_iter : int
        Number of training iterations (one trajectory is presented per iteration).
    strategy : str
        Sampling strategy. One of:
        - "sequential": repeat the pool in order until enough samples are drawn.
        - "uniform": sample uniformly at random with replacement.
        - "shuffled_epochs": sample the pool in random order, repeating until enough samples are drawn.
    rng : np.random.Generator
        Random number generator for reproducibility.

    Returns
    -------
    np.ndarray
        Array of sampled trajectory indices of length `n_iter`.
    """

    if n_pool <= 0:
        raise ValueError("Trajectory pool must contain at least one trajectory.")

    if n_iter <= 0:
        raise ValueError("Number of training iterations must be positive.")

    if strategy == "sequential":
        reps = int(np.ceil(n_iter / n_pool))
        return np.tile(np.arange(n_pool), reps)[:n_iter]

    if strategy == "uniform":
        return rng.integers(0, n_pool, size=n_iter)

    if strategy == "shuffled_epochs":
        indices = []
        while len(indices) < n_iter:
            indices.extend(rng.permutation(n_pool))
        return np.asarray(indices[:n_iter])

    raise ValueError(f"Unknown sampling strategy: {strategy!r}")


def generate_training_signals(
    spec: TrajectorySpec,
    cfg: TrainingSignalConfig,
    resolution_ms: float,
    input_shift_ms: float = 0.0,
) -> TrainingSignals:
    """Build input trajectory + target rate arrays for one trajectory.

    The planner trajectory starts moving ``input_shift_ms`` earlier than the
    motor-command target, mirroring what ``generate_trajectory_minjerk`` does
    in the controller (``time_prep = sim.time_prep - m1_delay``).

    The raw trajectory is returned as-is — the tracking_neuron_nestml handles
    kp/base_rate conversion internally.  Target motor command rates are
    pre-computed since they feed a step_rate_generator directly.
    """
    init_rad = np.deg2rad(spec.init_angle_deg)
    tgt_rad = np.deg2rad(spec.target_angle_deg)

    # Raw trajectory for tracking_neuron_nestml (planner input)
    full_traj = generate_trajectory(
        init_angle_rad=init_rad,
        target_angle_rad=tgt_rad,
        resolution_ms=resolution_ms,
        time_prep_ms=cfg.time_prep_ms,
        time_move_ms=cfg.time_move_ms,
        time_locked_with_feedback_ms=0,
        time_post_ms=cfg.time_post_ms,
        m1_delay_ms=input_shift_ms,
    )

    # Motor commands → target rates for step_rate_generator
    full_mc = generate_motor_commands(
        init_angle_rad=init_rad,
        target_angle_rad=tgt_rad,
        resolution_ms=resolution_ms,
        time_prep_ms=cfg.time_prep_ms,
        time_move_ms=cfg.time_move_ms,
        time_locked_with_feedback_ms=0,
        time_post_ms=cfg.time_post_ms,
        inertia=cfg.inertia,
    )
    target_pos, target_neg = signal_to_pos_neg_rates(
        full_mc, cfg.m1_kp, cfg.m1_base_rate
    )

    return TrainingSignals(
        input_trajectory=full_traj,
        target_rates_pos=target_pos,
        target_rates_neg=target_neg,
    )
