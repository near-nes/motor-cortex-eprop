"""
m1_factory: Factory for creating/loading M1Network instances.
"""

import subprocess
from pathlib import Path

import structlog

from .config_schema import MotorControllerConfig
from .m1_network import M1Network
from .m1_training import train_m1

_log = structlog.get_logger("m1_factory")


class TrainingRequired(RuntimeError):
    """Raised when a trained M1 model is required but not available or outdated."""

    pass


def get_git_commit_hash(repo_dir: Path) -> str:
    """Safely retrieves the current git commit hash."""
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=repo_dir, stderr=subprocess.DEVNULL
            )
            .decode("ascii")
            .strip()
        )
    except Exception:
        return "unknown"


def _stamp_config(config: MotorControllerConfig) -> None:
    repo_root = Path(__file__).resolve().parent.parent
    config.git_commit = get_git_commit_hash(repo_root)


# Fields that do NOT affect trained weights: changing them should not
# invalidate the cache.  Everything else is assumed to matter.
_TRAINING_ONLY_EXCLUDE = {
    "git_commit": True,
    "recording": True,
    "plotting": True,
    "task": {"n_iter", "learning_start_ms", "gradient_batch_size"},
    "simulation": {"rng_seed", "print_time", "total_num_virtual_procs"},
    "training": {"trajectories", "time_prep_ms", "time_post_ms"},
}


def _check_saved_model(
    config: MotorControllerConfig, artifacts_dir: Path
) -> M1Network | None:
    """Load and return M1Network if saved model matches config, else None.

    Only compares architecture and task/signal fields — training procedure
    fields (n_iter, plotting, recording, etc.) are excluded.
    """
    config_file = artifacts_dir / "config.yaml"
    weights_file = artifacts_dir / "trained_weights.npz"

    if not (config_file.exists() and weights_file.exists()):
        return None

    saved = MotorControllerConfig.from_yaml(config_file).model_dump(
        exclude=_TRAINING_ONLY_EXCLUDE
    )
    requested = config.model_dump(exclude=_TRAINING_ONLY_EXCLUDE)
    diffs = [
        f"  {k}: requested={requested[k]!r}, saved={saved[k]!r}"
        for k in requested
        if requested[k] != saved[k]
    ]

    if diffs:
        _log.warning("config mismatch with saved model", diffs=diffs)
        return None

    _log.info("config matches, loading weights", artifacts_dir=str(artifacts_dir))
    network = M1Network(config)
    network.load_weights(weights_file)
    return network


def get_m1_or_raise(
    config: MotorControllerConfig,
    artifacts_dir: Path,
) -> M1Network:
    """
    Load a trained M1Network from artifacts_dir, or raise TrainingRequired.

    For controller usage: expects a pre-trained model with matching config.
    Does NOT build the NEST network, caller must call build_network().
    """
    _stamp_config(config)

    if network := _check_saved_model(config, artifacts_dir):
        return network

    config_file = artifacts_dir / "config.yaml"
    if config_file.exists():
        msg = (
            "M1 config mismatch — retraining required.\n"
            "Please retrain: python -m submodules.motor_cortex_eprop.motor_controller_model.run_m1"
            f" --output-dir {artifacts_dir}"
        )
    else:
        msg = (
            f"No trained M1 model found at {artifacts_dir}.\n"
            "Please train first: python -m submodules.motor_cortex_eprop.motor_controller_model.run_m1"
            f" --output-dir {artifacts_dir}"
        )
    raise TrainingRequired(msg)


def get_m1_or_train(
    config: MotorControllerConfig,
    artifacts_dir: Path,
    nest_module: str,
    force_retrain: bool = False,
) -> M1Network:
    """
    Load a trained M1Network if available, otherwise train a new one.

    Training REQUIRES resetting kernel.
    Does NOT build the NEST network. Caller must call build_network() after kernel setup.
    """
    _stamp_config(config)

    if not force_retrain and (network := _check_saved_model(config, artifacts_dir)):
        return network

    _log.info("training M1 model", artifacts_dir=str(artifacts_dir))
    network = train_m1(config, artifacts_dir, nest_module=nest_module)
    config.to_yaml(artifacts_dir / "config.yaml")
    return network
