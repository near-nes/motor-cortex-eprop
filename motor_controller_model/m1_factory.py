"""
m1_factory: Factory for creating/loading M1Network instances.
"""

import subprocess
from pathlib import Path
from typing import List, Dict, Tuple
import nest

from motor_controller_model.config_schema import MotorControllerConfig
from motor_controller_model.m1_network import M1Network


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


def get_trained_m1(
    config: MotorControllerConfig,
    training_data: List[Dict[str, Tuple[str, str]]],
    force_retrain: bool = False,
) -> M1Network:
    """
    Factory method to get an M1Network.

    Checks if a model with the exact same configuration exists.
    If so, loads it into a clean inference SNN.
    If not (or if force_retrain is True), trains a new one online.
    """
    repo_root = Path(__file__).resolve().parent.parent

    # Stamp the config with the git hash to track codebase versions
    config.git_commit = get_git_commit_hash(repo_root)

    # Use the config hash to create a unique folder for this parameter set
    short_hash = config.hash()[:8]
    artifacts_dir = repo_root / "sim_results" / f"m1_artifacts_{short_hash}"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    config_file = artifacts_dir / "config.yaml"
    weights_file = artifacts_dir / "trained_weights.npz"

    # --- 1. Check Cache ---
    if not force_retrain and config_file.exists() and weights_file.exists():
        try:
            cached_config = MotorControllerConfig.from_yaml(config_file)

            # Verify the loaded config matches our current requested parameters perfectly
            if cached_config.hash() == config.hash():
                print(
                    f"Factory: Found matching cached model ({short_hash}). Loading..."
                )
                network = M1Network(config, artifacts_dir)
                network.load_weights(weights_file)

                # Build a clean, optimized SNN for inference
                nest.ResetKernel()
                nest.SetKernelStatus({"resolution": config.simulation.step})
                network.build_network()

                return network
            else:
                print("Factory: Cached configuration does not match. Retraining...")
        except Exception as e:
            print(f"Factory: Error checking cache ({e}). Retraining...")
    else:
        if force_retrain:
            print("Factory: Force retrain requested.")
        else:
            print(f"Factory: No cache found for hash {short_hash}.")

    # --- 2. Train New Model ---
    print("Factory: Training new M1 model...")
    network = M1Network(config, artifacts_dir)
    network.train(training_data)

    # Save the config so future runs can instantly load from cache
    config.to_yaml(config_file)

    # --- 3. Prep for Integration ---
    # Purge the heavy training nodes/generators and build a lean inference network
    print("Factory: Purging training kernel and building inference network...")
    nest.ResetKernel()
    nest.SetKernelStatus({"resolution": config.simulation.step})
    network.build_network()

    return network
