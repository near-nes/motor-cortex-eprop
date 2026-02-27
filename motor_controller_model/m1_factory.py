"""
m1_factory: Factory for creating/loading M1Network instances.
"""
from pathlib import Path
from typing import List, Dict, Tuple

from motor_controller_model.config_schema import MotorControllerConfig
from motor_controller_model.m1_network import M1Network

def get_trained_m1(
    config: MotorControllerConfig,
    training_data: List[Dict[str, Tuple[str, str]]],
    force_retrain: bool = False
) -> M1Network:
    """
    Factory method to get an M1Network.
    
    Checks if a model with the same configuration exists in artifacts_dir.
    If so, loads it. If not (or if force_retrain is True), trains a new one.
    """
    # Use sim_results/m1_artifacts as the default directory for artifacts
    repo_root = Path(__file__).resolve().parent.parent
    
    # e.g., sim_results/m1_artifacts_a1b2c3d4
    short_hash = config.hash()[:8]
    artifacts_dir = repo_root / "sim_results" / f"m1_artifacts_{short_hash}"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    config_file = artifacts_dir / "config.yaml"
    weights_file = artifacts_dir / "trained_weights.npz"
    
    # Check cache
    if not force_retrain and config_file.exists() and weights_file.exists():
        try:
            cached_config = MotorControllerConfig.from_yaml(config_file)
            
            if cached_config.hash() == config.hash():
                print("Factory: Found matching cached model. Loading...")
                network = M1Network(config, artifacts_dir)
                network.load_weights(weights_file)
                return network
            else:
                print("Factory: Cached configuration does not match. Retraining...")
        except Exception as e:
            print(f"Factory: Error checking cache ({e}). Retraining...")
    else:
        if force_retrain:
            print("Factory: Force retrain requested.")
        else:
            print("Factory: No cache found.")

    # Train new model
    print("Factory: Training new M1 model...")
    network = M1Network(config, artifacts_dir)
    network.train(training_data)
    
    # Save config for future cache checks
    config.to_yaml(config_file)
        
    return network