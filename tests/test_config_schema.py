"""
Test script to verify Pydantic config schema works with the existing YAML file.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from motor_controller_model.config_schema import MotorControllerConfig


def test_config_loading():
    """Test loading the config.yaml file with Pydantic validation."""
    config_path = (
        Path(__file__).parent.parent
        / "motor_controller_model"
        / "config"
        / "config.yaml"
    )

    print(f"Loading config from: {config_path}")

    try:
        # Load config using Pydantic
        config = MotorControllerConfig.from_yaml(str(config_path))

        print("✓ Config loaded successfully!")
        print(f"\nConfig validation passed. Sample values:")
        print(f"  - Simulation step: {config.simulation.step} ms")
        print(f"  - Number of recurrent neurons: {config.neurons.n_rec}")
        print(f"  - RBF centers: {config.rbf.num_centers}")
        print(f"  - Excitatory learning rate: {config.synapses.exc.optimizer.eta}")
        print(f"  - Static delay: {config.synapses.static_delay} ms")
        print(f"  - Feedback delay: {config.synapses.feedback_delay} ms")
        print(f"  - Rate target delay: {config.synapses.rate_target_delay} ms")

        # Test converting back to dict
        config_dict = config.to_dict()
        print(
            f"\n✓ Successfully converted to dict with {len(config_dict)} top-level keys"
        )

        # Test modifying values
        config.neurons.n_rec = 500
        config.synapses.exc.optimizer.eta = 0.05
        print(f"\n✓ Successfully modified values:")
        print(f"  - New n_rec: {config.neurons.n_rec}")
        print(f"  - New learning rate: {config.synapses.exc.optimizer.eta}")

        return True

    except Exception as e:
        print(f"✗ Error loading config: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_config_loading()
    exit(0 if success else 1)
