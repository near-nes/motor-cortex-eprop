"""
Test script to verify Pydantic config schema works.
"""

from pathlib import Path

from motor_controller_model.config_schema import MotorControllerConfig


def test_config_loading():
    """Test loading and saving configuration via Pydantic validation."""
    # Generate a default config and test round-trip serialization
    config = MotorControllerConfig()
    config_path = Path("/tmp/test_m1_config.yaml")
    config.to_yaml(config_path)

    # Load config using Pydantic
    loaded = MotorControllerConfig.from_yaml(config_path)

    assert loaded.simulation.step == config.simulation.step
    assert loaded.neurons.n_rec == config.neurons.n_rec
    assert loaded.rbf.num_centers == config.rbf.num_centers
    assert loaded.synapses.syn.optimizer.eta == config.synapses.syn.optimizer.eta
    assert loaded.synapses.static_delay == config.synapses.static_delay
    assert loaded.synapses.feedback_delay == config.synapses.feedback_delay
    assert loaded.synapses.rate_target_delay == config.synapses.rate_target_delay

    # Test converting back to dict
    config_dict = config.to_dict()
    assert isinstance(config_dict, dict)
    assert "simulation" in config_dict

    # Test modifying values
    config.neurons.n_rec = 500
    config.synapses.syn.optimizer.eta = 0.05
    assert config.neurons.n_rec == 500
    assert config.synapses.syn.optimizer.eta == 0.05

    # Cleanup
    config_path.unlink(missing_ok=True)
