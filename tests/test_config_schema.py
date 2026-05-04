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
    repo_root = Path(__file__).parent.parent
    candidate_paths = [
        repo_root / "motor_controller_model" / "config" / "config.yaml",
        repo_root
        / "experiments"
        / "legacy_sequence"
        / "legacy_like_1500_timephases.yaml",
        repo_root / "results" / "legacy_sequence" / "config.yaml",
    ]

    config_path = next((p for p in candidate_paths if p.exists()), None)
    if config_path is None:
        raise FileNotFoundError(
            "No valid config YAML found in expected locations: "
            + ", ".join(str(p) for p in candidate_paths)
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


def test_inhibitory_plasticity_flag_defaults_and_override():
    """Inhibitory synapse plasticity flag should default to False and allow override."""
    default_cfg = MotorControllerConfig()
    assert default_cfg.synapses.inh.plastic is False

    overridden_cfg = MotorControllerConfig.model_validate(
        {"synapses": {"inh": {"plastic": True}}}
    )
    assert overridden_cfg.synapses.inh.plastic is True


def test_adaptive_excitatory_fraction_derives_population_counts():
    cfg = MotorControllerConfig.model_validate(
        {
            "neurons": {
                "n_rec": 100,
                "exc_ratio": 0.8,
                "exc_adapt_ratio": 0.25,
            }
        }
    )

    assert cfg.neurons.n_exc == 80
    assert cfg.neurons.n_exc_adapt == 20
    assert cfg.neurons.n_exc_regular == 60


def test_recurrent_adaptive_params_are_dumped():
    cfg = MotorControllerConfig.model_validate(
        {
            "neurons": {
                "rec": {
                    "adapt_beta": 2.5,
                    "adapt_tau": 75.0,
                }
            }
        }
    )

    rec_dump = cfg.neurons.rec.model_dump()
    assert rec_dump["adapt_beta"] == 2.5
    assert rec_dump["adapt_tau"] == 75.0


def test_recurrent_to_nest_params_includes_adapt_only_when_requested():
    cfg = MotorControllerConfig.model_validate(
        {
            "simulation": {"step": 1.0},
            "neurons": {
                "rec": {
                    "adapt_beta": 1.7,
                    "adapt_tau": 33.0,
                    "eligibility_tau_ms": 20.0,
                    "tau_reg_ms": 30.0,
                }
            },
        }
    )

    regular_params = cfg.neurons.rec.to_nest_params(step_ms=cfg.simulation.step)
    adaptive_params = cfg.neurons.rec.to_nest_params(
        step_ms=cfg.simulation.step, include_adapt=True
    )

    assert "adapt_beta" not in regular_params
    assert "adapt_tau" not in regular_params
    assert adaptive_params["adapt_beta"] == 1.7
    assert adaptive_params["adapt_tau"] == 33.0
    assert "kappa" in adaptive_params
    assert "kappa_reg" in adaptive_params


def test_readout_synapse_optimizer_can_be_configured_independently():
    cfg = MotorControllerConfig.model_validate(
        {
            "synapses": {
                "exc": {"optimizer": {"eta": 0.01, "Wmin": 0.0, "Wmax": 1.0}},
                "readout": {
                    "optimizer": {"eta": 0.2, "Wmin": 0.0, "Wmax": 2.0}
                },
            }
        }
    )

    assert cfg.synapses.exc.optimizer.eta == 0.01
    assert cfg.synapses.readout.optimizer.eta == 0.2
    assert cfg.synapses.exc.optimizer.Wmax == 1.0
    assert cfg.synapses.readout.optimizer.Wmax == 2.0


if __name__ == "__main__":
    success = test_config_loading()
    exit(0 if success else 1)
