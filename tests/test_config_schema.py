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
    assert loaded.synapses.exc.optimizer.eta == config.synapses.exc.optimizer.eta
    assert loaded.synapses.static_delay == config.synapses.static_delay
    assert loaded.synapses.feedback_delay == config.synapses.feedback_delay
    assert loaded.synapses.rate_target_delay == config.synapses.rate_target_delay

    # Test converting back to dict
    config_dict = config.to_dict()
    assert isinstance(config_dict, dict)
    assert "simulation" in config_dict

    # Test modifying values
    config.neurons.n_rec = 500
    config.synapses.exc.optimizer.eta = 0.05
    assert config.neurons.n_rec == 500
    assert config.synapses.exc.optimizer.eta == 0.05

    # Cleanup
    config_path.unlink(missing_ok=True)


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
    assert "V_m" not in regular_params
    assert "V_m" not in adaptive_params
    assert adaptive_params["adapt_beta"] == 1.7
    assert adaptive_params["adapt_tau"] == 33.0
    assert "kappa" in adaptive_params
    assert "kappa_reg" in adaptive_params


def test_directional_synapse_weights_can_be_configured_independently():
    cfg = MotorControllerConfig.model_validate(
        {
            "synapses": {
                "exc": {"optimizer": {"eta": 0.01, "Wmin": 0.0, "Wmax": 1.0}},
                "rec_out": {"weight": 3.0},
                "out_rec": {"weight": 5.0},
            }
        }
    )

    assert cfg.synapses.exc.optimizer.eta == 0.01
    assert cfg.synapses.exc.optimizer.Wmax == 1.0
    assert cfg.synapses.rec_out.weight == 3.0
    assert cfg.synapses.out_rec.weight == 5.0


def test_background_poisson_defaults_and_override():
    default_cfg = MotorControllerConfig()
    assert default_cfg.background.enabled is False
    assert default_cfg.background.rate_hz == 0.0

    overridden_cfg = MotorControllerConfig.model_validate(
        {
            "background": {
                "enabled": True,
                "rate_hz": 15.0,
                "weight": 0.5,
                "delay": 2.0,
                "start_ms": 10.0,
                "stop_ms": 100.0,
            }
        }
    )

    poisson_cfg = overridden_cfg.synapses.background
    assert poisson_cfg.enabled is True
    assert poisson_cfg.rate_hz == 15.0
    assert poisson_cfg.weight == 0.5
    assert poisson_cfg.delay == 2.0
    assert poisson_cfg.start_ms == 10.0
    assert poisson_cfg.stop_ms == 100.0
