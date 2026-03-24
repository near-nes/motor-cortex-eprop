"""Run sequence timing experiment from a YAML config."""

from __future__ import annotations

from pathlib import Path

import structlog

from motor_controller_model.config_schema import MotorControllerConfig
from motor_controller_model.m1_factory import get_m1_or_train
from motor_controller_model.run_m1 import run_inference_test

LOG = structlog.get_logger("sequence_timing_experiment")

REPO_ROOT = Path(__file__).resolve().parents[2]

# -----------------------------------------------------------------------------
# Experiment settings
# -----------------------------------------------------------------------------
NEST_MODULE = str(
    REPO_ROOT
    / "motor_controller_model/nestml_neurons/nestml_install/motor_neuron_module.so"
)
OUTPUT_ROOT = REPO_ROOT / "results"
RUN_NAME = "legacy_sequence"
FORCE_RETRAIN = True
CONFIG_PATH = (
    REPO_ROOT / "experiments/legacy_sequence/legacy_like_1500_timephases.yaml"
)


def main() -> None:
    artifacts_dir = OUTPUT_ROOT / RUN_NAME
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Missing experiment config: {CONFIG_PATH}")

    config = MotorControllerConfig.from_yaml(CONFIG_PATH)

    network = get_m1_or_train(
        config=config,
        artifacts_dir=artifacts_dir,
        force_retrain=FORCE_RETRAIN,
        nest_module=NEST_MODULE,
    )

    run_inference_test(config, network, artifacts_dir, NEST_MODULE)

    LOG.info("experiment complete", output_dir=str(artifacts_dir))


if __name__ == "__main__":
    main()
