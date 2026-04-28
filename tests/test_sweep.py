from pathlib import Path

from motor_controller_model.config_schema import MotorControllerConfig
from motor_controller_model.sweep import (
    apply_dotted_override,
    build_run_name,
    expand_run_specs,
    materialize_config,
    compute_training_quality_metrics,
)


def test_apply_dotted_override_updates_nested_dict():
    data = {"task": {"n_iter": 200}, "synapses": {"w_input": 20.0}}
    apply_dotted_override(data, "task.n_iter", 25)
    apply_dotted_override(data, "synapses.w_input", 42.0)
    assert data["task"]["n_iter"] == 25
    assert data["synapses"]["w_input"] == 42.0


def test_build_run_name_is_stable_and_safe():
    name = build_run_name({"task.input_shift_ms": 100.0, "task.n_iter": 20}, prefix="legacy sweep")
    assert name == "legacy_sweep__task_input_shift_ms_100__task_n_iter_20"


def test_expand_run_specs_from_axes():
    spec = {
        "run_name_prefix": "smoke",
        "fixed_overrides": {"task.learning_start_ms": 650.0},
        "axes": {"task.input_shift_ms": [50.0, 100.0], "task.n_iter": [10, 20]},
    }
    runs = expand_run_specs(spec)
    assert len(runs) == 4
    assert runs[0]["name"].startswith("smoke__")
    assert runs[0]["overrides"]["task.learning_start_ms"] == 650.0


def test_materialize_config_preserves_validation():
    base_path = (
        Path(__file__).resolve().parent.parent
        / "experiments"
        / "legacy_sequence"
        / "legacy_like_1500_timephases.yaml"
    )
    base = MotorControllerConfig.from_yaml(base_path)
    config = materialize_config(base, {"task.input_shift_ms": 75.0, "task.n_iter": 5})
    assert config.task.input_shift_ms == 75.0
    assert config.task.n_iter == 5


def test_compute_training_quality_metrics_penalizes_spike_rate_cv():
    loss = [3.0, 2.0, 1.5, 1.0]

    low_cv = compute_training_quality_metrics(loss, n_samples=2, spike_rate_cv=0.1)
    high_cv = compute_training_quality_metrics(loss, n_samples=2, spike_rate_cv=2.0)

    assert high_cv["training_success_score"] > low_cv["training_success_score"]
