"""Unit tests for the post-training convergence check."""

import pytest

from motor_controller_model.config_schema import ConvergenceConfig
from motor_controller_model.convergence import (
    TrainingDidNotConverge,
    check_firing_rate,
)


@pytest.fixture
def cfg() -> ConvergenceConfig:
    return ConvergenceConfig()


def test_healthy_rate_passes(cfg):
    result = check_firing_rate(15.0, cfg)
    assert result.ok
    assert result.reason is None
    assert result.mean_firing_rate_hz == 15.0


def test_dead_rate_fails(cfg):
    result = check_firing_rate(2.0, cfg)
    assert not result.ok
    assert result.reason == "dead"
    assert "below" in result.detail


def test_diverged_rate_fails(cfg):
    result = check_firing_rate(80.0, cfg)
    assert not result.ok
    assert result.reason == "diverged"
    assert "exceeds" in result.detail


def test_boundary_min_passes(cfg):
    # Exactly at min is healthy (inclusive).
    result = check_firing_rate(cfg.min_firing_rate_hz, cfg)
    assert result.ok


def test_boundary_max_passes(cfg):
    # Exactly at max is healthy (inclusive).
    result = check_firing_rate(cfg.max_firing_rate_hz, cfg)
    assert result.ok


def test_custom_bounds():
    cfg = ConvergenceConfig(min_firing_rate_hz=10.0, max_firing_rate_hz=20.0)
    assert check_firing_rate(15.0, cfg).ok
    assert check_firing_rate(9.9, cfg).reason == "dead"
    assert check_firing_rate(20.1, cfg).reason == "diverged"


def test_exception_is_runtime_error():
    assert issubclass(TrainingDidNotConverge, RuntimeError)
