"""Post-training convergence verification.

A trained network is considered to have converged only if its mean recurrent
firing rate, measured over the last training iteration, falls within the
healthy band defined in :class:`ConvergenceConfig`. Out-of-band rates indicate
either a dead population (rate below ``min_firing_rate_hz``) or diverged
dynamics (rate above ``max_firing_rate_hz``).
"""

from __future__ import annotations

from dataclasses import dataclass

from .config_schema import ConvergenceConfig


class TrainingDidNotConverge(RuntimeError):
    """Raised when post-training convergence checks fail."""


@dataclass
class ConvergenceResult:
    ok: bool
    reason: str | None
    detail: str | None
    mean_firing_rate_hz: float


def check_firing_rate(
    mean_firing_rate_hz: float, cfg: ConvergenceConfig
) -> ConvergenceResult:
    """Classify a training run as converged / dead / diverged based on rate."""
    if mean_firing_rate_hz < cfg.min_firing_rate_hz:
        return ConvergenceResult(
            ok=False,
            reason="dead",
            detail=(
                f"mean recurrent firing rate {mean_firing_rate_hz:.2f} Hz is below "
                f"min_firing_rate_hz={cfg.min_firing_rate_hz} Hz"
            ),
            mean_firing_rate_hz=mean_firing_rate_hz,
        )
    if mean_firing_rate_hz > cfg.max_firing_rate_hz:
        return ConvergenceResult(
            ok=False,
            reason="diverged",
            detail=(
                f"mean recurrent firing rate {mean_firing_rate_hz:.2f} Hz exceeds "
                f"max_firing_rate_hz={cfg.max_firing_rate_hz} Hz"
            ),
            mean_firing_rate_hz=mean_firing_rate_hz,
        )
    return ConvergenceResult(
        ok=True, reason=None, detail=None, mean_firing_rate_hz=mean_firing_rate_hz
    )
