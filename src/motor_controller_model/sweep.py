"""Config-driven sweep runner for motor controller experiments."""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
import time
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from .config_schema import MotorControllerConfig


def get_git_commit_hash(repo_dir: Path) -> str:
    """Return the current git commit hash, or 'unknown' if git is unavailable."""

    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=repo_dir,
                stderr=subprocess.DEVNULL,
            )
            .decode("ascii")
            .strip()
        )
    except Exception:
        return "unknown"


def apply_dotted_override(data: dict[str, Any], dotted_path: str, value: Any) -> None:
    """Apply a dotted-path override to a nested dictionary."""

    parts = dotted_path.split(".")
    current = data
    for part in parts[:-1]:
        node = current.get(part)
        if node is None:
            node = {}
            current[part] = node
        if not isinstance(node, dict):
            raise TypeError(
                f"Cannot override {dotted_path!r}; {part!r} is not a mapping"
            )
        current = node
    current[parts[-1]] = value


def format_scalar(value: Any) -> str:
    """Format a value for filesystem-safe run names."""

    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return f"{value:g}"
    text = str(value)
    return text.replace("/", "-").replace(" ", "_")


def sanitize_run_name(text: str) -> str:
    """Convert free-form text to a filesystem-safe label."""

    safe = []
    for char in text:
        if char.isalnum() or char in {"-", "_", "."}:
            safe.append(char)
        else:
            safe.append("_")
    cleaned = "".join(safe).strip("._")
    return cleaned or "run"


def build_run_name(overrides: dict[str, Any], prefix: str | None = None) -> str:
    """Create a stable run name from dotted overrides."""

    parts: list[str] = []
    if prefix:
        parts.append(sanitize_run_name(prefix))
    for key in sorted(overrides):
        parts.append(
            f"{sanitize_run_name(key.replace('.', '_'))}_{sanitize_run_name(format_scalar(overrides[key]))}"
        )
    return "__".join(parts) if parts else "run"


def load_spec(path: Path) -> dict[str, Any]:
    """Load a sweep spec from YAML."""

    with open(path, "r", encoding="utf-8") as handle:
        spec = yaml.safe_load(handle)
    if not isinstance(spec, dict):
        raise TypeError(f"Sweep spec at {path} must be a mapping")
    return spec


def resolve_base_config_path(spec: dict[str, Any], spec_path: Path) -> Path:
    """Resolve absolute path to base config from sweep spec."""

    base_config_ref = Path(spec["base_config"])
    if base_config_ref.is_absolute():
        return base_config_ref
    return (spec_path.parent / base_config_ref).resolve()


def resolve_sweep_root(
    *,
    spec: dict[str, Any],
    spec_path: Path,
    repo_root: Path,
    explicit_sweep_root: Path | None,
) -> Path:
    """Resolve output directory for this sweep run."""

    if explicit_sweep_root is not None:
        return explicit_sweep_root.resolve()

    sweep_name = sanitize_run_name(spec.get("sweep_name") or spec_path.stem)
    output_dir_ref = Path(spec.get("output_dir", repo_root / "results" / "sweeps"))
    output_dir = (
        output_dir_ref
        if output_dir_ref.is_absolute()
        else (spec_path.parent / output_dir_ref).resolve()
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return output_dir / sweep_name / timestamp


def write_resolved_spec(
    *,
    sweep_root: Path,
    spec: dict[str, Any],
    base_config_path: Path,
) -> None:
    """Write resolved sweep spec metadata once per sweep directory."""

    resolved_spec_path = sweep_root / "resolved_sweep_spec.yaml"
    if resolved_spec_path.exists():
        return

    resolved_spec = dict(spec)
    resolved_spec["base_config"] = str(base_config_path)
    resolved_spec["sweep_root"] = str(sweep_root)
    with open(resolved_spec_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(resolved_spec, handle, sort_keys=False)


def write_task_result(
    sweep_root: Path, task_index: int, result: dict[str, Any]
) -> None:
    """Write one result file for a single task-index run."""

    task_results_dir = sweep_root / "task_results"
    task_results_dir.mkdir(parents=True, exist_ok=True)
    task_result_path = task_results_dir / f"task_{task_index:05d}.json"
    with open(task_result_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_sweep_outputs(sweep_root: Path, summary: list[dict[str, Any]]) -> None:
    """Write summary and manifest for a full sweep run."""

    manifest_path = sweep_root / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as handle:
        for result in summary:
            handle.write(json.dumps(result, sort_keys=True) + "\n")

    with open(sweep_root / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")


def expand_run_specs(spec: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand the sweep spec into concrete run specifications."""

    if "runs" in spec:
        run_specs: list[dict[str, Any]] = []
        for index, run in enumerate(spec["runs"]):
            if not isinstance(run, dict):
                raise TypeError("Each entry in 'runs' must be a mapping")
            overrides = dict(run.get("overrides", {}))
            name = run.get("name") or build_run_name(
                overrides, prefix=spec.get("run_name_prefix")
            )
            run_specs.append(
                {
                    "index": index,
                    "name": sanitize_run_name(name),
                    "overrides": overrides,
                }
            )
        return run_specs

    axes = spec.get("axes", {})
    if not isinstance(axes, dict):
        raise TypeError("'axes' must be a mapping")

    keys = list(axes.keys())
    values = [axes[key] for key in keys]
    run_specs = []
    for index, combo in enumerate(itertools.product(*values) if values else [()]):
        overrides = dict(spec.get("fixed_overrides", {}))
        overrides.update(dict(zip(keys, combo)))
        name = build_run_name(overrides, prefix=spec.get("run_name_prefix"))
        run_specs.append({"index": index, "name": name, "overrides": overrides})
    return run_specs


def materialize_config(
    base_config: MotorControllerConfig, overrides: dict[str, Any]
) -> MotorControllerConfig:
    """Create a validated config object with dotted overrides applied."""

    data = base_config.model_dump()
    for key, value in overrides.items():
        apply_dotted_override(data, key, value)
    return MotorControllerConfig(**data)


def average_final_loss(loss: Any, n_samples: int) -> float:
    """Compute a robust final-loss summary from a training loss array."""

    loss_array = np.asarray(loss)
    if loss_array.size == 0:
        return float("nan")
    window = max(1, min(int(n_samples), int(loss_array.size)))
    return float(np.mean(loss_array[-window:]))


def compute_training_quality_metrics(
    loss: Any,
    n_samples: int,
    mean_firing_rate_hz: float | None = None,
    spike_rate_cv: float | None = None,
) -> dict[str, float | bool]:
    """Compute training-only quality metrics from the loss curve.

    These metrics are intended to judge training success without inference tests.
    Lower ``training_success_score`` is better.
    """

    loss_array = np.asarray(loss, dtype=float)
    if loss_array.size == 0:
        return {
            "best_training_loss": float("nan"),
            "initial_training_loss": float("nan"),
            "training_improvement_ratio": float("nan"),
            "final_to_best_ratio": float("nan"),
            "last_window_cv": float("nan"),
            "last_window_slope": float("nan"),
            "spike_rate_cv": float("nan"),
            "training_success_score": float("nan"),
            "training_success": False,
        }

    eps = 1e-12
    window = max(1, min(int(n_samples), int(loss_array.size)))
    final_loss = float(np.mean(loss_array[-window:]))
    best_loss = float(np.min(loss_array))
    initial_loss = float(np.mean(loss_array[:window]))

    improvement_ratio = float((initial_loss - final_loss) / (abs(initial_loss) + eps))
    final_to_best_ratio = float(final_loss / (best_loss + eps))

    last_window = loss_array[-window:]
    last_mean = float(np.mean(last_window))
    last_std = float(np.std(last_window))
    last_cv = float(last_std / (abs(last_mean) + eps))

    x = np.arange(loss_array.size, dtype=float)
    slope = float(np.polyfit(x, loss_array, 1)[0]) if loss_array.size >= 2 else 0.0

    # Lower is better: penalize noisy/unstable tails and lack of improvement.
    success_score = float(
        final_loss
        * (1.0 + max(0.0, final_to_best_ratio - 1.0))
        * (1.0 + last_cv)
        / (1.0 + max(0.0, improvement_ratio))
    )

    # Biological activity penalty: heavily penalize if mean firing rate is too low or too high.
    rate_penalty = 1.0
    biological_success = True

    if mean_firing_rate_hz is not None:
        min_healthy_rate = 2.0  # Hz (lower bound)
        max_healthy_rate = 40.0  # Hz (upper bound)

        if mean_firing_rate_hz < min_healthy_rate:
            # Penalize severely if network is almost dead
            rate_penalty = 1.0 + (min_healthy_rate - mean_firing_rate_hz) * 10.0
            biological_success = False
        elif mean_firing_rate_hz > max_healthy_rate:
            # Penalize if network is firing too fast
            rate_penalty = 1.0 + (mean_firing_rate_hz - max_healthy_rate) * 0.5
            biological_success = False

    # Apply penalty (higher score is worse)
    success_score *= rate_penalty

    if spike_rate_cv is not None and np.isfinite(spike_rate_cv):
        # Penalize uneven firing across recurrent neurons.
        success_score *= 1.0 + max(0.0, float(spike_rate_cv))

    # Conservative training-only success gate.
    success = bool(
        np.isfinite(success_score)
        and (improvement_ratio >= 0.1)
        and (final_to_best_ratio <= 1.2)
        and (last_cv <= 0.25)
        and biological_success  # Must also have healthy biological activity
    )

    return {
        "best_training_loss": best_loss,
        "initial_training_loss": initial_loss,
        "training_improvement_ratio": improvement_ratio,
        "final_to_best_ratio": final_to_best_ratio,
        "last_window_cv": last_cv,
        "last_window_slope": slope,
        "mean_firing_rate_hz": (
            mean_firing_rate_hz if mean_firing_rate_hz is not None else float("nan")
        ),
        "spike_rate_cv": (
            spike_rate_cv if spike_rate_cv is not None else float("nan")
        ),
        "training_success_score": success_score,
        "training_success": success,
    }


def run_single_spec(
    *,
    repo_root: Path,
    sweep_root: Path,
    base_config: MotorControllerConfig,
    run_spec: dict[str, Any],
    base_config_path: Path,
    nest_module: str,
    force_retrain: bool,
    dry_run: bool,
) -> dict[str, Any]:
    """Materialize, launch, and summarize one run."""

    run_dir = sweep_root / run_spec["name"]
    run_dir.mkdir(parents=True, exist_ok=True)

    config = materialize_config(base_config, run_spec["overrides"])
    config.git_commit = get_git_commit_hash(repo_root)

    input_config_path = run_dir / "config.yaml"
    config.to_yaml(input_config_path)

    run_meta: dict[str, Any] = {
        "index": run_spec["index"],
        "name": run_spec["name"],
        "overrides": run_spec["overrides"],
        "config_hash": config.hash(),
        "git_commit": config.git_commit,
        "input_config": str(input_config_path),
        "base_config": str(base_config_path),
        "run_dir": str(run_dir),
        "status": "dry_run" if dry_run else "pending",
    }

    with open(run_dir / "run_spec.json", "w", encoding="utf-8") as handle:
        json.dump(run_meta, handle, indent=2, sort_keys=True)
        handle.write("\n")

    if dry_run:
        return run_meta

    command = [
        sys.executable,
        "-m",
        "motor_controller_model.run_m1",
        "--config",
        str(input_config_path),
        "--output-dir",
        str(sweep_root),
        "--run-name",
        run_spec["name"],
        "--nest-module",
        nest_module,
    ]
    if force_retrain:
        command.append("--force-retrain")

    started_at = time.time()
    result = subprocess.run(command, cwd=repo_root, check=False)
    run_meta["returncode"] = result.returncode
    run_meta["runtime_s"] = round(time.time() - started_at, 3)

    if result.returncode != 0:
        run_meta["status"] = "failed"
        with open(run_dir / "run_spec.json", "w", encoding="utf-8") as handle:
            json.dump(run_meta, handle, indent=2, sort_keys=True)
            handle.write("\n")
        return run_meta

    loss_path = run_dir / "training_loss.npy"
    if loss_path.exists():
        loss = np.load(loss_path)
        run_meta["training_loss_points"] = int(loss.size)
        run_meta["final_training_loss"] = average_final_loss(
            loss, len(config.training.trajectories)
        )

        # Attempt to load mean firing rate if available, and include it in the training quality metrics.
        rate_path = run_dir / "mean_firing_rate.json"
        mean_rate = None
        spike_cv = None
        if rate_path.exists():
            with open(rate_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
                mean_rate = payload.get("mean_firing_rate_hz")
                spike_cv = payload.get("spike_rate_cv")

        run_meta.update(
            compute_training_quality_metrics(
                loss, len(config.training.trajectories), mean_rate, spike_cv
            )
        )
    else:
        run_meta["training_loss_points"] = 0
        run_meta["final_training_loss"] = None
        run_meta["training_success"] = False
        run_meta["training_success_score"] = None
    run_meta["status"] = "completed"

    with open(run_dir / "run_spec.json", "w", encoding="utf-8") as handle:
        json.dump(run_meta, handle, indent=2, sort_keys=True)
        handle.write("\n")

    return run_meta


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description="Run config-driven parameter sweeps")
    parser.add_argument("--spec", type=Path, required=True, help="Sweep spec YAML file")
    parser.add_argument(
        "--task-index",
        type=int,
        default=None,
        help="Run only one sweep task by zero-based index",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Materialize configs and manifest entries without launching training",
    )
    parser.add_argument(
        "--force-retrain",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Force retraining for every run",
    )
    parser.add_argument(
        "--sweep-root",
        type=Path,
        default=None,
        help="Explicit sweep output directory (useful for Slurm arrays)",
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Run post-sweep analysis after completion (ignored with --task-index)",
    )
    parser.add_argument(
        "--promote",
        action="store_true",
        help="When used with --analyze, also promote best run into <sweep_root>/best",
    )
    return parser.parse_args()


def main() -> None:
    """Run a sweep described by a YAML spec."""

    args = parse_args()
    spec = load_spec(args.spec)

    repo_root = Path(__file__).resolve().parent.parent.parent
    base_config_path = resolve_base_config_path(spec, args.spec)
    base_config = MotorControllerConfig.from_yaml(base_config_path)

    sweep_root = resolve_sweep_root(
        spec=spec,
        spec_path=args.spec,
        repo_root=repo_root,
        explicit_sweep_root=args.sweep_root,
    )
    sweep_root.mkdir(parents=True, exist_ok=True)
    write_resolved_spec(
        sweep_root=sweep_root, spec=spec, base_config_path=base_config_path
    )

    run_specs = expand_run_specs(spec)
    if args.task_index is not None:
        if args.task_index < 0 or args.task_index >= len(run_specs):
            raise IndexError(
                f"task-index {args.task_index} is out of range for {len(run_specs)} runs"
            )
        run_specs = [run_specs[args.task_index]]

    force_retrain = (
        spec.get("force_retrain", True)
        if args.force_retrain is None
        else args.force_retrain
    )
    nest_module = spec.get("nest_module", "motor_neuron_module")

    summary: list[dict[str, Any]] = []
    for run_spec in run_specs:
        result = run_single_spec(
            repo_root=repo_root,
            sweep_root=sweep_root,
            base_config=base_config,
            run_spec=run_spec,
            base_config_path=base_config_path,
            nest_module=nest_module,
            force_retrain=force_retrain,
            dry_run=args.dry_run,
        )
        summary.append(result)

    if args.task_index is not None:
        write_task_result(sweep_root, args.task_index, summary[0])
        return

    write_sweep_outputs(sweep_root, summary)

    if args.analyze:
        command = [
            sys.executable,
            "-m",
            "motor_controller_model.analyze_sweep",
            "--sweep-root",
            str(sweep_root),
        ]
        if args.promote:
            command.append("--promote")
        subprocess.run(command, cwd=repo_root, check=True)


if __name__ == "__main__":
    main()
