"""Config-driven sweep runner for motor controller experiments."""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from .config_schema import MotorControllerConfig


@dataclass(frozen=True)
class SweepRunSpec:
    """One materialized run inside a sweep."""

    index: int
    name: str
    overrides: dict[str, Any]
    config: MotorControllerConfig
    run_dir: Path


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
            raise TypeError(f"Cannot override {dotted_path!r}; {part!r} is not a mapping")
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


def expand_run_specs(spec: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand the sweep spec into concrete run specifications."""

    if "runs" in spec:
        run_specs: list[dict[str, Any]] = []
        for index, run in enumerate(spec["runs"]):
            if not isinstance(run, dict):
                raise TypeError("Each entry in 'runs' must be a mapping")
            overrides = dict(run.get("overrides", {}))
            name = run.get("name") or build_run_name(overrides, prefix=spec.get("run_name_prefix"))
            run_specs.append({"index": index, "name": sanitize_run_name(name), "overrides": overrides})
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


def materialize_config(base_config: MotorControllerConfig, overrides: dict[str, Any]) -> MotorControllerConfig:
    """Create a validated config object with dotted overrides applied."""

    data = base_config.model_dump()
    for key, value in overrides.items():
        apply_dotted_override(data, key, value)
    return MotorControllerConfig(**data)


def average_final_loss(loss: Any, n_samples: int) -> float:
    """Compute a robust final-loss summary from a training loss array."""

    import numpy as np

    loss_array = np.asarray(loss)
    if loss_array.size == 0:
        return float("nan")
    window = max(1, min(int(n_samples), int(loss_array.size)))
    return float(np.mean(loss_array[-window:]))


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

    import numpy as np

    loss_path = run_dir / "training_loss.npy"
    if loss_path.exists():
        loss = np.load(loss_path)
        run_meta["training_loss_points"] = int(loss.size)
        run_meta["final_training_loss"] = average_final_loss(loss, len(config.training.trajectories))
    else:
        run_meta["training_loss_points"] = 0
        run_meta["final_training_loss"] = None
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
    return parser.parse_args()


def main() -> None:
    """Run a sweep described by a YAML spec."""

    args = parse_args()
    spec = load_spec(args.spec)

    repo_root = Path(__file__).resolve().parent.parent
    base_config_ref = Path(spec["base_config"])
    base_config_path = base_config_ref if base_config_ref.is_absolute() else (args.spec.parent / base_config_ref).resolve()
    base_config = MotorControllerConfig.from_yaml(base_config_path)

    sweep_name = sanitize_run_name(spec.get("sweep_name") or args.spec.stem)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir_ref = Path(spec.get("output_dir", repo_root / "results" / "sweeps"))
    output_dir = output_dir_ref if output_dir_ref.is_absolute() else (args.spec.parent / output_dir_ref).resolve()
    sweep_root = output_dir / sweep_name / timestamp
    sweep_root.mkdir(parents=True, exist_ok=True)

    resolved_spec_path = sweep_root / "resolved_sweep_spec.yaml"
    resolved_spec = dict(spec)
    resolved_spec["base_config"] = str(base_config_path)
    resolved_spec["sweep_root"] = str(sweep_root)
    with open(resolved_spec_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(resolved_spec, handle, sort_keys=False)

    run_specs = expand_run_specs(spec)
    if args.task_index is not None:
        if args.task_index < 0 or args.task_index >= len(run_specs):
            raise IndexError(
                f"task-index {args.task_index} is out of range for {len(run_specs)} runs"
            )
        run_specs = [run_specs[args.task_index]]

    force_retrain = spec.get("force_retrain", True) if args.force_retrain is None else args.force_retrain
    nest_module = spec.get("nest_module", "motor_neuron_module")

    manifest_path = sweep_root / "manifest.json"
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
        with open(manifest_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(result, sort_keys=True) + "\n")

    with open(sweep_root / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")


if __name__ == "__main__":
    main()