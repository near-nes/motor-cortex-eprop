"""Analyze sweep outputs and report the best run.

This utility reads sweep metadata (``summary.json`` or ``manifest.json``)
and ranks completed runs by a numeric metric.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import yaml


def load_json(path: Path) -> Any:
    """Load JSON from disk."""

    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load JSONL records from disk."""

    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_spec(path: Path) -> dict[str, Any]:
    """Load a sweep spec YAML file."""

    with open(path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Sweep spec at {path} must be a mapping")
    return payload


def resolve_sweep_root(*, spec_path: Path | None, sweep_root: Path | None, latest: bool) -> Path:
    """Resolve sweep output root from explicit path or spec + latest folder."""

    if sweep_root is not None:
        return sweep_root.resolve()

    if spec_path is None:
        raise ValueError("Provide either --sweep-root, or --spec with --latest")

    spec = load_spec(spec_path)
    repo_root = Path(__file__).resolve().parent.parent
    output_dir_ref = Path(spec.get("output_dir", repo_root / "results" / "sweeps"))
    output_dir = (
        output_dir_ref
        if output_dir_ref.is_absolute()
        else (spec_path.parent / output_dir_ref).resolve()
    )
    sweep_name = str(spec.get("sweep_name") or spec_path.stem)
    candidates_root = (output_dir / sweep_name).resolve()

    if not latest:
        raise ValueError("When using --spec, also pass --latest or provide --sweep-root")

    if not candidates_root.exists():
        raise FileNotFoundError(f"Sweep directory not found: {candidates_root}")

    candidates = [p for p in candidates_root.iterdir() if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No timestamped sweep directories under: {candidates_root}")

    # Sweep directories use timestamp names (YYYYMMDD_HHMMSS), so lexical sort works.
    return sorted(candidates)[-1]


def load_records(sweep_root: Path) -> list[dict[str, Any]]:
    """Load run records from summary.json or manifest.json."""

    summary_path = sweep_root / "summary.json"
    if summary_path.exists():
        data = load_json(summary_path)
        if isinstance(data, list):
            return data
        raise TypeError(f"Expected list in {summary_path}")

    manifest_path = sweep_root / "manifest.json"
    if manifest_path.exists():
        return load_jsonl(manifest_path)

    task_results_dir = sweep_root / "task_results"
    if task_results_dir.exists():
        task_files = sorted(task_results_dir.glob("task_*.json"))
        if task_files:
            return [load_json(path) for path in task_files]

    raise FileNotFoundError(
        f"Could not find summary.json, manifest.json, or task_results/ under {sweep_root}"
    )


def extract_metric(record: dict[str, Any], metric_key: str) -> float | None:
    """Extract and normalize metric value from a record."""

    value = record.get(metric_key)
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def rank_completed_runs(
    records: list[dict[str, Any]], metric_key: str, mode: str
) -> list[tuple[float, dict[str, Any]]]:
    """Return completed runs with numeric metric values, sorted by rank."""

    completed = [record for record in records if record.get("status") == "completed"]
    if not completed:
        raise RuntimeError("No completed runs found")

    usable: list[tuple[float, dict[str, Any]]] = []
    for record in completed:
        metric_value = extract_metric(record, metric_key)
        if metric_value is not None:
            usable.append((metric_value, record))

    if not usable:
        raise RuntimeError(f"No completed runs with numeric metric '{metric_key}'")

    reverse = mode == "max"
    usable.sort(key=lambda item: item[0], reverse=reverse)
    return usable


def resolve_best_run_dir(best_record: dict[str, Any], sweep_root: Path) -> Path:
    """Resolve and validate directory path for the best run."""

    best_run_dir_raw = best_record.get("run_dir")
    if not isinstance(best_run_dir_raw, str) or not best_run_dir_raw:
        raise RuntimeError("Best run does not include a valid run_dir")

    best_run_dir = Path(best_run_dir_raw)
    if not best_run_dir.is_absolute():
        best_run_dir = (sweep_root / best_run_dir).resolve()
    if not best_run_dir.exists():
        raise FileNotFoundError(f"Best run directory not found: {best_run_dir}")
    return best_run_dir


def promote_best_run(
    best_record: dict[str, Any], sweep_root: Path, promote_dir: Path | None
) -> Path:
    """Copy best run artifacts into a stable destination directory."""

    best_run_dir = resolve_best_run_dir(best_record, sweep_root)

    default_promote_root = sweep_root / "best"
    destination = promote_dir.resolve() if promote_dir is not None else default_promote_root

    if destination.exists():
        shutil.rmtree(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(best_run_dir, destination)
    return destination


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description="Analyze a sweep and report the best run")
    parser.add_argument(
        "--sweep-root",
        type=Path,
        default=None,
        help="Path to one timestamped sweep output directory",
    )
    parser.add_argument(
        "--spec",
        type=Path,
        default=None,
        help="Sweep spec YAML (used with --latest to find newest sweep dir)",
    )
    parser.add_argument(
        "--latest",
        action="store_true",
        help="When used with --spec, analyze the latest timestamped sweep output",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="final_training_loss",
        help="Metric key in summary/manifest records (default: final_training_loss)",
    )
    parser.add_argument(
        "--mode",
        choices=["min", "max"],
        default="min",
        help="Whether lower or higher metric is better",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top runs to print (default: 5)",
    )
    parser.add_argument(
        "--promote",
        action="store_true",
        help="Copy the best run to a stable directory",
    )
    parser.add_argument(
        "--promote-dir",
        type=Path,
        default=None,
        help="Target directory for promoted best run (default: <sweep_root>/best)",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point."""

    args = parse_args()
    sweep_root = resolve_sweep_root(
        spec_path=args.spec.resolve() if args.spec is not None else None,
        sweep_root=args.sweep_root,
        latest=args.latest,
    )

    records = load_records(sweep_root)
    try:
        usable = rank_completed_runs(records, args.metric, args.mode)
    except RuntimeError as exc:
        raise RuntimeError(f"{exc} in {sweep_root}") from exc

    best_metric, best_record = usable[0]
    top_k = min(max(1, args.top_k), len(usable))
    top_rows = usable[:top_k]

    promoted_dir: Path | None = None
    if args.promote:
        promoted_dir = promote_best_run(best_record, sweep_root, args.promote_dir)

    print(f"Sweep root: {sweep_root}")
    print(
        f"Best run by {args.mode}({args.metric}): "
        f"{best_record.get('name')} ({best_metric:.8g})"
    )
    print("Top runs:")
    for rank, (metric_value, record) in enumerate(top_rows, start=1):
        runtime = record.get("runtime_s")
        runtime_text = f", runtime_s={runtime}" if runtime is not None else ""
        print(
            f"{rank}. name={record.get('name')}, {args.metric}={metric_value:.8g}{runtime_text}"
        )

    report_path = sweep_root / "best_run_report.json"
    report_payload = {
        "sweep_root": str(sweep_root),
        "metric": args.metric,
        "mode": args.mode,
        "top_k": top_k,
        "best": {
            "name": best_record.get("name"),
            "metric_value": best_metric,
            "run_dir": best_record.get("run_dir"),
            "input_config": best_record.get("input_config"),
            "overrides": best_record.get("overrides"),
        },
        "top_runs": [
            {
                "rank": rank,
                "name": record.get("name"),
                "metric_value": metric_value,
                "run_dir": record.get("run_dir"),
                "input_config": record.get("input_config"),
                "overrides": record.get("overrides"),
            }
            for rank, (metric_value, record) in enumerate(top_rows, start=1)
        ],
    }
    if promoted_dir is not None:
        report_payload["promoted_dir"] = str(promoted_dir)

    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report_payload, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(f"Wrote report: {report_path}")
    if promoted_dir is not None:
        print(f"Promoted best run to: {promoted_dir}")


if __name__ == "__main__":
    main()
