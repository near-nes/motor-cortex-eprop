#!/usr/bin/env bash
# Local sweep runner (no SLURM).
#
# Usage:
# bash experiments/epropplus_corrected_params/run_sweep_local.sh [--parallel N] [--spec SPEC_PATH]
#
# Examples:
# bash experiments/epropplus_corrected_params/run_sweep_local.sh --parallel 2
# bash experiments/epropplus_corrected_params/run_sweep_local.sh --parallel 3 --spec update_eprop_neuron_model_sweep.yaml

set -euo pipefail

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"

DEFAULT_SPEC_PATH="${REPO_ROOT}/experiments/epropplus_continuous_sequence_v2/sweep.yaml"
SPEC_PATH="$DEFAULT_SPEC_PATH"
PARALLEL_JOBS=4  # Conservative default for 16 cores, 20GB RAM

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --parallel)
      PARALLEL_JOBS="$2"
      shift 2
      ;;
    --spec)
      SPEC_PATH="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
  esac
done

read_sweep_metadata() {
  python - "$SPEC_PATH" <<'PY'
import itertools
import sys
from datetime import datetime
from pathlib import Path

import yaml


def sanitize_run_name(text: str) -> str:
    safe = []
    for char in text:
        if char.isalnum() or char in {"-", "_", "."}:
            safe.append(char)
        else:
            safe.append("_")
    cleaned = "".join(safe).strip("._")
    return cleaned or "run"


spec_path = Path(sys.argv[1]).resolve()
with open(spec_path, "r", encoding="utf-8") as f:
    spec = yaml.safe_load(f)

if "runs" in spec:
    runs = len(spec["runs"])
else:
    axes = spec.get("axes", {}) or {}
    values = [axes[key] for key in axes]
    runs = len(list(itertools.product(*values))) if values else 1

base_config_ref = Path(spec["base_config"])
if not base_config_ref.is_absolute():
    base_config_ref = (spec_path.parent / base_config_ref).resolve()
with open(base_config_ref, "r", encoding="utf-8") as f:
    base_cfg = yaml.safe_load(f)
cpus_per_task = int(base_cfg.get("simulation", {}).get("total_num_virtual_procs", 4))

repo_root = Path.cwd()
sweep_name = sanitize_run_name(spec.get("sweep_name") or spec_path.stem)
output_dir_ref = Path(spec.get("output_dir", repo_root / "results" / "sweeps"))
if output_dir_ref.is_absolute():
    output_dir = output_dir_ref
else:
    output_dir = (spec_path.parent / output_dir_ref).resolve()
sweep_root = (output_dir / sweep_name / datetime.now().strftime("%Y%m%d_%H%M%S")).resolve()

print(runs)
print(cpus_per_task)
print(sweep_root)
PY
}

run_analysis() {
  local sweep_root="$1"
  echo ""
  echo "========== Running analysis =========="
  cd "$REPO_ROOT"
  python -m motor_controller_model.analyze_sweep --sweep-root "$sweep_root" --promote
}

# Validation
if [[ ! -f "$SPEC_PATH" ]]; then
  echo "Sweep spec not found: $SPEC_PATH" >&2
  exit 1
fi

if [[ ! "$PARALLEL_JOBS" =~ ^[0-9]+$ ]] || (( PARALLEL_JOBS < 1 )); then
  echo "PARALLEL_JOBS must be a positive integer" >&2
  exit 1
fi

mkdir -p "${REPO_ROOT}/report/local"

cd "$REPO_ROOT"

# Read sweep metadata
mapfile -t SWEEP_META < <(read_sweep_metadata)
RUNS="${SWEEP_META[0]:-}"
CPUS_PER_TASK="${SWEEP_META[1]:-}"
SWEEP_ROOT="${SWEEP_META[2]:-}"

if [[ ! "$RUNS" =~ ^[0-9]+$ ]] || (( RUNS < 1 )); then
  echo "Could not determine a valid number of runs from $SPEC_PATH" >&2
  exit 1
fi

if [[ ! "$CPUS_PER_TASK" =~ ^[0-9]+$ ]] || (( CPUS_PER_TASK < 1 )); then
  echo "simulation.total_num_virtual_procs in base config must be a positive integer" >&2
  exit 1
fi

if (( PARALLEL_JOBS > RUNS )); then
  PARALLEL_JOBS="$RUNS"
fi

if [[ -z "$SWEEP_ROOT" ]]; then
  echo "Could not resolve sweep output directory" >&2
  exit 1
fi

mkdir -p "$SWEEP_ROOT"

echo "Starting local sweep: $RUNS runs with max concurrency $PARALLEL_JOBS (cpus-per-task: $CPUS_PER_TASK)"
echo "Sweep root: $SWEEP_ROOT"
echo ""

run_tasks() {
  # Run tasks with controlled parallelism using GNU parallel or xargs.
  if command -v parallel &> /dev/null; then
    seq 0 $((RUNS - 1)) | parallel --halt soon,fail=1 -j "$PARALLEL_JOBS" \
      "python -m motor_controller_model.sweep --spec '$SPEC_PATH' --sweep-root '$SWEEP_ROOT' --task-index {}"
  else
    seq 0 $((RUNS - 1)) | xargs -P "$PARALLEL_JOBS" -I {} \
      python -m motor_controller_model.sweep --spec "$SPEC_PATH" --sweep-root "$SWEEP_ROOT" --task-index {}
  fi
}

if run_tasks; then
  echo ""
  echo "========== Sweep completed successfully =========="
  run_analysis "$SWEEP_ROOT"
else
  echo ""
  echo "========== Sweep failed ==========" >&2
  exit 1
fi
