#!/usr/bin/env bash
# Single-file Slurm sweep launcher.
#
# Usage:
# sbatch experiments/update_eprop_neuron_model_test/submit_sweep_slurm.sh
#
# Behavior:
# - Dispatcher job (no SLURM_ARRAY_TASK_ID):
#   - reads sweep metadata from YAML files
#   - submits an array job of this same script
#   - submits dependent best-run analysis job
# - Worker job (with SLURM_ARRAY_TASK_ID): runs one sweep condition.
#
# To change parallelism, edit PARALLEL_JOBS below.

#SBATCH --job-name=m1_sweep
#SBATCH --output=report/slurm/m1_sweep_%A_%a.out
#SBATCH --error=report/slurm/m1_sweep_%A_%a.err
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G

set -euo pipefail

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"

DEFAULT_SPEC_PATH="${REPO_ROOT}/experiments/update_eprop_neuron_model_test/update_eprop_neuron_model_sweep.yaml"
SPEC_PATH="$DEFAULT_SPEC_PATH"
PARALLEL_JOBS=20

parse_job_id() {
  local submit_output="$1"
  echo "$submit_output" | awk '{print $4}'
}

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

submit_analysis_job() {
  local array_job_id="$1"
  local sweep_root="$2"

  local submit_out
  submit_out="$(sbatch \
    --dependency "afterok:${array_job_id}" \
    --cpus-per-task 1 \
    --mem 4G \
    --time 00:30:00 \
    --job-name m1_sweep_analyze \
    --output report/slurm/m1_sweep_analyze_%j.out \
    --error report/slurm/m1_sweep_analyze_%j.err \
    --wrap "cd '$REPO_ROOT' && python -m motor_controller_model.analyze_sweep --sweep-root '$sweep_root' --promote")"

  local analysis_job_id
  analysis_job_id="$(parse_job_id "$submit_out")"
  if [[ "$analysis_job_id" =~ ^[0-9]+$ ]]; then
    echo "Submitted analysis job $analysis_job_id (afterok:$array_job_id)"
  else
    echo "Submitted analysis dependency, raw response: $submit_out"
  fi
}

if [[ ! -f "$SPEC_PATH" ]]; then
  echo "Sweep spec not found: $SPEC_PATH" >&2
  exit 1
fi

if [[ ! "$PARALLEL_JOBS" =~ ^[0-9]+$ ]] || (( PARALLEL_JOBS < 1 )); then
  echo "PARALLEL_JOBS must be a positive integer" >&2
  exit 1
fi

mkdir -p "${REPO_ROOT}/report/slurm"

cd "$REPO_ROOT"

# Dispatcher mode: automatically submit correctly sized array.
if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
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

  echo "Dispatching $RUNS runs with max concurrency $PARALLEL_JOBS and cpus-per-task $CPUS_PER_TASK"
  ARRAY_SUBMIT_OUT="$(sbatch \
    --cpus-per-task "$CPUS_PER_TASK" \
    --array "0-$((RUNS - 1))%${PARALLEL_JOBS}" \
    --export "ALL,SPEC_PATH=$SPEC_PATH,SWEEP_ROOT=$SWEEP_ROOT" \
    "$SCRIPT_PATH")"

  ARRAY_JOB_ID="$(parse_job_id "$ARRAY_SUBMIT_OUT")"
  if [[ ! "$ARRAY_JOB_ID" =~ ^[0-9]+$ ]]; then
    echo "Could not parse array job id from: $ARRAY_SUBMIT_OUT" >&2
    exit 1
  fi

  echo "Submitted array job $ARRAY_JOB_ID"
  submit_analysis_job "$ARRAY_JOB_ID" "$SWEEP_ROOT"
  exit 0
fi

# Worker mode: run one condition for this array task.
if [[ -z "${SWEEP_ROOT:-}" ]]; then
  echo "SWEEP_ROOT is not set for worker mode" >&2
  exit 1
fi

python -m motor_controller_model.sweep \
  --spec "$SPEC_PATH" \
  --sweep-root "$SWEEP_ROOT" \
  --task-index "$SLURM_ARRAY_TASK_ID"
