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
# Slurm will schedule jobs based on available resources.

#SBATCH --job-name=m1_sweep
#SBATCH --time=08:00:00
#SBATCH --output=report/slurm/m1_sweep_%A_%a.out
#SBATCH --error=report/slurm/m1_sweep_%A_%a.err
#SBATCH --cpus-per-task=24
#SBATCH --mem=32G
#SBATCH --partition=blaustein

set -euo pipefail

# Get repo root from current working directory (SLURM_SUBMIT_DIR)
# This works because Slurm jobs start in the submission directory
REPO_ROOT="$(/usr/bin/git rev-parse --show-toplevel)"

DEFAULT_SPEC_PATH="${REPO_ROOT}/experiments/update_eprop_neuron_model_test/sweep_dynamics.yaml"
SPEC_PATH="$DEFAULT_SPEC_PATH"
MAX_ARRAY_TASKS=1000

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
  local array_job_ids="$1"
  local sweep_root="$2"

  local submit_out
  local report_dir="${REPO_ROOT}/report/slurm"
  mkdir -p "$report_dir"
  submit_out="$(sbatch \
    --dependency "afterok:${array_job_ids}" \
    --cpus-per-task 1 \
    --mem 4G \
    --time 00:30:00 \
    --partition blaustein \
    --job-name m1_sweep_analyze \
    --output "$report_dir/m1_sweep_analyze_%j.out" \
    --error "$report_dir/m1_sweep_analyze_%j.err" \
    --wrap "bash -lc 'cd \"$REPO_ROOT\" && source \"$REPO_ROOT/env_load_hambach.sh\" && python -m motor_controller_model.analyze_sweep --sweep-root \"$sweep_root\" --promote'")"

  local analysis_job_id
  analysis_job_id="$(parse_job_id "$submit_out")"
  if [[ "$analysis_job_id" =~ ^[0-9]+$ ]]; then
    echo "Submitted analysis job $analysis_job_id (afterok:$array_job_ids)"
  else
    echo "Submitted analysis dependency, raw response: $submit_out"
  fi
}

if [[ ! -f "$SPEC_PATH" ]]; then
  echo "Sweep spec not found: $SPEC_PATH" >&2
  exit 1
fi

REPORT_DIR="${REPO_ROOT}/report/slurm"
mkdir -p "$REPORT_DIR"

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

  if [[ -z "$SWEEP_ROOT" ]]; then
    echo "Could not resolve sweep output directory" >&2
    exit 1
  fi

  mkdir -p "$SWEEP_ROOT"

  echo "Dispatching $RUNS runs with cpus-per-task $CPUS_PER_TASK"
  echo "Results will be saved to: $SWEEP_ROOT"
  ARRAY_JOB_IDS=()
  CHUNK_START=0
  while (( CHUNK_START < RUNS )); do
    CHUNK_END=$((CHUNK_START + MAX_ARRAY_TASKS - 1))
    if (( CHUNK_END >= RUNS )); then
      CHUNK_END=$((RUNS - 1))
    fi

    CHUNK_SIZE=$((CHUNK_END - CHUNK_START + 1))
    ARRAY_SUBMIT_OUT="$(sbatch \
      --cpus-per-task "$CPUS_PER_TASK" \
      --array "0-$((CHUNK_SIZE - 1))" \
      --chdir "$REPO_ROOT" \
      --output "$REPORT_DIR/m1_sweep_%A_%a.out" \
      --error "$REPORT_DIR/m1_sweep_%A_%a.err" \
      --export "ALL,SPEC_PATH=$SPEC_PATH,SWEEP_ROOT=$SWEEP_ROOT,REPO_ROOT=$REPO_ROOT,TASK_OFFSET=$CHUNK_START" \
      "$0")"

    ARRAY_JOB_ID="$(parse_job_id "$ARRAY_SUBMIT_OUT")"
    if [[ ! "$ARRAY_JOB_ID" =~ ^[0-9]+$ ]]; then
      echo "Could not parse array job id from: $ARRAY_SUBMIT_OUT" >&2
      exit 1
    fi

    echo "Submitted array job $ARRAY_JOB_ID for tasks $CHUNK_START-$CHUNK_END"
    ARRAY_JOB_IDS+=("$ARRAY_JOB_ID")
    CHUNK_START=$((CHUNK_END + 1))
  done

  ANALYSIS_DEPENDENCY="$(IFS=:; echo "${ARRAY_JOB_IDS[*]}")"
  submit_analysis_job "$ANALYSIS_DEPENDENCY" "$SWEEP_ROOT"
  exit 0
fi

# Worker mode: run one condition for this array task.
if [[ -z "${SWEEP_ROOT:-}" ]]; then
  echo "SWEEP_ROOT is not set for worker mode" >&2
  exit 1
fi

TASK_OFFSET="${TASK_OFFSET:-0}"
if [[ ! "$TASK_OFFSET" =~ ^[0-9]+$ ]]; then
  echo "TASK_OFFSET must be a non-negative integer" >&2
  exit 1
fi

TASK_INDEX="$((TASK_OFFSET + SLURM_ARRAY_TASK_ID))"

cd "$REPO_ROOT"
echo "Worker $SLURM_ARRAY_TASK_ID (global $TASK_INDEX): Working directory: $(pwd)"

# Activate environment
source "$REPO_ROOT/env_load_hambach.sh"
echo "Worker $SLURM_ARRAY_TASK_ID: Python: $(which python)"
echo "Worker $SLURM_ARRAY_TASK_ID: LD_LIBRARY_PATH=$LD_LIBRARY_PATH"

# Verify module import
python -c "import motor_controller_model; print('Module OK')" || {
  echo "Worker $SLURM_ARRAY_TASK_ID: Failed to import motor_controller_model" >&2
  exit 1
}

python -m motor_controller_model.sweep \
  --spec "$SPEC_PATH" \
  --sweep-root "$SWEEP_ROOT" \
  --task-index "$TASK_INDEX"
