#!/usr/bin/env bash

#SBATCH --job-name=m1_sweep
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --output=report/m1_sweep_%A_%a.out
#SBATCH --error=report/m1_sweep_%A_%a.err

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"

DEFAULT_SPEC_PATH="${REPO_ROOT}/experiments/epropplus_no_post_zero/sweep.yaml"
SPEC_PATH="${SPEC_PATH:-$DEFAULT_SPEC_PATH}"

REPORT_DIR="${REPO_ROOT}/report/slurm"
MAX_ARRAY_TASKS=1000
WORKER_MEM=64G

mkdir -p "$REPORT_DIR"

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

die() {
    echo "$*" >&2
    exit 1
}

parse_job_id() {
    awk '{print $4}' <<< "$1"
}

require_positive_int() {
    local value="$1"
    local name="$2"

    [[ "$value" =~ ^[0-9]+$ ]] && (( value > 0 )) \
        || die "$name must be a positive integer"
}

array_output_path() {
    echo "$REPORT_DIR/m1_sweep_%A_%a.out"
}

array_error_path() {
    echo "$REPORT_DIR/m1_sweep_%A_%a.err"
}

analysis_output_path() {
    echo "$REPORT_DIR/m1_sweep_analyze_%j.out"
}

analysis_error_path() {
    echo "$REPORT_DIR/m1_sweep_analyze_%j.err"
}

# ----------------------------------------------------------------------
# Sweep metadata
# ----------------------------------------------------------------------

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

cpus_per_task = int(
    base_cfg.get("simulation", {}).get("total_num_virtual_procs", 4)
)

repo_root = Path.cwd()

sweep_name = sanitize_run_name(
    spec.get("sweep_name") or spec_path.stem
)

output_dir_ref = Path(
    spec.get("output_dir", repo_root / "results" / "sweeps")
)

if output_dir_ref.is_absolute():
    output_dir = output_dir_ref
else:
    output_dir = (spec_path.parent / output_dir_ref).resolve()

sweep_root = (
    output_dir
    / sweep_name
    / datetime.now().strftime("%Y%m%d_%H%M%S")
).resolve()

print(runs)
print(cpus_per_task)
print(sweep_root)
PY
}

# ----------------------------------------------------------------------
# Submission helpers
# ----------------------------------------------------------------------

submit_array_chunk() {
    local chunk_start="$1"
    local chunk_end="$2"
    local cpus_per_task="$3"
    local sweep_root="$4"

    local chunk_size=$((chunk_end - chunk_start + 1))

    sbatch \
        --cpus-per-task "$cpus_per_task" \
        --mem "$WORKER_MEM" \
        --array "0-$((chunk_size - 1))" \
        --chdir "$REPO_ROOT" \
        --output "$(array_output_path)" \
        --error "$(array_error_path)" \
        --export "ALL,SPEC_PATH=$SPEC_PATH,SWEEP_ROOT=$sweep_root,REPO_ROOT=$REPO_ROOT,TASK_OFFSET=$chunk_start" \
        "$0"
}

submit_analysis_job() {
    local dependency_ids="$1"
    local sweep_root="$2"

    local submit_out

    submit_out="$(
        sbatch \
            --dependency "afterok:${dependency_ids}" \
            --job-name m1_sweep_analyze \
            --cpus-per-task 1 \
            --mem 4G \
            --time 00:30:00 \
            --output "$(analysis_output_path)" \
            --error "$(analysis_error_path)" \
            --wrap "bash -lc '
                cd \"$REPO_ROOT\"
                source \"$REPO_ROOT/env_load_hambach.sh\"
                python -m motor_controller_model.analyze_sweep \
                    --sweep-root \"$sweep_root\" \
                    --top-k 10 \
                    --promote
            '"
    )"

    local analysis_job_id
    analysis_job_id="$(parse_job_id "$submit_out")"

    if [[ "$analysis_job_id" =~ ^[0-9]+$ ]]; then
        echo "Submitted analysis job $analysis_job_id"
    else
        echo "Submitted analysis dependency: $submit_out"
    fi
}

# ----------------------------------------------------------------------
# Dispatcher
# ----------------------------------------------------------------------

dispatch() {
    mapfile -t meta < <(read_sweep_metadata)

    local runs="${meta[0]:-}"
    local cpus_per_task="${meta[1]:-}"
    local sweep_root="${meta[2]:-}"

    require_positive_int "$runs" "runs"
    require_positive_int "$cpus_per_task" "cpus_per_task"

    [[ -n "$sweep_root" ]] \
        || die "Could not resolve sweep output directory"

    mkdir -p "$sweep_root"

    echo "Dispatching $runs runs"
    echo "CPUs per task: $cpus_per_task"
    echo "Sweep root: $sweep_root"

    local -a array_job_ids=()

    local chunk_start=0

    while (( chunk_start < runs )); do
        local chunk_end=$((chunk_start + MAX_ARRAY_TASKS - 1))

        if (( chunk_end >= runs )); then
            chunk_end=$((runs - 1))
        fi

        local submit_out
        submit_out="$(
            submit_array_chunk \
                "$chunk_start" \
                "$chunk_end" \
                "$cpus_per_task" \
                "$sweep_root"
        )"

        local array_job_id
        array_job_id="$(parse_job_id "$submit_out")"

        [[ "$array_job_id" =~ ^[0-9]+$ ]] \
            || die "Could not parse job id from: $submit_out"

        echo "Submitted array job $array_job_id for tasks ${chunk_start}-${chunk_end}"

        array_job_ids+=("$array_job_id")

        chunk_start=$((chunk_end + 1))
    done

    local dependency
    dependency="$(IFS=:; echo "${array_job_ids[*]}")"

    submit_analysis_job "$dependency" "$sweep_root"
}

# ----------------------------------------------------------------------
# Worker
# ----------------------------------------------------------------------

run_worker() {
    [[ -n "${SWEEP_ROOT:-}" ]] \
        || die "SWEEP_ROOT is not set"

    local task_offset="${TASK_OFFSET:-0}"

    [[ "$task_offset" =~ ^[0-9]+$ ]] \
        || die "TASK_OFFSET must be a non-negative integer"

    local task_index=$((task_offset + SLURM_ARRAY_TASK_ID))

    echo "Worker ${SLURM_ARRAY_TASK_ID}"
    echo "Global task index: ${task_index}"
    echo "Working directory: $(pwd)"

    source "$REPO_ROOT/env_load_hambach.sh"

    echo "Python: $(which python)"
    echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}"

    python -c "import motor_controller_model; print('Module OK')" \
        || die "Failed to import motor_controller_model"

    python -m motor_controller_model.sweep \
        --spec "$SPEC_PATH" \
        --sweep-root "$SWEEP_ROOT" \
        --task-index "$task_index"
}

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

main() {
    [[ -f "$SPEC_PATH" ]] \
        || die "Sweep spec not found: $SPEC_PATH"

    cd "$REPO_ROOT"

    if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
        dispatch
    else
        run_worker
    fi
}

main "$@"