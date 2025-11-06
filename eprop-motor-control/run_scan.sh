#!/bin/bash
# NOTE: We intentionally do NOT hard-code --array here; submit script computes it.
#SBATCH --job-name=eprop_scan
#SBATCH --output=./report/eprop_scan_%A_%a.out
#SBATCH --error=./report/eprop_scan_%A_%a.err
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4    # 4 tasks per node
#SBATCH --cpus-per-task=32     # 32 cores per task
#SBATCH --mem=48G              # Total memory (adapt as needed)
#SBATCH --exclusive

# Source the Python environment and set LD_LIBRARY_PATH
source "$(git rev-parse --show-toplevel)/env_load_hambach.sh"

if [[ -z "$SLURM_ARRAY_TASK_ID" ]]; then
	echo "Error: SLURM_ARRAY_TASK_ID is not set. Did you forget to submit as an array?" >&2
	exit 10
fi

if [[ -n "${OFFSET:-}" ]]; then
	GLOBAL_ID=$(( OFFSET + SLURM_ARRAY_TASK_ID ))
else
	GLOBAL_ID=$SLURM_ARRAY_TASK_ID
fi
export GLOBAL_ID
echo "Running GLOBAL_ID=$GLOBAL_ID (array task $SLURM_ARRAY_TASK_ID offset ${OFFSET:-0})"
bash run_single_task.sh "$GLOBAL_ID"