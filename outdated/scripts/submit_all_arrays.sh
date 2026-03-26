#!/bin/bash
# Usage: ./submit_all_arrays.sh [--chunk-size N]
# Simple submission script: counts total jobs and submits one or more fixed-size arrays.
# Defaults to chunk size 1000 (common SLURM MaxArraySize). No auto-shrinking complexity.

set -euo pipefail

chunk_size=1000
while [[ $# -gt 0 ]]; do
	case $1 in
		--chunk-size)
			chunk_size=$2; shift 2 ;;
		-h|--help)
			echo "Usage: $0 [--chunk-size N]"; exit 0 ;;
		*) echo "Unknown argument: $1" >&2; exit 2 ;;
	esac
done

num_jobs=$(bash run_single_task.sh count)
if [[ -z "$num_jobs" || ! $num_jobs =~ ^[0-9]+$ ]]; then
	echo "Failed to obtain job count" >&2
	exit 1
fi

last=$(( num_jobs - 1 ))
echo "Total jobs: $num_jobs (index range 0-$last)" 

if (( num_jobs <= chunk_size )); then
	echo "Submitting single array 0-$last"
	sbatch --array=0-$last run_scan.sh
	exit 0
fi

echo "Submitting in chunks of $chunk_size (using OFFSET so each array starts at 0)"
offset=0
chunk=0
while (( offset < num_jobs )); do
	remaining=$(( num_jobs - offset ))
	this_size=$chunk_size
	if (( remaining < chunk_size )); then this_size=$remaining; fi
	last_local=$(( this_size - 1 ))
	echo "Chunk $chunk: global ${offset}-$(( offset + this_size - 1 )) as local 0-$last_local (OFFSET=$offset)"
	sbatch --export=ALL,OFFSET=${offset} --array=0-${last_local} run_scan.sh
	offset=$(( offset + this_size ))
	chunk=$(( chunk + 1 ))
done

echo "All chunks submitted."
