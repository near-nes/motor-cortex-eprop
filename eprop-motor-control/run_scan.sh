#!/bin/bash
#SBATCH --job-name=epropnet
#SBATCH --output=./report/nest_check_%j.out
#SBATCH --error=./report/nest_check_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G

# Ensure output directory exists
mkdir -p report

# Activate Python environment
source "$(git rev-parse --show-toplevel)/env_load_hambach.sh"
export LD_LIBRARY_PATH="/users/shimoura/gits/motor-controller-model/nestml-neurons/nestml_install:$LD_LIBRARY_PATH"

# Run Python scripts
python eprop-reaching-task.py --use-manual-rbf &
python eprop-reaching-task.py &
wait
python trained_weights_net.py