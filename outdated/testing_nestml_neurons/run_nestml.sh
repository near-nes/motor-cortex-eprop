#!/bin/bash
#SBATCH --job-name=nestml
#SBATCH --time=01:00:00
#SBATCH --output=nestml_compile_%j.out
#SBATCH --error=nestml_compile_%j.err

# Activate Python environment
source "$(git rev-parse --show-toplevel)/env_load_hambach.sh"

# Run the main Python script
python compile_nestml_neurons.py