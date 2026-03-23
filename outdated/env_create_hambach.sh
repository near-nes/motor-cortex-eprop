#!/bin/bash
#SBATCH --job-name=env_inst
#SBATCH --output=nest_env_create_%j.out
#SBATCH --time=00:01:00

module load stable/25.07 ias6 gcc

# Create and activate the Python venv
python -m venv --system-site-packages venv
source "$(git rev-parse --show-toplevel)/venv/bin/activate"

python -c "import nest; print('nest imported successfully')"
