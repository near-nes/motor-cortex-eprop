module load stable/25.07 ias6 gcc
source "$(git rev-parse --show-toplevel)/venv/bin/activate"
export LD_LIBRARY_PATH="/users/shimoura/gits/motor-controller-model/nestml-neurons/nestml_install:$LD_LIBRARY_PATH"