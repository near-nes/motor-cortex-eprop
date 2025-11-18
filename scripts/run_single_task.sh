#!/bin/bash
# This script either (a) prints the total number of parameter combinations (mode: count)
# or (b) runs a single task given a TASK_ID (default when passed a number).

# --- Define parameter grids (single source of truth) ---
learning_rates=(0.002 0.005 0.01 0.03 0.05)
n_recs=(100 200 300)
num_centers=(10 15 20)
w_inputs=(50.0 100.0 150.0 200.0)
w_recs=(40.0 60.0 80.0)
rbf_widths=(0.05 0.1 0.15 0.2 0.25)
n_iter=500   # Fixed (not scanned)
buffer_size=50  # Fixed (not scanned)

# Compute total combinations (excluding fixed n_iter)
total_combos=$(( ${#learning_rates[@]} * ${#n_recs[@]} * ${#num_centers[@]} * ${#w_inputs[@]} * ${#w_recs[@]} * ${#rbf_widths[@]} ))

if [[ "$1" == "count" || "$1" == "count_only" ]]; then
    echo "$total_combos"
    exit 0
fi

TASK_ID=$1
if [[ -z "$TASK_ID" ]]; then
    echo "Usage: $0 <TASK_ID>|count" >&2
    exit 1
fi
if ! [[ $TASK_ID =~ ^[0-9]+$ ]]; then
    echo "Error: TASK_ID must be a non-negative integer (or use 'count')." >&2
    exit 2
fi
if (( TASK_ID < 0 || TASK_ID >= total_combos )); then
    echo "Error: TASK_ID $TASK_ID out of range (0..$((total_combos-1)))." >&2
    exit 3
fi

# Calculate parameter indices for this task (mixed-radix decomposition)
lr_idx=$(( TASK_ID % ${#learning_rates[@]} ))
stride=$(( TASK_ID / ${#learning_rates[@]} ))
n_rec_idx=$(( stride % ${#n_recs[@]} ))
stride=$(( stride / ${#n_recs[@]} ))
num_centers_idx=$(( stride % ${#num_centers[@]} ))
stride=$(( stride / ${#num_centers[@]} ))
w_input_idx=$(( stride % ${#w_inputs[@]} ))
stride=$(( stride / ${#w_inputs[@]} ))
w_rec_idx=$(( stride % ${#w_recs[@]} ))
stride=$(( stride / ${#w_recs[@]} ))
rbf_width_idx=$(( stride % ${#rbf_widths[@]} ))

# Assign parameter values
lr=${learning_rates[$lr_idx]}
n_rec=${n_recs[$n_rec_idx]}
num_center=${num_centers[$num_centers_idx]}
w_input=${w_inputs[$w_input_idx]}
w_rec=${w_recs[$w_rec_idx]}
rbf_w=${rbf_widths[$rbf_width_idx]}

echo "Running task $TASK_ID / $((total_combos-1)) with parameters:"
echo "  learning_rate   = $lr"
echo "  neurons.n_rec   = $n_rec"
echo "  task.n_iter     = $n_iter"
echo "  rbf.num_centers = $num_center"
echo "  synapses.w_input= $w_input"
echo "  synapses.w_rec  = $w_rec"
echo "  rbf.width       = $rbf_w"
echo "  neurons.rb.buffer_size = $buffer_size"

scan_param_str="neurons.n_rec,task.n_iter,rbf.num_centers,synapses.w_input,synapses.w_rec,rbf.width,neurons.rb.buffer_size"
scan_values_str="$n_rec;$n_iter;$num_center;$w_input;$w_rec;$rbf_w;$buffer_size"

python -m motor_controller_model.eprop_reaching_task \
  --trajectory-files ../dataset_motor_training/stage1/trajectories_90_to_140.txt,../dataset_motor_training/stage1/trajectories_90_to_20.txt \
  --target-files ../dataset_motor_training/stage1/spikes_from_90_to_140.txt,../dataset_motor_training/stage1/spikes_from_90_to_20.txt \
  --learning-rate "$lr" \
  --scan-param "$scan_param_str" \
  --scan-values "$scan_values_str" \
  --plastic-input-to-rec \
  # --use-manual-rbf
