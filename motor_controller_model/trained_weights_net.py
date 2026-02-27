
"""
trained_weights_net.py
----------------------

This module loads trained weights from a previously trained e-prop motor control network
and runs a test simulation using NEST. It reproduces the network setup and input encoding
used in the main training module (eprop_reaching_task.py) for consistency.

Key steps:
- Loads configuration parameters from config.yaml.
- Loads trained weights from .npz file.
- Loads spike input data (same format as training).
- Sets up NEST simulation with neuron populations and connections.
- Applies loaded weights to recurrent and output connections.
- Runs the simulation and plots spike raster and loaded weight matrices.

Input connections:
- Parrot neurons relay spike input from planner neurons to rb_neurons.
- Each rb_neuron is connected to a group of excitatory and inhibitory recurrent neurons,
  matching the grouping logic in eprop_reaching_task.py.

This module is intended for post-training evaluation and visualization, using the same
spike input format as the training procedure.

Run as a module:
    python -m motor_controller_model.trained_weights_net
Outputs are saved in sim_results/ at the repository root.

Author: Renan Oliveira Shimoura
"""

import numpy as np
import nest
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
import sys
import json

from motor_controller_model.config_schema import MotorControllerConfig

mpl.use("Agg")

# Add dataset path for imports
from motor_controller_model.dataset_motor_training.load_dataset import load_data_file

# --------------------------------------------------------------------------------------
# Load configuration parameters from tutorial_results/config.yaml
# --------------------------------------------------------------------------------------

results_dir = Path(__file__).resolve().parent.parent / "tutorial_results"
config_path = results_dir / "config.yaml"

if config_path.exists():
    config_obj = MotorControllerConfig.from_yaml(config_path)
    config = config_obj.to_dict()
else:
    print(f"Warning: Config file not found at {config_path}. Using schema defaults.")
    config_obj = MotorControllerConfig()
    config = config_obj.to_dict()

# Use same variable names as eprop-reaching-task.py for consistency
n_rec = int(config["neurons"]["n_rec"])
n_out = int(config["neurons"]["n_out"])
step_ms = float(config["simulation"]["step"])
sequence = float(config["task"]["sequence"])
silent_period = float(config["task"]["silent_period"])
n_iter = 1  # Only one iteration for testing
num_centers = int(config["rbf"]["num_centers"])
scale_rate = float(config["rbf"]["scale_rate"])

# Duration matches eprop-reaching-task.py
n_timesteps_per_sequence = int(round((sequence + silent_period) / step_ms))
n_samples_per_trajectory_to_use = int(config["task"]["n_samples_per_trajectory_to_use"])
trajectory_ids_to_use = config["task"]["trajectory_ids_to_use"]
n_samples = len(trajectory_ids_to_use) * n_samples_per_trajectory_to_use
duration_task = n_timesteps_per_sequence * n_samples * n_iter * step_ms
duration = duration_task + step_ms

# --------------------------------------------------------------------------------------
# Load trained weights from .npz file
# --------------------------------------------------------------------------------------

weights_path = (
    Path(__file__).resolve().parent.parent
    / "tutorial_results"
    / "trained_weights.npz"
)
if not weights_path.exists():
    raise FileNotFoundError(f"Trained weights file not found: {weights_path}")

weights = np.load(weights_path, allow_pickle=True)
rec_rec_weights = weights.get("rec_rec").item() # Store recurrent network weights as a dictionary
rec_out_weights = weights.get("rec_out").item() # Store rec to out weights as a dictionary
rb_rec_weights = weights.get("rb_rec").item()   # Store input to rec weights as a dictionary

if rec_rec_weights is None or rec_out_weights is None:
    raise KeyError("Missing 'rec_rec' or 'rec_out' arrays in trained_weights.npz")

# --------------------------------------------------------------------------------------
# Load spike input data (same format as training)
# --------------------------------------------------------------------------------------

def load_spike_data(file_path):
    """
    Load spike data from various file formats.
    
    Supports:
    1. NEST .dat format with headers (sender, time_ms columns)
    2. Comma-separated format (neuron_id,spike_time)
    3. Whitespace-separated format (neuron_id spike_time)
    
    Args:
        file_path (str): Path to the spike data file
        
    Returns:
        numpy.ndarray: 2D array with columns [neuron_id, spike_time]
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Count lines starting with # and check for NEST format
    skip_lines = 0
    has_nest_header = False
    
    for i, line in enumerate(lines):
        if line.startswith('#'):
            skip_lines += 1
        elif 'sender' in line and 'time_ms' in line:
            # Found the NEST column header line
            skip_lines = i + 1
            has_nest_header = True
            break
        else:
            # First non-comment line without NEST header
            break
    
    # Check if it's NEST .dat format
    if has_nest_header:
        # NEST format - skip all comment and header lines
        data = np.loadtxt(file_path, skiprows=skip_lines)
        return data  # Already in [sender, time_ms] format
    
    # Check if comma-separated (look at first non-comment line)
    first_data_line = lines[skip_lines] if skip_lines < len(lines) else lines[0]
    if ',' in first_data_line:
        data = np.loadtxt(file_path, delimiter=',', skiprows=skip_lines)
        return data
    
    # Otherwise assume whitespace-separated
    else:
        data = np.loadtxt(file_path, skiprows=skip_lines)
        return data

# Define spike input file paths (matching the tutorial notebook)
base_data_dir = (
    Path(__file__).resolve().parent
    / "dataset_motor_training"
    / "input_ouput_data"
)

# Trajectory 1: 9020 (90° → 20°)
input_9020_pos = str(base_data_dir / 'N200_9020_planner_p.dat')
input_9020_neg = str(base_data_dir / 'N200_9020_planner_n.dat')

# Trajectory 2: 90140 (90° → 140°)
input_90140_pos = str(base_data_dir / 'N200_90140_planner_p.dat')
input_90140_neg = str(base_data_dir / 'N200_90140_planner_n.dat')

# List of input spike file pairs (pos, neg)
input_spike_files = [
    (input_9020_pos, input_9020_neg),
    (input_90140_pos, input_90140_neg),
]

# Load and organize spike input data (matching eprop_reaching_task.py approach)
input_pos_data_all = []
input_neg_data_all = []

for input_pos_file, input_neg_file in input_spike_files:
    spikes_pos = load_spike_data(input_pos_file)
    spikes_neg = load_spike_data(input_neg_file)
    input_pos_data_all.append(spikes_pos)
    input_neg_data_all.append(spikes_neg)

# Get unique senders and create neuron mapping (same as training code)
all_senders_pos = set()
all_senders_neg = set()
for data in input_pos_data_all:
    all_senders_pos.update(data[:, 0].astype(int))
for data in input_neg_data_all:
    all_senders_neg.update(data[:, 0].astype(int))

n_input_neurons = len(all_senders_pos) + len(all_senders_neg)
sender_to_idx = {}
for idx, sender in enumerate(sorted(all_senders_pos) + sorted(all_senders_neg)):
    sender_to_idx[sender] = idx

# Collect spike times across all trajectories with time offsets (same as training)
spike_times_per_neuron = [[] for _ in range(n_input_neurons)]
n_trajectories = len(input_pos_data_all)

for iter_num in range(n_iter):
    for traj_idx in range(n_trajectories):
        time_offset = (traj_idx + iter_num * n_trajectories) * n_timesteps_per_sequence * step_ms
        for data in [input_pos_data_all[traj_idx], input_neg_data_all[traj_idx]]:
            for sender_id, spike_time in data:
                neuron_idx = sender_to_idx[int(sender_id)]
                adjusted_time = spike_time + time_offset + silent_period
                spike_times_per_neuron[neuron_idx].append(adjusted_time)

# Compute total duration for both trajectories
duration = n_trajectories * n_timesteps_per_sequence * step_ms + step_ms
print(f"Total simulation duration: {duration} ms ({n_trajectories} trajectories)")

# --------------------------------------------------------------------------------------
# Set up NEST simulation
# --------------------------------------------------------------------------------------

nest.ResetKernel()
nest.SetKernelStatus({"resolution": step_ms, "rng_seed": 1234})


# ----------------------------------------------------------------------------------------
# Create neuron populations and input generators
# ----------------------------------------------------------------------------------------

# Load and install the custom rb_neuron module
nestml_install_dir = Path(__file__).resolve().parent / "nestml_neurons" / "nestml_install"
module_path = nestml_install_dir / "motor_neuron_module.so"

# Check if module exists and try to install it
if not module_path.exists():
    print("Compiled module not found. Compiling NESTML neurons...")
    from motor_controller_model.nestml_neurons.compile_nestml_neurons import compile_nestml_neurons
    compile_nestml_neurons()
    # Re-setup the kernel since compilation resets it
    nest.ResetKernel()
    nest.SetKernelStatus({"resolution": step_ms, "rng_seed": 1234})

# Install using the full path to the module
try:
    nest.Install(str(module_path))
    print("motor_neuron_module installed successfully.")
except Exception as e:
    print(f"Warning: Could not install motor_neuron_module: {e}")
    print("Attempting to use installed module...")
    nest.Install("motor_neuron_module")

# Create neuron populations
n_rb = num_centers
nrns_rb = nest.Create("rb_neuron", n_rb)

# Set up rb_neuron parameters
params_rb_neuron = config["neurons"]["rb"].copy()
params_rb_neuron["simulation_steps"] = int(duration / step_ms + 1)
nest.SetStatus(nrns_rb, params_rb_neuron)

# Set the desired rates for each rb_neuron (spike-input mode)
shift_min_rate = config["rbf"]["shift_min_rate"]
desired_upper_hz = float(config["rbf"].get("desired_upper_hz"))
desired_rates = np.linspace(shift_min_rate, desired_upper_hz, n_rb)
for i, nrn in enumerate(nrns_rb):
    nest.SetStatus(nrn, {"desired": desired_rates[i]})

# Create recurrent and output neuron populations
params_nrn_rec = config["neurons"]["rec"]
params_nrn_out = config["neurons"]["out"]

nrns_rec = nest.Create("eprop_iaf_bsshslm_2020", n_rec, params_nrn_rec)
nrns_out = nest.Create("eprop_readout_bsshslm_2020", n_out, params_nrn_out)
spike_recorder = nest.Create("spike_recorder")
spike_recorder_rb = nest.Create("spike_recorder")  # Recorder for rb_neurons
spike_recorder_input = nest.Create("spike_recorder")  # Recorder for input parrot neurons

# Create parrot neurons to relay input spikes (same count as in spike data)
nrns_parrot_input = nest.Create("parrot_neuron", n_input_neurons)

# Create spike generators for input spikes
gens_input_spikes = nest.Create("spike_generator", n_input_neurons)

# Set spike times for each input neuron (matching training approach)
# Sort spike times for each neuron to ensure non-descending order
for idx in range(len(gens_input_spikes)):
    sorted_spike_times = sorted(spike_times_per_neuron[idx])
    nest.SetStatus(gens_input_spikes[idx:idx+1], {"spike_times": sorted_spike_times})

# --------------------------------------------------------------------------------------
# Connect input to recurrent neurons via rb_neuron (using parrot neurons)
# --------------------------------------------------------------------------------------

params_conn_all_to_all = {"rule": "all_to_all", "allow_autapses": False}
params_conn_one_to_one = {"rule": "one_to_one", "allow_autapses": False}
params_syn_static = {"synapse_model": "static_synapse", "weight": 1.0, "delay": step_ms}

# Spike generators to parrot neurons (one-to-one)
nest.Connect(gens_input_spikes, nrns_parrot_input, params_conn_one_to_one, params_syn_static)

# Parrot neurons to rb_neurons (all-to-all connection)
nest.Connect(nrns_parrot_input, nrns_rb, params_conn_all_to_all, params_syn_static)

# Get source and target neuron IDs for rb_rec connections
nrns_rb_ids = rb_rec_weights["source"] + min(nrns_rb.tolist())
nrns_rec_ids = rb_rec_weights["target"] + min(nrns_rec.tolist())

nest.Connect(
    nrns_rb_ids,
    nrns_rec_ids,
    params_conn_one_to_one,
    {
        "synapse_model": "static_synapse",
        "weight": rb_rec_weights["weight"],
        "delay": [step_ms] * len(rb_rec_weights["weight"]),
    },
)

# --------------------------------------------------------------------------------------
# Connect recurrent and output neurons using loaded weights
# --------------------------------------------------------------------------------------

# Connect rec to rec with trained weights
rec_source_ids = rec_rec_weights["source"] + min(nrns_rec.tolist())
rec_target_ids = rec_rec_weights["target"] + min(nrns_rec.tolist())
nest.Connect(
    rec_source_ids,
    rec_target_ids,
    params_conn_one_to_one,
    syn_spec={
        "synapse_model": "static_synapse",
        "weight": rec_rec_weights["weight"],
        "delay": [step_ms] * len(rec_rec_weights["weight"]),
    },
)

# Connect rec to out with trained weights
nrns_rec_ids = rec_out_weights["source"] + min(nrns_rec.tolist())
nrns_out_ids = rec_out_weights["target"] + min(nrns_out.tolist())
nest.Connect(
    nrns_rec_ids,
    nrns_out_ids,
    params_conn_one_to_one,
    syn_spec={
        "synapse_model": "static_synapse",
        "weight": rec_out_weights["weight"],
        "delay": [step_ms] * len(rec_out_weights["weight"]),
    },
)

# --------------------------------------------------------------------------------------
# Record spikes and run simulation
# --------------------------------------------------------------------------------------

nest.Connect(nrns_rec, spike_recorder)
nest.Connect(nrns_rb, spike_recorder_rb)
nest.Connect(nrns_parrot_input, spike_recorder_input)

print(f"Simulating for {duration} ms...")
nest.Simulate(duration)
print("Simulation complete.")

# --------------------------------------------------------------------------------------
# Extract weights from the network for comparison
# --------------------------------------------------------------------------------------


def get_weights(pop_pre, pop_post):
    conns = nest.GetConnections(pop_pre, pop_post).get(["source", "target", "weight"])
    if not conns["source"]:
        return np.zeros((len(pop_post), len(pop_pre)))
    senders = np.array(conns["source"]) - np.min(conns["source"])
    targets = np.array(conns["target"]) - np.min(conns["target"])
    weight_matrix = np.zeros((len(pop_post), len(pop_pre)))
    weight_matrix[targets, senders] = conns["weight"]
    return weight_matrix


rec_rec_weights_extracted = get_weights(nrns_rec, nrns_rec) # recurrent to recurrent
rec_out_weights_extracted = get_weights(nrns_rec, nrns_out) # recurrent to output
rb_rec_weights_extracted = get_weights(nrns_rb, nrns_rec)  # input to recurrent

# --------------------------------------------------------------------------------------
# Plot results: spike raster and loaded weights vs extracted weights
# --------------------------------------------------------------------------------------

# Create output directory if it doesn't exist
output_dir = Path(__file__).resolve().parent.parent / "sim_results"
output_dir.mkdir(parents=True, exist_ok=True)

events = spike_recorder.get("events")
events_rb = spike_recorder_rb.get("events")
events_input = spike_recorder_input.get("events")

# Calculate PSTH for excitatory recurrent neurons only
exc_ratio = config["neurons"]["exc_ratio"]
n_rec_exc = int(n_rec * exc_ratio)
exc_neuron_ids = nrns_rec[:n_rec_exc].tolist()

# Filter spikes to only excitatory neurons
exc_spike_mask = np.isin(events["senders"], exc_neuron_ids)
exc_spike_times = events["times"][exc_spike_mask]

bin_size_ms = 10.0  # 10 ms bins
bins = np.arange(0, duration + bin_size_ms, bin_size_ms)
spike_counts, _ = np.histogram(exc_spike_times, bins=bins)
firing_rate = spike_counts / (n_rec_exc * bin_size_ms / 1000.0)  # Convert to Hz
bin_centers = (bins[:-1] + bins[1:]) / 2

fig_raster, axs_raster = plt.subplots(
    4, 1, figsize=(10, 12), gridspec_kw={"height_ratios": [1, 1, 4, 2]}, sharex=True
)
axs_raster[0].scatter(events_input["times"], events_input["senders"], s=2, color="#2ca02c", alpha=0.7)
axs_raster[0].set_ylabel("Input Neuron ID", fontsize=10)
axs_raster[0].set_title("Raster Plot of Input (Parrot) Neurons", fontsize=11)
axs_raster[0].grid(True, linestyle='--', alpha=0.3)

axs_raster[1].scatter(events_rb["times"], events_rb["senders"], s=2, color="#1f77b4", alpha=0.7)
axs_raster[1].set_ylabel("rb_neuron ID", fontsize=10)
axs_raster[1].set_title("Raster Plot of RBF Neurons", fontsize=11)
axs_raster[1].grid(True, linestyle='--', alpha=0.3)

axs_raster[2].scatter(events["times"], events["senders"], s=2, color='black', alpha=0.7)
axs_raster[2].set_ylabel("Recurrent Neuron ID", fontsize=10)
axs_raster[2].set_title("Raster Plot of Recurrent Neurons", fontsize=11)
axs_raster[2].grid(True, linestyle='--', alpha=0.3)

axs_raster[3].plot(bin_centers, firing_rate, color="black", linewidth=1.5, alpha=1.0)
axs_raster[3].fill_between(bin_centers, 0, firing_rate, color="black", alpha=0.2)
axs_raster[3].set_xlabel("Time (ms)", fontsize=11)
axs_raster[3].set_ylabel("Firing Rate (Hz)", fontsize=10)
axs_raster[3].set_title("Population Firing Rate (PSTH) - Excitatory Neurons", fontsize=11)
axs_raster[3].grid(True, linestyle='--', alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / "trained_weights_raster_plot.png", dpi=300)
print("Saved raster plot to sim_results/trained_weights_raster_plot.png")

# Plot loaded weights and extracted weights for verification
colors = {"blue": "#1f77b4", "red": "#d62728", "white": "#ffffff"}
cmap = mpl.colors.LinearSegmentedColormap.from_list(
    "cmap", ((0.0, colors["red"]), (0.5, colors["white"]), (1.0, colors["blue"]))
)

# Convert loaded weights from dict to matrix for plotting
def dict_to_matrix(weights_dict, n_pre, n_post):
    mat = np.zeros((n_post, n_pre))
    src = np.array(weights_dict["source"])
    tgt = np.array(weights_dict["target"])
    w = np.array(weights_dict["weight"])
    src -= src.min()
    tgt -= tgt.min()
    mat[tgt, src] = w
    return mat

rec_rec_weights_mat = dict_to_matrix(rec_rec_weights, n_rec, n_rec)
rec_out_weights_mat = dict_to_matrix(rec_out_weights, n_rec, n_out)
rb_rec_weights_mat = dict_to_matrix(rb_rec_weights, n_rb, n_rec)

vmin = min(
    np.min(rec_rec_weights_mat),
    np.min(rec_out_weights_mat),
    np.min(rec_rec_weights_extracted),
    np.min(rec_out_weights_extracted),
)
vmax = max(
    np.max(rec_rec_weights_mat),
    np.max(rec_out_weights_mat),
    np.max(rec_rec_weights_extracted),
    np.max(rec_out_weights_extracted),
)
norm = mpl.colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

fig, axs = plt.subplots(2, 2, figsize=(14, 10))
pc0 = axs[0, 0].pcolormesh(rec_rec_weights_mat, cmap=cmap, norm=norm)
axs[0, 0].set_title("Loaded rec_rec weights")
axs[0, 0].set_xlabel("Presynaptic neuron")
axs[0, 0].set_ylabel("Postsynaptic neuron")
plt.colorbar(pc0, ax=axs[0, 0])
pc1 = axs[0, 1].pcolormesh(rec_out_weights_mat, cmap=cmap, norm=norm)
axs[0, 1].set_title("Loaded rec_out weights")
axs[0, 1].set_xlabel("Presynaptic neuron")
axs[0, 1].set_ylabel("Postsynaptic neuron")
plt.colorbar(pc1, ax=axs[0, 1])
pc2 = axs[1, 0].pcolormesh(rec_rec_weights_extracted, cmap=cmap, norm=norm)
axs[1, 0].set_title("Extracted rec_rec weights")
axs[1, 0].set_xlabel("Presynaptic neuron")
axs[1, 0].set_ylabel("Postsynaptic neuron")
plt.colorbar(pc2, ax=axs[1, 0])
pc3 = axs[1, 1].pcolormesh(rec_out_weights_extracted, cmap=cmap, norm=norm)
axs[1, 1].set_title("Extracted rec_out weights")
axs[1, 1].set_xlabel("Presynaptic neuron")
axs[1, 1].set_ylabel("Postsynaptic neuron")
plt.colorbar(pc3, ax=axs[1, 1])
plt.tight_layout()
plt.savefig(output_dir / "trained_weights_comparison.png")
print("Saved weight comparison plot to sim_results/trained_weights_comparison.png")

# --- Plot input-to-recurrent weights (rb_neuron to rec) ---

fig_rbrec, axs_rbrec = plt.subplots(1, 2, figsize=(12, 5))
vmin_rb = -np.max(rb_rec_weights_extracted)
vmax_rb = np.max(rb_rec_weights_extracted)
norm_rb = mpl.colors.TwoSlopeNorm(vmin=vmin_rb, vcenter=0, vmax=vmax_rb)
if rb_rec_weights is not None:
    pc_rb_loaded = axs_rbrec[0].pcolormesh(rb_rec_weights_mat, cmap=cmap, norm=norm_rb)
    axs_rbrec[0].set_title("Loaded rb_rec weights (input to recurrent)")
    axs_rbrec[0].set_xlabel("Input neuron (rb_neuron)")
    axs_rbrec[0].set_ylabel("Recurrent neuron")
    plt.colorbar(pc_rb_loaded, ax=axs_rbrec[0])
else:
    axs_rbrec[0].set_title("Loaded rb_rec weights not present")
    axs_rbrec[0].axis("off")

pc_rb_extracted = axs_rbrec[1].pcolormesh(
    rb_rec_weights_extracted, cmap=cmap, norm=norm_rb
)
axs_rbrec[1].set_title("Extracted rb_rec weights (input to recurrent)")
axs_rbrec[1].set_xlabel("Input neuron (rb_neuron)")
axs_rbrec[1].set_ylabel("Recurrent neuron")
plt.colorbar(pc_rb_extracted, ax=axs_rbrec[1])
plt.tight_layout()
plt.savefig(output_dir / "trained_weights_rb_rec_comparison.png")
print("Saved input-to-recurrent weight comparison plot to sim_results/trained_weights_rb_rec_comparison.png")
print("All plots complete.")
