
"""
trained_weights_net.py
----------------------

This module loads trained weights from a previously trained e-prop motor control network
and runs a test simulation using NEST. It reproduces the network setup and input encoding
used in the main training module (eprop_reaching_task.py) for consistency.

Key steps:
- Loads configuration parameters from config.yaml.
- Loads trained weights from .npz file.
- Loads and resamples trajectory data as input.
- Sets up NEST simulation with neuron populations and connections.
- Applies loaded weights to recurrent and output connections.
- Runs the simulation and plots spike raster and loaded weight matrices.

Input connections:
- Each rb_neuron is connected to a group of excitatory and inhibitory recurrent neurons,
  matching the grouping logic in eprop_reaching_task.py.

This module is intended for post-training evaluation and visualization.

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
import yaml

mpl.use("Agg")

# Add dataset path for imports
from motor_controller_model.dataset_motor_training.load_dataset import load_data_file

# --------------------------------------------------------------------------------------
# Load configuration parameters from config.yaml
# --------------------------------------------------------------------------------------

config_path = Path(__file__).resolve().parent / "config" / "config.yaml"
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

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
    Path(__file__).resolve().parent
    / "../sim_results"
    / "default_plastic_False_manualRBF_False"
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
# Load and resample trajectory data
# --------------------------------------------------------------------------------------

dataset_path = (
    Path(__file__).resolve().parent.parent
    / "dataset_motor_training"
    / "dataset_spikes.gdf"
)
training_dataset = load_data_file(str(dataset_path))
sample_ids = [
    tid * config["task"]["samples_per_trajectory_in_dataset"] + j
    for tid in trajectory_ids_to_use
    for j in range(n_samples_per_trajectory_to_use)
]
assert len(sample_ids) == n_samples

# Resample each trajectory to match simulation time steps
trajectories = []
for idx, sample_id in enumerate(sample_ids):
    traj_num = int(training_dataset[sample_id][0][0])
    traj_file = dataset_path.parent / f"trajectory{traj_num}.txt"
    traj_data = np.loadtxt(traj_file)
    orig_num_pts, orig_dur = len(traj_data), len(traj_data) * 0.1  # 0.1ms resolution
    resampled_time = np.linspace(0, orig_dur, int(sequence / step_ms), endpoint=False)
    orig_time = np.arange(orig_num_pts) * 0.1
    trajectory_signal = np.interp(resampled_time, orig_time, traj_data)
    # Prepend zeros for silent period
    if silent_period > 0:
        silent_steps = int(silent_period / step_ms)
        trajectory_signal = np.concatenate((np.zeros(silent_steps), trajectory_signal))
    trajectories.append(trajectory_signal)

# Concatenate and tile for all iterations
input_spk_rate = np.concatenate(trajectories) * scale_rate
input_spk_rate = np.tile(input_spk_rate, n_iter)
in_rate_times = np.arange(len(input_spk_rate)) * step_ms + step_ms

# --------------------------------------------------------------------------------------
# Set up NEST simulation
# --------------------------------------------------------------------------------------

nest.ResetKernel()
nest.SetKernelStatus({"resolution": step_ms, "rng_seed": 1234})


# ----------------------------------------------------------------------------------------
# Create neuron populations and input generator
# ----------------------------------------------------------------------------------------

# Create neuron populations and input generator
nest.Install("motor_neuron_module")
n_rb = num_centers
nrns_rb = nest.Create("rb_neuron", n_rb)
params_rb_neuron = config["neurons"]["rb"].copy()
params_rb_neuron["simulation_steps"] = int(duration / step_ms + 1)
params_rb_neuron["sdev"] = scale_rate * config["rbf"]["width"]
params_rb_neuron["max_peak_rate"] = scale_rate / step_ms
nest.SetStatus(nrns_rb, params_rb_neuron)

# Set the desired center for each rb_neuron's receptive field
angle_centers = np.linspace(0.0, np.pi, n_rb)
desired_rates = angle_centers * scale_rate
for i, nrn in enumerate(nrns_rb):
    nest.SetStatus(nrn, {"desired": desired_rates[i]})

# Create recurrent and output neuron populations
params_nrn_rec = config["neurons"]["rec"]
params_nrn_out = config["neurons"]["out"]

gen_poisson_in = nest.Create("inhomogeneous_poisson_generator")
nrns_rec = nest.Create("eprop_iaf_bsshslm_2020", n_rec, params_nrn_rec)
nrns_out = nest.Create("eprop_readout_bsshslm_2020", n_out, params_nrn_out)
spike_recorder = nest.Create("spike_recorder")
spike_recorder_rb = nest.Create("spike_recorder")  # Recorder for input neurons

# Set input rates for Poisson generator
nest.SetStatus(
    gen_poisson_in, {"rate_times": in_rate_times, "rate_values": input_spk_rate}
)

# --------------------------------------------------------------------------------------
# Connect input to recurrent neurons via rb_neuron
# --------------------------------------------------------------------------------------

params_conn_all_to_all = {"rule": "all_to_all", "allow_autapses": False}
params_conn_one_to_one = {"rule": "one_to_one", "allow_autapses": False}
params_syn_static = {"synapse_model": "static_synapse", "weight": 1.0, "delay": step_ms}

# Poisson generator to rb_neuron
nest.Connect(gen_poisson_in, nrns_rb, params_conn_all_to_all, params_syn_static)

# Get source and target neuron IDs 
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

nest.Simulate(duration)

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

events = spike_recorder.get("events")
events_rb = spike_recorder_rb.get("events")

fig_raster, axs_raster = plt.subplots(
    2, 1, figsize=(8, 8), gridspec_kw={"height_ratios": [1, 5]}, sharex=True
)
axs_raster[0].scatter(events_rb["times"], events_rb["senders"], s=2, color="#1f77b4")
axs_raster[0].set_ylabel("Input Neuron ID")
axs_raster[0].set_title("Raster Plot of Input (rb_neuron) Neurons")
axs_raster[1].scatter(events["times"], events["senders"], s=2)
axs_raster[1].set_xlabel("Time (ms)")
axs_raster[1].set_ylabel("Recurrent Neuron ID")
axs_raster[1].set_title("Raster Plot of Recurrent Neurons")
plt.tight_layout()
plt.savefig("../sim_results/trained_weights_raster_plot.png")

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
plt.savefig("../sim_results/trained_weights_comparison.png")

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
plt.savefig("../sim_results/trained_weights_rb_rec_comparison.png")
