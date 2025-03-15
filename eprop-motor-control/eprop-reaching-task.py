# -*- coding: utf-8 -*-
#
# eprop_reaching_task.py
#
# This file is part of NEST.
#
# Copyright (C) 2004 The NEST Initiative
#
# NEST is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#
# NEST is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with NEST.  If not, see <http://www.gnu.org/licenses/>.

r"""
Tutorial on learning to perform a reaching task with e-prop
-----------------------------------------------------------

Training a regression model using supervised e-prop plasticity to perform a reaching task

Description
~~~~~~~~~~~

This script demonstrates supervised learning of a regression task with a recurrent spiking neural network that
is equipped with the eligibility propagation (e-prop) plasticity mechanism by Bellec et al. [1]_.

In this task, the network learns to generate a reaching trajectory. The network learns to reproduce with its
overall spiking activity a one-dimensional, one-second-long target signal which represents the reaching trajectory.

Learning in the neural network model is achieved by optimizing the connection weights with e-prop plasticity.
This plasticity rule requires a specific network architecture depicted in Figure 1. The neural network model
consists of a recurrent network that receives frozen noise input from Poisson generators and projects onto one
readout neuron. The readout neuron compares the network signal :math:`y` with the teacher target signal
:math:`y*`, which it receives from a rate generator. In scenarios with multiple readout neurons, each individual
readout signal denoted as :math:`y_k` is compared with a corresponding target signal represented as
:math:`y_k^*`. The network's training error is assessed by employing a mean-squared error loss.

Details on the event-based NEST implementation of e-prop can be found in [2]_.

References
~~~~~~~~~~

.. [1] Bellec G, Scherr F, Subramoney F, Hajek E, Salaj D, Legenstein R, Maass W (2020). A solution to the
       learning dilemma for recurrent networks of spiking neurons. Nature Communications, 11:3625.
       https://doi.org/10.1038/s41467-020-17236-y

.. [2] Korcsak-Gorzo A, Stapmanns J, Espinoza Valverde JA, Dahmen D, van Albada SJ, Bolten M, Diesmann M.
       Event-based implementation of eligibility propagation (in preparation)
"""

# %% ###########################################################################################################
# Import libraries
# ~~~~~~~~~~~~~~~~
# We begin by importing all libraries required for the simulation, analysis, and visualization.

import matplotlib as mpl
import matplotlib.pyplot as plt
import nest
import copy
import numpy as np
from cycler import cycler
from IPython.display import Image

# Import the function to load the dataset
import sys
from pathlib import Path

# Add the parent directory to the system path
sys.path.append(str(Path(__file__).resolve().parent.parent / "dataset_motor_training"))
from load_dataset import load_data_file


# %% ###########################################################################################################
# Setup
# ~~~~~

# %% ###########################################################################################################
# Initialize random generator
# ...........................
# We seed the numpy random generator, which will generate random initial weights as well as random input and
# output.

rng_seed = 1  # numpy random seed
np.random.seed(rng_seed)  # fix numpy random seed

# %% ###########################################################################################################
# Define timing of task
# .....................
# The task's temporal structure is then defined, once as time steps and once as durations in milliseconds.

n_batch = 1  # batch size, 1 in reference [2]
n_iter = 50  # number of iterations, 2000 in reference [2]

steps = {
    "sequence": 650,  # time steps of one full sequence (650 ms with steps of 1.0 ms)
}

steps["learning_window"] = steps["sequence"]  # time steps of window with non-zero learning signals
steps["task"] = n_iter * n_batch * steps["sequence"]  # time steps of task

steps.update(
    {
        "offset_gen": 1,  # offset since generator signals start from time step 1
        "delay_in_rec": 1,  # connection delay between input and recurrent neurons
        "delay_rec_out": 1,  # connection delay between recurrent and output neurons
        "delay_out_norm": 1,  # connection delay between output neurons for normalization
        "extension_sim": 1,  # extra time step to close right-open simulation time interval in Simulate()
    }
)

steps["delays"] = steps["delay_in_rec"] + steps["delay_rec_out"] + steps["delay_out_norm"]  # time steps of delays

steps["total_offset"] = steps["offset_gen"] + steps["delays"]  # time steps of total offset

steps["sim"] = steps["task"] + steps["total_offset"] + steps["extension_sim"]  # time steps of simulation

duration = {"step": 1.0}  # ms, temporal resolution of the simulation

duration.update({key: value * duration["step"] for key, value in steps.items()})  # ms, durations

# %% ###########################################################################################################
# Set up simulation
# .................
# As last step of the setup, we reset the NEST kernel to remove all existing NEST simulation settings and
# objects and set some NEST kernel parameters, some of which are e-prop-related.

params_setup = {
    "eprop_learning_window": duration["learning_window"],
    "eprop_reset_neurons_on_update": True,  # if True, reset dynamic variables at start of each update interval
    "eprop_update_interval": duration["sequence"],  # ms, time interval for updating the synaptic weights
    "print_time": True,  # if True, print time progress bar during simulation, set False if run as code cell
    "resolution": duration["step"],
    "total_num_virtual_procs": 1,  # number of virtual processes, set in case of distributed computing
}

####################

nest.ResetKernel()
nest.set(**params_setup)

# %% ###########################################################################################################
# Create neurons
# ~~~~~~~~~~~~~~
# We proceed by creating a certain number of recurrent and readout neurons and setting their parameters.
# Additionally, we already create an output target rate generator, which we will configure later.

n_in = 1  # number of input neurons
n_rec = 100  # number of recurrent neurons
n_out = 2  # Updated number of readout neurons

# divide the recurrent neurons into excitatory and inhibitory neurons
n_rec_exc = int(n_rec * 0.8)  # number of excitatory recurrent neurons
n_rec_inh = n_rec - n_rec_exc  # number of inhibitory recurrent neurons

params_nrn_rec = {
    "C_m": 250.0,  # pF, membrane capacitance - takes effect only if neurons get current input (here not the case)
    "c_reg": 300.0,  # firing rate regularization scaling
    "E_L": 0.0,  # mV, leak / resting membrane potential
    "f_target": 10.0,  # spikes/s, target firing rate for firing rate regularization
    "gamma": 0.3,  # scaling of the pseudo derivative
    "I_e": 0.0,  # pA, external current input
    "regular_spike_arrival": False,  # If True, input spikes arrive at end of time step, if False at beginning
    "surrogate_gradient_function": "piecewise_linear",  # surrogate gradient / pseudo-derivative function
    "t_ref": 2.0,  # ms, duration of refractory period
    "tau_m": 20.0,  # ms, membrane time constant
    "V_m": 0.0,  # mV, initial value of the membrane voltage
    "V_th": 20.0,  # mV, spike threshold membrane voltage
}

params_nrn_out = {
    "C_m": 250.0,
    "E_L": 0.0,
    "I_e": 0.0,
    "loss": "mean_squared_error",  # loss function
    "regular_spike_arrival": False,
    "tau_m": 20.0,
    "V_m": 0.0,
}

####################

# Create inhomogeneous Poisson generator for input
gen_poisson_in = nest.Create("inhomogeneous_poisson_generator", n_in)

# Create Poisson generator for background
gen_poisson_background = nest.Create("poisson_generator", n_in, {"rate": 100.0})

# The suffix _bsshslm_2020 follows the NEST convention to indicate in the model name the paper
# that introduced it by the first letter of the authors' last names and the publication year.

nrns_rec_exc = nest.Create("eprop_iaf_bsshslm_2020", n_rec_exc, params_nrn_rec)
nrns_rec_inh = nest.Create("eprop_iaf_bsshslm_2020", n_rec_inh, params_nrn_rec)
nrns_rec = nrns_rec_exc + nrns_rec_inh

nrns_out = nest.Create("eprop_readout_bsshslm_2020", n_out, params_nrn_out)
gen_rate_target = nest.Create("step_rate_generator", n_out)

# %% ###########################################################################################################
# Create recorders
# ~~~~~~~~~~~~~~~~
# We also create recorders, which, while not required for the training, will allow us to track various dynamic
# variables of the neurons, spikes, and changes in synaptic weights. To save computing time and memory, the
# recorders, the recorded variables, neurons, and synapses can be limited to the ones relevant to the
# experiment, and the recording interval can be increased (see the documentation on the specific recorders). By
# default, recordings are stored in memory but can also be written to file.

n_record = 1  # number of neurons to record dynamic variables from - this script requires n_record >= 1
n_record_w = 3  # number of senders and targets to record weights from - this script requires n_record_w >=1

if n_record == 0 or n_record_w == 0:
    raise ValueError("n_record and n_record_w >= 1 required")

params_mm_rec = {
    "interval": duration["step"],  # interval between two recorded time points
    "record_from": ["V_m", "surrogate_gradient", "learning_signal"],  # dynamic variables to record
    "start": duration["offset_gen"] + duration["delay_in_rec"],  # start time of recording
    "stop": duration["offset_gen"] + duration["delay_in_rec"] + duration["task"],  # stop time of recording
}

params_mm_out = {
    "interval": duration["step"],
    "record_from": ["V_m", "readout_signal", "readout_signal_unnorm", "target_signal", "error_signal"],
    "start": duration["total_offset"],
    "stop": duration["total_offset"] + duration["task"],
}

params_wr = {
    "senders": nrns_rec[:n_record_w],  # limit senders to subsample weights to record
    "targets": nrns_rec[:n_record_w] + nrns_out,  # limit targets to subsample weights to record from
    "start": duration["total_offset"],
    "stop": duration["total_offset"] + duration["task"],
}

params_sr = {
    "start": duration["total_offset"],
    "stop": duration["total_offset"] + duration["task"],
}

####################

mm_rec = nest.Create("multimeter", params_mm_rec)
mm_out = nest.Create("multimeter", params_mm_out)
sr = nest.Create("spike_recorder", params_sr)
wr = nest.Create("weight_recorder", params_wr)

nrns_rec_record = nrns_rec[:n_record]

# %% ###########################################################################################################
# Create connections
# ~~~~~~~~~~~~~~~~~~
# Now, we define the connectivity and set up the synaptic parameters, with the synaptic weights drawn from
# normal distributions. After these preparations, we establish the enumerated connections of the core network,
# as well as additional connections to the recorders.

params_conn_all_to_all = {"rule": "all_to_all", "allow_autapses": False}
params_conn_one_to_one = {"rule": "one_to_one"}
params_conn_bernoulli = {"rule": "pairwise_bernoulli", "p": 0.1, "allow_autapses": False}

w_default = 100.0  # default weight strength
w_rec = 100.0      # recurrent-to-recurrent weight strength
g = 4.0            # inhibitory-to-excitatory weight ratio

params_common_syn_eprop = {
    "optimizer": {
        "type": "gradient_descent",  # algorithm to optimize the weights
        "batch_size": n_batch,
        "eta": 1e-3,  # learning rate
        "Wmin": -1000.0,  # pA, minimal limit of the synaptic weights
        "Wmax": 1000.0,  # pA, maximal limit of the synaptic weights
    },
    "average_gradient": False,  # if True, average the gradient over the learning window
    "weight_recorder": wr,
}

params_syn_eprop_exc = {
    "optimizer": {
        "Wmin": 0.0,  # pA, minimal limit of the synaptic weights
        "Wmax": 1000.0,  # pA, maximal limit of the synaptic weights
    },
    "average_gradient": False,  # if True, average the gradient over the learning window
    "weight_recorder": wr,
}

# Note: for some reason, when setting Wmax to zero, an error occurs in the simulation. Therefore, we set it to 1.0.
# The error: NESTErrors.BadProperty: BadProperty in SLI function CopyModel_l_l_D: weight ≤ maximal weight Wmax required.
params_syn_eprop_inh = {
    "optimizer": {
        "Wmin": -1000.0,  # pA, minimal limit of the synaptic weights
        "Wmax": 1.0,  # pA, maximal limit of the synaptic weights
    },
    "average_gradient": False,  # if True, average the gradient over the learning window
    "weight_recorder": wr,
}

params_syn_base = {
    "synapse_model": "eprop_synapse_bsshslm_2020",
    "delay": duration["step"],  # ms, dendritic delay
    "tau_m_readout": params_nrn_out["tau_m"],  # ms, for technical reasons pass readout neuron membrane time constant
}

params_syn_input = {
    "synapse_model": "static_synapse",
    "delay": duration["step"],
    "weight": nest.math.redraw(nest.random.normal(
        mean=w_default, std=w_default*0.1
    ), min=0.0, max=1000.),
}

# Define the parameters for the recurrent connections
params_syn_rec_exc = copy.deepcopy(params_syn_base)
params_syn_rec_exc["weight"] = nest.math.redraw(nest.random.normal(
        mean=w_rec, std=w_rec*0.1
    ), min=0.0, max=1000.)
params_syn_rec_exc["synapse_model"] = "eprop_synapse_bsshslm_2020_exc"

params_syn_rec_inh = copy.deepcopy(params_syn_base)
params_syn_rec_inh["weight"] = nest.math.redraw(nest.random.normal(
        mean=-w_rec*g, std=g*w_rec*0.1
    ), min=-1000., max=0.0)
params_syn_rec_inh["synapse_model"] = "eprop_synapse_bsshslm_2020_inh"

params_syn_feedback = {
    "synapse_model": "eprop_learning_signal_connection_bsshslm_2020",
    "delay": duration["step"],
    "weight": nest.math.redraw(nest.random.normal(
        mean=w_rec, std=w_rec*0.1
    ), min=0.0, max=1000.),
}

params_syn_rate_target = {
    "synapse_model": "rate_connection_delayed",
    "delay": duration["step"],
    "receptor_type": 2,  # receptor type over which readout neurons receive target signals
}

params_syn_static = {
    "synapse_model": "static_synapse",
    "delay": duration["step"],
}

####################

nest.SetDefaults("eprop_synapse_bsshslm_2020", params_common_syn_eprop)
nest.CopyModel("eprop_synapse_bsshslm_2020", "eprop_synapse_bsshslm_2020_exc", params_syn_eprop_exc)
nest.CopyModel("eprop_synapse_bsshslm_2020", "eprop_synapse_bsshslm_2020_inh", params_syn_eprop_inh)

nest.Connect(gen_poisson_in, nrns_rec, params_conn_all_to_all, params_syn_input)            # Input to recurrent neurons
nest.Connect(gen_poisson_background, nrns_rec, params_conn_all_to_all, params_syn_input)    # Background to recurrent neurons

# Connect recurrent neurons to themselves
nest.Connect(nrns_rec_exc, nrns_rec, params_conn_bernoulli, params_syn_rec_exc)  # Excitory to all
nest.Connect(nrns_rec_inh, nrns_rec, params_conn_bernoulli, params_syn_rec_inh)  # Inhibitory to all

# Connect recurrent neurons to readout neurons
nest.Connect(nrns_rec_exc[:int(n_rec_exc/2)], nrns_out[0], params_conn_all_to_all, params_syn_rec_exc)  # Half of the excitatory neurons to readout 1
nest.Connect(nrns_rec_exc[int(n_rec_exc/2):], nrns_out[1], params_conn_all_to_all, params_syn_rec_exc)  # Half of the excitatory neurons to readout 2

# Connect readout neurons to recurrent neurons
nest.Connect(nrns_out[0], nrns_rec_exc[int(n_rec_exc/2):], params_conn_all_to_all, params_syn_feedback)  # Readout 1 to half of the excitatory neurons
nest.Connect(nrns_out[1], nrns_rec_exc[:int(n_rec_exc/2)], params_conn_all_to_all, params_syn_feedback)  # Readout 2 to half of the excitatory neurons
nest.Connect(nrns_out[0], nrns_rec_inh[int(n_rec_inh/2):], params_conn_all_to_all, params_syn_feedback)  # Readout 1 to inhibitory neurons
nest.Connect(nrns_out[1], nrns_rec_inh[:int(n_rec_inh/2)], params_conn_all_to_all, params_syn_feedback)  # Readout 2 to inhibitory neurons

# Connect readout neurons to target signal generator
nest.Connect(gen_rate_target[0], nrns_out[0], params_conn_one_to_one, params_syn_rate_target) # Readout 1 to target signal generator
nest.Connect(gen_rate_target[1], nrns_out[1], params_conn_one_to_one, params_syn_rate_target) # Readout 2 to target signal generator

# Connect recorders to neurons
nest.Connect(nrns_rec, sr, params_conn_all_to_all, params_syn_static)
nest.Connect(mm_rec, nrns_rec_record, params_conn_all_to_all, params_syn_static)
nest.Connect(mm_out, nrns_out, params_conn_all_to_all, params_syn_static)

# %% ###########################################################################################################
# Load training dataset
# ~~~~~~~~~~~~~~~~~~~~~
# We load the trajectory data from a file. The trajectory data is a one-dimensional signal that the network
# should learn to reproduce. We resample the trajectory data to match the new resolution.
# Load the training dataset and extract the trajectory number
dataset_path = Path(__file__).resolve().parent.parent / "dataset_motor_training" / "dataset_spikes.gdf"
training_dataset = load_data_file(str(dataset_path))
trajectory_num = int(training_dataset[0][0][0])
id_pos = training_dataset[0][1]
time_pos = training_dataset[0][2]
id_neg = training_dataset[0][3]
time_neg = training_dataset[0][4]

# Load the trajectory data used as input for the network
trajectory_file = dataset_path.parent / f"trajectory{trajectory_num}.txt"
trajectory_data = np.loadtxt(trajectory_file)

# Convert spiking times in time_pos and time_neg to rates 
# by dividing the number of spikes by the total time
# resolution: duration["step"] = 1.0 ms
# duration sequence: steps["sequence"] = 650 ms
desired_targets = {
    "pos": 1e4*np.histogram(time_pos, bins=int(duration["sequence"]), range=(0, duration["sequence"]))[0] / duration["sequence"],
    "neg": 1e4*np.histogram(time_neg, bins=int(duration["sequence"]), range=(0, duration["sequence"]))[0] / duration["sequence"],
}

# Smooth the target signals
desired_targets["pos"] = np.convolve(desired_targets["pos"], np.ones(20) / 10, mode="same")
desired_targets["neg"] = np.convolve(desired_targets["neg"], np.ones(20) / 10, mode="same")


# %% ###########################################################################################################
# Create input
# ~~~~~~~~~~~~

# Resample trajectory data to match the new resolution
trajectory_data = trajectory_data[::10] * 1e2

# Set the rates of the inhomogeneous Poisson generator based on the trajectory data
params_gen_poisson_in = {
    "rate_times": np.arange(0.0, duration["task"], duration["step"]) + duration["offset_gen"],
    "rate_values": np.tile(trajectory_data, n_iter * n_batch),
}

####################

nest.SetStatus(gen_poisson_in, params_gen_poisson_in)

# %% ###########################################################################################################
# Create output
# ~~~~~~~~~~~~~
# Use the trajectory data as the target signal for the reaching task.

params_gen_rate_target = []

params_gen_rate_target = [
    {
        "amplitude_times": np.arange(0.0, duration["task"], duration["step"]) + duration["total_offset"],
        "amplitude_values": np.tile(desired_targets[key], n_iter * n_batch),
    }
    for key in desired_targets.keys()
]

####################

nest.SetStatus(gen_rate_target, params_gen_rate_target)

# %% ###########################################################################################################
# Force final update
# ~~~~~~~~~~~~~~~~~~
# Synapses only get active, that is, the correct weight update calculated and applied, when they transmit a
# spike. To still be able to read out the correct weights at the end of the simulation, we force spiking of the
# presynaptic neuron and thus an update of all synapses, including those that have not transmitted a spike in
# the last update interval, by sending a strong spike to all neurons that form the presynaptic side of an eprop
# synapse. This step is required purely for technical reasons.

gen_spk_final_update = nest.Create("spike_generator", 1, {"spike_times": [duration["task"] + duration["delays"]]})

nest.Connect(gen_spk_final_update, nrns_rec, "all_to_all", {"weight": 1000.0})

# %% ###########################################################################################################
# Read out pre-training weights
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Before we begin training, we read out the initial weight matrices so that we can eventually compare them to
# the optimized weights.


def get_weights(pop_pre, pop_post):
    conns = nest.GetConnections(pop_pre, pop_post).get(["source", "target", "weight"])
    conns["senders"] = np.array(conns["source"]) - np.min(conns["source"])
    conns["targets"] = np.array(conns["target"]) - np.min(conns["target"])

    conns["weight_matrix"] = np.zeros((len(pop_post), len(pop_pre)))
    conns["weight_matrix"][conns["targets"], conns["senders"]] = conns["weight"]
    return conns


weights_pre_train = {
    "in_rec": get_weights(nrns_rec, nrns_rec),
    "rec_rec": get_weights(nrns_rec, nrns_rec),
    "rec_out": get_weights(nrns_rec, nrns_out),
}

# %% ###########################################################################################################
# Simulate
# ~~~~~~~~
# We train the network by simulating for a set simulation time, determined by the number of iterations and the
# batch size and the length of one sequence.

nest.Simulate(duration["sim"])

# %% ###########################################################################################################
# Read out post-training weights
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# After the training, we can read out the optimized final weights.

weights_post_train = {
    "in_rec": get_weights(nrns_rec, nrns_rec),
    "rec_rec": get_weights(nrns_rec, nrns_rec),
    "rec_out": get_weights(nrns_rec, nrns_out),
}

# %% ###########################################################################################################
# Read out recorders
# ~~~~~~~~~~~~~~~~~~
# We can also retrieve the recorded history of the dynamic variables and weights, as well as detected spikes.

events_mm_rec = mm_rec.get("events")
events_mm_out = mm_out.get("events")
events_sr = sr.get("events")
events_wr = wr.get("events")

# %% ###########################################################################################################
# Evaluate training error
# ~~~~~~~~~~~~~~~~~~~~~~~
# We evaluate the network's training error by calculating a loss - in this case, the mean squared error between
# the integrated recurrent network activity and the target rate.

readout_signal = events_mm_out["readout_signal"]
target_signal = events_mm_out["target_signal"]

error = (readout_signal - target_signal) ** 2
loss = 0.5 * np.add.reduceat(error, np.arange(0, steps["task"], steps["sequence"]))

# %% ###########################################################################################################
# Plot results
# ~~~~~~~~~~~~
# Then, we plot a series of plots.

do_plotting = True  # if True, plot the results

if not do_plotting:
    exit()

colors = {
    "blue": "#2854c5ff",
    "red": "#e04b40ff",
    "white": "#ffffffff",
}

plt.rcParams.update(
    {
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.prop_cycle": cycler(color=[colors["blue"], colors["red"]]),
    }
)

# %% ###########################################################################################################
# Plot training error
# ...................
# We begin with a plot visualizing the training error of the network: the loss plotted against the iterations.

fig, ax = plt.subplots()

ax.plot(range(1, n_iter + 1), loss)
ax.set_ylabel(r"$E = \frac{1}{2} \sum_{t,k} \left( y_k^t -y_k^{*,t}\right)^2$")
ax.set_xlabel("training iteration")
ax.set_xlim(1, n_iter)
ax.xaxis.get_major_locator().set_params(integer=True)

fig.tight_layout()
fig.savefig("training_error.png")  # Save the figure

# %% ###########################################################################################################
# Plot spikes and dynamic variables
# .................................
# This plotting routine shows how to plot all of the recorded dynamic variables and spikes across time. We take
# one snapshot in the first iteration and one snapshot at the end.


def plot_recordable(ax, events, recordable, ylabel, xlims):
    for sender in set(events["senders"]):
        idc_sender = events["senders"] == sender
        idc_times = (events["times"][idc_sender] > xlims[0]) & (events["times"][idc_sender] < xlims[1])
        ax.plot(events["times"][idc_sender][idc_times], events[recordable][idc_sender][idc_times], lw=0.5)
    ax.set_ylabel(ylabel)
    margin = np.abs(np.max(events[recordable]) - np.min(events[recordable])) * 0.1
    ax.set_ylim(np.min(events[recordable]) - margin, np.max(events[recordable]) + margin)


def plot_spikes(ax, events, nrns, ylabel, xlims):
    idc_times = (events["times"] > xlims[0]) & (events["times"] < xlims[1])
    idc_sender = np.isin(events["senders"][idc_times], nrns.tolist())
    senders_subset = events["senders"][idc_times][idc_sender]
    times_subset = events["times"][idc_times][idc_sender]

    ax.scatter(times_subset, senders_subset, s=0.1)
    ax.set_ylabel(ylabel)
    if senders_subset.size > 0:
        margin = np.abs(np.max(senders_subset) - np.min(senders_subset)) * 0.1
        ax.set_ylim(np.min(senders_subset) - margin, np.max(senders_subset) + margin)
    else:
        ax.set_ylim(0, 1)


for xlims in [(0, steps["sequence"]), (steps["task"] - steps["sequence"], steps["task"])]:
    fig, axs = plt.subplots(8, 1, sharex=True, figsize=(4, 12))

    plot_spikes(axs[0], events_sr, nrns_rec, r"$z_j$" + "\n", xlims)

    plot_recordable(axs[1], events_mm_rec, "V_m", r"$v_j$" + "\n(mV)", xlims)
    plot_recordable(axs[2], events_mm_rec, "surrogate_gradient", r"$\psi_j$" + "\n", xlims)
    plot_recordable(axs[3], events_mm_rec, "learning_signal", r"$L_j$" + "\n(pA)", xlims)

    plot_recordable(axs[4], events_mm_out, "V_m", r"$v_k$" + "\n(mV)", xlims)
    plot_recordable(axs[5], events_mm_out, "target_signal", r"$y^*_k$" + "\n", xlims)
    plot_recordable(axs[6], events_mm_out, "readout_signal", r"$y_k$" + "\n", xlims)
    plot_recordable(axs[7], events_mm_out, "error_signal", r"$y_k-y^*_k$" + "\n", xlims)

    axs[-1].set_xlabel(r"$t$ (ms)")
    axs[-1].set_xlim(*xlims)

    for ax in axs:
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.set_ylabel(ax.get_ylabel(), fontsize=14)
        for line in ax.get_lines():
            line.set_linewidth(1.5)  # Increase the linewidth

    fig.align_ylabels()
    fig.tight_layout()
    fig.savefig(f"spikes_and_dynamic_variables_{xlims[0]}_{xlims[1]}.png", dpi=300)  # Save the figure

# %% ###########################################################################################################
# Plot weight time courses
# ........................
# Similarly, we can plot the weight histories. Note that the weight recorder, attached to the synapses, works
# differently than the other recorders. Since synapses only get activated when they transmit a spike, the weight
# recorder only records the weight in those moments. That is why the first weight registrations do not start in
# the first time step and we add the initial weights manually.


def plot_weight_time_course(ax, events, nrns_senders, nrns_targets, label, ylabel):
    for sender in nrns_senders.tolist():
        for target in nrns_targets.tolist():
            idc_syn = (events["senders"] == sender) & (events["targets"] == target)
            idc_syn_pre = (weights_pre_train[label]["source"] == sender) & (
                weights_pre_train[label]["target"] == target
            )

            times = [0.0] + events["times"][idc_syn].tolist()
            weights = [weights_pre_train[label]["weight"][idc_syn_pre]] + events["weights"][idc_syn].tolist()

            ax.step(times, weights, c=colors["blue"])
        ax.set_ylabel(ylabel)

fig, axs = plt.subplots(2, 1, sharex=True, figsize=(3, 4))

plot_weight_time_course(
    axs[0], events_wr, nrns_rec[:n_record_w], nrns_rec[:n_record_w], "rec_rec", r"$W_\text{rec}$ (pA)"
)
plot_weight_time_course(axs[1], events_wr, nrns_rec[:n_record_w], nrns_out, "rec_out", r"$W_\text{out}$ (pA)")

axs[-1].set_xlabel(r"$t$ (ms)")
axs[-1].set_xlim(0, steps["task"])

fig.align_ylabels()
fig.tight_layout()
fig.savefig("weight_time_courses.png")  # Save the figure

# %% ###########################################################################################################
# Plot weight matrices
# ....................
# If one is not interested in the time course of the weights, it is possible to read out only the initial and
# final weights, which requires less computing time and memory than the weight recorder approach. Here, we plot
# the corresponding weight matrices before and after the optimization.

cmap = mpl.colors.LinearSegmentedColormap.from_list(
    "cmap", ((0.0, colors["blue"]), (0.5, colors["white"]), (1.0, colors["red"]))
)

fig, axs = plt.subplots(2, 2, sharex="col", sharey="row")

all_w_extrema = []

for k in weights_pre_train.keys():
    w_pre = weights_pre_train[k]["weight"]
    w_post = weights_post_train[k]["weight"]
    all_w_extrema.append([np.min(w_pre), np.max(w_pre), np.min(w_post), np.max(w_post)])

# Center the color map at zero
vmin = np.min(all_w_extrema)
vmax = np.max(all_w_extrema)
norm = mpl.colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

args = {"cmap": cmap, "norm": norm}

for i, weights in zip([0, 1], [weights_pre_train, weights_post_train]):
    axs[0, i].pcolormesh(weights["rec_rec"]["weight_matrix"], **args)
    cmesh = axs[1, i].pcolormesh(weights["rec_out"]["weight_matrix"], **args)

    axs[1, i].set_xlabel("recurrent\nneurons")

axs[0, 0].set_ylabel("recurrent\nneurons")
axs[1, 0].set_ylabel("readout\nneurons")
fig.align_ylabels(axs[:, 0])

axs[0, 0].text(0.5, 1.1, "pre-training", transform=axs[0, 0].transAxes, ha="center")
axs[0, 1].text(0.5, 1.1, "post-training", transform=axs[0, 1].transAxes, ha="center")

axs[1, 0].yaxis.get_major_locator().set_params(integer=True)

cbar = plt.colorbar(cmesh, cax=axs[1, 1].inset_axes([1.1, 0.2, 0.05, 0.8]), label="weight (pA)")

fig.tight_layout()
fig.savefig("weight_matrices.png")  # Save the figure

# plt.show()