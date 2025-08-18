r"""
Training a regression model using supervised e-prop plasticity to perform a reaching task.

Description
~~~~~~~~~~~

This script demonstrates supervised learning of a regression task with a recurrent spiking
neural network (SNN) that is equipped with the eligibility propagation (e-prop) plasticity
mechanism by Bellec et al. [1]_.

In this task, the network learns to generate a reaching trajectory. The network learns to
reproduce with its overall spiking activity a one-dimensional, one-second-long target signal
which represents the reaching trajectory.

Learning in the SNN is achieved by optimizing the connection weights with e-prop plasticity.
This plasticity rule requires a specific network architecture. The model consists of a
recurrent network that receives input encoding the trajectory and projects onto readout
neurons. The readout neurons compare the network's filtered output signal, $y$, with a
teacher target signal, $y^*$. The network's training error is assessed by employing a
mean-squared error loss.

Details on the event-based NEST implementation of e-prop can be found in [2]_.

This script includes two methods for encoding the input trajectory:
1.  **rb_neuron**: A custom NEST neuron model that performs Radial Basis Function (RBF)
    encoding internally. This is the default and recommended method.
2.  **Manual RBF**: A manual implementation where RBF activation is calculated in NumPy
    and fed to Poisson generators. This can be enabled with the `--use-manual-rbf` flag.

Author: Renan Oliveira Shimoura

References
~~~~~~~~~~

.. [1] Bellec G, Scherr F, Subramoney F, Hajek E, Salaj D, Legenstein R, Maass W (2020).
       A solution to the learning dilemma for recurrent networks of spiking neurons.
       Nature Communications, 11:3625. https://doi.org/10.1038/s41467-020-17236-y

.. [2] Korcsak-Gorzo A, Stapmanns J, Espinoza Valverde JA, Dahmen D, van Albada SJ,
       Bolten M, Diesmann M. Event-based implementation of eligibility propagation
       (in preparation)
"""

# %% ###########################################################################################################
# Import libraries
# ~~~~~~~~~~~~~~~~
# We begin by importing all libraries required for the simulation, analysis, and visualization.

import matplotlib

# matplotlib.use("Agg")

import nest
import copy
import numpy as np
import yaml
import sys
from pathlib import Path
import os
import itertools

# Add the parent directory to the system path for dataset import
sys.path.append(str(Path(__file__).resolve().parent.parent / "dataset_motor_training"))
from load_dataset import load_data_file
from plot_results import (
    plot_training_error,
    plot_spikes_and_dynamics,
    plot_weight_time_courses,
    plot_weight_matrices,
    plot_all_loss_curves,
)


def run_simulation(
    n_rec=None,
    n_out=None,
    learning_rate_exc=None,
    learning_rate_inh=None,
    exc_ratio=None,
    config_path_override=None,
    result_dir=None,
    plot_results=True,
    plastic_input_to_rec=False,
    use_manual_rbf=False,
    trajectory_files=None,
    target_files=None,
    **override_kwargs,
):
    """
    Run the e-prop reaching task simulation with optional parameter overrides.

    Args:
        n_rec (int, optional): Number of recurrent neurons.
        n_out (int, optional): Number of output neurons.
        learning_rate_exc (float, optional): Learning rate for excitatory synapses.
        learning_rate_inh (float, optional): Learning rate for inhibitory synapses.
        exc_ratio (float, optional): Ratio of excitatory neurons.
        config_path_override (str/Path, optional): Path to a custom config file.
        result_dir (str, optional): Directory for result files.
        plot_results (bool): Whether to generate and save plots.
        plastic_input_to_rec (bool): If True, make input-to-recurrent connections plastic.
        use_manual_rbf (bool): If True, use manual RBF implementation instead of rb_neuron.
        trajectory_files (list of str, optional): List of custom trajectory file paths to use.
        target_files (list of str, optional): List of custom target file paths to use.
        override_kwargs (dict): Other config overrides using dot notation (e.g., 'rbf.num_centers').
    """
    # %% ###########################################################################################################
    # Load configuration
    # ~~~~~~~~~~~~~~~~~~~~
    # Load the base configuration from the YAML file and override any parameters
    # specified through function arguments or keyword arguments.
    config_path = (
        Path(config_path_override)
        if config_path_override
        else Path(__file__).resolve().parent / "config" / "config.yaml"
    )
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Override config with function arguments
    if n_rec is not None:
        config["neurons"]["n_rec"] = n_rec
    if n_out is not None:
        config["neurons"]["n_out"] = n_out
    if exc_ratio is not None:
        config["neurons"]["exc_ratio"] = exc_ratio
    # Map user-friendly 'learning_rate' to NEST's 'eta'
    if learning_rate_exc is not None:
        config["synapses"]["exc"]["optimizer"]["eta"] = learning_rate_exc
    if learning_rate_inh is not None:
        config["synapses"]["inh"]["optimizer"]["eta"] = learning_rate_inh
    # Remove any accidental learning_rate keys to avoid confusion
    config["synapses"]["exc"]["optimizer"].pop("learning_rate", None)
    config["synapses"]["inh"]["optimizer"].pop("learning_rate", None)

    # Apply any other keyword argument overrides using dot notation
    for k, v in override_kwargs.items():
        keys = k.split(".")
        d = config
        for key in keys[:-1]:
            d = d[key]
        d[keys[-1]] = v

    # %% ###########################################################################################################
    # Setup Simulation
    # ~~~~~~~~~~~~~~~~
    # Extract parameters from the config and set up the simulation kernel and timing.

    # Extract simulation and task parameters
    sim_cfg = config["simulation"]
    task_cfg = config["task"]
    step_ms = sim_cfg["step"]
    silent_period = task_cfg["silent_period"]

    # Define the timing structure of the experiment
    duration = {
        "step": step_ms,
        "sequence": task_cfg["sequence"],  # Original active sequence length
        "silent_period": silent_period,  # Duration of silent period
        "total_sequence_with_silence": task_cfg["sequence"]
        + silent_period,  # Total sequence length
        "learning_window": task_cfg["sequence"]
        + silent_period,  # Learning window covers total sequence
        "extension_sim": step_ms,
    }

    # Number of timesteps for the total sequence including silence
    n_timesteps_per_sequence = int(
        round(duration["total_sequence_with_silence"] / step_ms)
    )

    trajectory_ids_to_use = task_cfg["trajectory_ids_to_use"]
    n_samples_per_trajectory_to_use = int(task_cfg["n_samples_per_trajectory_to_use"])
    n_iter = int(task_cfg["n_iter"])
    # Determine number of samples (trajectories) actually used.
    if trajectory_files is not None and len(trajectory_files) > 0:
        n_samples = len(trajectory_files)
        duration["n_trajectories"] = n_samples  # Each file considered unique trajectory
    else:
        n_samples = len(trajectory_ids_to_use) * n_samples_per_trajectory_to_use
        duration["n_trajectories"] = len(trajectory_ids_to_use)
    duration["task"] = n_timesteps_per_sequence * n_samples * n_iter * step_ms
    duration["sim"] = duration["task"] + duration["extension_sim"]

    # Set up NEST kernel
    params_setup = {
        "eprop_learning_window": duration["learning_window"],
        "eprop_reset_neurons_on_update": False,
        "eprop_update_interval": duration["total_sequence_with_silence"],
        "print_time": sim_cfg["print_time"],
        "resolution": duration["step"],
        "total_num_virtual_procs": sim_cfg["total_num_virtual_procs"],
        "rng_seed": sim_cfg["rng_seed"],
    }
    nest.ResetKernel()
    nest.set(**params_setup)

    # %% ###########################################################################################################
    # Create Neurons
    # ~~~~~~~~~~~~~~
    # We create input, recurrent, and readout neurons based on the configuration.
    # The input layer's implementation depends on the `use_manual_rbf` flag.
    nrn_cfg = config["neurons"]
    rbf_cfg = config["rbf"]
    num_centers = int(rbf_cfg["num_centers"])

    if use_manual_rbf:
        # Manual RBF: Input layer is a set of Poisson generators
        print("Using manual RBF implementation.")
        n_in = num_centers
        gen_poisson_in = nest.Create("inhomogeneous_poisson_generator", n_in)

        if plastic_input_to_rec:
            # Create parrot neurons to allow plastic connections from input to recurrent neurons
            parrot_neurons = nest.Create("parrot_neuron", n_in)
    else:
        # Default: Use the custom 'rb_neuron' model for RBF encoding
        print("Using 'rb_neuron' for RBF implementation.")
        nest.Install("motor_neuron_module")

        # Define parameters for the rb_neuron
        params_rb_neuron = nrn_cfg["rb"]
        params_rb_neuron["simulation_steps"] = int(
            duration["sim"] / duration["step"] + 1
        )
        params_rb_neuron["sdev"] = rbf_cfg["scale_rate"] * rbf_cfg["width"]
        params_rb_neuron["max_peak_rate"] = rbf_cfg["scale_rate"] / duration["step"]

        # Create the input layer as rb_neurons
        n_rb = num_centers
        nrns_rb = nest.Create("rb_neuron", n_rb)

        # Set the parameters for the rb_neuron
        nest.SetStatus(nrns_rb, params_rb_neuron)

        # Create Poisson generators for the input layer
        gen_poisson_in = nest.Create("inhomogeneous_poisson_generator")

    # Create recurrent and readout populations
    n_rec = int(nrn_cfg["n_rec"])
    n_out = int(nrn_cfg["n_out"])
    exc_ratio = nrn_cfg["exc_ratio"]
    n_rec_exc = int(n_rec * exc_ratio)
    n_rec_inh = n_rec - n_rec_exc
    params_nrn_rec = nrn_cfg["rec"]
    params_nrn_out = nrn_cfg["out"]

    nrns_rec_exc = nest.Create("eprop_iaf_bsshslm_2020", n_rec_exc, params_nrn_rec)
    nrns_rec_inh = nest.Create("eprop_iaf_bsshslm_2020", n_rec_inh, params_nrn_rec)
    nrns_rec = nrns_rec_exc + nrns_rec_inh

    nrns_out = nest.Create("eprop_readout_bsshslm_2020", n_out, params_nrn_out)
    gen_rate_target = nest.Create("step_rate_generator", n_out)

    # %% ###########################################################################################################
    # Create Recorders
    # ~~~~~~~~~~~~~~~~
    # We set up recorders to monitor membrane potentials, signals, spikes, and weights.
    rec_cfg = config["recording"]
    n_record, n_record_w = rec_cfg["n_record"], rec_cfg["n_record_w"]
    if n_record < 1 or n_record_w < 1:
        raise ValueError("'n_record' and 'n_record_w' must be at least 1.")

    start_time = duration["step"]
    stop_time = duration["task"]
    params_mm_rec = {
        **rec_cfg["mm_rec"],
        "interval": duration["step"],
        "start": start_time,
        "stop": stop_time,
    }
    params_mm_out = {
        **rec_cfg["mm_out"],
        "interval": duration["step"],
        "start": start_time,
        "stop": stop_time,
    }
    params_wr = {
        "senders": nrns_rec[:n_record_w],
        "targets": nrns_rec[:n_record_w] + nrns_out,
        "start": start_time,
        "stop": stop_time,
    }
    params_sr = {"start": start_time, "stop": stop_time}

    mm_rec = nest.Create("multimeter", params_mm_rec)
    mm_out = nest.Create("multimeter", params_mm_out)
    spike_recorder = nest.Create("spike_recorder", params_sr)
    weight_recorder = nest.Create("weight_recorder", params_wr)
    nrns_rec_record = nrns_rec[:n_record]

    # %% ###########################################################################################################
    # Create Connections
    # ~~~~~~~~~~~~~~~~~~
    # Define connectivity rules and synapse parameters, then build the network.
    syn_cfg = config["synapses"]
    gradient_batch_size = int(task_cfg["gradient_batch_size"])
    w_input, w_rec, g = syn_cfg["w_input"], syn_cfg["w_rec"], syn_cfg["g"]

    # Define connection rule parameters
    params_conn_all_to_all = {"rule": "all_to_all", "allow_autapses": False}
    params_conn_one_to_one = {"rule": "one_to_one"}
    params_conn_bernoulli = {
        "rule": "pairwise_bernoulli",
        "p": syn_cfg["conn_bernoulli_p"],
        "allow_autapses": False,
    }

    # Create custom e-prop synapse models for excitatory and inhibitory connections
    params_syn_eprop_exc = {
        "optimizer": {**syn_cfg["exc"]["optimizer"], "batch_size": gradient_batch_size},
        "average_gradient": syn_cfg["average_gradient"],
        "weight_recorder": weight_recorder,
    }
    params_syn_eprop_inh = {
        "optimizer": {**syn_cfg["inh"]["optimizer"], "batch_size": gradient_batch_size},
        "weight": syn_cfg["inh"]["weight"],
        "average_gradient": syn_cfg["average_gradient"],
        "weight_recorder": weight_recorder,
    }
    # Copy base models to create specific plastic synapse types
    nest.CopyModel(
        "eprop_synapse_bsshslm_2020",
        "eprop_synapse_bsshslm_2020_exc",
        params_syn_eprop_exc,
    )
    nest.CopyModel(
        "eprop_synapse_bsshslm_2020",
        "eprop_synapse_bsshslm_2020_inh",
        params_syn_eprop_inh,
    )

    # Create synapse parameters for recurrent connections
    params_syn_rec_exc = {
        "synapse_model": "eprop_synapse_bsshslm_2020_exc",
        "delay": duration["step"],
        "tau_m_readout": params_nrn_out["tau_m"],
        "weight": nest.math.redraw(
            nest.random.normal(mean=w_rec, std=w_rec * 0.1), min=0.0, max=1000.0
        ),
    }
    params_syn_rec_inh = {
        "synapse_model": "eprop_synapse_bsshslm_2020_inh",
        "delay": duration["step"],
        "tau_m_readout": params_nrn_out["tau_m"],
        "weight": nest.math.redraw(
            nest.random.normal(mean=-w_rec * g, std=g * w_rec * 0.1),
            min=-1000.0,
            max=0.0,
        ),
    }

    # Create synapse parameters for feedback connections
    params_syn_feedback = {
        "synapse_model": "eprop_learning_signal_connection_bsshslm_2020",
        "delay": duration["step"],
        "weight": nest.math.redraw(
            nest.random.normal(mean=w_rec, std=w_rec * 0.1), min=0.0, max=1000.0
        ),
    }
    # Create synapse parameters for rate-based target connections
    params_syn_rate_target = {
        "synapse_model": "rate_connection_delayed",
        "delay": duration["step"],
        "receptor_type": syn_cfg["receptor_type"],
    }
    params_syn_static = {"synapse_model": "static_synapse", "delay": duration["step"]}

    # Connect input layer to recurrent layer
    if use_manual_rbf:
        params_syn_input = {
            "synapse_model": "static_synapse",
            "delay": duration["step"],
            "weight": nest.math.redraw(
                nest.random.normal(mean=w_input, std=w_input * 0.1), min=0.0, max=1000.0
            ),
        }
        if plastic_input_to_rec:
            params_syn_input["synapse_model"] = "eprop_synapse_bsshslm_2020_exc"
            # Connect Poisson generators to parrot neurons for plastic connections
            nest.Connect(
                gen_poisson_in,
                parrot_neurons,
                params_conn_one_to_one,
                params_syn_static,
            )
            # Connect parrot neurons to recurrent neurons
            nest.Connect(
                parrot_neurons, nrns_rec, params_conn_all_to_all, params_syn_input
            )
        else:
            for i, poisson_node in enumerate(gen_poisson_in):
                nest.Connect(
                    poisson_node,
                    nrns_rec_exc[
                        int(i * n_rec_exc / n_in) : int((i + 1) * n_rec_exc / n_in)
                    ],
                    params_conn_all_to_all,
                    params_syn_input,
                )
                nest.Connect(
                    poisson_node,
                    nrns_rec_inh[
                        int(i * n_rec_inh / n_in) : int((i + 1) * n_rec_inh / n_in)
                    ],
                    params_conn_all_to_all,
                    params_syn_input,
                )
    else:
        # rb_neuron connectivity
        params_syn_input_to_rb = {
            "synapse_model": "static_synapse",
            "delay": duration["step"],
            "weight": 1.0,
        }
        params_syn_rb_to_rec = {
            "synapse_model": "static_synapse",
            "delay": duration["step"],
            "weight": nest.math.redraw(
                nest.random.normal(mean=w_input, std=w_input * 0.1), min=0.0, max=1000.0
            ),
        }
        nest.Connect(
            gen_poisson_in, nrns_rb, params_conn_all_to_all, params_syn_input_to_rb
        )
        if plastic_input_to_rec:
            params_syn_rb_to_rec["synapse_model"] = "eprop_synapse_bsshslm_2020_exc"
            nest.Connect(
                nrns_rb, nrns_rec, params_conn_all_to_all, params_syn_rb_to_rec
            )
        else:
            for i, input_node in enumerate(nrns_rb):
                nest.Connect(
                    input_node,
                    nrns_rec_exc[
                        int(i * n_rec_exc / n_rb) : int((i + 1) * n_rec_exc / n_rb)
                    ],
                    params_conn_all_to_all,
                    params_syn_rb_to_rec,
                )
                nest.Connect(
                    input_node,
                    nrns_rec_inh[
                        int(i * n_rec_inh / n_rb) : int((i + 1) * n_rec_inh / n_rb)
                    ],
                    params_conn_all_to_all,
                    params_syn_rb_to_rec,
                )

    # Connect recurrent neurons to each other
    nest.Connect(nrns_rec_exc, nrns_rec, params_conn_bernoulli, params_syn_rec_exc)
    nest.Connect(nrns_rec_inh, nrns_rec, params_conn_bernoulli, params_syn_rec_inh)

    # Connect recurrent neurons to readout neurons
    nest.Connect(nrns_rec_exc, nrns_out, params_conn_all_to_all, params_syn_rec_exc)
    # nest.Connect(nrns_rec_inh, nrns_out, params_conn_all_to_all, params_syn_rec_inh)

    # Connect readout neurons to the target signal generator
    nest.Connect(nrns_out, nrns_rec_exc, params_conn_all_to_all, params_syn_feedback)
    # nest.Connect(nrns_out, nrns_rec_inh, params_conn_all_to_all, params_syn_feedback)

    # Connect the target signal generators to the readout neurons
    nest.Connect(
        gen_rate_target[0], nrns_out[0], params_conn_one_to_one, params_syn_rate_target
    )
    nest.Connect(
        gen_rate_target[1], nrns_out[1], params_conn_one_to_one, params_syn_rate_target
    )

    # Connect the recorder devices
    nest.Connect(nrns_rec, spike_recorder, params_conn_all_to_all, params_syn_static)
    nest.Connect(mm_rec, nrns_rec_record, params_conn_all_to_all, params_syn_static)
    nest.Connect(mm_out, nrns_out, params_conn_all_to_all, params_syn_static)

    # %% ###########################################################################################################
    # Load and Prepare Data
    # ~~~~~~~~~~~~~~~~~~~~~~~
    # Load the trajectory and target spike data, then resample and process it.
    if trajectory_files is not None and len(trajectory_files) > 0:
        sample_ids = list(range(n_samples))
        training_dataset = None
        dataset_path = None
    else:
        dataset_path = (
            Path(__file__).resolve().parent.parent
            / "dataset_motor_training"
            / "sample_data"
            / "dataset_spikes.gdf"
        )
        training_dataset = load_data_file(str(dataset_path))
        sample_ids = [
            tid * task_cfg["samples_per_trajectory_in_dataset"] + j
            for tid in trajectory_ids_to_use
            for j in range(n_samples_per_trajectory_to_use)
        ]
        assert len(sample_ids) == n_samples

    # Number of timesteps for the active sequence
    n_timesteps_per_stimulus = int(round(task_cfg["sequence"] / duration["step"]))
    trajectories, desired_targets_list = [], {"pos": [], "neg": []}
    for idx, sample_id in enumerate(sample_ids):
        # Use custom trajectory file if provided
        if trajectory_files is not None and len(trajectory_files) > 0:
            traj_file = Path(trajectory_files[idx % len(trajectory_files)])
            traj_data = np.loadtxt(traj_file)
        else:
            traj_num = int(training_dataset[sample_id][0][0])
            traj_file = dataset_path.parent / f"trajectory{traj_num}.txt"
            traj_data = np.loadtxt(traj_file)
        orig_num_pts, orig_dur = (
            len(traj_data),
            len(traj_data) * 0.1,
        )  # 0.1ms is original resolution
        resampled_time = np.linspace(
            0, orig_dur, n_timesteps_per_stimulus, endpoint=False
        )
        orig_time = np.arange(orig_num_pts) * 0.1
        # Resample the trajectory to match the simulation time steps
        trajectory_signal = np.interp(resampled_time, orig_time, traj_data)
        trajectories.append(trajectory_signal)

        # Use custom target file if provided
        for i, key in enumerate(["pos", "neg"]):
            if target_files is not None and idx < len(target_files):
                # Custom file: each row neuron_id,spike_time (comma or whitespace separated)
                arr = np.loadtxt(
                    target_files[idx],
                    delimiter=(
                        "," if "," in open(target_files[idx]).readline() else None
                    ),
                )
                if key == "pos":
                    spike_times = arr[arr[:, 0] <= 50][:, 1]
                else:
                    spike_times = arr[arr[:, 0] > 50][:, 1]
            else:
                spike_times = training_dataset[sample_id][2 * i + 2]
            target_hist = np.histogram(
                spike_times,
                bins=n_timesteps_per_stimulus,
                range=(0, duration["sequence"]),
            )[0]
            desired_targets_list[key].append(
                np.convolve(target_hist, np.ones(20) / 10, mode="same")
            )

    # Add silent period BEFORE EACH trajectory (target signals)
    if silent_period > 0:  # Only add if silent duration is positive
        silent_steps = int(silent_period / duration["step"])
        for k in desired_targets_list:
            for i in range(len(desired_targets_list[k])):
                # Prepend zeros to each target sequence so every trajectory starts with silence
                desired_targets_list[k][i] = np.concatenate(
                    (np.zeros(silent_steps), desired_targets_list[k][i])
                )

    # %% ###########################################################################################################
    # Create Input and Output Signals
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Convert the processed data into timed rate signals for the NEST generators.
    shift_min_rate = rbf_cfg["shift_min_rate"]

    if use_manual_rbf:

        def gaussian_rbf(x, center, width):
            return np.exp(-((x - center) ** 2) / (2 * width**2))

        # Standard deviation for Gaussian RBF
        width = rbf_cfg["width"]

        # Define the number of centers and create RBF inputs
        
        # Get the minimum angle from all trajectory samples
        min_angle = np.min([np.min(traj) for traj in trajectories])
        if min_angle > 0.5:
            min_angle = 0.5

        centers = np.linspace(min_angle, np.pi, num_centers)
        rbf_inputs_list = []
        silent_steps = int(silent_period / duration["step"]) if silent_period > 0 else 0
        zeros_block = (
            np.zeros((silent_steps, num_centers)) if silent_steps > 0 else None
        )
        for trajectory_sample in trajectories:
            rbf_inputs_for_sample = np.zeros((n_timesteps_per_stimulus, num_centers))
            for i, center in enumerate(centers):
                rbf_inputs_for_sample[:, i] = gaussian_rbf(
                    trajectory_sample, center, width
                )
            # Prepend silence for EACH trajectory (not only once at start)
            if silent_steps > 0:
                rbf_inputs_for_sample = np.vstack((zeros_block, rbf_inputs_for_sample))
            rbf_inputs_list.append(rbf_inputs_for_sample)

        # Stack the per-trajectory (silence + active) blocks and scale to firing rates.
        rate_based_rbf_inputs = (
            np.vstack(rbf_inputs_list) * rbf_cfg["scale_rate"] / duration["step"]
            + shift_min_rate
        )

        # Tile the signals for all training iterations.
        tiled_rbf_inputs = np.tile(rate_based_rbf_inputs, (n_iter, 1))

        # Set the rate for each Poisson generator.
        in_rate_times = (
            np.arange(tiled_rbf_inputs.shape[0]) * duration["step"] + duration["step"]
        )
        params_gen_poisson_in = [
            {"rate_times": in_rate_times, "rate_values": tiled_rbf_inputs[:, i]}
            for i in range(num_centers)
        ]
        nest.SetStatus(gen_poisson_in, params_gen_poisson_in)
    else:  # rb_neuron input creation
        # Insert silence BEFORE EACH trajectory so that update intervals align with sequences
        if silent_period > 0:
            silent_steps = int(silent_period / duration["step"])
            zeros_block = np.zeros(silent_steps)
            trajectories_with_silence = [
                np.concatenate((zeros_block, tr)) for tr in trajectories
            ]
        else:
            trajectories_with_silence = trajectories

        input_spk_rate = (
            np.concatenate(trajectories_with_silence) * rbf_cfg["scale_rate"]
            + shift_min_rate
        )

        input_spk_rate = np.tile(input_spk_rate, n_iter)
        in_rate_times = (
            np.arange(len(input_spk_rate)) * duration["step"] + duration["step"]
        )
        nest.SetStatus(
            gen_poisson_in, {"rate_times": in_rate_times, "rate_values": input_spk_rate}
        )
        # Set the desired center for each rb_neuron's receptive field
        # Get the minimum angle from all trajectory samples
        min_angle = np.min([np.min(traj) for traj in trajectories])
        if min_angle > 0.5:
            min_angle = 0.5
        angle_centers = np.linspace(min_angle, np.pi, num_centers)
        desired_rates = angle_centers * rbf_cfg["scale_rate"] + shift_min_rate
        print(f"Setting desired rates for rb_neurons: {desired_rates}")
        print(params_rb_neuron)
        for i, nrn in enumerate(nrns_rb):
            nest.SetStatus(nrn, {"desired": desired_rates[i]})

    # Create target signals for the readout neurons
    concatenated_targets = {
        k: np.concatenate(v) for k, v in desired_targets_list.items()
    }
    target_amp_times = (
        np.arange(len(concatenated_targets["pos"]) * n_iter) * duration["step"]
        + duration["step"]
    )
    params_gen_rate_target = [
        {
            "amplitude_times": target_amp_times,
            "amplitude_values": np.tile(concatenated_targets[k], n_iter),
        }
        for k in concatenated_targets
    ]
    nest.SetStatus(gen_rate_target, params_gen_rate_target)

    # %% ###########################################################################################################
    # Pre-Simulation Finalization
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Force a final spike after the task to ensure all weights are updated for readout.
    # Also, read out the initial weight matrices for later comparison.
    final_update_time = duration["task"] + duration["step"]
    gen_spk_final_update = nest.Create(
        "spike_generator", 1, {"spike_times": [final_update_time]}
    )
    nest.Connect(gen_spk_final_update, nrns_rec, "all_to_all", {"weight": 1000.0})

    def get_weights(pop_pre, pop_post):
        conns = nest.GetConnections(pop_pre, pop_post).get(
            ["source", "target", "weight"]
        )
        conns["source"] = conns["source"] - np.min(conns["source"])
        conns["target"] = conns["target"] - np.min(conns["target"])

        conns["len_source"] = len(pop_pre)
        conns["len_target"] = len(pop_post)

        return conns

    weights_pre_train = {
        "rec_rec": get_weights(nrns_rec, nrns_rec),
        "rec_out": get_weights(nrns_rec, nrns_out),
    }

    # %% ###########################################################################################################
    # Simulate and Process Results
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # We run the simulation, read out the recorded data, and calculate the training error.
    # Debug info about expected weight update cadence
    total_sequences = n_samples * n_iter
    expected_updates = total_sequences / gradient_batch_size
    print(
        f"Starting simulation for {duration['sim']:.2f} ms... (sequences per iter incl. silence: {total_sequences}; "
        f"update interval: {duration['total_sequence_with_silence']} ms; gradient_batch_size={gradient_batch_size}; "
        f"expected weight updates ≈ {expected_updates})"
    )
    nest.Simulate(duration["sim"])
    print("Simulation finished.")

    weights_post_train = {
        "rec_rec": get_weights(nrns_rec, nrns_rec),
        "rec_out": get_weights(nrns_rec, nrns_out),
        "rb_rec": get_weights(
            (
                nrns_rb
                if not use_manual_rbf
                else (parrot_neurons if plastic_input_to_rec else gen_poisson_in)
            ),
            nrns_rec,
        ),
    }
    events_mm_rec, events_mm_out, events_sr, events_wr = (
        mm_rec.get("events"),
        mm_out.get("events"),
        spike_recorder.get("events"),
        weight_recorder.get("events"),
    )

    # Evaluate training error (loss)
    readout_signal = events_mm_out["readout_signal"]
    target_signal = events_mm_out["target_signal"]
    senders = events_mm_out["senders"]

    loss_list = []
    for sender in set(senders):
        idc = senders == sender
        error = (readout_signal[idc] - target_signal[idc]) ** 2
        loss_list.append(
            0.5
            * np.add.reduceat(
                error,
                np.arange(
                    0,
                    int(duration["task"]),
                    int(duration["total_sequence_with_silence"]),
                ),
            )
        )

    loss = np.sum(loss_list, axis=0)

    # %% ###########################################################################################################
    # Plotting and Saving
    # ~~~~~~~~~~~~~~~~~~~~~
    # Generate and save plots and simulation results if requested.
    out_dir = result_dir if result_dir else "."

    # Save the final weights after training
    print("Saving trained weights...")
    np.savez(
        os.path.join(out_dir, "trained_weights.npz"),
        rec_rec=weights_post_train["rec_rec"],
        rec_out=weights_post_train["rec_out"],
        rb_rec=weights_post_train["rb_rec"],
    )
    print("Trained weights saved.")

    if plot_results and config.get("plotting", {}).get("do_plotting", True):
        print("Generating and saving plots...")
        colors = {"blue": "#1f77b4", "red": "#d62728", "white": "#ffffff"}
        plot_training_error(loss, os.path.join(out_dir, "training_error.png"))
        plot_spikes_and_dynamics(
            events_sr,
            events_mm_rec,
            events_mm_out,
            nrns_rec,
            n_record,
            duration,
            colors,
            os.path.join(out_dir, "spikes_and_dynamics.png"),
        )
        plot_weight_time_courses(
            events_wr,
            weights_pre_train,
            nrns_rec,
            nrns_out,
            n_record_w,
            colors,
            duration,
            os.path.join(out_dir, "weight_time_courses.png"),
        )
        plot_weight_matrices(
            weights_pre_train,
            weights_post_train,
            colors,
            os.path.join(out_dir, "weight_matrices.png"),
        )

    if result_dir:
        import json

        def make_json_serializable(obj):
            if isinstance(obj, dict):
                return {k: make_json_serializable(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [make_json_serializable(i) for i in obj]
            if isinstance(obj, Path):
                return str(obj)
            if isinstance(obj, (np.ndarray, np.generic)):
                return obj.tolist()
            return obj

        config_serializable = make_json_serializable(config)
        config_serializable["model_run_settings"] = {
            "plastic_input_to_rec": plastic_input_to_rec,
            "use_manual_rbf": use_manual_rbf,
        }
        np.savez(
            os.path.join(out_dir, "results.npz"),
            loss=loss,
            readout_signal=readout_signal,
            target_signal=target_signal,
        )
        with open(os.path.join(out_dir, "config.json"), "w") as f:
            json.dump(config_serializable, f, indent=2)
    print(f"Results saved to {out_dir}")

    return {
        "spikes": events_sr,                # dict with 'senders' and 'times'
        "output_multimeter": events_mm_out, # dict with 'senders', 'times', 'readout_signal', 'target_signal'
        "loss": loss,                       # 1D array
        "duration": duration,               # dict
        "nrns_rec": nrns_rec                # array of recurrent neuron IDs
    }


def collect_scan_results(results_dir, output_csv="scan_summary.csv"):
    import csv, json
    import numpy as np

    def flatten_dict(d, parent_key="", sep="."):
        """Flattens a nested dictionary."""
        items = []
        for k, v in d.items():
            new_key = parent_key + sep + k if parent_key else k
            if isinstance(v, dict):
                items.extend(flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    rows = []
    param_keys = set()
    temp_rows = []
    for folder in os.listdir(results_dir):
        config_path = os.path.join(results_dir, folder, "config.json")
        results_path = os.path.join(results_dir, folder, "results.npz")
        if os.path.exists(config_path) and os.path.exists(results_path):
            with open(config_path) as f:
                config = json.load(f)
            loss = np.load(results_path)["loss"]
            if len(loss) > 0:
                n_last = min(10, len(loss))
                final_loss = float(np.mean(loss[-n_last:]))
            else:
                final_loss = None

            # Flatten the entire config dictionary and create a row
            row = flatten_dict(config)
            row["folder"] = folder
            row["final_loss"] = final_loss
            temp_rows.append(row)

    if temp_rows:
        all_keys = set().union(*(r.keys() for r in temp_rows))
        for k in all_keys:
            values = [r.get(k, None) for r in temp_rows]

            hashable_values = [tuple(v) if isinstance(v, list) else v for v in values]

            if (
                len(set(hashable_values)) > 1
                or (
                    len(set(hashable_values)) == 1
                    and list(set(hashable_values))[0] is not None
                )
            ) and k not in ["folder", "final_loss"]:
                param_keys.add(k)

    main_cols = sorted(param_keys) + ["folder", "final_loss"]

    with open(os.path.join(results_dir, output_csv), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=main_cols)
        writer.writeheader()
        for row in temp_rows:
            filtered = {k: row.get(k, None) for k in main_cols}
            writer.writerow(filtered)
    print(f"Scan summary saved to {output_csv} with columns: {main_cols}")


# ---
# Main execution block
# This block parses command-line arguments to run single or multiple simulations (parameter scans).
# ---
if __name__ == "__main__":
    import argparse

    # ---
    # Usage Examples:
    #
    # 1. Run a single simulation with default parameters:
    #    python eprop-reaching-task.py
    #
    # 2. Run with manual RBF encoding:
    #    python eprop-reaching-task.py --use-manual-rbf
    #
    # 3. Make input-to-recurrent connections plastic:
    #    python eprop-reaching-task.py --plastic-input-to-rec
    #
    # 4. Scan a single parameter (e.g., number of recurrent neurons):
    #    python eprop-reaching-task.py --scan-param neurons.n_rec --scan-values 100,200,300
    #
    # 5. Set both excitatory and inhibitory learning rates simultaneously:
    #    python eprop-reaching-task.py --learning-rate 0.005
    #
    # 6. Perform a grid search over multiple parameters:
    #    python eprop-reaching-task.py --scan-param learning_rate_exc,rbf.num_centers --scan-values "0.01,0.001;10,20"
    #    (Note: Use quotes around --scan-values to avoid shell parsing issues with semicolons)
    #
    # 7. Combine a fixed parameter change with a scan:
    #    python eprop-reaching-task.py --learning-rate 0.001 --scan-param neurons.n_rec --scan-values 150,250
    # ---

    parser = argparse.ArgumentParser(
        description="Run e-prop reaching task parameter scan.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--scan-param",
        type=str,
        default=None,
        help="Comma-separated list of config parameters to scan (e.g., neurons.n_rec).",
    )
    parser.add_argument(
        "--scan-values",
        type=str,
        default=None,
        help='Semicolon-separated lists of values for each scan parameter (e.g., "100,200;10,20").',
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Set both excitatory and inhibitory learning rates simultaneously.",
    )
    parser.add_argument("--no-plot", action="store_true", help="Disable all plotting.")
    parser.add_argument(
        "--plastic-input-to-rec",
        action="store_true",
        help="Make input-to-recurrent connections plastic and fully connected.",
    )
    parser.add_argument(
        "--use-manual-rbf",
        action="store_true",
        help="Use manual RBF implementation instead of the default 'rb_neuron' model.",
    )
    parser.add_argument(
        "--trajectory-files",
        type=str,
        default=None,
        help="Comma-separated list of custom trajectory file paths to use instead of defaults.",
    )
    parser.add_argument(
        "--target-files",
        type=str,
        default=None,
        help="Comma-separated list of custom target file paths to use instead of defaults.",
    )
    args = parser.parse_args()

    # Prepare for single run or parameter scan
    results_dir = os.path.join(os.path.dirname(__file__), "sim_results")
    os.makedirs(results_dir, exist_ok=True)

    # Parse custom file arguments
    trajectory_files = (
        args.trajectory_files.split(",") if args.trajectory_files else None
    )
    target_files = args.target_files.split(",") if args.target_files else None
    base_kwargs = {
        "plot_results": not args.no_plot,
        "plastic_input_to_rec": args.plastic_input_to_rec,
        "use_manual_rbf": args.use_manual_rbf,
        "trajectory_files": trajectory_files,
        "target_files": target_files,
    }
    if args.learning_rate is not None:
        base_kwargs["learning_rate_exc"] = args.learning_rate
        base_kwargs["learning_rate_inh"] = args.learning_rate

    scan_params = (
        [p.strip() for p in args.scan_param.split(",")] if args.scan_param else []
    )
    scan_values = []
    if args.scan_values:
        for val_list in args.scan_values.split(";"):
            vals = [
                float(v) if v.replace(".", "", 1).replace("-", "", 1).isdigit() else v
                for v in val_list.split(",")
            ]
            scan_values.append(vals)

    # Execute simulation runs
    if scan_params and scan_values:
        if len(scan_params) != len(scan_values):
            print(
                f"ERROR: Number of scan parameters ({len(scan_params)}) does not match number of scan value lists ({len(scan_values)}).\n"
                f"Each parameter must have a corresponding list of values.\n"
                f"Example: --scan-param a,b --scan-values '1,2;3,4' for a=[1,2], b=[3,4]"
            )
            exit(1)
        print(f"--- Starting Parameter Scan ---")
        for combo in itertools.product(*scan_values):
            param_dict = dict(zip(scan_params, combo))
            scenario_kwargs = {**base_kwargs, **param_dict}

            # Map 'learning_rate' to both exc and inh if present
            if "learning_rate" in scenario_kwargs:
                scenario_kwargs["learning_rate_exc"] = scenario_kwargs["learning_rate"]
                scenario_kwargs["learning_rate_inh"] = scenario_kwargs["learning_rate"]
            # If only one of exc/inh is present, set only that one
            if "learning_rate_exc" in scenario_kwargs:
                scenario_kwargs["learning_rate_exc"] = scenario_kwargs[
                    "learning_rate_exc"
                ]
            if "learning_rate_inh" in scenario_kwargs:
                scenario_kwargs["learning_rate_inh"] = scenario_kwargs[
                    "learning_rate_inh"
                ]
            # If not present in scan, use base_kwargs
            if (
                "learning_rate_exc" not in scenario_kwargs
                and base_kwargs.get("learning_rate_exc") is not None
            ):
                scenario_kwargs["learning_rate_exc"] = base_kwargs["learning_rate_exc"]
            if (
                "learning_rate_inh" not in scenario_kwargs
                and base_kwargs.get("learning_rate_inh") is not None
            ):
                scenario_kwargs["learning_rate_inh"] = base_kwargs["learning_rate_inh"]

            folder_name = "_".join(
                f"{k.replace('.', '_')}_{v}" for k, v in param_dict.items()
            )
            # Add learning rates to folder name if present
            lr_exc = scenario_kwargs.get("learning_rate_exc")
            lr_inh = scenario_kwargs.get("learning_rate_inh")
            if lr_exc is not None:
                folder_name += f"_lr_exc_{lr_exc}"
            if lr_inh is not None and lr_inh != lr_exc:
                folder_name += f"_lr_inh_{lr_inh}"
            folder_name += (
                f"_plastic_{args.plastic_input_to_rec}_manualRBF_{args.use_manual_rbf}"
            )
            sim_dir = os.path.join(results_dir, folder_name)
            os.makedirs(sim_dir, exist_ok=True)
            print(f"\nRunning scenario: {param_dict}")
            run_simulation(result_dir=sim_dir, **scenario_kwargs)
        print("\n--- Parameter Scan Finished ---")
        collect_scan_results(results_dir)
    else:
        folder_name = f"default_plastic_{args.plastic_input_to_rec}_manualRBF_{args.use_manual_rbf}"
        # Add learning rates to folder name if present
        lr_exc = base_kwargs.get("learning_rate_exc")
        lr_inh = base_kwargs.get("learning_rate_inh")
        if lr_exc is not None:
            folder_name += f"_lr_exc_{lr_exc}"
        if lr_inh is not None and lr_inh != lr_exc:
            folder_name += f"_lr_inh_{lr_inh}"
        sim_dir = os.path.join(results_dir, folder_name)
        os.makedirs(sim_dir, exist_ok=True)
        print("--- Running Single Default Scenario ---")
        run_simulation(result_dir=sim_dir, **base_kwargs)
        print("\n--- Default Scenario Finished ---")

    # Plot summary graph comparing all runs
    # if not args.no_plot and os.path.exists(results_dir):
    #     print("\nGenerating summary loss curve plot...")
    #     plot_all_loss_curves(results_dir, showfig=False)
    #     print("Summary plot saved.")
