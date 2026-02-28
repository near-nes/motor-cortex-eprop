"""
M1Network: Encapsulates the e-prop M1 network for spike-based input tasks.
"""

try:
    from interfaces.m1_base import M1SubModule
except ImportError:
    print("Warning: interfaces.m1_base not found. Running in standalone mode.")

    class M1SubModule:
        pass


import nest
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple

from motor_controller_model.config_schema import MotorControllerConfig

from motor_controller_model.utils import load_spike_data
from motor_controller_model.plot_results import (
    plot_training_error,
    plot_spikes_and_dynamics,
    plot_weight_matrices,
)


class M1Network(M1SubModule):
    def __init__(self, config: MotorControllerConfig, artifacts_dir: Path):
        self.config = config
        self.artifacts_dir = Path(artifacts_dir)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.trained = False

        # Placeholders for network objects
        self.nrns_rec = None
        self.nrns_out = None
        self.nrns_rb = None
        self.nrns_parrot = None

    def _install_nestml_module(self):
        """Ensures the custom NESTML module is compiled and installed, with correct kernel setup."""
        nestml_install_dir = (
            Path(__file__).resolve().parent / "nestml_neurons" / "nestml_install"
        )
        module_path = nestml_install_dir / "motor_neuron_module.so"
        if not module_path.exists():
            print("Compiled module not found. Compiling NESTML neurons...")
            from motor_controller_model.nestml_neurons.compile_nestml_neurons import (
                compile_nestml_neurons,
            )

            compile_nestml_neurons()
        try:
            nest.Install(str(module_path))
            print("motor_neuron_module installed successfully.")
        except Exception as e:
            print(f"Failed to install module from {module_path}: {e}")
            try:
                nest.Install("motor_neuron_module")
            except Exception:
                pass

    def _setup_nest_kernel(self, duration_dict):
        # 1. Reset kernel to ensure clean state before installing module and setting parameters
        nest.ResetKernel()

        # 2. Install the NESTML module FIRST (while thread count is still 1)
        self._install_nestml_module()

        # 3. Set all kernel parameters
        params_setup = {
            "eprop_learning_window": duration_dict["learning_window"],
            "eprop_reset_neurons_on_update": False,
            "eprop_update_interval": duration_dict["total_sequence_with_silence"],
            "print_time": self.config.simulation.print_time,
            "resolution": self.config.simulation.step,
            "total_num_virtual_procs": self.config.simulation.total_num_virtual_procs,
            "rng_seed": self.config.simulation.rng_seed,
        }
        nest.set(**params_setup)

    def train(self, training_data: List[Dict[str, Tuple[str, str]]]):
        """
        Train the network using the provided spike input data.

        Args:
            training_data: List of dicts, each containing:
                {'input': (pos_file, neg_file), 'output': (pos_file, neg_file)}
        """
        print(f"Initializing M1 training with {len(training_data)} trajectories...")

        # 1. Timing and Duration Setup
        step_ms = self.config.simulation.step
        sequence_ms = self.config.task.sequence
        silent_ms = self.config.task.silent_period
        total_seq_ms = sequence_ms + silent_ms
        n_iter = self.config.task.n_iter
        n_samples = len(training_data)

        # Determine learning window
        learning_start = self.config.task.learning_start
        learning_end = self.config.task.learning_end
        learning_window = max(
            0.0, min(sequence_ms, learning_end) - max(0.0, learning_start)
        )

        duration = {
            "step": step_ms,
            "sequence": sequence_ms,
            "silent_period": silent_ms,
            "total_sequence_with_silence": total_seq_ms,
            "learning_window": learning_window,
            "task": int(round(total_seq_ms / step_ms)) * n_samples * n_iter * step_ms,
            "n_trajectories": n_samples,
        }
        duration["sim"] = duration["task"] + step_ms

        self._setup_nest_kernel(duration)

        # 2. Data Loading and Preparation
        input_spikes_list = []
        desired_targets_list = {"pos": [], "neg": []}
        n_timesteps_per_stimulus = int(round(sequence_ms / step_ms))

        for sample in training_data:
            # Input spikes
            in_pos = load_spike_data(sample["input"][0])
            in_neg = load_spike_data(sample["input"][1])
            input_spikes_list.append((in_pos, in_neg))

            # Target spikes -> Rate signal
            out_pos = load_spike_data(sample["output"][0])
            out_neg = load_spike_data(sample["output"][1])

            for key, data in zip(["pos", "neg"], [out_pos, out_neg]):
                hist = np.histogram(
                    data[:, 1], bins=n_timesteps_per_stimulus, range=(0, sequence_ms)
                )[0]
                smoothed = np.convolve(hist, np.ones(50) / 10, mode="same")

                # Prepend silence
                if silent_ms > 0:
                    silent_steps = int(silent_ms / step_ms)
                    smoothed = np.concatenate((np.zeros(silent_steps), smoothed))

                # Apply input shift (shift target forward)
                if self.config.task.input_shift_ms > 0:
                    shift_steps = int(self.config.task.input_shift_ms / step_ms)
                    shifted = np.roll(smoothed, shift_steps)
                    shifted[:shift_steps] = 0.0
                    smoothed = shifted

                desired_targets_list[key].append(smoothed)

        # 3. Create Neurons
        # RB Neurons
        rbf_cfg = self.config.rbf
        n_rb = rbf_cfg.num_centers

        # Logic for sdev and max_peak_rate in spike input mode
        sdev = (
            rbf_cfg.sdev_hz
            if rbf_cfg.sdev_hz is not None
            else rbf_cfg.desired_upper_hz * rbf_cfg.width
        )
        max_peak = (
            rbf_cfg.max_peak_rate_hz
            if rbf_cfg.max_peak_rate_hz is not None
            else rbf_cfg.desired_upper_hz
        )

        params_rb = {
            "kp": rbf_cfg.kp,
            "base_rate": rbf_cfg.base_rate,
            "buffer_size": rbf_cfg.buffer_size,
            "simulation_steps": int(duration["sim"] / step_ms + 1),
            "sdev": sdev,
            "max_peak_rate": max_peak,
        }
        self.nrns_rb = nest.Create("rb_neuron", n_rb)
        nest.SetStatus(self.nrns_rb, params_rb)

        # Set desired rates for rb_neurons
        desired_rates = np.linspace(
            rbf_cfg.shift_min_rate, rbf_cfg.desired_upper_hz, n_rb
        )
        for i, nrn in enumerate(self.nrns_rb):
            nest.SetStatus(nrn, {"desired": desired_rates[i]})

        # Input Spike Generators & Parrot Neurons
        all_senders = set()
        for pos, neg in input_spikes_list:
            all_senders.update(pos[:, 0].astype(int))
            all_senders.update(neg[:, 0].astype(int))
        sender_to_idx = {s: i for i, s in enumerate(sorted(all_senders))}
        n_input_total = len(all_senders)

        spike_times_per_neuron = [[] for _ in range(n_input_total)]

        for iter_num in range(n_iter):
            for traj_idx, (pos, neg) in enumerate(input_spikes_list):
                offset = (traj_idx + iter_num * n_samples) * total_seq_ms
                for data in [pos, neg]:
                    for s_id, t in data:
                        idx = sender_to_idx[int(s_id)]
                        spike_times_per_neuron[idx].append(t + offset + silent_ms)

        self.nrns_parrot = nest.Create("parrot_neuron", n_input_total)
        spike_gens = nest.Create("spike_generator", n_input_total)
        for i, times in enumerate(spike_times_per_neuron):
            nest.SetStatus(spike_gens[i : i + 1], {"spike_times": sorted(times)})
        nest.Connect(spike_gens, self.nrns_parrot, "one_to_one")
        nest.Connect(
            self.nrns_parrot,
            self.nrns_rb,
            "all_to_all",
            {
                "synapse_model": "static_synapse",
                "delay": self.config.synapses.static_delay,
                "weight": 1.0,
            },
        )

        # Recurrent & Output Neurons
        n_rec = self.config.neurons.n_rec
        n_out = self.config.neurons.n_out
        n_exc = int(n_rec * self.config.neurons.exc_ratio)
        n_inh = n_rec - n_exc

        self.nrns_rec = nest.Create(
            "eprop_iaf_bsshslm_2020", n_exc, self.config.neurons.rec.model_dump()
        ) + nest.Create(
            "eprop_iaf_bsshslm_2020", n_inh, self.config.neurons.rec.model_dump()
        )
        self.nrns_out = nest.Create(
            "eprop_readout_bsshslm_2020", n_out, self.config.neurons.out.model_dump()
        )

        # Target Rate Generators
        gen_rate_target = nest.Create("step_rate_generator", n_out)
        target_amp_times = (
            np.arange(len(desired_targets_list["pos"][0]) * n_samples * n_iter)
            * step_ms
            + step_ms
        )
        concat_targets = {
            k: np.tile(np.concatenate(v), n_iter)
            for k, v in desired_targets_list.items()
        }

        nest.SetStatus(
            gen_rate_target[0],
            {
                "amplitude_times": target_amp_times,
                "amplitude_values": concat_targets["pos"],
            },
        )
        nest.SetStatus(
            gen_rate_target[1],
            {
                "amplitude_times": target_amp_times,
                "amplitude_values": concat_targets["neg"],
            },
        )

        # 4. Connections
        syn_cfg = self.config.synapses
        w_rec = syn_cfg.w_rec

        nest.CopyModel(
            "eprop_synapse_bsshslm_2020",
            "eprop_synapse_exc",
            {
                "optimizer": {
                    **syn_cfg.exc.optimizer.model_dump(),
                    "batch_size": self.config.task.gradient_batch_size,
                },
                "average_gradient": syn_cfg.average_gradient,
            },
        )
        nest.CopyModel(
            "eprop_synapse_bsshslm_2020",
            "eprop_synapse_inh",
            {
                "optimizer": {
                    **syn_cfg.inh.optimizer.model_dump(),
                    "batch_size": self.config.task.gradient_batch_size,
                },
                "weight": syn_cfg.inh.weight,
                "average_gradient": syn_cfg.average_gradient,
            },
        )

        # RB -> Rec
        nest.Connect(
            self.nrns_rb,
            self.nrns_rec,
            "all_to_all",
            {
                "synapse_model": "eprop_synapse_exc",
                "delay": syn_cfg.static_delay,
                "weight": nest.math.redraw(
                    nest.random.normal(mean=syn_cfg.w_input, std=syn_cfg.w_input * 0.1),
                    min=0.0,
                    max=1000.0,
                ),
            },
        )

        # Rec -> Rec
        nest.Connect(
            self.nrns_rec[:n_exc],
            self.nrns_rec,
            {
                "rule": "pairwise_bernoulli",
                "p": syn_cfg.conn_bernoulli_p,
                "allow_autapses": False,
            },
            {
                "synapse_model": "eprop_synapse_exc",
                "delay": step_ms,
                "tau_m_readout": self.config.neurons.out.tau_m,
                "weight": nest.math.redraw(
                    nest.random.normal(mean=w_rec, std=w_rec * 0.1), min=0.0, max=1000.0
                ),
            },
        )

        nest.Connect(
            self.nrns_rec[n_exc:],
            self.nrns_rec,
            {
                "rule": "pairwise_bernoulli",
                "p": syn_cfg.conn_bernoulli_p,
                "allow_autapses": False,
            },
            {
                "synapse_model": "eprop_synapse_inh",
                "delay": step_ms,
                "tau_m_readout": self.config.neurons.out.tau_m,
                "weight": nest.math.redraw(
                    nest.random.normal(
                        mean=-w_rec * syn_cfg.g, std=syn_cfg.g * w_rec * 0.1
                    ),
                    min=-1000.0,
                    max=0.0,
                ),
            },
        )

        # Rec -> Out
        nest.Connect(
            self.nrns_rec[:n_exc],
            self.nrns_out,
            "all_to_all",
            {
                "synapse_model": "eprop_synapse_exc",
                "delay": step_ms,
                "tau_m_readout": self.config.neurons.out.tau_m,
                "weight": nest.math.redraw(
                    nest.random.normal(mean=w_rec, std=w_rec * 0.1), min=0.0, max=1000.0
                ),
            },
        )

        # Feedback & Target
        nest.Connect(
            self.nrns_out,
            self.nrns_rec[:n_exc],
            "all_to_all",
            {
                "synapse_model": "eprop_learning_signal_connection_bsshslm_2020",
                "delay": syn_cfg.feedback_delay,
                "weight": nest.math.redraw(
                    nest.random.normal(mean=w_rec, std=w_rec * 0.1), min=0.0, max=1000.0
                ),
            },
        )

        nest.Connect(
            gen_rate_target[0],
            self.nrns_out[0],
            "one_to_one",
            {
                "synapse_model": "rate_connection_delayed",
                "delay": syn_cfg.rate_target_delay,
                "receptor_type": syn_cfg.receptor_type,
            },
        )
        nest.Connect(
            gen_rate_target[1],
            self.nrns_out[1],
            "one_to_one",
            {
                "synapse_model": "rate_connection_delayed",
                "delay": syn_cfg.rate_target_delay,
                "receptor_type": syn_cfg.receptor_type,
            },
        )

        # 5. Recorders
        rec_cfg = self.config.recording

        # Readout multimeter
        mm_out = nest.Create(
            "multimeter",
            {
                **rec_cfg.mm_out.model_dump(),
                "interval": step_ms,
                "start": step_ms,
                "stop": duration["task"],
            },
        )
        nest.Connect(mm_out, self.nrns_out)

        # Recurrent layer multimeter
        mm_rec = nest.Create(
            "multimeter",
            {
                **rec_cfg.mm_rec.model_dump(),
                "interval": step_ms,
                "start": step_ms,
                "stop": duration["task"],
            },
        )
        nrns_rec_record = self.nrns_rec[: rec_cfg.n_record]
        nest.Connect(mm_rec, nrns_rec_record)

        # Recurrent layer spike recorder
        spike_recorder = nest.Create(
            "spike_recorder", {"start": step_ms, "stop": duration["task"]}
        )
        nest.Connect(self.nrns_rec, spike_recorder)

        # Force final update
        gen_final = nest.Create(
            "spike_generator", 1, {"spike_times": [duration["task"] + step_ms]}
        )
        nest.Connect(gen_final, self.nrns_rec, "all_to_all", {"weight": 1000.0})

        # Capture pre-training weights for the matrix plots
        weights_pre_train = {
            "rec_rec": self._get_weights(self.nrns_rec, self.nrns_rec),
            "rec_out": self._get_weights(self.nrns_rec, self.nrns_out),
        }

        # 6. Simulation
        print(f"Simulating for {duration['sim']} ms...")
        nest.Simulate(duration["sim"])
        self.trained = True
        self.save_weights()

        # Capture post-training weights
        weights_post_train = {
            "rec_rec": self._get_weights(self.nrns_rec, self.nrns_rec),
            "rec_out": self._get_weights(self.nrns_rec, self.nrns_out),
            "rb_rec": self._get_weights(self.nrns_rb, self.nrns_rec),
        }

        # 7. Loss Calculation & Plotting
        events_mm_out = mm_out.get("events")
        readout_signal = events_mm_out["readout_signal"]
        target_signal = events_mm_out["target_signal"]
        senders = events_mm_out["senders"]

        # Calculate MSE loss per trajectory block
        loss_list = []
        for sender in set(senders):
            idc = senders == sender
            error = (readout_signal[idc] - target_signal[idc]) ** 2
            task_steps = int(duration["task"] / step_ms)
            seq_with_silence_steps = int(
                duration["total_sequence_with_silence"] / step_ms
            )
            loss_list.append(
                0.5
                * np.add.reduceat(
                    error, np.arange(0, task_steps, seq_with_silence_steps)
                )
            )
        loss = np.sum(loss_list, axis=0)

        # Generate the plots
        if self.config.plotting.do_plotting:
            print("Generating plots...")
            colors = {"blue": "#1f77b4", "red": "#d62728", "white": "#ffffff"}

            plot_training_error(loss, self.artifacts_dir / "training_error.png")

            events_sr = spike_recorder.get("events")
            events_mm_rec = mm_rec.get("events")
            plot_spikes_and_dynamics(
                events_sr,
                events_mm_rec,
                events_mm_out,
                self.nrns_rec,
                rec_cfg.n_record,
                duration,
                colors,
                self.artifacts_dir / "spikes_and_dynamics.png",
            )

            plot_weight_matrices(
                weights_pre_train,
                weights_post_train,
                colors,
                self.artifacts_dir / "weight_matrices.png",
            )
            print(f"Plots saved to {self.artifacts_dir}")

    def _get_weights(self, pop_pre, pop_post):
        conns = nest.GetConnections(pop_pre, pop_post).get(
            ["source", "target", "weight"]
        )
        if not len(conns["source"]):  # Safe check for empty connections
            return {
                "source": [],
                "target": [],
                "weight": [],
                "len_source": len(pop_pre),
                "len_target": len(pop_post),
            }

        conns["source"] = np.array(conns["source"]) - np.min(conns["source"])
        conns["target"] = np.array(conns["target"]) - np.min(conns["target"])
        conns["len_source"] = len(pop_pre)
        conns["len_target"] = len(pop_post)
        return conns

    def save_weights(self):
        """Saves trained weights to artifacts directory and keeps them in memory."""
        self.saved_weights = {
            "rec_rec": self._get_weights(self.nrns_rec, self.nrns_rec),
            "rec_out": self._get_weights(self.nrns_rec, self.nrns_out),
            "rb_rec": self._get_weights(self.nrns_rb, self.nrns_rec),
        }
        np.savez(self.artifacts_dir / "trained_weights.npz", **self.saved_weights)
        print(f"Weights saved to {self.artifacts_dir / 'trained_weights.npz'}")

    def load_weights(self, weights_path: Path):
        """Loads weights from file directly into memory."""
        if not weights_path.exists():
            raise FileNotFoundError(f"Weights file not found: {weights_path}")

        # Load npz and convert numpy object arrays back to standard dictionaries
        loaded = np.load(weights_path, allow_pickle=True)
        self.saved_weights = {key: loaded[key].item() for key in loaded.files}

        self.trained = True
        print(f"Weights successfully loaded into memory from {weights_path}")

    def connect(self, source_population):
        """
        Connect source population (Planner) to this M1 submodule.
        Pipes Planner spikes directly into the M1 RB neurons for inference.
        """
        if getattr(self, "nrns_rb", None) is not None:
            nest.Connect(
                source_population,
                self.nrns_rb,
                "all_to_all",
                {"synapse_model": "static_synapse", "weight": 1.0},
            )
        else:
            print("Warning: nrns_rb not found. Did you call build_network()?")

    def get_output_pops(self):
        """
        Get the output populations from this M1 submodule.
        Returns the positive and negative readout neurons.
        """
        if self.nrns_out is not None and len(self.nrns_out) >= 2:
            # nrns_out[0] is positive, nrns_out[1] is negative
            return self.nrns_out[0], self.nrns_out[1]
        return None, None

    def build_network(self, simulation_time_ms: float = None):
        """
        Builds a clean SNN in NEST for inference/integration.
        Uses exact saved topology and static synapses for maximum performance.
        """
        if getattr(self, "saved_weights", None) is None:
            print("Error: No weights loaded. Cannot build network.")
            return

        print("Building clean M1 network for integration...")
        self._install_nestml_module()

        step_ms = self.config.simulation.step

        # Calculate simulation time based on task parameters if not provided
        if simulation_time_ms is None:
            sequence_ms = self.config.task.sequence
            silent_ms = self.config.task.silent_period

            n_timesteps_per_seq = int(round((sequence_ms + silent_ms) / step_ms))
            n_samples_per_traj = self.config.task.n_samples_per_trajectory_to_use
            n_trajectories = len(self.config.task.trajectory_ids_to_use)
            n_samples = n_trajectories * n_samples_per_traj

            duration_task = n_timesteps_per_seq * n_samples * step_ms
            simulation_time_ms = duration_task + step_ms

        # Calculate exact array size for rb_neuron to prevent NEST memory leaks
        simulation_steps = int(simulation_time_ms / step_ms + 1)

        # 1. Create Neurons
        rbf_cfg = self.config.rbf
        n_rb = rbf_cfg.num_centers
        sdev = (
            rbf_cfg.sdev_hz
            if rbf_cfg.sdev_hz is not None
            else rbf_cfg.desired_upper_hz * rbf_cfg.width
        )
        max_peak = (
            rbf_cfg.max_peak_rate_hz
            if rbf_cfg.max_peak_rate_hz is not None
            else rbf_cfg.desired_upper_hz
        )

        params_rb = {
            "kp": rbf_cfg.kp,
            "base_rate": rbf_cfg.base_rate,
            "buffer_size": rbf_cfg.buffer_size,
            "simulation_steps": simulation_steps,
            "sdev": sdev,
            "max_peak_rate": max_peak,
        }
        self.nrns_rb = nest.Create("rb_neuron", n_rb, params_rb)

        desired_rates = np.linspace(
            rbf_cfg.shift_min_rate, rbf_cfg.desired_upper_hz, n_rb
        )
        for i, nrn in enumerate(self.nrns_rb):
            nest.SetStatus(nrn, {"desired": desired_rates[i]})

        n_rec = self.config.neurons.n_rec
        n_out = self.config.neurons.n_out
        n_exc = int(n_rec * self.config.neurons.exc_ratio)
        n_inh = n_rec - n_exc

        self.nrns_rec = nest.Create(
            "eprop_iaf_bsshslm_2020", n_exc, self.config.neurons.rec.model_dump()
        ) + nest.Create(
            "eprop_iaf_bsshslm_2020", n_inh, self.config.neurons.rec.model_dump()
        )
        self.nrns_out = nest.Create(
            "eprop_readout_bsshslm_2020", n_out, self.config.neurons.out.model_dump()
        )

        # 2. Explicit Topology Injection
        def apply_explicit_topology(pop_pre, pop_post, key, delay_val):
            w_dict = self.saved_weights.get(key)
            if not w_dict or len(w_dict["source"]) == 0:
                print(f"Warning: No weights found for {key}")
                return

            sources = list(
                np.array(w_dict["source"], dtype=int) + min(pop_pre.tolist())
            )
            targets = list(
                np.array(w_dict["target"], dtype=int) + min(pop_post.tolist())
            )
            weights = list(w_dict["weight"])
            delays = [delay_val] * len(weights)

            nest.Connect(
                sources,
                targets,
                "one_to_one",
                {"synapse_model": "static_synapse", "weight": weights, "delay": delays},
            )

        apply_explicit_topology(self.nrns_rb, self.nrns_rec, "rb_rec", step_ms)
        apply_explicit_topology(self.nrns_rec, self.nrns_rec, "rec_rec", step_ms)
        apply_explicit_topology(self.nrns_rec, self.nrns_out, "rec_out", step_ms)

        print("M1 Network successfully built and ready for integration.")
