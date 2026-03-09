"""
M1Network: Encapsulates the e-prop M1 network for spike-based input tasks.
"""

from pathlib import Path
from typing import Protocol, Tuple

import nest
import numpy as np
import structlog

from .config_schema import MotorControllerConfig


class M1SubModule(Protocol):
    """Structural interface for M1 submodule implementations."""

    def connect(self, source_population): ...
    def get_output_pops(self) -> Tuple: ...


def get_weights(pop_pre, pop_post):
    """Extract connection weights between two populations as a dictionary."""
    conns = nest.GetConnections(pop_pre, pop_post).get(["source", "target", "weight"])
    if not len(conns["source"]):
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


class M1Network:
    def __init__(self, config: MotorControllerConfig):
        self._log = structlog.get_logger("m1_network")
        self.config = config
        self.trained = False

        # Placeholders for network objects
        self.nrns_rec = None
        self.nrns_out = None
        self.nrns_rb = None
        self.nrns_parrot = None

    def save_weights(self, path: Path):
        """Saves trained weights to file and keeps them in memory."""
        self.saved_weights = {
            "rec_rec": get_weights(self.nrns_rec, self.nrns_rec),
            "rec_out": get_weights(self.nrns_rec, self.nrns_out),
            "rb_rec": get_weights(self.nrns_rb, self.nrns_rec),
        }
        np.savez(path, **self.saved_weights)
        self._log.debug("weights saved", path=str(path))

    def load_weights(self, weights_path: Path):
        """Loads weights from file directly into memory."""
        if not weights_path.exists():
            raise FileNotFoundError(f"Weights file not found: {weights_path}")

        # Load npz and convert numpy object arrays back to standard dictionaries
        loaded = np.load(weights_path, allow_pickle=True)
        self.saved_weights = {key: loaded[key].item() for key in loaded.files}

        self.trained = True
        self._log.debug("weights loaded", path=str(weights_path))

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
            self._log.warning("nrns_rb not found, call build_network() first")

    def get_output_pops(self):
        """
        Get the output populations from this M1 submodule.
        Returns the positive and negative readout neurons.
        """
        if self.nrns_out is not None and len(self.nrns_out) >= 2:
            return self.nrns_out[0], self.nrns_out[1]
        return None, None

    def build_network(
        self,
        simulation_time_ms: float = None,
        train: bool = False,
        output_neuron_model: str = "eprop_readout_bsshslm_2020",
    ):
        """
        Builds the M1 SNN in NEST.

        When train=False (default): uses saved weights with static synapses for inference.
        When train=True: creates plastic e-prop synapses with random initial weights for training.

        Args:
            simulation_time_ms: Total simulation duration in ms. If None, calculated from config.
            train: If True, build for training with plastic synapses. If False, build for inference.
            output_neuron_model: Neuron model for output layer (ignored when train=True).
        """
        if not train and getattr(self, "saved_weights", None) is None:
            self._log.error("no weights loaded, cannot build network for inference")
            return

        self._log.debug(
            "building M1 network", mode="training" if train else "inference"
        )

        step_ms = self.config.simulation.step

        # Calculate simulation time if not provided
        if simulation_time_ms is None:
            sequence_ms = self.config.task.sequence
            silent_ms = self.config.task.silent_period
            n_timesteps_per_seq = int(round((sequence_ms + silent_ms) / step_ms))
            n_samples_per_traj = self.config.task.n_samples_per_trajectory_to_use
            n_trajectories = len(self.config.task.trajectory_ids_to_use)
            n_samples = n_trajectories * n_samples_per_traj
            duration_task = n_timesteps_per_seq * n_samples * step_ms
            simulation_time_ms = duration_task + step_ms

        simulation_steps = int(simulation_time_ms / step_ms + 1)

        # 1. Create RB Neurons
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
        self.nrns_rb = nest.Create("rb_neuron_nestml", n_rb)
        nest.SetStatus(self.nrns_rb, params_rb)

        desired_rates = np.linspace(
            rbf_cfg.shift_min_rate, rbf_cfg.desired_upper_hz, n_rb
        )
        for i, nrn in enumerate(self.nrns_rb):
            nest.SetStatus(nrn, {"desired": desired_rates[i]})

        # 2. Create Recurrent & Output Neurons
        n_rec = self.config.neurons.n_rec
        n_out = self.config.neurons.n_out
        n_exc = int(n_rec * self.config.neurons.exc_ratio)

        self.nrns_rec = nest.Create(
            "eprop_iaf_bsshslm_2020", n_exc, self.config.neurons.rec.model_dump()
        ) + nest.Create(
            "eprop_iaf_bsshslm_2020",
            n_rec - n_exc,
            self.config.neurons.rec.model_dump(),
        )

        if train:
            # Training: always use eprop readout for learning signal
            self.nrns_out = nest.Create(
                "eprop_readout_bsshslm_2020",
                n_out,
                self.config.neurons.out.model_dump(),
            )
        else:
            # Inference: use specified output neuron model
            self.nrns_out = nest.Create(output_neuron_model, n_out)
            if output_neuron_model == "eprop_readout_bsshslm_2020":
                nest.SetStatus(self.nrns_out, self.config.neurons.out.model_dump())
            else:
                nest.SetStatus(self.nrns_out, {"simulation_steps": simulation_steps})

        # 3. Create Connections
        if train:
            self._connect_for_training(step_ms, n_exc)
        else:
            self._connect_from_saved_weights(step_ms)

    def _connect_for_training(self, step_ms, n_exc):
        """Create plastic e-prop connections for training."""
        syn_cfg = self.config.synapses
        w_rec = syn_cfg.w_rec
        n_rec = self.config.neurons.n_rec

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

        # Rec -> Rec (exc)
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

        # Rec -> Rec (inh)
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

        # Feedback: Out -> Rec (learning signal)
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

    def _connect_from_saved_weights(self, step_ms):
        """Inject exact saved topology with static synapses for inference."""

        def apply_explicit_topology(pop_pre, pop_post, key, delay_val):
            w_dict = self.saved_weights.get(key)
            if not w_dict or len(w_dict["source"]) == 0:
                self._log.warning("no weights found", key=key)
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

        for label, pre, post, key in [
            ("rb→rec", self.nrns_rb, self.nrns_rec, "rb_rec"),
            ("rec→rec", self.nrns_rec, self.nrns_rec, "rec_rec"),
            ("rec→out", self.nrns_rec, self.nrns_out, "rec_out"),
        ]:
            self._log.debug(
                "connecting",
                conn=label,
                pre_type=nest.GetStatus(pre[0], "model"),
                post_type=nest.GetStatus(post[0], "model"),
                pre_ids=[pre[0].tolist(), "...", pre[-1].tolist()],
                post_ids=[post[0].tolist(), "...", post[-1].tolist()],
            )
            apply_explicit_topology(pre, post, key, step_ms)
            self._log.debug("connected", conn=label)

        self._log.debug("M1 network built and ready for integration")
