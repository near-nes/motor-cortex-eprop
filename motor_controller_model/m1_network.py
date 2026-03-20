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
    # consider a more robust global_to_local_ids-like function
    conns["source"] = np.array(conns["source"]) - np.min(conns["source"])
    conns["target"] = np.array(conns["target"]) - np.min(conns["target"])
    conns["len_source"] = len(pop_pre)
    conns["len_target"] = len(pop_post)
    return conns


class M1Network:
    def __init__(self, config: MotorControllerConfig):
        self._log: structlog.BoundLogger = structlog.get_logger("m1_network")
        self.config = config
        self.trained = False

        self.nrns_rec = None
        self.nrns_out_p = None
        self.nrns_out_n = None
        self.nrns_rb = None
        self.nrns_parrot = None

    def save_weights(self, path: Path):
        """Saves trained weights to file and keeps them in memory."""
        nrns_out = self.nrns_out_p + self.nrns_out_n
        self.saved_weights = {
            "rec_rec": get_weights(self.nrns_rec, self.nrns_rec),
            "rec_out": get_weights(self.nrns_rec, nrns_out),
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

    def connect(self, source_population, params={}):
        """
        Connect source population (Planner) to this M1 submodule.
        Pipes Planner spikes directly into the M1 RB neurons for inference.
        """
        nest.Connect(
            source_population,
            self.nrns_rb,
            "all_to_all",
            {"synapse_model": "static_synapse", "weight": 1.0, **params},
        )
        return nest.GetConnections(source_population, self.nrns_rb)

    def get_output_pops(self):
        """
        Get the output populations from this M1 submodule.
        Returns the positive and negative readout neuron populations.
        """
        return self.nrns_out_p, self.nrns_out_n

    def build_network(
        self,
        simulation_time_ms: float,
        train: bool = False,
        output_neuron_model: str = "eprop_readout_bsshslm_2020",
        output_neuron_params: dict | None = None,
        n_out: int | None = None,
    ):
        """
        Builds the M1 SNN in NEST.

        When train=False (default): uses saved weights with static synapses for inference.
        When train=True: creates plastic e-prop synapses with random initial weights for training.

        Args:
            simulation_time_ms: Total simulation duration in ms.
            train: If True, build for training with plastic synapses. If False, build for inference.
            output_neuron_model: Neuron model for output layer (ignored when train=True).
            output_neuron_params: Parameters for the output neuron model. If None,
                uses config neurons.out params. ``simulation_steps`` is always
                added automatically.
            n_out: Total number of output neurons (must be even). If None, uses
                config n_out (default 2). When larger than n_out, the trained rec→out
                weights are tiled so each half of the population receives the same
                projection, giving independent Poisson realizations that reduce noise.
        """
        if not train and getattr(self, "saved_weights", None) is None:
            raise ValueError(f"no weights loaded, cannot build network for inference")

        n_out = n_out or self.config.neurons.n_out

        if n_out % 2 != 0:
            raise ValueError(
                f"n_out_pop must be even (got {n_out}), "
                "each half maps to pos/neg output channel"
            )
        if train and n_out != self.config.neurons.n_out:
            raise ValueError(
                "n_out_pop override is only supported for inference, not training"
            )

        self._log.debug(
            "building M1 network", mode="training" if train else "inference"
        )

        step_ms = self.config.simulation.step
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
        n_per_channel = n_out // 2
        n_exc = int(n_rec * self.config.neurons.exc_ratio)

        self.nrns_rec = nest.Create(
            "eprop_iaf_bsshslm_2020", n_exc, self.config.neurons.rec.model_dump()
        ) + nest.Create(
            "eprop_iaf_bsshslm_2020",
            n_rec - n_exc,
            self.config.neurons.rec.model_dump(),
        )

        if train:
            self.nrns_out_p = nest.Create(
                "eprop_readout_bsshslm_2020",
                n_per_channel,
                self.config.neurons.out.model_dump(),
            )
            self.nrns_out_n = nest.Create(
                "eprop_readout_bsshslm_2020",
                n_per_channel,
                self.config.neurons.out.model_dump(),
            )
        else:
            out_params = output_neuron_params or self.config.neurons.out.model_dump()
            self.nrns_out_p = nest.Create(output_neuron_model, n_per_channel)
            nest.SetStatus(self.nrns_out_p, out_params)
            self.nrns_out_n = nest.Create(output_neuron_model, n_per_channel)
            nest.SetStatus(self.nrns_out_n, out_params)

        # 3. Create Connections
        if train:
            self._connect_for_training(step_ms, n_exc)
        else:
            self._connect_from_saved_weights()

    def _connect_for_training(self, step_ms, n_exc):
        """Create plastic e-prop connections for training.

        Uses a single synapse model for all plastic connections (no Dale's law).
        All recurrent neurons connect to output and receive the learning signal.
        Weight initialization follows N(0, 1/sqrt(n_pre)) per the NEST e-prop
        reference example.
        """
        syn_cfg = self.config.synapses
        n_rec = self.config.neurons.n_rec
        n_rb = self.config.rbf.num_centers
        optimizer_cfg = syn_cfg.syn.optimizer

        nest.CopyModel(
            "eprop_synapse_bsshslm_2020",
            "eprop_synapse_m1",
            {
                "optimizer": {
                    **optimizer_cfg.model_dump(),
                    "batch_size": self.config.task.gradient_batch_size,
                },
                "average_gradient": syn_cfg.average_gradient,
            },
        )

        # Weight init: 1 / sqrt(n_pre), mean=0, matching the NEST e-prop
        # reference example. Works because V_th is set low (0.03 mV).
        w_std_input = 1.0 / np.sqrt(n_rb)
        w_std_rec = 1.0 / np.sqrt(n_rec)

        # RB -> Rec
        nest.Connect(
            self.nrns_rb,
            self.nrns_rec,
            "all_to_all",
            {
                "synapse_model": "eprop_synapse_m1",
                "delay": syn_cfg.static_delay,
                "weight": nest.math.redraw(
                    nest.random.normal(mean=0.0, std=w_std_input),
                    min=optimizer_cfg.Wmin,
                    max=optimizer_cfg.Wmax,
                ),
            },
        )

        # Rec -> Rec (all neurons, no E/I split)
        nest.Connect(
            self.nrns_rec,
            self.nrns_rec,
            {
                "rule": "pairwise_bernoulli",
                "p": syn_cfg.conn_bernoulli_p,
                "allow_autapses": False,
            },
            {
                "synapse_model": "eprop_synapse_m1",
                "delay": step_ms,
                "tau_m_readout": self.config.neurons.out.tau_m,
                "weight": nest.math.redraw(
                    nest.random.normal(mean=0.0, std=w_std_rec),
                    min=optimizer_cfg.Wmin,
                    max=optimizer_cfg.Wmax,
                ),
            },
        )

        # Rec -> Out (all recurrent neurons)
        nrns_out = self.nrns_out_p + self.nrns_out_n
        nest.Connect(
            self.nrns_rec,
            nrns_out,
            "all_to_all",
            {
                "synapse_model": "eprop_synapse_m1",
                "delay": step_ms,
                "tau_m_readout": self.config.neurons.out.tau_m,
                "weight": nest.math.redraw(
                    nest.random.normal(mean=0.0, std=w_std_rec),
                    min=optimizer_cfg.Wmin,
                    max=optimizer_cfg.Wmax,
                ),
            },
        )

        # Feedback: Out -> Rec (learning signal to ALL recurrent neurons)
        nest.Connect(
            nrns_out,
            self.nrns_rec,
            "all_to_all",
            {
                "synapse_model": "eprop_learning_signal_connection_bsshslm_2020",
                "delay": syn_cfg.feedback_delay,
                "weight": nest.math.redraw(
                    nest.random.normal(mean=0.0, std=w_std_rec),
                    min=optimizer_cfg.Wmin,
                    max=optimizer_cfg.Wmax,
                ),
            },
        )

    def _connect_from_saved_weights(self):
        """Inject exact saved topology with static synapses for inference."""

        def apply_saved(pop_pre, pop_post, key):
            w_dict = self.saved_weights.get(key)
            if not w_dict or len(w_dict["source"]) == 0:
                self._log.warning("no weights found", key=key)
                return

            pre_ids = pop_pre.tolist()
            post_ids = pop_post.tolist()
            sources = [pre_ids[s] for s in w_dict["source"]]
            targets = [post_ids[t] for t in w_dict["target"]]
            weights = list(w_dict["weight"])

            nest.Connect(
                sources,
                targets,
                "one_to_one",
                {"synapse_model": "static_synapse", "weight": weights},
            )

        apply_saved(self.nrns_rb, self.nrns_rec, "rb_rec")
        apply_saved(self.nrns_rec, self.nrns_rec, "rec_rec")

        # rec→out: tile trained weights across output populations
        w_dict = self.saved_weights.get("rec_out")

        base_sources = np.array(w_dict["source"], dtype=int)
        base_targets = np.array(w_dict["target"], dtype=int)
        base_weights = np.array(w_dict["weight"])
        rec_ids = np.array(self.nrns_rec)
        self.conns_rec_out = []

        for trained_idx, pop in enumerate([self.nrns_out_p, self.nrns_out_n]):
            mask = base_targets == trained_idx
            src = rec_ids[base_sources[mask]]
            wts = base_weights[mask]
            pop_ids = np.array(pop)

            # Each output neuron gets same trained projection:
            # sources: [s0, s1, ..., sK, s0, s1, ..., sK, ...]
            # weights: [w0, w1, ..., wK, w0, w1, ..., wK, ...]
            # targets: [t0, t0, ..., t0, t1, t1, ..., t1, ...]
            nest.Connect(
                np.tile(src, len(pop_ids)),
                np.repeat(pop_ids, len(src)),
                "one_to_one",
                {
                    "synapse_model": "static_synapse",
                    "weight": np.tile(wts, len(pop_ids)),
                },
            )
            # for s, t, w in zip(
            #     np.tile(src, len(pop_ids)).tolist(),
            #     np.repeat(pop_ids, len(src)).tolist(),
            #     np.tile(wts, len(pop_ids)).tolist(),
            # ):
            #     nest.Connect(
            #         [s],
            #         [t],
            #         "one_to_one",
            #         {
            #             "synapse_model": "static_synapse",
            #             "weight": [0],  # 0
            #         },
            #     )
            #     self.conns_rec_out.append((s, t, w))

        self._log.debug("M1 network built and ready for integration")

    def connect_rec_out(self):
        self._log.debug("connecting updated weights")
        for s, t, w in self.conns_rec_out:
            conn = nest.GetConnections(
                nest.NodeCollection([s]), nest.NodeCollection([t])
            )
            nest.SetStatus(conn, {"weight": w})
        self._log.debug("connected updated weights rec to out")
