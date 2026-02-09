import numpy as np
import structlog
import yaml
from interfaces.m1_base import M1SubModule

_log = structlog.get_logger("M1MotorCortexEprop").bind(module="M1Eprop")


class M1MotorCortexEprop(M1SubModule):
    def __init__(
        self,
        config_path,
        weights_path,
        sim_steps,
        nest_instance,
        expected_delay,
        resolution,
    ):
        self.nest = nest_instance
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        self.sim_steps = sim_steps
        self._validate_delay(expected_delay)

        weights = np.load(weights_path, allow_pickle=True)
        self.rec_rec_weights = weights.get("rec_rec").item()
        self.rec_out_weights = weights.get("rec_out").item()
        self.rb_rec_weights = weights.get("rb_rec").item()

        self.n_rec = int(self.config["neurons"]["n_rec"])
        self.n_out = int(self.config["neurons"]["n_out"])
        self.step_ms = resolution
        self.num_centers = int(self.config["rbf"]["num_centers"])
        self.scale_rate = float(self.config["rbf"]["scale_rate"])

        self.nrns_rb = None
        self.nrns_rec = None
        self.nrns_out = None
        _log.debug("parameters loaded")

        self._create_network()
        _log.debug("network created")
        self._connect_network()
        _log.debug("network connected, EPROP initialization complete")

    def _validate_delay(self, expected_delay):
        """Validate that runtime M1 delay matches training configuration."""
        # Check if delay is specified in the config
        config_delay = self.config.get("m1_delay")

        if config_delay is None:
            _log.debug(
                f"WARNING: M1 config does not specify m1_delay. Runtime expects {expected_delay}ms delay."
            )
            _log.debug(
                f"         If this M1 was not trained with {expected_delay}ms delay, performance may be degraded."
            )
        else:
            # Compare delays with tolerance
            tolerance_ms = 0.1
            if abs(expected_delay - config_delay) > tolerance_ms:
                raise ValueError(
                    f"M1 delay mismatch: M1 was trained with delay={config_delay}ms "
                    f"but runtime config specifies m1_delay={expected_delay}ms. "
                    f"Please retrain M1 with matching delay or update configuration."
                )
            else:
                _log.debug(f"M1 delay validation passed: {expected_delay}ms")

    def _create_network(self):
        self.n_rb = self.num_centers
        self.nrns_rb = self.nest.Create("rb_neuron_nestml", self.n_rb)
        self.rec_rb = self.nest.Create(
            "spike_recorder", {"record_to": "ascii", "label": "mc_m1_rb"}
        )
        self.rec_out = self.nest.Create(
            "spike_recorder", {"record_to": "ascii", "label": "mc_m1_out"}
        )
        self.rec_rec = self.nest.Create(
            "spike_recorder", {"record_to": "ascii", "label": "mc_m1_recurrent"}
        )
        # params_rb_neuron = self.config["neurons"]["rb"].copy()
        params_rb_neuron = {
            **self.config["neurons"]["rb"].copy(),
            "sdev": 3600.0,
            "max_peak_rate": 800.0,
            "simulation_steps": self.sim_steps,
        }
        # params_rb_neuron["sdev"] = self.scale_rate * self.config["rbf"]["width"]
        # params_rb_neuron["max_peak_rate"] = self.scale_rate / self.step_ms
        self.nest.SetStatus(self.nrns_rb, params_rb_neuron)
        desired_rates = np.linspace(
            self.config["rbf"]["shift_min_rate"],
            float(self.config["rbf"]["desired_upper_hz"]),
            self.n_rb,
        )
        _log.debug("desired rates:")
        _log.debug(desired_rates)
        for i, nrn in enumerate(self.nrns_rb):
            self.nest.SetStatus(nrn, {"desired": desired_rates[i]})

        params_nrn_rec = self.config["neurons"]["rec"]

        self.nrns_rec = self.nest.Create(
            "eprop_iaf_bsshslm_2020", self.n_rec, params_nrn_rec
        )
        # self.nrns_out = self.nest.Create(
        #     "iaf_cond_alpha",
        #     self.n_out,
        # )
        self.nrns_out = self.nest.Create(
            "basic_neuron_nestml",
            self.n_out,
        )
        self.nest.SetStatus(self.nrns_out, {"simulation_steps": self.sim_steps})

    def _connect_network(self):
        params_conn_one_to_one = {"rule": "one_to_one", "allow_autapses": False}

        self.nest.Connect(self.nrns_rb, self.rec_rb)
        self.nest.Connect(self.nrns_out, self.rec_out)
        self.nest.Connect(self.nrns_rec, self.rec_rec)

        nrns_rb_ids = self.rb_rec_weights["source"] + min(self.nrns_rb.tolist())
        nrns_rec_ids = self.rb_rec_weights["target"] + min(self.nrns_rec.tolist())

        self.nest.Connect(
            nrns_rb_ids,
            nrns_rec_ids,
            params_conn_one_to_one,
            {
                "synapse_model": "static_synapse",
                "weight": self.rb_rec_weights["weight"],
                "delay": [self.step_ms] * len(self.rb_rec_weights["weight"]),
            },
        )

        rec_source_ids = self.rec_rec_weights["source"] + min(self.nrns_rec.tolist())
        rec_target_ids = self.rec_rec_weights["target"] + min(self.nrns_rec.tolist())
        self.nest.Connect(
            rec_source_ids,
            rec_target_ids,
            params_conn_one_to_one,
            syn_spec={
                "synapse_model": "static_synapse",
                "weight": self.rec_rec_weights["weight"],
                "delay": [self.step_ms] * len(self.rec_rec_weights["weight"]),
            },
        )

        nrns_rec_ids = self.rec_out_weights["source"] + min(self.nrns_rec.tolist())
        nrns_out_ids = self.rec_out_weights["target"] + min(self.nrns_out.tolist())
        _log.debug(f"connecting these out ids: {nrns_out_ids}")
        self.nest.Connect(
            nrns_rec_ids,
            nrns_out_ids,
            params_conn_one_to_one,
            syn_spec={
                "synapse_model": "static_synapse",
                "weight": self.rec_out_weights["weight"],
                "delay": [self.step_ms] * len(self.rec_out_weights["weight"]),
            },
        )

    def connect(self, source):
        params_conn_all_to_all = {"rule": "all_to_all", "allow_autapses": False}
        params_syn_static = {
            "synapse_model": "static_synapse",
            "weight": 1.0,
            "delay": self.step_ms,
        }

        self.nest.Connect(
            source, self.nrns_rb, params_conn_all_to_all, params_syn_static
        )
        return

    def get_output_pops(self):
        return (self.nrns_out[0], self.nrns_out[1])

    def get_rb_neurons(self):
        return self.nrns_rb

    def get_recurrent_neurons(self):
        return self.nrns_rec
