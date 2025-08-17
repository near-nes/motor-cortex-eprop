import numpy as np
import yaml
from interfaces.m1_base import M1SubModule


class M1MotorCortexEprop(M1SubModule):
    def __init__(self, config_path, weights_path, sim_steps, nest_instance):
        self.nest = nest_instance
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        self.sim_steps = sim_steps
        weights = np.load(weights_path, allow_pickle=True)
        self.rec_rec_weights = weights.get("rec_rec").item()
        self.rec_out_weights = weights.get("rec_out").item()
        self.rb_rec_weights = weights.get("rb_rec").item()

        self.n_rec = int(self.config["neurons"]["n_rec"])
        self.n_out = int(self.config["neurons"]["n_out"])
        self.step_ms = 0.1
        self.num_centers = int(self.config["rbf"]["num_centers"])
        self.scale_rate = float(self.config["rbf"]["scale_rate"])

        self.nrns_rb = None
        self.nrns_rec = None
        self.nrns_out = None

        self._create_network()
        self._connect_network()

    def _create_network(self):
        self.n_rb = self.num_centers
        self.nrns_rb = self.nest.Create("rb_neuron_nestml", self.n_rb)
        self.rec_rb = self.nest.Create(
            "spike_recorder", {"record_to": "ascii", "label": "mc_m1_rb"}
        )
        self.rec_out = self.nest.Create(
            "spike_recorder", {"record_to": "ascii", "label": "mc_m1_out"}
        )
        params_rb_neuron = self.config["neurons"]["rb"].copy()
        params_rb_neuron["sdev"] = 200
        params_rb_neuron["max_peak_rate"] = 2800
        # params_rb_neuron["sdev"] = self.scale_rate * self.config["rbf"]["width"]
        # params_rb_neuron["max_peak_rate"] = self.scale_rate / self.step_ms
        params_rb_neuron["simulation_steps"] = self.sim_steps
        self.nest.SetStatus(self.nrns_rb, params_rb_neuron)

        angle_centers = np.linspace(0.0, np.pi, self.n_rb)
        desired_rates = angle_centers * self.scale_rate
        print("desired rates:")
        print(desired_rates)
        for i, nrn in enumerate(self.nrns_rb):
            self.nest.SetStatus(nrn, {"desired": desired_rates[i]})

        params_nrn_rec = self.config["neurons"]["rec"]

        self.nrns_rec = self.nest.Create(
            "eprop_iaf_bsshslm_2020", self.n_rec, params_nrn_rec
        )
        self.nrns_out = self.nest.Create("iaf_cond_alpha", self.n_out)

    def _connect_network(self):
        params_conn_one_to_one = {"rule": "one_to_one", "allow_autapses": False}

        self.nest.Connect(self.nrns_rb, self.rec_rb)
        self.nest.Connect(self.nrns_out, self.rec_out)

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
            "weight": 100.0,
            "delay": self.step_ms,
        }

        self.nest.Connect(
            source, self.nrns_rb, params_conn_all_to_all, params_syn_static
        )
        return

    def get_output_pops(self):
        return (self.nrns_out[0], self.nrns_out[1])
