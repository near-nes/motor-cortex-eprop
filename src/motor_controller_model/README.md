
# Motor Controller Model Package

This package contains code for simulating and analyzing motor control experiments using the e-prop (eligibility propagation) learning algorithm in spiking neural networks. It supports training and evaluation of recurrent spiking networks for reaching tasks, with flexible configuration, parameter sweeps, and automated result visualization.

All main code is now located in the `motor_controller_model/` package directory. For a high-level overview and environment setup, see the [main README](../../README.md) in the repository root.

## Structure

- [`run_m1.py`](run_m1.py): Training entry point (CLI)
- [`m1_training.py`](m1_training.py): Training logic - builds network, runs e-prop learning, saves weights
- [`m1_network.py`](m1_network.py): NEST network construction
- [`m1_factory.py`](m1_factory.py): Factory for building/caching trained M1 instances
- [`signals.py`](signals.py): Signal generation and encoding utilities
- [`config_schema.py`](config_schema.py): Pydantic configuration schema and YAML loader
- [`run_m1.py`](run_m1.py): Standalone inference verification after training
- [`utils.py`](utils.py): Shared utilities
- [`nestml_neurons/`](nestml_neurons/): Custom NESTML neuron models and compilation script

**Note:** Simulation results are saved in the `results/` directory at the repository root, not within this package directory.

## Usage

### Prerequisites

Ensure you have set up your environment following one of the methods in the [main README](../../README.md)
After environment setup, the package should be installed with `pip install -e .` from the repository root.

### Running Experiments

Compile NESTML neurons and train the network:
```bash
python -m motor_controller_model.nestml_neurons.compile_nestml_neurons
python -m motor_controller_model.run_m1 --force-retrain \
    --nest-module "./src/motor_controller_model/nestml_neurons/nestml_install/motor_neuron_module.so"
```

Run from an experiment YAML (single general runner):
```bash
python -m motor_controller_model.run_m1 \
    --config experiments/legacy_sequence/legacy_like_1500_timephases.yaml \
    --output-dir results \
    --run-name legacy_sequence \
    --force-retrain \
    --nest-module "./src/motor_controller_model/nestml_neurons/nestml_install/motor_neuron_module.so"
```

Notes:
- `--config` is optional; if omitted, built-in defaults are used (backward compatible).
- `--run-name` is optional; if set, artifacts are saved under `--output-dir/<run-name>`.

### Parameter Sweeps

Use the sweep runner for config-driven sweeps with one run directory per trial:
```bash
python -m motor_controller_model.sweep \
    --spec experiments/legacy_sequence/legacy_like_1500_timephases_sweep.yaml
```

For local testing, you can run sweep + best-run analysis in one command:
```bash
python -m motor_controller_model.sweep \
    --spec experiments/epropplus_corrected_params/sweep.yaml \
    --analyze \
    --promote
```

The sweep runner writes a timestamped root directory containing per-run configs, a JSONL manifest, and a summary file. For Slurm, point each array task at the same spec and pass `--task-index $SLURM_ARRAY_TASK_ID` so local testing and cluster execution stay identical.

After a sweep finishes, you can rank runs and pick the best one. By default, ranking is training-only and uses `training_success_score` (with fallback to `final_training_loss` for older sweeps):
```bash
python -m motor_controller_model.analyze_sweep \
    --spec experiments/epropplus_corrected_params/sweep.yaml \
    --latest
```

This prints the best run and top-k ranking, and writes `best_run_report.json` in the selected sweep directory.

To force pure final-loss ranking explicitly:
```bash
python -m motor_controller_model.analyze_sweep \
    --spec experiments/epropplus_corrected_params/sweep.yaml \
    --latest \
    --metric final_training_loss
```

To also copy the best run to a stable location:
```bash
python -m motor_controller_model.analyze_sweep \
    --spec experiments/epropplus_corrected_params/sweep.yaml \
    --latest \
    --promote
```

By default, the best run is copied to `<sweep_root>/best/`. You can override this with `--promote-dir`.

### Network and learning-signal path

Training creates a planner population, an RBF population, an excitatory recurrent
population, an inhibitory recurrent population, and positive and negative readout
populations. The planner is connected to the RBF population, and RBF-to-recurrent
connections use `eprop_synapse_m1_exc` and target all recurrent neurons. Recurrent
excitatory projections also use `eprop_synapse_m1_exc`. Inhibitory recurrent
projections use `eprop_synapse_m1_inh` when `synapses.inh.plastic: true`, which is
enabled by the experiment configurations; otherwise they use `static_synapse`.

The readout projection originates from excitatory recurrent neurons and uses
`eprop_synapse_m1_exc`. The readout populations send the e-prop learning signal
through `eprop_learning_signal_connection` to all recurrent neurons, including
inhibitory neurons. Therefore, a synapse model being named `eprop_synapse` does not
by itself establish effective learning: the corresponding postsynaptic recurrent
neurons must also receive this feedback connection. The current training path
creates one positive planner population; the negative channel is represented in
the target/readout pair but its planner population is not connected.


## Key Configuration Options

- `task.learning_start_ms`: Absolute start time (ms) for learning window within each sequence. NEST zeros error/target/readout signals before this time. E.g., with sequence=1500ms and learning_start_ms=800ms, learning is active from t=800 to t=1500.
- `task.input_shift_ms`: Temporal delay (ms) to shift planner input backwards, allowing M1 time to compute its output.

## Results

Simulation results and plots are saved in the `results/` directory at the repository root, organized by experiment configuration. Each run creates a subfolder with files such as:

- `training_error.png`: The training loss curve.
- `spikes_and_dynamics.png`: Visualization of network activity.
- `weight_matrices.png` & `weight_time_courses.png`: Weight visualizations.
- `trained_weights.npz`: Saved `rb_rec`, `rec_rec`, and `rec_out` weights.
- `training_loss.npy`: Per-sequence training loss.
- `standalone_inference_test.png`: Inference verification plot.


## Notes

- All outputs are saved by default in `results/` at the repository root.
- For environment setup and installation instructions, see the [main README](../../README.md).
- Legacy scripts and the monolithic training script (`eprop_reaching_task.py`) have been moved to `outdated/` in the repository root.


## License
No license file or license declaration is currently included in this repository.