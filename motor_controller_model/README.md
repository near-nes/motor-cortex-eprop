
# Motor Controller Model Package

This package contains code for simulating and analyzing motor control experiments using the e-prop (eligibility propagation) learning algorithm in spiking neural networks. It supports training and evaluation of recurrent spiking networks for reaching tasks, with flexible configuration, parameter sweeps, and automated result visualization.

All main code is now located in the `motor_controller_model/` package directory. For a high-level overview and environment setup, see the [main README](../README.md) in the repository root.

## Structure

- [`run_m1.py`](run_m1.py): Training entry point (CLI)
- [`m1_training.py`](m1_training.py): Training logic — builds network, runs e-prop learning, saves weights
- [`m1_network.py`](m1_network.py): NEST network construction
- [`m1_factory.py`](m1_factory.py): Factory for building/caching trained M1 instances
- [`signals.py`](signals.py): Signal generation and encoding utilities
- [`config_schema.py`](config_schema.py): Pydantic config schema (loaded from `config/config.yaml`)
- [`trained_weights_net.py`](trained_weights_net.py): Post-training inference/verification
- [`utils.py`](utils.py): Shared utilities
- [`nestml_neurons/`](nestml_neurons/): Custom NESTML neuron models and compilation script

**Note:** Simulation results are saved in the `sim_results/` directory at the repository root, not within this package directory.

## Usage

### Prerequisites

Ensure you have set up your environment following one of the methods in the [main README](../README.md)
After environment setup, the package should be installed with `pip install -e .` from the repository root.

### Running Experiments

Compile NESTML neurons and train the network:
```bash
python -m motor_controller_model.nestml_neurons.compile_nestml_neurons
python -m motor_controller_model.run_m1 --force-retrain \
    --nest-module "./motor_controller_model/nestml_neurons/nestml_install/motor_neuron_module.so"
```



## Key Configuration Options

- `task.learning_window_ms`: Duration (ms) of the learning window, anchored to the **end** of each update interval. NEST zeros error/target/readout signals before this window.
- `task.input_shift_ms`: Temporal delay (ms) to shift planner input backwards, allowing M1 time to compute its output.

## Results

Simulation results and plots are saved in the `sim_results/` directory at the repository root, organized by experiment configuration. Each run creates a subfolder with files such as:

- `training_error.png`: The training loss curve.
- `spikes_and_dynamics.png`: Visualization of network activity.
- `weight_matrices.png` & `weight_time_courses.png`: Weight visualizations.
- `results.npz`: Raw results data.


## Notes

- All outputs are saved by default in `results/` at the repository root.
- For environment setup and installation instructions, see the [main README](../README.md).
- Legacy scripts and the monolithic training script (`eprop_reaching_task.py`) have been moved to `outdated/` in the repository root.


## License
<Specify your license here>