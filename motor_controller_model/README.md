
# Motor Controller Model Package

This package contains code for simulating and analyzing motor control experiments using the e-prop (eligibility propagation) learning algorithm in spiking neural networks. It supports training and evaluation of recurrent spiking networks for reaching tasks, with flexible configuration, parameter sweeps, and automated result visualization.

All main code is now located in the `motor_controller_model/` package directory. For a high-level overview and environment setup, see the [main README](../README.md) in the repository root.

## Structure

- [`eprop_reaching_task.py`](eprop_reaching_task.py): Main script for running the reaching task experiment with e-prop learning.
- [`plot_results.py`](plot_results.py): Script for visualizing experiment results (loss curves, weights, spikes).
- [`analyse_scan_results.py`](analyse_scan_results.py): Script for analyzing parameter scan results and generating heatmaps.
- [`trained_weights_net.py`](trained_weights_net.py): Script for loading, analyzing, or visualizing trained network weights.
- [`config/`](config/): Configuration files (e.g., [`config.yaml`](config/config.yaml)) for experiment parameters.
- [`dataset_motor_training/`](dataset_motor_training/): Trajectory data, spike datasets, and utilities. See its [README](dataset_motor_training/README.md) for details.
- [`nestml_neurons/`](nestml_neurons/): NESTML neuron model files and compilation scripts. See its [README](nestml_neurons/README.md) for details.
- [`testing_nestml_neurons/`](testing_nestml_neurons/): Scripts and notebooks for testing custom neuron models. See its [README](testing_nestml_neurons/README.md) for details.

**Note:** Simulation results are saved in the `sim_results/` directory at the repository root, not within this package directory.

## Usage

### Prerequisites

Ensure you have set up your environment following one of the methods in the [main README](../README.md):
- **Option A:** conda/mamba environment (recommended) - See [`environment.yml`](../environment.yml)
- **Option B:** pyenv/venv with manual dependencies - See [`pyproject.toml`](../pyproject.toml) for Python packages

After environment setup, the package should be installed with `pip install -e .` from the repository root.

### Running Experiments
To run a default experiment with the standard configuration:

```bash
python -m motor_controller_model.eprop_reaching_task
```

This uses parameters from [`config/config.yaml`](config/config.yaml) and saves results into a new subfolder under `sim_results/` at the repository root.

### Analysis and Visualization
To analyze results or visualize outputs:

```bash
python -m motor_controller_model.plot_results
python -m motor_controller_model.analyse_scan_results
```

### Trained Weights Analysis
After running experiments, analyze trained network weights:

```bash
python -m motor_controller_model.trained_weights_net
```

### NESTML Neuron Compilation & Testing
To compile and install custom neuron models (NESTML):

```bash
python -m motor_controller_model.nestml_neurons.compile_nestml_neurons
```

Or use the provided Jupyter notebooks in [`nestml_neurons/`](nestml_neurons/) for interactive compilation, installation, and testing. See the [nestml_neurons README](nestml_neurons/README.md) for more details.

### Command-Line Options
You can pass options to the main experiment script, e.g.:

- Set both excitatory and inhibitory learning rates:
  ```bash
  python -m motor_controller_model.eprop_reaching_task --learning-rate 0.001
  ```
- Use manual RBF implementation:
  ```bash
  python -m motor_controller_model.eprop_reaching_task --use-manual-rbf
  ```
- Make input-to-recurrent connections plastic:
  ```bash
  python -m motor_controller_model.eprop_reaching_task --plastic-input-to-rec
  ```

- Use custom target spike files (supports NEST .dat format with paired pos/neg files):
  ```bash
  # Paired files (new NEST .dat format):
  python -m motor_controller_model.eprop_reaching_task \
      --target-files "path/N200_9020_mc_m1_p.dat;path/N200_9020_mc_m1_n.dat"
  
  # Paired input-output spike data (planner → M1, requires rb_neuron mode):
  python -m motor_controller_model.eprop_reaching_task \
      --target-files "path/N200_9020_planner_p.dat;path/N200_9020_planner_n.dat@path/N200_9020_mc_m1_p.dat;path/N200_9020_mc_m1_n.dat"
  
  # Single file (old format, backward compatible):
  python -m motor_controller_model.eprop_reaching_task \
      --target-files "path/spikes_from_90_to_20.txt"
  ```
  *Note: Paired files are separated by semicolon (`;`), input-output pairing uses `@` separator. 
  Spike input mode (with `@`) creates parrot neurons that feed the rb_neuron layer.
  The code auto-detects NEST .dat format, comma-separated, or whitespace-separated files.*

### Parameter Sweep Examples

- **Scan a single parameter:**
  ```bash
  python -m motor_controller_model.eprop_reaching_task --scan-param neurons.n_rec --scan-values 100,200,300
  ```

- **Scan multiple parameters (grid search):**
  ```bash
  python -m motor_controller_model.eprop_reaching_task --scan-param learning_rate_exc,rbf.num_centers --scan-values "0.01,0.001;10,20"
  ```
  *Note: Use quotes around `--scan-values` to avoid shell parsing issues with semicolons.*

- **Disable plotting:**
  ```bash
  python -m motor_controller_model.eprop_reaching_task --no-plot
  ```


## Configuration

Experiment parameters are set in [`config/config.yaml`](config/config.yaml). You can adjust simulation, task, RBF encoding, and neuron parameters there. These can be overridden at runtime with command-line arguments (see examples above).

## New / Important Options

- `task.learning_start` and `task.learning_end` (ms): Specify an explicit learning window inside each sequence. When both are provided, the learning window used by NEST's e-prop kernel is `learning_end - learning_start` (clamped to [0, sequence]). Use this to exclude the preparation period (TIME_PREP) from eligibility accumulation and weight updates.
- `input_shift_ms`: Aligns teacher/target signals with network processing delays. For trajectory (rate) input mode the input trajectories are shifted; for spike-input (paired planner→M1) mode the processed target signals are shifted forward by zero-padding so alignment is equivalent across input modes.
- Diagnostic outputs (spike-input mode): When running with spike-input data (`--target-files` using the `@` pairing) and plotting enabled, the simulation now saves quick diagnostic figures in the run `result_dir`:
  - `diag_target_vs_readout_seq0.png`: target vs readout for the first sequence
  - `diag_spike_raster_seq0.png`: spike raster for the first sequence

Note: NEST interprets `eprop_learning_window` relative to the update times defined by `eprop_update_interval` (set from the sequence length + silent period). If you want the learning window to cover an absolute interval earlier in the sequence, either align `learning_end` with the sequence end (easy option) or schedule explicit update triggers at the desired times.

## Results

Simulation results and plots are saved in the `sim_results/` directory at the repository root, organized by experiment configuration. Each run creates a subfolder with files such as:

- `training_error.png`: The training loss curve.
- `spikes_and_dynamics.png`: Visualization of network activity.
- `weight_matrices.png` & `weight_time_courses.png`: Weight visualizations.
- `results.npz`: Raw results data.

To aggregate and compare results across runs, use:
```bash
python -m motor_controller_model.plot_results
```

## Notes

- All outputs are saved by default in `sim_results/` at the repository root.
- For environment setup and installation instructions, see the [main README](../README.md).
- The script [`eprop_network_dale-law-not-applied.py`](eprop_network_dale-law-not-applied.py) contains an older version of the network implementation without Dale's law and is included for reference only.


## License
<Specify your license here>