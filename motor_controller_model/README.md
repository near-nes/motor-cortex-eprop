
# Motor Controller Model Package

This package contains code for simulating and analyzing motor control experiments using the e-prop (eligibility propagation) learning algorithm in spiking neural networks. It supports training and evaluation of recurrent spiking networks for reaching tasks, with flexible configuration, parameter sweeps, and automated result visualization.

All main code is now located in the `motor_controller_model/` package directory. For a high-level overview, see the main `README.md` in the repository root.

## Structure

- `eprop_reaching_task.py`: Main script for running the reaching task experiment with e-prop learning.
- `plot_results.py`: Script for visualizing experiment results (loss curves, weights, spikes).
- `analyse_scan_results.py`: Script for analyzing parameter scan results and generating heatmaps.
- `config/`: Configuration files (e.g., `config.yaml`) for experiment parameters.
- `sim_results/`: Stores simulation results, including plots and data. Subfolders are created for each run.
- `dataset_motor_training/`, `nestml_neurons/`, `testing_nestml_neurons/`: See their respective README files for details.
- `trained_weights_net.py`: Script for loading, analyzing, or visualizing trained network weights.

## Usage

### Environment Setup
Ensure you have installed all dependencies as described in the main `README.md` and `pyproject.toml`. Use the provided environment files if needed.

### Running Experiments
To run a default experiment with the standard configuration:

```bash
python -m motor_controller_model.eprop_reaching_task
```

This uses parameters from `config/config.yaml` and saves results into a new subfolder under `sim_results/` at the repository root.

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

Or use the provided Jupyter notebooks in `nestml_neurons/` for interactive compilation, installation, and testing.

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

## Notes
- All outputs are saved by default in `sim_results/` at the repository root.
- Refer to the main `README.md` for environment setup and general usage.




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

Experiment parameters are set in `motor_controller_model/config/config.yaml`. You can adjust simulation, task, RBF encoding, and neuron parameters there. These can be overridden at runtime with command-line arguments.


## Environment Setup

The recommended environment includes NEST, NESTML, and build tools. See `motor_controller_model/environment.yml` for details. Key dependencies:

- nest-simulator
- nestml
- numpy, pandas, matplotlib, h5py, statsmodels
- cmake, make, boost, gsl

Create and activate the environment:

```bash
mamba env create -f motor_controller_model/environment.yml
mamba activate motor-controller
```

This will install all necessary packages, including the NEST simulator and NESTML.


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


## Note on Dale's Law Version

The script `eprop_network_dale-law-not-applied.py` contains an older version of the network implementation without Dale's law and is included here for reference only.


## License
<Specify your license here>