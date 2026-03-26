# Motor Controller Spiking Neural Network (SNN) Model with e-prop

This repository contains code, data, and neuron models for simulating and analyzing motor control experiments using spiking neural networks and the e-prop (eligibility propagation) learning algorithm. The network models motor cortex (M1) activity for a movement task, using e-prop learning as implemented in NEST.

<p align="center">
   <img src="overview_network.png" alt="Motor controller SNN schematic" width="800"/>
</p>

**Figure 1. Schematic of the spiking neural network (SNN) architecture for motor control.**  
On the left, labeled "input neurons," time-varying joint angles are received as external signals. These signals are projected into a central recurrent network (reservoir), which models the motor cortex (M1). Within the reservoir, black triangles represent individual excitatory spiking neurons, while the red circle at the bottom denotes the inhibitory neuron population—crucial for dynamically balancing network activity via recurrent inhibition.  
The recurrent network outputs to two readout neurons via e-prop synapses: a red circle ("pos" channel) and a blue circle ("neg" channel), each corresponding to a motor output direction. Gray arrows from these readout neurons point to target signals, illustrating supervised learning via error comparison. The purple arrow labeled "B" represents the feedback path by which the readout neurons send the e-prop learning signal (global error signal) back to the recurrent network, enabling biologically plausible online synaptic adaptation.  
Colors and shapes explicitly encode network roles: labeled input neurons (left), excitatory reservoir neurons (black triangles), inhibitory population (red circle), and output channels (red and blue circles).


## Repository Structure

- [`motor_controller_model/`](motor_controller_model/) — Main package containing all code for running motor control experiments, training spiking networks, analyzing results, and visualizing outputs. See its [README](motor_controller_model/README.md) for detailed usage and options.
- [`motor_controller_model/dataset_motor_training/`](motor_controller_model/dataset_motor_training/) — Contains trajectory data, spike datasets, and utilities for dataset handling. Includes a [README](motor_controller_model/dataset_motor_training/README.md) describing the dataset format.
- [`motor_controller_model/nestml_neurons/`](motor_controller_model/nestml_neurons/) — NESTML neuron model files and scripts for compiling custom neuron modules. See its [README](motor_controller_model/nestml_neurons/README.md) for details.
- `sim_results/` — Output directory for simulation results, plots, and data (created automatically).
- [`environment.yml`](environment.yml) — Conda/mamba environment specification with all required dependencies.
- [`pyproject.toml`](pyproject.toml) — Python package configuration and pip dependencies.

## Getting Started

### 1. Environment Setup

**Option A: Full conda/mamba environment (Recommended)**

This is the easiest approach as it handles all system-level and Python dependencies automatically.

```bash
# Create environment from specification
mamba env create -f environment.yml
mamba activate motor-controller

# Install the package in editable mode
pip install -e .
```

- Uses [`environment.yml`](environment.yml) to install all dependencies including `nest-simulator`
- **Important:** `nest-simulator` must be installed via mamba/conda, not pip (not available on PyPI)
- Key dependencies: nest-simulator, nestml, numpy, pandas, matplotlib, h5py, statsmodels, cmake, make, boost, gsl

**Option B: Manual environment (pyenv, venv, or system Python)**

If you prefer to manage Python versions with venv:

```bash
# Example with venv
python -m venv venv
source venv/bin/activate

# Install the package with Python dependencies
pip install -e .
```

- Python dependencies are listed in [`pyproject.toml`](pyproject.toml) and installed automatically
- **You must manually install** system-level dependencies:
  - `nest-simulator` (via conda, OS package manager, or [build from source](https://nest-simulator.readthedocs.io/))

### 2. Compile NESTML Neurons

```bash
python -m motor_controller_model.nestml_neurons.compile_nestml_neurons
```

### 3. Quick Start

Train the M1 network and run a standalone inference test:
```bash
python -m motor_controller_model.run_m1 --force-retrain \
    --nest-module "./motor_controller_model/nestml_neurons/nestml_install/motor_neuron_module.so"
```

When running inside the controller devcontainer, use the `custom_stdp_module` NEST module:
```bash
python -m motor_controller_model.run_m1 --output-dir /sim/controller/artifacts/m1/ --force-retrain --nest-module=custom_stdp_module
```

This trains the network (or loads cached weights if config matches), then runs an inference test and saves results and plots to the output directory.

**For detailed usage, command-line options, and parameter sweeps**, see the [motor_controller_model README](motor_controller_model/README.md).

### 4. Additional Resources

- **Network Architecture:** See [`overview_network.png`](overview_network.png) for a visual summary of the spiking neural network architecture (Figure 1 above)
- **Package Documentation:** See [`motor_controller_model/README.md`](motor_controller_model/README.md) for detailed API usage
- **Outdated files:** Legacy scripts, tutorials, and the monolithic training script are in [`outdated/`](outdated/).


## License
<Specify your license here>
