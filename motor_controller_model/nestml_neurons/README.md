# nestml_neurons

This folder contains NESTML neuron model files and scripts for compiling custom neuron modules for use in motor control simulations.

## Structure
- `controller_module.nestml`: Main NESTML neuron model definition.
- `compile_nestml_neurons.py`: Script to compile and install neuron models using NESTML.
- `nestml_target/`: Generated C++ code and build artifacts from NESTML compilation.
- `report/`: Documentation and analysis of neuron model behavior (if present).

## Usage
To compile and install neuron models, run:

```bash
python -m motor_controller_model.nestml_neurons.compile_nestml_neurons
```

Or use the provided Jupyter notebooks for interactive compilation and testing.

Refer to the main `README.md` for environment setup and package usage.
