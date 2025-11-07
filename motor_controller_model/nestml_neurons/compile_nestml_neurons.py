
# Compile and install NESTML neuron models for motor controller simulations.
#
# Usage:
#     python -m motor_controller_model.nestml_neurons.compile_nestml_neurons
#
# This script generates NEST target code from NESTML files and installs the neuron module for use in simulations.

from pynestml.frontend.pynestml_frontend import generate_nest_target
from pathlib import Path
import nest


def compile_nestml_neurons():
    """
    Compile and install the NESTML motor neuron module.
    
    This function generates NEST target code from NESTML files and installs
    the motor_neuron_module for use in simulations.
    """
    nestml_file_path = Path(__file__).resolve().parent
    nestml_target_path = nestml_file_path / "nestml_target"
    nestml_install_path = nestml_file_path / "nestml_install"

    print(f"Generating NEST target code from {nestml_file_path} to {nestml_target_path}")

    generate_nest_target(
        input_path=str(nestml_file_path),
        target_path=str(nestml_target_path),
        install_path=str(nestml_install_path),
        module_name="motor_neuron_module",
    )

    nest.ResetKernel()
    try:
        nest.Install("motor_neuron_module")
        print("NESTML target code generated and installed successfully.")
    except nest.NESTError:
        raise RuntimeError("Failed to install compiled NESTML module")


if __name__ == "__main__":
    compile_nestml_neurons()
