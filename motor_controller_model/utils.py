from pathlib import Path

import numpy as np
import structlog

_log = structlog.get_logger("m1_utils")


def install_nestml_module(module_name: str = "motor_neuron_module"):
    """Install the custom NESTML module containing M1 neuron models.

    Tries installing the pre-compiled module first. If that fails,
    compiles from source and retries.
    """
    import nest

    try:
        nest.Install(module_name)
        return
    except Exception:
        pass

    _log.debug("module not installed, compiling and installing NESTML neurons")
    nestml_install_dir = (
        Path(__file__).resolve().parent / "nestml_neurons" / "nestml_install"
    )
    module_path = nestml_install_dir / f"{module_name}.so"

    from .nestml_neurons.compile_nestml_neurons import compile_nestml_neurons

    compile_nestml_neurons()
    nest.Install(str(module_path))


def load_spike_data(file_path):
    """
    Load spike data from various file formats.
    Supports:
    1. NEST .dat format with headers (sender, time_ms columns)
    2. Comma-separated format (neuron_id,spike_time)
    3. Whitespace-separated format (neuron_id spike_time)
    Args:
        file_path (str): Path to the spike data file
    Returns:
        numpy.ndarray: 2D array with columns [neuron_id, spike_time]
    """
    with open(file_path, "r") as f:
        lines = [line for line in f if line.strip() and not line.startswith("#")]
    data = []
    for line in lines:
        parts = line.replace(",", " ").split()
        if len(parts) >= 2:
            try:
                data.append([float(parts[0]), float(parts[1])])
            except Exception:
                continue
    return np.array(data)
