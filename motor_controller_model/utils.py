import numpy as np

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
    with open(file_path, 'r') as f:
        lines = [line for line in f if line.strip() and not line.startswith('#')]
    data = []
    for line in lines:
        parts = line.replace(',', ' ').split()
        if len(parts) >= 2:
            try:
                data.append([float(parts[0]), float(parts[1])])
            except Exception:
                continue
    return np.array(data)
