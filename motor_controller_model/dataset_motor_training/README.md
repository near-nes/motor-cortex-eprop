
# dataset_motor_training

This folder contains resources and scripts for preparing, analyzing, and visualizing motor training datasets used in neural network modeling and motor control experiments.

## Structure
- `sample_data/`: Example datasets, spike trains, trajectories, and a README for sample data details.
- `stage1/`: Processed spike and trajectory files for specific experimental stages.
- `input_ouput_data/`: Target spike files in NEST .dat format (e.g., N200_9020_mc_m1_p.dat, N200_9020_mc_m1_n.dat).
- `load_dataset.py`: Script for loading and preprocessing datasets.
- `plot_dataset.py`: Script for visualizing dataset contents.
- `utils.py`: Utility functions for data handling and processing.
- Jupyter notebooks for data checking, encoding, and exploration.

## Data Formats

### Trajectory Files
Trajectory files contain joint angle or position data over time, typically in plain text format with one value per line.

### Target Spike Files
The training code supports multiple spike file formats:

1. **NEST .dat format** (e.g., `input_ouput_data/N200_9020_mc_m1_p.dat`):
   - Tab-separated with headers: `sender` and `time_ms`
   - Separate files for positive and negative populations
   - Auto-detected by header content

2. **Comma-separated format** (e.g., `stage1/spikes_from_90_to_20.txt`):
   - Format: `neuron_id,spike_time`
   - Can contain both populations in single file (split by neuron_id threshold)

3. **Whitespace-separated format**:
   - Format: `neuron_id spike_time`
   - Similar to comma-separated but with spaces/tabs

The format is automatically detected when loading files. See the main package README for usage examples.

### Paired input-output (planner → M1) spike files

The code supports a paired input-output format used for spike-input experiments where planner spikes are provided separately from M1 target spikes. This is passed to the main script via the `--target-files` argument using an `@` separator between input and output parts, and a semicolon (`;`) to separate positive/negative files. Example:

```
planner_p.dat;planner_n.dat@mc_m1_p.dat;mc_m1_n.dat
```

Here, `planner_p.dat;planner_n.dat` are the planner (input) pos/neg files and `mc_m1_p.dat;mc_m1_n.dat` are the M1 (output/target) pos/neg files. The loader auto-detects NEST `.dat` headers, comma-separated, and whitespace-separated formats.

## Usage
Use the provided scripts and notebooks to:
- Load and preprocess motor training data
- Visualize and analyze dataset features
- Encode trajectories and spike trains for modeling

Refer to the `sample_data/README.md` for details about the example datasets.
