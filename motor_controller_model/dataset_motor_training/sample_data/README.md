This dataset contains 60 entries of initial and target end-effector positions couples and corresponding spikes produced as output by the motor cortex, as a sum of both the feedforward and feedback contributions. The recorded population is made up of a total of 100 neurons, equally split into two subgroups encoding positive and negative quantities, respectively. 
The dataset features ~10 examples for each of the 6 trajectories, and each line is formatted as follows:

[trajectory ID] [senders_pos] [times_pos] [senders_neg] [times_neg]
 
where senders contains the IDs of spiking neurons and times the instants at which spikes are generated.

Each line refers to a simulation starting with a 150 ms pause and preparation followed by 500 ms of movement execution.

Note: this sample dataset follows the repository convention of a short prep/silent period followed by the active sequence. When using these files as `--target-files` in spike-input mode, you can provide paired planner→M1 files using the `@` syntax documented in `motor_controller_model/dataset_motor_training/README.md`.