#!/bin/bash

# Run default eprop_reaching_task.py
# python eprop_reaching_task.py

# wait

# python eprop_reaching_task.py --use-manual-rbf

# wait

# Run eprop_reaching_task.py with scan parameters and two sets of trajectory files
# python eprop_reaching_task.py \
#   --trajectory-files ../dataset_motor_training/stage1/trajectories_90_to_140.txt,../dataset_motor_training/stage1/trajectories_90_to_20.txt \
#   --target-files ../dataset_motor_training/stage1/spikes_from_90_to_140.txt,../dataset_motor_training/stage1/spikes_from_90_to_20.txt \
#   --scan-param learning_rate,neurons.n_rec,task.n_iter,rbf.num_centers,synapses.w_input,synapses.w_rec,rbf.width \
#   --scan-values "0.005;300;1000;20;200;60;0.2" \

#   wait

# python eprop_reaching_task.py \
# --trajectory-files ../dataset_motor_training/stage1/trajectories_90_to_140.txt,../dataset_motor_training/stage1/trajectories_90_to_20.txt \
# --target-files ../dataset_motor_training/stage1/spikes_from_90_to_140.txt,../dataset_motor_training/stage1/spikes_from_90_to_20.txt \
# --scan-param learning_rate,neurons.n_rec,task.n_iter,rbf.num_centers,synapses.w_input,synapses.w_rec,rbf.width \
# --scan-values "0.03;300;500;20;100;40;0.1" \
# --use-manual-rbf \
# --plastic-input-to-rec \

# wait

python -m motor_controller_model.eprop_reaching_task \
--trajectory-files ../dataset_motor_training/stage1/trajectories_90_to_140.txt,../dataset_motor_training/stage1/trajectories_90_to_20.txt \
--target-files ../dataset_motor_training/stage1/spikes_from_90_to_140.txt,../dataset_motor_training/stage1/spikes_from_90_to_20.txt \
--scan-param learning_rate,neurons.n_rec,task.n_iter,rbf.num_centers,synapses.w_input,synapses.w_rec,rbf.width \
--scan-values "0.002;200;500;15;200;40;0.25" \
# --plastic-input-to-rec \
