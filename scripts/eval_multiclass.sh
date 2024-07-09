#!/bin/bash

cd /home/ekirby/workspace/DiT-3D/metrics/PyTorchEMD
pip install .

cd /home/ekirby/workspace/DiT-3D
config=config/config_multiclass_s_4_cross_voxel_max.yaml
weights=checkpoints/multiclass_xs_4_cross_voxel_max_2a_1/last.ckpt

python train_dit3d.py -c $config -w $weights --test