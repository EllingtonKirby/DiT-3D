#!/bin/bash

cd /home/ekirby/workspace/DiT-3D/metrics/PyTorchEMD
pip install .

cd /home/ekirby/workspace/DiT-3D

python train_dit3d.py -c config/ablation_2/b_4_1a_self_voxel.yaml