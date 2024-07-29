#!/bin/bash

cd /home/ekirby/workspace/DiT-3D/metrics/PyTorchEMD
pip install .

cd /home/ekirby/workspace/DiT-3D

python train_dit3d.py -c config/ablation_4/xs_4_1a_cross_pointnet_noff.yaml -w checkpoints/xs_4_1a_cross_pointnet_impcgf_4ch_noff/last.ckpt --test
python train_dit3d.py -c config/ablation_4/xs_4_1a_cross_pointnet_reordered.yaml  -w checkpoints/xs_4_1a_cross_pointnet_impcgf_4ch_reordered/last.ckpt --test
