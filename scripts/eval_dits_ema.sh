#!/bin/bash

cd /home/ekirby/workspace/DiT-3D/metrics/PyTorchEMD
pip install .

cd /home/ekirby/workspace/DiT-3D

python train_dit3d.py -c config/ablation_1/Xs4_1a_cross_point.yaml -w checkpoints/xs_4_1a_cross_point/last.ckpt --test
python train_dit3d.py -c config/ablation_1/Xs4_1a_cross_voxel.yaml -w checkpoints/xs_4_1a_cross_voxel/last.ckpt --test
python train_dit3d.py -c config/ablation_1/Xs4_1a_self_point.yaml  -w checkpoints/xs_4_1a_self_point/last.ckpt --test
# python train_dit3d.py -c config/ablation_1/Xs4_1a_self_voxel.yaml  -w checkpoints/xs_4_1a_self_voxel/last.ckpt --test
python train_dit3d.py -c config/ablation_1/Xs4_2a_cross_point.yaml -w checkpoints/xs_4_2a_cross_point/last.ckpt --test
python train_dit3d.py -c config/ablation_1/Xs4_2a_cross_voxel.yaml -w checkpoints/xs_4_2a_cross_voxel/last.ckpt --test
python train_dit3d.py -c config/ablation_1/Xs4_2a_self_point.yaml  -w checkpoints/xs_4_2a_self_point/last.ckpt --test
python train_dit3d.py -c config/ablation_1/Xs4_2a_self_voxel.yaml  -w checkpoints/xs_4_2a_self_voxel/last.ckpt --test