#!/bin/bash
cd /home/ekirby/workspace/DiT-3D

python test_w_param.py -c config/ablation_1/Xs4_2a_cross_voxel.yaml -w checkpoints/xs_4_2a_cross_voxel/last.ckpt
python test_w_param.py -c config/ablation_1/Xs4_2a_cross_point.yaml -w checkpoints/xs_4_2a_cross_point/last.ckpt
