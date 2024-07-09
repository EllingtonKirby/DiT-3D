#!/bin/bash
cd /home/ekirby/workspace/DiT-3D

python test_w_param.py -c config/ablation_1/Xs4_1a_self_point.yaml -w checkpoints/xs_4_1a_self_point/last.ckpt
python test_w_param.py -c config/ablation_1/Xs4_1a_self_voxel.yaml -w checkpoints/xs_4_1a_self_voxel/last.ckpt