#!/bin/bash
cd /home/ekirby/workspace/DiT-3D

config=config/config_bikes_s_4_cross_voxelized.yaml
weights=checkpoints/bikes_s_4_cross_voxel_pad_1/last.ckpt

python object_gen_eval.py -c $config -w $weights -n bikes_s_4_cross_voxel_pad_eval_1 -e 10 -m 50 -t recreate

python object_gen_eval.py -c $config -w $weights -n bikes_s_4_cross_voxel_pad_eval_2 -e 10 -m 50 -t interpolate

config=config/config_bikes_s_4_cross_voxel_sorted.yaml
weights=checkpoints/bikes_s_4_2_cross_voxel_sorted_2/last.ckpt

python object_gen_eval.py -c $config -w $weights -n bikes_s_4_cross_voxel_sorted_eval_1 -e 10 -m 50 -t recreate

python object_gen_eval.py -c $config -w $weights -n bikes_s_4_cross_voxel_sorted_eval_2 -e 10 -m 50 -t interpolate