#!/bin/bash
cd /home/ekirby/workspace/DiT-3D

config=config/config_bikes_s_4_cross_voxel_pretrained.yaml
weights=checkpoints/bikes_s_4_cross_voxel_sorted_pretrained_2/last.ckpt

# python object_gen_eval.py -c $config -w $weights -n bikes_xs_4_cross_voxel_pretrained_eval_1 -e 5 -m 50 -t recreate
python object_gen_eval.py -c $config -w $weights -n bikes_xs_4_cross_voxel_pretrained_eval_uncond -e 2 -m 50 -t interpolate