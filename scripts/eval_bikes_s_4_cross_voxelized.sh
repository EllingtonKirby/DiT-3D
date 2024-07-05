#!/bin/bash
cd /home/ekirby/workspace/DiT-3D

# config=config/config_bikes_s_4_cross_voxelized.yaml
# weights=checkpoints/bikes_s_4_cross_voxel_pad_1/last.ckpt

# python object_gen_eval.py -c $config -w $weights -n bikes_s_4_cross_voxel_pad_eval_1 -e 10 -m 50 -t recreate

# python object_gen_eval.py -c $config -w $weights -n bikes_s_4_cross_voxel_pad_eval_2 -e 10 -m 50 -t interpolate

config=config/config_bikes_s_4_padded.yaml
weights=checkpoints/bikes_s_4_padded_2/last.ckpt

python object_gen_eval.py -c $config -w $weights -n bikes_s_4_padded_2_eval_1 -e 5 -m 50 -t recreate
python object_gen_eval.py -c $config -w $weights -n bikes_s_4_padded_2_eval_2 -e 5 -m 50 -t interpolate