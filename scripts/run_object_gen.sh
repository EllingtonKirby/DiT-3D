#!/bin/bash
# cd /home/ekirby/workspace/DiT-3D/metrics/PyTorchEMD
# pip install .

cd /home/ekirby/workspace/DiT-3D

config=config/ablation_2/xs_4_1a_self_voxel_norm.yaml
weights=checkpoints/xs_4_1a_self_voxel_norm_2/last.ckpt
python object_gen_eval.py -c $config -w $weights -n xs_4_1a_self_voxel_normed_2_eval_14 -e 1 -t recreate -cls None -ds 10 -ind 14
python object_gen_eval.py -c $config -w $weights -n xs_4_1a_self_voxel_normed_2_eval_14_b -e 1 -t interpolate -cls None -ind 14

# config=config/ablation_1/Xs4_2a_self_point.yaml
# weights=checkpoints/xs_4_2a_self_point/last.ckpt
# python object_gen_eval.py -c $config -w $weights -n xs_4_2a_self_point_eval_14 -e 1 -t recreate -cls None -ds 10 -ind 14
# python object_gen_eval.py -c $config -w $weights -n xs_4_2a_self_point_eval_14_b -e 1 -t interpolate -cls None -ind 14

# config=config/ablation_1/Xs4_2a_cross_voxel.yaml
# weights=checkpoints/xs_4_2a_cross_voxel/last.ckpt
# python object_gen_eval.py -c $config -w $weights -n xs_4_2a_cross_voxel_eval_14 -e 1 -t recreate -cls None -ds 10 -ind 14
# python object_gen_eval.py -c $config -w $weights -n xs_4_2a_cross_voxel_eval_14_b -e 1 -t interpolate -cls None -ind 14

# config=config/config_bikes_s_4_cross_voxel_pretrained.yaml
# weights=checkpoints/bikes_s_4_cross_voxel_sorted_pretrained_2/last.ckpt
# python object_gen_eval.py -c $config -w $weights -n xs_4_2a_cross_voxel_pretrained_eval_14 -e 1 -t recreate -cls None -ds 10 -ind 14
# python object_gen_eval.py -c $config -w $weights -n xs_4_2a_cross_voxel_pretrained_eval_14_b -e 1 -t interpolate -cls None -ind 14

# config=config/config_multiclass_s_4_cross_voxel_max.yaml
# weights=checkpoints/multiclass_xs_4_cross_voxel_max_2a_1/last.ckpt
# python object_gen_eval.py -c $config -w $weights -n multiclass_xs_4_2a_cross_voxel_eval_14 -e 1 -t recreate -cls vehicle.bicycle -ds 10 -ind 14
# python object_gen_eval.py -c $config -w $weights -n multiclass_xs_4_2a_cross_voxel_eval_14_b -e 1 -t interpolate -cls vehicle.bicycle -ind 14
