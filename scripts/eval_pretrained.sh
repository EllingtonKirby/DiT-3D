#!/bin/bash
cd /home/ekirby/workspace/DiT-3D/metrics/PyTorchEMD
pip install .

cd /home/ekirby/workspace/DiT-3D

config=config/ablation_2/xs_4_1a_self_point_pretrained.yaml
weights=checkpoints/xs_4_1a_self_point_pretrained/last.ckpt

python train_dit3d.py -c $config -w $weights --test
# python test_w_param.py -c $config -w $weights
# python object_gen_eval.py -c $config -w $weights -n bikes_xs_4_cross_voxel_pretrained_eval_3 -e 5 -m 50 -t recreate -ds 10 -cls None
# python object_gen_eval.py -c $config -w $weights -n bikes_xs_4_cross_voxel_pretrained_eval_4 -e 5 -m 50 -t interpolate -cls None