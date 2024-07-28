#!/bin/bash

cd /home/ekirby/workspace/DiT-3D/metrics/PyTorchEMD
pip install .

cd /home/ekirby/workspace/DiT-3D

config=config/ablation_3/xs_4_1a_cross_pointnet_impcfg_4ch_gen.yaml
weights=checkpoints/xs_4_1a_cross_pointnet_impcgf_4ch/last.ckpt
python gen_instance_pool.py -c $config -w $weights -n 1 -s train -r /home/ekirby/scania/ekirby/datasets/nuscenes_generated_bikes
python gen_instance_pool.py -c $config -w $weights -n 1 -s val -r /home/ekirby/scania/ekirby/datasets/nuscenes_generated_bikes