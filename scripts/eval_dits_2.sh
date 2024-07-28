#!/bin/bash

cd /home/ekirby/workspace/DiT-3D/metrics/PyTorchEMD
pip install .

cd /home/ekirby/workspace/DiT-3D

python train_dit3d.py -c config/ablation_3/xs_4_1a_pixart_modadaln_pnet_impcfg_4ch.yaml -w checkpoints/xs_4_1a_pixart_pointnet_modadaln_impcfg_4ch_2/last.ckpt --test
python train_dit3d.py -c config/ablation_3/xs_4_1a_just_cross_4ch.yaml  -w checkpoints/xs_4_1a_just_cross_4ch/last.ckpt --test
