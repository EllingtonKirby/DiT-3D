#!/bin/bash
cd /home/ekirby/workspace/DiT-3D

# python train_dit3d.py -c config/config_bikes_s_4.yaml -w checkpoints/bikes_s_4_2/last.ckpt --test
# python train_dit3d.py -c config/config_bikes_s_4_cross.yaml -w checkpoints/bikes_s_4_cross_1/last.ckpt --test
# python train_dit3d.py -c config/config_bikes_s_4_flash.yaml -w checkpoints/bikes_s_4_flash_1/last.ckpt --test
# python train_dit3d.py -c config/config_bikes_s_4_padded.yaml -w checkpoints/bikes_s_4_padded_2/last.ckpt --test
# python train_dit3d.py -c config/config_bikes_s_4_cross_voxelized.yaml -w checkpoints/bikes_s_4_cross_voxel_pad_2/last.ckpt --test
# python train_dit3d.py -c config/config_bikes_s_4_cross_voxel_sorted.yaml -w checkpoints/bikes_s_4_cross_voxel_sorted_3/last.ckpt --test
python train_dit3d.py -c config/config_bikes_s_4_cross_voxel_pretrained.yaml -w checkpoints/bikes_s_4_cross_voxel_sorted_pretrained_2/last.ckpt --test
python train_dit3d.py -c config/config_bikes_s_4_flash_sorted.yaml -w checkpoints/bikes_s_4_flash_sorted_1/last.ckpt --test