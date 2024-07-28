# !/bin/bash
# cd /home/ekirby/workspace/DiT-3D/metrics/PyTorchEMD
# pip install .

cd /home/ekirby/workspace/DiT-3D

config=config/ablation_3/xs_4_1a_pixart_modadaln_pnet_impcfg_4ch.yaml
weights=checkpoints/xs_4_1a_pixart_pointnet_modadaln_impcfg_4ch_2/last.ckpt
# python object_gen_eval_impcfg.py -c $config -w $weights -n xs_4_1a_cross_pnet_impcfg_4ch_eval_14_f -e 1 -t recreate -cls None -ds 1 -ind 14
python object_gen_eval_impcfg.py -c $config -w $weights -n xs_4_1a_pixart_pnet_2_impcfg_4ch_eval_14_a -e 1 -t interpolate -cls None -ind 14

# config=config/ablation_3/xs_4_1a_just_cross_4ch.yaml
# weights=checkpoints/xs_4_1a_just_cross_4ch/last.ckpt
# python object_gen_eval_impcfg.py -c $config -w $weights -n xs_4_1a_just_cross_eval_14_a -e 1 -t interpolate -cls None -ind 14

# config=config/ablation_3/xs_4_1a_pixart_pnet_impcfg.yaml
# weights=checkpoints/xs_4_1a_pixart_pointnet_dimsinadaln_impcfg/last.ckpt
# python object_gen_eval_impcfg.py -c $config -w $weights -n xs_4_1a_pixart_pnet_impcfg_eval_14_a -e 1 -t interpolate -cls None -ind 14

# config=config/config_bikes_s_4_cross_voxel_pretrained.yaml
# weights=checkpoints/bikes_s_4_cross_voxel_sorted_pretrained_2/last.ckpt
# python object_gen_eval.py -c $config -w $weights -n xs_4_2a_cross_voxel_pretrained_eval_14 -e 1 -t recreate -cls None -ds 10 -ind 14
# python object_gen_eval.py -c $config -w $weights -n xs_4_2a_cross_voxel_pretrained_eval_14_b -e 1 -t interpolate -cls None -ind 14

# config=config/config_multiclass_s_4_cross_voxel_max.yaml
# weights=checkpoints/multiclass_xs_4_cross_voxel_max_2a_1/last.ckpt
# python object_gen_eval.py -c $config -w $weights -n multiclass_xs_4_2a_cross_voxel_eval_14 -e 1 -t recreate -cls vehicle.bicycle -ds 10 -ind 14
# python object_gen_eval.py -c $config -w $weights -n multiclass_xs_4_2a_cross_voxel_eval_14_b -e 1 -t interpolate -cls vehicle.bicycle -ind 14
