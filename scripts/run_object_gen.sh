#!/bin/bash
cd /home/ekirby/workspace/DiT-3D/metrics/PyTorchEMD
pip install .

cd /home/ekirby/workspace/DiT-3D
config=config/ablation_1/Xs4_1a_self_voxel.yaml
weights=checkpoints/xs_4_1a_self_voxel/last.ckpt

# python object_gen_eval.py -c $config -w $weights -n mult_cars_eval_1 -e 5 -m 300 -t recreate -cls vehicle.car -ds 10
# python object_gen_eval.py -c $config -w $weights -n mult_cars_eval_2 -e 2 -m 300 -t interpolate -cls vehicle.car

python object_gen_eval.py -c $config -w $weights -n xs_4_1a_self_voxel_eval_1 -e 5 -m 50 -t recreate -cls None -ds 10
python object_gen_eval.py -c $config -w $weights -n xs_4_1a_self_voxel_eval_2 -e 2 -m 50 -t interpolate -cls None

python object_gen_eval.py -c $config -w $weights -n xs_4_1a_self_voxel_eval_3 -e 5 -m 50 -t recreate -cls None -ds 10 -s val
python object_gen_eval.py -c $config -w $weights -n xs_4_1a_self_voxel_eval_4 -e 2 -m 50 -t interpolate -cls None -s val

# python object_gen_eval.py -c $config -w $weights -n mult_motos_eval_1 -e 5 -m 100 -t recreate -cls vehicle.motorcycle -ds 10
# python object_gen_eval.py -c $config -w $weights -n mult_motos_eval_2 -e 2 -m 100 -t interpolate -cls vehicle.motorcycle

# python object_gen_eval.py -c $config -w $weights -n mult_peds_eval_1 -e 5 -m 50 -t recreate -cls human.pedestrian.adult -ds 10
# python object_gen_eval.py -c $config -w $weights -n mult_meds_eval_2 -e 2 -m 50 -t interpolate -cls human.pedestrian.adult