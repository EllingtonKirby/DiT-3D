#!/bin/bash
cd /home/ekirby/workspace/DiT-3D
python object_gen_eval.py -c config/config_bikes_s_4.yaml -w checkpoints/bikes_s_4_2/last.ckpt -n bikes_s_4_2_eval_2 -e 10 -m 50 -t recreate

python object_gen_eval.py -c config/config_bikes_s_4.yaml -w checkpoints/bikes_s_4_2/last.ckpt -n bikes_s_4_2_eval_3 -e 10 -m 50 -t interpolate