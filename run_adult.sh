#!/bin/bash
set -e
set -x

python train_tabnet.py \
       --csv-path data/adult.csv \
       --target-name "<=50K" \
       --categorical-features workclass,education,marital.status,occupation,relationship,race,sex,native.country \
       --feature_dim 16 \
       --output_dim 16 \
       --batch-size 4096 \
       --virtual-batch-size 128 \
       --batch-momentum 0.98 \
       --gamma 1.5 \
       --n_steps 5 \
       --decay-every 2500 \
       --lambda-sparsity 0.0001 \
       --max-steps 7700

