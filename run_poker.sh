#!/bin/bash
set -e
set -x

python train_tabnet.py \
       --csv-path data/poker_train.csv \
       --target-name CLASS \
       --task classification \
       --categorical-features S1,S2,S3,S4,S5 \
       --feature-dim 24 \
       --output-dim 8 \
       --lambda-sparsity 0.001 \
       --batch-size 512 \
       --virtual-batch-size 256 \
       --batch-momentum 0.8 \
       --n_steps 4 \
       --gamma 1.5 \
       --lr 0.02 \
       --decay-every 10000 \
       --max-steps 71000
