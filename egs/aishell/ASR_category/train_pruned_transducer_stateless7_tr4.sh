#!/bin/bash

export CUDA_VISIBLE_DEVICES=3
basedir="pruned_transducer_stateless7"
task="pruned10_rnnt_lr0.035"
base_expdir="experiments"
expdir="${base_expdir}/${task}"

rm -rf ${expdir}

${basedir}/train.py \
    --world-size 1 \
    --master-port 10145 \
    --num-epochs 30 \
    --start-epoch 1 \
    --exp-dir ${expdir} \
    --max-duration 200 \
    --prune-range 10  \
    --base-lr 0.035 \

