#!/bin/bash

export CUDA_VISIBLE_DEVICES=2
basedir="pruned_transducer_stateless7_category_0623"
task="pr10_wt_category50_wt_weight0.1_lr0.035"
base_expdir="experiments"
expdir="${base_expdir}/${task}"
[ -e ${basedir}/${task} ] && rm -rf ${basedir}/${task}
${basedir}/train.py \
    --world-size 1 \
    --master-port 10145 \
    --num-epochs 30 \
    --start-epoch 1 \
    --exp-dir ${expdir} \
    --max-duration 200 \
    --prune-range 10  \
    --num-category 50 \
    --category-alpha 0.1 \
    --base-lr 0.035 \

