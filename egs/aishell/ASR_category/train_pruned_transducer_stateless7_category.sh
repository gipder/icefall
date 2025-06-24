#!/bin/bash

export CUDA_VISIBLE_DEVICES=4,5
basedir="pruned_transducer_stateless7_category"
task="category_second_trial_N2_lr0.035"
base_expdir="experiments"
expdir="${base_expdir}/${task}"
#[ -e ${basedir}/${task} ] && rm -rf ${basedir}/${task}
${basedir}/train.py \
    --world-size 2 \
    --master-port 10145 \
    --num-epochs 30 \
    --start-epoch 1 \
    --exp-dir ${expdir} \
    --max-duration 100 \
    --prune-range 5  \
    --base-lr 0.035 \
    --num-category 2 \

