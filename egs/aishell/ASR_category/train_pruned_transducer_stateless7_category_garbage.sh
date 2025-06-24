#!/bin/bash

export CUDA_VISIBLE_DEVICES=5
basedir="pruned_transducer_stateless7_category_0611"
task="garbage"
base_expdir="experiments"
expdir="${base_expdir}/${task}"
#[ -e ${basedir}/${task} ] && rm -rf ${basedir}/${task}
${basedir}/train.py \
    --world-size 1 \
    --master-port 10145 \
    --num-epochs 30 \
    --start-epoch 1 \
    --exp-dir ${expdir} \
    --max-duration 100 \
    --prune-range 5  \
    --base-lr 0.035 \
    --num-category 3 \

