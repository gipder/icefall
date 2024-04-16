#!/bin/bash

export CUDA_VISIBLE_DEVICES=2,3
basedir="pruned_transducer_stateless3"
task="large_model_pruned5_wt_gt_wo_kd_wt_stateless"
#[ -e ${basedir}/${task} ] && rm -rf ${basedir}/${task}
${basedir}/train.py \
    --world-size 2 \
    --master-port 23546 \
    --num-epochs 30 \
    --start-epoch 1 \
    --use-fp16 0 \
    --exp-dir ${PWD}/experiments/${task} \
    --max-duration 100 \
    --prune-range 5 \
    --datatang-prob 0.0