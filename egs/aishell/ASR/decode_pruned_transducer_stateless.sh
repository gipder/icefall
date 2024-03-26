#!/bin/bash

export CUDA_VISIBLE_DEVICES=3
basedir="pruned_transducer_stateless3"
task="large_model_pruned5_wt_gt_wo_kd_wt_stateless"
#[ -e ${basedir}/${task} ] && rm -rf ${basedir}/${task}
${basedir}/decode.py \
    --epoch 30 \
    --avg 1 \
    --exp-dir ${PWD}/experiments/${task} \
    --max-duration 100 \
    --decoding-method modified_beam_search \
    --beam-size 4 \
