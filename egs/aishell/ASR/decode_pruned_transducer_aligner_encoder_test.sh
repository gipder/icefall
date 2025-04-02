#!/bin/bash

export CUDA_VISIBLE_DEVICES=${1:-"5"}
basedir="pruned_transducer_stateless7_tmp"
task="85m_prune5_wt_gt_wo_kd"
#[ -e ${basedir}/${task} ] && rm -rf ${basedir}/${task}
${basedir}/decode.py \
    --epoch 30 \
    --avg 1 \
    --use-averaged-model False \
    --exp-dir ${PWD}/experiments/${task} \
    --max-duration 400 \
    --decoding-method "_deprecated_modified_beam_search" \
    --beam-size 4 \
