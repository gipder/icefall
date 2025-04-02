#!/bin/bash

export CUDA_VISIBLE_DEVICES=${1:-"0"}
basedir="pruned_transducer_stateless7"
task="revised_mc_sampling_loss_wt_hard_target_wt_simple_loss_wt_pr2_in_pruned_transducer7_lr0.050_wt_w0.01_wt_sam3"
#[ -e ${basedir}/${task} ] && rm -rf ${basedir}/${task}
${basedir}/decode.py \
    --epoch 30 \
    --avg 1 \
    --use-averaged-model False \
    --exp-dir ${PWD}/experiments/${task} \
    --max-duration 400 \
    --decoding-method modified_beam_search \
    --beam-size 4 \
