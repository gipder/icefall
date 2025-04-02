#!/bin/bash

export CUDA_VISIBLE_DEVICES=1
basedir="pruned_transducer_stateless7_pkd_sampling"
#task="6m_prune5_wt_gt_wt_kd_wt_kd_scale0.01_wt_sampling_wt_sam_scale0.1"
#task="6m_prune5_wt_gt_wt_kd_wt_kd_scale0.01_wo_sampling"
task="teacher_model_large"
#[ -e ${basedir}/${task} ] && rm -rf ${basedir}/${task}
${basedir}/decode.py \
    --epoch 30 \
    --avg 1 \
    --use-averaged-model True \
    --exp-dir ${PWD}/experiments/${task} \
    --max-duration 400 \
    --decoding-method modified_beam_search \
    --beam-size 4 \
