#!/bin/bash

export CUDA_VISIBLE_DEVICES=2
basedir="pruned_transducer_stateless7_pkd_sampling"
#task="6m_prune5_wt_gt_wt_kd_wt_kd_scale0.25_wt_sampling_wt_sam_scale1.0"
task="6m_prune5_wt_gt_wt_kd_wt_kd_scale0.1_wo_sampling"
#task="6m_prune5_wt_gt_wo_kd"
#[ -e ${basedir}/${task} ] && rm -rf ${basedir}/${task}
for b in False True
do
  ${basedir}/decode.py \
      --epoch 30 \
      --avg 1 \
      --use-averaged-model $b \
      --exp-dir ${PWD}/experiments/${task} \
      --max-duration 500 \
      --num-encoder-layers "1,1,1,1,1" \
      --feedforward-dims "256,256,512,512,256" \
      --nhead "4,4,4,4,4" \
      --encoder-dims "196,196,196,196,196" \
      --encoder-unmasked-dims "196,196,196,196,196" \
      --decoding-method modified_beam_search \
      --beam-size 4
done
