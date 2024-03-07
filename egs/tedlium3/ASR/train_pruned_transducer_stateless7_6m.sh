#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1
basedir="pruned_transducer_stateless7_pkd_sampling"
task="6m_pruned5_wt_gt_wo_kd"
#[ -e ${basedir}/${task} ] && rm -rf ${basedir}/${task}
${basedir}/train.py \
    --world-size 2 \
    --master-port 13546 \
    --num-epochs 30 \
    --start-epoch 1 \
    --exp-dir ${PWD}/experiments/${task} \
    --base-lr 0.035 \
    --max-duration 100 \
    --prune-range 5 \
    --keep-last-k 6 \
    --num-encoder-layers "1,1,1,1,1" \
    --feedforward-dims "256,256,512,512,256" \
    --nhead "4,4,4,4,4" \
    --encoder-dims "196,196,196,196,196" \
    --encoder-unmasked-dims "196,196,196,196,196" \
