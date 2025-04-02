#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1
basedir="pruned_transducer_stateless7"
task="revised_mc_sampling_loss_wt_hard_target_wt_simple_loss_wt_pr2_in_pruned_transducer7_lr0.050_wt_w0.01_wt_sam3"
base_expdir="experiments"
expdir="${base_expdir}/${task}"

${basedir}/train.py \
    --world-size 2 \
    --master-port 10101 \
    --num-epochs 30 \
    --start-epoch 1 \
    --exp-dir ${expdir} \
    --max-duration 200 \
    --base-lr 0.050 \
    --keep-last-k 6 \
    --use-aligner-encoder False \
    --use-monte-carlo-ce False \
    --use-soft-target False \
    --mono-alignment-policy "simple_loss" \
    --prune-range 2 \
    --use-mc-sampling True \
    --mc-sampling-num 3 \
    --mc-sampling-weight 0.01
