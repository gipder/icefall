#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1
basedir="pruned_transducer_stateless7_vscode"
task="garbage"
base_expdir="experiments"
expdir="${base_expdir}/${task}"

${basedir}/train.py \
    --world-size 1 \
    --master-port 10101 \
    --num-epochs 31 \
    --start-epoch 31 \
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
    --mc-sampling-weight 1.00
