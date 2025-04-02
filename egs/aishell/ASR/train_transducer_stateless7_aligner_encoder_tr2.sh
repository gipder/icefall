#!/bin/bash

export CUDA_VISIBLE_DEVICES=2,3
basedir="transducer_stateless7"
task="aligner_encoder_baseline"
base_expdir="experiments"
expdir="${base_expdir}/${task}"

${basedir}/train.py \
    --world-size 2 \
    --master-port 10223 \
    --num-epochs 30 \
    --start-epoch 1 \
    --exp-dir ${expdir} \
    --max-duration 200 \
    --base-lr 0.035 \
    --keep-last-k 6 \
    --use-aligner-encoder True


