#!/bin/bash

basedir="pruned_transducer_stateless7_garbage"
task="garbage"
[ -e "${basedir}/${task}" ] && rm -rf "${basedir}/${task}"
${basedir}/train.py \
  --world-size 1 \
  --num-epoch 1 \
  --exp-dir ${basedir}/${task} \
  --full-libri 0 \
  --max-duration 50 \
  --joiner-dim 128 \
  --decoder-dim 128 \
  --encoder-unmasked-dims "64,64,64,64,64" \
  --attention-dims "64,64,64,64,64" \
  --encoder-dims "128,128,128,128,128" \
  --nhead "2,2,2,2,2" \
  --feedforward-dims "128,128,128,128,128" \
  --num-encoder-layers "2,2,2,2,2" \
  --compress-time-axis ${1:-"True"}


