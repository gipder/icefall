#!/bin/bash

basedir="transducer_stateless2_baseline"
task="baseline"
#[ -e ${basedir}/${task} ] && rm -rf ${basedir}/${task}
${basedir}/train.py \
    --world-size 2 \
    --master-port 13546 \
    --num-epochs 30 \
    --start-epoch 0 \
    --exp-dir ${basedir}/${task} \
    --full-libri 0 \
    --max-duration 150 \
    --lr-factor 2.5
