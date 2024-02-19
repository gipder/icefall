#!/bin/bash

export CUDA_VISIBLE_DEVICES=2,3
basedir="transducer_stateless7_tmp"
task="6m_non_pruned_wt_pseudo_y_wt_1best_alignment_wt_seq_wt_kd_scale_0p01"
base_expdir="experiments"
expdir="${base_expdir}/${task}"
#[ -e ${basedir}/${task} ] && rm -rf ${basedir}/${task}
${basedir}/train.py \
    --world-size 2 \
    --master-port 23546 \
    --num-epochs 30 \
    --start-epoch 1 \
    --exp-dir ${expdir} \
    --full-libri 0 \
    --max-duration 100 \
    --base-lr 0.035 \
    --num-encoder-layers "1,1,1,1,1" \
    --feedforward-dims "256,256,512,512,256" \
    --nhead "4,4,4,4,4" \
    --encoder-dims "196,196,196,196,196" \
    --encoder-unmasked-dims "196,196,196,196,196" \
    --keep-last-k 6 \
    --use-kd True \
    --use-efficient False \
    --use-ctc False \
    --kd-loss-scale 0.01 \
    --teacher-checkpoint ${base_expdir}/teacher_model_70m/epoch-30.pt \
    --teacher-num-encoder-layers "2,4,3,2,4" \
    --teacher-feedforward-dims "1024,1024,2048,2048,1024" \
    --teacher-nhead "8,8,8,8,8" \
    --teacher-encoder-dims "384,384,384,384,384" \
    --use-teacher-simple-proj False \
    --use-beam-search True \
    --use-beam-search-alignment True \
    --dump-hyp ${base_expdir}/train-clean-100.from_test_loader.pkl \
    --use-1best False \
    --use-sequence True \
