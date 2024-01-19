#!/bin/bash

export CUDA_VISIBLE_DEVICES=1,3
basedir="pruned_transducer_stateless7_pkd_pseudo_gt_cache_check"
task="6m_prune5_wt_pseudo_labels_wt_alignment_retrain_cache_check"
#task="garbage2"
base_expdir="experiments"
expdir="${base_expdir}/${task}"
threshold=0.97
#[ -e ${basedir}/${task} ] && rm -rf ${basedir}/${task}
${basedir}/train.test.py \
    --world-size 2 \
    --master-port 29546 \
    --num-epochs 30 \
    --start-epoch 1 \
    --exp-dir ${expdir} \
    --full-libri 0 \
    --max-duration 100 \
    --prune-range 5  \
    --num-encoder-layers "1,1,1,1,1" \
    --feedforward-dims "256,256,512,512,256" \
    --nhead "4,4,4,4,4" \
    --encoder-dims "196,196,196,196,196" \
    --encoder-unmasked-dims "196,196,196,196,196" \
    --keep-last-k 6 \
    --use-pkd True \
    --use-ctc False \
    --pkd-loss-scale 0.5 \
    --teacher-checkpoint ${base_expdir}/teacher_model_70m/epoch-30.pt \
    --teacher-num-encoder-layers "2,4,3,2,4" \
    --teacher-feedforward-dims "1024,1024,2048,2048,1024" \
    --teacher-nhead "8,8,8,8,8" \
    --teacher-encoder-dims "384,384,384,384,384" \
    --use-teacher-ctc-alignment False \
    --use-efficient False \
    --use-time-compression False \
    --time-compression-threshold ${threshold} \
    --use-teacher-simple-proj False \
    --use-alphas False \
    --use-beam-search True \
    --dump-hyp ${base_expdir}/train-clean-100.pkl \
    --use-teacher-ctc-alignment False \
