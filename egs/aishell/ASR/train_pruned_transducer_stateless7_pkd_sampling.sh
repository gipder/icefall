#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1
basedir="pruned_transducer_stateless7_pkd_sampling"
task="6m_prune5_wt_gt_wo_kd"
base_expdir="experiments"
expdir="${base_expdir}/${task}"
threshold=0.97
#[ -e ${basedir}/${task} ] && rm -rf ${basedir}/${task}
${basedir}/train.py \
    --world-size 2 \
    --master-port 15586 \
    --num-epochs 30 \
    --start-epoch 1 \
    --exp-dir ${expdir} \
    --max-duration 100 \
    --prune-range 5  \
    --base-lr 0.035 \
    --num-encoder-layers "1,1,1,1,1" \
    --feedforward-dims "256,256,512,512,256" \
    --nhead "4,4,4,4,4" \
    --encoder-dims "196,196,196,196,196" \
    --encoder-unmasked-dims "196,196,196,196,196" \
    --keep-last-k 6 \
    --use-pkd False \
    --pkd-range 5 \
    --use-ctc False \
    --pkd-loss-scale 0.01


#    --teacher-checkpoint ${base_expdir}/teacher_model_70m/epoch-30.pt \
#    --teacher-num-encoder-layers "2,4,3,2,4" \
#    --teacher-feedforward-dims "1024,1024,2048,2048,1024" \
#    --teacher-nhead "8,8,8,8,8" \
#    --teacher-encoder-dims "384,384,384,384,384" \
#    --use-teacher-ctc-alignment False \
#    --use-efficient False \
#    --use-time-compression False \
#    --time-compression-threshold ${threshold} \
#    --use-teacher-simple-proj False \
#    --use-beam-search False \
#    --use-beam-search-alignment False \
#    --dump-hyp ${base_expdir}/train-clean-100.from_test_loader.pkl \
#    --use-sq-sampling True \
#    --sq-sampling-num 1

