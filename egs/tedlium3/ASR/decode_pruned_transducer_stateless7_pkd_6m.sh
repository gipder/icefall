#!/bin/bash
#pushd /icefall/egs/librispeech/ASR
#--decoding-method beam_search \
export CUDA_VISIBLE_DEVICES=0
basedir="pruned_transducer_stateless7_pkd_sampling"
task="6m_pruned5_wt_gt_wo_kd"

IN=${1:-"30"}
for epoch in 10
do
	for avg in 1
	do
		python3 $PWD/$basedir/decode.py --epoch $epoch \
				--avg $avg \
				--exp-dir $PWD/experiments/$task \
				--bpe-model ./data/lang_bpe_500/bpe.model \
				--decoding-method modified_beam_search \
        --use-averaged-model False \
				--max-duration 100 \
        --num-encoder-layers "1,1,1,1,1" \
        --feedforward-dims "256,256,512,512,256" \
        --nhead "4,4,4,4,4" \
        --encoder-dims "196,196,196,196,196" \
        --encoder-unmasked-dims "196,196,196,196,196" \
				--beam-size 4
	done
done
