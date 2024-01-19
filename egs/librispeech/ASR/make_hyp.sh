#!/bin/bash
#pushd /icefall/egs/librispeech/ASR
#--decoding-method beam_search \
export CUDA_VISIBLE_DEVICES=3

basedir="pruned_transducer_stateless7_pkd_pseudo_gt_cache_check"
task="teacher_model_70m"

IN=${1:-"30"}
for epoch in $IN
do
	for avg in 1
	do
		python3 $PWD/$basedir/make_hyp.py --epoch $epoch \
				--avg $avg \
				--exp-dir $PWD/experiments/$task \
				--bpe-model ./data/lang_bpe_500/bpe.model \
				--decoding-method modified_beam_search_for_kd \
        --save-hypotheses train-clean-100.pkl \
        --use-averaged-model False \
				--max-duration 200 \
				--beam-size 4
	done
done
