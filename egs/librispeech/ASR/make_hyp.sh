#!/bin/bash
#pushd /icefall/egs/librispeech/ASR
#--decoding-method beam_search \
export CUDA_VISIBLE_DEVICES=3

basedir="pruned_transducer_stateless7_pkd_sampling"
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
				--decoding-method modified_beam_search_for_kd_nbest \
        --save-hypotheses train-clean-100.nbest.from_test_loader.pkl \
        --use-averaged-model False \
				--max-duration 300 \
				--beam-size 4 \
        --topk 4
	done
done
