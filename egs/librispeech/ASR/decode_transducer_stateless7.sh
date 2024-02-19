#!/bin/bash
#pushd /icefall/egs/librispeech/ASR
#--decoding-method beam_search \

basedir="transducer_stateless7"
task="baseline"

IN=${1:-"30"}
for epoch in $IN
do
	for avg in 1
	do
		python3 $PWD/$basedir/decode.py --epoch $epoch \
				--avg $avg \
				--exp-dir $PWD/$basedir/$task \
				--bpe-model ./data/lang_bpe_500/bpe.model \
				--decoding-method modified_beam_search \
        --use-averaged-model False \
				--max-duration 200 \
				--beam-size 4
	done
done
