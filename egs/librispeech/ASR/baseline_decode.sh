#!/bin/bash
#pushd /icefall/egs/librispeech/ASR
#--decoding-method beam_search \

IN=${1:-"29"}
for epoch in $IN
do
	for avg in 1 15
	do
		task=transducer_stateless2_baseline
        exp=baseline
		python3 $PWD/$task/decode.py --epoch $epoch \
				--avg $avg \
				--exp-dir $PWD/$task/$exp \
				--bpe-model ./data/lang_bpe_500/bpe.model \
				--decoding-method greedy_search \
				--max-duration 600 \
				--beam-size 4
	done
done

popd

