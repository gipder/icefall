#!/bin/bash

export CUDA_VISIBLE_DEVICES="5"
basedir="pruned_transducer_stateless7"
task=pruned_rnnt_lr0.055
${basedir}/decode.py \
	--epoch 30 \
	--avg 1 \
	--use-averaged-model False \
	--exp-dir ${PWD}/experiments/${task} \
	--max-duration 400 \
	--decoding-method modified_beam_search \
	--beam-size 4
