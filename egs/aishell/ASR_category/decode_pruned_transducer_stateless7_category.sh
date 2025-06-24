#!/bin/bash

export CUDA_VISIBLE_DEVICES="0"
basedir="pruned_transducer_stateless7_category_0611"
task=wt_category50_wt_weight0.1_lr0.055
${basedir}/decode.py \
	--epoch 30 \
	--avg 1 \
	--use-averaged-model False \
	--num-category 50 \
	--exp-dir ${PWD}/experiments/${task} \
	--max-duration 1200 \
	--decoding-method modified_beam_search \
	--beam-size 4

