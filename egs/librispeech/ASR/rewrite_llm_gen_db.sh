#!/bin/bash

IN=$1
OUT=$2

rewrite_llm_gen_db=${PWD}/pruned_transducer_stateless7_pkd_sampling/pickle_rewrite.py

python3 $rewrite_llm_gen_db $IN $OUT
