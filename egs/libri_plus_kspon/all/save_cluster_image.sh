#!/bin/bash

for cluster_type in "embedding"; do
#for cluster_type in "ipa"; do
for top_k in 3 5 10 15; do
python pruned_transducer_stateless7_mix_category_260221_integration/my_utils.py \
    --tokens data/lang_char_aishell_4000/tokens.txt \
    --embedding experiments_full/data/token_embeddings.pt \
    --num-category-list "100" \
    --top-k "$top_k"     \
    --out-dir "experiments_full/data/cluster_visualization/aishell_4000" \
    --random-state 42 \
    --cluster-type "$cluster_type"
done
done