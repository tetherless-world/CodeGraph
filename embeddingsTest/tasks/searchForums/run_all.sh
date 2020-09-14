#!/bin/bash

# run each embedding
tar xzf stackoverflow_matches_codesearchnet_5k_USE.tar.gz

sh run_specific_embedding.sh stackoverflow_matches_codesearchnet_5k_USE USE

rm - rf stackoverflow_matches_codesearchnet_5k_USE

tar xzf stackoverflow_matches_codesearchnet_5k_bert_base.tar.gz

sh run_specific_embedding.sh stackoverflow_matches_codesearchnet_5k_bert_base bert

rm -rf stackoverflow_matches_codesearchnet_5k_bert_base

tar xzf stackoverflow_matches_codesearchnet_5k_roberta_base.tar.gz

sh run_specific_embedding.sh stackoverflow_matches_codesearchnet_5k_robert_base roberta

rm -rf stackoverflow_matches_codesearchnet_5k_roberta_base
