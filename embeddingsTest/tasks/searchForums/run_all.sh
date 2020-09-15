#!/bin/bash

# run each embedding, assuming all datasets are mounted in /data
tar xzf /data/stackoverflow_matches_codesearchnet_5k_USE.tar.gz

sh run_specific_embedding.sh /data/stackoverflow_matches_codesearchnet_5k_USE USE

rm - rf /data/stackoverflow_matches_codesearchnet_5k_USE

tar xzf /data/stackoverflow_matches_codesearchnet_5k_bert_base.tar.gz

sh run_specific_embedding.sh /data/stackoverflow_matches_codesearchnet_5k_bert_base bert

rm -rf /data/stackoverflow_matches_codesearchnet_5k_bert_base

tar xzf /data/stackoverflow_matches_codesearchnet_5k_roberta_base.tar.gz

sh run_specific_embedding.sh /data/stackoverflow_matches_codesearchnet_5k_robert_base roberta

rm -rf /data/stackoverflow_matches_codesearchnet_5k_roberta_base

# compare search for title versus search for all content
python evaluate_search_content.py --search_titles_file /data/stackoverflow_matches_codesearchnet_5k.json --search_contents_file /data/stackoverflow_matches_codesearchnet_5k_content.json --top_k 50 > content_title_50

python evaluate_search_content.py --search_titles_file /data/stackoverflow_matches_codesearchnet_5k.json --search_contents_file /data/stackoverflow_matches_codesearchnet_5k_content.json --top_k 100 > content_title_100

python evaluate_search_content.py --search_titles_file /data/stackoverflow_matches_codesearchnet_5k.json --search_contents_file /data/stackoverflow_matches_codesearchnet_5k_content.json --top_k 200 > content_title_200

python evaluate_search_content.py --search_titles_file /data/stackoverflow_matches_codesearchnet_5k.json --search_contents_file /data/stackoverflow_matches_codesearchnet_5k_content.json --top_k 400 > content_title_400

python evaluate_search_content.py --search_titles_file /data/stackoverflow_matches_codesearchnet_5k.json --search_contents_file /data/stackoverflow_matches_codesearchnet_5k_content.json --top_k 800 > content_title_800
