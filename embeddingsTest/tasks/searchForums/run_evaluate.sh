#!/bin/bash
# Get search results

if [ -f stackoverflow_matches_codesearchnet_5k.json ]; then
    echo "using stackoverflow_matches_codesearchnet_5k.json"
else
    wget https://archive.org/download/merge-15-22.2.format/stackoverflow_matches_codesearchnet_5k.json.zip
    unzip stackoverflow_matches_codesearchnet_5k.json
fi

if [ -f stackoverflow_matches_codesearchnet_5k_content.json ]; then
    echo "using stackoverflow_matches_codesearchnet_5k_content.json"
else
    wget https://archive.org/download/merge-15-22.2.format/stackoverflow_matches_codesearchnet_5k_content.json
fi

# compare search for title versus search for all content
python evaluate_search_content.py --search_titles_file stackoverflow_matches_codesearchnet_5k.json --search_contents_file stackoverflow_matches_codesearchnet_5k_content.json --top_k 50 > results/content_title_50
python evaluate_search_content.py --search_titles_file stackoverflow_matches_codesearchnet_5k.json --search_contents_file stackoverflow_matches_codesearchnet_5k_content.json --top_k 100 > results/content_title_100
python evaluate_search_content.py --search_titles_file stackoverflow_matches_codesearchnet_5k.json --search_contents_file stackoverflow_matches_codesearchnet_5k_content.json --top_k 200 > results/content_title_200
python evaluate_search_content.py --search_titles_file stackoverflow_matches_codesearchnet_5k.json --search_contents_file stackoverflow_matches_codesearchnet_5k_content.json --top_k 400 > results/content_title_400
python evaluate_search_content.py --search_titles_file stackoverflow_matches_codesearchnet_5k.json --search_contents_file stackoverflow_matches_codesearchnet_5k_content.json --top_k 800 > results/content_title_800
