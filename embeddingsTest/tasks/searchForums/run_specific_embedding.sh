#!/bin/bash

python search_task.py --top_k 50 --search_file_results stackoverflow_matches_codesearchnet_5k.json --embedding_dict_dir $1 --add_all_content False --embed_type $2 > $2results50_title

python search_task.py --top_k 100 --search_file_results stackoverflow_matches_codesearchnet_5k.json --embedding_dict_dir $1 --add_all_content False --embed_type $2 >$2results100_title

python search_task.py --top_k 200 --search_file_results stackoverflow_matches_codesearchnet_5k.json --embedding_dict_dir $1 --add_all_content False --embed_type $2 > $2results200_title

python search_task.py --top_k 400 --search_file_results stackoverflow_matches_codesearchnet_5k.json --embedding_dict_dir $1 --add_all_content False --embed_type $2 > $2results400_title

python search_task.py --top_k 800 --search_file_results stackoverflow_matches_codesearchnet_5k.json --embedding_dict_dir $1 --add_all_content False --embed_type $2 > $2results800_title

python search_task.py --top_k 50 --search_file_results stackoverflow_matches_codesearchnet_5k.json --embedding_dict_dir $1 --add_all_content True --embed_type $2 > $2results50_all

python search_task.py --top_k 100 --search_file_results stackoverflow_matches_codesearchnet_5k.json --embedding_dict_dir $1 --add_all_content True --embed_type $2 > $2results100_all

python search_task.py --top_k 200 --search_file_results stackoverflow_matches_codesearchnet_5k.json --embedding_dict_dir $1 --add_all_content True --embed_type $2 > $2results200_all

python search_task.py --top_k 400 --search_file_results stackoverflow_matches_codesearchnet_5k.json --embedding_dict_dir $1 --add_all_content True --embed_type $2 > $2results400_all

python search_task.py --top_k 800 --search_file_results stackoverflow_matches_codesearchnet_5k.json --embedding_dict_dir $1 --add_all_content True --embed_type $2 > $2results800_all
