#!/bin/bash
# unzip the embedding dictionary file first with:
# tar xzf stackoverflow_matches_codesearchnet_5k_USE.tar.gz
# Pass directory into arg 1
# Pass embed_type: USE, bert, roberta as arg 2
#
python search_task.py --top_k 50 --search_file_results /data/embedding/stackoverflow_matches_codesearchnet_5k.json --embedding_dict_dir $1 --add_all_content False --embed_type $2 > $2results50_title

python search_task.py --top_k 100 --search_file_results /data/embedding/stackoverflow_matches_codesearchnet_5k.json --embedding_dict_dir $1 --add_all_content False --embed_type $2 >$2results100_title

python search_task.py --top_k 200 --search_file_results /data/embedding/stackoverflow_matches_codesearchnet_5k.json --embedding_dict_dir $1 --add_all_content False --embed_type $2 > $2results200_title

python search_task.py --top_k 400 --search_file_results /data/embedding/stackoverflow_matches_codesearchnet_5k.json --embedding_dict_dir $1 --add_all_content False --embed_type $2 > $2results400_title

python search_task.py --top_k 800 --search_file_results /data/embedding/stackoverflow_matches_codesearchnet_5k.json --embedding_dict_dir $1 --add_all_content False --embed_type $2 > $2results800_title

python search_task.py --top_k 50 --search_file_results /data/embedding/stackoverflow_matches_codesearchnet_5k.json --embedding_dict_dir $1 --add_all_content True --embed_type $2 > $2results50_all

python search_task.py --top_k 100 --search_file_results /data/embedding/stackoverflow_matches_codesearchnet_5k.json --embedding_dict_dir $1 --add_all_content True --embed_type $2 > $2results100_all

python search_task.py --top_k 200 --search_file_results /data/embedding/stackoverflow_matches_codesearchnet_5k.json --embedding_dict_dir $1 --add_all_content True --embed_type $2 > $2results200_all

python search_task.py --top_k 400 --search_file_results /data/embedding/stackoverflow_matches_codesearchnet_5k.json --embedding_dict_dir $1 --add_all_content True --embed_type $2 > $2results400_all

python search_task.py --top_k 800 --search_file_results /data/embedding/stackoverflow_matches_codesearchnet_5k.json --embedding_dict_dir $1 --add_all_content True --embed_type $2 > $2results800_all
