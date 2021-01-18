#!/bin/bash

export PYTHONPATH=../:.

python hierarchy_stats.py --eval_file /data/blanca/hierarchy_test.json --docstrings_file /data/download/merge-15-22.2.format.json --classmap /data/download/classes.map --classfail /data/download/classes.fail --embed_type finetuned --model_dir /data/BERTOverflow-tuned/hierarchy_small/dccstor/m3/blanca/BERTOverflow/-2021-01-13_11-19-41/0_Transformer/ > ./tuned_hierarchy_small

python hierarchy_stats.py --eval_file /data/blanca/hierarchy_test.json --docstrings_file /data/download/merge-15-22.2.format.json --classmap /data/download/classes.map --classfail /data/download/classes.fail --embed_type bertoverflow --model_dir /data/BERTOverflow/ > ./bertoverflow_hierarchy_small

python hierarchy_stats.py --eval_file /data/blanca/hierarchy_test.json --docstrings_file /data/download/merge-15-22.2.format.json --classmap /data/download/classes.map --classfail /data/download/classes.fail --embed_type bert --model_dir /dev/null > ./bert_hierarchy_small

python hierarchy_stats.py --eval_file /data/blanca/hierarchy_test.json --docstrings_file /data/download/merge-15-22.2.format.json --classmap /data/download/classes.map --classfail /data/download/classes.fail --embed_type xlm --model_dir /dev/null > ./xlm_hierarchy_small

python hierarchy_stats.py --eval_file /data/blanca/hierarchy_test.json --docstrings_file /data/download/merge-15-22.2.format.json --classmap /data/download/classes.map --classfail /data/download/classes.fail --embed_type distilbert_para --model_dir /dev/null > ./distilbert_para_hierarchy_small 

python hierarchy_stats.py --eval_file /data/blanca/hierarchy_test.json --docstrings_file /data/download/merge-15-22.2.format.json --classmap /data/download/classes.map --classfail /data/download/classes.fail --embed_type msmacro --model_dir /dev/null > ./msmarco_hiearchy_small

python hierarchy_stats.py --eval_file /data/blanca/hierarchy_test.json --docstrings_file /data/download/merge-15-22.2.format.json --classmap /data/download/classes.map --classfail /data/download/classes.fail --embed_type USE --model_dir /dev/null > ./USE_hiearchy_small





