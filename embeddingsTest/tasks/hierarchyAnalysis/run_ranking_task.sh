#!/bin/bash

nohup python hierarchy_stats.py --top_k 10 --class2superclass_file class2superclasses.csv --docstrings_file /data/embedding/merge-15-22.2.format.json --classmap classes.map --classfail classes.fail --embed_type USE > results/ranking_USE &

nohup python hierarchy_stats.py --top_k 10 --class2superclass_file class2superclasses.csv --docstrings_file /data/embedding/merge-15-22.2.format.json --classmap classes.map --classfail classes.fail --embed_type bert > results/ranking_bert &

nohup python hierarchy_stats.py --top_k 10 --class2superclass_file class2superclasses.csv --docstrings_file /data/embedding/merge-15-22.2.format.json --classmap classes.map --classfail classes.fail --embed_type roberta > results/ranking_roberta &
