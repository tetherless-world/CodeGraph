#!/bin/bash

nohup python hierarchy_stats.py --top_k 10 --class2superclass_file /data/class2superclasses.csv --docstrings_file /data/merge-15-22.2.format.json --classmap /data/classes.map --classfail /data/classes.fail --embed_type USE > ranking_USE &

nohup python hierarchy_stats.py --top_k 10 --class2superclass_file /data/class2superclasses.csv --docstrings_file /data/merge-15-22.2.format.json --classmap /data/classes.map --classfail /data/classes.fail --embed_type bert > ranking_bert &

nohup python hierarchy_stats.py --top_k 10 --class2superclass_file /data/class2superclasses.csv --docstrings_file /data/merge-15-22.2.format.json --classmap /data/classes.map --classfail /data/classes.fail --embed_type roberta > ranking_roberta &
