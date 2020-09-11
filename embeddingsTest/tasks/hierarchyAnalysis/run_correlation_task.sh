#!/bin/bash

nohup python correlation_eval.py --top_k 10 --class2superclass_file class2superclasses.csv --docstrings_file /data/embedding/merge-15-22.2.format.json --classmap classes.map --classfail classes.fail --embed_type USE > results/corr_use.results

nohup python correlation_eval.py --top_k 10 --class2superclass_file class2superclasses.csv --docstrings_file /data/embedding/merge-15-22.2.format.json --classmap classes.map --classfail classes.fail --embed_type bert > results/corr_bert.results

nohup python correlation_eval.py --top_k 10 --class2superclass_file class2superclasses.csv --docstrings_file /data/embedding/merge-15-22.2.format.json --classmap classes.map --classfail classes.fail --embed_type roberta > results/corr_roberta.results
