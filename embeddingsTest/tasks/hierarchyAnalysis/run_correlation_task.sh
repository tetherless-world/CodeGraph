#!/bin/bash

nohup python correlation_eval.py --top_k 10 --class2superclass_file /data/class2superclasses.csv --docstrings_file /data/merge-15-22.2.format.json --classmap /data/classes.map --classfail /data/classes.fail --embed_type USE > corr_use.results

nohup python correlation_eval.py --top_k 10 --class2superclass_file /data/class2superclasses.csv --docstrings_file /data/merge-15-22.2.format.json --classmap /data/classes.map --classfail /data/classes.fail --embed_type bert > corr_bert.results

nohup python correlation_eval.py --top_k 10 --class2superclass_file /data/class2superclasses.csv --docstrings_file /data/embedding/merge-15-22.2.format.json --classmap /data/classes.map --classfail /data/classes.fail --embed_type roberta > corr_roberta.results
