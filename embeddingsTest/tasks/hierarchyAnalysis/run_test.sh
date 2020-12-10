#!/bin/bash

export PYTHONPATH=../:.

nohup python hierarchy_stats.py --eval_file /data/blanca/hierarchy_test.json --class2superclass_file /data/download/class2superclasses.csv --docstrings_file /data/download/merge-15-22.2.format.json --classmap /data/download/classes.map --classfail /data/download/classes.fail --embed_type USE > /tmp/hierachy/USE 2>&1 &

