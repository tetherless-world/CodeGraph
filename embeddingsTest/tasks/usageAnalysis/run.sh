#!/bin/bash

ME=$0
DIR=`dirname $ME`

DATA=$1

bash $DIR/run_use.sh $DATA

if [ -f /tmp/out_usage_bert ]; then
    echo "using /tmp/out_usage_bert"
else
    PYTHONPATH=.. python staticAnalysis2.py $DATA/classes2superclass.out $DATA/merge-15-22.2.format.json $DATA/classes.map $DATA/usage.txt bert >/tmp/out_usage_bert
fi

if [ -f /tmp/out_usage_roberta ]; then
    echo "using /tmp/out_usage_roberta"
else
    PYTHONPATH=.. python staticAnalysis2.py $DATA/classes2superclass.out $DATA/merge-15-22.2.format.json $DATA/classes.map $DATA/usage.txt roberta >/tmp/out_usage_roberta
fi

bash $DIR/plot.sh
