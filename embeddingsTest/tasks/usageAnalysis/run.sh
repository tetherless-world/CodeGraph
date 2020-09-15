#!/bin/bash

ME=$0
DIR=`dirname $ME`

DATA=$1

cd $DATA

bash $DIR/run_use.sh $DATA

if [ -f /tmp/out_usage_bert ]; then
    echo "using /tmp/out_usage_bert"
else
    PYTHONPATH=.. python staticAnalysis2.py classes2superclass.out merge-15-22.2.format.json classes.map usage.txt bert >/tmp/out_usage_bert
fi

if [ -f /tmp/out_usage_roberta ]; then
    echo "using /tmp/out_usage_roberta"
else
    PYTHONPATH=.. python staticAnalysis2.py classes2superclass.out merge-15-22.2.format.json classes.map usage.txt roberta >/tmp/out_usage_roberta
fi

bash $DIR/plot.sh
