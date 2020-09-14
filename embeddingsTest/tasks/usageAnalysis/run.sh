#!/bin/bash

ME=$0
DIR=`dirname $ME`

if [[ -f classes2superclass.out ]]; then
    echo "using classes2superclass.out"
else
    wget https://archive.org/download/merge-15-22.2.format/classes2superclass.out
fi

if [[ -f merge-15-22.2.format.json ]]; then
    echo "using merge-15-22.2.format.json"
else
    wget https://archive.org/download/merge-15-22.2.format/merge-15-22.2.format.json
fi

if [[ -f classes.map ]]; then
    echo "using classes.map"
else
    wget https://archive.org/download/merge-15-22.2.format/classes.map
fi

if [[ -f usage.txt ]]; then
    echo "using usage.txt"
else
    wget https://archive.org/download/merge-15-22.2.format/usage.txt
fi

if [ -f /tmp/out_usage_USE ]; then
    echo "using /tmp/out_usage_USE"
else
    PYTHONPATH=.. python staticAnalysis2.py classes2superclass.out merge-15-22.2.format.json classes.map usage.txt USE >/tmp/out_usage_USE
fi

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

awk -f mrr-map.awk -v start=103 /tmp/out_usage_USE > mrr-map-use.dat
awk -f mrr-map.awk -v start=103 /tmp/out_usage_bert > mrr-map-bert.dat
awk -f mrr-map.awk -v start=103 /tmp/out_usage_roberta > mrr-map-roberta.dat

gnuplot mrr-map.gnuplot > mrr-map.pdf
