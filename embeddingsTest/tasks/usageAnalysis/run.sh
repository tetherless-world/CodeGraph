#!/bin/bash

ME=$0
DIR=`diname $ME`

INCLUDES=$DIR/..

wget https://archive.org/download/merge-15-22.2.format/classes2superclass.out
wget https://archive.org/download/merge-15-22.2.format/merge-15-22.2.format.json
wget https://archive.org/download/merge-15-22.2.format/classes.map
wget https://archive.org/download/merge-15-22.2.format/usage.txt

PYTHONPATH=$INCLUDES python staticAnalysis2.py classes2superclass.out merge-15-22.2.format.json classes.map usage.txt USE >/tmp/out_usage_USE

PYTHONPATH=$INCLUDES python staticAnalysis2.py classes2superclass.out merge-15-22.2.format.json classes.map usage.txt bert >/tmp/out_usage_bert

PYTHONPATH=$INCLUDES python staticAnalysis2.py classes2superclass.out merge-15-22.2.format.json classes.map usage.txt roberta >/tmp/out_usage_roberta
