#!/bin/bash

ME=$0
DIR=`dirname $ME`

DATA=$1

cd $DIR

if [ -f /tmp/out_usage_USE ]; then
    echo "using /tmp/out_usage_USE"
else
    PYTHONPATH=.. python staticAnalysis2.py $DATA/classes2superclass.out $DATA/merge-15-22.2.format.json $DATA/classes.map $DATA/usage.txt USE >/tmp/out_usage_USE
fi

