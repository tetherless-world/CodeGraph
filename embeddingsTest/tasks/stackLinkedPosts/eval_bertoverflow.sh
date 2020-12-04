#!/bin/bash

if [ -f /tmp/bertoverflow_links.out ]; then
    echo "using bertoverflow_links.out"
else
    PYTHONPATH=.. python eval.py /tmp/dataset.json /tmp/testset.json bertoverflow /tmp/true_bertoverflow.json /tmp/false_bertoverflow.json > /tmp/bertoverflow_links.out 
fi

