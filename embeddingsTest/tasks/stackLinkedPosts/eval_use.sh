#!/bin/bash

if [ -f /tmp/use_links.out ]; then
    echo "using use_links.out"
else
    PYTHONPATH=.. python eval.py /tmp/dataset.json /tmp/testset.json use /tmp/true_use.json /tmp/false_use.json > /tmp/use_links.out 
fi

