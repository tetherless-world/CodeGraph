#!/bin/bash

if [ -f /tmp/roberta_links.out ]; then
    echo "using roberta_links.out"
else
    PYTHONPATH=.. python eval.py /tmp/dataset.json /tmp/testset.json roberta /tmp/true_roberta.json /tmp/false_roberta.json > /tmp/roberta_links.out 
fi
