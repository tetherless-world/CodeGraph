#!/bin/bash

if [ -f /tmp/bert_links.out ]; then
    echo "using bert_links.out"
else
    PYTHONPATH=.. python eval.py /tmp/dataset.json /tmp/testset.json bert /tmp/true_bert.json /tmp/false_bert.json > /tmp/bert_links.out 
fi

