#!/bin/bash

if [ -f stackoverflow_data_ranking.json ]; then
    echo "using stackoverflow_data_ranking.json"
else
    wget https://archive.org/download/merge-15-22.2.format/stackoverflow_data_ranking.json.tar.gz
    tar xzf stackoverflow_data_ranking.json.tar.gz
fi

if [ -f /tmp/dataset.json ]; then
    echo "using /tmp/dataset.json"
else
    python dataset.py /data/stackoverflow_dump/pickled/stackoverflow_data_ranking.json /tmp/dataset.json /tmp/testset.json
fi

if [ -f /tmp/use_links.out ]; then
    echo "using use_links.out"
else
    PYTHONPATH=.. python eval.py /tmp/dataset.json /tmp/testset.json use /tmp/true_use.json /tmp/false_use.json > /tmp/use_links.out 
fi

if [ -f /tmp/bert_links.out ]; then
    echo "using bert_links.out"
else
    PYTHONPATH=.. python eval.py /tmp/dataset.json /tmp/testset.json bert /tmp/true_bert.json /tmp/false_bert.json > /tmp/bert_links.out 
fi

if [ -f /tmp/roberta_links.out ]; then
    echo "using roberta_links.out"
else
    PYTHONPATH=.. python eval.py /tmp/dataset.json /tmp/testset.json roberta /tmp/true_roberta.json /tmp/false_roberta.json > /tmp/roberta_links.out 
fi

