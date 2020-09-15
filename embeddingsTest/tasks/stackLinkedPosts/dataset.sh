#!/bin/bash

DATA=$1

if [ -f /tmp/dataset.json ]; then
    echo "using /tmp/dataset.json"
else
    python dataset.py $DATA/stackoverflow_data_ranking.json /tmp/dataset.json /tmp/testset.json
fi
