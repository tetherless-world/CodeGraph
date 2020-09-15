#!/bin/bash

DATA=$1

bash dataset.py $DATA

bash eval_use.sh

bash eval_bert.sh

bash eval_roberta.sh

bash plot.sh
