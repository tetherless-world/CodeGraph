#!/bin/bash

DATA=$1

bash dataset.sh $DATA

bash eval_use.sh

bash eval_bert.sh

bash eval_roberta.sh

bash eval_bertoverflow.sh

bash plot.sh use
