#!/bin/bash

DATA=$1

bash dataset.sh $DATA

bash eval_use.sh

xdbash eval_bert.sh

bash eval_roberta.sh

bash plot.sh
