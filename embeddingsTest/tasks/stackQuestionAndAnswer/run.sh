#!/bin/bash

DATA=$1

if [ -f /tmp/ranking_metrics_roberta.out ]; then
    echo "using ranking_metrics_roberta.out"
else
    PYTHONPATH=.. python rank_answers.py $DATA/stackoverflow_data_ranking.json roberta-base-nli-mean-tokens $DATA/stackoverflow_data_ranking_title_all_roberta_base/  > /tmp/ranking_metrics_roberta.out
fi

if [ -f /tmp/ranking_metrics_bert.out ]; then
    echo "using ranking_metrics_bert.out"
else
    PYTHONPATH=.. python rank_answers.py $DATA/stackoverflow_data_ranking.json bert-base-nli-mean-tokens $DATA/stackoverflow_data_ranking_title_all_bert_base/  > /tmp/ranking_metrics_bert.out
fi

if [ -f /tmp/ranking_metrics_use.out ]; then
    echo "using ranking_metrics_use.out"
else
    PYTHONPATH=.. python rank_answers.py $DATA/stackoverflow_data_ranking.json https://tfhub.dev/google/universal-sentence-encoder/4 $DATA/stackoverflow_data_ranking_title_all_USE/  > /tmp/ranking_metrics_use.out
fi

cat /tmp/ranking_metrics_bert.out
echo "--------------------------------"
cat /tmp/ranking_metrics_roberta.out
echo "--------------------------------"
cat /tmp/ranking_metrics_use.out
echo "--------------------------------"

