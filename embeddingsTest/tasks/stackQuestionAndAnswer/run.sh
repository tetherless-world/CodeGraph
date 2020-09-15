#!/bin/bash

if [ -f stackoverflow_data_ranking.json.tar.gz ]; then
    echo "stackoverflow_data_ranking.json.tar.gz"
else
    wget -O - https://archive.org/download/merge-15-22.2.format/stackoverflow_data_ranking.json.tar.gz
    tar -xzf stackoverflow_data_ranking.json.tar.gz
fi

# get precomputed embeddings for stackoverflow ranking, for RoBERTa
if [ -f stackoverflow_data_ranking_title_all_bert_base.tar.gz ]; then
    echo "using stackoverflow_data_ranking_title_all_bert_base.tar.gz"
else
    wget https://archive.org/download/merge-15-22.2.format/stackoverflow_data_ranking_title_all_bert_base.tar.gz
    tar -xzf stackoverflow_data_ranking_title_all_bert_base.tar.gz
fi

# get precomputed embeddings for stackoverflow ranking, for BERT
if [ -f stackoverflow_data_ranking_title_all_roberta_base.tar.gz ]; then
    echo "using stackoverflow_data_ranking_title_all_roberta_base.tar.gz"
else
    wget https://archive.org/download/merge-15-22.2.format/stackoverflow_data_ranking_title_all_roberta_base.tar.gz
    tar -xzf stackoverflow_data_ranking_title_all_roberta_base.tar.gz
fi

# get precomputed embeddings for stackoverflow ranking, for USE
if [ -f stackoverflow_data_ranking_title_all_USE.tar.gz ]; then
    echo "using stackoverflow_data_ranking_title_all_USE.tar.gz"
else
    wget https://archive.org/download/merge-15-22.2.format/stackoverflow_data_ranking_title_all_USE.tar.gz
    tar -xzf stackoverflow_data_ranking_title_all_USE.tar.gz
fi


if [ -f /tmp/ranking_metrics_roberta.out ]; then
    echo "using ranking_metrics_roberta.out"
else
    PYTHONPATH=.. python rank_answers.py ./stackoverflow_data_ranking.json roberta-base-nli-mean-tokens ./stackoverflow_data_ranking_title_all_roberta_base/  > /tmp/ranking_metrics_roberta.out
fi

if [ -f /tmp/ranking_metrics_bert.out ]; then
    echo "using ranking_metrics_bert.out"
else
    PYTHONPATH=.. python rank_answers.py ./stackoverflow_data_ranking.json bert-base-nli-mean-tokens ./stackoverflow_data_ranking_title_all_bert_base/  > /tmp/ranking_metrics_bert.out
fi

if [ -f /tmp/ranking_metrics_use.out ]; then
    echo "using ranking_metrics_use.out"
else
    PYTHONPATH=.. python rank_answers.py ./stackoverflow_data_ranking.json https://tfhub.dev/google/universal-sentence-encoder/4 ./stackoverflow_data_ranking_title_all_USE/  > /tmp/ranking_metrics_use.out
fi

cat /tmp/ranking_metrics_bert.out
echo "--------------------------------"
cat /tmp/ranking_metrics_roberta.out
echo "--------------------------------"
cat /tmp/ranking_metrics_use.out
echo "--------------------------------"

