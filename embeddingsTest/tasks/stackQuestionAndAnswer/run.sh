#!/bin/bash

DATA=$1

if [ -f $DATA/stackoverflow_data_ranking.json ]; then
    echo "using stackoverflow_data_ranking.json"
elif [ -f  $DATA/stackoverflow_data_ranking.json.tar.gz ]; then
    echo "untar stackoverflow_data_ranking"
    tar -xzf $DATA/stackoverflow_data_ranking.json.tar.gz -C $DATA/
else
    echo "download stackoverflow_data_ranking"
    wget -O - https://archive.org/download/merge-15-22.2.format/stackoverflow_data_ranking.json.tar.gz -P $DATA/
    tar -xzf $DATA/stackoverflow_data_ranking.json.tar.gz -C $DATA/
fi

# get precomputed embeddings for stackoverflow ranking, for RoBERTa
if [ -d $DATA/stackoverflow_data_ranking_title_all_bert_base ]; then
    echo "using stackoverflow_data_ranking_title_all_bert_base"
elif [ -f $DATA/stackoverflow_data_ranking_title_all_bert_base.tar.gz ]; then
    echo "untar stackoverflow_data_ranking_title_all_bert_base"
    tar -xzf $DATA/stackoverflow_data_ranking_title_all_bert_base.tar.gz -C $DATA/
else
    echo "download stackoverflow_data_ranking_title_all_bert_base"
    wget https://archive.org/download/merge-15-22.2.format/stackoverflow_data_ranking_title_all_bert_base.tar.gz -P $DATA/
    tar -xzf $DATA/stackoverflow_data_ranking_title_all_bert_base.tar.gz -C $DATA/
fi

# get precomputed embeddings for stackoverflow ranking, for BERT
if [ -d $DATA/stackoverflow_data_ranking_title_all_roberta_base ]; then
    echo "using stackoverflow_data_ranking_title_all_roberta_base"
elif [ -f $DATA/stackoverflow_data_ranking_title_all_roberta_base.tar.gz ]; then
    echo "untar stackoverflow_data_ranking_title_all_roberta_base"
    tar -xzf $DATA/stackoverflow_data_ranking_title_all_roberta_base.tar.gz -C $DATA/
else
    echo "download stackoverflow_data_ranking_title_all_roberta_base"
    wget https://archive.org/download/merge-15-22.2.format/stackoverflow_data_ranking_title_all_roberta_base.tar.gz -P $DATA/
    tar -xzf $DATA/stackoverflow_data_ranking_title_all_roberta_base.tar.gz -C $DATA/
fi

# get precomputed embeddings for stackoverflow ranking, for USE
if [ -d $DATA/stackoverflow_data_ranking_title_all_USE ]; then
    echo "using stackoverflow_data_ranking_title_all_USE"
elif [ -f $DATA/stackoverflow_data_ranking_title_all_USE.tar.gz ]; then
    tar -xzf $DATA/stackoverflow_data_ranking_title_all_USE.tar.gz -C $DATA/
    echo "untar stackoverflow_data_ranking_title_all_USE"
else
    echo "download stackoverflow_data_ranking_title_all_USE"
    wget https://archive.org/download/merge-15-22.2.format/stackoverflow_data_ranking_title_all_USE.tar.gz -P $DATA/
    tar -xzf $DATA/stackoverflow_data_ranking_title_all_USE.tar.gz -C $DATA/
fi




PYTHONPATH=.. python rank_answers.py $DATA/stackoverflow_data_ranking.json roberta-base-nli-mean-tokens $DATA/stackoverflow_data_ranking_title_all_roberta_base/  > /tmp/ranking_metrics_roberta.out
PYTHONPATH=.. python rank_answers.py $DATA/stackoverflow_data_ranking.json bert-base-nli-mean-tokens $DATA/stackoverflow_data_ranking_title_all_bert_base/  > /tmp/ranking_metrics_bert.out
PYTHONPATH=.. python rank_answers.py $DATA/stackoverflow_data_ranking.json https://tfhub.dev/google/universal-sentence-encoder/4 $DATA/stackoverflow_data_ranking_title_all_USE/  > /tmp/ranking_metrics_use.out

cat /tmp/ranking_metrics_bert.out
echo "--------------------------------"
cat /tmp/ranking_metrics_roberta.out
echo "--------------------------------"
cat /tmp/ranking_metrics_use.out
echo "--------------------------------"

