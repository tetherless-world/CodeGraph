#!/bin/bash
# fetch all data

cd $1

################ common ###############################
if [ -f merge-15-22.2.format.json ]; then
    echo "using merge-15-22.2.format.json"
else
    wget https://archive.org/download/merge-15-22.2.format/merge-15-22.2.format.json
fi

if [ -f class2superclasses.csv ]; then
    echo "using class2superclasses.csv"
else
    wget https://archive.org/download/merge-15-22.2.format/class2superclasses.csv
fi

if [ -f classes.map ]; then
    echo "using classes.map"
else
    wget https://archive.org/download/merge-15-22.2.format/classes.map
fi


################  usage task  ############################

if [[ -f usage.txt ]]; then
    echo "using usage.txt"
else
    wget https://archive.org/download/merge-15-22.2.format/usage.txt
fi


################stack link task###########################

if [ -f stackoverflow_data_ranking.json ]; then
    echo "using stackoverflow_data_ranking.json"
else
    https://archive.org/download/merge-15-22.2.format/stackoverflow_data_ranking.json.tar.gz 
    tar xzf stackoverflow_data_ranking.json.tar.gz 
fi


################hierarchy task############################

if [ -f classes.fail ]; then
    echo "using classes.fail"
else
    wget https://archive.org/download/merge-15-22.2.format/classes.fail
fi


############# for search task ##########################
# get search results
if [ -f stackoverflow_matches_codesearchnet_5k.json ]; then
    echo "using stackoverflow_matches_codesearchnet_5k.json"
else
    wget https://archive.org/download/merge-15-22.2.format/stackoverflow_matches_codesearchnet_5k.json.zip
    unzip stackoverflow_matches_codesearchnet_5k.json.zip
    rm stackoverflow_matches_codesearchnet_5k.json.zip
fi

# get precomputed embeddings for search, for USE
if [ -f stackoverflow_matches_codesearchnet_5k_USE.tar.gz ]; then
    echo "using stackoverflow_matches_codesearchnet_5k_USE"
else
    wget https://archive.org/download/merge-15-22.2.format/stackoverflow_matches_codesearchnet_5k_USE.tar.gz
fi

# get precomputed embeddings for search, for BERT
if [ -f stackoverflow_matches_codesearchnet_5k_bert_base.tar.gz ]; then
    echo "using stackoverflow_matches_codesearchnet_5k_bert_base"
else
    wget https://archive.org/download/merge-15-22.2.format/stackoverflow_matches_codesearchnet_5k_bert_base.tar.gz
fi

#get precomputed embeddings for search, for RoBERTa
if [ -f stackoverflow_matches_codesearchnet_5k_roberta_base.tar.gz ]; then
    echo "using stackoverflow_matches_codesearchnet_5k_roberta_base.tar.gz"
else
    wget https://archive.org/download/merge-15-22.2.format/stackoverflow_matches_codesearchnet_5k_roberta_base.tar.gz
fi

############# for stackoverflow ranking tasks ##########################
# get search results
if [ -f stackoverflow_data_ranking.json.tar.gz ]; then
    echo "stackoverflow_data_ranking.json.tar.gz"
else
    wget https://archive.org/download/merge-15-22.2.format/stackoverflow_data_ranking.json.tar.gz
fi

# get precomputed embeddings for stackoverflow ranking, for RoBERTa
if [ -f stackoverflow_data_ranking_title_all_bert_base.tar.gz ]; then
    echo "using stackoverflow_data_ranking_title_all_bert_base.tar.gz"
else
    wget https://archive.org/download/merge-15-22.2.format/stackoverflow_data_ranking_title_all_bert_base.tar.gz
fi

# get precomputed embeddings for stackoverflow ranking, for BERT
if [ -f stackoverflow_data_ranking_title_all_roberta_base.tar.gz ]; then
    echo "using stackoverflow_data_ranking_title_all_roberta_base.tar.gz"
else
    wget https://archive.org/download/merge-15-22.2.format/stackoverflow_data_ranking_title_all_roberta_base.tar.gz
fi

# get precomputed embeddings for stackoverflow ranking, for USE
if [ -f stackoverflow_data_ranking_title_all_USE.tar.gz ]; then
    echo "using stackoverflow_data_ranking_title_all_USE.tar.gz"
else
    wget https://archive.org/download/merge-15-22.2.format/stackoverflow_data_ranking_title_all_USE.tar.gz
fi
