## Stack Overflow Question and Answer Analysis

The script in this directory, totalStatsAnalysisPreEmbedded.py, is used to perform a few different analyses on Stack Overflow Question and Answer data using pre-computed embeddings.   

The script takes in a couple data sources that are downloadable from this project's content folder on the internet archive: https://archive.org/details/merge-15-22.2.format.  

Those files are:
- stackoverflow_data_ranking.json (a 500K dataset with stack overflow question and its corresonding answer data)
- Pre-computed embeddings for the above dataset. In particular, we pre-computed embeddings using BERT, RoBERTa and USE. 

```
wget https://archive.org/download/merge-15-22.2.format/stackoverflow_data_ranking_title_all_USE.tar.gz
wget https://archive.org/download/merge-15-22.2.format/stackoverflow_data_ranking_title_all_bert_base.tar.gz
wget https://archive.org/download/merge-15-22.2.format/stackoverflow_data_ranking_title_all_roberta_base.tar.gz
```


By default, the script will perform MRR, NDCG, and paired T test analyses given the pre-computed USE embedding above.

To run this task:
```
python totalStatsAnalysisPreEmbedded.py <path_to_stackoverflow_data_ranking.json> <language_model_name> <pre_computed_embeddings>
```

For example, to evaluate this task using BERT, one can use:
```
python totalStatsAnalysisPreEmbedded.py ./stackoverflow_data_ranking.json bert-base-nli-mean-tokens ./stackoverflow_data_ranking_title_all_bert_base/ 
```

