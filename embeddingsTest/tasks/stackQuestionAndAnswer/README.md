## Stack Overflow Question and Answer Analysis

The script in this directory, rank_answers.py, is used to perform a few different analyses on Stack Overflow Question and Answer data using pre-computed embeddings.   

The script takes in the following dataset files (available at internet archive: https://archive.org/details/merge-15-22.2.format).  
- stackoverflow_data_ranking.json (a 500K dataset with stack overflow question and its corresonding answer data)
- Pre-computed embeddings for the above dataset. In particular, we pre-computed embeddings using BERT, RoBERTa and USE. 


The script `run.sh` downloads all required data files (if necessary) and run the answer ranking task. It takes as input the location where data will be downloaded; `/tmp` in the following example:
 ```
chmod +x ./run.sh /tmp/
./run.sh
```

It downloads a 500K questions dataset with its answers as well as pre-computed embeddings using BERT, RoBERTa and USE language models. 
Then, it calls `rank_answers.py` to rank the answers based on the each embedding type and report the standard ranking metrics MRR and NDCG. 
`

For example, the following shows how this task is run using BERT embeddings:
```
python rank_answers.py ./stackoverflow_data_ranking.json bert-base-nli-mean-tokens ./stackoverflow_data_ranking_title_all_bert_base/ 
```

This should produce:
```
MRR: standard error of the mean  0.000439046368059151
Mean reciprocal rank is: 0.5343055530751452
Calculating NDCG with model bert-base-nli-mean-tokens
NDCG: standard error of the mean  0.0002539788918591201
Average NDCG: 0.7661997598261399
Calculating T statistic with model bert-base-nli-mean-tokens
Result of T statistic calculation is: Ttest_relResult(statistic=318.6342868996597, pvalue=0.0)
Number of forum posts with stackoverflow links =  70999
Average number of links per post:  1.3670333384977253

```
