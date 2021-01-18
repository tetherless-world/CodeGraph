python staticAnalysis_regression.py $DATA/merge-15-22.2.format.json $DATA/classes.map /data/blanca/usage_test.json finetuned /data/BERTOverflow-tuned/usage/data/BERTOverflow/-2021-01-12_15-28-39/0_Transformer/ > ./bert_overflow_tuned.results

python staticAnalysis_regression.py $DATA/merge-15-22.2.format.json $DATA/classes.map /data/blanca/usage_test.json bert /dev/null > ./bert.results

python staticAnalysis_regression.py $DATA/merge-15-22.2.format.json $DATA/classes.map /data/blanca/usage_test.json xlm /dev/null > ./xlm.results

python staticAnalysis_regression.py $DATA/merge-15-22.2.format.json $DATA/classes.map /data/blanca/usage_test.json msmacro /dev/null > ./msmarco.results

python staticAnalysis_regression.py $DATA/merge-15-22.2.format.json $DATA/classes.map /data/blanca/usage_test.json bertoverflow /data/BERTOverflow > ./bert_overflow.results

python staticAnalysis_regression.py $DATA/merge-15-22.2.format.json $DATA/classes.map /data/blanca/usage_test.json distilbert_para /dev/null > ./distilbert_para.results

python staticAnalysis_regression.py $DATA/merge-15-22.2.format.json $DATA/classes.map /data/blanca/usage_test.json USE /dev/null > ./USE.results
