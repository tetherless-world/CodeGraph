#!/bin/sh

wget -nc https://ia801500.us.archive.org/24/items/stackoverflow_questions_per_class_func_3M_filtered_new/stackoverflow_questions_per_class_func_3M_filtered_new.json
wget -nc https://archive.org/download/classes2superclass/classes2superclass.out
python embedDocstringsEvaluateStackOverFlow-relevantBuckets.py stackoverflow_questions_per_class_func_3M_filtered_new.json 10 T 4
#The Stackoverflow post is embedded with One masking: The particular class associated with the stackoverflow post is 
#masked and then embedded and evaluated for precision of predicting the closest docstrings(after embedding), 
#for the model USE by Sliding window technique, each bucket/bag of the doclabel assigned to a fixed number of sentences = bucket/bag size (4), 
#with fixed nearest neighbors, since more than one vector created from the same doclabel, results can be misleading ,
#so instead more neighbors are calculated and the first 10 nearest unique neighbors are picked

python allEmbedDocstringsEvaluateStackOverFlowAllMask.py stackoverflow_questions_per_class_func_3M_filtered_new.json 10
#when all the relevant classes associated with the stackoverflow post (Across all rows) are masked and then embedded, 
#and evaluated for precision of predicting the closest docstrings(after embedding), for the models-Roberta and Bert (base)

python embedDocstringsEvaluateStackOverFlowAllMask-lengthanalysis-newJson.py stackoverflow_questions_per_class_func_3M_filtered_new.json 10 
#when all the relevant classes associated with the stackoverflow post (Across all rows) are masked and then embedded, and evaluated for
#precision of predicting the closest docstrings(after embedding), for the model USE

python allModelsembedDocstringsEvaluateStackOverFlow.py stackoverflow_questions_per_class_func_3M_filtered_new.json 10 T
No masking:The Stackoverflow post is embedded as-is. One masking: The particular class associated with the stackoverflow post is masked and then embedded and evaluated for precision of predicting the closest docstrings(after embedding), for the models: 'bert-base-nli-stsb-mean-tokens', 'bert-large-nli-stsb-mean-tokens','roberta-base-nli-stsb-mean-tokens','roberta-large-nli-stsb-mean-tokens', 'distilbert-base-nli-stsb-mean-tokens', corresponding output in stackNewJson_NoOrOneMask_'+model_name+'_.txt ##provide input file path of the above json for example ../../data/codeGraph/stackoverflow_questions_per_class_func_3M_filtered_new.json
python embedDocstringsEvaluateStackOverFlow-lengthanalysis-newJson.py stackoverflow_questions_per_class_func_3M_filtered_new.json 10 T
 #The Stackoverflow One masked: The particular class associated with the stackoverflow post is masked and then embedded and evaluated for 
 #precision of predicting the closest docstrings(after embedding), for the model USE 
 
python embedDocstringsEvaluateStackOverFlow-relevantBuckets.py stackoverflow_questions_per_class_func_3M_filtered_new.json 10 F 4
#The Stackoverflow post is embedded with no  masking: The  stackoverflow post is embedded and evaluated for precision of predicting the closest 
#docstrings(after embedding), 
#for the model USE by Sliding window technique, each bucket/bag of the doclabel assigned to a fixed number of sentences = bucket/bag size (4), 
#with fixed nearest neighbors, since more than one vector created from the same doclabel, results can be misleading ,
#so instead more neighbors are calculated and the first 10 nearest unique neighbors are picked

python embedDocstringsEvaluateStackOverFlow-relevantBuckets.py stackoverflow_questions_per_class_func_3M_filtered_new.json 10 T 11
#The Stackoverflow post is embedded with One masking: The particular class associated with the stackoverflow post is 
#masked and then embedded and evaluated for precision of predicting the closest docstrings(after embedding), 
#for the model USE by Sliding window technique, each bucket/bag of the doclabel assigned to a fixed number of sentences = bucket/bag size (11), 
#with fixed nearest neighbors, since more than one vector created from the same doclabel, results can be misleading ,
#so instead more neighbors are calculated and the first 10 nearest unique neighbors are picked

python embedDocstringsEvaluateStackOverFlow-relevantBuckets.py stackoverflow_questions_per_class_func_3M_filtered_new.json 10 F 11
#The Stackoverflow post is embedded with no  masking: The  stackoverflow post is embedded and evaluated for precision of predicting the closest 
#docstrings(after embedding), 
#for the model USE by Sliding window technique, each bucket/bag of the doclabel assigned to a fixed number of sentences = bucket/bag size (11), 
#with fixed nearest neighbors, since more than one vector created from the same doclabel, results can be misleading ,
#so instead more neighbors are calculated and the first 10 nearest unique neighbors are picked

python embedDocstringsEvaluateStackOverFlow-lengthanalysis-newJson.py stackoverflow_questions_per_class_func_3M_filtered_new.json 10 F
# The Stackoverflow post is embedded as-is and evaluated for precision of predicting the closest docstrings(after embedding), for the model USE 

python allModelsembedDocstringsEvaluateStackOverFlow.py stackoverflow_questions_per_class_func_3M_filtered_new.json 10 F

 #The Stackoverflow is embedded and evaluated for 
 #precision of predicting the closest docstrings(after embedding), for the model USE 





