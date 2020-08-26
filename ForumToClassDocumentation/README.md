
Configurations and how to run:

File name: embedDocstringsEvaluateStackOverFlow-relevantBuckets.py 


File description:
No masking: The Stackoverflow post is embedded as-is. One masking: The particular class associated with the stackoverflow post is masked and then embedded and evaluated for precision of predicting the closest docstrings(after embedding), for the model USE by Sliding window technique,  each bucket/bag of the doclabel assigned  to a fixed number of sentences = bucket/bag size, with fixed nearest neighbors, since more than one vector created from the same doclabel, results can be misleading , so instead more neighbors are calculated and the first k nearest  unique neighbors are picked


dataset at : https://ia801500.us.archive.org/24/items/stackoverflow_questions_per_class_func_3M_filtered_new/stackoverflow_questions_per_class_func_3M_filtered_new.json 

How to run:

python embedDocstringsEvaluateStackOverFlow-relevantBuckets.py ../../data/codeGraph/stackoverflow_questions_per_class_func_3M_filtered_new.json 5 T 4


provide input file path of the above dataset for  while running the code: 
->../../data/codeGraph/stackoverflow_questions_per_class_func_3M_filtered_new.json  is the location of the dataset
->where 5 is the number of nearest neighbors
-> T is one masked,F is no masked configuration
->4 is the bucket_size
->output1.txt and output2.txt files are output produced

-----------------------------------------------------------------------------------------------------------------------------------------------------


File name: allEmbedDocstringsEvaluateStackOverFlowAllMask.py


File description:
when all the relevant classes associated with the stackoverflow post (Across all rows) are masked and then embedded, and evaluated for precision of predicting the closest docstrings(after embedding), for the models:  'bert-base-nli-stsb-mean-tokens', 'bert-large-nli-stsb-mean-tokens','roberta-base-nli-stsb-mean-tokens','roberta-large-nli-stsb-mean-tokens', 'distilbert-base-nli-stsb-mean-tokens', corresponding output in stackNewJsonAllMask_'+model_name+'_.txt  
##provide input file path of the above json for example ../../data/codeGraph/stackoverflow_questions_per_class_func_3M_filtered_new.json

 dataset at https://ia801500.us.archive.org/24/items/stackoverflow_questions_per_class_func_3M_filtered_new/stackoverflow_questions_per_class_func_3M_filtered_new.json 


for example, python allEmbedDocstringsEvaluateStackOverFlowAllMask.py ../../data/codeGraph/stackoverflow_questions_per_class_func_3M_filtered_new.json 5

../../data/codeGraph/stackoverflow_questions_per_class_func_3M_filtered_new.json 
where 5 is the number of nearest neighbors

output files produced: output1.txt and  finalOut_'+model_name+'_.txt

-----------------------------------------------------------------------------------------------------------------------------------------------------

File name: embedDocstringsEvaluateStackOverFlowAllMask-lengthanalysis-newJson.py


File description:
when all the relevant classes associated with the stackoverflow post (Across all rows) are masked and then embedded, and evaluated for precision of predicting the closest docstrings(after embedding), for the model USE

 dataset at https://ia801500.us.archive.org/24/items/stackoverflow_questions_per_class_func_3M_filtered_new/stackoverflow_questions_per_class_func_3M_filtered_new.json 


for example, python embedDocstringsEvaluateStackOverFlowAllMask-lengthanalysis-newJson.py ../../data/codeGraph/stackoverflow_questions_per_class_func_3M_filtered_new.json 5
where 5 is the number of nearest neighbors


output files produced: output1.txt and  output2.txt

-----------------------------------------------------------------------------------------------------------------------------------------------------
