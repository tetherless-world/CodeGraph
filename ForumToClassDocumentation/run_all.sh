#!/bin/sh

wget -nc https://ia801500.us.archive.org/24/items/stackoverflow_questions_per_class_func_3M_filtered_new/stackoverflow_questions_per_class_func_3M_filtered_new.json
wget -nc https://archive.org/download/classes2superclass/classes2superclass.out

python embedDocstringsEvaluateStackOverFlow-relevantBuckets.py stackoverflow_questions_per_class_func_3M_filtered_new.json 10 T 4
python allEmbedDocstringsEvaluateStackOverFlowAllMask.py stackoverflow_questions_per_class_func_3M_filtered_new.json 10
python embedDocstringsEvaluateStackOverFlowAllMask-lengthanalysis-newJson.py stackoverflow_questions_per_class_func_3M_filtered_new.json 10 
python allModelsembedDocstringsEvaluateStackOverFlow.py stackoverflow_questions_per_class_func_3M_filtered_new.json 10 T
python embedDocstringsEvaluateStackOverFlow-lengthanalysis-newJson.py stackoverflow_questions_per_class_func_3M_filtered_new.json 10 T
python embedDocstringsEvaluateStackOverFlow-relevantBuckets.py stackoverflow_questions_per_class_func_3M_filtered_new.json 10 F 4
python embedDocstringsEvaluateStackOverFlow-relevantBuckets.py stackoverflow_questions_per_class_func_3M_filtered_new.json 10 T 11
python embedDocstringsEvaluateStackOverFlow-relevantBuckets.py stackoverflow_questions_per_class_func_3M_filtered_new.json 10 F 11
python embedDocstringsEvaluateStackOverFlow-lengthanalysis-newJson.py stackoverflow_questions_per_class_func_3M_filtered_new.json 10 F
python allModelsembedDocstringsEvaluateStackOverFlow.py stackoverflow_questions_per_class_func_3M_filtered_new.json 10 F





