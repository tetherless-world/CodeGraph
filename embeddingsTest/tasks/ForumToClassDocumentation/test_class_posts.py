
import json
import tensorflow_hub as hub
import numpy as np
from bs4 import BeautifulSoup
import sys
import re
import math, scipy
from sentence_transformers import SentenceTransformer
import random
import statistics
from scipy import stats as stat
import pickle
from utils.util import get_model, embed_sentences
# parse input data file and remove duplicates for analysis
# also calls all necessary analysis functions
import os
from pathlib import Path
import numpy as np

# def embed_sentences(sentences, model, embed_type ):
#     if embed_type == 'USE':
#         sentence_embeddings = model(sentences)
#     else:
#         sentence_embeddings = model.encode(sentences)
#     return sentence_embeddings

if __name__ == '__main__':
    dataSetPath = sys.argv[1]
    embed_type = sys.argv[2]
    model_path = sys.argv[3]
    #/tmp/true_bert.json /tmp/false_bert.json
    trueFileName = sys.argv[4]
    falseFileName = sys.argv[5]
    trues = []
    falses = []
    model = get_model(embed_type, model_path)
    with open(dataSetPath, 'r', encoding="UTF-8") as data_file:
        data = json.load(data_file)
        i = 0
        for jsonObject in data:
            srcEmbed = embed_sentences([jsonObject['docstring']], model, embed_type)
            dstEmbed = embed_sentences([jsonObject['text']], model, embed_type)
#            linkedDist = np.linalg.norm(srcEmbed - dstEmbed) ** 2
            from scipy.spatial import distance
            linkedDist = distance.cosine(srcEmbed, dstEmbed)
            
            if jsonObject['label'] == 1:
                trues.append(linkedDist)
            else:
                falses.append(linkedDist)

    print('number of relevant: ', len(trues), 'number of irrelevant: ', len(falses))

    with open(trueFileName, 'w') as trueFile:
        trueFile.write(json.dumps(trues, indent=2))

    with open(falseFileName, 'w') as falseFile:
        falseFile.write(json.dumps(falses, indent=2))

    print(np.mean(np.asarray(trues)))
    print(np.mean(np.asarray(falses)))
    
    print('Total number of samples = ', len(data))
    print(scipy.stats.ttest_ind(trues, falses))

