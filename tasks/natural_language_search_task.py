#!/usr/bin/env python
# coding: utf-8

# In[1]:


import ijson
import json
import tensorflow_hub as hub
import faiss
import numpy as np
from bs4 import BeautifulSoup
import sys
import re
from statistics import mean 
from statistics import pstdev
import nltk
nltk.download('punkt')
from sentence_transformers import SentenceTransformer
import copy

from nltk.tokenize import sent_tokenize


# In[14]:


embed = hub.load('https://tfhub.dev/google/universal-sentence-encoder/4')
index = faiss.IndexFlatL2(512)
docMessages = []
embeddingtolabelmap = {}
labeltotextmap = {}
labeltourl = {}
embedCollect = set()
duplicateClassDocString=set()
with open('stackoverflow_questions_per_class_func_3M_filtered_new.json', 'r') as data:
    jsonCollect = ijson.items(data, 'results.bindings.item')
    i = 0
    for jsonObject in jsonCollect:
        objectType = jsonObject['class_func_type']['value'].replace(
            'http://purl.org/twc/graph4code/ontology/', '')

        #comment out the line below for docstring analysis
        if objectType != 'Class':
            continue
        label = jsonObject['class_func_label']['value']
        docLabel = label
        #uncomment the line below for docstring analysis
        #docStringText = jsonObject['docstr']['value'] 
        docStringText = jsonObject['content']['value'] + " " + jsonObject['answerContent']['value']
        url = jsonObject['q']['value']
        
        soup = BeautifulSoup(docStringText, 'html.parser')
        for code in soup.find_all('code'):
            code.decompose()
        docStringText = soup.get_text()
        
        if docStringText in embedCollect:
            if docLabel in duplicateClassDocString:
                pass
            else:
                duplicateClassDocString.add(docLabel)
                embeddedDocText = embed([docStringText])[0]
                embeddingtolabelmap[tuple(embeddedDocText.numpy().tolist())].append(docLabel)
        else:
            duplicateClassDocString.add(docLabel)
            embedCollect.add(docStringText)
            embeddedDocText = embed([docStringText])[0]
            newText = np.asarray(embeddedDocText, dtype=np.float32).reshape(1, -1)
            docMessages.append(embeddedDocText.numpy().tolist())
            index.add(newText)
            embeddingtolabelmap[tuple(embeddedDocText.numpy().tolist())] = [docLabel]

            labeltotextmap[docLabel] = docStringText
            labeltourl[docLabel] = url
        i += 1


# In[15]:


data = [
'convert int to string',
'priority queue',
'string to date',
'sort string list',
'save list to file',
'postgresql connection',
'confusion matrix',
'set working directory',
'group by count',
'binomial distribution',
'aes encryption',
'linear regression',
'socket recv timeout',
'write csv',
'convert decimal to hex',
'export to excel',
'scatter plot',
'convert json to csv',
'pretty print json',
'replace in file',
'k means clustering',
'connect to sql',
'html encode string',
'finding time elapsed using a timer',
'parse binary file to custom class',
'get current ip address',
'convert int to bool',
'read text file line by line',
'get executable path',
'httpclient post json',
'get inner html',
'convert string to number',
'format date',
'readonly array',
'filter array',
'map to json',
'parse json file',
'get current observable value',
'get name of enumerated value',
'encode url',
'create cookie',
'how to empty array',
'how to get current date',
'how to make the checkbox checked',
'initializing array',
'how to reverse a string',
'read properties file',
'copy to clipboard',
'convert html to pdf',
'json to xml conversion',
'how to randomly pick a number',
'normal distribution',
'nelder mead optimize',
'hash set for counting distinct elements',
'how to get database table name',
'deserialize json',
'find int in string',
'get current process id',
'regex case insensitive',
'custom http error response',
'how to determine a string is a valid word',
'html entities replace',
'set file attrib hidden',
'sorting multiple arrays based on another arrays sorted order',
'string similarity levenshtein',
'how to get html of website',
'buffered file reader read text',
'encrypt aes ctr mode',
'matrix multiply',
'print model summary',
'unique elements',
'extract data from html content',
'heatmap from 3d coordinates',
'get all parents of xml node',
'how to extract zip file recursively',
'underline text in label widget',
'unzipping large files',
'copying a file to a path',
'get the description of a http status code',
'randomly extract x items from a list',
'convert a date string into yyyymmdd',
'convert a utc time to epoch',
'all permutations of a list',
'extract latitude and longitude from given input',
'how to check if a checkbox is checked',
'converting uint8 array to image',
'memoize to disk  - persistent memoization',
'parse command line argument',
'how to read the contents of a .gz compressed file?',
'sending binary data over a serial connection',
'extracting data from a text file',
'positions of substrings in string',
'reading element from html - <td>',
'deducting the median from each column',
'concatenate several file remove header lines',
'parse query string in url',
'fuzzy match ranking',
'output to html file',
'how to read .csv file in an efficient way?',
]


# In[16]:


querytolabel = {}
labelproperties = {}
labeltoproperties = {}

for query in data:
    i = 1
    embeddedText = embed([query])
    embeddingVector = embeddedText[0]
    embeddingArray = np.asarray(embeddingVector, dtype=np.float32).reshape(1, -1)
    
    D, I = index.search(embeddingArray, 5)
    for dist, i_index in zip(D[0], I[0]):
#         reconstructed = index.reconstruct(int(i_index))
#         embeddinglabel = embeddingtolabelmap[tuple(reconstructed.tolist())]
#         mainlabel = labeltotextmap[embeddinglabel[0]]
        embedding = docMessages[int(i_index)]
        adjustedembedding = tuple(embedding)
        mainlabel = embeddingtolabelmap[adjustedembedding]
        maintext = labeltotextmap[mainlabel[0]]

        #Use this to get the URL for the stackoverflow posts
        labelproperties['url'] = labeltourl[mainlabel[0]]
        #Use this to get the label for the docstring search
        #labelproperties['class_func_label'] = mainlabel[0]

        labelproperties['index'] = str(i_index)
        labelproperties['distance'] = str(dist)
        labelproperties['text'] = maintext
        labeltoproperties['label' + str(i)] = copy.deepcopy(labelproperties)
        labelproperties.clear()
        i = i + 1

    querytolabel[query] = copy.deepcopy(labeltoproperties)
    labeltoproperties.clear()


# In[17]:


res1 = dict(list(querytolabel.items())[:len(querytolabel)//3])
res2 = dict(list(querytolabel.items())[len(querytolabel)//3:2*len(querytolabel)//3])
res3 = dict(list(querytolabel.items())[2*len(querytolabel)//3:])

with open("results_corpus1.json", "w") as outfile:  
    json.dump(res1, outfile, indent = 4) 
with open("results_corpus2.json", "w") as outfile:  
    json.dump(res2, outfile, indent = 4) 
with open("results_corpus3.json", "w") as outfile:  
    json.dump(res3, outfile, indent = 4) 


# In[ ]:




 


# In[ ]:





# In[ ]:




