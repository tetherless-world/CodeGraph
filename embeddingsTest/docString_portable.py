
import tensorflow as tf
import ijson
from absl import logging

import tensorflow as tf

import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import seaborn as sns
import faiss                   # make faiss available
import sys

def batch_embed_evaluate(input_path,output_path):

    print("input_path",input_path)
    docMap = {}
    classDocStrings = {}
    with open(input_path,'rb') as data:
        docStringObjects = ijson.items(data, 'item')
        for docString in docStringObjects:
            if 'klass' in docString:
                if 'class_docstring' in docString:
                    if 'class_docstring' != None:
                        classDocStrings[docString['klass']] = docString['class_docstring']
    docMap = {}
    with open(input_path,'rb') as data:
        docStringObjects = ijson.items(data, 'item')
        for docString in docStringObjects:
            if docString['module'] != None:
                totalLabel = docString['module']
            else:
                totalLabel = 'noModule'
            className = 'noClass'
            if 'klass' in docString:
                if docString['klass'] != None:
                    className = docString['klass']
            totalLabel = totalLabel + ' ' + className

            totalText = ''
            if className != 'noClass':
                totalText = totalText + className

            classDocString = ''

            if className in classDocStrings:
                totalText = totalText + ' ' + classDocStrings[className]
            docMap[totalLabel] = totalText
    docItems = []
    for label, text in docMap.items():
        for thing in text:
            if thing == None:
                print(label)
        docItems.append((label, text))

    docStringCompiled= docItems
    #@title Load the Universal Sentence Encoder's TF Hub module



    def embed(input):
        module_url = "https://tfhub.dev/google/universal-sentence-encoder/4" #@param ["https://tfhub.dev/google/universal-sentence-encoder/4", "https://tfhub.dev/google/universal-sentence-encoder-large/5"]
        model = hub.load(module_url)
        return model(input)
    def randomize(x, y):
        ##Randomizes the order of data samples and their corresponding labels
        import numpy as np
        permutation = np.random.permutation(len(y))
        y=np.asarray(y)
        shuffled_x = x[permutation]
        shuffled_y = y[permutation]

        return shuffled_x, shuffled_y

    def get_next_batch(x, y, start, end):
        ##get the specified batch
        x_batch = x[start:end]
        y_batch = y[start:end]
        return x_batch, y_batch

    size_to_train=len(docStringCompiled) ##ive run for a max of 130000 till now
    batch_size=100
    embeded_docnames=[]
    embeded_docmessages=[]
    embeded_stringdocmessages=[]

    index = faiss.IndexFlatL2(512)
    shuffled_docnames,shuffled_docmessages=randomize(np.asarray([seq[0] for seq in docStringCompiled]),np.asarray([seq[1] for seq in docStringCompiled]))

    for i in range(int(size_to_train/batch_size)):
        print("iteration",i)
    # Reduce logging output.
        docStringsset1,docStringsset2=get_next_batch(shuffled_docnames,shuffled_docmessages,i*batch_size,(i+1)*batch_size)
        message_embeddings = embed(docStringsset2)
        message_embeddings=tf.make_tensor_proto(message_embeddings)
        message_embeddings=tf.make_ndarray(message_embeddings)
        embeded_docnames.append(docStringsset1)
        #embeded_docmessages.append(message_embeddings)
        #embeded_stringdocmessages.append(docStringsset2)
        index.add(message_embeddings)

    k=11
    embeded_distance_index_info=[]
    embeded_distance_info=[]
    for i in range(len(embeded_docmessages)):
    # we want to see 6 nearest neighborS
        D, I = index.search(embeded_docmessages[i], k)
        embeded_distance_index_info.append(I)
        embeded_distance_info.append(D)
    sys.stdout = open(output_path, "w")

    for i in range(len(embeded_docmessages)):
        for j in range(len(embeded_docmessages[i])):
            print("-------------------------------------------------------------")
            print("document name  : \n"+str(embeded_docnames[i][j])+"\n")
            for k in range(len(embeded_distance_index_info[i][j])):
                print("\n close to document :",shuffled_docnames[embeded_distance_index_info[i][j][k]])
                print("\n with a distance :",embeded_distance_info[i][j][k])


    sys.stdout.close()
if __name__ == '__main__':
    itemMap = batch_embed_evaluate(sys.argv[1:][0],sys.argv[1:][1])
    #print("job finished")
