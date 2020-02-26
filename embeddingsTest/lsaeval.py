import nltk
import pickle
import os 
import numpy as np
from nltk.stem import WordNetLemmatizer
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from soupclean import clean_text
from sklearn.decomposition import TruncatedSVD
#nltk.download('stopwords')
#nltk.download('punkt')
#nltk.download('wordnet')
#from nltk.corpus import stopwords
#wordnet_lemmatizer = WordNetLemmatizer()
#file=open('scipy.linalg.det.json')

#with open('scipy.linalg.det.json') as f:
 # data = json.load(f)
#listData=data['stackoverflow']
#contentCorpus=""
#for i in listData:
#    contentCorpus=contentCorpus+i['_source']['content']
#projectStopwords=stopwords.words('english')
#additionalStopWords=['<','>','/p','.']
#for i in additionalStopWords:
#    projectStopwords.append(i)
#tokens = nltk.tokenize.word_tokenize(contentCorpus)
#tokens = [wordnet_lemmatizer.lemmatize(t) for t in tokens] 
#tokens = [t for t in tokens if t not in projectStopwords]
#print(tokens)
jsonFileDir="../../../../data/data/datascience_stackexchange_graph_v2/all"
files=[]
for i in os.listdir(jsonFileDir):
    files.append(jsonFileDir+"/"+i)
    
#files=['numpy.linalg.linalg.inv.json','scipy.linalg.det.json','sklearn.linear_model.LinearRegression.json','sklearn.linear_model.logistic.softmax.json',
#'sklearn.metrics.regression.mean_squared_error.json',
#'sklearn.metrics.scorer.f1_score.json']
wordArr=[]
moduleArr=[]
lookupDict={}
lookdownDict={}
currentRow=0
j=0
interested_file_len=100
for i in files:
   moduleArr.append(i)
   if j == interested_file_len:
       break

   #print("this file is causing an error",i)
   try:
       words=clean_text(i)
   except:
       pass
   wordArr.append(words)
   for word in words:
       if word not in lookupDict:
          lookupDict[word]=currentRow
          currentRow=currentRow+1
          lookdownDict[currentRow-1]=word
   j=j+1
#X=np.zeros((currentRow,len(files)))
##documentrelevance
X=np.zeros((interested_file_len,currentRow))

for j in range(len(wordArr)):
    for word in wordArr[j]:
       # X[lookupDict[word],j]=X[lookupDict[word],j]+1
       ##document relevance
       X[j,lookupDict[word]]=X[j,lookupDict[word]]+1
pickle.dump( X, open( "save.x", "wb" ) )
pickle.dump(moduleArr,open("save.moduleArr","wb"))
#svd=TruncatedSVD(n_components=1)
#Z=svd.fit_transform(X)
#for i in range(len(Z)):
#    print(Z[i],lookdownDict[i])
