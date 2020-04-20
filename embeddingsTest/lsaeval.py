import nltk
import pickle
import os 
import numpy as np
from nltk.stem import WordNetLemmatizer
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#from soupclean import clean_text
from shortParseFiles import parse_text
from sklearn.decomposition import TruncatedSVD
#nltk.download('stopwords')
#nltk.download('punkt')
#nltk.download('wordnet')
#from nltk.corpus import stopwords
#wordnet_lemmatizer = WordNetLemmatizer()
#file=open('scipy.linalg.det.json')

dataTuple=parse_text()
print(dataTuple[1][0])
wordArr=[]
moduleArr=[]
lookupDict={}
lookdownDict={}
currentRow=0
j=0
interested_file_len=100
for i in files:
   moduleArr.append(i.replace(jsonFileDir,""))
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
