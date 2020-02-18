import nltk 
import numpy as np
from nltk.stem import WordNetLemmatizer
import json
from testscript import clean_text
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
files=['numpy.linalg.linalg.inv.json','scipy.linalg.det.json']
wordArr=[]
moduleArr=[]
lookupDict={}
lookdownDict={}
currentRow=0
for i in files:
   moduleArr.append(i)
   words=clean_text(i)
   wordArr.append(words)
   for word in words:
       if word not in lookupDict:
          lookupDict[word]=currentRow
          currentRow=currentRow+1
          lookdownDict[currentRow-1]=word

X=np.zeros((currentRow,len(files)))
for j in range(len(wordArr)):
    for word in wordArr[j]:
        X[lookupDict[word],j]=X[lookupDict[word],j]+1
svd=TruncatedSVD(n_components=1)
Z=svd.fit_transform(X)
for i in range(len(Z)):
    print(Z[i],lookdownDict[i])
       
       
       



