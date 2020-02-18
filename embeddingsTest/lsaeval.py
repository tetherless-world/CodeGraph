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
tokens=clean_text('scipy.linalg.det.json')
print(tokens)


