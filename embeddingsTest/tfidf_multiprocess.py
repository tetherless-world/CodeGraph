
from bs4 import BeautifulSoup
import json
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
from soupclean import clean_text
import os
import sys
import tarfile
from math import log
from multiprocessing import Pool
def getAllText(path):
    word_file_arr=[]
    for f in os.listdir(path): 
                        with open(path+"/"+f) as file:
                          # yield (f, "file causing issue")
                           try:
                               j=json.load(file,strict=False)
                               if  f is not  None and j is not None:

                                    stopset = set(stopwords.words('english'))
                                    documentation = j
                                    text = ''
                                    for section in documentation['stackoverflow']:
                                        plaintext = section['_source']['content']
                                        text += ' ' + plaintext
                                    if not text:
                                       return ''
                                    soup = BeautifulSoup(text, 'html.parser')
                                    for code in soup.find_all('code'):
                                              code.decompose()
                                    tokenized_text = word_tokenize(soup.get_text())
                                    final_text = [word.lower() for word in tokenized_text if word not in stopset and word not in string.punctuation]
                                    for i in final_text:
                                       word_file_arr.append((i,f))
                           except:
                              pass
    return word_file_arr
if __name__ == "__main__" :
    workerPool = Pool(48)
    jsonFileDir = "../../../../data/data/datascience_stackexchange_graph_v2/all/first500"
    
    #files.sort()
    #fullFiles = [jsonFileDir +'/' + target for target in files]
    cleanedTexts = workerPool.map(getAllText, jsonFileDir)
    print(cleanedTexts)                           
                            
