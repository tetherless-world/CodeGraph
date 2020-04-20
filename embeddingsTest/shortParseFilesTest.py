
from mrjob.protocol import JSONValueProtocol
from mrjob.compat import jobconf_from_env
from bs4 import BeautifulSoup
import json
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
from mrjob.job import MRJob
#bconf_from_env('mapreduce.map.input.file')
from mrjob.step import MRStep
from soupclean import clean_text
import os
import sys
import tarfile


class parseAllFiles(MRJob):
    #INPUT_PROTOCOL = JSONValueProtocol 
    def steps(self):


        return [MRStep(mapper_raw=self.mapper_getAllFiles,reducer=self.reducerAsASorter)]
        #return [MRStep(mapper_raw=self.mapper_getAllFiles)]
        

    def mapper_getAllFiles(self, path, uri):
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
                                   yield (i,f)
                           except:
                            
                               yield ("file causing issue",i)

                          # yield (f, json.load(file,strict=False))




    def reducerAsASorter(self, key, tokens):

        # print("here")
        arr=[]   
        for i in tokens:
            arr.append(i)
        yield (key,arr)


if __name__ == '__main__':
    parseAllFiles.run()
    
