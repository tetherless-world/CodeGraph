
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
from math import log


class parseAllFiles(MRJob):
    #INPUT_PROTOCOL = JSONValueProtocol 
    def steps(self):

         #return  [MRStep(mapper_raw=self.mapper_getAllFiles,reducer=self.reducerAsTermFrequency),MRStep(mapper=self.mapper_to_find_words_in_a_doc,reducer=self.reducer_to_aggregate_words_in_a_doc),
                                  # MRStep(mapper=self.mapper_to_numberofwordsforidf)]
        return [MRStep(mapper_raw=self.mapper_getAllFiles,reducer=self.reducerAsTermFrequency),MRStep(mapper=self.mapper_to_find_words_in_a_doc,reducer=self.reducer_to_aggregate_words_in_a_doc),
              MRStep(mapper=self.mapper_to_numberofwordsforidf,reducer=self.reducer_to_numberofwordsforidf),MRStep(mapper=self.mapper_tfidf_compute)]
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
                                   yield (f,i,len(final_text)),1
                           except:

                               yield (f,"",1),0

                          # yield (f, json.load(file,strict=False))
##files to words, length of relevant words and 1 to finally coiunt everything


    def reducerAsTermFrequency(self,doc_word_doclen,  count):

        # unique doc, word combination is injured
        yield (doc_word_doclen[0],doc_word_doclen[1],doc_word_doclen[2]),sum(count)
   
    def mapper_to_find_words_in_a_doc(self,doc_word_doclen,count):
        yield doc_word_doclen[0],(doc_word_doclen[1],doc_word_doclen[2],count)

    def reducer_to_aggregate_words_in_a_doc(self,doc,word_doclen_count):
        total_words=0
        count_arr=[]
        doc_len_arr=[]
        word=[]
        for iter in word_doclen_count:
            total_words += iter[2]
            count_arr.append(iter[2])
            word.append(iter[0])
            doc_len_arr.append(iter[1])
        denominator_for_tf=[total_words]*len(word)
        #for i in range(len(word)):
         #   denominator_for_tf.append(total_words)
        ##for each document, all words and then the 
        for iter in range(len(word)):
            yield (word[iter],doc,doc_len_arr[iter]),(count_arr[iter],denominator_for_tf[iter])



    def mapper_to_numberofwordsforidf(self,word_doc_doclen,count_denominator):

    ##broadcast multiple words so the reducer can take care of
        yield word_doc_doclen[0],(word_doc_doclen[1],count_denominator[0],count_denominator[1],word_doc_doclen[2],1)
    def reducer_to_numberofwordsforidf(self,word,doc_doclen_count_denominator_idfcount):
        total_documents=0
        docs=[]
        doclengths=[]
        denominators=[]
        counts=[]
        for val in doc_doclen_count_denominator_idfcount:
            total_documents=total_documents+1
            docs.append(val[0])
            doclengths.append(val[3])
            counts.append(val[1])
            denominators.append(val[2])
        ##each document info is maintained

        denominatoridf=[total_documents]*total_documents
        ##transmit these words for the correspondign document
        for iter in range(total_documents):
            yield (word,docs[iter],doclengths[iter]),(counts[iter],denominators[iter],denominatoridf[iter])

    def mapper_tfidf_compute(self,word_docs_doclengths,c_d_di):
        if c_d_di[0]== 0 or c_d_di[1]== 0 or word_docs_doclengths[2] == 0 or c_d_di[2] == 0:
            yield (word_docs_doclengths[0],word_docs_doclengths[1]),0
        else:
            yield (word_docs_doclengths[0],word_docs_doclengths[1]),(c_d_di[0]/c_d_di[1])*log(500/c_d_di[2])
        ##return word, document and tfidf
if __name__ == '__main__':
            parseAllFiles.run()


