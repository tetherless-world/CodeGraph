
from bs4 import BeautifulSoup
import json
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
from mrjob.job import MRJob
from mrjob.step import MRStep
from soupclean import clean_text
import os
import sys


class parseAllFiles(MRJob):

    def steps(self):

        # return [MRStep(mapper=self.mapper_getAllFiles,reducer=self.reducerAsASorter)]

        return [MRStep(mapper=self.mapper_getAllFiles)]

    def mapper_getAllFiles(self, _, l):
        stopset = set(stopwords.words('english'))
        documentation = json.load(l)
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
        final_text = [word.lower() for word in tokenized_text if word
                      not in stopset and word not in string.punctuation]
        yield (map.input.file, final_text)

    def reducerAsASorter(self, key, tokens):

        # print("here")

        yield (key, tokens)



if __name__ == '__main__':
    parseAllFiles.run()
    #print parseAllFiles.run()

