
from bs4 import BeautifulSoup
import json
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string

def clean_text(file):
        stopset = set(stopwords.words('english'))
        try:
           documentation = json.load(open(file, encoding = 'utf-8'))
        except json.JSONDecodeError:
           return []
        text = ''
        if type(documentation) is not dict:
            return []
        for section in documentation['stackoverflow']:
                plaintext = section['_source']['content']
                text += ' ' + plaintext
        if not text:
                   return []
        soup = BeautifulSoup(text, 'html.parser')
        for code in soup.find_all('code'):
                code.decompose()
        tokenized_text = word_tokenize(soup.get_text())
        final_text = [word.lower() for word in tokenized_text if word not in stopset and word not in string.punctuation]
        return final_text
