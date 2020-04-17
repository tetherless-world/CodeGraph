
from bs4 import BeautifulSoup
import ijson
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string

# this is left commented here for reference
'''def clean_text(file):
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
        return final_text'''
if __name__ == '__main__':
	file = '/data/merge-15-22.2.format.json'
	with open(file, 'r') as data:
		docStringObjects = ijson.items(data, 'item')
		docMap = {}
		for docString in docStringObjects:
			if 'function_docstring' in docString:
				name = docString['module']
				if name not in docMap:
					docMap[name] = []
				docMap[name].append((docString['function'], docString['function_docstring']))
		print(docMap['httpretty'])
