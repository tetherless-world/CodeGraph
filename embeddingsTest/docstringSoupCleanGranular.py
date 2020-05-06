
from bs4 import BeautifulSoup
import ijson
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string

# this is left commented here for reference
def clean_docstrings():
	stopset = set(stopwords.words('english'))
	file = '/data/merge-15-22.2.format.json'
	docMap = {}
	with open(file, 'r') as data:
		docStringObjects = ijson.items(data, 'item')	
		for docString in docStringObjects:
			if 'function_docstring' in docString:
				name = docString['function']
				if docString['function_docstring'] != None:
					if name not in docMap:
						docMap[name] = []
					docMap[name].append(docString['function_docstring'])
			elif 'class_docstring' in docString:
				name = docString['klass']
				if docString['class_docstring'] != None:
					if name not in docMap:
						docMap[name] = []
					docMap[name].append(docString['class_docstring'])
							
	docItems = []
	for label, text in docMap.items():
		space = ' '
		for thing in text:
			if thing == None:
				print(label)
		combinedModuleText = space.join(text)
		docItems.append((label, combinedModuleText))
	#return docMap
	return docItems
			#print(docMap['httpretty'])

if __name__ == '__main__':
	itemMap = clean_docstrings()
	print(itemMap[0][1])
	print(itemMap[0][0])
