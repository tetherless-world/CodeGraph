
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
			if 'module' in docString:
				name = docString['module']
				if name not in docMap:
					docMap[name] = []
				for element in docString:
					if 'docstring' in element:
						if docString[element] != None:
							docMap[name].append(docString[element])
						else:
							pass
	docItems = []
	for label, text in docMap.items():
		space = ' '
		combinedModuleText = space.join(text)
		docItems.append((label, combinedModuleText))
	#return docMap
	return docItems
			#print(docMap['httpretty'])
