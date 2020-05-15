from bs4 import BeautifulSoup
import ijson
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import pickleFiles as pf

# this is left commented here for reference
def clean_docstrings():
	stopset = set(stopwords.words('english'))
	file = '../../data/merge-15-22.2.format.json'
	classDocStrings = {}
	with open(file, 'r', encoding ="utf-8") as data:
		docStringObjects = ijson.items(data, 'item')
		for docString in docStringObjects:
			if 'klass' in docString:
				if 'class_docstring' in docString:
					if 'class_docstring' != None:
						classDocStrings[docString['klass']] = docString['class_docstring']
	docMap = {}
	with open(file, 'r', encoding = 'utf-8') as data:
		docStringObjects = ijson.items(data, 'item')	
		for docString in docStringObjects:
			if docString['module'] != None:
				totalLabel = docString['module']
			else:
				totalLabel = 'noModule'
			className = 'noClass'
			functionName = 'noFunction'
			if 'klass' in docString:
				if docString['klass'] != None:
					className = docString['klass']	
			totalLabel = totalLabel + ' ' + className 
			if 'function' in docString:
				if docString['function'] != None:
					functionName = docString['function']
			totalLabel = totalLabel + ' ' + functionName
			totalText = '' 
			if className != 'noClass':
				totalText = totalText + className
			if functionName != 'noFunction':
				totalText = totalText + ' ' + functionName
			functionDocString = ''
			classDocString = ''
			if 'function_docstring' in docString:
				functionDocString = docString['function_docstring']
				if functionDocString != None:
					totalText = totalText + ' ' + functionDocString
			if className in classDocStrings:
				totalText = totalText + ' ' + classDocStrings[className]
			docMap[totalLabel] = totalText	
	docItems = []
	for label, text in docMap.items():
		for thing in text:
			if thing == None:
				print(label)	
		docItems.append((label, text))
	#return docMap
	return docItems
			#print(docMap['httpretty'])

if __name__ == '__main__':
	itemMap = clean_docstrings()
	print(itemMap[0][1])
	print(itemMap[0][0])
	pf.store_text(itemMap, 'docstringGranularExtendedText.p')
