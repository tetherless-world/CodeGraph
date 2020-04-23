
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
							'''print("This one is perfectly fine.")
							input()'''
							docMap[name].append(docString[element])
						else:
							'''print("Something is wrong with this text element.")
							print("The element is", element)
							print("The module name is", name)
							print("The element contents are", docString[element])
							print(docString)'''
			#print(docMap['httpretty'])
	for module, moduleTextList in docMap.items():
		space = ' '
#		print(moduleTextList)
		combinedModuleText = space.join(moduleTextList)
		tokenized_text = word_tokenize(combinedModuleText)
		final_text = [word.lower() for word in tokenized_text if word not in stopset and word not in string.punctuation]
		docMap[module] = final_text

	print(docMap['httpretty'])
