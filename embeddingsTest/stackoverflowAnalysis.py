import ijson
from bs4 import BeautifulSoup

def analyze_posts():
	with open('../../data/codeGraph/stackoverflow_questions_per_class_func_1M_.json', 'r') as data:
		docStringObjects = ijson.items(data, 'results.bindings.item')
		i = 0
		for docString in docStringObjects:
			if i == 2000:
				break
			classType = 'class' #placeholder
			if classType != 'class':
				continue
		 	classLabel = docString['class_func']['value'].replace('http://purl.org/twc/graph4code/python/', '') #placeholder
			title = docString['title']['value']	
			text = docString['content']['value']
			soup = BeautifulSoup(text, 'html.parser')
			for code in soup.find_all('code'):
				code.decompose()
			cleanedText = soup.get_text()
			maskedText = cleanedText.replace(classLabel, '')
			i += 1


if __name__ == '__main__':
	analyze_posts()
