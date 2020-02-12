import json
from nltk.tokenize import word_tokenize
import re
from nltk.corpus import stopwords
from gensim.models.doc2vec import Doc2Vec, TaggedDocument


# this function cleans and prints out the contents of the first stack
# overflow post for a module
def clean_text(file):
	stopset = set(stopwords.words('english'))
	documentation = json.load(open(file, encoding = 'utf-8'))
	text = documentation['stackoverflow'][0]['_source']['content']
	# legacy code below, preserving in case we need it
	'''text = re.sub(r'<p>', r'', text)
	text = re.sub(r'<\/p>', r'', text)
	text = re.sub(r'<(.*)>.*<\/\1*>', r'', text, flags=re.DOTALL)'''
	text = re.sub(r'<code>[^>]*</code>', r'', text, flags=re.DOTALL)
	text = re.sub(r'<[^>]+>', r'', text)
	tokenized_text = word_tokenize(text)
	final_text = [word for word in tokenized_text if word not in stopset]
	# for debugging purposes
	print("\nOutput:\n")
	print(final_text)

inputfile = input("Please enter name of json file you wish to parse.\n")
clean_text(inputfile)
