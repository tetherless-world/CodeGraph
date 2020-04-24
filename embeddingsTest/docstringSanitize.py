from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string

def sanitize_text(textTuple):
	text = textTuple[1]
	tokenized_text = word_tokenize(text)
	final_text = [word.lower() for word in tokenized_text if word not in stopset and word not in string.punctuation]
	return (textTuple[0], final_text)
