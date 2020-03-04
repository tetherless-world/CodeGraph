from sklearn.feature_extraction.text import TfidfVectorizer
import shortParseFiles as sp

def compute_tfidf():
    text = sp.parse_text()
    textcollection = text[0]
    corpus = []
    for module in textcollection:
        newsent = ''
        for word in module:
              newsent += word + ' '
        corpus.append(newsent)
    vectorizer = TfidfVectorizer()
    fitted = vectorizer.fit_transform(corpus)
     #the prints below are just for demonstration purposes,
     #feel free to remove them.
    print(vectorizer.get_feature_names()[0])
    print(vectorizer.get_feature_names()[5683])
    print(fitted.shape)
    return fitted
