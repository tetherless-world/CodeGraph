import ijson
from bs4 import BeautifulSoup
import re
import tensorflow_hub as hub
import faiss
import pickle
import numpy as np
import sys
def analyze_posts():
        with open('../../data/codeGraph/stackoverflow_questions_per_class_func_1M_filtered.json', 'r') as data:
                docStringObjects = ijson.items(data, 'results.bindings.item')
                originalOut = sys.stdout
                with open('../../data/codeGraph/docstringplusAllstackOverFlowSimilarityAnalysis.txt', 'w') as outputFile:
                        sys.stdout=outputFile
                        print("Fetching embedding model.")
                        embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
                        print("Embedding model fetched.")
                        totalDocs = 0
                        correctMatches = 0
                        sumdist=0
                        computedDistance=0
                        for docString in docStringObjects:
                                totalDocs += 1
                                classType = docString['class_func_type']['value'].replace('http://purl.org/twc/graph4code/ontology/', '')
                                if classType != 'Class':
                                        continue
                                computedDistance+=1
                                classLabel = docString['class_func_label']['value']
                                title = docString['title']['value']
                                text = docString['content']['value']
                                text += " " + docString['answerContent']['value']
                                soup = BeautifulSoup(text, 'html.parser')
                                for code in soup.find_all('code'):
                                        code.decompose()
                                cleanedText = soup.get_text()
                                splitLabel = classLabel.lower().split('.')
                                wholePattern = re.compile(classLabel.lower(), re.IGNORECASE)
                                maskedText = wholePattern.sub(' ', cleanedText)#cleanedText.replace(classLabel, ' ')
                                for labelPart in splitLabel:
                                        partPattern = re.compile(labelPart, re.IGNORECASE)
                                        maskedText = partPattern.sub(' ', maskedText)#maskedText.replace(labelPart, ' ')
                                embeddedText = embed([maskedText])
                                embeddingVector = embeddedText[0]
                                embeddingArray = np.asarray(embeddingVector, dtype=np.float32).reshape(1, -1)
                                docstringText=docString['docstr']['value']
                                embeddedDS = embed([docstringText])
                                embeddingVectorDS = embeddedDS[0]
                                embeddingArrayDS= np.asarray(embeddingVectorDS, dtype=np.float32).reshape(1, -1)
                                dist = np.linalg.norm(embeddingArrayDS-embeddingArray)
                                sumdist+=dist
                                print('\n---------------------------------')
                                print('\nTitle of Stack Overflow Post:', title)
                                print('\nOriginal Text of Stack Overflow Post:', cleanedText)
                                print('\nClass associated with Post:', classLabel)
                                print("\n The euclidean distance between the two vectors corresponding to this class for stackoverflow and docstring embeddings is", dist)
                        print("Average embedded euclidean distance between the  two vectors corresponding to this class for stackoverflow and docstring embeddings is:",sumdist/computedDistance)
                        sys.stdout = originalOut


if __name__ == '__main__':
	analyze_posts()
