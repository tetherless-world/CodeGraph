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
                i = 0
                k = 11
                #print("print working")
                embed = hub.load('https://tfhub.dev/google/universal-sentence-encoder/4')
                loadedIndex = faiss.read_index('faiss_index.saved')
                with open('embeded_docmessages.pickle', 'rb') as inMessages:
                        embedded_docmessages = pickle.load(inMessages)
                with open('embeddingtolabelmap.pickle', 'rb') as inEmbedding:
                        embeddingtolabelmap = pickle.load(inEmbedding)
                with open('labeltotextmap.pickle', 'rb') as inText:
                        labeltotextmap = pickle.load(inText)
                with open('duplicate_documents.pickle', 'rb') as inDuplicateDocs:
                        duplicateDocToText=pickle.load(inDuplicateDocs)
                originalOut = sys.stdout
                with open('../../data/codeGraph/UpdatedAllstackOverFlowSimilarityAnalysis.txt', 'w') as outputFile:
                        sys.stdout=outputFile
                        # IMPORTANT NOTE: I have no clue why they are called docstrings,
                        # must have been silly mistake on my part they really represent
                        # stack overflow posts
                        totalDocs = 0
                        correctMatches = 0
                        for docString in docStringObjects:
                                totalDocs += 1
                                classType = docString['class_func_type']['value'].replace('http://purl.org/twc/graph4code/ontology/', '')
                                if classType != 'Class':
                                        continue
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
                                D, I = loadedIndex.search(embeddingArray, k)
                                distances = D[0]
                                indices = I[0]
                                print('\n---------------------------------')
                                print('\nTitle of Stack Overflow Post:', title)
                                print('\nOriginal Text of Stack Overflow Post:', cleanedText)
                                print('\nClass associated with Post:', classLabel)
                                print('\nIndices of related vectors:', indices)
                                print('Distances to each related vector:', distances)

                                labelCollect = []
                                for p in range(0, k):
                                        properIndex = indices[p]
                                        embedding = embedded_docmessages[properIndex]
                                        adjustedembedding = tuple(embedding.tolist())
                                        label = embeddingtolabelmap[adjustedembedding]
                                        labelCollect.append(label)
                                        print('\nName of document in ranking order', p, 'is:', label)
                                        print('\nText of document', p, 'is:', labeltotextmap[label])
                                        if labeltotextmap[label] in duplicateDocToText:
                                             similar = duplicateDocToText[labeltotextmap[label]]
                                             print('\nThis document also has similarities to class:', similar)
                                if classLabel in labelCollect:
                                     correctMatches += 1  
                        print("\nTotal Number of Posts Parsed:", totalDocs)
                        print("Number of Posts Matched With Docstring:", correctMatches)
                        print("Percentage of Posts Correctly Matched (Correct/Total)", (correctMatches/totalDocs) * 100)
                        sys.stdout = originalOut


if __name__ == '__main__':
	analyze_posts()
