from bs4 import BeautifulSoup
import ijson
import string
import pickleFiles as pf
import tensorflow as tf
import tensorflow_hub as hub

# this is left commented here for reference
def clean_docstrings():
    classDocStrings = {}
    docMap = {}
    embeddingList = []
    embeddingMapping = {}
    print("Fetching embedding model.")
    embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
    print("Embedding model fetched.")
    with open('../../data/codeGraph/merge-15-22.2.format.json','rb') as data:
        docStringObjects = ijson.items(data, 'item')
        parsedLines = 0
        for docString in docStringObjects:
            parsedLines += 1
            if parsedLines % 1000 == 0:
                print('Parsed', parsedLines, 'lines') 
            if 'class_docstring' in docString:
                if 'class_docstring' != None:
                    text = docString['class_docstring']
                    className = docString['klass']
                    totalText = className + ' ' + text
                    computedEmbedding = embed([totalText])
                    embeddingVector = tuple(computedEmbedding[0].numpy().tolist())
                    embeddingList.append(embeddingVector) 
                    embeddingMapping[embeddingVector] = className
                    classDocStrings[className] = totalText
                else:
                    print("Class docstring empty, skipped")
    return (embeddingList, embeddingMapping, classDocStrings)

if __name__ == '__main__':
    itemMap = clean_docstrings()
    firstembedding = itemMap[0][0]
    print(firstembedding)
    print(itemMap[1][firstembedding])
    print(itemMap[2]['httpretty.core.EmptyRequestHeaders'])
