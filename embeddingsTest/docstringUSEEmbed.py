from bs4 import BeautifulSoup
import ijson
import string
import pickleFiles as pf
import tensorflow as tf
import tensorflow_hub as hub

# this is left commented here for reference
def clean_docstrings():
    classDocStrings = {}
    with open('../../data/codeGraph/merge-15-22.2.format.json','rb') as data:
        docStringObjects = ijson.items(data, 'item')
        for docString in docStringObjects:
            if 'klass' in docString:
                if 'class_docstring' in docString:
                    if 'class_docstring' != None:
                        classDocStrings[docString['klass']] = docString['class_docstring']
    docMap = {}
    embeddingList = []
    embeddingMapping = {}
    print("Fetching embedding model.")
    embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
    print("Embedding model fetched.")
    with open('../../data/codeGraph/merge-15-22.2.format.json','rb') as data:
        docStringObjects = ijson.items(data, 'item')
        embeddedDocument = 0
        for docString in docStringObjects:
            embeddedDocument += 1
            if embeddedDocument % 100 == 0:
                print('Embedding', embeddedDocument, 'documents')
            if docString['module'] != None:
                totalLabel = docString['module']
            else:
                totalLabel = 'noModule'
            className = 'noClass'
    #         functionName = 'noFunction'
            if 'klass' in docString:
                if docString['klass'] != None:
                    className = docString['klass']  
            totalLabel = totalLabel + ' ' + className 
    #         if 'function' in docString:
    #             if docString['function'] != None:
    #                 functionName = docString['function']
    #         totalLabel = totalLabel + ' ' + functionName
            totalText = '' 
            if className != 'noClass':
                totalText = totalText + className
    #         if functionName != 'noFunction':
    #             totalText = totalText + ' ' + functionName
    #         functionDocString = ''
            classDocString = ''
    #         if 'function_docstring' in docString:
    #             functionDocString = docString['function_docstring']
    #             if functionDocString != None:
    #                 totalText = totalText + ' ' + functionDocString
            if className in classDocStrings:
                totalText = totalText + ' ' + classDocStrings[className]
            computedEmbedding = embed([totalText])
            embeddingVector = tuple(computedEmbedding[0].numpy().tolist())
            embeddingList.append(embeddingVector) 
            embeddingMapping[embeddingVector] = totalLabel
    return (embeddingList, embeddingMapping, classDocStrings)

if __name__ == '__main__':
    itemMap = clean_docstrings()
    firstembedding = itemMap[0][0]
    print(firstembedding)
    print(itemMap[1][firstembedding])
    print(itemMap[2]['httpretty.core.EmptyRequestHeaders'])
