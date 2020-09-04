import ijson
import tensorflow_hub as hub
import faiss
import numpy as np
from bs4 import BeautifulSoup


def build_index_docs(docPath):
    embed = hub.load('https://tfhub.dev/google/universal-sentence-encoder/4')
    index = faiss.IndexFlatIP(512)
    classesToDocs = {}
    docsToClasses = {}
    embedList = {}

    with open(docPath, 'r') as data:
        jsonCollect = ijson.items(data, 'item')
        i = 0
        for jsonObject in jsonCollect:
            if 'class_docstring' not in jsonObject:
                continue
            className = jsonObject['klass']
            docStringText = jsonObject['class_docstring']

            soup = BeautifulSoup(docStringText, 'html.parser')
            for code in soup.find_all('code'):
                code.decompose()  # this whole block might be unnecessary
            docStringText = soup.get_text()

            if docStringText in docsToClasses:
                docClasses = docsToClasses[docStringText]

                if className in docClasses:
                    pass

                else:
                    docClasses.append(className)

            else:
                docsToClasses[docStringText] = [className]

            classesToDocs[className] = docStringText

    docList = np.array(list(docsToClasses.keys()))
    embeddedDocText = np.array(embed(docList))
    faiss.normalize_L2(embeddedDocText)
    index.add(embeddedDocText)

    return (index, docList, docsToClasses, embeddedDocText, classesToDocs)


def compute_neighbor_docstrings(query_neighbors, docList):
    docstringsToNeighbors = {}

    for docStringIndex, embeddedDocStringNeighbors in enumerate(query_neighbors):
        docString = docList[docStringIndex]

        neighborDocstrings = []
        for neighborDocStringIndex in embeddedDocStringNeighbors:
            neighborDocstrings.append(docList[neighborDocStringIndex])

        docstringsToNeighbors[docString] = neighborDocstrings

    return docstringsToNeighbors