import ijson
import tensorflow_hub as hub
import faiss
import numpy as np
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer

embed = None

def get_model(embed_type):
    global embed
    if embed:
        return embed
    if embed_type == 'USE':
        model_path = 'https://tfhub.dev/google/universal-sentence-encoder/4'
        embed = hub.load(model_path)
    elif embed_type == 'bert':
        model_path = 'bert-base-nli-stsb-mean-tokens'
        embed = SentenceTransformer(model_path)
    elif embed_type == 'roberta':
        model_path = 'roberta-base-nli-stsb-mean-tokens'
        embed = SentenceTransformer(model_path)
    elif embed_type == 'distilbert':
        model_path = 'distilbert-base-nli-stsb-wkpooling'
        embed = SentenceTransformer(model_path)
    return embed


def embed_sentences(sentences, embed_type):
    embed = get_model(embed_type)
    if embed_type == 'USE':
        sentence_embeddings = embed(sentences)
    else:
        sentence_embeddings = embed.encode(sentences)
    return sentence_embeddings


def build_index_docs(docPath, embedType):
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
    embeddedDocText = np.array(embed_sentences(docList, embedType))
    faiss.normalize_L2(embeddedDocText)
    index = faiss.IndexFlatIP(len(embeddedDocText[0]))
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
