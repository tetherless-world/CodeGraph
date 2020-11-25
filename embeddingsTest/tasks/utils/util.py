import ijson
import tensorflow_hub as hub
import faiss
import numpy as np
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer, models

embed = None

def get_model(embed_type):
    global embed
    if embed:
        return embed
    if embed_type == 'USE':
        model_path = 'https://tfhub.dev/google/universal-sentence-encoder/4'
        embed = hub.load(model_path)
    elif embed_type == 'bertoverflow':
        model_path = '/data/BERTOverflow'
        word_embedding_model = models.Transformer(model_path, max_seq_length=256)
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
        embed = SentenceTransformer(modules=[word_embedding_model, pooling_model])
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


def build_index_docs(docPath, embedType, valid_classes=None, generate_dict=False):
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
            if valid_classes and className not in valid_classes:
                continue
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

    if generate_dict:
        doc2embedding = {}
        for index, doc in enumerate(docList):
            doc2embedding[doc] = embeddedDocText[index]
        return (index, docList, docsToClasses, embeddedDocText, classesToDocs, doc2embedding)
    else:
        return (index, docList, docsToClasses, embeddedDocText, classesToDocs)


def compute_neighbor_docstrings(query_neighbors, docList):
    docstringsToNeighbors = {}

    for docStringIndex, embeddedDocStringNeighbors in enumerate(query_neighbors):
        docString = docList[docStringIndex]

        i = 0
        neighborDocstrings = []
        for neighborDocStringIndex in embeddedDocStringNeighbors:
            if i != 0:
                neighborDocstrings.append(docList[neighborDocStringIndex])
            i = i + 1

        docstringsToNeighbors[docString] = neighborDocstrings
        
    return docstringsToNeighbors
