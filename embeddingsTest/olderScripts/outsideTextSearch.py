import tensorflow as tf
import tensorflow_hub as tf_hub
import pickle
import faiss
import numpy as np

def compute_neighbors_in_index(inputFile):
        print('Fetching embedding model.')
        embed = tf_hub.load('https://tfhub.dev/google/universal-sentence-encoder/4')
        print("Embedding model fetched.")
        textsToCompare = []
        embedded_docmessages = []
        embeddingtolabelmap = {}
        labeltotextmap = {}
        duplicateDocToText= {}
        index = faiss.read_index('faiss_index.saved')
        with open('embeded_docmessages.pickle', 'rb') as inMessages:
                embedded_docmessages = pickle.load(inMessages)
        with open('embeddingtolabelmap.pickle', 'rb') as inEmbedding:
                embeddingtolabelmap = pickle.load(inEmbedding)
        with open('labeltotextmap.pickle', 'rb') as inText:
                labeltotextmap = pickle.load(inText)
        with open('duplicate_documents.pickle', 'rb') as inDuplicateDocs:
                duplicateDocToText=pickle.load(inDuplicateDocs)
        k = 11
        # the input file is expected to contain any number of queries,
        # separated by having each query begin on a new line
        with open(inputFile, 'r') as dataIn:
                for line in dataIn:
                        embeddedText = embed([line])
                        embeddingVector = embeddedText[0]
                        embeddingArray = np.asarray(embeddingVector, dtype = np.float32).reshape(1,-1)
                        D, I = index.search(embeddingArray, 11)
                        distances = D[0]
                        indices = I[0]
                        print('\n--------------------------------------')
                        print('\nText to be analyzed is:', line)
                        print('\nIndices of related vectors:', indices)
                        print('Distances to each related vector:', distances)
                        for i in range(0, k):
                                properIndex = indices[i]
                                embedding = embedded_docmessages[properIndex]
                                adjustedembedding = tuple(embedding.tolist())
                                label = embeddingtolabelmap[adjustedembedding]
                                print('\nName of document in ranking order', i, 'is:', label)
                                print('\nText of document', i, 'is:', labeltotextmap[label])
                similar = duplicateDocToText[labeltotextmap[label]]
                print('\nThis document also has similarities to class:', similar)

if __name__ == '__main__':
        compute_neighbors_in_index('sampleText.txt')

