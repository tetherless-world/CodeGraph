import unrelatedParseFiles as up
from gensim.models.doc2vec import Doc2Vec
import os

if __name__ == "__main__":
	jsonFileDir = '/data/data/datascience_stackexchange_graph_v2/all/'
	files = [i for i in os.listdir(jsonFileDir) if i != 'first500']
	model = Doc2Vec.load("stressTestOutput.model")
	for file in files:
		print("Documents most similar to", file, "are:")
		similarities = model.docvecs.most_similar(file, '', 10)
		for i in range(0, 5):
			print('\t', str(i+1) + '.', similarities[i][0])
