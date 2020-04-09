import unrelatedParseFiles as up
from gensim.models.doc2vec import Doc2Vec

if __name__ == "__main__":
	model = Doc2Vec.load("stressTestOutput.model")
	similar_doc = model.docvecs.most_similar('tensorflow.python.add.json','', 10)
	print(similar_doc)
