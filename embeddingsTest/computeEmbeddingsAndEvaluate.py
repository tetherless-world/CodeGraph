import unrelatedParseFiles as up
from gensim.models.doc2vec import Doc2Vec
import os
import fivehrParseFiles as sp
import pickleFiles as pf
import os.path as op
from gensim.models.doc2vec import Doc2Vec, TaggedDocument



if __name__ == "__main__":
	jsonFileDir = '/data/data/datascience_stackexchange_graph_v2/all/first500'
	files = [i for i in os.listdir(jsonFileDir) if i != 'first500']
	model = Doc2Vec.load("stressTestOutput.model")
	if op.isfile('fivehrPickledFiles.p'):
	   print("Pickled data detected")
	   dataTuple = pf.load_text("fivehrPickledFiles.p")
           #print("Embeddings for:",file=open("embeddings_output.txt", "a"))
	   for i in range(len(dataTuple[0])):
               
               #print("inferred vecto for")
               print(dataTuple[1][i],file=open("embeddings_output.txt", "a") )
               inferred_vector = model.infer_vector(dataTuple[0][i])
               print(inferred_vector,file=open("embeddings_output.txt", "a"))

