from gensim.models.doc2vec import Doc2Vec
import os
import fivehrParseFiles as sp
import pickleFiles as pf
import os.path as op
from gensim.models.doc2vec import Doc2Vec, TaggedDocument



if __name__ == "__main__":
        jsonFileDir = '/data/data/datascience_stackexchange_graph_v2/all'
        files = [i for i in os.listdir(jsonFileDir) if i != 'first500']
        model = Doc2Vec.load("newStressTestOutput.model")
        if op.isfile('fivehrPickledFiles.p'):
           print("Pickled data detected")
           dataTuple = pf.load_text("fivehrPickledFiles.p")

           print("Embeddings for:",file=open("output_new/embeddings_output.txt", "a"))
           ##uncomment for vector outputs
           for i in range(len(dataTuple[0])):

               #print("inferred vecto for")
               print(dataTuple[1][i],file=open("output_new/embeddings_output.txt", "a") )
               inferred_vector = model.infer_vector(dataTuple[0][i])
               print(inferred_vector,file=open("output_new/embeddings_output.txt", "a"))
           accuracy=0
           for i in range(len(dataTuple[0])):
                   inferred_vector = model.infer_vector(dataTuple[0][i])
                   most_similar= model.docvecs.most_similar([inferred_vector],topn=1)
                   if most_similar[0][0] == dataTuple[1][i]:
                      accuracy=accuracy+1
                   else:
                    print("Files not accurately embedded",dataTuple[1][i],file=open("output_new/embeddings_wrongly_done.txt","a"))
                    print("\n predicted instead",most_similar[0][0],file=open("output_new/embeddings_wrongly_done.txt","a"))
           print("Training accuracy for embeddings using Doc2vec",accuracy/len(dataTuple[0]),file=open("output_new/embeddings_accuracy.txt","a"))
        			
			
