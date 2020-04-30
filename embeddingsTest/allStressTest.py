import fivehrParallelParseFiles as sp
import pickleFiles as pf
import os.path as op
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
numofEpochs=300
##tune this later
if op.isfile('allPickledFiles.p'):
        print("Pickled data detected")
        dataTuple = pf.load_text("allPickledFiles.p")
        data = []
        for i in range(0, len(dataTuple[0])):
                data.append(TaggedDocument(dataTuple[0][i], [dataTuple[1][i]]))
        model = Doc2Vec(vector_size=300,alpha=0.01,min_alpha=0.0001,min_count=2,dm=1, workers=48)
        model.build_vocab(data)
        model.train(data,total_examples=model.corpus_count,epochs=numofEpochs)
        model.save('newStressTestOutput.model')
        print("Model saved")
else:   
        print("No pickled data detected.")
        dataTuple = sp.parse_text()
        data = []
        for i in range(0, len(dataTuple[0])):
                data.append(TaggedDocument(dataTuple[0][i], [dataTuple[1][i]]))
        model = Doc2Vec(vector_size=600,alpha=0.01,min_alpha=0.0001,min_count=3,dm=1, workers=96)
        model.build_vocab(data)
        model.train(data,total_examples=model.corpus_count,epochs=numofEpochs)
        model.save('newStressTestOutput.model')
        print("Model saved")
