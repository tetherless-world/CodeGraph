import fivehrParallelParseFiles as sp
import pickleFiles as pf
import os.path as op
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
numofEpochs=300
##tune this later
if op.isfile('cleanedDocStringText.p'):
        print("Pickled data detected")
        dataList = pf.load_text("cleanedDocStringText.p")
        data = []
        print('Text loaded from pickled data')
        for i in range(0, len(dataList)):
                data.append(TaggedDocument(dataList[i][1], [dataList[i][0]]))
        print('List of data documents created.')
        model = Doc2Vec(vector_size=300,alpha=0.01,min_alpha=0.0001,min_count=2,dm=1, workers=48)
        print('Model initialized.')
        model.build_vocab(data)
        print('Vocab initialized. Training beginning.')
        model.train(data,total_examples=model.corpus_count,epochs=numofEpochs)
        model.save('docstringStressTestOutput.model')
        print("Model saved")
else:   
        print("No pickled data detected.")
