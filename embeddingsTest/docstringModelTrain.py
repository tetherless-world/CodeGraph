import pickleFiles as pf
import os.path as op
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

##tune this later
hyperparameters=[{"numofEpochs":100,"vector_size":30,"min_count":5},{"numofEpochs":30,"vector_size":100,"min_count":2},{"numofEpochs":30,"vector_size":500,"min_count":2},{"numofEpochs":100,"vector_size":300,"min_count":3},{"numofEpochs":30,"vector_size":30,"min_count":10}]
for params in hyperparameters:
      if op.isfile('cleanedDocStringText.p'):
              print("Pickled data detected")
              dataList = pf.load_text("cleanedDocStringText.p")
              data = []
              print('Text loaded from pickled data')
              for i in range(0, len(dataList)):
                      data.append(TaggedDocument(dataList[i][1], [dataList[i][0]]))
              print('List of data documents created.')
              model = Doc2Vec(vector_size=params['vector_size'],alpha=0.01,min_alpha=0.0001,min_count=params['min_count'],dm=1, workers=96)
              print('Model initialized.')
              model.build_vocab(data)
              print('Vocab initialized. Training beginning.')
              model.train(data,total_examples=model.corpus_count,epochs=params['numofEpochs'])
              model.save('docstringStressTestOutput.model')
              print("Model saved")
      else:   
              print("No pickled data detected.")
                                            
      
      
      model = Doc2Vec.load("docstringStressTestOutput.model")
      if op.isfile('cleanedDocStringText.p'):
        print("Pickled data detected")
        dataTuple = pf.load_text("cleanedDocStringText.p")

        #print("Embeddings for:",file=open("output_new/embeddings_output.txt", "a"))
        ##uncomment for vector outputs

        #for i in range(len(dataTuple)):

            #print("inferred vecto for")
          #  print(dataTuple[i][0],file=open("output_new/embeddings_output.txt", "a") )
            # inferred_vector = model.infer_vector(dataTuple[i][1])
            #print(inferred_vector,file=open("output_new/embeddings_output.txt", "a"))
        accuracy=0
        print("number of docs",len(dataTuple))
        iters=len(dataTuple)
        total_words=0
        for i in range(iters):
                inferred_vector = model.infer_vector(dataTuple[i][1])
                most_similar= model.docvecs.most_similar([inferred_vector],topn=1)
                total_words=total_words+len(dataTuple[i][1])
                if most_similar[0][0] == dataTuple[i][0]:
                    accuracy=accuracy+1
                else:
                  pass
                  # # print("Files not accurately embedded",dataTuple[i][0],file=open("output_new/embeddings_wrongly_done.txt","a"))
                  # print("\n predicted instead",most_similar[0][0],file=open("output_new/embeddings_wrongly_done.txt","a"))
        print("Training accuracy for embeddings using Doc2vec with hyperparameters vector_size="+str(params['vector_size'])+", min_count="+str(params['min_count'])+", numofEpochs"+str(params['numofEpochs']),accuracy/iters,file=open("output_new/embeddings_accuracy.txt","a"))
        print("Total Words in this model",total_words)
