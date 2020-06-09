import faiss 
import csv
import numpy as np
if __name__ == '__main__':
        embeddingspath = '../../data/codeGraph/embeddings.csv'
        embeddingsToLabelPath='../../data/codeGraph/embeddingtolabel.csv'
        textToLabelPath='../../data/codeGraph/labeltotext.csv'
        embeded_docmessages=[]
        index = faiss.IndexFlatL2(512)

        with open(embeddingspath, 'r') as embeddings, open(embeddingsToLabelPath,'r') as embeddingsToLabels,open(textToLabelPath,'r') as textToLabels:
                i = 0
                for (line1,line2,line3) in zip(embeddings,embeddingsToLabels,textToLabels):
                        if i == 1:
                                break
                        newline = line1.rstrip()
                        parsedline = newline.split(',')
                        embeded_docmessages.append(parsedline)
                        index.add(np.asarray(parsedline,dtype=np.float32).reshape(1,-1)
                                 )
                        print(parsedline)
                        newline = line2.rstrip()
                        parsedline = newline.split(',')
                        print(parsedline)

                        newline = line3.rstrip()
                        parsedline = newline.split(',')
                        print(parsedline)
                        
                        
                        i += 1
                k=2
                    
                embeded_distance_index_info=[]
                embeded_distance_info=[]
                embeded_docmessages=np.asarray(embeded_docmessages,dtype=np.float32)
        for i in range(embeded_docmessages.shape[0]):
            # we want to see 2 nearest neighborS
#                     print(embeded_docmessages[i].reshape(-1,1).shape)
                    D, I = index.search(embeded_docmessages[i].reshape(1,-1), k)
                    embeded_distance_index_info.append(I)
                    embeded_distance_info.append(D)
                    print(embeded_distance_index_info)
#                         sys.stdout = open(output_path, "w")

        for i in range(embeded_docmessages.shape[0]):
                    print("-------------------------------------------------------------")
#                         print("document name  : \n"+str(embeded_docnames[i][j])+"\n")
                    for k in range(len(embeded_distance_index_info[i])):
                        print("\n close to document :",[embeded_distance_index_info[i][k]])
                        print("\n with a distance :",embeded_distance_info[i][k])




