import faiss 
import csv
import numpy as np
import sys

if __name__ == '__main__':
        embeddingspath = '../../data/codeGraph/embeddings.csv'
        embeddingsToLabelPath='../../data/codeGraph/embeddingtolabel.csv'
        textToLabelPath='../../data/codeGraph/labeltotext.csv'
        embeded_docmessages=[]
        index = faiss.IndexFlatL2(512)

        with open(embeddingspath, 'r') as embeddings, open(embeddingsToLabelPath,'r') as embeddingsToLabels,open(textToLabelPath,'r') as textToLabels:
                #i = 0
                embeddingtolabellist = []
                labeltotextlist = []
                for (line1,line2,line3) in zip(embeddings,embeddingsToLabels,textToLabels):
                        newline = line1.rstrip()
                        parsedline = newline.split(',')
                        embeded_docmessages.append(parsedline)
                        linetoadd = np.asarray(parsedline, dtype=np.float32).reshape(1,-1)
                        index.add(linetoadd)
                        #print(parsedline)
                        newline = line2.rstrip()
                        parsedline = newline.split(',')
                        embeddingtolabellist.append(parsedline[1])
                        #print(parsedline)

                        newline = line3.rstrip()
                        parsedline = newline.split(',')
                        try:
                            labeltotextlist.append(parsedline[1])
                        # this was originally included to find a bug caused by not omitting
                        # carriage return in the csv, feel free to leave this part out
                        except IndexError:
                            print(newline)
                            print('\n\n\n\n\nSEPARATOR\n\n\n\n\n')
                            print(parsedline)
                            exit()
                        #print(parsedline)
                        
                        
                        #i += 1
                k=11
                    
                embeded_distance_index_info=[]
                embeded_distance_info=[]
                embeded_docmessages=np.asarray(embeded_docmessages,dtype=np.float32)
        for i in range(embeded_docmessages.shape[0]):
            # we want to see 2 nearest neighborS
#                     print(embeded_docmessages[i].reshape(-1,1).shape)
                    D, I = index.search(embeded_docmessages[i].reshape(1,-1), k)
                    embeded_distance_index_info.append(I)
                    embeded_distance_info.append(D)
                    #print(embeded_distance_index_info)
#                         sys.stdout = open(output_path, "w")

#        print(embeded_distance_index_info) 
#        print(embeded_distance_info)
        originalOut = sys.stdout
        with open('../../data/codeGraph/similarityAnalysis.txt', 'w') as outputFile:
            sys.stdout = outputFile
            for i in range(embeded_docmessages.shape[0]):
                        print("\n-------------------------------------------------------------")
                        print("Document name is:", embeddingtolabellist[i])
                        print("\nText for document is:", labeltotextlist[i]) 
#                         print("document name  : \n"+str(embeded_docnames[i][j])+"\n")
                        for p in range(len(embeded_distance_index_info[i])):
                            # call to tolist() here is optional, just looks better for output imo
                            print("\nIndices of related vectors:", embeded_distance_index_info[i][p].tolist()) 
                            print("Distances to each related vector:", embeded_distance_info[i][p].tolist())
                            for f in range(0, k):
                                print('\nDocument related by', str(f) +'th position is:', embeddingtolabellist[embeded_distance_index_info[i][p][f]])
                                print('\nText for document', str(f), 'is:', labeltotextlist[embeded_distance_index_info[i][p][f]])
            sys.stdout = originalOut




