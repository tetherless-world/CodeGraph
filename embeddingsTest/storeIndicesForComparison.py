import faiss 
import csv
import numpy as np
import sys
import pickle
if __name__ == '__main__':
        embeddingspath = '../../data/codeGraph/embeddings.csv'
        embeddingsToLabelPath='../../data/codeGraph/embeddingtolabel.csv'
        textToLabelPath='../../data/codeGraph/labeltotext.csv'
        discardedPath='../../data/codeGraph/discardedDocuments.csv'
        embeded_docmessages=[]
        index = faiss.IndexFlatL2(512)

        with open(embeddingspath, 'r') as embeddings, open(embeddingsToLabelPath,'r') as embeddingsToLabels,open(textToLabelPath,'r') as textToLabels,open(discardedPath,'r') as discarded:
                embeddingtolabelmap = {}
                labeltotextmap = {}
                duplicate_documents_to_original={}
                for line in discarded:
                    newline = line.rstrip()
                    parsedline = newline.split(',')
                    try:
                        
                        duplicate_documents_to_original[parsedline[1]] =    parsedline[0]
                    # this was originally included to find a bug caused by not omitting
                    # carriage return in the csv, feel free to leave this part out
                    except IndexError:
                        print(newline)
                        print('\n\n\n\n\nSEPARATOR\n\n\n\n\n')
                        print(parsedline)
                        exit()
                    
                for (line1,line2,line3) in zip(embeddings,embeddingsToLabels,textToLabels):

                        newline = line1.rstrip()
                        parsedline = newline.split(',')
                        embeded_docmessages.append(parsedline)
                        linetoadd = np.asarray(parsedline, dtype=np.float32).reshape(1,-1)
                        index.add(linetoadd)
                        #print(parsedline)
                        newline = line2.rstrip()
                        parsedline = newline.split(',')
                        splitembedding = parsedline[0].split(';')
                        arrayembedding = np.asarray(splitembedding, dtype=np.float32).reshape(1,-1)
                        arrayembedding = arrayembedding.tolist()
                        #print(arrayembedding)
                        finalembedding = tuple(arrayembedding[0])
                        #print(finalembedding)
                        embeddingtolabelmap[finalembedding] = parsedline[1]
                        #print(parsedline)

                        newline = line3.rstrip()
                        parsedline = newline.split(',')
                        try:
                            labeltotextmap[parsedline[0]] = parsedline[1]    
                        # this was originally included to find a bug caused by not omitting
                        # carriage return in the csv, feel free to leave this part out
                        except IndexError:
                            print(newline)
                            print('\n\n\n\n\nSEPARATOR\n\n\n\n\n')
                            print(parsedline)
                            exit()
                        #print(parsedline)
                  
                        
                        
#                        i += 1
                k=11
                    
                embeded_distance_index_info=[]
                embeded_distance_info=[]
                embeded_docmessages=np.asarray(embeded_docmessages,dtype=np.float32)
                with open('embeded_docmessages.pickle', 'wb') as f:
                    pickle.dump(embeded_docmessages, f, pickle.HIGHEST_PROTOCOL)
                faiss.write_index(index, 'faiss_index.saved')
                with open('labeltotextmap.pickle', 'wb') as f:
                    pickle.dump(labeltotextmap, f, pickle.HIGHEST_PROTOCOL)
                with open('embeddingtolabelmap.pickle', 'wb') as f:
                    pickle.dump(embeddingtolabelmap, f, pickle.HIGHEST_PROTOCOL)
                with open('duplicate_documents.pickle', 'wb') as f:
                    pickle.dump(duplicate_documents_to_original, f, pickle.HIGHEST_PROTOCOL)
                print("Program terminated")

    




