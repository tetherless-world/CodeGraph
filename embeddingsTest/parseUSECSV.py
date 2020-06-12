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
#                i = 0
                embeddingtolabelmap = {}
                labeltotextmap = {}
                encounteredTexts = set()
                discardedDocuments = {}
                for (line1,line2,line3) in zip(embeddings,embeddingsToLabels,textToLabels):
#                        if i == 4000:
#                            break
                        newline = line3.rstrip()
                        parsedline = newline.split(',')
                        label = parsedline[0]
                        text = parsedline[1]
                        if text in encounteredTexts:
                            print("Duplicate eliminated.")
                            discardedDocuments[label] = text
                            continue
                        else:
                            encounteredTexts.add(text)
                        try:
                            labeltotextmap[label] = text    
                        # this was originally included to find a bug caused by not omitting
                        # carriage return in the csv, feel free to leave this part out
                        except IndexError:
                            print(newline)
                            print('\n\n\n\n\nSEPARATOR\n\n\n\n\n')
                            print(parsedline)
                            exit()
                        #print(parsedline)

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
                        
#                        i += 1
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
        with open('../../data/codeGraph/finalSimilarityAnalysis.txt', 'w') as outputFile:
            sys.stdout = outputFile
            for i in range(embeded_docmessages.shape[0]):
                        print("\n-------------------------------------------------------------")
                        adjustedtopembedding = tuple(embeded_docmessages[i].tolist())
                        toplabel = embeddingtolabelmap[adjustedtopembedding]
                        print('\nName of document is:', toplabel)
                        print('\nText of document is:', labeltotextmap[toplabel])
                        for p in range(len(embeded_distance_index_info[i])):
                            # call to tolist() here is optional, just looks better for output imo
                            print("\nIndices of related vectors:", embeded_distance_index_info[i][p].tolist()) 
                            print("Distances to each related vector:", embeded_distance_info[i][p].tolist())
                            for f in range(0, k):
                                numpyembedding = embeded_docmessages[embeded_distance_index_info[i][p][f]] 
                                adjustedembedding = tuple(numpyembedding.tolist())
                                label = embeddingtolabelmap[adjustedembedding]
                                print('\nName of document in ranking order', f, 'is:', label)
                                print('\nText of document', f, 'is:', labeltotextmap[label])
            sys.stdout = originalOut 
        with open('../../data/codeGraph/discardedDocuments.txt', 'w') as discardFile:
            sys.stdout = discardFile
            for discardLabel, discardText in discardedDocuments.items():
                print(discardLabel + ',' + discardText)
            sys.stdout = originalOut
