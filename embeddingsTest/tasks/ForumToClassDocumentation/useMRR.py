## MRR computation with hits and mean  for the model USE

##droppedClassWithLessLength  set can be used to adjust the length of the docstring characters to be dropped
##embeddingtolabelmap is used to retrieve labels for finding information about the nearest neighbors
## build_index() embeds Docstrings
##embedCollect used to have unique embeddings for lookup
##index contains FAISS indices
##evaluate_neighbors() compute the nearest neighbors for string that is embedded
##k number of neighbors to be computed
##classToSuperClass is used for class, to super class relationship
##for no masking and all masking, comment out the lines indicated below

##provide input file path of the above json for example ../../data/codeGraph/stackoverflow_questions_per_class_func_3M_filtered_new.json

### dataset at https://ia801500.us.archive.org/24/items/stackoverflow_questions_per_class_func_3M_filtered_new/stackoverflow_questions_per_class_func_3M_filtered_new.json 


##for example, python useMRR.py ../../data/codeGraph/stackoverflow_questions_per_class_func_3M_filtered_new.json 10 T
##where 10 is the number of nearest neighbors
##T is one masked,F is no masked configuartion

##output file output1.txt and USEMRR.txt

import ijson
import tensorflow_hub as hub
import faiss
import numpy as np
from bs4 import BeautifulSoup
import sys
import re
from statistics import mean 
from statistics import pstdev
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
from scipy import stats
def build_index(input1,nn,maskFlag):
    embed = hub.load('https://tfhub.dev/google/universal-sentence-encoder/4')
    index = faiss.IndexFlatL2(512)
    docMessages = []
    embeddingtolabelmap = {}
    labeltotextmap = {}
    embedCollect = set()
    duplicateClassDocString=set()
    droppedClassWithLessLength=set()
    docStringLength_avg=[]
    docLabelToTextForSentenceTokenizationAndAnalysis= {}
    with open(input1, 'r') as data,open('output1.txt', 'w') as outputFile:
        jsonCollect = ijson.items(data, 'results.bindings.item')
        i = 0
        originalout = sys.stdout
        sys.stdout = outputFile

        for jsonObject in jsonCollect:
            objectType = jsonObject['class_func_type']['value'].replace(
                'http://purl.org/twc/graph4code/ontology/', '')
            if objectType != 'Class':
                continue
            label = jsonObject['class_func_label']['value']
            docLabel = label
            docStringText = jsonObject['docstr']['value']# + ' ' + str(i)
            soup = BeautifulSoup(docStringText, 'html.parser')
            for code in soup.find_all('code'):
                code.decompose()
            docStringText = soup.get_text()
            
            if docStringText in embedCollect:
                ##had to include this because the same class was getting addded into the mapped array
                if docLabel in duplicateClassDocString:
                    pass
                    
                else:
                    if len(docStringText) < -1:
#                         print("has less than 300 character,class:",docLabel,"docstring:",docStringText)
                        droppedClassWithLessLength.add(docLabel)
                        continue
                    duplicateClassDocString.add(docLabel)
                    docStringLength_avg.append(len(docStringText))
                    print("doclabel",docLabel)
                    docLabelToTextForSentenceTokenizationAndAnalysis[docLabel]=docStringText
                    embeddedDocText = embed([docStringText])[0]
                    embeddingtolabelmap[tuple(
                    embeddedDocText.numpy().tolist())].append(docLabel)
            else:
                if len(docStringText) < -1:
#                         print("has less than 300 character,class:",docLabel,"docstring:",docStringText)
                        droppedClassWithLessLength.add(docLabel)
                        continue
                duplicateClassDocString.add(docLabel)
                embedCollect.add(docStringText)
                docStringLength_avg.append(len(docStringText))
                docLabelToTextForSentenceTokenizationAndAnalysis[docLabel]=docStringText
                print("doclabel",docLabel)
                print(len(docStringText))
                embeddedDocText = embed([docStringText])[0]
                newText = np.asarray(
                embeddedDocText, dtype=np.float32).reshape(1, -1)
                docMessages.append(embeddedDocText.numpy().tolist())
                index.add(newText)
                embeddingtolabelmap[tuple(
                embeddedDocText.numpy().tolist())] = [docLabel]
#                 if  docLabel == 'pysnmp.smi.rfc1902.ObjectType':
#                     print("text for pysnmp.smi.rfc1902.ObjectType' is")
#                     print(docStringText)
#            labeltotextmap[docLabel] = docStringText
            i += 1
        sys.stdout=originalout

        return (index, docMessages, embeddingtolabelmap, docStringLength_avg,droppedClassWithLessLength,docLabelToTextForSentenceTokenizationAndAnalysis,input1,nn,maskFlag)


def evaluate_neighbors(index, docMessages, embeddingtolabelmap,docStringLength_avg,droppedClassWithLessLength,docLabelToTextForSentenceTokenizationAndAnalysis,input1,nn,mask):
    k = int(nn)
    fp=0
    fn=0
    tp=0
    tn=0
    efp=0
    efn=0
    etp=0
    etn=0
    rr=0
    rrcount=0
    mrr=[]
    positivepresent=False
    exactpositivepresent=False
    totaldocs=0
    embed = hub.load('https://tfhub.dev/google/universal-sentence-encoder/4')
    originalout = sys.stdout
    with open(input1, 'r') as data, open('USEMRR.txt', 'w') as outputFile:
        hits=0
        jsonCollect = ijson.items(data, 'results.bindings.item')
        sys.stdout = outputFile
        stack_overflow_length=[]
        for jsonObject in jsonCollect:
            totaldocs+=1
            objectType = jsonObject['class_func_type']['value'].replace(
                'http://purl.org/twc/graph4code/ontology/', '')
            if objectType != 'Class':
                continue
            title = jsonObject['title']['value']
            classLabel = jsonObject['class_func_label']['value']
            if classLabel in droppedClassWithLessLength:
                continue  

            stackText = jsonObject['content_wo_code']+ \
                " " + jsonObject['answer_wo_code']
            soup = BeautifulSoup(stackText, 'html.parser')

            for code in soup.find_all('code'):
                code.decompose()
            stackText = soup.get_text()
#             if len(stackText) < 50:
#                 continue
            print('Class associated with post:', classLabel,'\n')
            print("Text of the associated post:",stackText,'\n')
            splitLabel = classLabel.lower().split('.')
            wholePattern = re.compile(classLabel.lower(), re.IGNORECASE)
            maskedText = wholePattern.sub(' ', stackText)
            for labelPart in splitLabel:
                    partPattern = re.compile(labelPart, re.IGNORECASE)
                    maskedText = partPattern.sub(' ', maskedText)#maskedText.replace(labelPart, ' ')
                    
            if mask == 'F':
                embeddedText = embed([stackText])#[maskedText])
                outputstackmessage="without masking"
            else:
                embeddedText = embed([maskedText])#[maskedText])
                outputstackmessage="with one masking"

            embeddingVector = embeddedText[0]
            embeddingArray = np.asarray(
                embeddingVector, dtype=np.float32).reshape(1, -1)
            D, I = index.search(embeddingArray, k)
            distances = D[0]
            indices = I[0]
#             print("Distances of related vectors:", distances)
#             print("Indices of related vectors:", indices)
            positivepresent=False
            exactpositivepresent=False
            for p in range(0, k):
                properIndex = indices[p]
                embedding = docMessages[properIndex]
                adjustedembedding = tuple(embedding)
                label = embeddingtolabelmap[adjustedembedding]
                ##multiple docstrings associated with the same embedding mapped
                ##array of labels mapped
                j=0
                for l in label:
                    
                    if l.startswith(classLabel.split(".")[0]):
                        pass
                        
                        if not positivepresent and not exactpositivepresent:
                            rr=p+1
                        positivepresent=True
                    else:
                        print("class not associated",l)
                    if l == classLabel:
                        if not exactpositivepresent and not positivepresent:
                            rr=p+1
                        exactpositivepresent=True
                        
                        
                    j=j+1
            if exactpositivepresent:
                hits+=1
#             if exactpositivepresent:

                mrr.append(1/rr)
            else:
#                  pass
                  mrr.append(0)

        print("--------------------------------------------- \n")
        print("MRR")
        print(sum(mrr)/len(mrr))
        print("sum of MRR",sum(mrr))
        print("lenght of MRR",len(mrr))
        print("Mean",stats.sem(mrr))
        print("Hits",hits/len(mrr))
        sys.stdout=originalout

if __name__ == '__main__':
    dataTuple = build_index(sys.argv[1:][0],sys.argv[1:][1],sys.argv[1:][2])
    print("Completed building index.")
    evaluate_neighbors(dataTuple[0], dataTuple[1], dataTuple[2], dataTuple[3],dataTuple[4],dataTuple[5],dataTuple[6],dataTuple[7],dataTuple[8])
