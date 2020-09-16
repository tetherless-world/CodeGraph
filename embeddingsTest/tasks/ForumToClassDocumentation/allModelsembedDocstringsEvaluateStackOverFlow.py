## No masking: The Stackoverflow post is embedded as-is. One masking: The particular class associated with the stackoverflow post is masked and then embedded and evaluated for precision of predicting the closest docstrings(after embedding), for the models:  'bert-base-nli-stsb-mean-tokens', 'bert-large-nli-stsb-mean-tokens','roberta-base-nli-stsb-mean-tokens','roberta-large-nli-stsb-mean-tokens', 'distilbert-base-nli-stsb-mean-tokens', corresponding output in stackNewJson_NoOrOneMask_'+model_name+'_.txt
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


##for example, python allModelsembedDocstringsEvaluateStackOverFlow.py ../../data/codeGraph/stackoverflow_questions_per_class_func_3M_filtered_new.json 5 T
##T is one masked,F is no masked configuartion
##../../data/codeGraph/stackoverflow_questions_per_class_func_3M_filtered_new.json 
##where 5 is the number of nearest neighbors

##output file output1.txt and  stackNewJson_NoOrOneMask_T'+model_name+'_.txt

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
from sentence_transformers import SentenceTransformer

from nltk.tokenize import sent_tokenize

def run_differentmodels(input_file,nn,mask):
        modelList = [ 'bert-base-nli-stsb-mean-tokens','roberta-base-nli-stsb-mean-tokens']
        for i in modelList:
                print("running model",i)
                dataTuple = build_index(i,input_file)
                evaluate_neighbors(dataTuple[0], dataTuple[1], dataTuple[2], dataTuple[3],dataTuple[4],dataTuple[5],i,input_file,int(nn),mask)
                
    
def build_index(model,input_file):
    embed = hub.load('https://tfhub.dev/google/universal-sentence-encoder/4')

    docMessages = []
    embeddingtolabelmap = {}
    labeltotextmap = {}
    embedCollect = set()
    duplicateClassDocString=set()
    droppedClassWithLessLength=set()
    docStringLength_avg=[]
    docLabelToTextForSentenceTokenizationAndAnalysis= {}

    with open(input_file, 'r') as data,open('output1.txt', 'w') as outputFile:
        jsonCollect = ijson.items(data, 'results.bindings.item')
        i = 0
        transformer = SentenceTransformer(model)
        embeddedDocTextLen = len(transformer.encode(["index builder"])[0])
        index = faiss.IndexFlatL2(embeddedDocTextLen)
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
#                         
                        droppedClassWithLessLength.add(docLabel)
                        continue
                    duplicateClassDocString.add(docLabel)
                    docStringLength_avg.append(len(docStringText))
                    print("doclabel",docLabel)
                    docLabelToTextForSentenceTokenizationAndAnalysis[docLabel]=docStringText
                    embeddedDocText = transformer.encode([docStringText])
                    print("embeddedDoctext shape",np.asarray(embeddedDocText, dtype=np.float32).shape)

                    embeddingtolabelmap[np.asarray(embeddedDocText, dtype=np.float32).tobytes()] = [docLabel]

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
                
                embeddedDocText = transformer.encode([docStringText])  
                print("embeddedDoctext shape",np.asarray(embeddedDocText, dtype=np.float32).shape)
                docMessages.append(np.asarray(embeddedDocText, dtype=np.float32))
                index.add(np.asarray(embeddedDocText, dtype=np.float32))
                embeddingtolabelmap[np.asarray(embeddedDocText, dtype=np.float32).tobytes()] = [docLabel]
#                 if  docLabel == 'pysnmp.smi.rfc1902.ObjectType':
#                     print("text for pysnmp.smi.rfc1902.ObjectType' is")
#                     print(docStringText)
#            labeltotextmap[docLabel] = docStringText
            i += 1
        sys.stdout=originalout

        return (index, docMessages, embeddingtolabelmap, docStringLength_avg,droppedClassWithLessLength,docLabelToTextForSentenceTokenizationAndAnalysis)


def evaluate_neighbors(index, docMessages, embeddingtolabelmap,docStringLength_avg,droppedClassWithLessLength,docLabelToTextForSentenceTokenizationAndAnalysis,model,input_file,nn,mask):
    k = nn
    fp=0
    fn=0
    tp=0
    tn=0
    efp=0
    efn=0
    etp=0
    etn=0
    positivepresent=False
    exactpositivepresent=False
    totaldocs=0
    embed = hub.load('https://tfhub.dev/google/universal-sentence-encoder/4')
    originalout = sys.stdout
    transformer = SentenceTransformer(model)
    with open(input_file, 'r') as data, open('stackNewJson_NoOrOneMask_'+mask+model+'_.txt', 'w') as outputFile:
        
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
#              print('\nTitle of Stack Overflow Post:', title)
            print('Class associated with post:', classLabel)
            splitLabel = classLabel.lower().split('.')
            wholePattern = re.compile(classLabel.lower(), re.IGNORECASE)
            maskedText = wholePattern.sub(' ', stackText)
            for labelPart in splitLabel:
                    partPattern = re.compile(labelPart, re.IGNORECASE)
                    maskedText = partPattern.sub(' ', maskedText)#maskedText.replace(labelPart, ' ')
            ##uncomment this line and replace with masked Text for one masking,  else stackText for no masking

            if mask == 'F':
                embeddedText = transformer.encode([stackText])
                outputstackmessage="without masking"
            else:
                embeddedText = transformer.encode([maskedText])
                outputstackmessage="with one masking"
            D, I = index.search(  np.asarray(embeddedText, dtype=np.float32), k)
            distances = D[0]
            indices = I[0]

#             print("Distances of related vectors:", distances)
#             print("Indices of related vectors:", indices)
            positivepresent=False
            exactpositivepresent=False
            for p in range(0, k):
                properIndex = indices[p]
                embedding = docMessages[properIndex]
                adjustedembedding = np.asarray(embedding, dtype=np.float32).tobytes()
                label = embeddingtolabelmap[adjustedembedding]
                ##multiple docstrings associated with the same embedding mapped
                ##array of labels mapped
                j=0
                for l in label:
                    
                    if l.startswith(classLabel.split(".")[0]):
                        positivepresent=True
                        if j == 0:
                             print("\n True positive label being contributed by \n",l)
                        else:
                            print("and \t",l)
                    else:
                        print("class not associated",l)
                    if l == classLabel:
                        exactpositivepresent=True
                        print("\n Exact positive label being contributed by \n",l)
                    j=j+1
                        
            if not positivepresent:
                fp=fp+1
                print("Loose False Positive Present \n")
                print("Investigating the reason with sentence tokenized docstring for:", classLabel,"\n")
                print(sent_tokenize(docLabelToTextForSentenceTokenizationAndAnalysis[classLabel]))
            else:
                tp=tp+1
#                 print("Loose True Positive Present -------------------------------------------------------- \n")
            if not exactpositivepresent:
                efp=efp+1
#                 print("match  False Positive Present ------------------------------------------------------- \n")
            else:
                etp=etp+1
            
#                 print("match True Positive Present -------------------------------------------------------- \n")
        print("--------------------------------------------- \n")
        

        print(tp/(tp+fp), " Loose Precision at" +str(nn)+" "+outputstackmessage +model)
        print(etp/(etp+efp), "Exact Precision at" +str(nn)+" "+outputstackmessage +model)

        sys.stdout=originalout

if __name__ == '__main__':
    run_differentmodels(sys.argv[1:][0],sys.argv[1:][1],sys.argv[1:][2])
        
