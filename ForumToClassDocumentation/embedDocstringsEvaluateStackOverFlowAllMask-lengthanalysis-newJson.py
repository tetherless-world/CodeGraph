## when all the relevant classes associated with the stackoverflow post (Across all rows) are masked and then embedded, and evaluated for precision of predicting the closest docstrings(after embedding), for the model USE


##droppedClassWithLessLength  set can be used to adjust the length of the docstring characters to be dropped
##embeddingtolabelmap is used to retrieve labels for finding information about the nearest neighbors
## build_index() embeds Docstrings
##embedCollect used to have unique embeddings for lookup
##index contains FAISS indices
##evaluate_neighbors() compute the nearest neighbors for string that is embedded
##k number of neighbors to be computed
##classToSuperClass is used for class, to super class relationship




##provide input file path of the above json for example ../../data/codeGraph/stackoverflow_questions_per_class_func_3M_filtered_new.json

### dataset at https://ia801500.us.archive.org/24/items/stackoverflow_questions_per_class_func_3M_filtered_new/stackoverflow_questions_per_class_func_3M_filtered_new.json 


##for example, python embedDocstringsEvaluateStackOverFlowAllMask-lengthanalysis-newJson.py ../../data/codeGraph/stackoverflow_questions_per_class_func_3M_filtered_new.json 5

##where 5 is the number of nearest neighbors

##output file output1.txt and allMaskOutput.txt
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
def build_index(input_file,nn):
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
    with open(input_file, 'r') as data,open('./output1.txt', 'w') as outputFile:
        originalout = sys.stdout
        sys.stdout= outputFile
        jsonCollect = ijson.items(data, 'results.bindings.item')
        i = 0
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
                if docLabel in duplicateClassDocString:
                    pass
                    
                else:
                    if len(docStringText) < -1:
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

        return (index, docMessages, embeddingtolabelmap, docStringLength_avg,droppedClassWithLessLength,docLabelToTextForSentenceTokenizationAndAnalysis,input_file,int(nn))


def evaluate_neighbors(index, docMessages, embeddingtolabelmap,docStringLength_avg,droppedClassWithLessLength,docLabelToTextForSentenceTokenizationAndAnalysis,input_file,nn):
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
   
    with open(input_file, 'r') as data, open('allMaskOutput.txt', 'w') as outputFile:
        originalout = sys.stdout
        sys.stdout= outputFile
        firstJsonCollect = ijson.items(data, 'results.bindings.item') 
        postMap = {}
        for jsonObject in firstJsonCollect:
            objectType = jsonObject['class_func_type']['value'].replace('http://purl.org/twc/graph4code/ontology/','')
            if objectType != 'Class':
                continue
            stackText = jsonObject['content_wo_code']+ \
                " " + jsonObject['answer_wo_code']
            soup = BeautifulSoup(stackText, 'html.parser')
            for code in soup.find_all('code'):
                code.decompose()
            stackText = soup.get_text()
            classLabel = jsonObject['class_func_label']['value']
            if stackText in postMap:
                postMap[stackText].append(classLabel) 
            else:
                postMap[stackText] = [classLabel]
        data.close()
        newData = open('../../data/codeGraph/stackoverflow_questions_per_class_func_3M_filtered_new.json', 'r')
        jsonCollect = ijson.items(newData, 'results.bindings.item')
        sys.stdout = outputFile
        for jsonObject in jsonCollect:
            totaldocs+=1
            objectType = jsonObject['class_func_type']['value'].replace(
                'http://purl.org/twc/graph4code/ontology/', '')
            if objectType != 'Class':
                continue
            title = jsonObject['title']['value']
            classLabel = jsonObject['class_func_label']['value']
            originalStackText = jsonObject['content_wo_code']+ \
                " " + jsonObject['answer_wo_code']
            if classLabel in droppedClassWithLessLength:
                continue  
            soup = BeautifulSoup(originalStackText, 'html.parser')
            for code in soup.find_all('code'):
                code.decompose()
            stackText = soup.get_text()
#             if len(stackText) < 50:
#                 continue
#              print('\nTitle of Stack Overflow Post:', title)
            print('Class associated with post:', classLabel, '\n')
            print('Text of post before masking:', stackText, '\n')
            maskedText = None
            for foundLabel in postMap[stackText]: 
                splitLabel = foundLabel.lower().split('.')
                wholePattern = re.compile(foundLabel.lower(), re.IGNORECASE)
                maskedText = wholePattern.sub(' ', stackText)
                for labelPart in splitLabel:
                    partPattern = re.compile(labelPart, re.IGNORECASE)
                    maskedText = partPattern.sub(' ', maskedText)#maskedText.replace(labelPart, ' ')
            print('Text of post after masking:', maskedText, '\n')
            embeddedText = embed([maskedText])#[stackText])
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
                print("Loose False Positive Present  \n")
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
                print("match True Positive Present \n")
            print("--------------------------------------------- \n")


        print(tp/(tp+fp), " Loose Precision at "+str(k)+" with all  masking ")
        print(etp/(etp+efp), "Exact Precision at "+str(k)+" with all masking ")

        sys.stdout=originalout

if __name__ == '__main__':
    dataTuple = build_index(sys.argv[1:][0],sys.argv[1:][1])
    print("Completed building index.")
    evaluate_neighbors(dataTuple[0], dataTuple[1], dataTuple[2], dataTuple[3],dataTuple[4],dataTuple[5],dataTuple[6],dataTuple[7])
