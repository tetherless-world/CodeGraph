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
def build_index():
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
    with open('../../data/codeGraph/stackoverflow_questions_per_class_func_3M_filtered_new.json', 'r') as data,open('./lengthAnalysisDocstrings_NewJson.txt', 'w') as outputFile:
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

        return (index, docMessages, embeddingtolabelmap, docStringLength_avg,droppedClassWithLessLength,docLabelToTextForSentenceTokenizationAndAnalysis)


def evaluate_neighbors(index, docMessages, embeddingtolabelmap,docStringLength_avg,droppedClassWithLessLength,docLabelToTextForSentenceTokenizationAndAnalysis):
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

    with open('../../data/codeGraph/stackoverflow_questions_per_class_func_3M_filtered_new.json', 'r') as data, open('../../data/codeGraph/classes2superclass.out', 'r') as class2superclass, open('../../data/codeGraph/classes.map', 'r') as classes,open('docstringsEmbeddedOutk=10.txt', 'w') as outputFile:
        originalOut=sys.stdout
        sys.stdout=outputFile
        getHierarchy = ijson.items(class2superclass, 'results.bindings.item')
        classToSuperClass={}
        classToSubClassCount={}

        for iterateInHierarchy in getHierarchy:
            superClass = iterateInHierarchy['superclass']['value'].replace('http://purl.org/twc/graph4code/python/','')
            class_sub = iterateInHierarchy['class']['value'].replace('http://purl.org/twc/graph4code/python/','')
            classToSuperClass[class_sub]=superClass
            if superClass in classToSubClassCount:
                classToSubClassCount[superClass]=classToSubClassCount[superClass]+1 ##calculate number of classes sharing same super class 
            else:
                classToSubClassCount[superClass]=1
             

           
        firstJsonCollect = ijson.items(data, 'results.bindings.item') 
        jsonCollect = ijson.items(data, 'results.bindings.item')
        i = 0
        correctHierarchy=0
        wrongHierarchy=0
        correctclass=0
        exactcorrectclass=0
        wrongclass=0
        exactwrongclass=0
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
  

            embeddedText = embed([docStringText])#[stackText])
            embeddingVector = embeddedText[0]
            embeddingArray = np.asarray(
                embeddingVector, dtype=np.float32).reshape(1, -1)
            if docLabel not in classToSuperClass:
                        continue
            
            k = classToSubClassCount[classToSuperClass[docLabel]] ##get only certain number of nearest neighbors that belong to the same class
            if k > 10:
                continue
            print("\n")
            print("---------------------------------------- \n")
            print("The class whose neighbors are being computed:",docLabel,"its super class: ",classToSuperClass[docLabel])
            print("the count of the number of classes sharing the same superclass of this particular class:",k,"\n")

            D, I = index.search(embeddingArray, k)
            distances = D[0]
            indices = I[0]
#             print("Distances of related vectors:", distances)
#             print("Indices of related vectors:", indices)
            positivepresent=False
            exactpositivepresent=True

            for p in range(0, k):
                if p == 0:
                    continue
                properIndex = indices[p]
                embedding = docMessages[properIndex]
                adjustedembedding = tuple(embedding)
                label = embeddingtolabelmap[adjustedembedding]
                ##multiple docstrings associated with the same embedding mapped
                ##array of labels mapped
                j=0
                for l in label:
                    if docLabel not in classToSuperClass or l not in classToSuperClass:
                        continue
           ##not all labels present
                    if classToSuperClass[l]==classToSuperClass[docLabel]:
                        
                        positivepresent=True
                        correctHierarchy=correctHierarchy+1
                        print("class correctly predicted (in terms of hierarchy): ",l,"its super class:",classToSuperClass[l],"target super class:", classToSuperClass[docLabel],"\n")

                    else:
                        exactpositivepresent=False
                        print("class wrongly predicted (in terms of hierarchy):",l,"its super class:",classToSuperClass[l],"target super class:", classToSuperClass[docLabel],"\n")
                        wrongHierarchy=wrongHierarchy+1
            if exactpositivepresent:
                exactcorrectclass=exactcorrectclass+1
            else:
                exactwrongclass=exactwrongclass+1
            if positivepresent:
                correctclass=correctclass+1
            else:
                wrongclass=wrongclass+1
                
                
        print("items correctly classified",correctHierarchy)
        print("items wrongly classfied",wrongHierarchy)

        print("hierarchy accuracy %=",correctHierarchy*100/(correctHierarchy+wrongHierarchy))

        print("overall soft accuracy %=",correctclass*100/(correctclass+wrongclass))

        print("overall hard accuracy %=",exactcorrectclass*100/(exactcorrectclass+exactwrongclass))
        sys.stdout=originalOut

if __name__ == '__main__':
    dataTuple = build_index()
    print("Completed building index.")
    evaluate_neighbors(dataTuple[0], dataTuple[1], dataTuple[2], dataTuple[3],dataTuple[4],dataTuple[5])
