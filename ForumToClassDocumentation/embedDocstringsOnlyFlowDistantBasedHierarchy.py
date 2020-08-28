##the file constructs mapping of class to superclass and distance corresponding to them based on output only, enables analysis--> output detailedMappingsOut
##droppedClassWithLessLength  set can be used to adjust the length of the docstring characters to be dropped
##embeddingtolabelmap is used to retrieve labels for finding information about the nearest neighbors
## build_index() embeds Docstrings
##embedCollect used to have unique embeddings for lookup
##index contains FAISS indices
##evaluate_neighbors() compute the nearest neighbors for string that is embedded
##k number of neighbors to be computed
##classToSuperClass is used for class, to super class relationship
### dataset at https://ia801500.us.archive.org/24/items/stackoverflow_questions_per_class_func_3M_filtered_new/stackoverflow_questions_per_class_func_3M_filtered_new.json 


##Other data needed 
##https://archive.org/download/classes2superclass/classes2superclass.out
##output 1 and output2.txt are produced
##python embedDocstringsOnlyFlowDistantBasedHierarchy.py ../../data/codeGraph/stackoverflow_questions_per_class_func_3M_filtered_new.json ../../data/codeGraph/classes2superclass.out 

##https://ia601401.us.archive.org/30/items/classes2superclass/class2top10neighbors_withScore.txt
##is produced using this hierarchy related output2.txt 

##output2.txt format: knearest_neighbors class FAISS_distance
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
def build_index(input1,input2):
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
    with open(input1, 'r') as data,open('output1.txt', 'w') as outputFile,open(input2, 'r') as class2superclass:
        jsonCollect = ijson.items(data, 'results.bindings.item')
        i = 0
        getHierarchy = ijson.items(class2superclass, 'results.bindings.item')

        classToSuperClass={}
        classToSubClassCount={}
        mainPackage={}
        classesFilteredOutDueToObject={}
        isSuperClass={}
        for iterateInHierarchy in getHierarchy:
            superClass = iterateInHierarchy['superclass']['value'].replace('http://purl.org/twc/graph4code/python/','')
            class_sub = iterateInHierarchy['class']['value'].replace('http://purl.org/twc/graph4code/python/','')
            classToSuperClass[class_sub]=superClass
            isSuperClass[superClass]=1
            if superClass == 'Object' or  superClass == 'object' :
                
                classesFilteredOutDueToObject[class_sub]=1
        
        ##if a class only has superclass of object and no subclasses then its eliminated
        ##the class filtered out has a subclass  

                    
                 
                      

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
            if docLabel not in classToSuperClass:
                        continue

            if docLabel in classesFilteredOutDueToObject:
                if docLabel in isSuperClass:
                    pass
                else:
                    
                    continue
                    
            
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

        return (index, docMessages, embeddingtolabelmap, docStringLength_avg,droppedClassWithLessLength,docLabelToTextForSentenceTokenizationAndAnalysis,input1,input2)


def evaluate_neighbors(index, docMessages, embeddingtolabelmap,docStringLength_avg,droppedClassWithLessLength,docLabelToTextForSentenceTokenizationAndAnalysis,input1,input2):
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

    with open(input1, 'r') as data, open(input2, 'r') as class2superclass,open('output2.txt', 'w') as outputFile:
        originalOut=sys.stdout
        sys.stdout=outputFile
        getHierarchy = ijson.items(class2superclass, 'results.bindings.item')
        classToSuperClass={}
        classToSubClassCount={}
        mainPackage={}
        classesFilteredOutDueToObject={}
        isSuperClass={}
        for iterateInHierarchy in getHierarchy:
            superClass = iterateInHierarchy['superclass']['value'].replace('http://purl.org/twc/graph4code/python/','')
            class_sub = iterateInHierarchy['class']['value'].replace('http://purl.org/twc/graph4code/python/','')
            classToSuperClass[class_sub]=superClass
            isSuperClass[superClass]=1
            if superClass == 'Object' or  superClass == 'object' :
                
                classesFilteredOutDueToObject[class_sub]=1
        
        ##if a class only has superclass of object and no subclasses then its eliminated
       
                        

                
                
#             if superClass in classToSubClassCount:
#                 classToSubClassCount[superClass]=classToSubClassCount[superClass]+1 ##calculate number of classes sharing same super class 
#             else:
#                 classToSubClassCount[superClass]=1
                
             
        
           
        firstJsonCollect = ijson.items(data, 'results.bindings.item') 
        jsonCollect = ijson.items(data, 'results.bindings.item')
        i = 0
        correctHierarchy=0
        wrongHierarchy=0
        correctclass=0
        exactcorrectclass=0
        wrongclass=0
        exactwrongclass=0
        completedClass={}
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
            if docLabel in completedClass:
            
                     continue
            if docLabel in classesFilteredOutDueToObject:
                if docLabel in isSuperClass:
                    pass
                else:
#                         print("Class skipped",docLabel)
                        continue
                    
            completedClass[docLabel]=1
            
#             k = classToSubClassCount[classToSuperClass[docLabel]] ##get only certain number of nearest neighbors that belong to the same class
#             if k > 10:
#                 continue
            k=11

#             print("The class whose neighbors are being computed:",docLabel,"its super class: ",classToSuperClass[docLabel])
#             print("the count of the number of classes sharing the same superclass of this particular class:",k,"\n")

            D, I = index.search(embeddingArray, k)
            distances = D[0]
            indices = I[0]
#             print("Distances of related vectors:", distances)
#             print("Indices of related vectors:", indices)
            positivepresent=False
            exactpositivepresent=True
            j=0
#             print("-------------------------------------------------- \n")
#             print("Docstring text corresponding to: ", docLabel,"\n") 
#             print(docStringText,"\n")
            for p in range(0, k):
                if p == 0:
                    continue
                if j == 11:
                    break
                properIndex = indices[p]
                actualDistance=distances[p]
                embedding = docMessages[properIndex]
                adjustedembedding = tuple(embedding)
                label = embeddingtolabelmap[adjustedembedding]
                ##multiple docstrings associated with the same embedding mapped
                ##array of labels mapped
                
                for l in label:
                    j+=1
                    if j== 11:
                        break

                    if docLabel not in classToSuperClass or l not in classToSuperClass:
                        continue
           ##not all labels present
                    if classToSuperClass[l]==classToSuperClass[docLabel]:
                        
                        positivepresent=True
                        correctHierarchy=correctHierarchy+1
                        print(l,docLabel,actualDistance)
#                         print("mapping:",l,docLabel)
#                         print("Distance",actualDistance)
#                         print("\n")
#                         print("text of the neighbor \n")
#                         print(docLabelToTextForSentenceTokenizationAndAnalysis[l])
#                         print("\n")
                        

                    else:
                        exactpositivepresent=False
                        print(l,docLabel,actualDistance)
#                         print("Distance",actualDistance)
#                         print("\n")
#                         print("text of the neighbor \n")
#                         print(docLabelToTextForSentenceTokenizationAndAnalysis[l])
#                         print("\n")
                        wrongHierarchy=wrongHierarchy+1
            if exactpositivepresent:
                exactcorrectclass=exactcorrectclass+1
            else:
                exactwrongclass=exactwrongclass+1
            if positivepresent:
                correctclass=correctclass+1
            else:
                wrongclass=wrongclass+1
                

        sys.stdout=originalOut

if __name__ == '__main__':
    dataTuple = build_index(sys.argv[1:][0],sys.argv[1:][1])
    print("Completed building index.")
    evaluate_neighbors(dataTuple[0], dataTuple[1], dataTuple[2], dataTuple[3],dataTuple[4],dataTuple[5],dataTuple[6],dataTuple[7])
