import ijson
import tensorflow_hub as hub
import faiss
import numpy as np
from bs4 import BeautifulSoup
import sys
import re


def build_index():
    embed = hub.load('https://tfhub.dev/google/universal-sentence-encoder/4')
    index = faiss.IndexFlatL2(512)
    docMessages = []
    embeddingtolabelmap = {}
    labeltotextmap = {}
    embedCollect = set()
    duplicateClassDocString=set()
    with open('../../data/codeGraph/stackoverflow_questions_per_class_func_1M_filtered.json', 'r') as data:
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
                    duplicateClassDocString.add(docLabel)
                    embeddedDocText = embed([docStringText])[0]
                    embeddingtolabelmap[tuple(
                    embeddedDocText.numpy().tolist())].append(docLabel)
            else:
                duplicateClassDocString.add(docLabel)
                embedCollect.add(docStringText)
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
        return (index, docMessages, embeddingtolabelmap)#, labeltotextmap)


def evaluate_neighbors(index, docMessages, embeddingtolabelmap, labeltotextmap):
    k =1 
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
    with open('../../data/codeGraph/stackoverflow_questions_per_class_func_1M_filtered.json', 'r') as data, open('./resultsFromNoDupeEmbeddingDocStringThenStackOverflowAllMaskingPr@1.txt', 'w') as outputFile:
        firstJsonCollect = ijson.items(data, 'results.bindings.item') 
        postMap = {}
        for jsonObject in firstJsonCollect:
            objectType = jsonObject['class_func_type']['value'].replace('http://purl.org/twc/graph4code/ontology/','')
            stackText = jsonObject['content']['value'] + \
                " " + jsonObject['answerContent']['value']
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
        newData = open('../../data/codeGraph/stackoverflow_questions_per_class_func_1M_filtered.json', 'r')
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
            originalStackText = jsonObject['content']['value'] + \
                " " + jsonObject['answerContent']['value']
            soup = BeautifulSoup(originalStackText, 'html.parser')
            for code in soup.find_all('code'):
                code.decompose()
            stackText = soup.get_text()
            print('\nTitle of Stack Overflow Post:', title)
            print('Class associated with post:', classLabel)
            print('Text of post before masking:', stackText)
            maskedText = None
            for foundLabel in postMap[stackText]: 
                splitLabel = foundLabel.lower().split('.')
                wholePattern = re.compile(foundLabel.lower(), re.IGNORECASE)
                maskedText = wholePattern.sub(' ', stackText)
                for labelPart in splitLabel:
                    partPattern = re.compile(labelPart, re.IGNORECASE)
                    maskedText = partPattern.sub(' ', maskedText)#maskedText.replace(labelPart, ' ')
            print('Text of post after masking:', maskedText)

## masking removed for now

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
                            print("\t and",l)
                    if l == classLabel:
                        exactpositivepresent=True
                        print("\n Exact positive label being contributed by \n",l)
                    j=j+1
                        
                    
            if not positivepresent:
                fp=fp+1
#                 print("Loose False Positive Present ------------------------------------------------------- \n")
            else:
                tp=tp+1
#                 print("Loose True Positive Present -------------------------------------------------------- \n")
            if not exactpositivepresent:
                efp=efp+1
#                 print("match  False Positive Present ------------------------------------------------------- \n")
            else:
                etp=etp+1
#                 print("match True Positive Present -------------------------------------------------------- \n")
                

        print(tp/(tp+fp), " Loose Precision at 1 with all  masking ")
        print(etp/(etp+efp), "Exact Precision at 1 with all masking ")

        sys.stdout=originalout

if __name__ == '__main__':
    dataTuple = build_index()
    print("Completed building index.")
    evaluate_neighbors(dataTuple[0], dataTuple[1], dataTuple[2], None)#dataTuple[3])
