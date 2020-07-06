
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
                embeddedDocText = embed([docStringText])[0]
                embeddingtolabelmap[tuple(
                embeddedDocText.numpy().tolist())].append(docLabel)
            else:
                embedCollect.add(docStringText)
                embeddedDocText = embed([docStringText])[0]
                newText = np.asarray(
                embeddedDocText, dtype=np.float32).reshape(1, -1)
                docMessages.append(embeddedDocText.numpy().tolist())
                index.add(newText)
                embeddingtolabelmap[tuple(
                embeddedDocText.numpy().tolist())] = [docLabel]
#            labeltotextmap[docLabel] = docStringText
            i += 1
        return (index, docMessages, embeddingtolabelmap)#, labeltotextmap)


def evaluate_neighbors(index, docMessages, embeddingtolabelmap, labeltotextmap):
    k = 10
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
    with open('../../data/codeGraph/stackoverflow_questions_per_class_func_1M_filtered.json', 'r') as data, open('../../data/codeGraph/resultsFromNoDupeEmbeddingDocStringThenStackOverflowWithoutMaskingPr@10.txt', 'w') as outputFile:
        jsonCollect = ijson.items(data, 'results.bindings.item')
        sys.stdout = outputFile
        for jsonObject in jsonCollect:
            totaldocs+=1
            objectType = jsonObject['class_func_type']['value'].replace(
                'http://purl.org/twc/graph4code/ontology/', '')
            if objectType != 'Class':
                continue
            title = jsonObject['title']['value']
            classLabel = jsonObject['class_func_label']['value']
            stackText = jsonObject['content']['value'] + \
                " " + jsonObject['answerContent']['value']
            soup = BeautifulSoup(stackText, 'html.parser')
            for code in soup.find_all('code'):
                code.decompose()
            stackText = soup.get_text()
            print('\nTitle of Stack Overflow Post:', title)
            print('Class associated with post:', classLabel)
            print('Text of post before masking:', stackText)
#             splitLabel = classLabel.lower().split('.')
#             wholePattern = re.compile(classLabel.lower(), re.IGNORECASE)
#             maskedText = wholePattern.sub(' ', stackText)
#             for labelPart in splitLabel:
#                     partPattern = re.compile(labelPart, re.IGNORECASE)
#                     maskedText = partPattern.sub(' ', maskedText)#maskedText.replace(labelPart, ' ')
#             print('Text of post after masking:', maskedText)

## masking removed for now

            embeddedText = embed([stackText])
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
                ##array of labels mapped
                for l in label:
                    if l.startswith(classLabel.split(".")[0]):
                        positivepresent=True
                        print("\n True positive label being contributed by \n",label)
                    if l == classLabel:
                        exactpositivepresent=True
                        print("\n Exact positive label being contributed by \n",label)
                        
                    
            if not positivepresent:
                fp=fp+1
                print("Loose False Positive Present ------------------------------------------------------- \n")
            else:
                tp=tp+1
                print("Loose True Positive Present -------------------------------------------------------- \n")
            if not exactpositivepresent:
                efp=efp+1
                print("Loose False Positive Present ------------------------------------------------------- \n")
            else:
                etp=etp+1
                print("Loose True Positive Present -------------------------------------------------------- \n")
                

        print(tp/(tp+fp), " Loose Precision at 10 without masking ")
        print(etp/(etp+efp), " Loose Precision at 10 without masking ")

        sys.stdout=originalout

if __name__ == '__main__':
    dataTuple = build_index()
    print("Completed building index.")
    evaluate_neighbors(dataTuple[0], dataTuple[1], dataTuple[2], None)#dataTuple[3])
