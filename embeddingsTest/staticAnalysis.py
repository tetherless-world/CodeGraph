
import ijson
import tensorflow_hub as hub
import faiss
import numpy as np
from bs4 import BeautifulSoup
import sys
import re


def build_index_stack():
    embed = hub.load('https://tfhub.dev/google/universal-sentence-encoder/4')
    index = faiss.IndexFlatL2(512)
    docMessages = []
    embeddingtolabelmap = {}
    labeltotextmap = {}
    embedCollect = set()
    duplicateClassDocString=set()
    with open('../../data/codeGraph/stackoverflow_questions_per_class_func_3M_filtered_new.json', 'r') as data:
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
                code.decompose() # this whole block might be unnecessary
            docStringText = soup.get_text()
            if docStringText in embedCollect:
                if docLabel in duplicateClassDocString:
                    pass
                    
                else:
                    duplicateClassDocString.add(docLabel)
                    embeddedDocText = embed([docStringText])[0] #this whole section might also be unnecessary
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

def build_index_docs():
    embed = hub.load('https://tfhub.dev/google/universal-sentence-encoder/4')
    index = faiss.IndexFlatL2(512)
    docMessages = []
    embeddingtolabelmap = {}
    labeltotextmap = {}
    embedCollect = set()
    duplicateClassDocString=set()
    with open('../../data/codeGraph/merge-15-22.2.format.json', 'r') as data:
        jsonCollect = ijson.items(data, 'item')
        i = 0
        for jsonObject in jsonCollect:
            if 'class_docstring' not in jsonObject:
                continue
            label = jsonObject['klass']
            docLabel = label
            docStringText = jsonObject['class_docstring']# + ' ' + str(i)
            soup = BeautifulSoup(docStringText, 'html.parser')
            for code in soup.find_all('code'):
                code.decompose() # this whole block might be unnecessary
            docStringText = soup.get_text()
            if docStringText in embedCollect:
                if docLabel in duplicateClassDocString:
                    pass
                    
                else:
                    duplicateClassDocString.add(docLabel)
                    embeddedDocText = embed([docStringText])[0] #this whole section might also be unnecessary
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


def build_static_map():
    staticMap = {}
    with open('../../data/codeGraph/usage.txt', 'r') as staticData:
        matchString = '(.+) (\d+) \[(.+)\]'
        for line in staticData:
            pattern = re.compile(matchString)
            adjustedLine = pattern.match(line)
            if adjustedLine == None:
                print("Found violation.")
                print(line)
            separateClasses = adjustedLine.group(3).strip().split(', ')
            staticMap[adjustedLine.group(1)] = separateClasses
    return staticMap

def testValuation(index, staticMap, docMessages, embeddingtolabelmap, labeltotextmap):
    with open('../../data/codeGraph/merge-15-22.2.format.json', 'r') as inputData:
        k = 10
        embed = hub.load('https://tfhub.dev/google/universal-sentence-encoder/4')
        encounteredClasses = set()
        jsonCollect = ijson.items(inputData, 'item')
        totaldocs = 0
        for jsonObject in jsonCollect:
            if 'class_docstring' in jsonObject:
                docText = jsonObject['class_docstring'] 
                classLabel = jsonObject['klass']
                soup = BeautifulSoup(docText, 'html.parser')
                for code in soup.find_all('code'):
                    code.decompose()
                docText = soup.get_text()
#                print('Class associated with post:', classLabel)
#                print('Text of docstring:', docText)
                splitLabel = classLabel.lower().split('.')
    #             wholePattern = re.compile(classLabel.lower(), re.IGNORECASE)
    #             maskedText = wholePattern.sub(' ', stackText)
    #             for labelPart in splitLabel:
    #                     partPattern = re.compile(labelPart, re.IGNORECASE)
    #                     maskedText = partPattern.sub(' ', maskedText)#maskedText.replace(labelPart, ' ')
    #             print('Text of post after masking:', maskedText)
                embeddedText = embed([docText])#[maskedText])
                embeddingVector = embeddedText[0]
                embeddingArray = np.asarray(
                    embeddingVector, dtype=np.float32).reshape(1, -1)
                D, I = index.search(embeddingArray, k)
                distances = D[0]
                indices = I[0]
    #             print("Distances of related vectors:", distances)
    #             print("Indices of related vectors:", indices)
                if classLabel in staticMap:
                    viableClasses = staticMap[classLabel]
                else:
#                    print("Skipped due to not being in analysis file.")
    #                input()
                    continue
                if len(viableClasses) < 2:
#                    print("Skipped due to analysis classes being too small")
    #                input()
                    continue
                if len(viableClasses) > 10:
#                    print("Skipped due to analysis classes being too large")
    #                input()
                    continue
#                print("And statically analyzed similar classes are:", viableClasses)
                for p in range(0, k):
                    properIndex = indices[p]
                    embedding = docMessages[properIndex]
                    adjustedembedding = tuple(embedding)
                    labelList = embeddingtolabelmap[adjustedembedding]
#                    print("List of Labels related in", p, "position", labelList)
                    for label in labelList:
                        if label in viableClasses:
                            print("And found in statically related classes")
                    ##multiple docstrings associated with the same embedding mapped
                    ##array of labels mapped
                    j = 0 
                totaldocs += 1
    #            input()
        print("Why are we here?")
        print("Total number of unskipped docstrings", totaldocs)


def evaluate_neighbors_docs(index, staticMap, docMessages, embeddingtolabelmap, labeltotextmap):
    k = 10
    totaldocs=0
    embed = hub.load('https://tfhub.dev/google/universal-sentence-encoder/4')
    encounteredClasses = set()
    with open('../../data/codeGraph/merge-15-22.2.format.json', 'r') as data:
        jsonCollect = ijson.items(data, 'item')
        percents = []
        for jsonObject in jsonCollect:
            if 'class_docstring' not in jsonObject:
                continue
            classLabel = jsonObject['klass']
            if classLabel in encounteredClasses: #this might not be what we want to do,
            # the problem is that we have multiple classes for a given docstring
            # so maybe we want to have the docstring text being identical as the condition?
                print("FYI we skipped one")
#                input()
                continue
            else:
                encounteredClasses.add(classLabel)
            docText = jsonObject['class_docstring']
            soup = BeautifulSoup(docText, 'html.parser')
            for code in soup.find_all('code'): #probably don't need this
                code.decompose()
            docText = soup.get_text()
#            print('Class associated with docstring:', classLabel)
#            print('Text of docstring:', docText)
#             wholePattern = re.compile(classLabel.lower(), re.IGNORECASE)
#             maskedText = wholePattern.sub(' ', stackText)
#             for labelPart in splitLabel:
#                     partPattern = re.compile(labelPart, re.IGNORECASE)
#                     maskedText = partPattern.sub(' ', maskedText)#maskedText.replace(labelPart, ' ')
#             print('Text of post after masking:', maskedText)
            embeddedText = embed([docText])#[maskedText])
            embeddingVector = embeddedText[0]
            embeddingArray = np.asarray(
                embeddingVector, dtype=np.float32).reshape(1, -1)
            D, I = index.search(embeddingArray, k+1)
            distances = D[0]
            indices = I[0]
#             print("Distances of related vectors:", distances)
#             print("Indices of related vectors:", indices)
            if classLabel in staticMap:
                viableClasses = staticMap[classLabel]
            else:
#                print("Skipped due to not being in analysis file.")
#                input()
                continue
            if len(viableClasses) < 2:
#                print("Skipped due to analysis classes being too small")
#                input()
                continue
            if len(viableClasses) > 10:
#                print("Skipped due to analysis classes being too large")
#                input()
                continue
#            print("And statically analyzed similar classes are:", viableClasses)
            correctMatch = 0
            for p in range(1, k+1):#weird adjustments here to skip first result
                properIndex = indices[p]
                embedding = docMessages[properIndex]
                adjustedembedding = tuple(embedding)
                labelList = embeddingtolabelmap[adjustedembedding]
#                print("List of Labels related in", p, "position", labelList)
                for label in labelList:
                    if label in viableClasses:
#                        print("And found in statically related classes")
                        correctMatch += 1
                ##multiple docstrings associated with the same embedding mapped
                ##array of labels mapped
                correctPercent = correctMatch/k
                if correctPercent > 1:
                    print("THIS IS NOT SUPPOSED TO HAPPEN") # technically this could if
                    # same class is present in statically analyzed classes
            totaldocs += 1
            percents.append(correctPercent)
        print("Total number of unskipped docstrings", totaldocs)
        print("Average correct percent matched:", sum(percents)/len(percents))
                    

        #sys.stdout=originalout

                


def evaluate_neighbors_stack(index, staticMap, docMessages, embeddingtolabelmap, labeltotextmap):
    k = 10
    totaldocs=0
    embed = hub.load('https://tfhub.dev/google/universal-sentence-encoder/4')
    encounteredClasses = set()
    with open('../../data/codeGraph/stackoverflow_questions_per_class_func_3M_filtered_new.json', 'r') as data:
        jsonCollect = ijson.items(data, 'results.bindings.item')
        for jsonObject in jsonCollect:
            objectType = jsonObject['class_func_type']['value'].replace(
                'http://purl.org/twc/graph4code/ontology/', '')
            if objectType != 'Class':
                continue
            title = jsonObject['title']['value']
            classLabel = jsonObject['class_func_label']['value']
            if classLabel in encounteredClasses: #this might not be what we want to do,
            # the problem is that we have multiple classes for a given docstring
            # so maybe we want to have the docstring text being identical as the condition?
                print("FYI we skipped one")
#                input()
                continue
            else:
                encounteredClasses.add(classLabel)
            docText = jsonObject['docstr']['value']
            soup = BeautifulSoup(docText, 'html.parser')
            for code in soup.find_all('code'):
                code.decompose()
            docText = soup.get_text()
            print('\nTitle of Stack Overflow Post:', title)
            print('Class associated with post:', classLabel)
            print('Text of docstring:', docText)
            splitLabel = classLabel.lower().split('.')
#             wholePattern = re.compile(classLabel.lower(), re.IGNORECASE)
#             maskedText = wholePattern.sub(' ', stackText)
#             for labelPart in splitLabel:
#                     partPattern = re.compile(labelPart, re.IGNORECASE)
#                     maskedText = partPattern.sub(' ', maskedText)#maskedText.replace(labelPart, ' ')
#             print('Text of post after masking:', maskedText)
            embeddedText = embed([docText])#[maskedText])
            embeddingVector = embeddedText[0]
            embeddingArray = np.asarray(
                embeddingVector, dtype=np.float32).reshape(1, -1)
            D, I = index.search(embeddingArray, k)
            distances = D[0]
            indices = I[0]
#             print("Distances of related vectors:", distances)
#             print("Indices of related vectors:", indices)
            if classLabel in staticMap:
                viableClasses = staticMap[classLabel]
            else:
                print("Skipped due to not being in analysis file.")
#                input()
                continue
            if len(viableClasses) < 2:
                print("Skipped due to analysis classes being too small")
#                input()
                continue
            if len(viableClasses) > 10:
                print("Skipped due to analysis classes being too large")
#                input()
                continue
            print("And statically analyzed similar classes are:", viableClasses)
            for p in range(0, k):
                properIndex = indices[p]
                embedding = docMessages[properIndex]
                adjustedembedding = tuple(embedding)
                labelList = embeddingtolabelmap[adjustedembedding]
                print("List of Labels related in", p, "position", labelList)
                for label in labelList:
                    if label in viableClasses:
                        print("And found in statically related classes")
                ##multiple docstrings associated with the same embedding mapped
                ##array of labels mapped
                j = 0 
            totaldocs += 1
#            input()
        print("Total number of unskipped docstrings", totaldocs)
                    

        #sys.stdout=originalout

if __name__ == '__main__':
    dataTuple = build_index_docs()
    print("Completed building index.")
    staticAnalysis = build_static_map()
    evaluate_neighbors_docs(dataTuple[0], staticAnalysis, dataTuple[1], dataTuple[2], None)#dataTuple[3])
#    testValuation(dataTuple[0], staticAnalysis, dataTuple[1], dataTuple[2], None)#dataTuple[3])

