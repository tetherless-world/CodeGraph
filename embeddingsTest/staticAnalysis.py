
import ijson
import tensorflow_hub as hub
import faiss
import numpy as np
from bs4 import BeautifulSoup
import sys
import re


def build_class_mapping():
    classMap = {}
    with open('../../data/codeGraph/new_classes.map', 'r') as inputFile:
        for line in inputFile:
            lineComponents = line.rstrip().split(' ')
            if len(lineComponents) < 2:
                classMap[lineComponents[0]] = lineComponents[0]
            else:
                classMap[lineComponents[0]] = lineComponents[1]
    return classMap

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


def evaluate_neighbors_docs(index, staticMap, docMessages, embeddingtolabelmap, labeltotextmap, classMap):
    k = 10
    original = 0
    adjusted = 0
    totaldocs=0 
    notFoundSkipped = 0
    smallSkipped = 0
    largeSkipped = 0
    unSkipped = 0
    embed = hub.load('https://tfhub.dev/google/universal-sentence-encoder/4')
    encounteredClasses = set()
    with open('../../data/codeGraph/merge-15-22.2.format.json', 'r') as data:
        jsonCollect = ijson.items(data, 'item')
        averagePrecisions = []
        for jsonObject in jsonCollect:
            if 'class_docstring' not in jsonObject:
                continue
            classLabel = jsonObject['klass']
            if classLabel in encounteredClasses: #this might not be what we want to do,
            # the problem is that we have multiple classes for a given docstring
            # so maybe we want to have the docstring text being identical as the condition?
#                input()
                continue
            else:
                encounteredClasses.add(classLabel)
            docText = jsonObject['class_docstring']
            totaldocs += 1
            embeddedText = embed([docText])#[maskedText])
            embeddingVector = embeddedText[0]
            embeddingArray = np.asarray(
                embeddingVector, dtype=np.float32).reshape(1, -1)
            D, I = index.search(embeddingArray, k+1)
            distances = D[0]
            indices = I[0]
            print("Distances of related vectors:", distances)
            print("Indices of related vectors:", indices)
            adjustedLabel = classLabel
            if classLabel in classMap:
                adjustedLabel = classMap[classLabel]
                adjusted += 1
            else:
                original += 1
            if adjustedLabel in staticMap:
                viableClasses = staticMap[adjustedLabel]
            else:
#                print("Skipped due to not being in analysis file.")
#                input()
                notFoundSkipped += 1
                continue
            if len(viableClasses) < 1:
#                print("Skipped due to analysis classes being too small")
#                input()
                smallSkipped += 1
                continue
            if len(viableClasses) > 10:
#                print("Skipped due to analysis classes being too large")
#                input()
                largeSkipped += 1
                continue
#            print("And statically analyzed similar classes are:", viableClasses)
            unSkipped += 1
#            input()
            correctMatch = 0
            precisions = []
            for p in range(1, k+1):#weird adjustments here to skip first result
                properIndex = indices[p]
                embedding = docMessages[properIndex]
                adjustedembedding = tuple(embedding)
                labelList = embeddingtolabelmap[adjustedembedding]
#                print("List of Labels related in", p, "position", labelList)
                pAtK = 0
                isRelevant = 0
                for label in labelList:
                    if label in viableClasses:
#                        print("And found in statically related classes")
                        correctMatch += 1
                        isRelevant = 1
                        break
                if isRelevant == 1:
                    precisions.append(correctMatch/k)
                else:
                    precisions.append(0.0)
            if correctMatch == 0:
                averagePrecisions.append(0.0)
            else:
                averagePrecision = sum(precisions)/correctMatch
                averagePrecisions.append(averagePrecision)

        meanAveragePrecision = sum(averagePrecisions)/len(averagePrecisions)
        print("Mean average precision is", meanAveragePrecision)
        print("Number of class names kept original", original)
        print("Number of class names adjusted", adjusted)
        print("Total number of docstrings", totaldocs)
        print("Total number of not found docstrings", notFoundSkipped)
        print("Total number of small skipped", smallSkipped)
        print("Total number of large skipped", largeSkipped)
        print("Total number unskipped", unSkipped)
#        print("Average correct percent matched:", sum(percents)/len(percents))
                    

        #sys.stdout=originalout

                



if __name__ == '__main__':
    dataTuple = build_index_docs()#[0,0,0,0,0]
    print("Completed building index.")
    staticAnalysis = build_static_map()
    classMap = build_class_mapping()
    evaluate_neighbors_docs(dataTuple[0], staticAnalysis, dataTuple[1], dataTuple[2], None, classMap)#dataTuple[3])
#    testValuation(dataTuple[0], staticAnalysis, dataTuple[1], dataTuple[2], None)#dataTuple[3])

