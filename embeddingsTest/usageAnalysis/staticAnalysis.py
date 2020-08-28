
import ijson
import tensorflow_hub as hub
import faiss
import numpy as np
from bs4 import BeautifulSoup
import sys
import re


def build_class_mapping(mapPath):
    classMap = {}
    with open(mapPath, 'r') as inputFile:
        for line in inputFile:
            lineComponents = line.rstrip().split(' ')
            if len(lineComponents) < 2:
                classMap[lineComponents[0]] = lineComponents[0]
            else:
                classMap[lineComponents[0]] = lineComponents[1]
    return classMap

def build_index_docs(docPath):
    embed = hub.load('https://tfhub.dev/google/universal-sentence-encoder/4')
    index = faiss.IndexFlatL2(512)
    docMessages = []
    embeddingtolabelmap = {}
    labeltotextmap = {}
    embedCollect = set()
    duplicateClassDocString=set()
    with open(docPath, 'r') as data:
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


def build_static_map(usagePath):
    staticMap = {}
    with open(usageData, 'r') as staticData:
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

def build_sibling_maps(hierarchyPath):
    childToParentMap = {}
    parentToChildMap = {}
    with open(hierarchyPath, 'r') as data:
        jsonCollect = ijson.items(data, 'results.bindings.item')
        for jsonObject in jsonCollect:
            child = jsonObject['class']['value']
            parent = jsonObject['superclass']['value'].replace('http://purl.org/twc/graph4code/python/', '')
            if child in childToParentMap:
                childToParentMap[child].append(parent)
            else:
                childToParentMap[child] = [parent]
            if parent in parentToChildMap:
                parentToChildMap[parent].append(child)
            else:
                parentToChildMap[parent] = [child]

    return (childToParentMap, parentToChildMap)


def evaluate_neighbors_docs(index, staticMap, docMessages, embeddingtolabelmap, labeltotextmap, classMap, docPath):
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
    with open(docPath, 'r') as data:
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
#            print("Distances of related vectors:", distances)
#            print("Indices of related vectors:", indices)
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
                    pAtK = correctMatch/k
                    precisions.append(pAtK)
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

def compareOverlap(mapTuple, staticMap):
    # don't forget to possibly add class2ClassMapping
    childToParentMap = mapTuple[0]
    parentToChildMap = mapTuple[1]
    overlapPercents = []
    for targetClass, relatedClasses in staticMap.items(): 
        if len(relatedClasses) > 10:
            continue
        overlapNumerator = 0
        overlapDenominator = 0
        overlapDenominator = len(relatedClasses)
        classParents = []
        if targetClass in childToParentMap:
            classParents = childToParentMap[targetClass]
        classChildren = []
        if targetClass in parentToChildMap:
            classChildren = parentToChildMap[targetClass]
        classSiblingLL = []
        for parent in classParents:
            if parent in parentToChildMap:
                classSiblingLL.append(parentToChildMap[parent])
        for usageClass in relatedClasses:
            if usageClass in classParents:
#                print("Matched in parents")
                overlapNumerator += 1
            elif usageClass in classChildren:
#                print("Matched in children")
                overlapNumerator += 1
            else:
                for siblingList in classSiblingLL:
                    if usageClass in siblingList:
#                        print("Matched in siblings")
                        overlapNumerator += 1
                        break
        '''print("For target class", targetClass, "we have these parents:", classParents)
        print("And these children:", classChildren)
        print("And these siblings:", classSiblingLL)
        print("And these related classes by usage", relatedClasses)
        print("And results of overlap comparison are: " + str(overlapNumerator) + \
        "/" + str(overlapDenominator))
        input()'''
        overlapPercent = overlapNumerator/overlapDenominator
        overlapPercents.append(overlapPercent)
    print("Average overlap is", sum(overlapPercents)/len(overlapPercents))

        
        



if __name__ == '__main__':
    hierarchyPath = input("Please enter path to class hierarchy data.")
    docPath = input("Please enter path to docstrings dataset.")
    classPath = input("Please enter path to old to new class conversion.")
    usagePath = input("Please enter path to usage data.")
    hierarchyMaps = build_sibling_maps(hierarchyPath)
#    dataTuple = build_index_docs(docPath)#[0,0,0,0,0]
#    print("Completed building index.")
    staticAnalysis = build_static_map(usagePath)
    compareOverlap(hierarchyMaps, staticAnalysis)
#    classMap = build_class_mapping(classPath)
#    evaluate_neighbors_docs(dataTuple[0], staticAnalysis, dataTuple[1], dataTuple[2], None, classMap, docPath)#dataTuple[3])

