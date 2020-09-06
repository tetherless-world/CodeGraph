
import ijson
import tensorflow_hub as hub
import faiss
import numpy as np
from bs4 import BeautifulSoup
import sys
import re
import sys
from metrics_eval import ranking_metrics
from utils import util

embedType = 'USE'

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


def build_static_map(usagePath, docsToClasses):
    staticMap = {}
    with open(usagePath, 'r') as staticData:
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


def compute_neighbor_docs(query_distances, query_neighbors, index, docList, docsToClasses, embeddedDocText):
    classesToNeighbors = {}
    
    for docStringIndex, embeddedDocStringNeighbors in enumerate(query_neighbors):
        docString = docList[ docStringIndex ]

        allNeighborClasses = set()
        for neighborDocStringIndex in embeddedDocStringNeighbors:
            neighborClasses = docsToClasses[ docList[ neighborDocStringIndex ] ]
            for neighborClass in neighborClasses:
                allNeighborClasses.add(neighborClass)

        for klass in docsToClasses[ docList[docStringIndex] ]:
            classesToNeighbors[ klass ] = allNeighborClasses

    return classesToNeighbors


def compareOverlap(mapTuple, staticMap):
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
                overlapNumerator += 1
            elif usageClass in classChildren:
                overlapNumerator += 1
            else:
                for siblingList in classSiblingLL:
                    if usageClass in siblingList:
                        overlapNumerator += 1
                        break
        # below are some optional metrics that you can uncomment to enable
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

def evaluate_static_analysis(classesToDocs, docstringsToDocstringNeighbors, usagePath):
    with open(usagePath, 'r') as staticData:
        maxCount = 0
        maxId = 0
        ids = {}
        predicted = {}
        expected = {}
        matchString = '(.+) (\d+) \[(.+)\]'
        for line in staticData:
            pattern = re.compile(matchString)
            adjustedLine = pattern.match(line)
            if adjustedLine == None:
                print("Found violation.")
                print(line)
            count = int(adjustedLine.group(2))
            if count > maxCount:
                maxCount = count

            if adjustedLine.group(1) in classesToDocs:
                otherDocstrings = set()
                thisone = classesToDocs[ adjustedLine.group(1) ]
                myneighbors = docstringsToDocstringNeighbors[ thisone ]
                overlap = 0
                outside = 0
                otherClasses = adjustedLine.group(3).strip().split(', ')
                for otherClass in otherClasses:
                    if otherClass in classesToDocs:
                        otherDocstrings.add(classesToDocs[otherClass])
                        if classesToDocs[otherClass] in myneighbors:
                            overlap = overlap + 1
                        else:
                            outside = outside + 1

                print(str(adjustedLine.group(1)) + " " + str(overlap) + " " + str(outside) + " " + str(count))

                expectedIds = []
                for n in otherDocstrings:
                    if n in ids:
                        expectedIds.append(ids[n])
                    else:
                        ids[n] = maxId
                        expectedIds.append(maxId)
                        maxId = maxId + 1

                if len(expectedIds) == 0:
                    continue
                
                if count in expected:
                    expected[count].append(np.array(expectedIds))
                else:
                    expected[count] = [np.array(expectedIds)]
                    
                predictedIds = []
                for n in myneighbors:
                    if n in ids:
                        predictedIds.append(ids[n])
                    else:
                        ids[n] = maxId
                        predictedIds.append(maxId)
                        maxId = maxId + 1

                if count in predicted:
                    predicted[count].append(np.array(predictedIds))
                else:
                    predicted[count] = [np.array(predictedIds)]

    countExpected = []
    countPredicted = []
    for i in range(maxCount, 0, -1):
        if i in predicted:
            countExpected.extend(expected[i])
            countPredicted.extend(predicted[i])
            print("size: " + str(len(countExpected)))
            print(str(i) + ": mrr: " + str(ranking_metrics.mrr(countExpected, countPredicted)))
            print(str(i) + ": map@10: " + str(ranking_metrics.map(countExpected, countPredicted, 10)))

            
if __name__ == '__main__':
    if len(sys.argv) > 5:
        embedType = sys.argv[5]
        
    util.get_model(embedType)
    hierarchyPath = sys.argv[1]
    docPath = sys.argv[2]
    classPath = sys.argv[3]
    usagePath = sys.argv[4]
#    hierarchyMaps = build_sibling_maps(hierarchyPath)
    (index, docList, docsToClasses, embeddedDocText, classesToDocs) = util.build_index_docs(docPath, embedType)
    top_k = 10
    query_distances, query_neighbors = index.search(embeddedDocText, top_k+1)
    classesToDocstringNeighbors = compute_neighbor_docs(query_distances, query_neighbors, index, docList, docsToClasses, embeddedDocText)
    docstringsToDocstringNeighbors = util.compute_neighbor_docstrings(query_neighbors, docList)
#    print(str(classesToDocstringNeighbors))
#    classesToUsageNeighbors = build_static_map(usagePath)
#    compareOverlap(hierarchyMaps, staticAnalysis)
#    classMap = build_class_mapping(classPath)
    evaluate_static_analysis(classesToDocs, docstringsToDocstringNeighbors, usagePath);

