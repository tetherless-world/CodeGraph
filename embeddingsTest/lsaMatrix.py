import numpy as np
import re

def computeMatrix(mrOutput):
	inputFile = open(mrOutput, 'r')
	docList = set()
	wordList = []
	docAndWordDict = {}
	for line in inputFile:
		searchedLine = re.search('\["(.+)", "(.+\.json)"\]\t(.+)', line)
		wordName = searchedLine.group(1)
		docName = searchedLine.group(2)
		num = searchedLine.group(3)
		trueNum = float(num)
		docList.add(docName)
		if wordName not in wordList:
			wordList.append(wordName)
		if docName not in docAndWordDict:
			docAndWordDict[docName] = {}
			docAndWordDict[docName][wordName] = trueNum
		else:
			docAndWordDict[docName][wordName] = trueNum

	wholeMatrix = []
	for document in docList:
		data = []
		for word in wordList:
			if word in docAndWordDict[document]:
				data.append(docAndWordDict[document][word])
			else:
				data.append(0)
		wholeMatrix.append(data)
	finalMatrix = np.array(wholeMatrix)
	inputFile.close()
	return finalMatrix
