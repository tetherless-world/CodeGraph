import numpy as np
import re

# mrOutput is filename of MRJob output to compute the matrix from
def computeMatrix(mrOutput):
	inputFile = open(mrOutput, 'r')
	docList = set()
	wordList = []
	docAndWordDict = {}
	totalEntries = 0
	for line in inputFile:
		totalEntries += 1
	currentEntry = 1
	inputFile.close()
	inputFile = open(mrOutput, 'r')
	for line in inputFile:
		'''if currentEntry%50 == 0:
			print("Parsing entry", currentEntry, "out of", totalEntries)'''
		searchedLine = re.search('\["(.+)", "(.+\.json)"\]\t(.+)', line)
		try:
			wordName = searchedLine.group(1)
			docName = searchedLine.group(2)
			num = searchedLine.group(3)	
		except:
			print("This line is not parseable by the regex", line)
			continue
		trueNum = float(num)
		docList.add(docName)
		if wordName not in wordList:
			wordList.append(wordName)
		if docName not in docAndWordDict:
			docAndWordDict[docName] = {}
			docAndWordDict[docName][wordName] = trueNum
		else:
			docAndWordDict[docName][wordName] = trueNum
		currentEntry += 1

	wholeMatrix = []
	totalDocs = 0
	for document in docList:
		totalDocs += 1
	currentDoc = 1
	for document in docList:
		'''if currentDoc%5 == 0:
			print("Going through column for document", currentDoc, "out of", totalDocs)'''
		data = []
		for word in wordList:
			if word in docAndWordDict[document]:	
				data.append(docAndWordDict[document][word])
			else:	
				data.append(0)
		wholeMatrix.append(data)
		currentDoc += 1
	finalMatrix = np.array(wholeMatrix)
	inputFile.close()
	return finalMatrix

if __name__ == "__main__":
	testMatrix = computeMatrix("output_try2")
