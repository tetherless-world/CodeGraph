from soupclean import clean_text
import os
from multiprocessing import Pool

def parse_text():
#	jsonFileDir = "../../../../data/data/datascience_stackexchange_graph_v2/all/first500"
	jsonFileDir = "../../../../data/data/datascience_stackexchange_graph_v2/all"
	files = [i for i in os.listdir(jsonFileDir)]
	print('Total number of files is', len(files))
#	files = files[0:150]
	files.sort()
	fullFiles = [jsonFileDir +'/' + target for target in files]
#	cleanedTexts = [clean_text(targetFile) for targetFile in fullFiles]
	workerPool = Pool(48)
	cleanedTexts = workerPool.map(clean_text, fullFiles)
	textList = []
	labelList = []
	for i in range(0, 500):
		textList.append(cleanedTexts[i])
		labelList.append(files[i])
		'''for target in files:
		print("Cleaned", cleanedFiles, "files out of", numFiles)
		actualpath = jsonFileDir + '/' + target
		cleaned_text = clean_text(actualpath)
		textList.append(cleaned_text)
		labelList.append(target)
		cleanedFiles += 1'''
	return (textList, labelList)
print()
print()
result = parse_text()
print(result[1][1])
print(result[0][1])
