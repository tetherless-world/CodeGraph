from soupclean import clean_text
import os

def parse_text():
	jsonFileDir = "../../../../data/data/datascience_stackexchange_graph_v2/all/first500"
	files = [i for i in os.listdir(jsonFileDir)]
	files.sort()
	textList = []
	labelList = []
	numFiles = len(files)
	cleanedFiles = 0
	for target in files:
		print("Cleaned", cleanedFiles, "files out of", numFiles)
		actualpath = jsonFileDir + '/' + target
		cleaned_text = clean_text(actualpath)
		textList.append(cleaned_text)
		labelList.append(target)
		cleanedFiles += 1
	return (textList, labelList)
