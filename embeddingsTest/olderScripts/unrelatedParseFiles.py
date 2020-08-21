from soupclean import clean_text
import os

def parse_text():
	jsonFileDir = "/data/data/stats_stackexchange_graph_v2/all/"
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
		if cleanedFiles == 3:
			break
	return (textList, labelList)
