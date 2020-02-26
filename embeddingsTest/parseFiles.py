from soupclean import clean_text
import os


def parse_text():
	jsonFileDir = "../../../../data/data/datascience_stackexchange_graph_v2/all"
	files = [i for i in os.listdir(jsonFileDir)]
	files.sort()
	textList = []
	labelList = []
	for target in files:
		actualpath = jsonFileDir + '/' + target
		cleaned_text = clean_text(actualpath)
		textList.append(cleaned_text)
		labelList.append(target)
	return (textList, labelList)
