from soupclean import clean_text
import os

def parse_text():
	jsonFileDir = "../../../../data/data/datascience_stackexchange_graph_v2/all"
	files = [i for i in os.listdir(jsonFileDir)]
	textList = []
	labelList = []
	i = 0
	for target in files:
		if i == 5:
			break
		else:
			i += 1
		actualpath = jsonFileDir + '/' + target
		cleaned_text = clean_text(actualpath)
		textList.append(cleaned_text)
		labelList.append(target)
	return (textList, labelList)

textTest = parse_text()
print(textTest[0][0])
print(textTest[1][0])
