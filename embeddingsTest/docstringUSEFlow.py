import docstringUSEEmbed as de

if __name__ == '__main__':
	itemMap = de.clean_docstrings()
	with open('../../data/codeGraph/embeddings.csv', 'w') as embedcsv:
		for embedding in itemMap[0]:
			for i in range(0,511):
				embedcsv.write(str(embedding[i]))
				embedcsv.write(',')
			embedcsv.write(str(embedding[511]))
			embedcsv.write('\n')	
	with open('../../data/codeGraph/embeddingtolabel.csv', 'w') as labelcsv:
		for embedding, label in itemMap[1].items():
			for i in range(0,511):
				labelcsv.write(str(embedding[i]))
				labelcsv.write(';')
			labelcsv.write(str(embedding[511]))
			labelcsv.write(',')
			labelcsv.write(label)
			labelcsv.write('\n')
	#docstring text may contain other things that may invalidate the csv, I tried
	#to remove as many as possible
	with open('../../data/codeGraph/labeltotext.csv', 'w') as textcsv:
		for label, text in itemMap[2].items():
			text = text.replace('\n', ' ')
			text = text.replace(',', ';')
			text = text.replace('\r', ' ')
			textcsv.write(label)
			textcsv.write(',')
			textcsv.write(text)
			textcsv.write('\n')
