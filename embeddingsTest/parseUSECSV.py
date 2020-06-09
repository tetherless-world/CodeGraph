if __name__ == '__main__':
	embeddingspath = '../../data/embeddings.csv'
	with open(embeddingspath, 'r') as embeddings:
		i = 0
		for line in embeddings:
			if i == 1000000:
				break
			newline = line.rstrip()
			parsedline = newline.split(',')
			i += 1
