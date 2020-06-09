if __name__ == '__main__':
	dupeMap = {}
	lineTotal = 0
	with open('./embeddings.csv', 'r') as embedcsv:
		for line in embedcsv:
			lineTotal += 1
			print ("Parsed a line!")
			adjustedline = line.rstrip()
			splitline = adjustedline.split(',')
			embeddingTuple = tuple(splitline)
			if embeddingTuple in dupeMap:
				print("Found a duplicate!")
				print(embeddingTuple)
				break
			else:
				dupeMap[embeddingTuple] = ''	
	print("There are", lineTotal, "lines!")

	with open('./embeddingtolabel.csv', 'r') as labelcsv:
		
