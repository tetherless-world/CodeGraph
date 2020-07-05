##docstring and stackoverflow texts analysed, old methods of evaluation
import ijson
import tensorflow_hub as hub
import faiss
import numpy as np
from bs4 import BeautifulSoup

def build_index():
	embed = hub.load('https://tfhub.dev/google/universal-sentence-encoder/4')
	index = faiss.IndexFlatL2(512)
	docMessages = []
	embeddingtolabelmap = {}
	labeltotextmap = {}
	with open('../../data/codeGraph/stackoverflow_questions_per_class_func_1M_filtered.json', 'r') as data:
		jsonCollect = ijson.items(data, 'results.bindings.item')
		i = 0
		for jsonObject in jsonCollect:
			objectType = jsonObject['class_func_type']['value'].replace('http://purl.org/twc/graph4code/ontology/', '')
			if objectType != 'Class':
				continue
			label = jsonObject['class_func_label']['value']
			docLabel = label + " docstring " + str(i)
			docStringText = jsonObject['docstr']['value'] + ' ' + str(i)
			soup = BeautifulSoup(docStringText, 'html.parser')
			for code in soup.find_all('code'):
				code.decompose()
			docStringText = soup.get_text()
			embeddedDocText = embed([docStringText])[0]
			newText = np.asarray(embeddedDocText, dtype = np.float32).reshape(1, -1)
			index.add(newText)
			docMessages.append(embeddedDocText.numpy().tolist())
			embeddingtolabelmap[tuple(embeddedDocText.numpy().tolist())] = docLabel 
			labeltotextmap[docLabel] = docStringText

			stackLabel = label + " stack " + str(i)
			stackQuestion = jsonObject['content']['value']
			stackAnswer = jsonObject['answerContent']['value']
			stackText = stackQuestion + " " + stackAnswer + ' ' + str(i)
			soup = BeautifulSoup(stackText, 'html.parser')
			for code in soup.find_all('code'):
				code.decompose()
			stackText = soup.get_text()
			embeddedStackText = embed([stackText])[0]
			newStackText = np.asarray(embeddedStackText, dtype=np.float32).reshape(1,-1)
			index.add(newStackText)
			docMessages.append(embeddedStackText.numpy().tolist())
			embeddingtolabelmap[tuple(embeddedStackText.numpy().tolist())] = stackLabel
			labeltotextmap[stackLabel] = stackText
			i += 1
		return (index, docMessages, embeddingtolabelmap, labeltotextmap)

def evaluate_neighbors(index, docMessages, embeddingtolabelmap, labeltotextmap):
	k = 11
	embed = hub.load('https://tfhub.dev/google/universal-sentence-encoder/4')
	with open('../../data/codeGraph/stackoverflow_questions_per_class_func_1M_filtered.json', 'r') as data:
		jsonCollect = ijson.items(data, 'results.bindings.item')
		for jsonObject in jsonCollect:
			objectType = jsonObject['class_func_type']['value'].replace('http://purl.org/twc/graph4code/ontology/', '')
			if objectType != 'Class':
				continue
			title = jsonObject['title']['value']
			classLabel = jsonObject['class_func_label']['value']
			stackText = jsonObject['content']['value'] + " " + jsonObject['answerContent']['value']
			soup = BeautifulSoup(stackText, 'html.parser')
			for code in soup.find_all('code'):
				code.decompose()
			stackText = soup.get_text()
			print('\nTitle of Stack Overflow Post:', title)
			print('Class associated with post:', classLabel)
			print('Text of post:', stackText)

			print("\nAnalyzing stackoverflow portion:")
			
			embeddedText = embed([stackText])
			embeddingVector = embeddedText[0]
			embeddingArray = np.asarray(embeddingVector, dtype = np.float32).reshape(1,-1)
			D, I = index.search(embeddingArray, k)
			distances = D[0]
			indices = I[0]
			print("Distances of related vectors:", distances)
			print("Indices of reltaed vectors:", indices)
			
			for p in range(0, k):
				properIndex = indices[p]
				embedding = docMessages[properIndex]
				adjustedembedding = tuple(embedding)
				label = embeddingtolabelmap[adjustedembedding]
				print('\nName of document in ranking order', p, 'is:', label)
				print('\nText of document', p, 'is:', labeltotextmap[label])
			input()

if __name__ == '__main__':
	dataTuple = build_index()	
	evaluate_neighbors(dataTuple[0], dataTuple[1], dataTuple[2], dataTuple[3])
