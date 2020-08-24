import fivehrParseFiles as sp
import pickleFiles as pf
import os.path as op
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

if op.isfile('fivehrPickledFiles.p'):
	print("Pickled data detected")
	dataTuple = pf.load_text("fivehrPickledFiles.p")
	data = []
	for i in range(0, len(dataTuple[0])):
		data.append(TaggedDocument(dataTuple[0][i], [dataTuple[1][i]]))
	model = Doc2Vec(size=20,alpha=0.025,min_alpha=0.00025,min_count=1,dm=1, workers=48)
	model.build_vocab(data)
	for epoch in range(100):
		print('Iteration', epoch)
		model.train(data,total_examples=model.corpus_count,epochs=model.iter)
		model.alpha -= 0.0002
		model.min_alpha = model.alpha
	model.save('stressTestOutput.model')
	print("Model saved")
else:
	print("No pickled data detected.")
	dataTuple = sp.parse_text()
	data = []
	for i in range(0, len(dataTuple[0])):
		data.append(TaggedDocument(dataTuple[0][i], [dataTuple[1][i]]))
	model = Doc2Vec(size=20,alpha=0.025,min_alpha=0.00025,min_count=1,dm=1, workers=96)
	model.build_vocab(data)
	for epoch in range(100):
		print('Iteration', epoch)
		model.train(data,total_examples=model.corpus_count,epochs=model.iter)
		model.alpha -= 0.0002
		model.min_alpha = model.alpha
	model.save('stressTestOutput.model')
	print("Model saved")
