from soupclean import clean_text
from gensim.models.doc2vec import Doc2Vec, TaggedDocument


test_files = ['../scipy.linalg.det.json', '../numpy.reshape.json', '../numpy.linalg.linalg.inv.json', '../sklearn.metrics.scorer.f1_score.json', '../sklearn.linear_model.LinearRegression.json', '../sklearn.linear_model.logistic.softmax.json', '../sklearn.metrics.regression.mean_squared_error.json']
for i in range(len(test_files)):
	print("Document", test_files[i], "has tag", i)
data = []
for i in range(len(test_files)):
	data.append(TaggedDocument(clean_text(test_files[i]), [test_files[i]]))
model = Doc2Vec(size=20, alpha=0.025, min_alpha=0.00025, min_count=1, dm=1)
model.build_vocab(data)

for epoch in range(100):
	print('iteration {0}'.format(epoch))
	model.train(data, total_examples=model.corpus_count, epochs = model.iter)
	model.alpha -= 0.0002
	model.min_alpha = model.alpha
model.save("output.model")
print("Model Saved")
