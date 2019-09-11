from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
import nltk
import json
import os

nltk.download('stopwords')

for d in os.listdir('/tmp/sklearn'):
    if d != 'sklearn.linear_model.json':
        continue
    seen_classes = set([])
    with open(os.path.join('/tmp/sklearn', d), 'r') as f:
        models = json.load(f)
        with open('/tmp/outfile', 'w') as out:
            for model in models:
                if model['class'] in seen_classes:
                    continue
                seen_classes.add(model['class'])
                sentence_groups = model['class_usage_doc']
                if sentence_groups is None:
                    continue
                for sentences in sentence_groups:
                    sent_tokenize_list = sent_tokenize(sentences['description'])
                    for s in sent_tokenize_list:
                        s = s.replace('\n', ' ')
                        s = nltk.word_tokenize(s)
                        out.write(' '.join(s) + ' \n')
