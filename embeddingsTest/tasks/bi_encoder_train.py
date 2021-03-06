"""
This examples show how to train a Bi-Encoder for the MS Marco dataset (https://github.com/microsoft/MSMARCO-Passage-Ranking).
The queries and passages are passed independently to the transformer network to produce fixed sized embeddings.
These embeddings can then be compared using cosine-similarity to find matching passages for a given query.
For training, we use MultipleNegativesRankingLoss. There, we pass triplets in the format:
(query, positive_passage, negative_passage)
Negative passage are hard negative examples, that where retrieved by lexical search. We use the negative
passages (the triplets) that are provided by the MS MARCO dataset.
Running this script:
python train_bi-encoder.py
"""
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, LoggingHandler, util, models, evaluation, losses, InputExample
import logging
from datetime import datetime
import gzip
import os
import tarfile
from collections import defaultdict
from torch.utils.data import IterableDataset
import sys
import tqdm


#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

# The  model we want to fine-tune
model_name = sys.argv[1]

train_batch_size = 8           #Increasing the train batch size improves the model performance, but requires more GPU memory
epochs = 2

num_dev_queries = 20           #Number of queries we want to use to evaluate the performance while training
num_max_dev_negatives = 200     #For every dev query, we use up to 200 hard negatives and add them to the dev corpus

# We construct the SentenceTransformer bi-encoder from scratch
word_embedding_model = models.Transformer(model_name, max_seq_length=350)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

model_save_path = sys.argv[2] + '/training_bi-encoder-'+model_name.replace("/", "-")+'-'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


data_folder = sys.argv[3]

#### Read the corpus files, that contain all the passages. Store them in the corpus dict
corpus = {}         #dict in the format: passage_id -> passage. Stores all existent passages
collection_filepath = os.path.join(data_folder, 'stackoverflow_matches_codesearchnet_5k_train_collection.tsv')

with open(collection_filepath, 'r', encoding='utf8') as fIn:
    for line in fIn:
        pid, passage = line.strip().split("\t")
        corpus[pid] = passage


### Read the train queries, store in queries dict
queries = {}        #dict in the format: query_id -> query. Stores all training queries
queries_filepath = os.path.join(data_folder, 'stackoverflow_matches_codesearchnet_5k_train_queries.tsv')

with open(queries_filepath, 'r', encoding='utf8') as fIn:
    for line in fIn:
        qid, query = line.strip().split("\t")
        queries[qid] = query


train_eval_filepath = os.path.join(data_folder, 'stackoverflow_matches_codesearchnet_5k_test_blanca-qidpidtriples.train.tsv')

dev_queries = {}
dev_corpus = {}
dev_rel_docs = {}

num_negatives = defaultdict(int)

with open(train_eval_filepath, 'rt') as fIn:
    lines = fIn.readlines()
    import random
    random.shuffle(lines)

    for line in lines:
        qid, pos_id, neg_id = line.strip().split()

        if len(dev_queries) <= num_dev_queries or qid in dev_queries:
            dev_queries[qid] = queries[qid]

            #Ensure the corpus has the positive
            dev_corpus[pos_id] = corpus[pos_id]

            if qid not in dev_rel_docs:
                dev_rel_docs[qid] = set()

            dev_rel_docs[qid].add(pos_id)

            if num_negatives[qid] < num_max_dev_negatives:
                dev_corpus[neg_id] = corpus[neg_id]
                num_negatives[qid] += 1

logging.info("Dev queries: {}".format(len(dev_queries)))
logging.info("Dev Corpus: {}".format(len(dev_corpus)))


# Create the evaluator that is called during training
ir_evaluator = evaluation.InformationRetrievalEvaluator(dev_queries, dev_corpus, dev_rel_docs, name='ms-marco-train_eval')


def get_train_samples(dev_samples, queries, corpus):
    # Read our training file. qidpidtriples consists of triplets (qid, positive_pid, negative_pid)
    train_filepath = os.path.join(data_folder,
                                  'stackoverflow_matches_codesearchnet_5k_test_blanca-qidpidtriples.train.tsv')

    train_samples = []
    with open(train_filepath, 'rt') as fIn:
        for line in tqdm.tqdm(fIn, unit_scale=True):
            qid, pos_id, neg_id = line.strip().split()

            if qid in dev_samples:
                continue

            query_text = queries[qid]
            pos_text = corpus[pos_id]
            neg_text = corpus[neg_id]

            train_samples.append(InputExample(texts=[query_text, pos_text, neg_text]))
    return train_samples


# We create a DataLoader to load our train samples
train_samples = get_train_samples(queries=queries, corpus=corpus, dev_samples=dev_queries)
print(len(train_samples))
# For training the SentenceTransformer model, we need a dataset, a dataloader, and a loss used for training.
train_dataloader = DataLoader(train_samples, shuffle=False, batch_size=train_batch_size)
train_loss = losses.MultipleNegativesRankingLoss(model=model)

# Train the model
model.fit(train_objectives=[(train_dataloader, train_loss)],
          evaluator=ir_evaluator,
          epochs=epochs,
          warmup_steps=1000,
          output_path=model_save_path,
          evaluation_steps=5000,
          use_amp=True
          )

#Save latest model
model.save(model_save_path+'-latest')