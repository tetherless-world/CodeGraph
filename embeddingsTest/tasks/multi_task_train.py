"""
This is an example how to train SentenceTransformers in a multi-task setup.
The system trains BERT on the AllNLI and on the STSbenchmark dataset.
"""
from torch.utils.data import DataLoader
import math
from sentence_transformers import models, losses, InputExample
from sentence_transformers import SentencesDataset, LoggingHandler, SentenceTransformer, util
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, BinaryClassificationEvaluator
from sentence_transformers.readers import *
from sentence_transformers.cross_encoder import CrossEncoder
#from sentence_transformers.cross_encoder.evaluation import CERerankingEvaluator
import logging
from datetime import datetime
import gzip
import csv
import os
import sys
import json
from sklearn.model_selection import train_test_split

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

def create_hirerachy_examples(fl):
    train_hierarchy_samples = []
    with open(fl) as f:
        data = json.load(f)
        max_distance = 0
        for obj in data:
            if obj['distance'] > max_distance:
                max_distance = obj['distance']
        for obj in data:
            # flip the meaning of similarity, since the more distant the two classes, the closer to 0 it should be
            dist = (max_distance - obj['distance']) / max_distance
            train_hierarchy_samples.append(InputExample(texts=[obj['class1'], obj['class2']], label=dist))
    return train_hierarchy_samples

def create_linked_posts(fl):
    train_linked_posts = []
    with open(fl) as f:
        data = json.load(f)
        for obj in data:
            if obj['class'] == 'relevant':
                label = 1
            else:
                label = 0

            train_linked_posts.append(InputExample(texts=[obj['text_1'], obj['text_2']], label=label))
    return train_linked_posts

def create_train_class_posts(fl):
    train_class_posts = []
    with open(fl) as f:
        data = json.load(f)
        for obj in data:
            train_class_posts.append(InputExample(texts=[obj['docstring'], obj['text']], label=obj['label']))
    return train_class_posts

def create_train_usage(fl):
    train_usage = []
    with open(fl) as f:
        data = json.load(f)
        min_d = 10000000
        max_d = 0
        for obj in data:
            dist = obj['distance']
            if dist < min_d:
                min_d = dist
            if dist > max_d:
                max_d = dist
        for obj in data:
            dist = (max_d - obj['distance']) / (max_d - min_d)
            train_usage.append(InputExample(texts=[obj['class1'], obj['class2']], label=dist))
    return train_usage

def create_posts_ranking(fl):
    all_posts_ranking = []
    with open(fl) as f:
        data = json.load(f)
        for obj in data:
            answers = obj['answers']
            for answer in answers:
                dist = (len(answers) - answer['a_rank']) / len(answers)
                all_posts_ranking.append(
                    InputExample(texts=[obj['q_text'], answer['a_text']], label=answer['a_rank'] / len(answers)))
    return all_posts_ranking

def create_search(collection, query_file, train):
    corpus = {}
    with open(collection, 'r', encoding='utf8') as fIn:
        for line in fIn:
            pid, passage = line.strip().split("\t")
            corpus[pid] = passage

    queries = {}
    with open(query_file, 'r', encoding='utf8') as fIn:
        for line in fIn:
            qid, query = line.strip().split("\t")
            queries[qid] = query

    train_search = []
    with open(train, 'r', encoding='utf8') \
            as f:
        added_q = set()
        for line in f.readlines():
            qid, pos_id, neg_id = line.strip().split()
            query = queries[qid]
            passage = corpus[pos_id]
            neg_passage = corpus[neg_id]
            if qid not in added_q:
                train_search.append(InputExample(texts=[query, passage], label=1))
                added_q.add(qid)
            train_search.append(InputExample(texts=[query, neg_passage], label=0))
    return train_search

# Read the dataset

if __name__ == '__main__':
    
    model_name = sys.argv[1]
    batch_size = 16
    model_save_path = sys.argv[2] + model_name+'-'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


    # Use BERT for mapping tokens to embeddings
    word_embedding_model = models.Transformer(model_name)

    # Apply mean pooling to get one fixed sized sentence vector
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                               pooling_mode_mean_tokens=True,
                               pooling_mode_cls_token=False,
                               pooling_mode_max_tokens=False)

    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    # task 1 - class hierarchy prediction
    train_hierarchy_samples = create_hirerachy_examples('/data/blanca/hierarchy_train.json')
    train_data_hierarchy = SentencesDataset(train_hierarchy_samples, model=model)
    train_dataloader_hierarchy = DataLoader(train_data_hierarchy, shuffle=True, batch_size=batch_size)
    train_loss_hierarchy = losses.CosineSimilarityLoss(model=model)

    # task 2 - determine if two posts are linked
    train_linked_posts = create_linked_posts('/data/blanca/stackoverflow_data_linkedposts__train.json')
    train_data_linked_posts = SentencesDataset(train_linked_posts, model=model)
    train_dataloader_linked_posts = DataLoader(train_data_linked_posts, shuffle=True, batch_size=batch_size)
    train_loss_linked_posts = losses.ContrastiveLoss(model=model)

    # task 3 - determine if a post is related to a class's docstring
    train_class_posts = create_train_class_posts('/data/blanca/class_posts_train_data.json')
    train_data_class_posts = SentencesDataset(train_class_posts, model=model)
    train_dataloader_class_posts = DataLoader(train_data_class_posts, shuffle=True, batch_size=batch_size)
    train_loss_class_posts = losses.ContrastiveLoss(model=model)

    # task 4 - class usage prediction
    train_usage = create_train_usage('/data/blanca/usage_train.json')
    train_data_usage = SentencesDataset(train_usage, model=model)
    train_dataloader_usage = DataLoader(train_data_usage, shuffle=True, batch_size=batch_size)
    train_loss_usage = losses.CosineSimilarityLoss(model=model)

    # task 5 - predict ranks of a post's answers
    all_posts_ranking = create_posts_ranking('/data/blanca/stackoverflow_data_ranking_v2_train.json')
    train_posts_ranking, dev_posts_ranking = train_test_split(all_posts_ranking, test_size = 0.1)
    train_data_posts_ranking = SentencesDataset(train_posts_ranking, model=model)
    train_dataloader_posts_ranking = DataLoader(train_data_posts_ranking, shuffle=True, batch_size=batch_size)
    train_loss_posts_ranking = losses.CosineSimilarityLoss(model=model)

    # task 6 - predict search rank of a post
    train_search = create_search('/data/blanca/stackoverflow_matches_codesearchnet_5k_train_collection.tsv',
                             '/data/blanca/stackoverflow_matches_codesearchnet_5k_train_queries.tsv',
                             '/data/blanca/stackoverflow_matches_codesearchnet_5k_train_blanca-qidpidtriples.train.tsv')

    # We create a DataLoader to load our train samples
    train_dataloader_search = DataLoader(train_search, shuffle=True, batch_size=batch_size)
    train_loss_search = losses.ContrastiveLoss(model=model)

    logging.info("Use post ranking as the dev task")
    evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_posts_ranking, name='posts-ranking-dev')

    # Configure the training
    num_epochs = 10

    warmup_steps = math.ceil(len(train_posts_ranking) * num_epochs / batch_size * 0.1) #10% of train data for warm-up
    logging.info("Warmup-steps: {}".format(warmup_steps))


    train_objectives = [ (train_dataloader_hierarchy, train_loss_hierarchy),
                     (train_dataloader_linked_posts, train_loss_linked_posts),
                     (train_dataloader_class_posts, train_loss_class_posts),
                     (train_dataloader_usage, train_loss_usage),
                     (train_dataloader_posts_ranking, train_loss_posts_ranking)]

    # Train the model
    model.fit(train_objectives=train_objectives,
          evaluator=evaluator,
          epochs=num_epochs,
          evaluation_steps=1000,
          warmup_steps=warmup_steps,
          output_path=model_save_path
          )



  
