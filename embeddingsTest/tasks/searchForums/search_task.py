#!/usr/bin/env python
# coding: utf-8

# In[1]:

import json
import tensorflow_hub as hub
import faiss
import numpy as np
import sys
import re
import copy


# In[14]:

top_k = 50
embed = hub.load('https://tfhub.dev/google/universal-sentence-encoder/4')
add_all = True

with open(sys.argv[1], 'r') as f:
    data = json.load(f)
    queries_embedding = []
    embeddings = []
    index = faiss.IndexFlatIP(512)
    all_matches = []
    all_docs = []
    qids = {}
    for obj in data:
        query = obj['query']
        queries_embedding.extend(np.asarray(embed([query])))
        for match in obj['matches']:
            qid = match['question_id']
            # posts get repeated across queries sometimes - to avoid neural
            # embeddings reproducing the same result multiple times - ignore dups
            # across queries.
            if qid in qids:
                continue
            qids[match['question_id']] = 1
            all_text = match['question_title']
            if add_all:
                all_text = all_text + ' ' + match['question_text']
                for answer in match['answers']:
                    all_text = all_text + ' ' + answer['answer_text']
            all_docs.append(all_text)
            all_matches.append((query, match))
    if add_all:
        arr = np.array_split(np.asarray(all_docs), 10)
        for x in arr:
            embeddings.extend(embed(x))
        embeddings = np.asarray(embeddings)
    else:
        embeddings = np.asarray(embed(all_docs))
    num_queries = len(queries_embedding)
    queries_embedding = np.asarray(queries_embedding)
    index.add(embeddings)
    faiss.normalize_L2(embeddings)
    faiss.normalize_L2(queries_embedding)
    query_distances, query_neighbors = index.search(queries_embedding, top_k)

    num_matches_to_text = 0
    ranks = []

    for index, q in enumerate(query_neighbors):
        print(data[index]['query'])
        search_matches = []
        for m in data[index]['matches']:
            search_matches.append(m['question_id'])
        print(q)
        for k in q:
            print(all_matches[k][1]['question_title'])
            print(all_matches[k][1]['question_id'])
            if all_matches[k][1]['question_id'] in search_matches:
                num_matches_to_text += 1
                ranks.append(search_matches.index(all_matches[k][1]['question_id']))
    print('num of matches to text:' + str(num_matches_to_text))
    print('num queries:' + str(num_queries))
    print('average overlap with search' + str(num_matches_to_text/(num_queries * top_k)))
    print('mean search rank')
    print(np.mean(np.asarray(ranks)))
        


