#!/usr/bin/env python
# coding: utf-8

# In[1]:

import json
import faiss
import numpy as np
import argparse
import pickle
from sentence_transformers import SentenceTransformer
import tensorflow as tf
import tensorflow_hub as hub
import os
import sys
from utils.util import get_model
from scipy import stats as stat
from sklearn.metrics import ndcg_score

def embed_sentences(sentences, embed_type, model_dir=None):
    model = get_model(embed_type, model_dir)
    if embed_type == 'USE' and type(sentences) == str:
        sentences = [sentences]
    sentence_embeddings = model.encode(sentences)
    return sentence_embeddings


def run_analysis(top_k, search_file_name, add_all, embed_type, model_dir):
    top_k = top_k

    with open(search_file_name, 'r') as f:
        data = json.load(f)
        queries = []
        embeddings = []
        all_matches = []
        qids = {}
        query_2_matches = {}
        data_as_array = []
        for obj in data:
            query = obj['query']
            if query not in query_2_matches:
                query_2_matches[query] = []
            query_2_matches[query].append(obj)

        for query, obj in query_2_matches.items():

            queries.append(query)
            all_docs = []
            for match in obj:
                qid = match['q_id']
                # posts get repeated across queries sometimes - to avoid neural
                # embeddings reproducing the same result multiple times - ignore dups
                # across queries.
                if qid in qids:
                    continue
                qids[match['q_id']] = 1
                if add_all:
                    content = embed_sentences(match['q_title'] + ' ' + match['q_text'] + ' ' + match['a_text'], embed_type, model_dir)
                else:
                    content = embed_sentences(match['q_title'], embed_type, model_dir)
                # not performing this step seems to cause catastrophic
                # issues in the np.asarray(embeddings) step further down
                # suspect something is suboptimal about converting whatever
                # tensorflow returns into np arrays
                if embed_type == 'USE':
                    content = np.asarray(content)
                embeddings.append(content)
                all_matches.append((query, match))
            data_as_array.append((query, obj))
        print('ALL LOADED')
        queries_embedding = np.asarray(embed_sentences(queries, embed_type, model_dir))
        print('queries embedded')
        print(len(embeddings))
        embeddings = np.asarray(embeddings).squeeze(1)
        print('numpy array created for main embeddings')

        num_queries = len(queries_embedding)
        faiss.normalize_L2(embeddings)
        faiss.normalize_L2(queries_embedding)
        index = faiss.IndexFlatIP(len(embeddings[0]))
        print(embeddings.shape)
        index.add(embeddings)
        query_distances, query_neighbors = index.search(queries_embedding, top_k)

        num_matches_to_text = 0
        all_ndcg = []
        ranks_avgs = []
        recipRanks = []
        for index, q in enumerate(query_neighbors):
            ranks = []
            print(data_as_array[index][0])
            search_matches = []
            print('Actual matches (top-100):')
            for idx, m in enumerate(data_as_array[index][1] ):
                try:
                    if idx < 100:
                        print(f"{idx} -- {m['q_id']}:{m['q_title']}")
                except:
                    pass
                search_matches.append(m['q_id'])
            print(q)
            print('Returned matches:')
            y_true = []
            y_pred = []
            for idx, k in enumerate(q):
                # print(all_matches[k][1]['q_title'])
                # print(all_matches[k][1]['q_id'])
                if all_matches[k][1]['q_id'] in search_matches:
                    try:
                        print(f"{idx} -- {all_matches[k][1]['q_id']}:{all_matches[k][1]['q_title']}")
                    except:
                        pass
                    num_matches_to_text += 1
                    rank = search_matches.index(all_matches[k][1]['q_id'])
                    '''
                       # Corpus# 
                       True relevance score - scale from 0-10
                           true_relevance = {'d1': 10, 'd2': 9, 'd3':7, 'd4':6, 'd5':4}# Predicted relevence score
                           predicted_relevance = {'d1': 8, 'd2': 9, 'd3':6, 'd4':6, 'd5':5}# relevance list processed as array
                           true_rel = np.asarray([list(true_relevance.values())])
                           predicted_rel = np.asarray([list(predicted_relevance.values())])
                           >> ndcg_score(true_rel, predicted_rel)
                           >> 0.9826797611735933
                    '''
                    y_true.append(rank+1) #true scores of entities to be ranked.
                    y_pred.append(idx+1)
                    reciprocal = 1 / (rank+1)
                    recipRanks.append(reciprocal)
                    ranks.append(rank+1)
            q_mrr = np.mean(np.asarray(ranks))
            ranks_avgs.append(q_mrr)
            if len(y_true) > 0 and len(y_pred) > 0:
                print('y_true: ', y_true, ', y_pred: ', y_pred)
                q_ndcg = ndcg_score(np.asarray([y_true]), np.asarray([y_pred]))
                print(f'Question MRR: {q_mrr}, NDCG: {q_ndcg}')
                all_ndcg.append(q_ndcg)

        print('num of matches to text:' + str(num_matches_to_text))
        print('num queries:' + str(num_queries))
        print('average overlap with search:' + str(num_matches_to_text / (num_queries * top_k)))
        print('mean search rank:', np.mean(np.asarray(ranks_avgs)))
        meanRecipRank = sum(recipRanks) / len(recipRanks)
        print('MRR: standard error of the mean ', stat.sem(recipRanks))
        print("Mean reciprocal rank is:", meanRecipRank)
        print("Average NDCG:", (sum(all_ndcg) / len(all_ndcg)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run a set of queries on embedded queries')
    parser.add_argument('--top_k', type=int,
                        help='file containing all queries to be run')
    parser.add_argument('--search_file_results', type=str,
                        help='file containing the corpus of search results we need to run on')
    parser.add_argument('--embedding_dict_dir', type=str,
                        help='a pickle file containing a dictionary of USE/BERT/ROBERTA embeddings')
    parser.add_argument('--add_all_content', type=str, default='True',
                        help='True/False')
    parser.add_argument('--embed_type', type=str, default='bert',
                        help='USE or bert or roberta')
    parser.add_argument('--model_dir', type=str, default=None,
                        help='Model directory')

    args = parser.parse_args()

    """
    if (args.embed_type == 'USE'):
        for f in os.listdir(args.embedding_dict_dir):
            with open(args.embedding_dict_dir + '/' + f, "rb") as x:
                q_data = pickle.load(x)
                embedding_dict[f] = q_data
            x.close()        
"""


    if args.add_all_content == "True":
        add_all = True
    else:
        add_all = False

    run_analysis(args.top_k, args.search_file_results, add_all, args.embed_type, args.model_dir)

