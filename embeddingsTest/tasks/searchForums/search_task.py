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


def get_model(embed_type):
    if embed_type == 'USE':
        model_path = 'https://tfhub.dev/google/universal-sentence-encoder/4'
        embed = hub.load(model_path)
    elif embed_type == 'bert':
        model_path = 'bert-base-nli-stsb-mean-tokens'
        embed = SentenceTransformer(model_path)
    elif embed_type == 'roberta':
        model_path = 'roberta-base-nli-stsb-mean-tokens'
        embed = SentenceTransformer(model_path)
    elif embed_type == 'distilbert':
        model_path = 'distilbert-base-nli-stsb-wkpooling'
        embed = SentenceTransformer(model_path)
    return embed


def embed_questions(queries, embed_type):
    embed = get_model(embed_type)
    if embed_type == 'USE':
        sentence_embeddings = embed(queries)
    else:
        sentence_embeddings = embed.encode(queries)
    return sentence_embeddings


def run_analysis(top_k, search_file_name, embedding_dict, add_all, embed_type):
    top_k = top_k

    with open(search_file_name, 'r') as f:
        data = json.load(f)
        queries = []
        embeddings = []
        index = faiss.IndexFlatIP(512)
        all_matches = []
        all_docs = []
        qids = {}
        for obj in data:
            query = obj['query']
            queries.append(query)
            for match in obj['matches']:
                qid = match['question_id']
                # posts get repeated across queries sometimes - to avoid neural
                # embeddings reproducing the same result multiple times - ignore dups
                # across queries.
                if qid in qids:
                    continue
                qids[match['question_id']] = 1
                if add_all:
                    all_docs.append(embedding_dict[qid]['all_content'])
                else:
                    all_docs.append(embedding_dict[qid]['title'])
                all_matches.append((query, match))
        embeddings = np.asarray(all_docs)

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run a set of queries on embedded queries')
    parser.add_argument('--top_k', type=int,
                        help='file containing all queries to be run')
    parser.add_argument('--search_file_results', type=str,
                        help='file containing the corpus of search results we need to run on')
    parser.add_argument('--embedding_dict_file', type=str,
                        help='a pickle file containing a dictionary of USE/BERT/ROBERTA embeddings')
    parser.add_argument('--add_all_content', type=str,
                        help='True/False')
    parser.add_argument('--embed_type', type=str,
                        help='USE or bert or roberta')

    args = parser.parse_args()
    infile = open(args.embedding_dict, 'rb')
    embedding_dict = pickle.load(infile)
    infile.close()

    if args.add_all_content == "True":
        add_all = True
    else:
        add_all = False

    assert args.embed_type == 'USE' or args.embed_type == 'bert' or args.embed_type == "roberta"

    args = parser.parse_args(args.top_k, args.search_file_results, embedding_dict, add_all, args.embed_type)

