#!/usr/bin/env python
# coding: utf-8

# In[1]:

import json
import pickle
import argparse

def run_analysis(output_dir, search_file_name, search_file_content_name):
    subset_data = []
    question_ids = []
    with open(search_file_name, 'r') as f:
        data = json.load(f)
        for obj in data:
            query = obj['query']
            matches = {'query': query, 'matches':obj['matches'][0:20]}
            subset_data.append(matches)
            for m in matches['matches']:
                question_ids.append(m['question_id'])
            
    with open(output_dir + '/' + 'stackoverflow_matches_codesearchnet_5k.json', 'w') as f:
        json.dump(subset_data, f)

    with open(output_dir + '/' + 'search_question_ids', 'w') as f:
        for id in question_ids:
            f.write(str(id) + '\n')

    with open(search_file_content_name, 'r') as f:
        data = json.load(f)
        content_subset = []
        for obj in data:
            query = obj['query']
            matches = []
            for m in obj['matches']:
                if m['question_id'] in question_ids:
                    matches.append(m)
            content_subset.append({'query':query, 'matches':matches})
        with open(output_dir + '/' + 'stackoverflow_matches_codesearchnet_5k_content.json', 'w') as o:
            json.dump(content_subset, o)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'create a set of files for testing')
    parser.add_argument('--output_dir', type=str,
                        help='output_dir')
    parser.add_argument('--search_file_results', type=str,
                    help='file containing the corpus of search results we need to run on')

    parser.add_argument('--search_file_content', type=str,
                    help='file containing the corpus of search results we need to run on')


    args = parser.parse_args()
    run_analysis(args.output_dir, args.search_file_results, args.search_file_content)

