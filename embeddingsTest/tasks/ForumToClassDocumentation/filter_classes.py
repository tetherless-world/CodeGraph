import json
import sys
import re
import argparse

def read_into_results(ids, id2hits, docs, class2class, label):
    results = {}
    hits = 0
    for key in ids:
        l = ids[key]
        if key in class2class and class2class[key] in docs:
            val = docs[class2class[key]]
        else: 
            continue
        hits += len(l)
        results[key] = {'docstring': val,'posts': [id2hits[x] for x in l],'label': label}
    print('hits:' + str(hits))
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--positive_file", help="Files to get positives")
    parser.add_argument("--negative_file", help="Files to get negatives")
    parser.add_argument("--all_docstrings_file", help="Files to get docstrings")
    parser.add_argument("--class_map", help="class map")

    # parser.set_defaults(positive_file = '../test_data/class_matches_in_stackoverflow.json',
    #                     negative_file = '../test_data/class_matches_in_stackoverflow.json',
    #                     all_docstrings_file = '../../../code_knowledge_graph/data/full_docstrings-merge-15-22/all/merge-15-22.2.format.json',
    #                     class_map = '../test_data/classes.map')
    args = parser.parse_args()

    with open(args.positive_file) as f:
        data = json.load(f)
    ids = {}
    ids2hits = {}

    for hit in data:
        for clazz in hit['relevant_class_alias'][0]:
            pkg = ' ' + clazz.split('.')[0]
            clazz_part = clazz.split('.')[-1]
            match = False
            
            for answer in hit['answers']:            
                if clazz_part in hit['title'] and (pkg in answer['answer_text'] or pkg in hit['text'] or pkg in hit['title']) and (clazz_part in answer['answer_text'] or clazz_part in hit['text']):
                    match = True
            if match:
                if clazz not in ids:
                    ids[clazz] = set()
                if hit['id'] not in ids2hits:
                    ids[clazz].add(hit['id'])
                    ids2hits[hit['id']] = hit
                    break
    
            
    class2class = {}
    with open(args.class_map) as f:
        for line in f.readlines():
            arr = line.split()
            if len(arr) == 1:
                class2class[arr[0]] = arr[0]
            else:
                class2class[arr[1]] = arr[1]
                class2class[arr[0]] = arr[1]

    class2docstring = {}
    with open(args.all_docstrings_file) as f:
        data = json.load(f)
        for obj in data:
            if 'klass' not in obj or 'class_docstring' not in obj:
                continue
            klass = obj['klass']
            if klass not in class2class:
                continue
            class2docstring[class2class[klass]] = obj['class_docstring']
    
    
    results = read_into_results(ids, ids2hits, class2docstring, class2class, 1)
    
    with open('/tmp/class_matches_stackoverflow_positives.json', 'w') as f:
        json.dump(results, f, indent=4)

    print('number of positive classes:' + str(len(results.keys())))
    
    with open(args.negative_file) as f:
        data = json.load(f)
    
    ids = {}
    neg_ids2hits = {}
    num_samples = 0
    for hit in data:
        for clazz in hit['relevant_class_alias'][0]:
            if clazz in ids and len(ids[clazz])>10:
                print(f'clazz has more than 10 hits, continue')
                continue
            pkg = ' ' + clazz.split('.')[0]
            clazz_part = clazz.split('.')[-1]
            # if pkg not in hit['title'] and clazz_part not in hit['title'] and hit['id'] not in ids2hits and hit['id'] not in neg_ids2hits:
            if clazz_part not in hit['title'] and clazz_part not in hit['text'] and hit['id'] not in ids2hits and hit['id'] not in neg_ids2hits:
                neg_ids2hits[hit['id']] = hit
                if clazz not in ids:
                    ids[clazz] = set()
                ids[clazz].add(hit['id'])
                num_samples += 1
            if num_samples > 10000:
                print(f'Found {num_samples} samples, break')
                break
        if num_samples > 10000:
            print(f'Found {num_samples} samples, break')
            break

    results = read_into_results(ids, neg_ids2hits, class2docstring, class2class, 0)
    assert len(set(ids2hits.keys()).intersection(set(neg_ids2hits.keys()))) == 0
    with open('/tmp/class_matches_stackoverflow_negatives_v3.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    print('number of negative classes:' + str(len(results.keys())))

''''
v1:             # if pkg not in hit['title'] and clazz_part not in hit['title'] and hit['id'] not in ids2hits and hit['id'] not in neg_ids2hits:
hits:3685
number of positive classes:1753
hits:18010
number of negative classes:4881




v2:             if pkg not in hit['title'] and pkg not in hit['text'] and clazz_part not in hit['title'] and clazz_part not in hit['text'] and hit['id'] not in ids2hits and hit['id'] not in neg_ids2hits:

python filter_classes.py --positive_file /data/blanca/class_matches_in_stackoverflow_v5.json --negative_file /data/blanca/class_matches_in_stackoverflow_v4_1m_shuffled.json --all_docstrings_file /home/ibrahim/full_docstrings-merge-15-22/all/merge-15-22.2.format.json --class_map ../test_data/classes.map
number of positive classes:2261
Found 10001 samples, break
Found 10001 samples, break
hits:5003
number of negative classes:4685

'''
