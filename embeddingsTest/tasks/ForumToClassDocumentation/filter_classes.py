import json
import sys
import re

with open(sys.argv[1]) as f:
    data = json.load(f)
    ids = {}
    ids2hits = {}
    ids2negs = {}
    for hit in data:
        match = False
        ids2hits[hit['id']] = hit
        for clazz in hit['relevant_class_alias'][0]:
            pkg = ' ' + clazz.split('.')[0]
            clazz_part = clazz.split('.')[-1]

            for answer in hit['answers']:
            
                if clazz_part in hit['title'] and (pkg in answer['answer_text'] or pkg in hit['text'] or pkg in hit['title']) and (clazz_part in answer['answer_text'] or clazz_part in hit['text']):
                    
                    match = True
                    if clazz not in ids:
                        ids[clazz] = set()
                    ids[clazz].add(hit['id'])
                    break
                
            if not match and pkg not in hit['title'] and clazz_part not in hit['title']:
                if clazz not in ids2negs:
                    ids2negs[clazz] = set()
                ids2negs[clazz].add(hit['id'])

    results = {}
    for key in ids:
        l = ids[key]
        results[key] = [ids2hits[x] for x in l]
        
    with open('/tmp/class_matches_so.json', 'w') as f:
        json.dump(results, f, indent=4)
    print(len(ids))
    results = {}
    for key in ids2negs:
        l = ids2negs[key]
        results[key] = [ids2hits[x] for x in l]
    with open('/tmp/class_poor_matches_so.json', 'w') as f:
        json.dump(results, f, indent=4)
    print(len(results))
    print(len(set(ids2negs.keys()).intersection(set(ids.keys()))))
