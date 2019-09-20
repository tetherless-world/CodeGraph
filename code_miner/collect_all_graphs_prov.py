import requests
import json
import StaticAnalysisGraphBuilder
import AI4MLTagReader
import rdflib
import numpy as np
import os
import sys
import setlr
import rdflib

"""
  This class opens up a pandas dataframe of code from github, and calls WALA's apis to get control flow and data flow
  for each class.  The class is broken down into a set of entry points that correspond to each function in the dataset.
  Analysis starts at each entry point, and returns a graph of control flow and data flow edges.  These are 
  converted into an RDF graph and dumped in turtle.
"""
classes_to_superclasses = AI4MLTagReader.get_class_hierarchy()

url = 'http://localhost:4567/analyze_code'

SETL_FILE = 'codegraph.setl.ttl'

prov = rdflib.Namespace('http://www.w3.org/ns/prov#')

def main(inputdir, graphdir, jsondir, limit=-1):
    fails = 0
    success = 0

    limit = int(limit)
    for i, f in enumerate(os.listdir(inputdir)):
        if limit >= 0 and i > limit:
            break
        if not f.startswith('sample'):
            continue
        with open(os.path.join(inputdir, f)) as sample_file:
            source = sample_file.read()

        json_data = handle_call_to_analysis(source, f)
        if json_data:
            json_file = os.path.join(jsondir, f + '.json')
            with open(json_file,'w') as out:
                out.write(json.dumps(json_data,indent=4))
                
            fn = os.path.join(graphdir, f + '.trig')
            
            g = convert_to_rdf(json_file)
            with open(fn, 'wb') as out:
                out.write(g.serialize(format='trig'))
            success += 1
        else:
            fails += 1

    print("num successes:" + str(success))
    print('num failures' + str(fails))
    
    

def print_infrequent_edges(edge_map, limit):
    for key in edge_map:
        if len(edge_map.get(key)) < limit:
            print(key)


def compute_average_degree(edge_map):
    degree = []
    for key in edge_map:
        degree.append(len(edge_map.get(key)))
    np_array = np.asarray(degree)
    print(np.histogram(degree, density=False))
    return np_array.mean(), np_array.std()


def handle_call_to_analysis(source, file):
    if not source:
        return
    source = source.encode('utf-8')
    res = requests.post(url=url,
                        data=source,
                        headers={'Content-Type': 'application/octet-stream'})
    if not res.text:
        print("failed to analyze " + file + " with null result")
        return

    if res.text == '<html><body><h2>500 Internal Server Error</h2></body></html>':
        print("failed to analyze " + file + " with server error")
        return
    if res.text == '[]':
        print("failed to analyze " + file + " with empty list")
        return
    json_data = json.loads(res.text)
    if len(json_data) == 0:
        print("failed to analyze " + file + " with no turtles")
        return
    return json_data

def convert_to_rdf(json_file):
    setl_graph = rdflib.Graph()
    setl_graph.parse(SETL_FILE,format="turtle")
    cwd = os.getcwd()
    
    extract = setl_graph.value(rdflib.URIRef('http://purl.org/twc/codegraph/setl/codegraph_json'), prov.wasGeneratedBy)
    setl_graph.add((extract, prov.used,rdflib.URIRef('file://'+os.path.join(cwd,json_file))))
    
    results = setlr._setl(setl_graph)
    
    single_g = results[rdflib.URIRef('http://purl.org/twc/codegraph/setl/codegraph')]
    print("analyzed " + json_file)
    return single_g

if __name__ == '__main__':
    main(*sys.argv[1:])

