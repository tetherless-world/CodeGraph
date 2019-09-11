import requests
import json
import StaticAnalysisGraphBuilder
import AI4MLTagReader
import rdflib
import numpy as np
import os
import sys

"""
  This class opens up a pandas dataframe of code from github, and calls WALA's apis to get control flow and data flow
  for each class.  The class is broken down into a set of entry points that correspond to each function in the dataset.
  Analysis starts at each entry point, and returns a graph of control flow and data flow edges.  These are 
  converted into an RDF graph and dumped in turtle.
"""
classes_to_superclasses = AI4MLTagReader.get_class_hierarchy()

url = 'http://localhost:4567/analyze_code'


def main(inputdir, graphdir):
    fails = 0
    success = 0

    for f in os.listdir(inputdir):
        if not f.startswith('sample'):
            continue
        with open(os.path.join(inputdir, f)) as sample_file:
            source = sample_file.read()

        graph_tuple = handle_call_to_analysis(source, f)
        if graph_tuple:
            single_g = rdflib.Graph()
            StaticAnalysisGraphBuilder.addToGraph(single_g, graph_tuple)
            fn = os.path.join(graphdir, f + '.ttl')
            print(fn)
            with open(fn, 'wb') as out:
                out.write(single_g.serialize(format='turtle'))
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
    # print(json.dumps(json_data, indent=4))
    # print('************************')
    nodes, data_flow_edges, control_flow_edges = StaticAnalysisGraphBuilder.parse_wala_into_graph(json_data,
                                                                                               add_args=True)
    # print(nodes)
    print("analyzed " + file)
    return (nodes, data_flow_edges, control_flow_edges, file)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])

