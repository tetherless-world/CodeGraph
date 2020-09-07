from utils import util
import argparse
import networkx as nx
from metrics_eval import ranking_metrics
import numpy as np

def build_graph(class2superclasses):
    classgraph = nx.Graph()
    with open(class2superclasses) as f:
        for line in f:
            arr = line.split(',')
            clazz = arr[0]
            if len(arr[1]) < len('http://purl.org/twc/graph4code/python/'):
                continue
            superclazz = arr[1][len('http://purl.org/twc/graph4code/python/'):]
            if clazz not in classgraph.nodes():
                classgraph.add_node(clazz)
            if superclazz not in classgraph.nodes():
                classgraph.add_node(superclazz)
            superclazz = superclazz.strip()
            if superclazz != 'object':
                classgraph.add_edge(clazz, superclazz)
                #print('added edge:' + clazz + ' ->' + superclazz)
    f.close()
    return classgraph

def read_valid_classes(classmap, classfail):
    realclasses = set()
    with open(classmap) as f:
        for line in f:
            arr = line.split()
            if len(arr) > 1:
                realclasses.add(arr[1])
    with open(classfail) as f:
        for line in f:
            line = line.strip()
            realclasses.add(line)
    return realclasses

def evaluate_neighbors(docstring_to_neighbors, docsToClasses, classGraph, real_classes):
    expected = []
    expected_mrr_map = []
    predicted = []
    counter = 0
    class2ids = {}
    num_queries = 0
    no_related_classes_found = 0
    #z = 0
    for key in docstring_to_neighbors:
        #z += 1
        #if z > 20:
        #    break
        p = []
        if key not in docsToClasses:
            print('key not found:' + key)
            continue

        # this key might correspond to a number of classes, pick one that has at least one related class
        # classGraph
        clazzes = docsToClasses[key]
        clazz = None
        for c in clazzes:
            if c not in real_classes or c not in classGraph.nodes():
                # print('skipping:' + c)
                continue
            if len(list(nx.neighbors(classGraph, c))) > 0:
                clazz = c
                break
        if clazz is None:
            continue

        if clazz not in class2ids:
            class2ids[clazz] = counter
            counter += 1

        neighbors = docstring_to_neighbors[key]
        module = clazz.split('.')[0]
        
        for n in neighbors:
            if n == key:
                continue
            if n not in docsToClasses:
                print('neighbor not found' + n)
                continue
            neighbor_classes = docsToClasses[n]

            # we want to have only 10 neighbors so we need to ensure we add the correct class
            # if it has the same module as the clazz, we have some hope of using it
            # find the one with the least path distance
            max_dist = 10
            pathdist = 0
            min_class = None
            # print(clazz)
            for c in neighbor_classes:
                neighbor_module = c.split('.')[0]
                # print('neighbor:' + c)
                if neighbor_module != module:
                    continue
                try:
                    pathdist = min(len(nx.shortest_path(classGraph, clazz, c)), max_dist)
                    min_class = c
                    #print(c + '->' + clazz + ' has distance:' + str(pathdist))
                except:
                    pass

            if c not in class2ids:
                class2ids[c] = counter
                counter += 1
            # if the path distance > 0 then make the nearer ones more relevant by subtracting some max
            if pathdist > 0:
                pathdist = 10.0 - float(pathdist)
            else:
                pathdist = 0.0
            p.append([class2ids[c], pathdist])
        test_dist = [x for x in p if x[1] > 0]
        num_queries += 1
        if len(test_dist) == 0:
            no_related_classes_found += 1
            continue
        e = p.copy()
        e.sort(key=lambda x:x[1], reverse=True)
        e_mrr_map = []
        for x in e:
            if x[1] > 0:
                e_mrr_map.append(x[0])
        p = [x[0] for x in p]
        expected.append(e)
        predicted.append(p)
        expected_mrr_map.append(e_mrr_map)
        
    expected = np.array(expected)
    predicted = np.array(predicted)
    expected_mrr_map = np.array(expected_mrr_map)
    """
    print('expected_ncdg')
    print(expected)
    print('expected')
    print(expected_mrr_map)
    print('predicted')
    print(predicted) """
    print('ndcg:' + str(ranking_metrics.ndcg(expected, predicted, 10)))
    #print('mrr: ' + str(ranking_metrics.mrr(expected_mrr_map, predicted)))
    #print('map@10: ' + str(ranking_metrics.map(expected_mrr_map, predicted, 10)))
    print('total queries with some related class: ' + str(((num_queries - no_related_classes_found)/num_queries)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hierarchy prediction based on embeddings')
    parser.add_argument('--top_k', type=int,
                        help='file containing all queries to be run')
    parser.add_argument('--class2superclass_file', type=str,
                        help='csv of classes to superclasses')
    parser.add_argument('--docstrings_file', type=str,
                        help='docstrings for classes, functions etc')
    parser.add_argument('--classmap', type=str,
                        help='classes to real class names as determined by dynamic loading of class')
    parser.add_argument('--classfail', type=str,
                        help='classes that fail to load to determine real class mappings')
    parser.add_argument('--embed_type', type=str,
                        help='USE or bert or roberta')

    args = parser.parse_args()

    real_classes = read_valid_classes(args.classmap, args.classfail)
    index, docList, docsToClasses, embeddedDocText, classesToDocs = util.build_index_docs(args.docstrings_file)
    query_distances, query_neighbors = index.search(embeddedDocText, args.top_k + 1)
    docstringsToDocstringNeighbors = util.compute_neighbor_docstrings(query_neighbors, docList)
    classGraph = build_graph(args.class2superclass_file)
    evaluate_neighbors(docstringsToDocstringNeighbors, docsToClasses, classGraph, real_classes)

