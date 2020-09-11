from utils import util
import argparse
import networkx as nx
from metrics_eval import ranking_metrics
import numpy as np
import scipy

def build_graph(class2superclasses, valid_classes):
    classgraph = nx.Graph()
    with open(class2superclasses) as f:
        for line in f:
            arr = line.split(',')
            clazz = arr[0].strip()
            if len(arr[1]) < len('http://purl.org/twc/graph4code/python/'):
                continue
            superclazz = arr[1][len('http://purl.org/twc/graph4code/python/'):].strip()
            if clazz not in valid_classes or superclazz not in valid_classes:
                continue
            if clazz not in classgraph.nodes():
                classgraph.add_node(clazz)
            if superclazz not in classgraph.nodes():
                classgraph.add_node(superclazz)
            if superclazz != 'object':
                classgraph.add_edge(clazz, superclazz)
                #print('added edge:' + clazz + ' ->' + superclazz)
    print('graph has nodes:' + str(len(list(classgraph.nodes()))))
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
    mrr = []
    ndcg = []
    all_mrr = []
    all_ndcg = []
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
                    print(c + '->' + clazz + ' has distance:' + str(pathdist))
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
            all_ndcg.append(0.0)
            all_mrr.append(0.0)
            continue
        
        e = p.copy()
        e.sort(key=lambda x:x[1], reverse=True)
        p = [x[0] for x in p]
        e_mrr_map = [x[0] for x in e if x[1] > 0]
        ndcg_ind = ranking_metrics.ndcg(np.array([e]), np.array([p]), 10)
        print(e)
        print(p)
        print('ndcg:' + str(ndcg_ind))
        ndcg.append(ndcg_ind)
        print(e_mrr_map)
        print(p)
        mrr_ind = ranking_metrics.mrr(np.array([e_mrr_map]), np.array([p]))
        print('mrr: ' + str(mrr_ind))
        mrr.append(mrr_ind)
        all_ndcg.append(ndcg_ind)
        all_mrr.append(mrr_ind)

    ndcg_avg = np.array(ndcg).mean()
    mrr_avg = np.array(mrr).mean()
    all_ndcg_avg = np.array(all_ndcg).mean()
    all_mrr_avg = np.array(all_mrr).mean()
    
    """
    print('expected_ncdg')
    print(expected)
    print('expected')
    print(expected_mrr_map)
    print('predicted')
    print(predicted) """ 
    print('ndcg:' + str(ndcg_avg))
    print('mrr: ' + str(mrr_avg))
    print('se_ndcg:' + str(scipy.stats.sem(ndcg)))
    print('se_mrr:' + str(scipy.stats.sem(mrr)))
    print('all_ndcg:' + str(all_ndcg_avg))
    print('all_mrr: ' + str(all_mrr_avg))
    print('se__all_ndcg:' + str(scipy.stats.sem(all_ndcg)))
    print('se_all_mrr:' + str(scipy.stats.sem(all_mrr)))

    print('total queries with some related class: ' + str(((num_queries - no_related_classes_found)/num_queries)))
    print('total num queries:' + str(num_queries))

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
    index, docList, docsToClasses, embeddedDocText, classesToDocs = util.build_index_docs(args.docstrings_file, args.embed_type, real_classes)
    query_distances, query_neighbors = index.search(embeddedDocText, args.top_k + 1)
    docstringsToDocstringNeighbors = util.compute_neighbor_docstrings(query_neighbors, docList)
    classGraph = build_graph(args.class2superclass_file, real_classes)
    evaluate_neighbors(docstringsToDocstringNeighbors, docsToClasses, classGraph, real_classes)
