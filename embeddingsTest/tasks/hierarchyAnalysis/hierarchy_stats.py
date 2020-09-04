from utils import util
import argparse
import networkx as nx
from metrics_eval import ranking_metrics


def build_graph(class2superclasses):
    classgraph = nx.Graph()
    with open(class2superclasses) as f:
        for line in f:
            arr = line.split(',')
            clazz = arr[0]
            superclazz = arr[1][len('http://purl.org/twc/graph4code/python/')]
            if clazz not in classgraph.nodes():
                classgraph.add_node(clazz)
            if superclazz not in classgraph.nodes():
                classgraph.add_node(superclazz)
            if superclazz != 'object':
                classgraph.add_edge(clazz, superclazz)
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
    predicted = []
    counter = 0
    class2ids = {}
    for key in docstring_to_neighbors:
        if key not in docsToClasses:
            print('key not found:' + key)
            continue

        # this key might correspond to a number of classes, pick one that has at least one related class
        # classGraph
        clazzes = docsToClasses[key]
        clazz = None
        for c in clazzes:
            if c not in real_classes and c not in classGraph.nodes():
                print('skipping:' + c)
                continue
            if len(list(nx.neighbors(classGraph, c))) > 0:
                clazz = c
                break
        if clazz is None:
            continue

        if clazz not in class2ids:
                class2ids[clazz] = counter
                counter += 1

        related_classes = nx.bfs_predecessors(classGraph, clazz)
        for c in related_classes:
            if c not in class2ids:
                class2ids[c] = counter
                counter += 1
            expected.append(class2ids[c])

        neighbors = docstring_to_neighbors[key]
        for n in neighbors:
            if n not in docsToClasses:
                print('neighbor not found' + n)
                continue
            neighbor_classes = docsToClasses[n]

            # we want to have only 10 neighbors so we need to ensure we add the correct class
            # if its in related classes, and if its not then we add just one representative
            # for some class found thats not in the related set
            for c in neighbor_classes:
                if c in related_classes:
                    predicted.append(class2ids(c))
                else:
                    if c not in class2ids:
                        class2ids[c] = counter
                        counter += 1
                    predicted.append(class2ids[c])
                    break
    print('mrr: ' + str(ranking_metrics.mrr(expected, predicted)))
    print('map@10: ' + str(ranking_metrics.map(expected, predicted, 10)))


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
    query_distances, query_neighbors = index.search(embeddedDocText, args.top_k)
    docstringsToDocstringNeighbors = util.compute_neighbor_docstrings(query_neighbors, docList)
    classGraph = build_graph(args.class2superclass_file)
    evaluate_neighbors(docstringsToDocstringNeighbors, docsToClasses, classGraph, real_classes)








