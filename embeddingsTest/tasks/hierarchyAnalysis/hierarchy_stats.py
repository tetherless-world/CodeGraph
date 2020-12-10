from utils import util
import argparse
import networkx as nx
from metrics_eval import ranking_metrics
import numpy as np
import scipy
import sys

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hierarchy prediction based on embeddings')
    parser.add_argument('--eval_file', type=str,
                        help='train/test file')
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
    print('evaluation')

    with open(args.eval_file) as f:
        util.evaluate_regression(f, args.docstrings_file, args.embed_type)
