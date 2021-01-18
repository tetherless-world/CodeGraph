import sys
from utils import util
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import scipy
import json

embedType = 'USE'

def build_class_mapping(mapPath):
    classMap = {}
    with open(mapPath, 'r') as inputFile:
        for line in inputFile:
            lineComponents = line.rstrip().split(' ')
            if len(lineComponents) < 2:
                classMap[lineComponents[0]] = lineComponents[0]
            else:
                classMap[lineComponents[0]] = lineComponents[1]
    return classMap


def check(data, v):
    print(v)
    assert v in data
            
if __name__ == '__main__':
    if len(sys.argv) > 4:
        embedType = sys.argv[4]
        if len(sys.argv) > 5:
            model_dir = sys.argv[5]
        
    util.get_model(embedType, model_dir)

    docPath = sys.argv[1]
    classPath = sys.argv[2]
    usagePath = sys.argv[3]
    with open(usagePath) as f:
        util.evaluate_regression(f, docPath, embedType)
        

        
