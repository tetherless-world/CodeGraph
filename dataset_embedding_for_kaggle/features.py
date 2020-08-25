from autosklearn.metalearning.metafeatures import metafeatures, metafeature
import autosklearn.pipeline.util as putil
import sklearn.datasets as ds
import numpy as np
import random
import warnings
import math

def loadFile(path):
    """
    path: path to the csv file

    Returns: 2-D numpy array representing the csv file
    """

    #skip_header skips the first line in the csv file
    data = np.genfromtxt(path, dtype = np.uint8, delimiter=",", skip_header=1)
    return data

def takeSample(dataset, x_sampleNum):
    """
    dataset: 2-D numpy array (usually obtained from loadFile)
    x_sampleNum: int, number of rows to sample
    y_sampleNum: int, number of columns to sample

    Returns: 2-D numpy array sampled from the dataset parameter
    """
    #scale the number of sampled rows according to the number of columns
    #if the number of columns is too large
    if dataset.shape[1] > 800:
        x_sampleNum = math.floor((-1/50) * dataset.shape[1] + 1400)
        #ensure at least 1000 rows
        if x_sampleNum < 1000:
            x_sampleNum = 1000

    reservoir = dataset[:x_sampleNum]

    #sample rows
    for i in range(dataset.shape[0]):
        j = random.randrange(i+1)
        
        if j < x_sampleNum:
            reservoir[j] = dataset[i]

    return reservoir

def getFeatures(x, y, name):
    """
    x: numpy array, X of the dataset
    y: 1-D numpy array, Y of the dataset
    name: string, name of the dataset

    Returns: DatasetMetafeatures object
    (found in autosklearn/metalearning/metafeatures/metafeature.py)
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        metafeatures_labels = metafeatures.calculate_all_metafeatures(x, y, [False] * x.shape[1], name)
    return metafeatures_labels

def serialize(filename, features_vector):
    """
    filename: filename of .npy file to write to
    features_vector: features vector gotten from getFeatures()

    Each feature is an MetaFeatureValue object saved as follows in numpy array:
        PCASkewnessFirstPC (type: METAFEATURE, fold: 0, repeat: 0, value: 4.220950289131384, time: 0.002, comment: )
    
    example_array[0].value will give the corresponding value of the feature
    example_array[0].name will give the name of the corresponding feature
    """

    keys = features_vector.keys()
    values = []
    for x in keys:
       if features_vector[x].type_ == "METAFEATURE":
           values.append(features_vector[x])
    
    values = np.array(values)
    np.save(filename, values)
