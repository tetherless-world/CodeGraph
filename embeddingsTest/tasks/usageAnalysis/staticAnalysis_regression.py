import sys
from utils import util
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import scipy

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
        
    util.get_model(embedType)

    docPath = sys.argv[1]
    classPath = sys.argv[2]
    usagePath = sys.argv[3]
    with open(usagePath) as f:
        df = pd.read_csv(f, delimiter=' ', names=['class1', 'class2', 'shared_calls', 'pair_size'])
        print('pair-size and shared-calls')
        print(scipy.stats.pearsonr(df['shared_calls'].values, df['pair_size']))
        print(df['class1'].values)
        (index, docList, docsToClasses, embeddedDocText, classesToDocs, docToEmbedding) = util.build_index_docs(docPath, embedType, generate_dict=True)
        df = df[df['class1'].isin(classesToDocs.keys())]
        df = df[df['class2'].isin(classesToDocs.keys())]
        print(df)
        df['embedding1'] = df['class1'].apply(lambda x: docToEmbedding[classesToDocs[x]])
        df['embedding2'] = df['class2'].apply(lambda x: docToEmbedding[classesToDocs[x]])
        embed1 = df['embedding1'].values
        embed2 = df['embedding2'].values
        distance = []
        for idx in range(len(embed1)):
            distance.append(np.linalg.norm(embed1[idx] - embed2[idx]))

        model = linear_model.LinearRegression()
        new_df = df[['shared_calls', 'pair_size']]
        model.fit(new_df.iloc[:], distance)
        y_pred = model.predict(new_df.iloc[:])
        # The coefficients
        print('Coefficients: \n', model.coef_)
        print('Mean squared error: %.2f' % mean_squared_error(distance, y_pred))
        # The coefficient of determination: 1 is perfect prediction
        print('Coefficient of determination: %.2f' % r2_score(distance, y_pred))
        print('shared_calls')
        print(scipy.stats.pearsonr(df['shared_calls'].values, distance))
        print('pair_size')
        print(scipy.stats.pearsonr(df['pair_size'].values, distance))
        
