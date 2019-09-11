#!/usr/bin/env python
# coding: utf-8

# ### This kernel is about DecisionTreeClassifier, RandomForestClassifier
# #### In decisiontreeclassifier, we show the flow of max_depth(hyperparameter)
# #### In RandomForestClassifier, we show the brief show of classifying

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

# In[ ]:


import io
import pydot
from IPython.core.display import Image
from sklearn.tree import export_graphviz
from sklearn.datasets import load_iris
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

#Functions to show plots

def draw_decision_tree(model):
    dot_buf = io.StringIO()
    export_graphviz(model, out_file=dot_buf,feature_names=iris.feature_names[2:])
    graph = pydot.graph_from_dot_data(dot_buf.getvalue())[0]
    image = graph.create_png()
    return Image(image)


def plot_decision_regions(X, y, model, title):
    resolution = 0.01
    markers = ('s', '^', 'o')
    colors = ('red', 'blue', 'lightgreen')
    cmap = mpl.colors.ListedColormap(colors)

    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = model.predict(
        np.array([xx1.ravel(), xx2.ravel()]).T).reshape(xx1.shape)

    plt.contour(xx1, xx2, Z, cmap=mpl.colors.ListedColormap(['k']))
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8,
                    c=[cmap(idx)], marker=markers[idx], s=80, label=cl)

    plt.xlabel(iris.feature_names[2])
    plt.ylabel(iris.feature_names[3])
    plt.legend(loc='upper left')
    plt.title(title)

    return Z



# In[ ]:


#tree1 max_depth=1

iris = load_iris()
X = iris.data[:, [2, 3]]
y = iris.target

tree1 = DecisionTreeClassifier(criterion='entropy', max_depth=1, random_state=0).fit(X, y)
draw_decision_tree(tree1)


# In[ ]:


plot_decision_regions(X, y, tree1, "Depth 1")
plt.show()

# In[ ]:


#tree2 max_depth=1

tree2 = DecisionTreeClassifier(criterion='entropy', max_depth=2, random_state=0).fit(X, y)
draw_decision_tree(tree2)

# In[ ]:


plot_decision_regions(X, y, tree2, "Depth 2")
plt.show()

# In[ ]:


#tree3 max_depth=3

tree3 = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0).fit(X, y)
draw_decision_tree(tree3)

# In[ ]:


plot_decision_regions(X, y, tree3, "Depth 3")
plt.show()

# In[ ]:


#tree4 max_depth=4

tree4 = DecisionTreeClassifier(criterion='entropy', max_depth=4, random_state=0).fit(X, y)
draw_decision_tree(tree4)

# In[ ]:


plot_decision_regions(X, y, tree4, "Depth 4")
plt.show()

# In[ ]:


#tree5 max_depth=5

tree5 = DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=0).fit(X, y)
draw_decision_tree(tree5)

# In[ ]:


plot_decision_regions(X, y, tree5, "Depth 5")
plt.show()

# In[ ]:


rf = RandomForestClassifier(n_estimators=100,random_state=42)
rf.fit(X,y)
prediction = rf.predict(X)

# In[ ]:


print(prediction)
print(metrics.accuracy_score(y, prediction))
