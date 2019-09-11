#!/usr/bin/env python
# coding: utf-8

# ##### In this kernel I intended to do a quick review of KNN and a simple application on Titanic problem.
# ![](https://cdn.pixabay.com/photo/2012/10/25/23/34/pygmy-sloth-62869_960_720.jpg)
# 

# K-nearest neighbor is an instance-based and nonparametric algorithm. Nonparametric because it doesn't have a determinated number of parameters before training and instance based because it memorizes the training dataset. It's also considered a lazy learning due to this learning process.
# 
# ![](https://upload.wikimedia.org/wikipedia/commons/e/e5/KNN_detec.JPG)
# 
# KNN works to find the K-nearest neighbors of a new sample, based on a distance metric (usually Euclidean distance), and make the prediction. In classification problem, the new point can be classified based on majority vote and in regression problems the prediction can be based on average of k-neighbors.
# 
# The choose of **K** parameter is important because small k can lead to overfitting.
# 
# KNN is simple and powerful.

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import warnings
warnings.filterwarnings(action="ignore")

# In[ ]:


train=pd.read_csv("../input/train.csv")
test=pd.read_csv("../input/test.csv")
test2=pd.read_csv("../input/test.csv")
titanic=pd.concat([train, test], sort=False)
len_train=train.shape[0]

print(titanic.dtypes.sort_values())
titanic.head()

# ### NA's imputation

# In[ ]:


titanic.isnull().sum()[titanic.isnull().sum()>0]

# In[ ]:


train.Age=train.Age.fillna(train.Age.mean())
test.Age=test.Age.fillna(train.Age.mean())

train.Fare=train.Fare.fillna(train.Fare.mean())
test.Fare=test.Fare.fillna(train.Fare.mean())

train.Cabin=train.Cabin.fillna("unknow")
test.Cabin=test.Cabin.fillna("unknow")

train.Embarked=train.Embarked.fillna(train.Embarked.mode()[0])
test.Embarked=test.Embarked.fillna(train.Embarked.mode()[0])

# ### Dropping features I won't use on model

# In[ ]:


train.drop(['PassengerId','Name','Ticket','Cabin'],axis=1,inplace=True)
test.drop(['PassengerId','Name','Ticket','Cabin'],axis=1,inplace=True)

# ### Turning categorical into numerical

# In[ ]:


titanic=pd.concat([train, test], sort=False)
titanic=pd.get_dummies(titanic)
train=titanic[:len_train]
test=titanic[len_train:]

# In[ ]:


train.Survived=train.Survived.astype('int')

# ### Model

# In[ ]:


xtrain=train.drop("Survived",axis=1)
ytrain=train['Survived']
xtest=test.drop("Survived", axis=1)

# In[ ]:


KNN= make_pipeline(StandardScaler(),KNeighborsClassifier())
pgKNN=[{'kneighborsclassifier__n_neighbors':[3,12,20,22]}]
gsKNN= GridSearchCV(estimator=KNN, param_grid=pgKNN, scoring='accuracy', cv=2)
scoresKNN=cross_val_score(gsKNN, xtrain.astype(float), ytrain, scoring='accuracy', cv=5)
np.mean(scoresKNN)

# ### Submission

# In[ ]:


KNN.fit(xtrain, ytrain)
pred=KNN.predict(xtest)
output=pd.DataFrame({'PassengerId':test2['PassengerId'],'Survived':pred})
output.to_csv('submission.csv', index=False)

# References:
# * Hands-On Machine Learning with Scikit-Learn and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems from Aurelien Geron
# * Python Machine Learning: Machine Learning and Deep Learning with Python, scikit-learn, and TensorFlow from Sebastian Raschka e Vahid Mirjalili
# * Machine Learning with R from por Brett Lantz
