#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
base = "../input/"
# base  = "./"
print(os.listdir(base))

# Any results you write to the current directory are saved as output.

# In[ ]:


from xgboost import XGBClassifier
from xgboost import plot_importance
from sklearn import metrics

# In[ ]:


train = pd.read_csv(base+"train.csv")
test = pd.read_csv(base+"test.csv")

# In[ ]:


def col_null():
    print("train:",train.columns[train.isnull().any()].tolist())
    print("="*100)
    print("test:",test.columns[test.isnull().any()].tolist())

# In[ ]:


col_null()

# In[ ]:


def deal_f(row):
    flag = row["Cabin"]
    if pd.isnull(flag):
        return 1
    else:
        return 0
train["Cabin_flag"] = train.apply(deal_f,axis=1)
test["Cabin_flag"] = test.apply(deal_f,axis=1)
train.drop("Cabin",axis=1,inplace=True)
test.drop("Cabin",axis=1,inplace=True)

# In[ ]:


# test.Fare.isnull().sum()
# test.loc[test.Fare.isnull(),:]
# test.groupby(["Pclass"]).agg({"Fare":['mean']})
test.loc[test.Fare.isnull(),"Fare"] = 12.459678

# In[ ]:


# train['Embarked'].value_counts()
train.loc[train.Embarked.isnull(),'Embarked'] = "S"

# In[ ]:


for dataset in [train,test]:
    age_avg = dataset['Age'].mean()
    age_std = dataset['Age'].std()
    age_null_count = dataset['Age'].isnull().sum()
    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
    dataset.loc[dataset['Age'].isnull(),'Age'] = age_null_random_list
    dataset['Age'] = dataset['Age'].astype(int)
train['CategoricalAge'] = pd.qcut(train['Age'], 5,labels=False)
test['CategoricalAge'] = pd.qcut(test['Age'], 5,labels=False)

# In[ ]:


train["Sex"] = train["Sex"].map({"male":1,"female":0}).astype(np.int)
test["Sex"] = test["Sex"].map({"male":1,"female":0}).astype(np.int)

# In[ ]:


train["Embarked"] = train["Embarked"].map({"S":0,"C":1,"Q":2}).astype(np.int)
test["Embarked"] = test["Embarked"].map({"S":0,"C":1,"Q":2}).astype(np.int)
train.drop("Ticket",axis=1,inplace=True)
test.drop("Ticket",axis=1,inplace=True)

# In[ ]:


train.head()

# In[ ]:


train.Embarked.value_counts()

# In[ ]:


# colormap = plt.cm.RdBu
# plt.figure(figsize=(14,12))
# plt.title('Pearson Correlation of Features', y=1.05, size=15)
# sns.heatmap(train.astype(float).corr(),linewidths=0.1,vmax=1.0, 
#             square=True, cmap=colormap, linecolor='white', annot=True)

# In[ ]:


train.head()

# In[ ]:


train.Cabin_flag.value_counts()

# In[ ]:


train.info()

# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



