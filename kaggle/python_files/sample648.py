#!/usr/bin/env python
# coding: utf-8

# In[103]:


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

# In[132]:


import random
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
plt.style.use(style='ggplot')
random.seed(12)

# In[133]:


train = pd.read_csv('../input/train.csv')
test  = pd.read_csv('../input/test.csv')
train.head()

# In[134]:


train.shape

# In[135]:


train.SalePrice.describe()

# In[136]:


plt.hist(train.SalePrice, color='green')
plt.show()

# In[137]:


target = np.log(train.SalePrice)
plt.hist(target, color='green')
plt.show()

# In[138]:


numeric_features = train.select_dtypes(include=[np.number])
numeric_features.dtypes

# In[139]:


corr = numeric_features.corr()
print(corr['SalePrice'].sort_values(ascending=False)[:5])
print('\n')
print(corr['SalePrice'].sort_values(ascending=False)[-5:])

# In[140]:


quality_pivot = train.pivot_table(index='OverallQual', values='SalePrice', aggfunc=np.median)
quality_pivot.plot(kind='bar', color='green')
plt.xlabel('Overral Quality')
plt.ylabel('Median Sale Price')
plt.xticks(rotation=0)
plt.show()

# In[141]:


plt.scatter(x=train['GrLivArea'], y=np.log(train['SalePrice']), color='green')
plt.ylabel('Sale Price')
plt.xlabel('GrLivArea')
plt.show()

# In[142]:


plt.scatter(x=train['GarageArea'], y=np.log(train['SalePrice']), color='green')
plt.ylabel('Sale Price')
plt.xlabel('GrLivArea')
plt.show()

# In[143]:


# remove outliers
train = train[train['GarageArea'] < 1200]
plt.scatter(x=train['GarageArea'], y=np.log(train['SalePrice']), color='green')
plt.xlim(-200, 1600)
plt.ylabel('Sale Price')
plt.xlabel('Garage Area')
plt.show()

# In[144]:


nulls = pd.DataFrame(train.isnull().sum().sort_values(ascending=False)[:30])
nulls.columns = ['Null count']
nulls.index.name = 'Feature'
nulls

# In[145]:


categorical_features = train.select_dtypes(exclude=[np.number])
categorical_features.describe()

# In[146]:


train['street'] = pd.get_dummies(train.Street, drop_first=True)
test['street']  = pd.get_dummies(test.Street, drop_first=True)

# In[147]:


train.street.value_counts()

# In[148]:


data = train.select_dtypes(include=[np.number]).interpolate().dropna()
data.isnull().sum()

# In[149]:


y = np.log(train.SalePrice)
x =  data.drop(['Id', 'SalePrice'], axis=1)

# In[150]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=25, random_state=42)

# In[152]:


model = LinearRegression()
model.fit(x_train, y_train)

# In[153]:


model.score(x_test, y_test)

# In[154]:


prediction = model.predict(x_test)

# In[155]:


mean_squared_error(y_test, prediction)

# In[156]:


submission = pd.DataFrame()
submission['Id'] = test.Id
features = test.select_dtypes(include=[np.number]).drop(['Id'], axis=1).interpolate()
predictions = model.predict(features)

# In[129]:


final_predictions = np.exp(predictions)
submission['SalePrice'] = final_predictions

# In[130]:


submission.head()

# In[131]:


 submission.to_csv('submission2.csv', index=False)

# In[ ]:



