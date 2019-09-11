#!/usr/bin/env python
# coding: utf-8

# In[128]:


import numpy as np # linear algebra
import pandas as pd # data processing
import matplotlib.pyplot as plt # to plot inage, graph
import time

# In[129]:



# In[130]:


# dataset for digit (0-9)
from sklearn.datasets import load_digits

# In[131]:


# load dataset
digits = load_digits()

# In[132]:


digits.keys()

# In[133]:


# dataset description
digits.DESCR

# In[134]:


# already processed images
digits.images[0]

# In[135]:


# predictors,independent variables, features
digits.data

# In[136]:


# target variable, class, dependent variable
digits.target

# In[137]:


# There 1797 images (8 by 8 for a dimension of 64)
print('Image Data Shape', digits.images.shape)

# In[138]:


# 1797 labels
print('Label Data Shape', digits.target.shape)

# In[139]:


X = digits.images

# In[141]:


plt.figure(figsize=(20,10))
columns = 5
for i in range(5):
    plt.subplot(5 / columns + 1, columns, i + 1)
    plt.imshow(X[i],cmap=plt.cm.gray_r,interpolation='nearest')

# In[143]:


from sklearn.metrics import accuracy_score,confusion_matrix # metrics error
from sklearn.model_selection import train_test_split # resampling method

# In[144]:


X = digits.data
y = digits.target

# In[145]:


# since its a multi-class prediction, in order to prevent error we need some library
from sklearn.multiclass import OneVsRestClassifier

# In[146]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# In[147]:


from sklearn.neighbors import KNeighborsClassifier

# In[148]:


knn = OneVsRestClassifier(KNeighborsClassifier())

# In[149]:


knn.fit(X_train,y_train)

# In[150]:


# predict for one observation
knn.predict(X_test[0].reshape(1,-1))

# In[151]:


# predict for multiple observation (images) at once
knn.predict(X_test[0:10])

# In[152]:


# make prediction on entire test data
predictions = knn.predict(X_test)

# In[156]:


# 98%
print('KNN Accuracy: %.3f' % accuracy_score(y_test,predictions))

# In[154]:


# to create nice confusion metrics
import seaborn as sns

# In[155]:


cm = confusion_matrix(y_test,predictions)
plt.figure(figsize=(9,9))
sns.heatmap(cm,annot=True, fmt='.3f', linewidths=.5, square=True,cmap='Blues_r')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
all_sample_title = 'Accuracy Score: {0}'.format(accuracy_score(y_test,predictions))
plt.title(all_sample_title,size=15)

# # KNN accuracy is 98%

# In[ ]:



