#!/usr/bin/env python
# coding: utf-8

# 

# Here I am going to run Support Vector machine on the datasets and do cross validation and then use accuracy score as a parameter to judge the best combination of kernel and values of C which are the hyperparameters in SVM . Here I am taking 1 kernel at a time although we could have avoided it using GridSearchCV. 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

# # Importing all the necessary libraries

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt


# # Reading the comma separated values file into the dataframe

# In[ ]:


df = pd.read_csv('../input/voice.csv')
df.head()

# # Checking the correlation between each feature

# In[ ]:


df.corr()

# # Checking whether there is any null values 

# In[ ]:


df.isnull().sum()

# In[ ]:


df.shape[0]

# In[ ]:


print("Total number of labels: {}".format(df.shape[0]))
print("Number of male: {}".format(df[df.label == 'male'].shape[0]))
print("Number of female: {}".format(df[df.label == 'female'].shape[0]))

# Thus we can see there are equal number of male and female labels

# In[ ]:


df.shape

# There are 21 features and 3168 instances.

# # Separating features and labels

# In[ ]:


X=df.iloc[:, :-1]
X.head()

# # Converting string value to int type for labels

# In[ ]:


from sklearn.preprocessing import LabelEncoder
y=df.iloc[:,-1]

# Encode label category
# male -> 1
# female -> 0

gender_encoder = LabelEncoder()
y = gender_encoder.fit_transform(y)
y

# # Data Standardisation
# Standardization refers to shifting the distribution of each attribute to have a mean of zero and a standard deviation of one (unit variance). It is useful to standardize attributes for a model. Standardization of datasets is a common requirement for many machine learning estimators implemented in scikit-learn; they might behave badly if the individual features do not more or less look like standard normally distributed data.

# In[ ]:


# Scale the data to be between -1 and 1
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

# # Splitting dataset into training set and testing set for better generalisation

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# # Running SVM with default hyperparameter.

# In[ ]:


from sklearn.svm import SVC
from sklearn import metrics
svc=SVC() #Default hyperparameters
svc.fit(X_train,y_train)
y_pred=svc.predict(X_test)
print('Accuracy Score:')
print(metrics.accuracy_score(y_test,y_pred))

# We are getting a good accuracy score.But data are split into training and testing data randomly.Thus a lot depends on how the data got split. When we are not using Random state as a hyperparameter everytime data is splitted differently into training and testing testsa and we get different accuracy score. This is when K-fold Cross validation is a good option

# In[ ]:


svc=SVC(kernel='linear',C=1)
svc.fit(X_train,y_train)
y_pred=svc.predict(X_test)
print('Accuracy Score:')
print(metrics.accuracy_score(y_test,y_pred))

# In[ ]:


svc=SVC(kernel='rbf',C=1)
svc.fit(X_train,y_train)
y_pred=svc.predict(X_test)
print('Accuracy Score:')
print(metrics.accuracy_score(y_test,y_pred))

# Again with kernel as rbf we are getting a marginal less accuracy score s compared to linear kernel

# Thus with K-fold cross validation we are splitting data in K equal parts(in our case K=10).For every value of K we got different training and testing data picking 1/10th of the data a time and train all of them thus covering all the data.In the end we take mean of all the sets. Generally K=10 is taken

# In[ ]:


from sklearn.cross_validation import cross_val_score
svc=SVC(kernel='linear',C=1)
scores = cross_val_score(svc, X, y, cv=10, scoring='accuracy')
print(scores)

# In[ ]:


We can see above how the accuracy score is different everytime.This shows that accuracy score depends upon how the datasets got split.

# In[ ]:


print(scores.mean())

# In K-fold cross validation we generally take the mean of all the scores.

# ### Taking all the values of C and checking out the accuracy score and kernel as linear.

# In[ ]:


C_range=list(range(1,26))
acc_score=[]
for c in C_range:
    svc = SVC(kernel='linear', C=c)
    svc.fit(X_train,y_train)
    y_pred=svc.predict(X_test)
    acc_score.append(metrics.accuracy_score(y_test,y_pred))
print(acc_score)    

# In[ ]:


import matplotlib.pyplot as plt

C_values=list(range(1,26))

# plot the value of C for SVM (x-axis) versus the cross-validated accuracy (y-axis)
plt.plot(C_values,acc_score )
plt.xlabel('Value of C for SVC')
plt.ylabel('Cross-Validated Accuracy')

# From the above plot we can see that accuracy has been close to 97.8% for C=1 and then it drops below 97.65% and remains constant.Thus we can conclude that C=1 is the best hyperparameter for linear kernel.

# ### Taking kernel as **rbc** and and taking different values of C

# In[ ]:


C_range=list(range(1,41))
acc_score=[]
for c in C_range:
    svc = SVC(kernel='rbf', C=c)
    svc.fit(X_train,y_train)
    y_pred=svc.predict(X_test)
    acc_score.append(metrics.accuracy_score(y_test,y_pred))  
print(acc_score)        

# In[ ]:


import matplotlib.pyplot as plt

C_values=list(range(1,41))

# plot the value of C for SVM (x-axis) versus the cross-validated accuracy (y-axis)
plt.plot(C_values,acc_score )
plt.xlabel('Value of C for SVC with kernel rbf')
plt.ylabel('Cross-Validated Accuracy')

# We can see from the plot that the accuracy score is highest for C=3, and then drops at a bit and then again it is highest for C-=8,9,10,11,12,13.Also the accuracy score is slightly more than linear kernel.In this case it is around 98% particular values of C.

# ### Thus from the above two plots we can conclude that **rbf** kernel is performing better than the **linear** kernel.

# # Now performing SVM by taking hyperparameter C=1 and kernel as linear 
# 
# 
# ----------

# In[ ]:


from sklearn.svm import SVC
svc= SVC(kernel='linear',C=1)
svc.fit(X_train,y_train)
y_predict=svc.predict(X_test)
accuracy_score= metrics.accuracy_score(y_test,y_predict)
print(accuracy_score)

# # With K-fold cross validation(where K=10)

# In[ ]:


from sklearn.cross_validation import cross_val_score
svc=SVC(kernel='linear',C=1)
scores = cross_val_score(svc, X, y, cv=10, scoring='accuracy')
print(scores)

# Taking the mean of all the scores

# In[ ]:


print(scores.mean())

# The accuracy is slightly good without K-fold cross validation but it may fail to generalise the unseen data.Hence it is advisable to perform K-fold cross validation where all the data is covered so it may predict unseen data well.

# # Now performing SVM by taking hyperparameter C=8,9,10,11,12,13 and kernel as rbf

# In[ ]:


C_range=[8,9,10,11,12,13]
acc_score_rbf=[]
for c in C_range:
    svc= SVC(kernel='rbf',C=c)
    svc.fit(X_train,y_train)
    y_predict=svc.predict(X_test)
    acc_score_rbf= metrics.accuracy_score(y_test,y_predict)
    print(acc_score_rbf)
    

# Thus we can see that it is giving the same score

# # With K-fold cross validation(where K=10)

# In[ ]:


C_range=[8,9,10,11,12,13]
acc_score_rbf=[]
for c in C_range:
    svc=SVC(kernel='linear',C=c)
    scores = cross_val_score(svc, X, y, cv=10, scoring='accuracy')
    print(scores)

# In[ ]:


print(scores.mean())

# Thus we can conclude that kernel **rbf** and C in the range of 8 to 13 is the good choice since it is performing slightly better.
