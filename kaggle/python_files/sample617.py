#!/usr/bin/env python
# coding: utf-8

# In[ ]:




# In[61]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

#import numpy as np # linear algebra
#import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
#%matplotlib inline
import warnings
warnings.filterwarnings("ignore")
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
print(train.head())
# Any results you write to the current directory are saved as output.

# In[62]:


#heatmap for highlighting the null values
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')

# In[63]:


def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):

        if Pclass == 1:
            return 37

        elif Pclass == 2:
            return 29

        else:
            return 24

    else:
        return Age

# In[64]:


#to overwrite null values with the average value of that Pclass
train['Age'] = train[['Age','Pclass']].apply(impute_age,axis=1)
print(train.head())

# In[65]:


#to check if null values are removed in Age through heatmap
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')

# In[66]:


#dropping Cabin Column since it contains more of null values
train.drop('Cabin',axis=1,inplace=True)
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
#null valued recors are furthered dropped
train.dropna(inplace=True)

# In[67]:



#train.info()
#creating dummy variables for training and testing
sex = pd.get_dummies(train['Sex'],drop_first=True)
print(sex)




# In[68]:


embark = pd.get_dummies(train['Embarked'],drop_first=True)
print(embark)

# In[69]:


train.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)
print(train)

# In[70]:


train = pd.concat([train,sex,embark],axis=1)
print(train)

# In[71]:


#splitting into traina nd test sets 80-20
X_train, X_test, y_train, y_test = train_test_split(train.drop('Survived',axis=1), 
                                                    train['Survived'], test_size=0.20, 
                                                    random_state=101)

# In[72]:


#using logistic regression
print('------------Logistic Regrssion--------------')
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)
predictions = logmodel.predict(X_test)
#print(predictions)
print('Confusion Matrix')
print(confusion_matrix(y_test,predictions))


# In[73]:


print(classification_report(y_test,predictions))

# In[74]:


#decision tree model
print("\n-----------Decision Tree-------")
dt_model=DecisionTreeClassifier()
dt_model.fit(X_train,y_train)

dt_pred = dt_model.predict(X_test)
print('Confusion Matrix')
print(confusion_matrix(y_test,dt_pred))

# In[75]:


print(classification_report(y_test,dt_pred))

# In[76]:


#random forest tree model
print("\n-----------Random Forest---------")
rf_model = RandomForestClassifier(n_estimators=12)
rf_model.fit(X_train,y_train)
rf_pred = rf_model.predict(X_test)
print('Confusion Matrix')
print(confusion_matrix(y_test,rf_pred))

# In[77]:


print(classification_report(y_test,rf_pred))

# In[78]:


error_rate = []

# Will take some time
for i in range(1,50):
    
    rf_model = RandomForestClassifier(n_estimators=i)
    rf_model.fit(X_train,y_train)
    pred_i = rf_model.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))
    plt.figure(figsize=(10,6))
plt.plot(range(1,50),error_rate,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')
rf= RandomForestClassifier(n_estimators=30)
rf.fit(X_train,y_train)
rf_pre=rf.predict(X_test)

# In[79]:


#test file
print("\n----------------TEST SET-----------------\n")
#test = pd.read_csv('test.csv')
test.drop('Cabin',axis=1,inplace=True)
test['Fare'].fillna(test['Fare'].mean(), inplace=True)
test['Age'] = test[['Age','Pclass']].apply(impute_age,axis=1)
sex_test = pd.get_dummies(test['Sex'],drop_first=True)
embark_test= pd.get_dummies(test['Embarked'],drop_first=True)
test.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)
test = pd.concat([test,sex_test,embark_test],axis=1)
test_prediction = rf.predict(test)
#print(test_prediction)
test_pred = pd.DataFrame(test_prediction, columns= ['Survived'])
new_test = pd.concat([test, test_pred], axis=1, join='inner')
print(new_test.head())


# In[81]:


df= new_test[['PassengerId' ,'Survived']]
df.to_csv('predictions.csv' , index=False)
print('Saved file: ' + 'predictions.csv')
