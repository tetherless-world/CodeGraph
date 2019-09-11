#!/usr/bin/env python
# coding: utf-8

# In[163]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.simplefilter('ignore')
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

# In[164]:


training_data = pd.read_csv('../input/train.csv')

# In[165]:


training_data.head()

# In[166]:


training_data.isnull().values.any()

# In[167]:


#gives birdeye view of columns which might have null values
sns.heatmap(training_data.isnull(),yticklabels=False,cbar=False)

# In[168]:


sns.distplot(training_data['Age'].dropna(),kde=False,bins=30)

# In[169]:


sns.barplot(x='Pclass',y='Fare',data=training_data,ci=None)

# The passengers in class 1 were wealthier as compared to the ones in class 2 and 3

# In[170]:


sns.boxplot(x='Pclass',y='Age',data=training_data)

# Shows that as the class increasing the age decreases - wealthier passengers are older. We will use this data to fill in the Null values in age column!

# In[171]:


def compute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    if pd.isnull(Age):
        if Pclass == 1:
            return 37
        elif Pclass == 2:
            return 29
        else:
            return 25
    else:
        return Age

# In[172]:


training_data['Age'] = training_data[['Age','Pclass']].apply(compute_age,axis=1)

# In[173]:


#gives birdeye view of columns which might have null values
sns.heatmap(training_data.isnull(),yticklabels=False,cbar=False)

# As most of the cabin data is not defined, if we compute the missing data from the very less known data, it might lead to distorted computations. Hence it is best to not take this column into consideration at all.

# In[174]:


training_data.drop('Cabin',axis=1,inplace=True)

# In[175]:


training_data.isnull().values.any()

# In[176]:


null_columns=training_data.columns[training_data.isnull().any()]
training_data[null_columns].isnull().sum()

# In[177]:


training_data.dropna(inplace=True)

# In[178]:


training_data.isnull().values.any()

# In[179]:


def categorise_sex(cols):
    age = cols[0]
    sex = cols[1]
    
    if age<16:
        return 'child'
    else:
        return cols[1]

# In[180]:


training_data['Sex'] = training_data[['Age','Sex']].apply(categorise_sex,axis=1)

# In[181]:


sns.countplot(x='Survived',data=training_data,hue='Sex')

# Of the people who managed to survive, mostly were women and of the people who died, mostly were men.

# In[182]:


sns.countplot(x='Survived',hue='Pclass',data=training_data)

# Of the people who survived, most of them were the wealthier ones in class 1 while the ones who died were mostly the ones in the lowest passenger class.

# In[183]:


def is_alone(cols):
    siblings_or_spouse = cols[0]
    parents_or_child = cols[1]
    if (siblings_or_spouse == 0) & (parents_or_child == 0):
        return 1
    else:
        return 0

training_data['Is_Alone'] = training_data[['SibSp','Parch']].apply(is_alone,axis=1)

# In[184]:


training_data.head()

# In[185]:


sns.countplot(x='Survived',hue='Is_Alone',data=training_data)

# In[186]:


training_data.info()

# In[187]:


training_data['Pclass'] = training_data['Pclass'].astype('object')
training_data['Is_Alone'] = training_data['Is_Alone'].astype('object')

# In[188]:


embark = pd.get_dummies(training_data['Embarked'],drop_first=True)
sex = pd.get_dummies(training_data['Sex'],drop_first=True)
pclass = pd.get_dummies(training_data['Pclass'],drop_first=True)

# In[189]:


training_data = pd.concat([training_data,sex,embark,pclass],axis=1)

# In[190]:


training_data.head()

# In[191]:


training_data.drop(['Sex','Embarked','Name','Ticket','Pclass','SibSp','Parch'],axis=1,inplace=True)

# In[192]:


training_data.head()

# In[193]:


training_data.drop('PassengerId',axis=1,inplace=True)

# In[194]:


X = training_data.drop('Survived',axis=1)
y = training_data['Survived']

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)

from sklearn.linear_model import LogisticRegression

logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)

# In[195]:


pred = logmodel.predict(X_test)

# In[196]:


from sklearn.metrics import classification_report

print(classification_report(y_test,pred))

# In[197]:


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(X_train,y_train)
rfc_pred = rfc.predict(X_test)
print(classification_report(y_test,rfc_pred))

# In[198]:


#tuning RFClassifier to get best results

from sklearn.model_selection import GridSearchCV

n_folds = 5

# parameters to build the model on
parameters = {'max_depth': range(10, 30, 3),
              'min_samples_leaf': [3,4,5,6] }

# instantiate the model
rf = RandomForestClassifier(random_state=42)

rf = GridSearchCV(rf, param_grid=parameters,
                  cv=n_folds, 
                 scoring="accuracy")

rf.fit(X_train,y_train)

print('\n'+'Enter the best parameters: ',rf.best_params_)

rf_tuned = RandomForestClassifier(bootstrap=True,
                             max_depth=rf.best_params_['max_depth'],
                             min_samples_leaf=rf.best_params_['min_samples_leaf'],
                             n_estimators=100,
                             random_state=42)

rf_tuned.fit(X_train,y_train)

rf_tuned_pred = rf_tuned.predict(X_test)

print(classification_report(y_test,rf_tuned_pred))

# In[199]:


#Using SVM

from sklearn.svm import SVC

model = SVC()

model.fit(X_train,y_train)

SVM_predictions = model.predict(X_test)

print(classification_report(y_test,SVM_predictions))

# In[200]:


#tuning SVM to get best results
param_grid = {'C':[0.1,1,10,100,1000,10000,100000],'gamma':[1,.1,.01,.001,.0001,.00001]}
grid = GridSearchCV(SVC(),param_grid,verbose=3)

grid.fit(X_train,y_train)

# In[201]:


grid.best_params_

# In[202]:


grid.best_estimator_

# In[203]:


grid_predictions = grid.predict(X_test)
print(classification_report(y_test,grid_predictions))

# In[204]:


testing_data = pd.read_csv('../input/test.csv')

# In[205]:


testing_data.head()

# In[206]:


testing_data['Age'] = testing_data[['Age','Pclass']].apply(compute_age,axis=1)

# In[207]:


testing_data.head()

# In[208]:


testing_data['Sex'] = testing_data[['Age','Sex']].apply(categorise_sex,axis=1)

# In[209]:


testing_data['Is_Alone'] = testing_data[['SibSp','Parch']].apply(is_alone,axis=1)

# In[210]:


test_data = testing_data[['Pclass','Sex','Age','Fare','Embarked', 'Is_Alone']]

# In[211]:


test_data.head()

# In[212]:


#gives birdeye view of columns which might have null values
sns.heatmap(test_data.isnull(),yticklabels=False,cbar=False)

# In[213]:


test_data.isnull().values.any()

# In[214]:


null_columns=test_data.columns[test_data.isnull().any()]
test_data[null_columns].isnull().sum()

# In[215]:


ax = sns.boxplot(x='Pclass',y='Fare',data=test_data)
ax.set_ylim(0,100)

# In[216]:


def compute_fare(cols):
    Fare = cols[1]
    Pclass = cols[0]
    if pd.isnull(Fare):
        if Pclass == 1:
            return 60
        elif Pclass == 2:
            return 18
        else:
            return 15
    else:
        return Fare

# In[217]:


test_data['Fare'] = test_data[['Pclass','Fare']].apply(compute_fare,axis=1)

# In[218]:


test_data.isnull().values.any()

# In[219]:


test_data['Pclass'] = test_data['Pclass'].astype('object')
test_data['Is_Alone'] = test_data['Is_Alone'].astype('object')

# In[220]:


test_embark = pd.get_dummies(test_data['Embarked'],drop_first=True)
test_sex = pd.get_dummies(test_data['Sex'],drop_first=True)
test_pclass = pd.get_dummies(test_data['Pclass'],drop_first=True)

# In[221]:


test_data = pd.concat([test_data,test_embark,test_pclass,test_sex],axis=1)

# In[222]:


test_data.head()

# In[223]:


test_data.drop(['Sex','Pclass','Embarked'],axis=1,inplace=True)

# In[224]:


test_data.head()

# In[231]:


predictions = rf_tuned.predict(test_data)

# In[232]:


predictions = pd.Series(predictions)

# In[233]:


result = pd.concat([testing_data['PassengerId'],predictions],axis=1)

# In[234]:


result.columns = ['PassengerId','Survived']

# In[235]:


result.head()

# In[236]:


filename = 'Titanic Predictions - RF_TUNED.csv'

result.to_csv(filename,index=False)

print('Saved file: ' + filename)

# In[ ]:




# In[ ]:



