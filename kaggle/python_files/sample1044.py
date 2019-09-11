#!/usr/bin/env python
# coding: utf-8

# ##### In this kernel I intended to do a quick review of Random Forest and a simple application on Titanic problem (basic preprocessing and without feature engineering), which resulted in an accuracy of 79,9% and reach top 15% on this competition (in 05/20/2019).
# 
# ![](https://cdn.pixabay.com/photo/2015/09/09/16/05/forest-931706_960_720.jpg)

# Before Random Forest, let's remind something about Decision Trees.
# Roughly speaking, Decicion Trees split data making questions about feature values at each node. During this split there is the goal of maximizing the **information gain** at each split. Information gain can be understood as the difference between the impurity of a parent node and the sum of impurities of its child nodes. Thus, less impurity on child nodes, more information gain.
# 
# Random Forest is an ensemble of decision trees. Ensemble is a method that combines many models. The final prediction can be, for example, the average of them (regression) or the majority votes (classification). Random Forest consists in a group of decision trees built on samples of training dataset. More specifically, what happens is a sampling with replacement (bagging).
# 
# ![](https://upload.wikimedia.org/wikipedia/commons/7/76/Random_forest_diagram_complete.png)
# 
# 
# But why "random"? Because at each split on nodes, instead of choose the feature that maximize de information gain **among all features**, will be considered only a **random subset** of the features. This brings some diversity to the model.
# 
# Usually default hyperparameters work well and don't require tuning. But increase the number of trees can make a great difference in results despite impacting the computational cost.
# 
# Some of the advantages of Random Forest are the possibility of check feature importance and check partial dependence plots.
# Feature importance is calculated (in sklearn) considering how much a feature can reduce average impurity of model when being chosen to split a node. Partial dependence plot can help understanding the relationship between a feature and the target.
# 
# Random Forest is easy, robust and can collaborate with interpretability.

# # Application

# ### Exploratory data analysis

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from pdpbox import pdp, get_dataset, info_plots
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


train.drop(['PassengerId','Name'],axis=1,inplace=True)
test.drop(['PassengerId','Name'],axis=1,inplace=True)

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


RF=RandomForestClassifier(random_state=1)
scores_rf1=cross_val_score(RF,xtrain,ytrain,scoring='accuracy',cv=5)
np.mean(scores_rf1)

# In[ ]:


RF.fit(xtrain, ytrain)

# ### Feature Importance
# Here we can see the importance of variables like "Sex" and "Age".

# In[ ]:


importances=RF.feature_importances_
feature_importances=pd.Series(importances, index=xtrain.columns).sort_values(ascending=False)
sns.barplot(x= feature_importances[0:10] , y= feature_importances.index[0:10])
plt.xlabel("Feature Importance")
plt.ylabel("Features")
plt.show()

# ### Partial Dependence Plots
# https://pdpbox.readthedocs.io/en/latest/
# 
# Here we can see, basically, that increase on Age, SibSp and Parch seems impact negatively the survival while increase on Fare seems impact positively

# In[ ]:


pdp_data = pdp.pdp_isolate(model=RF, dataset=train, model_features=xtrain.columns, feature='Fare')
pdp.pdp_plot(pdp_data, 'Fare')
plt.show()

# In[ ]:


pdp_data = pdp.pdp_isolate(model=RF, dataset=train, model_features=xtrain.columns, feature='Age')
pdp.pdp_plot(pdp_data, 'Age')
plt.show()

# In[ ]:


pdp_data = pdp.pdp_isolate(model=RF, dataset=train, model_features=xtrain.columns, feature='SibSp')
pdp.pdp_plot(pdp_data, 'SibSp')
plt.show()

# In[ ]:


pdp_data = pdp.pdp_isolate(model=RF, dataset=train, model_features=xtrain.columns, feature='Parch')
pdp.pdp_plot(pdp_data, 'Parch')
plt.show()

# ### Submission

# In[ ]:


#increasing the number of trees
RF2=RandomForestClassifier(random_state=1, n_estimators= 100000)
RF2.fit(xtrain, ytrain)
pred=RF2.predict(xtest)
output=pd.DataFrame({'PassengerId':test2['PassengerId'],'Survived':pred})
output.to_csv('submission.csv', index=False)

# #### References:
# * Hands-On Machine Learning with Scikit-Learn and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems from Aurelien Geron
# * Python Machine Learning: Machine Learning and Deep Learning with Python, scikit-learn, and TensorFlow from  Sebastian Raschka e Vahid Mirjalili 
# * Machine Learning with R from por Brett Lantz 
