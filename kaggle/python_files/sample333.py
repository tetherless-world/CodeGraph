#!/usr/bin/env python
# coding: utf-8

# We'll be trying to predict hause price with regression models. 
# 
# **Let's get started!**
# ## Check out the data
# We've been able to get some data from your neighbor for housing prices as a csv set, let's get our environment ready with the libraries we'll need and then import the data!
# ### Import Libraries

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt 
import seaborn as sns 


import os
print(os.listdir("../input"))


# In[ ]:


train = pd.read_csv('../input/train.csv')

# In[ ]:


train.head()

# In[ ]:


X= train[['LotArea','Street', 'Neighborhood','Condition1', 'Condition2','BldgType','HouseStyle','OverallCond', 'Heating','CentralAir','Electrical','1stFlrSF','2ndFlrSF','BsmtHalfBath','FullBath','BedroomAbvGr','KitchenAbvGr','TotRmsAbvGrd','GarageCars','GarageArea','PoolArea', 'SalePrice']]

# In[ ]:


X.head()

# In[ ]:


X.info()

# We have to convert all columns into numeric

# In[ ]:


set(X['Street'])

# In[ ]:


X['Street'] = [1 if i == 'Grvl' 
               else 2 for i in X['Street'] ]

# In[ ]:


set(X['Neighborhood'])

# In[ ]:


X['Neighborhood'] = [1 if i == 'Blmngtn'
                     else 2 if i=='Blueste'
                     else 3 if i=='BrDale'
                     else 4 if i =='BrkSide'
                     else 5 if i =='ClearCr'
                     else 6 if i =='CollgCr'
                     else 7 if i =='Crawfor'
                     else 8 if i =='Edwards'
                     else 9 if i =='Gilbert'
                     else 10 if i =='IDOTRR'
                     else 11 if i =='MeadowV'
                     else 12 if i =='Mitchel'
                     else 13 if i =='NAmes'
                     else 14 if i =='NPkVill'
                     else 15 if i =='NWAmes'
                     else 16 if i =='NoRidge'
                     else 17 if i =='NridgHt'
                     else 18 if i =='OldTown'
                     else 19 if i =='SWISU'
                     else 20 if i =='Sawyer'
                     else 21 if i =='SawyerW'
                     else 22 if i =='Somerst'
                     else 23 if i =='StoneBr'
                     else 24 if i =='Timber'
                     else 25 for i in X['Neighborhood'] ]

# In[ ]:


set(X['Condition1'])

# In[ ]:


X['Condition1'] = [1 if i == 'Artery'
                   else 2 if i=='Feedr'
                   else 3 if i=='Norm'
                   else 4 if i =='PosA'
                   else 5 if i =='PosN'
                   else 6 if i =='RRAe'
                   else 7 if i =='RRAn'
                   else 8 if i =='RRNe'
                   else 9 for i in X['Condition1'] ]

# In[ ]:


set(X['Condition2'])

# In[ ]:


X['Condition2'] = [1 if i == 'Artery'
                   else 2 if i=='Feedr'
                   else 3 if i=='Norm'
                   else 4 if i =='PosA'
                   else 5 if i =='PosN'
                   else 6 if i =='RRAe'
                   else 7 if i =='RRAn'
                   else 8 if i =='RRNe'
                   else 9 for i in X['Condition2'] ]

# In[ ]:


set(X['BldgType'])

# In[ ]:


X['BldgType'] = [1 if i == '1Fam' 
                 else 2 if i=='2fmCon' 
                 else 3 if i=='Duplex' 
                 else 4 if i =='Twnhs' 
                 else 5  for i in X['BldgType'] ]

# In[ ]:


set(X['HouseStyle'])

# In[ ]:


X['HouseStyle'] = [1 if i == '1.5Fin'
                   else 2 if i=='1.5Unf'
                   else 3 if i=='1Story'
                   else 4 if i =='2.5Fin'
                   else 5 if i =='2.5Unf'
                   else 6 if i =='2Story'
                   else 7 if i =='SFoyer'
                   else 8 for i in X['HouseStyle'] ]

# In[ ]:


set(X['Heating'])

# In[ ]:


X['Heating'] = [1 if i == 'Floor' 
                else 2 if i=='GasA' 
                else 3 if i=='GasW' 
                else 4 if i =='Grav' 
                else 5 if i =='OthW' 
                else 6  for i in X['Heating'] ]

# In[ ]:


set(X['CentralAir'])

# In[ ]:


X['CentralAir'] = [1 if i=='Y' 
                   else 2 for i in X['CentralAir']]

# In[ ]:


set(X['Electrical'])

# In[ ]:


X['Electrical'] = [1 if i == 'FuseA' 
                   else 2 if i=='FuseF' 
                   else 3 if i=='FuseP' 
                   else 4 if i =='Mix' 
                   else 5 if i =='SBrkr' else 6  for i in X['Electrical'] ]

# In[ ]:


X.head() #Yes, All colums are numeric

# <a id="1"></a> <br>
# # **Data Visualization **
# 

# In[ ]:


from scipy.stats import norm
plt.figure(figsize=(15,8))
sns.distplot(X['SalePrice'], fit= norm,kde=True)
plt.show()

# In[ ]:


plt.figure(figsize=(15,8))
sns.heatmap(X.corr(),annot=True,cmap='coolwarm')
plt.show()

# In[ ]:


sns.pairplot(X, palette='rainbow')

# In[ ]:


sns.lmplot(x='1stFlrSF',y='SalePrice',data=X)

# In[ ]:


plt.figure(figsize=(16,8))
sns.boxplot(x='GarageCars',y='SalePrice',data=X)
plt.show()

# In[ ]:


plt.figure(figsize=(16,8))
sns.barplot(x='GarageArea',y = 'SalePrice',data=X, estimator=np.mean)
plt.show()

# In[ ]:


plt.figure(figsize=(16,8))
sns.barplot(x='FullBath',y = 'SalePrice',data=X)
plt.show()

# <a id="1"></a> <br>
# # **Linear Regression **
# 
# Let's now begin to train out regression model! We will need to first split up our data into an X array that contains the features to train on, and a y array with the target variable, in this case the Price column. We will toss out the Address column because it only has text info that the linear regression model can't use.

# In[ ]:


x=X[['LotArea','Street', 'Neighborhood','Condition1', 'Condition2','BldgType','HouseStyle','OverallCond', 'Heating','CentralAir','Electrical','1stFlrSF','2ndFlrSF','BsmtHalfBath','FullBath','BedroomAbvGr','KitchenAbvGr','TotRmsAbvGrd','GarageCars','GarageArea','PoolArea']].values
y=X[['SalePrice']].values


# <a id="1"></a> <br>
# ### **Train Test Split **
# Now let's split the data into a training set and a testing set. We will train out model on the training set and then use the test set to evaluate the model.

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=101)

# In[ ]:


# we are going to scale to data

y= y.reshape(-1,1)
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)
y_train = sc_X.fit_transform(y_train)
y_test = sc_y.fit_transform(y_test)

# <a id="1"></a> <br>
# ### **Creating and Training the Model **

# In[ ]:


from sklearn.linear_model import LinearRegression
lm = LinearRegression()

# In[ ]:


lm.fit(X_train,y_train)
print(lm)

# <a id="1"></a> <br>
# ### **Model Evaluation **
# Let's evaluate the model by checking out it's coefficients and how we can interpret them.

# In[ ]:


# print the intercept
print(lm.intercept_)

# In[ ]:


print(lm.coef_)

# <a id="1"></a> <br>
# ### **Predictions from our Model **
# Let's grab predictions off our test set and see how well it did!

# In[ ]:


predictions = lm.predict(X_test)
predictions= predictions.reshape(-1,1)

# In[ ]:


plt.figure(figsize=(15,8))
plt.scatter(y_test,predictions)
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')
plt.show()

# <a id="1"></a> <br>
# ### **Regression Evaluation Metrics**
# 
# Here are three common evaluation metrics for regression problems:
# 
# **Mean Absolute Error (MAE)** is the mean of the absolute value of the errors:
# 
# 1𝑛∑𝑖=1𝑛|𝑦𝑖−𝑦̂ 𝑖|
#  
# **Mean Squared Error (MSE)** is the mean of the squared errors:
# 
# 1𝑛∑𝑖=1𝑛(𝑦𝑖−𝑦̂ 𝑖)2
#  
#  
# **Root Mean Squared Error (RMSE)** is the square root of the mean of the squared errors:
# 
# 1𝑛∑𝑖=1𝑛(𝑦𝑖−𝑦̂ 𝑖)
#  
# **Comparing these metrics**:
# 
# MAE is the easiest to understand, because it's the average error.
# MSE is more popular than MAE, because MSE "punishes" larger errors, which tends to be useful in the real world.
# RMSE is even more popular than MSE, because RMSE is interpretable in the "y" units.
# All of these are loss functions, because we want to minimize them.

# In[ ]:


from sklearn import metrics

# In[ ]:


print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))

# <a id="1"></a> <br>
# # **Gradient Boosting Regression **

# In[ ]:


from sklearn import ensemble
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error, r2_score

# In[ ]:


params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,
          'learning_rate': 0.01, 'loss': 'ls'}
clf = ensemble.GradientBoostingRegressor(**params)

clf.fit(X_train, y_train)

# In[ ]:


clf_pred=clf.predict(X_test)
clf_pred= clf_pred.reshape(-1,1)

# In[ ]:


print('MAE:', metrics.mean_absolute_error(y_test, clf_pred))
print('MSE:', metrics.mean_squared_error(y_test, clf_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, clf_pred)))

# In[ ]:


plt.figure(figsize=(15,8))
plt.scatter(y_test,clf_pred, c= 'brown')
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')
plt.show()

# <a id="1"></a> <br>
# # **Decision Tree Regression **

# 
# The decision tree is a simple machine learning model for getting started with regression tasks.
# 
# **Background**
# A decision tree is a flow-chart-like structure, where each internal (non-leaf) node denotes a test on an attribute, each branch represents the outcome of a test, and each leaf (or terminal) node holds a class label. The topmost node in a tree is the root node. (see here for more details).
# 
# 

# In[ ]:


from sklearn.tree import DecisionTreeRegressor
dtreg = DecisionTreeRegressor(random_state = 0)
dtreg.fit(X_train, y_train)


# In[ ]:


dtr_pred = dtreg.predict(X_test)
dtr_pred= dtr_pred.reshape(-1,1)

# In[ ]:


print('MAE:', metrics.mean_absolute_error(y_test, dtr_pred))
print('MSE:', metrics.mean_squared_error(y_test, dtr_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, dtr_pred)))

# In[ ]:


plt.figure(figsize=(15,8))
plt.scatter(y_test,dtr_pred,c='green')
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')
plt.show()

# <a id="1"></a> <br>
# # **Support Vector Machine Regression **

# Support Vector Machine can also be used as a regression method, maintaining all the main features that characterize the algorithm (maximal margin). The Support Vector Regression (SVR) uses the same principles as the SVM for classification, with only a few minor differences. First of all, because output is a real number it becomes very difficult to predict the information at hand, which has infinite possibilities. In the case of regression, a margin of tolerance (epsilon) is set in approximation to the SVM which would have already requested from the problem. But besides this fact, there is also a more complicated reason, the algorithm is more complicated therefore to be taken in consideration. However, the main idea is always the same: to minimize error, individualizing the hyperplane which maximizes the margin, keeping in mind that part of the error is tolerated.
# 
# 
# ![](https://www.saedsayad.com/images/SVR_1.png)
# ![](https://www.saedsayad.com/images/SVR_2.png)
# ***Linear SVR***
#                                     ![](https://www.saedsayad.com/images/SVR_4.png)                     
#                                     
#                                     
#                                     
#                                     
# ***Non Linear SVR***
# 
# ![](https://www.saedsayad.com/images/SVR_6.png)
# ![](https://www.saedsayad.com/images/SVR_5.png)

# In[ ]:


from sklearn.svm import SVR
svr = SVR(kernel = 'rbf')
svr.fit(X_train, y_train)


# In[ ]:


svr_pred = svr.predict(X_test)
svr_pred= svr_pred.reshape(-1,1)

# In[ ]:


print('MAE:', metrics.mean_absolute_error(y_test, svr_pred))
print('MSE:', metrics.mean_squared_error(y_test, svr_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, svr_pred)))

# In[ ]:


plt.figure(figsize=(15,8))
plt.scatter(y_test,svr_pred, c='red')
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')
plt.show()

# <a id="1"></a> <br>
# # **Random Forest Regression **

# A Random Forest is an ensemble technique capable of performing both regression and classification tasks with the use of multiple decision trees and a technique called Bootstrap Aggregation, commonly known as bagging. What is bagging you may ask? Bagging, in the Random Forest method, involves training each decision tree on a different data sample where sampling is done with replacement.
# 
# ![](https://cdn-images-1.medium.com/max/800/1*jEGFJCm4VSG0OzoqFUQJQg.jpeg)

# In[ ]:


from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor(n_estimators = 50, random_state = 0)
rfr.fit(X_train, y_train)


# In[ ]:


rfr_pred= rfr.predict(X_test)
rfr_pred = rfr_pred.reshape(-1,1)

# In[ ]:


print('MAE:', metrics.mean_absolute_error(y_test, rfr_pred))
print('MSE:', metrics.mean_squared_error(y_test, rfr_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, rfr_pred)))

# In[ ]:


plt.figure(figsize=(15,8))
plt.scatter(y_test,rfr_pred, c='orange')
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')
plt.show()

# # Model Comparison

# **We can say the best working model by loking MSE rates The best working model is Support Vector Machine.**
# We are going to see the error rate. which one is better?
# 

# In[ ]:


error_rate=np.array([metrics.mean_squared_error(y_test, predictions),metrics.mean_squared_error(y_test, clf_pred),metrics.mean_squared_error(y_test, dtr_pred),metrics.mean_squared_error(y_test, svr_pred),metrics.mean_squared_error(y_test, rfr_pred)])

# In[ ]:


plt.figure(figsize=(16,5))
plt.plot(error_rate)

# Now we will use test data .

# In[ ]:


test = pd.read_csv('../input/test.csv')

# In[ ]:


test.info()

# In[ ]:


test.head()

# In[ ]:


x=test[['LotArea','Street', 'Neighborhood','Condition1', 'Condition2','BldgType','HouseStyle','OverallCond', 'Heating','CentralAir','Electrical','1stFlrSF','2ndFlrSF','BsmtHalfBath','FullBath','BedroomAbvGr','KitchenAbvGr','TotRmsAbvGrd','GarageCars','GarageArea','PoolArea']]


# In[ ]:


x.head()

# In[ ]:


x['Street'] = [1 if i == 'Grvl' 
               else 2 for i in x['Street'] ]

# In[ ]:


x['Neighborhood'] = [1 if i == 'Blmngtn'
                     else 2 if i=='Blueste'
                     else 3 if i=='BrDale'
                     else 4 if i =='BrkSide'
                     else 5 if i =='ClearCr'
                     else 6 if i =='CollgCr'
                     else 7 if i =='Crawfor'
                     else 8 if i =='Edwards'
                     else 9 if i =='Gilbert'
                     else 10 if i =='IDOTRR'
                     else 11 if i =='MeadowV'
                     else 12 if i =='Mitchel'
                     else 13 if i =='NAmes'
                     else 14 if i =='NPkVill'
                     else 15 if i =='NWAmes'
                     else 16 if i =='NoRidge'
                     else 17 if i =='NridgHt'
                     else 18 if i =='OldTown'
                     else 19 if i =='SWISU'
                     else 20 if i =='Sawyer'
                     else 21 if i =='SawyerW'
                     else 22 if i =='Somerst'
                     else 23 if i =='StoneBr'
                     else 24 if i =='Timber'
                     else 25 for i in x['Neighborhood'] ]

# In[ ]:


x['Condition1'] = [1 if i == 'Artery'
                   else 2 if i=='Feedr'
                   else 3 if i=='Norm'
                   else 4 if i =='PosA'
                   else 5 if i =='PosN'
                   else 6 if i =='RRAe'
                   else 7 if i =='RRAn'
                   else 8 if i =='RRNe'
                   else 9 for i in x['Condition1'] ]

# In[ ]:


x['Condition2'] = [1 if i == 'Artery'
                   else 2 if i=='Feedr'
                   else 3 if i=='Norm'
                   else 4 if i =='PosA'
                   else 5 if i =='PosN'
                   else 6 if i =='RRAe'
                   else 7 if i =='RRAn'
                   else 8 if i =='RRNe'
                   else 9 for i in x['Condition2'] ]

# In[ ]:


x['BldgType'] = [1 if i == '1Fam' 
                 else 2 if i=='2fmCon' 
                 else 3 if i=='Duplex' 
                 else 4 if i =='Twnhs' 
                 else 5  for i in x['BldgType'] ]


# In[ ]:


x['HouseStyle'] = [1 if i == '1.5Fin'
                   else 2 if i=='1.5Unf'
                   else 3 if i=='1Story'
                   else 4 if i =='2.5Fin'
                   else 5 if i =='2.5Unf'
                   else 6 if i =='2Story'
                   else 7 if i =='SFoyer'
                   else 8 for i in x['HouseStyle'] ]

# In[ ]:


x['Heating'] = [1 if i == 'Floor' 
                else 2 if i=='GasA' 
                else 3 if i=='GasW' 
                else 4 if i =='Grav' 
                else 5 if i =='OthW' 
                else 6  for i in x['Heating'] ]


# In[ ]:


x['CentralAir'] = [1 if i=='Y' 
                   else 2 for i in x['CentralAir']]

# In[ ]:


x['Electrical'] = [1 if i == 'FuseA' 
                   else 2 if i=='FuseF' 
                   else 3 if i=='FuseP' 
                   else 4 if i =='Mix' 
                   else 5 if i =='SBrkr' else 6  for i in x['Electrical'] ]


# In[ ]:


x.info()

# In[ ]:


x['GarageArea'].fillna(x['GarageArea'].mode()[0],inplace=True)
x['GarageCars'].fillna(x['GarageCars'].mode()[0],inplace=True)
x['BsmtHalfBath'].fillna(x['BsmtHalfBath'].mode()[0],inplace=True)

# In[ ]:


x = sc_X.fit_transform(x)

# In[ ]:


test_prediction_svr=svr.predict(x)
test_prediction_svr= test_prediction_svr.reshape(-1,1)

# In[ ]:


test_prediction_svr

# In[ ]:


test_prediction_svr =sc_y.inverse_transform(test_prediction_svr)

# In[ ]:


test_pred_svr = pd.DataFrame(test_prediction_svr, columns=['SalePrice'])

# In[ ]:


df= pd.concat([test,test_pred_svr], axis=1, join='inner')

# In[ ]:


df= df[['Id','SalePrice']]

# In[ ]:


df.to_csv('prediction.csv' , index=False)

# In[ ]:


df.head()

# In[ ]:




# In[ ]:



