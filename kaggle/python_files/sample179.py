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
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

# In[ ]:


# download .csv file from Kaggle Kernel

from IPython.display import HTML
import pandas as pd
import numpy as np
import base64

# function that takes in a dataframe and creates a text link to  
# download it (will only work for files < 2MB or so)
def create_download_link(df, title = "Download CSV file", filename = "data.csv"):  
    csv = df.to_csv()
    b64 = base64.b64encode(csv.encode())
    payload = b64.decode()
    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'
    html = html.format(payload=payload,title=title,filename=filename)
    return HTML(html)

# create a random sample dataframe
df = pd.DataFrame(np.random.randn(50, 4), columns=list('ABCD'))

# create a link to download the dataframe

# In[87]:


# Read data 
train_data = pd.read_csv("../input/train.csv")
test_data = pd.read_csv("../input/test.csv")

# In[89]:


# Take a quick look at the data structure
train_data.head(10)

# In[ ]:


# Take a quick look at the data structure
test_data.head()

# In[ ]:


# Take a quick look at the data structure
print("The size of train data", train_data.shape)
print("The size of test data", test_data.shape)

# From the above, you can notice that train and test data almost 
# share the same number of columns, except "Sale Price" in the training set.

# The `info()` method is useful to get a quick description of the data, in particular the total number of rows and each attributes types and number of non-null values.

# In[ ]:


train_data.info()

# In[ ]:


# The `info()` method is useful to get a quick description of the data, 
# in particular the total number of rows and each attributes types and number of non-null values.
test_data.info()

# In[ ]:


#correlation matrix
import matplotlib.pyplot as plt
import seaborn as sns


corrmat = train_data.corr()
f, ax = plt.subplots(figsize=(20, 9))
sns.heatmap(corrmat, vmax=.8, annot=True);

# We will analyze correlation using the following 
# (Reference: [Comprehensive data exploration with Python](https://www.kaggle.com/pmarcelino/comprehensive-data-exploration-with-python))
# 
# * Correlation matrix (heatmap style).
# * `SalePrice` correlation matrix (zoomed heatmap style).
# * Scatter plots between the most correlated variables (move like Jagger style).

# In[ ]:


# Correlation matrix (heatmap style).
# correlation matri


corrmat = train_data.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);

# In[ ]:


# most correlated features
corrmat = train_data.corr()
top_corr_features = corrmat.index[abs(corrmat["SalePrice"])>0.5]
plt.figure(figsize=(10,10))
g = sns.heatmap(train_data[top_corr_features].corr(),annot=True,cmap="RdYlGn")

# ### Observations based on heatmap
# 1. The variable `OverallQual` has white color and it seems that it has strong co-relation with `SalePrice` target variable. 
# 
# 2. The variable `TotalBsmtSF` and `1stFlrSF` show good correlation with `SalePrice` variable. 
# 
# 3. The variable `GrLiveArea` shows good correlation with `SalePrice` variable.
# 
# 4. The GarageX variables such as `GarageCars` and `GarageArea`  also show good correlation with `SalePrice` variable.  
# 
# 5. Apartfrom the above, feature variables such as **Lot** Variable (`LotFrontage`, `LotArea`), `YearBuilt`, `YearRemodAdd`, `MasVnrArea`, `TotalRmsAbvGrd`, `Fireplaces`, `GarageYrBlt` shows some correlation with target variable `SalePrice`. 
# 
# Let's dig this relationship more using Zoom heatmap.
# 
# We will use the following:
# *  Numerical variable: Scatter plot 
#     * Numerical variable  =  [`TotalBsmtSF`, `1stFlrSF`, `GrLivArea`,`GarageArea`, `LotFrontage`, `LotArea`, `MasVnrArea`]   
# *  Categorical variable: Box plot
#     * Categorical variable  = [`OverallQual`,`GarageCars`, `YearBuilt`, `YearRemodAdd`, `Fireplaces`, `GarageYrBlt`,`TotRmsAbvGrd`]
# 

# ### Relationship with Numerical variable

# In[ ]:


# Scatter Plot for numerical variables

li_cat_feats = [ 'TotalBsmtSF', '1stFlrSF', 'GrLivArea', 'GarageArea', 'LotFrontage', 'LotArea', 'MasVnrArea']   
target = 'SalePrice'
nr_rows = 2
nr_cols = 4

fig, axs = plt.subplots(nr_rows, nr_cols, figsize=(nr_cols*4,nr_rows*3))

for r in range(0,nr_rows):
    for c in range(0,nr_cols):  
        i = r*nr_cols+c
        if i < len(li_cat_feats):
            sns.scatterplot(x=li_cat_feats[i], y=target, data=train_data, ax = axs[r][c])
    
plt.tight_layout()    
plt.show()   

# ![](http://intranet.tdmu.edu.ua/data/kafedra/internal/distance/classes_stud/english/1course/Medical%20statistics/08.%20Types%20of%20correlation.files/image013.gif)

# In[ ]:


#li_cat_feats = list(categorical_feats)

# Box plot for categorical variables

li_cat_feats = ['OverallQual','GarageCars', 'YearBuilt', 'YearRemodAdd', 'Fireplaces', 'GarageYrBlt', 'TotRmsAbvGrd']
target = 'SalePrice'
nr_rows = 2
nr_cols = 4

fig, axs = plt.subplots(nr_rows, nr_cols, figsize=(nr_cols*4,nr_rows*3))

for r in range(0,nr_rows):
    for c in range(0,nr_cols):  
        i = r*nr_cols+c
        if i < len(li_cat_feats):
            sns.boxplot(x=li_cat_feats[i], y=target, data=train_data, ax = axs[r][c])
plt.tight_layout()    
plt.show()   

# ### Check missing values and Data imputation

# In[ ]:


# total = train_data.isnull().sum().sort_values(ascending=False)
# percent = (train_data.isnull().sum()/train_data.isnull().count()).sort_values(ascending=False)
# missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
# missing_data.head(19)

sns.set_style("whitegrid")
missing = train_data.isnull().sum()
missing = missing[missing > 0]
missing.sort_values(inplace=True)
missing.plot.bar()

# In[ ]:


# Fill NA value with Zero
train_data['MasVnrArea'].fillna(0, inplace=True)
test_data['MasVnrArea'].fillna(0, inplace=True)

# Fill NA value with Zero
train_data['GarageYrBlt'].fillna(0, inplace=True)
test_data['GarageYrBlt'].fillna(0, inplace=True)

# In[ ]:


# Identify where the Null Value in the column
print(train_data[train_data["Electrical"].isnull()]["Electrical"])
# Mode Imputation
train_data["Electrical"].value_counts()
# Fill the column value with mode of column
train_data["Electrical"].fillna('SBrkr', inplace=True)

# In[ ]:


# Fill the missing values in "LotFrontage" attributes with mean() - Mean Imputation
train_data["LotFrontage"].fillna(int(train_data["LotFrontage"].mean()), inplace=True)
test_data["LotFrontage"].fillna(int(test_data["LotFrontage"].mean()), inplace=True)

# In[ ]:


string_column = ['Alley', 'MasVnrType',  'BsmtQual',  'BsmtCond', 'BsmtExposure', 
                 'BsmtFinType1', 'BsmtFinType2', 'FireplaceQu', 'GarageType', 'GarageQual', 'GarageFinish' , 'GarageCond', 'PoolQC', 'Fence', 'MiscFeature' ]
for column in string_column:
    train_data[column] = train_data[column].fillna("None")
    test_data[column] = test_data[column].fillna("None")

# In[ ]:


# A histogram shows the number of instances (on the vertical axis) that have a given value range(on the horizontal axis)
import matplotlib.pyplot as plt
train_data.hist(bins=50, figsize=(20, 15))
plt.show()

# #### Observations, based on histogram
# 
# - As we see, the target variable `Sale Price` is not normally distributed.  This can reduce the performance of the ML regression models because some ML assumes normal distribution. Therefore we make a log transformation, the resulting distrbution looks much better.
# 
# - Like the target variable (`SalePrice`) , also some of the feature values (such as `GrLivArea`, `LotArea`, `1stFlrSF`, `GarageArea` , `LotFrontage` , `TotalBsmtSF`)  are not normally distributed and it is therefore better to use log values both in `train_data` and `test_data`. 

# In[ ]:


# Re-plotting the distribution of Sales Price
import seaborn as sns
sns.distplot(train_data["SalePrice"])

# ![](https://www.safaribooksonline.com/library/view/clojure-for-data/9781784397180/graphics/7180OS_01_180.jpg)
# 
# for more info  click [Here](https://www.google.co.in/url?sa=t&rct=j&q=&esrc=s&source=web&cd=3&cad=rja&uact=8&ved=0ahUKEwi7i-jN-K7XAhWJKo8KHbIHAV4QFgguMAI&url=http%3A%2F%2Fwhatis.techtarget.com%2Fdefinition%2Fskewness&usg=AOvVaw1LJhHdq4KFEYIpfdXjOlF-)

# ![skewness](https://i.stack.imgur.com/7iSYs.png)
# 

# Learn more about [skeness](https://whatis.techtarget.com/definition/skewness)

# As we see, the target variable `Sale Price` is not normally distributed.  This can reduce the performance of the ML regression models because some ML assumes normal distribution. Therefore we make a log transformation, the resulting distrbution looks much better. 
# 
# Reference: **[House Prices: EDA to ML (Beginner)](https://www.kaggle.com/dejavu23/house-prices-eda-to-ml-beginner#Plots-of-relation-to-target-for-all-numerical-features) **

# In[ ]:


train_data["SalePrice_Log"] = np.log(train_data['SalePrice'])
sns.distplot(train_data["SalePrice_Log"]);  

# In[ ]:


# Drop columns that are not necessary 
train_data.drop(['SalePrice'], axis=1, inplace=True)
# test_data.drop(['Id'], axis=1, inplace=True)

# ### Use encoding for text and categorical attributes

# In[ ]:


# Before we encode - drop not necessary columns
train_data.drop(["Utilities", "Condition2", "HouseStyle", "RoofStyle", "RoofMatl", "Exterior1st", "Exterior2nd", "Heating", "Electrical", "GarageQual", "PoolQC", "MiscFeature"], axis=1, inplace=True)
test_data.drop(["Utilities", "Condition2", "HouseStyle", "RoofStyle", "RoofMatl", "Exterior1st", "Exterior2nd", "Heating", "Electrical", "GarageQual", "PoolQC", "MiscFeature"], axis=1, inplace=True)

# In[ ]:


# test-data set - mode imputation
test_data["MSZoning"].fillna('RL',  inplace=True)
test_data["BsmtFinSF2"].fillna(0, inplace=True)
test_data["BsmtFullBath"].fillna(0, inplace=True)
test_data["BsmtHalfBath"].fillna(0, inplace=True)
test_data["KitchenQual"].fillna('TA', inplace=True)
test_data["Functional"].fillna('Typ', inplace=True)
test_data["GarageCars"].fillna(0, inplace=True)
test_data["SaleType"].fillna('WD', inplace=True)

# test-dataset - mean imputation
test_data["BsmtFinSF1"].fillna(int(test_data["BsmtFinSF1"].mean()), inplace=True)
test_data["BsmtUnfSF"].fillna(int(test_data["BsmtUnfSF"].mean()), inplace=True)
test_data["TotalBsmtSF"].fillna(int(test_data["TotalBsmtSF"].mean()), inplace=True)
test_data["GarageArea"].fillna(int(test_data["GarageArea"].mean()), inplace=True)

# In[ ]:


create_download_link(test_data)

# In[ ]:


# Apply encoding both on training and testing dataset
train_data = pd.get_dummies(train_data)
test_data = pd.get_dummies(test_data)

# #### Our dataset is ready for ML now

# In[ ]:


# now split the training data set into two : train_set and test_set
# train_set will be used to train the ML model
# test_set will be used to test the score of the model
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(train_data, test_size=0.2, random_state = 42)

# In[ ]:


# y_train
train_target_col = train_set["SalePrice_Log"]

# X_train
train_set_drop = train_set.drop(["SalePrice_Log"], axis=1)

#y_test
test_target_col = test_set["SalePrice_Log"]

# X_test
test_set_drop = test_set.drop(["SalePrice_Log"], axis=1)

# ### Linear Regression

# In[ ]:


# Linear Regression model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

lin_reg = LinearRegression()

# Train the model
lin_reg.fit(train_set_drop,train_target_col)

# predict
housing_predictions = lin_reg.predict(test_set_drop)

# Calculate the Root Mean Square Error
lin_mse = mean_squared_error(housing_predictions,test_target_col)
lin_rmse = np.sqrt(lin_mse)
lin_rmse
# result -- 23082.15859701882

# ### Cross- validation 

# In[ ]:


# Evaluate Linear Regression using Cross-Validation Technique
from sklearn.model_selection import cross_val_score

# prepare training set for cross validation
train_target_cross_val = train_data["SalePrice_Log"]
train_feature_cross_val = train_data.drop(["SalePrice_Log"], axis = 1)

scores = cross_val_score(lin_reg, train_feature_cross_val, train_target_cross_val, scoring="neg_mean_squared_error", cv=3)
linear_regression_scores = np.sqrt(-scores)
print("Score" , linear_regression_scores)
print("\n")
print("Mean" , linear_regression_scores.mean())
print("\n")
print("Standard deviation" , linear_regression_scores.std())

# ### Decision Tree Regression

# In[ ]:


# Decision Tree Regressor 
from sklearn.tree import DecisionTreeRegressor
tree_reg = DecisionTreeRegressor()
tree_reg.fit(train_set_drop,train_target_col)

housing_predictions = tree_reg.predict(test_set_drop)

# Calculate the Root Mean Square Error
tree_mse = mean_squared_error(housing_predictions,test_target_col)
tree_rmse = np.sqrt(tree_mse)
tree_rmse 
# result -- 0.21180931905855785

# In[ ]:


# Cross validation for tree regression
scores = cross_val_score(tree_reg, train_feature_cross_val, train_target_cross_val, scoring="neg_mean_squared_error", cv=10)
tree_regression_scores = np.sqrt(-scores)
print("Score" , tree_regression_scores)
print("\n")
print("Mean" , tree_regression_scores.mean())
print("\n")
print("Standard deviation" , tree_regression_scores.std())

# ### Random Forest Regression

# In[ ]:


# Random Forest Regression
from sklearn.ensemble import RandomForestRegressor
forest_reg = RandomForestRegressor()
forest_reg.fit(train_set_drop, train_target_col)
housing_predictions = forest_reg.predict(test_set_drop)

# Calculate the Root Mean Square Error
forest_mse = mean_squared_error(housing_predictions,test_target_col)
forest_rmse = np.sqrt(forest_mse)
forest_rmse 
# result -- 0.21180931905855785

# In[ ]:


# Cross validation for forest regressor
scores = cross_val_score(forest_reg, train_feature_cross_val, train_target_cross_val, scoring="neg_mean_squared_error", cv=10)
forest_regression_scores = np.sqrt(-scores)
print("Score" , forest_regression_scores)
print("\n")
print("Mean" , forest_regression_scores.mean())
print("\n")
print("Standard deviation" , forest_regression_scores.std())

# In[ ]:


predicted_prices = forest_reg.predict(test_data)

my_submission = pd.DataFrame({'Id': test_data.Id, 'SalePrice': predicted_prices})
# you could use any filename. We choose submission here
my_submission.to_csv('submission.csv', index=False)

# In[ ]:


pd.read_csv('submission.csv')
