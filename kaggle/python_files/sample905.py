#!/usr/bin/env python
# coding: utf-8

# # <div style="text-align: center">Statistical Analysis for Elo</div>
# ### <div align="center"><b>Quite Practical and Far from any Theoretical Concepts</b></div>
# <div style="text-align:center">last update: <b>19/02/2019</b></div>
# <img src='http://s8.picofile.com/file/8344134250/KOpng.png'>
# 
# >You are reading **10 Steps to Become a Data Scientist** and are now in the 8th step : 
# 
# 1. [Leren Python](https://www.kaggle.com/mjbahmani/the-data-scientist-s-toolbox-tutorial-1)
# 2. [Python Packages](https://www.kaggle.com/mjbahmani/the-data-scientist-s-toolbox-tutorial-2)
# 3. [Mathematics and Linear Algebra](https://www.kaggle.com/mjbahmani/linear-algebra-for-data-scientists)
# 4. [Programming &amp; Analysis Tools](https://www.kaggle.com/mjbahmani/20-ml-algorithms-15-plot-for-beginners)
# 5. [Big Data](https://www.kaggle.com/mjbahmani/a-data-science-framework-for-quora)
# 6. [Data visualization](https://www.kaggle.com/mjbahmani/top-5-data-visualization-libraries-tutorial)
# 7. [Data Cleaning](https://www.kaggle.com/mjbahmani/machine-learning-workflow-for-house-prices)
# 8. <font color="red">You are in the 8th step</font>
# 9. [A Comprehensive ML  Workflow with Python](https://www.kaggle.com/mjbahmani/a-comprehensive-ml-workflow-with-python)
# 10. [Deep Learning](https://www.kaggle.com/mjbahmani/top-5-deep-learning-frameworks-tutorial)
# 
# You can Fork and Run this kernel on **Github**:
# > ###### [ GitHub](https://github.com/mjbahmani/10-steps-to-become-a-data-scientist)
# 
# -------------------------------------------------------------------------------------------------------------
#  **I hope you find this kernel helpful and some <font color="red"><b>UPVOTES</b></font> would be very much appreciated**
#  
#  -----------

#  <a id="1"></a> <br>
# ## 1- Introduction
# **[Elo](https://www.cartaoelo.com.br/)** has defined a competition in **Kaggle**. A realistic and attractive data set for data scientists.
# on this notebook, I will provide a **comprehensive** approach to solve Elo Recommendation problem for **Beginners**.
#  <a id="11"></a> <br>
# ## 1-1 Kaggle kernels
# I have just listed some more important kernels that inspired my work and I've used them in this kernel:
# 1. [simple-python-lightgbm-example](https://www.kaggle.com/ezietsman/simple-python-lightgbm-example)
# 1. [simple-exploration-notebook-elo](https://www.kaggle.com/sudalairajkumar/simple-exploration-notebook-elo)
# 1. [prediction-using-xgboost](https://www.kaggle.com/harshit92/prediction-using-xgboost-3-771)
# 1. [elo-world](https://www.kaggle.com/mjbahmani/elo-world)
# 1. [simple-lightgbm-without-blending](https://www.kaggle.com/mfjwr1/simple-lightgbm-without-blending)
# <br>
# <br>
# I am open to getting your feedback for improving this **kernel**.
# 

# <a id="top"></a> <br>
# ## Notebook  Content
# 1. [Introduction](#1)
#     1. [Kaggle kernels](#11)
# 1. [Data Science Workflow for Elo](#2)
# 1. [Problem Definition](#3)
#     1. [About Elo](#31)
#     1. [Business View](#32)
#         1. [Real world Application Vs Competitions](#321)
# 1. [Problem feature](#4)
#     1. [Aim](#41)
#     1. [Variables](#42)
#     1. [ Inputs & Outputs](#43)
#     1. [Evaluation](#44)
# 1. [Select Framework](#5)
#     1. [Import](#51)
#     1. [Version](#52)
#     1. [Setup](#53)
# 1. [Exploratory data analysis](#6)
#     1. [Data Collection](#61)
#         1. [data_dictionary Analysis](#611)
#         1. [Explorer Dataset](#612)
#     1. [Data Cleaning](#62)
#     1. [Data Visualization](#63)
#         1. [countplot](#631)
#         1. [pie plot](#632)
#         1. [Histogram](#633)
#         1. [violin plot](#634)
#         1. [kdeplot](#635)
#     1. [Data Preprocessing](#64)
# 1. [Apply Learning](#7)
#     1. [Evaluation](#71)
# 1. [Conclusion](#8)
# 1. [References](#9)

# -------------------------------------------------------------------------------------------------------------
# 
#  **I hope you find this kernel helpful and some <font color="red"><b>UPVOTES</b></font> would be very much appreciated**
#  
#  -----------

# <a id="2"></a> <br>
# ## 2- A Data Science Workflow for Elo
# Of course, the same solution can not be provided for all problems, so the best way is to create a **general framework** and adapt it to new problem.
# 
# **You can see my workflow in the below image** :
# 
#  <img src="http://s8.picofile.com/file/8342707700/workflow2.png"  />
# 
# **You should feel free	to	adjust 	this	checklist 	to	your needs**
# ###### [Go to top](#top)

# <a id="3"></a> <br>
# ## 3- Problem Definition
# I think one of the important things when you start a new machine learning project is Defining your problem. that means you should understand business problem.( **Problem Formalization**)
# <img src='http://s8.picofile.com/file/8344103134/Problem_Definition2.png' width=400 height=400>
#  ><font color="red"><b>Note: </b></font>
# We are predicting a **loyalty score** for each card_id represented in test.csv and sample_submission.csv.

# <a id="31"></a> <br>
# ## 3-1 About Elo
#  [Elo](https://www.cartaoelo.com.br/) is one of the largest **payment brands** in Brazil, has built partnerships with merchants in order to offer promotions or discounts to cardholders. But 
# 1. do these promotions work for either the consumer or the merchant?
# 1. Do customers enjoy their experience? 
# 1. Do merchants see repeat business? 
# 
#  ><font color="red"><b>Note: </b></font>
# **Personalization is key**.
# 

# <a id="32"></a> <br>
# ## 3-2 Business View 
# **Elo** has built machine learning models to understand the most important aspects and preferences in their customers’ lifecycle, from food to shopping. But so far none of them is specifically tailored for an individual or profile. This is where you come in.
# 
# ###### [Go to top](#top)

# <a id="4"></a> <br>
# ## 4- Problem Feature
# Problem Definition has four steps that have illustrated in the picture below:
# 
# 
# 1. Aim
# 1. Variable
# 1. Inputs & Outputs
# 1. Evaluation
# <a id="41"></a> <br>
# 
# ### 4-1 Aim
# Develop algorithms to identify and serve the most relevant opportunities to individuals, by uncovering signal in customer loyalty.
# We are predicting a **loyalty score** for each card_id represented in test.csv and sample_submission.csv.
# 
# <a id="42"></a> <br>
# ### 4-2 Variables
# The data is formatted as follows:
# 
# train.csv and test.csv contain card_ids and information about the card itself - the first month the card was active, etc. train.csv also contains the target.
# 
# historical_transactions.csv and new_merchant_transactions.csv are designed to be joined with train.csv, test.csv, and merchants.csv. They contain information about transactions for each card, as described above.
# 
# merchants can be joined with the transaction sets to provide additional merchant-level information.
# 
# 
# <a id="43"></a> <br>
# ### 4-3 Inputs & Outputs
# we use **train.csv** and **test.csv** as Input and we should upload a  **submission.csv** as Output

# <a id="5"></a> <br>
# ## 5- Select Framework
# After problem definition and problem feature, we should select our **framework** to solve the **problem**.
# What we mean by the framework is that  the programming languages you use and by what modules the problem will be solved.
# ###### [Go to top](#top)

# <a id="51"></a> <br>
# ## 5-1 Import

# In[ ]:


from sklearn import model_selection, preprocessing, metrics
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
from pandas import get_dummies
import lightgbm as lgb
import matplotlib as mpl
import xgboost as xgb
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib
import warnings
import sklearn
import scipy
import numpy
import json
import sys
import csv
import os

# <a id="52"></a> <br>
# ## 5-2 version

# In[ ]:


print('matplotlib: {}'.format(matplotlib.__version__))
print('sklearn: {}'.format(sklearn.__version__))
print('scipy: {}'.format(scipy.__version__))
print('seaborn: {}'.format(sns.__version__))
print('pandas: {}'.format(pd.__version__))
print('numpy: {}'.format(np.__version__))
print('Python: {}'.format(sys.version))


# <a id="53"></a> <br>
# ## 5-3 Setup
# 
# A few tiny adjustments for better **code readability**

# In[ ]:


sns.set(style='white', context='notebook', palette='deep')
warnings.filterwarnings('ignore')
warnings.simplefilter(action='ignore', category=FutureWarning)
sns.set_style('white')
pd.set_option('display.max_columns', 500)

# <a id="6"></a> <br>
# ## 6- EDA
# By the end of the section, you'll be able to answer these questions and more, while generating graphics that are both insightful and beautiful.  then We will review analytical and statistical operations:
# 
# 1. Data Collection
# 1. Visualization
# 1. Data Cleaning
# 1. Data Preprocessing
# <img src="http://s9.picofile.com/file/8338476134/EDA.png" width=400 height=400>
# 
#  ###### [Go to top](#top)

# <a id="61"></a> <br>
# ## 6-1 Data Collection
# **Data collection** is the process of gathering and measuring data, information or any variables of interest in a standardized and established manner that enables the collector to answer or test hypothesis and evaluate outcomes of the particular collection.[techopedia]
# 
# I start Collection Data by the training and testing datasets into **Pandas DataFrames**.
# ###### [Go to top](#top)

# In[ ]:


import os
print(os.listdir("../input"))


# In[ ]:


main_train = pd.read_csv('../input/train.csv', parse_dates=["first_active_month"] )
main_test = pd.read_csv('../input/test.csv' ,parse_dates=["first_active_month"] )
main_merchants=pd.read_csv('../input/merchants.csv')
main_new_merchant_transactions=pd.read_csv('../input/new_merchant_transactions.csv')
main_historical_transactions = pd.read_csv("../input/historical_transactions.csv")

# In[ ]:


sample_submission = pd.read_csv("../input/sample_submission.csv")

# In[ ]:


sample_submission.shape

# In[ ]:


sample_submission.head()

# In[ ]:


print(main_train.info())

# In[ ]:


print(main_test.info())

# <a id="612"></a> <br>
# ## 6-1-2 Explorer Dataset
# 1- Dimensions of the dataset.
# 
# 2- Peek at the data itself.
# 
# 3- Statistical summary of all attributes.
# 
# 4- Breakdown of the data by the class variable.
# 
#  ><font color="red"><b>Note: </b></font> Don’t worry, each look at the data is **one command**. These are useful commands that you can use again and again on future projects.
# ###### [Go to top](#top)

#  ><font color="red"><b>Note: </b></font>
#  
# * All **data** is simulated and fictitious, and is not real customer data
# * Each **row** is an observation (also known as : sample, example, instance, record).
# * Each **column** is a feature (also known as: Predictor, attribute, Independent Variable, input, regressor, Covariate).
# ###### [Go to top](#top)

# In[ ]:


print("Shape of train set                 : ",main_train.shape)
print("Shape of test set                  : ",main_test.shape)
print("Shape of historical_transactions   : ",main_historical_transactions.shape)
print("Shape of merchants                 : ",main_merchants.shape)
print("Shape of new_merchant_transactions : ",main_new_merchant_transactions.shape)


# <a id="6121"></a> <br>
# ## 6-1-2-1 data_dictionary Analysis
# Elo Provides a excel file to describe about data(feature). It has four sheet and we have just read them with below code:

# In[ ]:


data_dictionary_train=pd.read_excel('../input/Data_Dictionary.xlsx',sheet_name='train')
data_dictionary_history=pd.read_excel('../input/Data_Dictionary.xlsx',sheet_name='history')
data_dictionary_new_merchant_period=pd.read_excel('../input/Data_Dictionary.xlsx',sheet_name='new_merchant_period')
data_dictionary_merchant=pd.read_excel('../input/Data_Dictionary.xlsx',sheet_name='merchant')

# <a id="613"></a> <br>
# ## 6-1-3 Features
# Features can be from following types:
# * numeric
# * categorical
# * ordinal
# * datetime
# * coordinates
# 
# Find the type of features in **Elo dataset**?!
# 
# For getting some information about the dataset you can use **info()** command.

# <a id="614"></a> <br>
# ## 6-1-4 Train Analysis

# you can use tails command to explorer dataset, such as 

# In[ ]:


main_train.tail()

# <a id="6141"></a> <br>
# ### 6-1-4-1 Train Description
# some info about train set

# In[ ]:


data_dictionary_train.head(10)
# what we know about train:

# We have three features that they are **Anonymized** 

# In[ ]:


main_train.tail()

# After loading the data via **pandas**, we should checkout what the content is, description and via the following:

# In[ ]:


main_train.describe()

# 1. The train set  is approximately twice the test set
# 2. The target data value is between -33.219281 and 17.965068

# ## 6-1-5 Test Analysis

# In[ ]:


print('----- test set--------')
print(main_test.head(5))

# In[ ]:


main_test.info()

# In[ ]:


main_test.describe()

# If you compare **describe()** function for  test and train, you find that they are so similar!

# <a id="615"></a> <br>
# ## 6-1-5 Historical Transactions Analysis

# In[ ]:


data_dictionary_history.head(10)
# what we know about history:

# In[ ]:


main_historical_transactions.head()

# In[ ]:


main_historical_transactions.shape

# <a id="614"></a> <br>
# ## 6-1-4 Merchant Analysis

# In[ ]:


main_merchants.head()

# In[ ]:


data_dictionary_merchant.head(30)
# what we know about merchant:

# ## 6-1-5 New Merchant Transactions Analysis

# In[ ]:


main_new_merchant_transactions.head()

# In[ ]:


data_dictionary_new_merchant_period.head(10)
# what we know about new_merchant_period:

# <a id="62"></a> <br>
# ## 6-2 Data Cleaning
# When dealing with real-world data, dirty data is the norm rather than the exception.
# 
# ###### [Go to top](#top)

# How many NA elements in every column!!
# 
# Good news, it is Zero!
# 
#  ><font color="red"><b>Note: </b></font> To check out how many null info are on the dataset, we can use **isnull().sum()**.

# In[ ]:


def check_missing_data(df):
    flag=df.isna().sum().any()
    if flag==True:
        total = df.isnull().sum()
        percent = (df.isnull().sum())/(df.isnull().count()*100)
        output = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
        data_type = []
        # written by MJ Bahmani
        for col in df.columns:
            dtype = str(df[col].dtype)
            data_type.append(dtype)
        output['Types'] = data_type
        return(np.transpose(output))
    else:
        return(False)

# In[ ]:


print ('for train :',check_missing_data(main_train))
print ('for test:',check_missing_data(main_test))

#  ><font color="red"><b>Note: </b></font> But if we had , we can just use **dropna()**(be careful sometimes you should not do this!)

# In[ ]:


# remove rows that have NA's
print('Before Droping',main_train.shape)
main_train = main_train.dropna()
print('After Droping',main_train.shape)

# 
# We can get a quick idea of how many instances (rows) and how many attributes (columns) the data contains with the shape property.

# To print dataset **columns**, we can use columns atribute.

# In[ ]:


main_train.columns

# <a id="63"></a> <br>
# ## 6-3 Data Visualization
# **Data visualization**  is the presentation of data in a pictorial or graphical format. It enables decision makers to see analytics presented visually, so they can grasp difficult concepts or identify new patterns.
# 
# > * Two** important rules** for Data visualization:
# >     1. Do not put too little information
# >     1. Do not put too much information
# 
# ###### [Go to top](#top)

# <a id="631"></a> <br>
# ## 6-3-1  Histogram

# Most of the targets almost have the value between +8 or -8,please check the plot below. and some of data have value (-30)

# In[ ]:


main_train["target"].hist();

# we should be careful about them!

# In[ ]:


main_train[main_train["target"]<-29].count()

# In[ ]:


# histograms
main_train.hist(figsize=(15,20))
plt.figure()

# In[ ]:


f,ax=plt.subplots(1,2,figsize=(20,10))
main_train[main_train['feature_3']==0].target.plot.hist(ax=ax[0],bins=20,edgecolor='black',color='red')
ax[0].set_title('feature_3= 0')
x1=list(range(0,85,5))
ax[0].set_xticks(x1)
main_train[main_train['feature_3']==1].target.plot.hist(ax=ax[1],color='green',bins=20,edgecolor='black')
ax[1].set_title('feature_3= 1')
x2=list(range(0,85,5))
ax[1].set_xticks(x2)
plt.show()

# In[ ]:


f,ax=plt.subplots(1,2,figsize=(18,8))
main_train['feature_3'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True)
ax[0].set_title('feature_3')
ax[0].set_ylabel('')
sns.countplot('feature_3',data=main_train,ax=ax[1])
ax[1].set_title('feature_3')
plt.show()

# In[ ]:


f,ax=plt.subplots(1,2,figsize=(18,8))
main_train[['feature_3','feature_2']].groupby(['feature_3']).mean().plot.bar(ax=ax[0])
ax[0].set_title('Survived vs feature_2')
sns.countplot('feature_3',hue='feature_2',data=main_train,ax=ax[1])
ax[1].set_title('feature_3:feature')
plt.show()

# <a id="632"></a> <br>
# ## 6-3-2  distplot

# In[ ]:


sns.distplot(main_train['target'])

# <a id="633"></a> <br>
# ## 6-3-3 violinplot

# In[ ]:


sns.violinplot(data=main_train, x="feature_1", y='target')

# <a id="634"></a> <br>
# ## 6-3-4 Scatter plot
# Scatter plot Purpose to identify the type of relationship (if any) between two quantitative variables

# In[ ]:


plt.figure(figsize=(8,6))
plt.scatter(range(main_train.shape[0]), np.sort(main_train['target'].values),marker='o',c='green')
plt.xlabel('index', fontsize=12)
plt.ylabel('Loyalty Score', fontsize=12)
plt.title('Explore: Target')
plt.show();

# In[ ]:


# Modify the graph above by assigning each species an individual color.
g = sns.FacetGrid(main_train, hue="feature_3", col="feature_2", margin_titles=True,
                  palette={1:"blue", 0:"red"} )
g=g.map(plt.scatter, "first_active_month", "target",edgecolor="w").add_legend();

# <a id="635"></a> <br>
# ## 6-3-5 Box
# In descriptive statistics, a box plot or boxplot is a method for graphically depicting groups of numerical data through their quartiles. Box plots may also have lines extending vertically from the boxes (whiskers) indicating variability outside the upper and lower quartiles, hence the terms box-and-whisker plot and box-and-whisker diagram.[wikipedia]

# In[ ]:


sns.boxplot(x="feature_3", y="feature_2", data=main_test )
plt.show()

# <a id="64"></a> <br>
# ## 6-4 Data Preprocessing
# **Data preprocessing** refers to the transformations applied to our data before feeding it to the algorithm.
#  
# Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
# there are plenty of steps for data preprocessing and we just listed some of them in general(Not just for Elo) :
# 1. removing Target column (id)
# 1. Sampling (without replacement)
# 1. Making part of iris unbalanced and balancing (with undersampling and SMOTE)
# 1. Introducing missing values and treating them (replacing by average values)
# 1. Noise filtering
# 1. Data discretization
# 1. Normalization and standardization
# 1. PCA analysis
# 1. Feature selection (filter, embedded, wrapper)
# 1. Etc.
# 
# >What methods of preprocessing can we run on  Elo?! 
# ###### [Go to top](#top)

# **<< Note 2 >>**
# in pandas's data frame you can perform some query such as "where"

# In[ ]:


main_train.where(main_train ['target']==1).count()

# As you can see in the below in python, it is so easy perform some query on the dataframe:

# In[ ]:


main_train[main_train['target']<-32].head(5)

# In[ ]:


main_train[main_train['target']==1].head(5)

# In[ ]:


main_train.feature_1.unique()

# In[ ]:


main_train.feature_2.unique()

# In[ ]:


main_train.feature_3.unique()

# In[ ]:


main_train.first_active_month.unique()

# **<< Note >>**
# >**Preprocessing and generation pipelines depend on a model type**

# <a id="641"></a> <br>
# ## 6-4-1Some New Feature

# In[ ]:


df_train=main_train
df_test=main_test

# In[ ]:


df_train["year"] = main_train["first_active_month"].dt.year
df_test["year"] = main_test["first_active_month"].dt.year

# In[ ]:


df_train["month"] = main_train["first_active_month"].dt.month
df_test["month"] = main_test["first_active_month"].dt.month

# <a id="642"></a> <br>
# ## 6-4-2 Feature Encoding
# In machine learning projects, one important part is feature engineering. It is very common to see categorical features in a dataset. However, our machine learning algorithm can only read numerical values. It is essential to encoding categorical features into numerical values.[3]

# In[ ]:


x_train = df_train.drop(["target","card_id","first_active_month"],axis=1)
x_test = df_test.drop(["card_id","first_active_month"],axis=1)

# In[ ]:


y_train = df_train["target"]
df_train = df_train.sample(frac=1, random_state = 7)

# In[ ]:



Trn_x,val_x,Trn_y,val_y = train_test_split(x_train,y_train,test_size =0.1,random_state = 7)
trn_x , test_x, trn_y, test_y = train_test_split(Trn_x , Trn_y, test_size =0.1, random_state = 7)

# <a id="7"></a> <br>
# ## 7- Apply Learning
# How to understand what is the best way to solve our problem?!
# 
# The answer is always "**It depends**." It depends on the **size**, **quality**, and **nature** of the **data**. It depends on what you want to do with the answer. It depends on how the **math** of the algorithm was translated into instructions for the computer you are using. And it depends on how much **time** you have. Even the most **experienced data scientists** can't tell which algorithm will perform best before trying them.(see a nice [cheatsheet](https://github.com/mjbahmani/10-steps-to-become-a-data-scientist/blob/master/cheatsheets/microsoft-machine-learning-algorithm-cheat-sheet-v7.pdf) for this section)
# Categorize the problem
# The next step is to categorize the problem. This is a two-step process.
# 
# 1. **Categorize by input**:
#     1. If you have labelled data, it’s a supervised learning problem.
#     1. If you have unlabelled data and want to find structure, it’s an unsupervised learning problem.
#     1. If you want to optimize an objective function by interacting with an environment, it’s a reinforcement learning problem.
# 1. **Categorize by output**.
#     1. If the output of your model is a number, it’s a regression problem.
#     1. If the output of your model is a class, it’s a classification problem.
#     1. If the output of your model is a set of input groups, it’s a clustering problem.
#     1. Do you want to detect an anomaly ? That’s anomaly detection
# 1. **Understand your constraints**
#     1. What is your data storage capacity? Depending on the storage capacity of your system, you might not be able to store gigabytes of classification/regression models or gigabytes of data to clusterize. This is the case, for instance, for embedded systems.
#     1. Does the prediction have to be fast? In real time applications, it is obviously very important to have a prediction as fast as possible. For instance, in autonomous driving, it’s important that the classification of road signs be as fast as possible to avoid accidents.
#     1. Does the learning have to be fast? In some circumstances, training models quickly is necessary: sometimes, you need to rapidly update, on the fly, your model with a different dataset.
# 1. **Find the available algorithms**
#     1. Now that you a clear understanding of where you stand, you can identify the algorithms that are applicable and practical to implement using the tools at your disposal. Some of the factors affecting the choice of a model are:
# 
#     1. Whether the model meets the business goals
#     1. How much pre processing the model needs
#     1. How accurate the model is
#     1. How explainable the model is
#     1. How fast the model is: How long does it take to build a model, and how long does the model take to make predictions.
#     1. How scalable the model is
# 

# <a id="71"></a> <br>
# ## 7-1 Evaluation
# Submissions are scored on the root mean squared error. RMSE(Root Mean Squared Error) is defined as:
# <img src='https://www.includehelp.com/ml-ai/Images/rmse-1.jpg'>
# where y^ is the predicted loyalty score for each card_id, and y is the actual loyalty score assigned to a card_id.
# 
#  ><font color="red"><b>Note: </b></font>
#  You must answer the following question:
# How does your company expect to use and benefit from **your model**.
# ###### [Go to top](#top)

# In[ ]:


# rmse
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# In[ ]:


# converting into xgb DMatrix
Train = xgb.DMatrix(trn_x,label = trn_y)
Validation = xgb.DMatrix(val_x, label = val_y)
Test = xgb.DMatrix(test_x)

# In[ ]:


params = {"booster":"gbtree","eta":0.1,'min_split_loss':0,'max_depth':6,
         'min_child_weight':1, 'max_delta_step':0,'subsample':1,'colsample_bytree':1,
         'colsample_bylevel':1,'reg_lambda':1,'reg_alpha':0,
         'grow_policy':'depthwise','max_leaves':0,'objective':'reg:linear','eval_metric':'rmse',
         'seed':7}
history ={}  # This will record rmse score of training and test set
eval_list =[(Train,"Training"),(Validation,"Validation")]

# In[ ]:


clf = xgb.train(params, Train, num_boost_round=119, evals=eval_list, obj=None, feval=None, maximize=False, 
          early_stopping_rounds=40, evals_result=history);

# In[ ]:


prediction = clf.predict(xgb.DMatrix(x_test))

# In[ ]:


submission = pd.DataFrame({
        "card_id": main_test["card_id"].values,
        "target": np.ravel(prediction)
    })

# <a id="8"></a> <br>
# # 8- Conclusion
# This kernel is not completed yet , I have tried to cover all the parts related to the process of **Elo  problem** with a variety of Python packages and I know that there are still some issues then I hope to get your feedback to improve it.
# 

# <a id="9"></a> <br>
# # 9- References & Credits
# 1. [hackernoon](https://hackernoon.com/choosing-the-right-machine-learning-algorithm-68126944ce1f)
# 1. [medium](https://medium.com/@pushkarmandot/https-medium-com-pushkarmandot-what-is-lightgbm-how-to-implement-it-how-to-fine-tune-the-parameters-60347819b7fc)
# 1. [encoding-categorical-features](https://towardsdatascience.com/encoding-categorical-features-21a2651a065c)

# you can Fork and Run this kernel on **Github**:
# > ###### [ GitHub](https://github.com/mjbahmani/10-steps-to-become-a-data-scientist)
# 

# Go to first step: [**Course Home Page**](https://www.kaggle.com/mjbahmani/10-steps-to-become-a-data-scientist)
# 
# Go to next step : [**Mathematics and Linear Algebra**](https://www.kaggle.com/mjbahmani/linear-algebra-for-data-scientists)
# 

# ### The kernel is not completed and will be updated soon  !!!
