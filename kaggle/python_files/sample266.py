#!/usr/bin/env python
# coding: utf-8

#  #  <div style="text-align: center">Probability of Earthquake: EDA, FE, +5 Models </div> 
# <img src='http://s8.picofile.com/file/8355280718/pro.png' width=400 height=400>
# <div style="text-align:center"> last update: <b>26/03/2019</b></div>
# 
# 
# 
# You can Fork code  and  Follow me on:
# > ###### [ GitHub](https://github.com/mjbahmani/10-steps-to-become-a-data-scientist)
# -------------------------------------------------------------------------------------------------------------
#  <b> I hope you find this kernel helpful and some <font color='red'>UPVOTES</font> would be very much appreciated.</b>
#     
#  -----------

#  <a id="top"></a> <br>
# ## Notebook  Content
# 1. [Introduction](#1)
# 1. [Load packages](#2)
#     1. [import](21)
#     1. [Setup](22)
#     1. [Version](23)
# 1. [Problem Definition](#3)
#     1. [Problem Feature](#31)
#     1. [Aim](#32)
#     1. [Variables](#33)
#     1. [Evaluation](#34)
# 1. [Exploratory Data Analysis(EDA)](#4)
#     1. [Data Collection](#41)
#     1. [Visualization](#42)
#         1. [Hist](#421)
#         1. [Time to failure histogram](#422)
#         1. [Distplot](#423)
#         1. [kdeplot](#424)
#     1. [Data Preprocessing](#43)
#         1. [Create new feature](#431)
#     1. [ML Explainability](#44)
#         1. [Permutation Importance](#441)
#         1. [Partial Dependence Plots](#442)
#         1. [SHAP Values](#443)
#         1. [Pdpbox](#444)
# 1. [Model Development](#5)
#     1. [SVM](#51)
#     1. [LGBM](#52)
#     1. [Catboost](#53)
# 1. [Submission](#6)
#     1. [Blending](#61)
# 1. [References](#7)
#     1. [References](#71)
# 

#  <a id="1"></a> <br>
# ## 1- Introduction
# **Forecasting earthquakes** is one of the most important problems in **Earth science**. If you agree, the earthquake forecast is likely to be related to the concepts of **probability**. In this kernel, I try to look at the prediction of the earthquake with the **help** of the concepts of **probability** .
# <img src='https://www.preventionweb.net/files/52472_largeImage.jpg' width=600 height=600 >
# For anyone taking first steps in data science, **Probability** is a must know concept. Concepts of probability theory are the backbone of many important concepts in data science like inferential statistics to Bayesian networks. It would not be wrong to say that the journey of mastering statistics begins with **probability**.
# 
# Before starting, I have to point out that I used the following great kernel:
# [https://www.kaggle.com/inversion/basic-feature-benchmark](https://www.kaggle.com/inversion/basic-feature-benchmark)

#  <a id="2"></a> <br>
#  ## 2- Load packages
#   <a id="21"></a> <br>
# ## 2-1 Import

# In[ ]:


from sklearn.model_selection import StratifiedKFold, KFold, RepeatedKFold
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from eli5.sklearn import PermutationImportance
from sklearn.tree import DecisionTreeRegressor
from catboost import CatBoostRegressor,Pool
import matplotlib.patches as patch
from scipy.stats import kurtosis
import matplotlib.pyplot as plt
from sklearn.svm import NuSVR
from scipy.stats import skew
from scipy.stats import norm
from scipy import linalg
from sklearn import tree
from sklearn import svm
import lightgbm as lgb
import xgboost as xgb
from tqdm import tqdm
import seaborn as sns
import pandas as pd
import numpy as np
import graphviz
import warnings
import random
import eli5
import shap  # package used to calculate Shap values
import time
import glob
import sys
import os

#  <a id="22"></a> <br>
# ##  2-2 Setup

# In[ ]:


warnings.filterwarnings('ignore')
plt.style.use('ggplot')
np.set_printoptions(suppress=True)
pd.set_option("display.precision", 15)

#  <a id="23"></a> <br>
# ## 2-3 Version
# 

# In[ ]:


print('pandas: {}'.format(pd.__version__))
print('numpy: {}'.format(np.__version__))
print('Python: {}'.format(sys.version))

# <a id="3"></a> 
# <br>
# ## 3- Problem Definition
# I think one of the important things when you start a new machine learning project is Defining your problem. that means you should understand business problem.( **Problem Formalization**)
# 
# Problem Definition has four steps that have illustrated in the picture below:
# <img src="http://s8.picofile.com/file/8344103134/Problem_Definition2.png" width=400 height=400>
# 
# > <font color='red'>Note</font> : **Current scientific studies related to earthquake forecasting focus on three key points:** 
# 1. when the event will occur
# 1. where it will occur
# 1. how large it will be.
# 

# <a id="31"></a> 
# ### 3-1 Problem Feature
# 
# 1.     Train.csv - A single, continuous training segment of experimental data.
# 1.     Test - A folder containing many small segments of test data.
# 1.     Slample_sumbission.csv - A sample submission file in the correct format.
# 

# <a id="32"></a> 
# ### 3-2 Aim
# In this competition, you will address <font color='red'><b>WHEN</b></font> the earthquake will take place.

# <a id="33"></a> 
# ### 3-3 Variables
# 
# 1.     **acoustic_data** - the seismic signal [int16]
# 1.     **time_to_failure** - the time (in seconds) until the next laboratory earthquake [float64]
# 1.     **seg_id**- the test segment ids for which predictions should be made (one prediction per segment)
# 

# <a id="34"></a> 
# ### 3-4 Evaluation
# Submissions are evaluated using the [**mean absolute error**](https://en.wikipedia.org/wiki/Mean_absolute_error) between the predicted time remaining before the next lab earthquake and the act remaining time.
# <img src='https://wikimedia.org/api/rest_v1/media/math/render/svg/3ef87b78a9af65e308cf4aa9acf6f203efbdeded'>

# <a id="4"></a> 
# ## 4- Exploratory Data Analysis(EDA)
#  In this section, we'll analysis how to use graphical and numerical techniques to begin uncovering the structure of your data. 
#  
# 1. Which variables suggest interesting relationships?
# 1. Which observations are unusual?
# 1. Analysis of the features!
# By the end of the section, you'll be able to answer these questions and more, while generating graphics that are both insightful and beautiful.  then We will review analytical and statistical operations:
# 
# 1. Data Collection
# 1. Visualization
# 1. Data Preprocessing
# 1. Data Cleaning

#  <a id="41"></a> <br>
# ## 4-1 Data Collection

# What we have in input!

# In[ ]:


print(os.listdir("../input/"))

# In[ ]:


train = pd.read_csv('../input/train.csv' , dtype={'acoustic_data': np.int16, 'time_to_failure': np.float32})

# In[ ]:


print("Train has: rows:{} cols:{}".format(train.shape[0], train.shape[1]))

# What we have submit!

# In[ ]:


submission = pd.read_csv('../input/sample_submission.csv', index_col='seg_id')
submission.head()

# In[ ]:


print("submission has: rows:{} cols:{}".format(submission.shape[0], submission.shape[1]))

# ### There are 2624 files in test.zip.

# In[ ]:


len(os.listdir(os.path.join('../input/', 'test')))

# Also we have **2624**  row same as number of test files in submission , so this clear that we should predict **time_to_failure** for all of test files.

# ## Memory usage: 3.5 GB

# In[ ]:


train.info()

# In[ ]:


train.shape

# ### Wow! so large(rows:629145480 columns:2) for playing with it, let's select just 150000 rows!

# In[ ]:


train.head()

#  <a id="42"></a> <br>
# ## 4-2 Visualization
# Because the size of the database is very large, for the visualization section, we only need to select a small subset of the data.

# In[ ]:


# we change the size of Dataset due to play with it (just loaded %0001)
mini_train= pd.read_csv("../input/train.csv",nrows=150000)

# In[ ]:


mini_train.describe()

# In[ ]:


mini_train.shape

# In[ ]:


mini_train.isna().sum()

# In[ ]:


type(mini_train)

# <a id="421"></a> 
# ### 4-2-1 Hist

# In[ ]:


#acoustic_data means signal
mini_train["acoustic_data"].hist();

# <a id="422"></a> 
# ### 4-2-2 Time to failure histogram

# In[ ]:


plt.plot(mini_train["time_to_failure"], mini_train["acoustic_data"])
plt.title("time_to_failure histogram")

# <a id="423"></a> 
# ### 4-2-3 Distplot

# In[ ]:


sns.distplot(mini_train.acoustic_data.values, color="Red", bins=100, kde=False)

# <a id="424"></a> 
# ### 4-2-4 kdeplot

# In[ ]:


sns.kdeplot(mini_train["acoustic_data"] )

#  <a id="43"></a> <br>
# ## 4-3 Data Preprocessing
# 

# Because we have only one feature(**acoustic_data**), and the size of the training set is very large( more that 60000000 rows), it is a good idea to reduce the size of the training set with making new segment  and also to increase the number of attributes by using **statistical attributes**.

# In[ ]:


# based on : https://www.kaggle.com/inversion/basic-feature-benchmark
rows = 150_000
segments = int(np.floor(train.shape[0] / rows))
segments

# In[ ]:


X_train = pd.DataFrame(index=range(segments), dtype=np.float64,
                       columns=['ave', 'std', 'max', 'min','sum','skew','kurt'])
y_train = pd.DataFrame(index=range(segments), dtype=np.float64,
                       columns=['time_to_failure'])

# ### y_train is our target for prediction

# In[ ]:


y_train.head()

# ### our train set with 4 new feature

# In[ ]:


X_train.head()

#  <a id="431"></a> <br>
# ### 4-3-1 Create New Features

# > <font color='red'>Note:</font>  
# **tqdm** means "progress" in Arabic (taqadum, تقدّم) and is an abbreviation for "I love you so much" in Spanish (te quiero demasiado). Instantly make your loops show a smart progress meter - just wrap any iterable with tqdm(iterable), and you're done! [https://tqdm.github.io/](https://tqdm.github.io/)

# In[ ]:


for segment in tqdm(range(segments)):
    seg = train.iloc[segment*rows:segment*rows+rows]
    x = seg['acoustic_data'].values
    y = seg['time_to_failure'].values[-1]
    
    y_train.loc[segment, 'time_to_failure'] = y
    X_train.loc[segment, 'ave'] = x.mean()
    X_train.loc[segment, 'std'] = x.std()
    X_train.loc[segment, 'max'] = x.max()
    X_train.loc[segment, 'min'] = x.min()
    X_train.loc[segment, 'sum'] = x.sum()
    X_train.loc[segment, 'skew'] =skew(x)
    X_train.loc[segment, 'kurt'] = kurtosis(x)

# In[ ]:


X_train.head()

# In[ ]:


y_train.head()

# ### Cheking missing Data

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


check_missing_data(X_train)

# ### Now we must create our X_test for submission

# In[ ]:


X_test = pd.DataFrame(columns=X_train.columns, dtype=np.float64, index=submission.index)

# In[ ]:


X_test.head()

# In[ ]:


for seg_id in  tqdm(X_test.index):
    seg = pd.read_csv('../input/test/' + seg_id + '.csv')
    
    x = seg['acoustic_data'].values
    
    X_test.loc[seg_id, 'ave'] = x.mean()
    X_test.loc[seg_id, 'std'] = x.std()
    X_test.loc[seg_id, 'max'] = x.max()
    X_test.loc[seg_id, 'min'] = x.min()
    X_test.loc[seg_id, 'sum'] = x.sum()
    X_test.loc[seg_id, 'skew'] =skew(x)
    X_test.loc[seg_id, 'kurt'] = kurtosis(x)

# In[ ]:


X_test.shape

# Now we have all of the data frames for applying ML algorithms. just adding some feature scaling.

# In[ ]:


scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)

# In[ ]:


X_test_scaled = scaler.transform(X_test)

# In[ ]:


X=X_train.copy()
y=y_train.copy()

#  <a id="44"></a> <br>
# ## 4-4 ML Explainability
# In this section, I want to try extract insights from models with the help of this excellent [Course](https://www.kaggle.com/learn/machine-learning-explainability) in Kaggle. The Goal behind of ML Explainability for Earthquake is:
# 
# 1. Extract insights from models.
# 1. Find the most inmortant feature in models.
# 1. Affect of each feature on the model's predictions.

#  <a id="441"></a> <br>
# ### 4-4-1 Permutation Importance
# In this section we will answer following question:
# 
# What features have the biggest impact on predictions?
# 
# how to extract insights from models?

# In[ ]:


train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
rfc_model = RandomForestRegressor(random_state=0).fit(train_X, train_y)

# Here is how to calculate and show importances with the [eli5](https://eli5.readthedocs.io/en/latest/) library:

# In[ ]:


perm = PermutationImportance(rfc_model, random_state=1).fit(val_X, val_y)

# In[ ]:


eli5.show_weights(perm, feature_names = val_X.columns.tolist(), top=7)

#  <a id="442"></a> <br>
# ### 4-4-2 Partial Dependence Plots
# While feature importance shows what variables most affect predictions, partial dependence plots show how a feature affects predictions. and partial dependence plots are calculated after a model has been fit. [partial-plots](https://www.kaggle.com/dansbecker/partial-plots)

# In[ ]:


train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
tree_model = DecisionTreeRegressor(random_state=0, max_depth=5, min_samples_split=5).fit(train_X, train_y)

# For the Partial of explanation, I use a Decision Tree which you can see below.

# In[ ]:


tree_graph = tree.export_graphviz(tree_model, out_file=None, feature_names=X.columns)

# In[ ]:


graphviz.Source(tree_graph)

#  <a id="443"></a> <br>
# ### 4-4-3 SHAP Values
# SHAP (SHapley Additive exPlanations) is a unified approach to explain the output of any machine learning model. SHAP connects game theory with local explanations, uniting several previous methods [1-7] and representing the only possible consistent and locally accurate additive feature attribution method based on expectations (see the SHAP NIPS paper for details).

# In[ ]:


row_to_show = 5
data_for_prediction = val_X.iloc[row_to_show]  # use 1 row of data here. Could use multiple rows if desired
data_for_prediction_array = data_for_prediction.values.reshape(1, -1)


tree_model.predict(data_for_prediction_array);

# In[ ]:


# Create object that can calculate shap values
explainer = shap.TreeExplainer(tree_model)

# Calculate Shap values
shap_values = explainer.shap_values(data_for_prediction)

# >**Note**: Shap can answer to this qeustion : how the model works for an individual prediction?

# In[ ]:


shap.initjs()
shap.force_plot(explainer.expected_value, shap_values, data_for_prediction)

#  <a id="444"></a> <br>
# ### 4-4-4 Pdpbox
# In this section, we see the impact of the main variables discovered in the previous sections by using the [pdpbox](https://pdpbox.readthedocs.io/en/latest/).

# In[ ]:


from matplotlib import pyplot as plt
from pdpbox import pdp, get_dataset, info_plots

# Create the data that we will plot
pdp_goals = pdp.pdp_isolate(model=tree_model, dataset=val_X, model_features=X.columns, feature='std')

# plot it
pdp.pdp_plot(pdp_goals, 'std')
plt.show()

# In[ ]:


# Create the data that we will plot
pdp_goals = pdp.pdp_isolate(model=rfc_model, dataset=val_X, model_features=X.columns, feature='kurt')

# plot it
pdp.pdp_plot(pdp_goals, 'kurt')
plt.show()

# In[ ]:


# Create the data that we will plot
pdp_goals = pdp.pdp_isolate(model=rfc_model, dataset=val_X, model_features=X.columns, feature='max')

# plot it
pdp.pdp_plot(pdp_goals, 'max')
plt.show()

# In[ ]:


# Create the data that we will plot
pdp_goals = pdp.pdp_isolate(model=rfc_model, dataset=val_X, model_features=X.columns, feature='min')

# plot it
pdp.pdp_plot(pdp_goals, 'min')
plt.show()

# <a id="5"></a> <br>
# # 5- Model Development
# 1. Svm
# 1. LGBM
# 1. Catboost
# 1. DT
# 1. Randomforest

# <a id="51"></a> <br>
# # 5-1 SVM

# In[ ]:


svm = NuSVR()
svm.fit(X_train_scaled, y_train.values.flatten())
y_pred_svm = svm.predict(X_train_scaled)

# In[ ]:


score = mean_absolute_error(y_train.values.flatten(), y_pred_svm)
print(f'Score: {score:0.3f}')

# <a id="52"></a> <br>
# # 5-2 LGBM

#  Defing folds for cross-validation

# In[ ]:


folds = KFold(n_splits=5, shuffle=True, random_state=42)

# LGBM params

# In[ ]:


params = {'objective' : "regression", 
               'boosting':"gbdt",
               'metric':"mae",
               'boost_from_average':"false",
               'num_threads':8,
               'learning_rate' : 0.001,
               'num_leaves' : 52,
               'max_depth':-1,
               'tree_learner' : "serial",
               'feature_fraction' : 0.85,
               'bagging_freq' : 1,
               'bagging_fraction' : 0.85,
               'min_data_in_leaf' : 10,
               'min_sum_hessian_in_leaf' : 10.0,
               'verbosity' : -1}

# In[ ]:


y_pred_lgb = np.zeros(len(X_test_scaled))
for fold_n, (train_index, valid_index) in tqdm(enumerate(folds.split(X))):
    print('Fold', fold_n, 'started at', time.ctime())
    X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
    y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]
        
    model = lgb.LGBMRegressor(**params, n_estimators = 22000, n_jobs = -1)
    model.fit(X_train, y_train, 
                    eval_set=[(X_train, y_train), (X_valid, y_valid)], eval_metric='mae',
                    verbose=1000, early_stopping_rounds=200)
            
    y_pred_valid = model.predict(X_valid)
    y_pred_lgb += model.predict(X_test_scaled, num_iteration=model.best_iteration_) / folds.n_splits

# <a id="53"></a> <br>
# # 5-3 Catboost 

# In[ ]:


train_pool = Pool(X,y)
cat_model = CatBoostRegressor(
                               iterations=3000,# change 25 to 3000 to get best performance 
                               learning_rate=0.03,
                               eval_metric='MAE',
                              )
cat_model.fit(X,y,silent=True)
y_pred_cat = cat_model.predict(X_test)

# <a id="6"></a> <br>
# # 6- Submission

# ### submission for svm

# In[ ]:


y_pred_svm= svm.predict(X_test_scaled)

# In[ ]:


submission['time_to_failure'] = y_pred_cat
submission.to_csv('submission_svm.csv')

# ### Submission for LGBM

# In[ ]:


submission['time_to_failure'] = y_pred_lgb
submission.to_csv('submission_lgb.csv')

# ### Submission for Catboost

# In[ ]:


submission['time_to_failure'] = y_pred_cat
submission.to_csv('submission_cat.csv')

# ### Submission for Randomforest

# In[ ]:


y_pred_rf=rfc_model.predict(X_test_scaled)

# In[ ]:



submission['time_to_failure'] = y_pred_rf
submission.to_csv('submission_rf.csv')

# <a id="61"></a> <br>
# # 6-1 Blending

# In[ ]:


blending = y_pred_svm*0.5 + y_pred_lgb*0.5 
submission['time_to_failure']=blending
submission.to_csv('submission_lgb_svm.csv')

# In[ ]:


blending = y_pred_svm*0.5 + y_pred_cat*0.5 
submission['time_to_failure']=blending
submission.to_csv('submission_cat_svm.csv')

# you can follow me on:
# > ###### [ GitHub](https://github.com/mjbahmani/)
# > ###### [Kaggle](https://www.kaggle.com/mjbahmani/)
# 
#  <b>I hope you find this kernel helpful and some <font color='red'>UPVOTES</font> would be very much appreciated.<b/>
#  

# <a id="7"></a> <br>
# # 7-References
# 1. [Basic Probability Data Science with examples](https://www.analyticsvidhya.com/blog/2017/02/basic-probability-data-science-with-examples/)
# 1. [How to self learn statistics of data science](https://medium.com/ml-research-lab/how-to-self-learn-statistics-of-data-science-c05db1f7cfc3)
# 1. [Probability statistics for data science- series](https://towardsdatascience.com/probability-statistics-for-data-science-series-83b94353ca48)
# 1. [basic-statistics-in-python-probability](https://www.dataquest.io/blog/basic-statistics-in-python-probability/)
# 1. [tutorialspoint](https://www.tutorialspoint.com/python/python_poisson_distribution.htm)
# 
# ## 7-1 Kaggle Kernels
# In the end , I want to thank all the kernels I've used in this notebook
# 1. [https://www.kaggle.com/inversion/basic-feature-benchmark](https://www.kaggle.com/inversion/basic-feature-benchmark)
# 1. [https://www.kaggle.com/dansbecker/permutation-importance](https://www.kaggle.com/dansbecker/permutation-importance)

# Go to first step: [Course Home Page](https://www.kaggle.com/mjbahmani/10-steps-to-become-a-data-scientist)
# 
# Go to next step : [Santander-ml-explainability](https://www.kaggle.com/mjbahmani/santander-ml-explainability)
# 
