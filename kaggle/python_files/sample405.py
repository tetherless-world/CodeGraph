#!/usr/bin/env python
# coding: utf-8

#  #  <div style="text-align: center">  Santander ML Explainability  </div> 
# ###  <div style="text-align: center">CLEAR DATA. MADE MODEL. </div> 
# <img src='https://galeria.bankier.pl/p/b/5/215103d7ace468-645-387-261-168-1786-1072.jpg' width=600 height=600>
# <div style="text-align:center"> last update: <b> 10/03/2019</b></div>
# 
# 
# 
# You can Fork code  and  Follow me on:
# 
# > ###### [ GitHub](https://github.com/mjbahmani/10-steps-to-become-a-data-scientist)
# > ###### [Kaggle](https://www.kaggle.com/mjbahmani/)
# -------------------------------------------------------------------------------------------------------------
#  <b>I hope you find this kernel helpful and some <font color='red'>UPVOTES</font> would be very much appreciated.</b>
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
#     1. [Data Preprocessing](#43)
# 1. [Machine Learning Explainability for Santander](#5)
#     1. [Permutation Importance](#51)
#     1. [How to calculate and show importances?](#52)
#     1. [What can be inferred from the above?](#53)
#     1. [Partial Dependence Plots](#54)
# 1. [Model Development](#6)
#     1. [lightgbm](#61)
#     1. [RandomForestClassifier](#62)
#     1. [DecisionTreeClassifier](#63)
#     1. [CatBoostClassifier](#64)
#     1. [Funny Combine](#65)
# 1. [References](#7)

#  <a id="1"></a> <br>
# ## 1- Introduction
# At [Santander](https://www.santanderbank.com) their mission is to help people and businesses prosper. they are always looking for ways to help our customers understand their financial health and identify which products and services might help them achieve their monetary goals.
# <img src='https://www.smava.de/kredit/wp-content/uploads/2015/12/santander-bank.png' width=400 height=400>
# 
# In this kernel we are going to create a **Machine Learning Explainability** for **Santander** based this perfect [course](https://www.kaggle.com/learn/machine-learning-explainability) in kaggle.
# ><font color="red"><b>Note: </b></font>
# how to extract **insights** from models?

# <a id="2"></a> <br>
# ## 2- A Data Science Workflow for Santander 
# Of course, the same solution can not be provided for all problems, so the best way is to create a **general framework** and adapt it to new problem.
# 
# **You can see my workflow in the below image** :
# 
#  <img src="http://s8.picofile.com/file/8342707700/workflow2.png"  />
# 
# **You should feel free	to	adjust 	this	checklist 	to	your needs**
# ###### [Go to top](#top)

#  <a id="2"></a> <br>
#  ## 2- Load packages
#   <a id="21"></a> <br>
# ## 2-1 Import

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from catboost import CatBoostClassifier,Pool
from IPython.display import display
import matplotlib.patches as patch
import matplotlib.pyplot as plt
from sklearn.svm import NuSVR
from scipy.stats import norm
from sklearn import svm
import lightgbm as lgb
import xgboost as xgb
import seaborn as sns
import pandas as pd
import numpy as np
import warnings
import time
import glob
import sys
import os
import gc

#  <a id="22"></a> <br>
# ##  2-2 Setup

# In[ ]:


# for get better result chage fold_n to 5
fold_n=5
folds = StratifiedKFold(n_splits=fold_n, shuffle=True, random_state=10)
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
# In this **challenge**, we should help this **bank**  identify which **customers** will make a **specific transaction** in the future, irrespective of the amount of money transacted. The data provided for this competition has the same structure as the real data we have available to solve this **problem**.
# 

# <a id="31"></a> 
# ### 3-1 Problem Feature
# 
# 1. train.csv - the training set.
# 1. test.csv - the test set. The test set contains some rows which are not included in scoring.
# 1. sample_submission.csv - a sample submission file in the correct format.
# 

# <a id="32"></a> 
# ### 3-2 Aim
# In this competition, The task is to predict the value of **target** column in the test set.

# <a id="33"></a> 
# ### 3-3 Variables
# 
# We are provided with an **anonymized dataset containing numeric feature variables**, the binary **target** column, and a string **ID_code** column.
# 
# The task is to predict the value of **target column** in the test set.

# <a id="34"></a> 
# ## 3-4 evaluation
# **Submissions** are evaluated on area under the [ROC curve](http://en.wikipedia.org/wiki/Receiver_operating_characteristic) between the predicted probability and the observed target.
# <img src='https://upload.wikimedia.org/wikipedia/commons/6/6b/Roccurves.png' width=300 height=300>

# In[ ]:


from sklearn.metrics import roc_auc_score, roc_curve

# <a id="4"></a> 
# ## 4- Exploratory Data Analysis(EDA)
#  In this section, we'll analysis how to use graphical and numerical techniques to begin uncovering the structure of your data. 
# *  Data Collection
# *  Visualization
# *  Data Preprocessing
# *  Data Cleaning
# <img src="http://s9.picofile.com/file/8338476134/EDA.png" width=400 height=400>

#  <a id="41"></a> <br>
# ## 4-1 Data Collection

# In[ ]:


print(os.listdir("../input/"))

# In[ ]:


# import Dataset to play with it
train= pd.read_csv("../input/train.csv")
test = pd.read_csv('../input/test.csv')

# In[ ]:


sample_submission = pd.read_csv('../input/sample_submission.csv')
sample_submission.head()

# In[ ]:


train.shape, test.shape, sample_submission.shape

# In[ ]:


train.head(5)

# # Reducing  memory size by ~50%
# Because we make a lot of calculations in this kernel, we'd better reduce the size of the data.
# 1. 300 MB before Reducing
# 1. 150 MB after Reducing

# In[ ]:


#Based on this great kernel https://www.kaggle.com/arjanso/reducing-dataframe-memory-size-by-65
def reduce_mem_usage(df):
    start_mem_usg = df.memory_usage().sum() / 1024**2 
    print("Memory usage of properties dataframe is :",start_mem_usg," MB")
    NAlist = [] # Keeps track of columns that have missing values filled in. 
    for col in df.columns:
        if df[col].dtype != object:  # Exclude strings
            
            # Print current column type
            print("******************************")
            print("Column: ",col)
            print("dtype before: ",df[col].dtype)
            
            # make variables for Int, max and min
            IsInt = False
            mx = df[col].max()
            mn = df[col].min()
            
            # Integer does not support NA, therefore, NA needs to be filled
            if not np.isfinite(df[col]).all(): 
                NAlist.append(col)
                df[col].fillna(mn-1,inplace=True)  
                   
            # test if column can be converted to an integer
            asint = df[col].fillna(0).astype(np.int64)
            result = (df[col] - asint)
            result = result.sum()
            if result > -0.01 and result < 0.01:
                IsInt = True

            
            # Make Integer/unsigned Integer datatypes
            if IsInt:
                if mn >= 0:
                    if mx < 255:
                        df[col] = df[col].astype(np.uint8)
                    elif mx < 65535:
                        df[col] = df[col].astype(np.uint16)
                    elif mx < 4294967295:
                        df[col] = df[col].astype(np.uint32)
                    else:
                        df[col] = df[col].astype(np.uint64)
                else:
                    if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)    
            
            # Make float datatypes 32 bit
            else:
                df[col] = df[col].astype(np.float32)
            
            # Print new column type
            print("dtype after: ",df[col].dtype)
            print("******************************")
    
    # Print final result
    print("___MEMORY USAGE AFTER COMPLETION:___")
    mem_usg = df.memory_usage().sum() / 1024**2 
    print("Memory usage is: ",mem_usg," MB")
    print("This is ",100*mem_usg/start_mem_usg,"% of the initial size")
    return df, NAlist

# Reducing for train data set

# In[ ]:


train, NAlist = reduce_mem_usage(train)
print("_________________")
print("")
print("Warning: the following columns have missing values filled with 'df['column_name'].min() -1': ")
print("_________________")
print("")
print(NAlist)

# Reducing for test data set

# In[ ]:


test, NAlist = reduce_mem_usage(test)
print("_________________")
print("")
print("Warning: the following columns have missing values filled with 'df['column_name'].min() -1': ")
print("_________________")
print("")
print(NAlist)

#  <a id="41"></a> <br>
# ##   4-1-1Data set fields

# In[ ]:


train.columns

# In[ ]:


print(len(train.columns))

# In[ ]:


print(train.info())

#  <a id="422"></a> <br>
# ## 4-2-2 numerical values Describe

# In[ ]:


train.describe()

#  <a id="42"></a> <br>
# ## 4-2 Visualization

# <a id="421"></a> 
# ## 4-2-1 hist

# In[ ]:


train['target'].value_counts().plot.bar();

# In[ ]:


f,ax=plt.subplots(1,2,figsize=(20,10))
train[train['target']==0].var_0.plot.hist(ax=ax[0],bins=20,edgecolor='black',color='red')
ax[0].set_title('target= 0')
x1=list(range(0,85,5))
ax[0].set_xticks(x1)
train[train['target']==1].var_0.plot.hist(ax=ax[1],color='green',bins=20,edgecolor='black')
ax[1].set_title('target= 1')
x2=list(range(0,85,5))
ax[1].set_xticks(x2)
plt.show()

#  <a id="422"></a> <br>
# ## 4-2-2 Mean Frequency

# In[ ]:


train[train.columns[2:]].mean().plot('hist');plt.title('Mean Frequency');

# <a id="423"></a> 
# ## 4-2-3 countplot

# In[ ]:


f,ax=plt.subplots(1,2,figsize=(18,8))
train['target'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True)
ax[0].set_title('target')
ax[0].set_ylabel('')
sns.countplot('target',data=train,ax=ax[1])
ax[1].set_title('target')
plt.show()

# <a id="424"></a> 
# ## 4-2-4 hist
# If you check histogram for all feature, you will find that most of them are so similar

# In[ ]:


train["var_0"].hist();

# In[ ]:


train["var_81"].hist();

# In[ ]:


train["var_2"].hist();

# <a id="426"></a> 
# ## 4-2-6 distplot
#  The target in data set is **imbalance**

# In[ ]:


sns.set(rc={'figure.figsize':(9,7)})
sns.distplot(train['target']);

# <a id="427"></a> 
# ## 4-2-7 violinplot

# In[ ]:


sns.violinplot(data=train,x="target", y="var_0")

# In[ ]:


sns.violinplot(data=train,x="target", y="var_81")

#  <a id="43"></a> <br>
# ## 4-3 Data Preprocessing
# Before we start this section let me intrduce you, some other compitation that they were similar to this:
# 
# 1. https://www.kaggle.com/artgor/how-to-not-overfit
# 1. https://www.kaggle.com/c/home-credit-default-risk
# 1. https://www.kaggle.com/c/porto-seguro-safe-driver-prediction

#  <a id="431"></a> <br>
# ## 4-3-1 Check missing data for test & train

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


check_missing_data(train)

# In[ ]:


check_missing_data(test)

#  <a id="432"></a> <br>
# ## 4-3-2 Binary Classification

# In[ ]:


train['target'].unique()

#  <a id="433"></a> <br>
# ## 4-3-3 Is data set imbalance?

# A large part of the data is unbalanced, but **how can we  solve it?**

# In[ ]:


train['target'].value_counts()

# In[ ]:


def check_balance(df,target):
    check=[]
    # written by MJ Bahmani for binary target
    print('size of data is:',df.shape[0] )
    for i in [0,1]:
        print('for target  {} ='.format(i))
        print(df[target].value_counts()[i]/df.shape[0]*100,'%')
    

# 1. **Imbalanced dataset** is relevant primarily in the context of supervised machine learning involving two or more classes. 
# 
# 1. **Imbalance** means that the number of data points available for different the classes is different
# 
# <img src='https://www.datascience.com/hs-fs/hubfs/imbdata.png?t=1542328336307&width=487&name=imbdata.png'>
# [Image source](http://api.ning.com/files/vvHEZw33BGqEUW8aBYm4epYJWOfSeUBPVQAsgz7aWaNe0pmDBsjgggBxsyq*8VU1FdBshuTDdL2-bp2ALs0E-0kpCV5kVdwu/imbdata.png)

# In[ ]:


check_balance(train,'target')

# ## 4-3-4 skewness and kurtosis

# In[ ]:


#skewness and kurtosis
print("Skewness: %f" % train['target'].skew())
print("Kurtosis: %f" % train['target'].kurt())

#  <a id="5"></a> <br>
# # 5- Machine Learning Explainability for Santander
# In this section, I want to try extract **insights** from models with the help of this excellent [**Course**](https://www.kaggle.com/learn/machine-learning-explainability) in Kaggle.
# The Goal behind of ML Explainability for Santander is:
# 1. All features are senseless named.(var_1, var2,...) but certainly the importance of each one is different!
# 1. Extract insights from models.
# 1. Find the most inmortant feature in models.
# 1. Affect of each feature on the model's predictions.
# <img src='http://s8.picofile.com/file/8353215168/ML_Explain.png'>
# 
# As you can see from the above, we will refer to three important and practical concepts in this section and try to explain each of them in detail.

#  <a id="51"></a> <br>
# ## 5-1 Permutation Importance
#  In this section we will answer following question:
#  1. What features have the biggest impact on predictions?
#  1. how to extract insights from models?

# ### Prepare our data for our model

# In[ ]:


cols=["target","ID_code"]
X = train.drop(cols,axis=1)
y = train["target"]


# In[ ]:


X_test  = test.drop("ID_code",axis=1)

# ### Create  a sample model to calculate which feature are more important.

# In[ ]:


train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
rfc_model = RandomForestClassifier(random_state=0).fit(train_X, train_y)

#  <a id="52"></a> <br>
# ## 5-2 How to calculate and show importances?

# ### Here is how to calculate and show importances with the [eli5](https://eli5.readthedocs.io/en/latest/) library:

# In[ ]:


import eli5
from eli5.sklearn import PermutationImportance

perm = PermutationImportance(rfc_model, random_state=1).fit(val_X, val_y)

# In[ ]:


eli5.show_weights(perm, feature_names = val_X.columns.tolist(), top=150)

# <a id="53"></a> <br>
# ## 5-3 What can be inferred from the above?
# 1. As you move down the top of the graph, the importance of the feature decreases.
# 1. The features that are shown in green indicate that they have a positive impact on our prediction
# 1. The features that are shown in white indicate that they have no effect on our prediction
# 1. The features shown in red indicate that they have a negative impact on our prediction
# 1.  The most important feature was **Var_110**.

# <a id="54"></a> <br>
# ## 5-4 Partial Dependence Plots
# While **feature importance** shows what **variables** most affect predictions, **partial dependence** plots show how a feature affects predictions.[6][7]
# and partial dependence plots are calculated after a model has been fit. [partial-plots](https://www.kaggle.com/dansbecker/partial-plots)

# In[ ]:


train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
tree_model = DecisionTreeClassifier(random_state=0, max_depth=5, min_samples_split=5).fit(train_X, train_y)

# For the sake of explanation, I use a Decision Tree which you can see below.

# In[ ]:


features = [c for c in train.columns if c not in ['ID_code', 'target']]

# In[ ]:


from sklearn import tree
import graphviz
tree_graph = tree.export_graphviz(tree_model, out_file=None, feature_names=features)

# In[ ]:


graphviz.Source(tree_graph)

# As guidance to read the tree:
# 
# 1. Leaves with children show their splitting criterion on the top
# 1. The pair of values at the bottom show the count of True values and False values for the target respectively, of data points in that node of the tree.
# ><font color="red"><b>Note: </b></font>
# Yes **Var_81** are more effective on our model.

# <a id="55"></a> <br>
# ## 5-5  Partial Dependence Plot
# In this section, we see the impact of the main variables discovered in the previous sections by using the [pdpbox](https://pdpbox.readthedocs.io/en/latest/).

# In[ ]:


from matplotlib import pyplot as plt
from pdpbox import pdp, get_dataset, info_plots

# Create the data that we will plot
pdp_goals = pdp.pdp_isolate(model=tree_model, dataset=val_X, model_features=features, feature='var_81')

# plot it
pdp.pdp_plot(pdp_goals, 'var_81')
plt.show()

# <a id="56"></a> <br>
# ## 5-6 Chart analysis
# 1. The y axis is interpreted as change in the prediction from what it would be predicted at the baseline or leftmost value.
# 1. A blue shaded area indicates level of confidence

# In[ ]:


# Create the data that we will plot
pdp_goals = pdp.pdp_isolate(model=tree_model, dataset=val_X, model_features=features, feature='var_82')

# plot it
pdp.pdp_plot(pdp_goals, 'var_82')
plt.show()

# In[ ]:


# Create the data that we will plot
pdp_goals = pdp.pdp_isolate(model=tree_model, dataset=val_X, model_features=features, feature='var_139')

# plot it
pdp.pdp_plot(pdp_goals, 'var_139')
plt.show()

# In[ ]:


# Create the data that we will plot
pdp_goals = pdp.pdp_isolate(model=tree_model, dataset=val_X, model_features=features, feature='var_110')

# plot it
pdp.pdp_plot(pdp_goals, 'var_110')
plt.show()

# <a id="57"></a> <br>
# ## 5-7 SHAP Values
# **SHAP** (SHapley Additive exPlanations) is a unified approach to explain the output of **any machine learning model**. SHAP connects game theory with local explanations, uniting several previous methods [1-7] and representing the only possible consistent and locally accurate additive feature attribution method based on expectations (see the SHAP NIPS paper for details).
# 
# <img src='https://raw.githubusercontent.com/slundberg/shap/master/docs/artwork/shap_diagram.png' width=400 height=400>
# [image credits](https://github.com/slundberg/shap)
# ><font color="red"><b>Note: </b></font>
# Shap can answer to this qeustion : **how the model works for an individual prediction?**

# In[ ]:


row_to_show = 5
data_for_prediction = val_X.iloc[row_to_show]  # use 1 row of data here. Could use multiple rows if desired
data_for_prediction_array = data_for_prediction.values.reshape(1, -1)


rfc_model.predict_proba(data_for_prediction_array);

# In[ ]:


import shap  # package used to calculate Shap values

# Create object that can calculate shap values
explainer = shap.TreeExplainer(rfc_model)

# Calculate Shap values
shap_values = explainer.shap_values(data_for_prediction)

# If you look carefully at the code where we created the SHAP values, you'll notice we reference Trees in  **shap.TreeExplainer(my_model)**. But the SHAP package has explainers for every type of model.
# 
# 1. shap.DeepExplainer works with Deep Learning models.
# 1. shap.KernelExplainer works with all models, though it is slower than other Explainers and it offers an approximation rather than exact Shap values.

# In[ ]:


shap.initjs()
shap.force_plot(explainer.expected_value[1], shap_values[1], data_for_prediction)

#  <a id="6"></a> <br>
# # 6- Model Development
# So far, we have used two  models, and at this point we add another model and we'll be expanding it soon.
# in this section you will see following model:
# 1. lightgbm
# 1. RandomForestClassifier
# 1. DecisionTreeClassifier
# 1. CatBoostClassifier

# ## 6-1 lightgbm

# In[ ]:


# params is based on following kernel https://www.kaggle.com/brandenkmurray/nothing-works
params = {'objective' : "binary", 
               'boost':"gbdt",
               'metric':"auc",
               'boost_from_average':"false",
               'num_threads':8,
               'learning_rate' : 0.01,
               'num_leaves' : 13,
               'max_depth':-1,
               'tree_learner' : "serial",
               'feature_fraction' : 0.05,
               'bagging_freq' : 5,
               'bagging_fraction' : 0.4,
               'min_data_in_leaf' : 80,
               'min_sum_hessian_in_leaf' : 10.0,
               'verbosity' : 1}

# In[ ]:


y_pred_lgb = np.zeros(len(X_test))
num_round = 1000000
for fold_n, (train_index, valid_index) in enumerate(folds.split(X,y)):
    print('Fold', fold_n, 'started at', time.ctime())
    X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
    y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]
    
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_valid, label=y_valid)
        
    lgb_model = lgb.train(params,train_data,num_round,#change 20 to 2000
                    valid_sets = [train_data, valid_data],verbose_eval=1000,early_stopping_rounds = 3500)##change 10 to 200
            
    y_pred_lgb += lgb_model.predict(X_test, num_iteration=lgb_model.best_iteration)/5

#  <a id="62"></a> <br>
# ## 6-2 RandomForestClassifier

# In[ ]:


y_pred_rfc = rfc_model.predict(X_test)

#  <a id="63"></a> <br>
# ## 6-3 DecisionTreeClassifier

# In[ ]:


y_pred_tree = tree_model.predict(X_test)

#  <a id="64"></a> <br>
# ## 6-4 CatBoostClassifier

# In[ ]:


train_pool = Pool(train_X,train_y)
cat_model = CatBoostClassifier(
                               iterations=3000,# change 25 to 3000 to get best performance 
                               learning_rate=0.03,
                               objective="Logloss",
                               eval_metric='AUC',
                              )
cat_model.fit(train_X,train_y,silent=True)
y_pred_cat = cat_model.predict(X_test)

# Now you can change your model and submit the results of other models.

# In[ ]:


submission_rfc = pd.DataFrame({
        "ID_code": test["ID_code"],
        "target": y_pred_rfc
    })
submission_rfc.to_csv('submission_rfc.csv', index=False)

# In[ ]:


submission_tree = pd.DataFrame({
        "ID_code": test["ID_code"],
        "target": y_pred_tree
    })
submission_tree.to_csv('submission_tree.csv', index=False)

# In[ ]:


submission_cat = pd.DataFrame({
        "ID_code": test["ID_code"],
        "target": y_pred_cat
    })
submission_cat.to_csv('submission_cat.csv', index=False)

# In[ ]:


# good for submit
submission_lgb = pd.DataFrame({
        "ID_code": test["ID_code"],
        "target": y_pred_lgb
    })
submission_lgb.to_csv('submission_lgb.csv', index=False)

#  <a id="65"></a> <br>
# ## 6-5 Funny Combine 

# In[ ]:


submission_rfc_cat = pd.DataFrame({
        "ID_code": test["ID_code"],
        "target": (y_pred_rfc +y_pred_cat)/2
    })
submission_rfc_cat.to_csv('submission_rfc_cat.csv', index=False)

# In[ ]:


submission_lgb_cat = pd.DataFrame({
        "ID_code": test["ID_code"],
        "target": (y_pred_lgb +y_pred_cat)/2
    })
submission_lgb_cat.to_csv('submission_lgb_cat.csv', index=False)

# In[ ]:


submission_rfc_lgb = pd.DataFrame({
        "ID_code": test["ID_code"],
        "target": (y_pred_rfc +y_pred_lgb)/2
    })
submission_rfc_lgb.to_csv('submission_rfc_lgb.csv', index=False)

# you can follow me on:
# > ###### [ GitHub](https://github.com/mjbahmani/)
# > ###### [Kaggle](https://www.kaggle.com/mjbahmani/)
# 
#  <b>I hope you find this kernel helpful and some <font color='red'>UPVOTES</font> would be very much appreciated.<b/>
#  

#  <a id="7"></a> <br>
# # 7- References & credits
# Thanks fo following kernels that help me to create this kernel.

# 1. [https://www.kaggle.com/dansbecker/permutation-importance](https://www.kaggle.com/dansbecker/permutation-importance)
# 1. [https://www.kaggle.com/dansbecker/partial-plots](https://www.kaggle.com/dansbecker/partial-plots)
# 1. [https://www.kaggle.com/miklgr500/catboost-with-gridsearch-cv](https://www.kaggle.com/miklgr500/catboost-with-gridsearch-cv)
# 1. [https://www.kaggle.com/dromosys/sctp-working-lgb](https://www.kaggle.com/dromosys/sctp-working-lgb)
# 1. [https://www.kaggle.com/gpreda/santander-eda-and-prediction](https://www.kaggle.com/gpreda/santander-eda-and-prediction)
# 1. [https://www.kaggle.com/dansbecker/permutation-importance](https://www.kaggle.com/dansbecker/permutation-importance)
# 1. [https://www.kaggle.com/dansbecker/partial-plots](https://www.kaggle.com/dansbecker/partial-plots)
# 1. [https://www.kaggle.com/dansbecker/shap-values](https://www.kaggle.com/dansbecker/shap-values)
# 1. [https://docs.microsoft.com/en-us/azure/machine-learning/studio/algorithm-choice](https://docs.microsoft.com/en-us/azure/machine-learning/studio/algorithm-choice)
# 1. [kernel https://www.kaggle.com/arjanso/reducing-dataframe-memory-size-by-65](kernel https://www.kaggle.com/arjanso/reducing-dataframe-memory-size-by-65)
# 1. [https://www.kaggle.com/brandenkmurray/nothing-works](https://www.kaggle.com/brandenkmurray/nothing-works)
