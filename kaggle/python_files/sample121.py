#!/usr/bin/env python
# coding: utf-8

# # Santander Customer Transaction Prediction
# 
# This kernel uses LGBM model to predict Customer Transaction.
# 
# **For LightGBM parameters optimization, please find my other kernel below, where I show how to take advantage of Bayesian Optimization to find optimal paramer ofr LightGBM:**
# 
# https://www.kaggle.com/fayzur/lgb-bayesian-parameters-finding-rank-average
# 
# 
# ## Notebook  Content
# 1. [Loading the data](#0) <br>    
# 1. [Training the model](#1)
# 1. [Submission](#2)

# <a id="0"></a> <br>
# ## 1. Loading the data

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
from sklearn import metrics
import gc

pd.set_option('display.max_columns', 200)

# In[ ]:


train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')

# We are given anonymized dataset containing 200 numeric feature variables from var_0 to var_199. Let's have a look train dataset:

# In[ ]:


train_df.head()

# Test dataset:

# In[ ]:


test_df.head()

# Distribution of target variable

# In[ ]:


train_df.target.value_counts()

# The problem is unbalance!
# We can build a quick model on this dataset considering unbalance to see how far we can go without Feature engineering! 

# <a id="1"></a> <br>
# ## 2. Training the model

# In[ ]:


#parameters were obtained using the same structure presented in the following kernel:

#   https://www.kaggle.com/fayzur/lgb-bayesian-parameters-finding-rank-average

param = {
    'num_leaves': 18,
     'max_bin': 63,
     'min_data_in_leaf': 5,
     'learning_rate': 0.010614430970330217,
     'min_sum_hessian_in_leaf': 0.0093586657313989123,
     'feature_fraction': 0.056701788569420042,
     'lambda_l1': 0.060222413158420585,
     'lambda_l2': 4.6580550589317573,
     'min_gain_to_split': 0.29588543202055562,
     'max_depth': 49,
     'save_binary': True,
     'seed': 1337,
     'feature_fraction_seed': 1337,
     'bagging_seed': 1337,
     'drop_seed': 1337,
     'data_random_seed': 1337,
     'objective': 'binary',
     'boosting_type': 'gbdt',
     'verbose': 1,
     'metric': 'auc',
     'is_unbalance': True,
     'boost_from_average': False
}


# The above parameters were obtained using the same structure presented in the following kernel:
# 
# https://www.kaggle.com/fayzur/lgb-bayesian-parameters-finding-rank-average

# Number of Kfolds:

# In[ ]:


nfold = 10

# In[ ]:


target = 'target'
predictors = train_df.columns.values.tolist()[2:]

# In[ ]:


gc.collect()

# In[ ]:


skf = StratifiedKFold(n_splits=nfold, shuffle=True, random_state=2019)

oof = np.zeros(len(train_df))
predictions = np.zeros(len(test_df))

i = 1
for train_index, valid_index in skf.split(train_df, train_df.target.values):
    print("\nfold {}".format(i))
    xg_train = lgb.Dataset(train_df.iloc[train_index][predictors].values,
                           label=train_df.iloc[train_index][target].values,
                           feature_name=predictors,
                           free_raw_data = False
                           )
    xg_valid = lgb.Dataset(train_df.iloc[valid_index][predictors].values,
                           label=train_df.iloc[valid_index][target].values,
                           feature_name=predictors,
                           free_raw_data = False
                           )   

    nround = 8523
    clf = lgb.train(param, xg_train, nround, valid_sets = [xg_valid], verbose_eval=250)
    oof[valid_index] = clf.predict(train_df.iloc[valid_index][predictors].values, num_iteration=nround) 
    
    predictions += clf.predict(test_df[predictors], num_iteration=nround) / nfold
    i = i + 1

print("\n\nCV AUC: {:<0.4f}".format(metrics.roc_auc_score(train_df.target.values, oof)))

# <a id="2"></a> <br>
# ## 2. Submission

# In[ ]:


sub_df = pd.DataFrame({"ID_code": test_df.ID_code.values})
sub_df["target"] = predictions
sub_df[:10]

# In[ ]:


sub_df.to_csv("Customer_Transaction.csv", index=False)

# Upvote if it is useful :)

# In[ ]:




# In[ ]:




# In[ ]:



