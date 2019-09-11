#!/usr/bin/env python
# coding: utf-8

# One of the most important things to establish for every Kaggle competition is whether there is a significatn differnece in distributions of the train and test sets. So far the CV validation scores for kernels and for public LB have been pretty close, but the local CV seems to be consistently about 0.01 better than the LB scores. It would be interesting, and potentially very valuable, to find out in a more quantitative and specific way how do these distributions compare. For that purpose we'll build an adverserial validation scheme - we'll run a CV classifier that tries to predict if any given question belongs to the train or the test set.
# 
# Firts, we'll have to deal with data gregation adn building of the combined train and test datasets. This work has already been doen in many of the kernels, and in this kernel we'll realy on [Rahul Bamola](https://www.kaggle.com/bamola)'s excelent [kernel](https://www.kaggle.com/bamola/elo-eda-and-modelling-lgbm).

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn import model_selection, preprocessing, metrics
import warnings
import datetime
warnings.filterwarnings("ignore")
import os
print(os.listdir("../input"))

# In[ ]:


#Loading Train and Test Data
df_train = pd.read_csv("../input/train.csv", parse_dates=["first_active_month"])
df_test = pd.read_csv("../input/test.csv", parse_dates=["first_active_month"])
print("{} observations and {} features in train set.".format(df_train.shape[0],df_train.shape[1]))
print("{} observations and {} features in test set.".format(df_test.shape[0],df_test.shape[1]))

# In[ ]:


# Let's explore New and Old Merchants
df_new_trans = pd.read_csv("../input/new_merchant_transactions.csv")
df_hist_trans = pd.read_csv("../input/historical_transactions.csv")


# In[ ]:


df_h = df_hist_trans.groupby("card_id").size().reset_index().rename({0:'transactions'},axis=1)
df_n = df_new_trans.groupby("card_id").size().reset_index().rename({0:'transactions'},axis=1)

# In[ ]:


print("Historic Transactions ---->   Average transactions per card : {:.0f}, Maximum transactions : {}.\nNew Transactions ---->   Average transactions per card : {:.0f}, Maximum transactions : {}. ".format(df_h['transactions'].mean(),df_h['transactions'].max(),df_n['transactions'].mean(),df_n['transactions'].max()))

# In[ ]:


#Comparting Historic Transactions vs New Transactions for Top 50 Merchants 
m_df_h = df_hist_trans.groupby("merchant_id").size().reset_index().rename({0:'transactions'},axis=1)
m_df_n = df_new_trans.groupby("merchant_id").size().reset_index().rename({0:'transactions'},axis=1)



# In[ ]:


print("Historic Transactions ---->   Average transactions per merchant : {:.0f}, Maximum transactions : {}.\nNew Transactions ---->   Average transactions per merchant : {:.0f}, Maximum transactions : {}. ".format(m_df_h['transactions'].mean(),m_df_h['transactions'].max(),m_df_n['transactions'].mean(),m_df_n['transactions'].max()))

# In[ ]:


df_train["year"] = df_train["first_active_month"].dt.year
df_test["year"] = df_test["first_active_month"].dt.year
df_train["month"] = df_train["first_active_month"].dt.month
df_test["month"] = df_test["first_active_month"].dt.month
df_train['elapsed_time'] = (datetime.date(2018, 2, 1) - df_train['first_active_month'].dt.date).dt.days
df_test['elapsed_time'] = (datetime.date(2018, 2, 1) - df_test['first_active_month'].dt.date).dt.days

df_hist_trans['authorized_flag'] = df_hist_trans['authorized_flag'].map({'Y':1, 'N':0})
df_new_trans['authorized_flag'] = df_new_trans['authorized_flag'].map({'Y':1, 'N':0})
#df_merch_trans = pd.concat([df_hist_trans,df_new_trans])

# In[ ]:


def aggregate_hist_transactions(trans):
    
    trans.loc[:, 'purchase_date'] = pd.DatetimeIndex(trans['purchase_date']).\
                                      astype(np.int64) * 1e-9
    
    agg_func = {
        'authorized_flag': ['sum', 'mean'],
        'merchant_id': ['nunique'],
        'city_id': ['nunique'],
        'purchase_amount': ['sum', 'median', 'max', 'min', 'std'],
        'installments': ['sum', 'median', 'max', 'min', 'std'],
        'purchase_date': [np.ptp],
        'month_lag': ['min', 'max']
        }
    agg_trans = trans.groupby(['card_id']).agg(agg_func)
    agg_trans.columns = ['hist_' + '_'.join(col).strip() 
                           for col in agg_trans.columns.values]
    agg_trans.reset_index(inplace=True)
    
    df = (trans.groupby('card_id')
          .size()
          .reset_index(name='transactions_count'))
    
    agg_trans = pd.merge(df, agg_trans, on='card_id', how='left')
    
    return agg_trans

def aggregate_new_transactions(trans):
    
    trans.loc[:, 'purchase_date'] = pd.DatetimeIndex(trans['purchase_date']).\
                                      astype(np.int64) * 1e-9
    
    agg_func = {
        'authorized_flag': ['sum', 'mean'],
        'merchant_id': ['nunique'],
        'city_id': ['nunique'],
        'purchase_amount': ['sum', 'median', 'max', 'min', 'std'],
        'installments': ['sum', 'median', 'max', 'min', 'std'],
        'purchase_date': [np.ptp],
        'month_lag': ['min', 'max']
        }
    agg_trans = trans.groupby(['card_id']).agg(agg_func)
    agg_trans.columns = ['new_' + '_'.join(col).strip() 
                           for col in agg_trans.columns.values]
    agg_trans.reset_index(inplace=True)
    
    df = (trans.groupby('card_id')
          .size()
          .reset_index(name='transactions_count'))
    
    agg_trans = pd.merge(df, agg_trans, on='card_id', how='left')
    
    return agg_trans

# In[ ]:


merch_hist = aggregate_hist_transactions(df_hist_trans)
merch_new = aggregate_new_transactions(df_new_trans)

# In[ ]:


#Merging history with training and test data
df_train = pd.merge(df_train, merch_hist, on='card_id',how='left')
df_test = pd.merge(df_test, merch_hist, on='card_id',how='left')

df_train = pd.merge(df_train, merch_new, on='card_id',how='left')
df_test = pd.merge(df_test, merch_new, on='card_id',how='left')

target = df_train['target']

use_cols  = [c for c in df_train.columns if c not in ['card_id', 'first_active_month', 'target']]

features = list(df_train[use_cols].columns)
categorical_features = [col for col in features if 'feature_' in col]

# In[ ]:


df_train = df_train[use_cols]
df_test = df_test[use_cols]

# In[ ]:


df_train.shape

# In[ ]:


df_test.shape

# In[ ]:


df_train['target'] = 0
df_test['target'] = 1

# In[ ]:


train_test = pd.concat([df_train, df_test], axis =0)

target = train_test['target'].values

# In[ ]:


param = {'num_leaves': 50,
         'min_data_in_leaf': 30, 
         'objective':'binary',
         'max_depth': 5,
         'learning_rate': 0.001,
         "min_child_samples": 20,
         "boosting": "gbdt",
         "feature_fraction": 0.9,
         "bagging_freq": 1,
         "bagging_fraction": 0.9 ,
         "bagging_seed": 17,
         "metric": 'auc',
         "lambda_l1": 0.1,
         "verbosity": -1}

folds = KFold(n_splits=5, shuffle=True, random_state=15)
oof = np.zeros(len(train_test))


for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_test.values, target)):
    print("fold n°{}".format(fold_))
    trn_data = lgb.Dataset(train_test.iloc[trn_idx][features], label=target[trn_idx], categorical_feature=categorical_features)
    val_data = lgb.Dataset(train_test.iloc[val_idx][features], label=target[val_idx], categorical_feature=categorical_features)

    num_round = 10000
    clf = lgb.train(param, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=100, early_stopping_rounds = 100)
    oof[val_idx] = clf.predict(train_test.iloc[val_idx][features], num_iteration=clf.best_iteration)


# We see that the oof AUC is **VERY** close to 0.5, so these two datasets seem very statistically similar. In other words, relying on your local validation should work very well for this competition. 
