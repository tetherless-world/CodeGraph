#!/usr/bin/env python
# coding: utf-8

# # Combining your model with a model without outlier

# Assuming that you have already finished your feature engineering and you have two dataset:
# 
# - ***train_clean.csv***
# - ***test_clean.csv***
# 
# In train_clean.csv, there's an **'outlier' column with values 1/0. **
# 
# Besides, you have your best LB submission:
# - ***3.695.csv*** (thanks  **Ashish Patel(阿希什)** My original model can't rich this score, so I try to use the idea to improve your submission to get better LB socre.）
# 
# The flows of this pipline is as follows:
# 1. Training a model using a training set without outliers. (we get: **Model_1**)
# 2. Training a model to classify outliers. (we get: **Model_2**)
# 3. Using **Model_2** to predict whether an card_id in test set is an outliers. (we get:**Outlier_Likelyhood**)
# 4. Spliting out the card_id from **Outlier_Likelyhood** with top 10% (or some other ratio) score. (we get:**Outlier_ID**)
# 5. Combining your submission using your **best submission (that is, your best model)** to predict **Outlier_ID** in test set and using **Model_1** to predict the rest of the test set.
# 
# The  basic idea behind this pipline is:
# 1. Training model without outliers make the model more accurate for non-outliers.
# 2. A great proportion of the error is caused by outliers, so we need to use a model training with outliers to predict them. How to find them out? build a classifier!

# In[ ]:


import numpy as np
import pandas as pd
import time
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import log_loss

# # Part 1 Training Model Without Outliers

# In[ ]:


df_train = pd.read_csv('../input/predicting-outliers-to-improve-your-score/train_clean.csv')
df_test = pd.read_csv('../input/predicting-outliers-to-improve-your-score/test_clean.csv')

# ## filtering out outliers

# In[ ]:


df_train = df_train[df_train['outliers'] == 0]
target = df_train['target']
del df_train['target']
features = [c for c in df_train.columns if c not in ['card_id', 'first_active_month','outliers']]
categorical_feats = [c for c in features if 'feature_' in c]

# ## parameters

# In[ ]:


param = {'objective':'regression',
         'num_leaves': 31,
         'min_data_in_leaf': 25,
         'max_depth': 7,
         'learning_rate': 0.01,
         'lambda_l1':0.13,
         "boosting": "gbdt",
         "feature_fraction":0.85,
         'bagging_freq':8,
         "bagging_fraction": 0.9 ,
         "metric": 'rmse',
         "verbosity": -1,
         "random_state": 2333}

# ## training model

# In[ ]:


folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=2333)
oof = np.zeros(len(df_train))
predictions = np.zeros(len(df_test))
feature_importance_df = pd.DataFrame()

for fold_, (trn_idx, val_idx) in enumerate(folds.split(df_train,df_train['outliers'].values)):
    print("fold {}".format(fold_))
    trn_data = lgb.Dataset(df_train.iloc[trn_idx][features], label=target.iloc[trn_idx])#, categorical_feature=categorical_feats)
    val_data = lgb.Dataset(df_train.iloc[val_idx][features], label=target.iloc[val_idx])#, categorical_feature=categorical_feats)

    num_round = 10000
    clf = lgb.train(param, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval= 100, early_stopping_rounds = 200)
    oof[val_idx] = clf.predict(df_train.iloc[val_idx][features], num_iteration=clf.best_iteration)
    
    fold_importance_df = pd.DataFrame()
    fold_importance_df["Feature"] = features
    fold_importance_df["importance"] = clf.feature_importance()
    fold_importance_df["fold"] = fold_ + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
    
    predictions += clf.predict(df_test[features], num_iteration=clf.best_iteration) / folds.n_splits

print("CV score: {:<8.5f}".format(mean_squared_error(oof, target)**0.5))

# In[ ]:


model_without_outliers = pd.DataFrame({"card_id":df_test["card_id"].values})
model_without_outliers["target"] = predictions

# # Part 2 Training Model For Outliers Classification

# In[ ]:


df_train = pd.read_csv('../input/predicting-outliers-to-improve-your-score/train_clean.csv')
df_test = pd.read_csv('../input/predicting-outliers-to-improve-your-score/test_clean.csv')

# ## using outliers column as labels instead of target column

# In[ ]:


target = df_train['outliers']
del df_train['outliers']
del df_train['target']

# In[ ]:


features = [c for c in df_train.columns if c not in ['card_id', 'first_active_month']]
categorical_feats = [c for c in features if 'feature_' in c]

# ## parameters

# In[ ]:


param = {'num_leaves': 31,
         'min_data_in_leaf': 30, 
         'objective':'binary',
         'max_depth': 6,
         'learning_rate': 0.01,
         "boosting": "rf",
         "feature_fraction": 0.9,
         "bagging_freq": 1,
         "bagging_fraction": 0.9 ,
         "bagging_seed": 11,
         "metric": 'binary_logloss',
         "lambda_l1": 0.1,
         "verbosity": -1,
         "random_state": 2333}

# ## training model

# In[ ]:


folds = KFold(n_splits=5, shuffle=True, random_state=15)
oof = np.zeros(len(df_train))
predictions = np.zeros(len(df_test))
feature_importance_df = pd.DataFrame()

start = time.time()


for fold_, (trn_idx, val_idx) in enumerate(folds.split(df_train.values, target.values)):
    print("fold n°{}".format(fold_))
    trn_data = lgb.Dataset(df_train.iloc[trn_idx][features], label=target.iloc[trn_idx], categorical_feature=categorical_feats)
    val_data = lgb.Dataset(df_train.iloc[val_idx][features], label=target.iloc[val_idx], categorical_feature=categorical_feats)

    num_round = 10000
    clf = lgb.train(param, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=100, early_stopping_rounds = 200)
    oof[val_idx] = clf.predict(df_train.iloc[val_idx][features], num_iteration=clf.best_iteration)
    
    fold_importance_df = pd.DataFrame()
    fold_importance_df["feature"] = features
    fold_importance_df["importance"] = clf.feature_importance()
    fold_importance_df["fold"] = fold_ + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
    
    predictions += clf.predict(df_test[features], num_iteration=clf.best_iteration) / folds.n_splits

print("CV score: {:<8.5f}".format(log_loss(target, oof)))

# In[ ]:


### 'target' is the probability of whether an observation is an outlier
df_outlier_prob = pd.DataFrame({"card_id":df_test["card_id"].values})
df_outlier_prob["target"] = predictions
df_outlier_prob.head()

# # Part 3 Combining Submission:
# So far so good !
# We now have three dataset:
# 
# 1. Best Submission
# 2. Prediction Using Model Without Outliers
# 3. Probability of Outliers In Test set
# 

# In[ ]:


# if the test set has the same ratio of outliers as training set, 
# then the numbuer of outliers in test is about: (1.06% outliers in training set)
123623*0.0106

# In[ ]:


# In case missing some predictable outlier, we choose top 25000 with highest outliers likelyhood.
outlier_id = pd.DataFrame(df_outlier_prob.sort_values(by='target',ascending = False).head(25000)['card_id'])

# In[ ]:


best_submission = pd.read_csv('../input/predicting-outliers-to-improve-your-score/3.695.csv')

# In[ ]:


most_likely_liers = best_submission.merge(outlier_id,how='right')
most_likely_liers.head()

# In[ ]:


for card_id in most_likely_liers['card_id']:
    model_without_outliers.loc[model_without_outliers['card_id']==card_id,'target']\
    = most_likely_liers.loc[most_likely_liers['card_id']==card_id,'target'].values

# In[ ]:


model_without_outliers.to_csv("combining_submission.csv", index=False)
