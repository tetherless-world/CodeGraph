#!/usr/bin/env python
# coding: utf-8

# # Background
# - As you know, there are features which have high cardinality in this competition.
# - I've studied and read some discussions, blogs and articles about high cardinality.
# - In this kernel, I'll experiment to see which encoding works better.
# - Label encoding, Frequency encoding and Mean encoding will be tested.
# - Because I'm student, I welcome your feedback on anything of this contents.!
# - Ok, Let's see!.
# - To compare, I forked great kernel https://www.kaggle.com/sudalairajkumar/simple-exploration-baseline-ga-customer-revenue.
# - I set the same rrandom number and same parameters for each cases.
# - I want to recommend you, this kernel. https://www.kaggle.com/vprokopev/mean-likelihood-encodings-a-comprehensive-study. 
# - This kernel experiments and explains these encodings. Very useful!

# In[ ]:


import pandas as pd
import numpy as np

import lightgbm as lgb
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
from sklearn import model_selection, preprocessing, metrics
import datetime\

import os
import json
import numpy as np
import pandas as pd
from pandas.io.json import json_normalize

# In[ ]:


def load_df(csv_path='../input/train.csv', nrows=None):
    JSON_COLUMNS = ['device', 'geoNetwork', 'totals', 'trafficSource']
    
    df = pd.read_csv(csv_path, 
                     converters={column: json.loads for column in JSON_COLUMNS}, 
                     dtype={'fullVisitorId': str}, # Important!!
                     nrows=nrows)
    
    for column in JSON_COLUMNS:
        column_as_df = json_normalize(df[column])
        column_as_df.columns = [f"{column}_{subcolumn}" for subcolumn in column_as_df.columns]
        df = df.drop(column, axis=1).merge(column_as_df, right_index=True, left_index=True)
    print(f"Loaded {os.path.basename(csv_path)}. Shape: {df.shape}")
    return df

# In[ ]:


df_train = load_df()
df_test = load_df("../input/test.csv")

# In[ ]:


df_train["totals_transactionRevenue"] = df_train["totals_transactionRevenue"].astype('float')

# In[ ]:


const_cols = [c for c in df_train.columns if df_train[c].nunique(dropna=False)==1 ]
const_cols

# In[ ]:


cols_to_drop = const_cols + ['sessionId']

df_train = df_train.drop(cols_to_drop + ["trafficSource_campaignCode"], axis=1)
df_test = df_test.drop(cols_to_drop, axis=1)

# In[ ]:


print(df_train.shape, df_test.shape)

# In[ ]:


# Impute 0 for missing target values
df_train["totals_transactionRevenue"].fillna(0, inplace=True)
train_y = df_train["totals_transactionRevenue"].values
train_id = df_train["fullVisitorId"].values
test_id = df_test["fullVisitorId"].values


# label encode the categorical variables and convert the numerical variables to float
cat_cols = ["channelGrouping", "device_browser", 
            "device_deviceCategory", "device_operatingSystem", 
            "geoNetwork_city", "geoNetwork_continent", 
            "geoNetwork_country", "geoNetwork_metro",
            "geoNetwork_networkDomain", "geoNetwork_region", 
            "geoNetwork_subContinent", "trafficSource_adContent", 
            "trafficSource_adwordsClickInfo.adNetworkType", 
            "trafficSource_adwordsClickInfo.gclId", 
            "trafficSource_adwordsClickInfo.page", 
            "trafficSource_adwordsClickInfo.slot", "trafficSource_campaign",
            "trafficSource_keyword", "trafficSource_medium", 
            "trafficSource_referralPath", "trafficSource_source",
            'trafficSource_adwordsClickInfo.isVideoAd',
            'trafficSource_isTrueDirect', 'device_isMobile']

# In[ ]:


df_train['date'] = df_train['date'].apply(lambda x: datetime.date(int(str(x)[:4]), int(str(x)[4:6]), int(str(x)[6:])))
df_train['date'] = pd.to_datetime(df_train['date'])

# In[ ]:


num_cols = ["totals_hits", "totals_pageviews", 
            "visitNumber", "visitStartTime", 
            'totals_bounces',  'totals_newVisits']    

for col in num_cols:
    df_train[col] = df_train[col].astype(float)
    df_test[col] = df_test[col].astype(float)

# In[ ]:


df_train_copy = df_train.copy()
df_test_copy = df_test.copy()

df_train = df_train_copy.copy()
df_test = df_test_copy.copy()

# In[ ]:


train_dates = df_train['date'].copy()

# # Label-encoding

# In[ ]:


for col in cat_cols:
    print(col)
    lbl = preprocessing.LabelEncoder()
    lbl.fit(list(df_train[col].values.astype('str')) + list(df_test[col].values.astype('str')))
    df_train[col] = lbl.transform(list(df_train[col].values.astype('str')))
    df_test[col] = lbl.transform(list(df_test[col].values.astype('str')))

# In[ ]:


df_train.shape

# In[ ]:


for col in df_train.columns:
    if col not in num_cols and col not in cat_cols:
        print(col)

# In[ ]:


not_use_cols = ['date', 'fullVisitorId', 'visitId', 'totals_transactionRevenue']

# In[ ]:


len(cat_cols) + len(num_cols)

# In[ ]:


len(not_use_cols)

# In[ ]:


# Split the train dataset into development and valid based on time 
dev_df = df_train[df_train['date']<=datetime.date(2017,5,31)]
val_df = df_train[df_train['date']>datetime.date(2017,5,31)]
dev_y = np.log1p(dev_df["totals_transactionRevenue"].values)
val_y = np.log1p(val_df["totals_transactionRevenue"].values)

use_cols = [col for col in df_train.columns if col not in not_use_cols]

dev_X = dev_df[use_cols] 
val_X = val_df[use_cols] 
test_X = df_test[use_cols] 

# custom function to run light gbm model
def run_lgb(train_X, train_y, val_X, val_y, test_X):
    params = {
        "objective" : "regression",
        "metric" : "rmse", 
        "num_leaves" : 30,
        "min_child_samples" : 100,
        "learning_rate" : 0.1,
        "bagging_fraction" : 0.7,
        "feature_fraction" : 0.5,
        "bagging_frequency" : 5,
        "bagging_seed" : 1989,
        "verbosity" : -1,
        'seed': 1989
    }
    
    lgtrain = lgb.Dataset(train_X, label=train_y)
    lgval = lgb.Dataset(val_X, label=val_y)
    model = lgb.train(params, lgtrain, 10000, valid_sets=[lgtrain, lgval], early_stopping_rounds=500, verbose_eval=100)
    
    pred_test_y = model.predict(test_X, num_iteration=model.best_iteration)
    return pred_test_y, model

# Training the model #
pred_test, model = run_lgb(dev_X, dev_y, val_X, val_y, test_X)

# In[ ]:


sub_df = pd.DataFrame({"fullVisitorId":test_id})
pred_test[pred_test<0] = 0
sub_df["PredictedLogRevenue"] = np.expm1(pred_test)
sub_df = sub_df.groupby("fullVisitorId")["PredictedLogRevenue"].sum().reset_index()
sub_df.columns = ["fullVisitorId", "PredictedLogRevenue"]
sub_df["PredictedLogRevenue"] = np.log1p(sub_df["PredictedLogRevenue"])
sub_df.to_csv("Label_encoding.csv", index=False)

# # Frequency encoding

# In[ ]:


def frequency_encoding(frame, col):
    freq_encoding = frame.groupby([col]).size()/frame.shape[0] 
    freq_encoding = freq_encoding.reset_index().rename(columns={0:'{}_Frequency'.format(col)})
    return frame.merge(freq_encoding, on=col, how='left')

# In[ ]:


df_train = df_train_copy.copy()
df_test = df_test_copy.copy()

# In[ ]:


len_train = df_train.shape[0]
df_all = pd.concat([df_train, df_test])

for col in tqdm(cat_cols):
    df_all = frequency_encoding(df_all, col)

df_train = df_all[:len_train]
df_test = df_all[len_train:]

print(df_train.shape, df_test.shape)

# In[ ]:


df_train.drop(cat_cols, axis=1, inplace=True)
df_test.drop(cat_cols, axis=1, inplace=True)

freq_cat_cols = ['{}_Frequency'.format(col) for col in cat_cols]

# Split the train dataset into development and valid based on time 
dev_df = df_train[train_dates<=datetime.date(2017,5,31)]
val_df = df_train[train_dates>datetime.date(2017,5,31)]

dev_y = np.log1p(dev_df["totals_transactionRevenue"].values)
val_y = np.log1p(val_df["totals_transactionRevenue"].values)

use_cols = [col for col in df_train.columns if col not in not_use_cols]

dev_X = dev_df[use_cols] 
val_X = val_df[use_cols] 
test_X = df_test[use_cols]  

# Training the model #
pred_test, model = run_lgb(dev_X, dev_y, val_X, val_y, test_X)

# In[ ]:


sub_df = pd.DataFrame({"fullVisitorId":test_id})
pred_test[pred_test<0] = 0
sub_df["PredictedLogRevenue"] = np.expm1(pred_test)
sub_df = sub_df.groupby("fullVisitorId")["PredictedLogRevenue"].sum().reset_index()
sub_df.columns = ["fullVisitorId", "PredictedLogRevenue"]
sub_df["PredictedLogRevenue"] = np.log1p(sub_df["PredictedLogRevenue"])
sub_df.to_csv("Freq_encoding.csv", index=False)

# # Mean encoding

# In[ ]:


from sklearn.model_selection import KFold

# In[ ]:


def mean_k_fold_encoding(col, alpha):
    target_name = 'totals_transactionRevenue'
    target_mean_global = df_train[target_name].mean()
    
    nrows_cat = df_train.groupby(col)[target_name].count()
    target_means_cats = df_train.groupby(col)[target_name].mean()
    target_means_cats_adj = (target_means_cats*nrows_cat + 
                             target_mean_global*alpha)/(nrows_cat+alpha)
    # Mapping means to test data
    encoded_col_test = df_test[col].map(target_means_cats_adj)
    
    kfold = KFold(n_splits=5, shuffle=True, random_state=1989)
    parts = []
    for trn_inx, val_idx in kfold.split(df_train):
        df_for_estimation, df_estimated = df_train.iloc[trn_inx], df_train.iloc[val_idx]
        nrows_cat = df_for_estimation.groupby(col)[target_name].count()
        target_means_cats = df_for_estimation.groupby(col)[target_name].mean()

        target_means_cats_adj = (target_means_cats * nrows_cat + 
                                target_mean_global * alpha) / (nrows_cat + alpha)

        encoded_col_train_part = df_estimated[col].map(target_means_cats_adj)
        parts.append(encoded_col_train_part)
        
    encoded_col_train = pd.concat(parts, axis=0)
    encoded_col_train.fillna(target_mean_global, inplace=True)
    encoded_col_train.sort_index(inplace=True)
    
    return encoded_col_train, encoded_col_test

# In[ ]:


df_train = df_train_copy.copy()
df_test = df_test_copy.copy()

# In[ ]:


for col in tqdm(cat_cols):
    temp_encoded_tr, temp_encoded_te = mean_k_fold_encoding(col, 5)
    new_feat_name = 'mean_k_fold_{}'.format(col)
    df_train[new_feat_name] = temp_encoded_tr.values
    df_test[new_feat_name] = temp_encoded_te.values

# In[ ]:


df_train.drop(cat_cols, axis=1, inplace=True)
df_test.drop(cat_cols, axis=1, inplace=True)

mean_cat_cols = ['mean_k_fold_{}'.format(col) for col in cat_cols]

# Split the train dataset into development and valid based on time 
dev_df = df_train[train_dates<=datetime.date(2017,5,31)]
val_df = df_train[train_dates>datetime.date(2017,5,31)]

dev_y = np.log1p(dev_df["totals_transactionRevenue"].values)
val_y = np.log1p(val_df["totals_transactionRevenue"].values)

use_cols = [col for col in df_train.columns if col not in not_use_cols]

dev_X = dev_df[use_cols] 
val_X = val_df[use_cols] 
test_X = df_test[use_cols] 

# Training the model #
pred_test, model = run_lgb(dev_X, dev_y, val_X, val_y, test_X)

# In[ ]:


sub_df = pd.DataFrame({"fullVisitorId":test_id})
pred_test[pred_test<0] = 0
sub_df["PredictedLogRevenue"] = np.expm1(pred_test)
sub_df = sub_df.groupby("fullVisitorId")["PredictedLogRevenue"].sum().reset_index()
sub_df.columns = ["fullVisitorId", "PredictedLogRevenue"]
sub_df["PredictedLogRevenue"] = np.log1p(sub_df["PredictedLogRevenue"])
sub_df.to_csv("mean_encoding.csv", index=False)

# # Label + Frequency + Mean encoding

# In[ ]:


df_train = df_train_copy.copy()
df_test = df_test_copy.copy()

len_train = df_train.shape[0]
df_all = pd.concat([df_train, df_test])

for col in tqdm(cat_cols):
    df_all = frequency_encoding(df_all, col)

df_train = df_all[:len_train]
df_test = df_all[len_train:]

print(df_train.shape, df_test.shape)

for col in tqdm(cat_cols):
    temp_encoded_tr, temp_encoded_te = mean_k_fold_encoding(col, 5)
    new_feat_name = 'mean_k_fold_{}'.format(col)
    df_train[new_feat_name] = temp_encoded_tr.values
    df_test[new_feat_name] = temp_encoded_te.values

# In[ ]:


for col in cat_cols:
    print(col)
    lbl = preprocessing.LabelEncoder()
    lbl.fit(list(df_train[col].values.astype('str')) + list(df_test[col].values.astype('str')))
    df_train[col] = lbl.transform(list(df_train[col].values.astype('str')))
    df_test[col] = lbl.transform(list(df_test[col].values.astype('str')))

# In[ ]:


freq_cat_cols = ['{}_Frequency'.format(col) for col in cat_cols]
mean_cat_cols = ['mean_k_fold_{}'.format(col) for col in cat_cols]

# In[ ]:


print(df_train.shape, df_test.shape)

# Split the train dataset into development and valid based on time 
dev_df = df_train[train_dates<=datetime.date(2017,5,31)]
val_df = df_train[train_dates>datetime.date(2017,5,31)]
dev_y = np.log1p(dev_df["totals_transactionRevenue"].values)
val_y = np.log1p(val_df["totals_transactionRevenue"].values)

use_cols = [col for col in df_train.columns if col not in not_use_cols]

dev_X = dev_df[use_cols] 
val_X = val_df[use_cols] 
test_X = df_test[use_cols]  

# Training the model #
pred_test, model = run_lgb(dev_X, dev_y, val_X, val_y, test_X)

# In[ ]:


sub_df = pd.DataFrame({"fullVisitorId":test_id})
pred_test[pred_test<0] = 0
sub_df["PredictedLogRevenue"] = np.expm1(pred_test)
sub_df = sub_df.groupby("fullVisitorId")["PredictedLogRevenue"].sum().reset_index()
sub_df.columns = ["fullVisitorId", "PredictedLogRevenue"]
sub_df["PredictedLogRevenue"] = np.log1p(sub_df["PredictedLogRevenue"])
sub_df.to_csv("all.csv", index=False)

# # Result
# | Encoding  | Training RMSE  | VALID RMSE  |  RMSE(Tr) / RMSE(vld)  | LB 
# |---|---|---|---|---|
# | Label encoding |  1.52503  | 1.69546  |  1.111755 |  1.4470
# |  Frequency encoding | 1.52039  | 1.69291 | 1.113471 | 1.4545
# |  Mean encoding |  1.52247 | 1.6955  | 1.113651  |  1.4448
# |Label + Fre + Mean (All) | 1.51965 | 1.69179 | 1.1132761 |  1.4417

# - Traning RMSE: All  < Freq < Mean < Label
# - Valid RMSE: All < Freq < Label < Mean
# - Tr/vld ratio: Label < All < Freq < Mean

# - Freq, Mean encoding tend to overfit to training set.
# - But LB of Mean encoding is lower than LB of Freq.

# # Conclusion

# - Of course, it's not easy to say the all(Label + Fre + Mean) is the best choice.
# - But, adding other encoding performs better than only using label encoding.

# # More
# - I know, this experiment is so simple. The result could be changed depending on hyper-parameters and algoritm.
# - But, I think adding various encoding give us better performance than using only one encoding strategy because applying various approachs (like ensemble)  commoly shows good result.

# In[ ]:




# In[ ]:



