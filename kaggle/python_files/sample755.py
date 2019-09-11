#!/usr/bin/env python
# coding: utf-8

# I made too much simple NN model.
# https://www.kaggle.com/artgor helped me.
# 
# I'm beginner, so there may be many strange point.
# Please give me a advise.
# 
# --------------------------------------------------------------
# 
# NNによるシンプルな回帰モデルを構築しました。
# https://www.kaggle.com/artgor　さんのkernelsを参考にさせていただきました。
# 
# 日本語で書かれたkernelsがほとんどなかったため
# 少しでも初心者の助けになればとこちらのkernelsを作成します。
# 
# なお、私自身も初心者ですので理解のできていない点が多くあるかと思います。
# ご指摘やアドバイスがあれば是非お願いします。

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
from tqdm import tqdm_notebook
from sklearn.preprocessing import StandardScaler
from sklearn.svm import NuSVR, SVR
from sklearn.metrics import mean_absolute_error
pd.options.display.precision = 15

import lightgbm as lgb
import xgboost as xgb
import time
import datetime

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, KFold, RepeatedKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import Ridge, RidgeCV
import gc
from catboost import CatBoostRegressor
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))


# Loading Data.
# 
# -------------------
# 
# データの読み込み。

# In[ ]:


X = pd.read_csv("../input/train.csv", nrows = 600000000, dtype={'acoustic_data': np.int16, 'time_to_failure': np.float64})

# Making Features.
# 
# -----------------------
# 
# 特徴量の作成。

# In[ ]:



rows = 150_000
train = X
segments = int(np.floor(train.shape[0] / rows))

X_tr = pd.DataFrame(index=range(segments), dtype=np.float64,
                       columns=['ave', 'std', 'max', 'min',
                               'av_change_abs', 'av_change_rate', 'abs_max', 'abs_min',
                               'std_first_50000', 'std_last_50000', 'std_first_10000', 'std_last_10000',
                               'avg_first_50000', 'avg_last_50000', 'avg_first_10000', 'avg_last_10000',
                               'min_first_50000', 'min_last_50000', 'min_first_10000', 'min_last_10000',
                               'max_first_50000', 'max_last_50000', 'max_first_10000', 'max_last_10000'])
y_tr = pd.DataFrame(index=range(segments), dtype=np.float64,
                       columns=['time_to_failure'])

total_mean = train['acoustic_data'].mean()
total_std = train['acoustic_data'].std()
total_max = train['acoustic_data'].max()
total_min = train['acoustic_data'].min()
total_sum = train['acoustic_data'].sum()
total_abs_max = np.abs(train['acoustic_data']).sum()

for segment in tqdm_notebook(range(segments)):
    seg = train.iloc[segment*rows:segment*rows+rows]
    x = seg['acoustic_data'].values
    y = seg['time_to_failure'].values[-1]
    
    y_tr.loc[segment, 'time_to_failure'] = y
    X_tr.loc[segment, 'ave'] = x.mean()
    X_tr.loc[segment, 'std'] = x.std()
    X_tr.loc[segment, 'max'] = x.max()
    X_tr.loc[segment, 'min'] = x.min()
    
    
    X_tr.loc[segment, 'av_change_abs'] = np.mean(np.diff(x))
    X_tr.loc[segment, 'av_change_rate'] = np.mean(np.nonzero((np.diff(x) / x[:-1]))[0])
    X_tr.loc[segment, 'abs_max'] = np.abs(x).max()
    X_tr.loc[segment, 'abs_min'] = np.abs(x).min()
    
    X_tr.loc[segment, 'std_first_50000'] = x[:50000].std()
    X_tr.loc[segment, 'std_last_50000'] = x[-50000:].std()
    X_tr.loc[segment, 'std_first_10000'] = x[:10000].std()
    X_tr.loc[segment, 'std_last_10000'] = x[-10000:].std()
    
    X_tr.loc[segment, 'avg_first_50000'] = x[:50000].mean()
    X_tr.loc[segment, 'avg_last_50000'] = x[-50000:].mean()
    X_tr.loc[segment, 'avg_first_10000'] = x[:10000].mean()
    X_tr.loc[segment, 'avg_last_10000'] = x[-10000:].mean()
    
    X_tr.loc[segment, 'min_first_50000'] = x[:50000].min()
    X_tr.loc[segment, 'min_last_50000'] = x[-50000:].min()
    X_tr.loc[segment, 'min_first_10000'] = x[:10000].min()
    X_tr.loc[segment, 'min_last_10000'] = x[-10000:].min()
    
    X_tr.loc[segment, 'max_first_50000'] = x[:50000].max()
    X_tr.loc[segment, 'max_last_50000'] = x[-50000:].max()
    X_tr.loc[segment, 'max_first_10000'] = x[:10000].max()
    X_tr.loc[segment, 'max_last_10000'] = x[-10000:].max()

# Normalize Features.
# 
# ------------------------
# 
# 作成した特徴量の正規化。

# In[ ]:


scaler = StandardScaler()
scaler.fit(X_tr)
X_train_scaled = pd.DataFrame(scaler.transform(X_tr), columns=X_tr.columns)

# Making data to predict.
# 
# --------------------------------
# 
# 訓練データの作成。

# In[ ]:


submission = pd.read_csv('../input/sample_submission.csv', index_col='seg_id')
X_test = pd.DataFrame(columns=X_tr.columns, dtype=np.float64, index=submission.index)
plt.figure(figsize=(22, 16))

for i, seg_id in enumerate(tqdm_notebook(X_test.index)):
    seg = pd.read_csv('../input/test/' + seg_id + '.csv')
    
    x = seg['acoustic_data'].values
    X_test.loc[seg_id, 'ave'] = x.mean()
    X_test.loc[seg_id, 'std'] = x.std()
    X_test.loc[seg_id, 'max'] = x.max()
    X_test.loc[seg_id, 'min'] = x.min()
        
    X_test.loc[seg_id, 'av_change_abs'] = np.mean(np.diff(x))
    X_test.loc[seg_id, 'av_change_rate'] = np.mean(np.nonzero((np.diff(x) / x[:-1]))[0])
    X_test.loc[seg_id, 'abs_max'] = np.abs(x).max()
    X_test.loc[seg_id, 'abs_min'] = np.abs(x).min()
    
    X_test.loc[seg_id, 'std_first_50000'] = x[:50000].std()
    X_test.loc[seg_id, 'std_last_50000'] = x[-50000:].std()
    X_test.loc[seg_id, 'std_first_10000'] = x[:10000].std()
    X_test.loc[seg_id, 'std_last_10000'] = x[-10000:].std()
    
    X_test.loc[seg_id, 'avg_first_50000'] = x[:50000].mean()
    X_test.loc[seg_id, 'avg_last_50000'] = x[-50000:].mean()
    X_test.loc[seg_id, 'avg_first_10000'] = x[:10000].mean()
    X_test.loc[seg_id, 'avg_last_10000'] = x[-10000:].mean()
    
    X_test.loc[seg_id, 'min_first_50000'] = x[:50000].min()
    X_test.loc[seg_id, 'min_last_50000'] = x[-50000:].min()
    X_test.loc[seg_id, 'min_first_10000'] = x[:10000].min()
    X_test.loc[seg_id, 'min_last_10000'] = x[-10000:].min()
    
    X_test.loc[seg_id, 'max_first_50000'] = x[:50000].max()
    X_test.loc[seg_id, 'max_last_50000'] = x[-50000:].max()
    X_test.loc[seg_id, 'max_first_10000'] = x[:10000].max()
    X_test.loc[seg_id, 'max_last_10000'] = x[-10000:].max()
    
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

# Trainig and Predict by NN.
# 
# --------------------------------------
# 
# NNによる学習および予測。

# In[ ]:


X_test_scaled.shape

# In[ ]:


from keras.layers import Input, Dense
from keras.models import Model
import tensorflow as tf

# This returns a tensor
inputs = Input(shape=(24,))

# a layer instance is callable on a tensor, and returns a tensor
x = Dense(64, activation='relu')(inputs)
x = Dense(64, activation='relu')(x)
predictions = Dense(1)(x)

# This creates a model that includes
# the Input layer and three Dense layers
model = Model(inputs=inputs, outputs=predictions)
model.compile(optimizer=tf.train.RMSPropOptimizer(0.001),
              loss='mse',
              metrics=['accuracy'])
model.fit(X_train_scaled, y_tr,epochs=10)  # starts training

# In[ ]:


y_pred_nn = model.predict(X_test_scaled).flatten()

# Making data to submit.
# 
# ---------------------------
# 
# 提出用データの作成。

# In[ ]:


sample = pd.read_csv("../input/sample_submission.csv")
sample['time_to_failure'] = y_pred_nn
sample.to_csv('submission.csv',index=False)

# Cross Validation.
# 
# ---------------------------
# 
# 交差検証。

# In[ ]:



from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss, roc_auc_score

n_estimators = [5,10,15]
functions = ['softmax', 'tanh', 'sigmoid', 'relu']
for funcs in functions:
    for n_est in  n_estimators:
        print(n_est)
        print(funcs)
        cv = KFold(n_splits=5, shuffle=True,random_state=0)
        for train, valid in cv.split(X_train_scaled, y_tr):
            x_train = X_train_scaled.iloc[train]
            x_valid = X_train_scaled.iloc[valid]
            y_train = y_tr.iloc[train]
            y_valid = y_tr.iloc[valid]
        
            inputs = Input(shape=(24,))
            x = Dense(64, activation=funcs)(inputs)
            x = Dense(64, activation=funcs)(x)
            predictions = Dense(1)(x)
            model = Model(inputs=inputs, outputs=predictions)
            model.compile(optimizer=tf.train.RMSPropOptimizer(0.001),
                      loss='mse',
                    metrics=['accuracy'])
            model.fit(x_train, y_train, epochs=n_est, verbose=0)
            y_pred_nn = model.predict(x_valid)
            y_pred = y_pred_nn
            print(mean_absolute_error(y_valid, y_pred))  
