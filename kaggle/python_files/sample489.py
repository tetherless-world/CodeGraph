#!/usr/bin/env python
# coding: utf-8

# In[97]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
from fastai.tabular import *
from fastai.callbacks import EarlyStoppingCallback
# Any results you write to the current directory are saved as output.

# In[98]:


root = Path("../input")
train_df = pd.read_csv(root/'train.csv')
test_df = pd.read_csv(root/'test.csv')
submission = pd.read_csv(root/'sample_submission.csv')

# In[99]:


train_df.head()

# In[100]:


train_df.describe()

# In[101]:


train_df.info()

# In[102]:


test_df.head()

# In[103]:


test_df.describe()

# In[104]:


test_df.info()

# In[105]:


submission.head()

# ## preprocessing

# In[106]:


procs = [FillMissing, Categorify, Normalize]

# ## split validation

# In[107]:


valid_idx = range(round(len(train_df) * 0.9), len(train_df))

# In[108]:


train_df['wheezy-copper-turtle-magic'].value_counts()

# In[109]:


all_cols = test_df.columns
dep_var = 'target'
cat_names = ['wheezy-copper-turtle-magic']
cont_names = list(set(all_cols) - set(['id', 'wheezy-copper-turtle-magic']))

# In[110]:


test = TabularList.from_df(test_df, cat_names=cat_names, cont_names=cont_names)

# In[111]:


data = (TabularList.from_df(train_df, cat_names=cat_names, cont_names=cont_names, procs=procs)
        .split_by_idx(valid_idx)
        .label_from_df(cols=dep_var)
        .add_test(test)
        .databunch(path='.', device=torch.device('cuda: 0'))
       )

# In[112]:


data.show_batch(rows=5)

# ## Define model

# In[113]:


len(data.test_ds.cont_names)

# In[174]:


learn = tabular_learner(data, layers=[512, 128], ps=[0.003, 0.001], metrics=accuracy, emb_drop=0.001,
                        callback_fns=[partial(EarlyStoppingCallback, monitor='accuracy', min_delta=0.0001, patience=5)]).to_fp16()

# In[175]:


learn.model

# In[176]:


learn.opt_func

# In[177]:


learn.loss_func

# ## Train model

# In[178]:


learn.lr_find()
learn.recorder.plot()

# In[179]:


lr = 1e-2

# In[180]:


learn.fit_one_cycle(100, lr)

# In[ ]:


learn.recorder.plot_losses()

# ## Predict

# In[ ]:


preds, _ = learn.get_preds(ds_type=DatasetType.Test)

# In[ ]:


submission['target'] = preds[:, 1].numpy()

# In[ ]:


submission.head()

# ## Save

# In[ ]:


submission.to_csv('submission.csv', index=None, encoding='utf-8')

# ## LGB

# In[ ]:


# from lightgbm import LGBMClassifier
# from sklearn.datasets import load_digits
# from sklearn.metrics import accuracy_score
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler

# data, target = train_df[cont_names].values, train_df['target'].values
# X_train, X_valid, y_train, y_valid = train_test_split(data, target, test_size=0.1)
# X_test = test_df[cont_names].values
# scaler = StandardScaler().fit(X_train)
# X_train = scaler.transform(X_train)
# X_test = scaler.transform(X_test)
# X_valid = scaler.transform(X_valid)

# params = {
#     'objective': 'binary',
#     'num_iterations': 1000,
#     'num_leaves': 10,
#     'learning_rate': 0.003,
#     'metric': ['auc', 'binary_logloss']
# }
# gbm = LGBMClassifier(**params)

# # 训练
# gbm.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], eval_metric='binary_logloss', early_stopping_rounds=15)

# # predict
# y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration_)
# print(f'Best iterations: {gbm.best_iteration_}')

# In[ ]:



