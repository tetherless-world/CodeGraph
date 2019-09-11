#!/usr/bin/env python
# coding: utf-8

# # lets get some instant gratification
# Overview of competition:
# - First kaggle competition to use "Synchronous KO" which allows for running kernels through public and private LB data prior to the deadline.
# - While this may seem like "instant gratification" it does require that the kernel runs first for the initial commit, and then again when submitting for a LB score- doubling the usual runtime before you can see a public LB score.
# - This does remove the need for a 2-stage competition and at the compeition deadline the final results can be revealed with no delay.
# - It's also important to note from the rules that "submissions that result in an error--either within the kernel or within the process of scoring--will count against your daily submission limit and will not return the specific error message."

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import os
import matplotlib.pylab as plt
print(os.listdir("../input"))

# In[ ]:


train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')

y_train = train_df['target'].copy()
id_train = train_df['id'].copy()
X_train = train_df.drop(['target', 'id'], axis=1)
id_text = test_df['id'].copy()
X_test = test_df.drop(['id'], axis=1)

# ## One of these things is not like the others...
# - When we display summary statistics of each feature we notice that one feature `wheezy-copper-turtle-magic` stands out.

# In[ ]:


train_df['wheezy-copper-turtle-magic'].plot(kind='hist', bins=500, figsize=(15, 5), title='Distribution of Feature wheezy-copper-turtle-magic')
plt.show()

# In[ ]:


# This feature is more categorical than continious
train_df['wheezy-copper-turtle-magic'] = train_df['wheezy-copper-turtle-magic'].astype('category')
test_df['wheezy-copper-turtle-magic'] = test_df['wheezy-copper-turtle-magic'].astype('category')
X_train['wheezy-copper-turtle-magic'] = X_train['wheezy-copper-turtle-magic'].astype('category')
X_test['wheezy-copper-turtle-magic'] = X_test['wheezy-copper-turtle-magic'].astype('category')

# # Lets look at some summary statistics of features
# - Removing `wheezy-copper-turtle-magic` from this analysis

# In[ ]:


cmap = cmap=sns.diverging_palette(5, 250, as_cmap=True)

train_df.drop(['target', 'wheezy-copper-turtle-magic'], axis=1). \
    describe().T\
    .sort_values('mean', ascending=False)\
    .drop('count', axis=1)\
    .T.style.background_gradient(cmap, axis=1)\
    .set_precision(2)

# In[ ]:


test_df.drop(['wheezy-copper-turtle-magic'], axis=1). \
    describe().T\
    .sort_values('mean', ascending=False)\
    .drop('count', axis=1)\
    .T.style.background_gradient(cmap, axis=1)\
    .set_precision(2)

# ## Target in the training set is almost 50/50 Split positive/negative

# In[ ]:


y_train.mean()

# ## 257 Features with some interesting names.
# - stealthy-beige-pinscher-golden?
# - nerdy-indigo-wolfhound-sorted?
# 
# This data looks simulated. And the names are funny but will just require a lot of useless typing..
# 
# If we want to we could rename the columns for those of us who worked on santander... :D

# In[ ]:


X_train.columns = ['var_{}'.format(x) for x in range(0, 256)]
X_test.columns = ['var_{}'.format(x) for x in range(0, 256)]

# In[ ]:


average_of_feat = train_df.groupby('target').agg(['mean']).T.reset_index().rename(columns={'level_0':'feature'}).drop('level_1', axis=1)

# In[ ]:


average_of_feat['pos_neg_diff'] = np.abs(average_of_feat[0] - average_of_feat[1])
average_of_feat.sort_values('pos_neg_diff', ascending=True) \
    .tail(20).set_index('feature')['pos_neg_diff'].plot(kind='barh',
                                                        title='Top 20 feature with biggest difference in mean between positive and negative class',
                                                       figsize=(15, 7),
                                                       color='grey')
plt.show()

# ## Plot positive vs negative feature distributions
# ..pretty boring - or is it?

# In[ ]:


fig, axes = plt.subplots(10, 2, figsize=(20, 30))
top20_diff = average_of_feat.sort_values('pos_neg_diff', ascending=True).tail(20)['feature'].values
ax_position = 0
for var in top20_diff:
    if var not in ['target','id']:
        for i, d in train_df.groupby('target'):
            d[var].plot(kind='hist', bins=100, alpha=0.5, title=var, label='target={}'.format(i), ax=axes.flat[ax_position])
        axes.flat[ax_position].legend()
        ax_position += 1
plt.show()

# # Baseline LightGBM Model

# In[ ]:


from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

param = {
    'bagging_freq': 3,
    'bagging_fraction': 0.8,
    'boost_from_average':'false',
    'boost': 'gbdt',
    'feature_fraction': 0.9,
    'learning_rate': 0.01,
    'max_depth': 8,  
    'metric':'auc',
    'min_data_in_leaf': 82,
    'min_sum_hessian_in_leaf': 10.0,
    'num_leaves': 10,
    'objective': 'binary', 
    'verbosity': 1
}
N_FOLDS = 5
folds = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=529)
oof = np.zeros(len(X_train))
predictions = np.zeros(len(X_test))
feature_importance_df = pd.DataFrame()

for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train, y_train)):
    print("Fold {}".format(fold_))
    trn_data = lgb.Dataset(X_train.iloc[trn_idx], label=y_train.iloc[trn_idx])
    val_data = lgb.Dataset(X_train.iloc[val_idx], label=y_train.iloc[val_idx])

    num_round = 1000000
    clf = lgb.train(param, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=500, early_stopping_rounds = 300)
    oof[val_idx] = clf.predict(X_train.iloc[val_idx], num_iteration=clf.best_iteration)
    
    fold_importance_df = pd.DataFrame()
    fold_importance_df["Feature"] = X_train.columns
    fold_importance_df["importance"] = clf.feature_importance()
    fold_importance_df["fold"] = fold_ + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
    
    predictions += clf.predict(X_test, num_iteration=clf.best_iteration) / folds.n_splits

print("CV score: {:<8.5f}".format(roc_auc_score(y_train, oof)))

# # Save The Results

# In[ ]:


ss = pd.read_csv('../input/sample_submission.csv')
ss['target'] = predictions

# In[ ]:


from datetime import datetime
run_id = "{:%m%d_%H%M}".format(datetime.now())

# In[ ]:


# Save Submission
submission_csv = 'submission_{:0.2f}CV_{}Folds_{}.csv'.format(roc_auc_score(y_train, oof), N_FOLDS, run_id)
print('Saving submission as {}'.format(submission_csv))
ss.to_csv(submission_csv, index=False)
ss.to_csv('submission.csv', index=False)
# Save Feature Importance
feature_importance_csv = 'fi_{:0.2f}CV_{}Folds_{}.csv'.format(roc_auc_score(y_train, oof), N_FOLDS, run_id)
print('Saving feature importance as {}'.format(feature_importance_csv))
feature_importance_df.to_csv(feature_importance_csv, index=False)

# Save OOF
oof_df = pd.DataFrame()
oof_df['oof'] = oof
oof_df['id'] = id_train
oof_df['target'] = y_train
oof_csv = 'oof_{:0.2f}CV_{}Folds_{}.csv'.format(roc_auc_score(y_train, oof), N_FOLDS, run_id)
print('Saving out-of-fold predictions as {}'.format(oof_csv))
oof_df.to_csv(oof_csv, index=False)
