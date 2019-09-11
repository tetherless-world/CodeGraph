#!/usr/bin/env python
# coding: utf-8

# ## Overview
# 
# The purpose of this kernel is to take a look at the data, come up with some insights, and attempt to create a predictive model or two. So let's get started!
# 
# ## Packages
# 
# First, let's load a few useful Python packages. This section will keep growing in subsequent versions of this EDA.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import operator
#import gc
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import describe

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import xgboost as xgb

# Now let us look at the input folder. Here we find all the relevant files for this competition.

# In[ ]:


print(os.listdir("../input"))

# We see that the input folder only contains three files ```train.csv```, ```test.csv```, and ```sample_submission.csv```. It seems that for this competition we don't have to do any complicated combination and mergers of files.
# 
# Now let's import and take a glimpse at these files.

# In[ ]:


train_df = pd.read_csv('../input/train.csv')
train_df.head()

# In[ ]:


train_df.shape

# In[ ]:


train_df.info()

# In[ ]:


train_df.isnull().values.any()

# A few things immediately stand out:
# 
# 1. As advertised, features are numeric and anonymized. 
# 2. Features seem sparse. We'll have to investigate this further. 
# 3. There are a LOT of features! Almost 5000! And they outnumber the number of rows in the training set!
# 4. There are less than 5000 training rows. In fact, there are fewer rows than columns, which means we'll have to invest a lot of effort into feature selection / feature engineering. 
# 5. The memory size of the train dataset is fairly large - 170 MB, which is to be expected. 
# 6. Pandas is treating 1845 features as float, and 3147 as integer. It is possible that some of those int features are one-hot-encoded or label-encoded categorical variables. We'll have to investigate this later, and possibly do some reverse-engineering. :)
# 7. There doesn't appear to be any missing values in the train set. This is, IMHO, overall a good thing, although a lot of times there is some signal in the missing values that's valuable and worth exploring. 
# 
# This is going to be a very, very, interesting competition. :)
# 
# Now let's take a look at the test dataset.

# In[ ]:


test_df = pd.read_csv('../input/test.csv')
test_df.head()

# In[ ]:


test_df.shape

# In[ ]:


test_df.info()

# In[ ]:


test_df.isnull().values.any()

# Here we see that the number of features in the test set (4992) matches the number in the train set. Sanity check is always a good thing, and at leas at this level, hte Kaggle people did not mess things up.
# 
# According to Pandas, there are no ```int``` values in the test set. Now I'm really curious about those ... 
# 
# There also doesn't seem to be any missing values in the test set. 
# 
# We also see that the number of rows in the test set far surpasses the number of rows in the train set. Yes, a very *interesting* competition indeed ...
# 
# 
# 
# Now let's see some basic descriptive statistics for the train and test dataframes.

# In[ ]:


train_df_describe = train_df.describe()
train_df_describe

# A few things to notice:
# 
# 1.  Target variable ranges over 4 orders of magnitude. (factor of 10,000)
# 2. Most features have 0.0 for 75% - another indication that we are probably dealing with sparse data.
# 3. Most features seem to have similarly wide spread of values as the target variable. Hmm, interesting ...
# 4. The standard deviation for most features seems larger than the feature mean. 
# 5.. There are a few features (such as ```d5308d8bc```, ```c330f1a67```) that seem to be filled with zeros. These will need to be eliminated.
# 
# Now let's look at the test set. 

# In[ ]:


test_df_describe = test_df.describe()
test_df_describe

# We see a similar distribution of various statistical aggregates, but by no means the same: seems like there soem substantial distribution shifts between the train and test sets. This will probably be another major concern when it comes to feature selection/engineering. 
# 
# Now let's do some plotting. We'll take a look at, naturally, the ```target``` variable. First, let's make a histogram of its raw value.

# In[ ]:


plt.figure(figsize=(12, 5))
plt.hist(train_df.target.values, bins=100)
plt.title('Histogram target counts')
plt.xlabel('Count')
plt.ylabel('Target')
plt.show()

# This is a highly skewed distribution, so let's try to re-plot it with with log transform of the target.

# In[ ]:


plt.figure(figsize=(12, 5))
plt.hist(np.log(1+train_df.target.values), bins=100)
plt.title('Histogram target counts')
plt.xlabel('Count')
plt.ylabel('Log 1+Target')
plt.show()

# As expected, this distribution looks much more, ahem, normal. This is probably one of the main reasons why the metric that we are trying to optimize for this competition is RMSLE - root mean square logarithmic error.
# 
# Another way of looking at the same distribution is with the help of violinplot.

# In[ ]:


sns.set_style("whitegrid")
ax = sns.violinplot(x=np.log(1+train_df.target.values))
plt.show()

# That's ... revealing. And it looks like a fairly nice distribution, albeit still fairly asymetrical.
# 
# Let's take a look at the statistics of the Log(1+target)

# In[ ]:


train_log_target = train_df[['target']]
train_log_target['target'] = np.log(1+train_df['target'].values)
train_log_target.describe()

# We see that the statistical properties of teh Log(1+Target) distribution are much more amenable.
# 
# Now let's take a look at columns with constant value.

# In[ ]:


constant_train = train_df.loc[:, (train_df == train_df.iloc[0]).all()].columns.tolist()
constant_test = test_df.loc[:, (test_df == test_df.iloc[0]).all()].columns.tolist()

# In[ ]:


print('Number of constant columns in the train set:', len(constant_train))
print('Number of constant columns in the test set:', len(constant_test))

# So this is interesting: there are 256 constant columns in the train set, but none in the test set. These constant columns are thus most likely an artifact of the way that the train and test sets were constructed, and not necessarily irrelevant in their own right. This is yet another byproduct of having a very small dataset. For most problems it would be useful to take a look at the description of these columns, but in this competition they are anonymized, and thus would not yield any useful information. 
# 
# So let's subset the colums that we'd use to just those that are not constant.

# In[ ]:


columns_to_use = test_df.columns.tolist()
del columns_to_use[0] # Remove 'ID'
columns_to_use = [x for x in columns_to_use if x not in constant_train] #Remove all 0 columns
len(columns_to_use)

# So we have the total of 4735 columns to work with. However, as mentioned earlier, most of these columns seem to be filled predominatly with zeros. Let's try to get a better sense of this data.

# In[ ]:


describe(train_df[columns_to_use].values, axis=None)

# If we treat all the train matrix values as if they belonged to a single row vector, we see a huge amount of varience, far exceeding the similar variance for the target variable.
# 
# Now let's plot it to see how diverse the numerical values are.

# In[ ]:


plt.figure(figsize=(12, 5))
plt.hist(train_df[columns_to_use].values.flatten(), bins=50)
plt.title('Histogram all train counts')
plt.xlabel('Count')
plt.ylabel('Value')
plt.show()

# Wow, not very diverse at all! Most of the values are heavily concentrated around 0. 
# 
# Maybe if we used the log plot things would be better.

# In[ ]:


plt.figure(figsize=(12, 5))
plt.hist(np.log(train_df[columns_to_use].values.flatten()+1), bins=50)
plt.title('Log Histogram all train counts')
plt.xlabel('Count')
plt.ylabel('Log value')
plt.show()

# Only marginal improvement - there is a verly small bump close to 15.
# 
# Can the violin plot help?

# In[ ]:


sns.set_style("whitegrid")
ax = sns.violinplot(x=np.log(train_df[columns_to_use].values.flatten()+1))
plt.show()

# Not really - the plot looks nicer, but the overall shape is pretty much the same. 
# 
# OK, let's take a look at the distribution of non-zero values.

# In[ ]:


train_nz = np.log(train_df[columns_to_use].values.flatten()+1)
train_nz = train_nz[np.nonzero(train_nz)]
plt.figure(figsize=(12, 5))
plt.hist(train_nz, bins=50)
plt.title('Log Histogram nonzero train counts')
plt.xlabel('Count')
plt.ylabel('Log value')
plt.show()

# In[ ]:


sns.set_style("whitegrid")
ax = sns.violinplot(x=train_nz)
plt.show()

# In[ ]:


describe(train_nz)

# OK, that's much more interesting. 
# 
# Let's do the same thing with the test data.

# In[ ]:


test_nz = np.log(test_df[columns_to_use].values.flatten()+1)
test_nz = test_nz[np.nonzero(test_nz)]
plt.figure(figsize=(12, 5))
plt.hist(test_nz, bins=50)
plt.title('Log Histogram nonzero test counts')
plt.xlabel('Count')
plt.ylabel('Log value')
plt.show()

# In[ ]:


sns.set_style("whitegrid")
ax = sns.violinplot(x=test_nz)
plt.show()

# In[ ]:


describe(test_nz)

# Again, we see that these distributions look similar, but they are definitely not the same. 
# 
# Now let's take a closer look at the shape and content of the train data. We want to get a better numerical grasp of the true extent of zeros.

# In[ ]:


train_df[columns_to_use].values.flatten().shape

# In[ ]:


((train_df[columns_to_use].values.flatten())==0).mean()

# So as we suspected, almost 97% of all values in the train dataframe are zeros. That looks pretty sparse to me, but let's see how much variation is there between different columns.

# In[ ]:


train_zeros = pd.DataFrame({'Percentile':((train_df[columns_to_use].values)==0).mean(axis=0),
                           'Column' : columns_to_use})
train_zeros.head()

# In[ ]:


describe(train_zeros.Percentile.values)

# So it seems that the vast majority of columns have 95+ percent of zeros in them. Let's see how would that look on a plot.

# In[ ]:


plt.figure(figsize=(12, 5))
plt.hist(train_zeros.Percentile.values, bins=50)
plt.title('Histogram percentage zeros train counts')
plt.xlabel('Count')
plt.ylabel('Value')
plt.show()

# In[ ]:


describe(np.log(train_df[columns_to_use].values+1), axis=None)

# In[ ]:


describe(test_df[columns_to_use].values, axis=None)

# In[ ]:


describe(np.log(test_df[columns_to_use].values+1), axis=None)

# In[ ]:


test_zeros = pd.DataFrame({'Percentile':(np.log(1+test_df[columns_to_use].values)==0).mean(axis=0),
                           'Column' : columns_to_use})
test_zeros.head()

# In[ ]:


describe(test_zeros.Percentile.values)

# OK, let's try to do some modeling. We'll start with a simple LighGBM regression, and see if that yields any results. First, let's set our target variable to be the log of 1 + target.

# In[ ]:


y = np.log(1+train_df.target.values)
y.shape

# In[ ]:


y

# In[ ]:


train = lgb.Dataset(train_df[columns_to_use],y ,feature_name = "auto")

# In[ ]:


params = {'boosting_type': 'gbdt', 
          'objective': 'regression', 
          'metric': 'rmse', 
          'learning_rate': 0.01, 
          'num_leaves': 100, 
          'feature_fraction': 0.4, 
          'bagging_fraction': 0.6, 
          'max_depth': 5, 
          'min_child_weight': 10}


clf = lgb.train(params,
        train,
        num_boost_round = 400,
        verbose_eval=True)

# In[ ]:



preds = clf.predict(test_df[columns_to_use])
preds

# In[ ]:


sample_submission = pd.read_csv("../input/sample_submission.csv")
sample_submission.target = np.exp(preds)-1
sample_submission.to_csv('simple_lgbm_1.csv', index=False)
sample_submission.head()

# Well, that's great - we made a prediction on the test set, and saved it to a file, which we were able to submit to the competition. Unfortunately, there was no way to tell how this model would perform on the unseen data. (This submission scored 1.53 on Public Leaderboard.)

# In[ ]:


nr_splits = 5
random_state = 1054

y_oof = np.zeros((y.shape[0]))
total_preds = 0


kf = KFold(n_splits=nr_splits, shuffle=True, random_state=random_state)
for i, (train_index, val_index) in enumerate(kf.split(y)):
    print('Fitting fold', i+1, 'out of', nr_splits)
    X_train, X_val  = train_df[columns_to_use].iloc[train_index], train_df[columns_to_use].iloc[val_index]
    y_train, y_val = y[train_index], y[val_index]
    
    train = lgb.Dataset(X_train,y_train ,feature_name = "auto")
    val = lgb.Dataset(X_val ,y_val ,feature_name = "auto")
    clf = lgb.train(params,train,num_boost_round = 400,verbose_eval=True)
    
    total_preds += clf.predict(test_df[columns_to_use])/nr_splits
    pred_oof = clf.predict(X_val)
    y_oof[val_index] = pred_oof
    print('Fold error', np.sqrt(mean_squared_error(y_val, pred_oof)))

print('Total error', np.sqrt(mean_squared_error(y, y_oof)))

# In[ ]:


params['max_depth'] = 4

y_oof_2 = np.zeros((y.shape[0]))
total_preds_2 = 0


kf = KFold(n_splits=nr_splits, shuffle=True, random_state=random_state)
for i, (train_index, val_index) in enumerate(kf.split(y)):
    print('Fitting fold', i+1, 'out of', nr_splits)
    X_train, X_val  = train_df[columns_to_use].iloc[train_index], train_df[columns_to_use].iloc[val_index]
    y_train, y_val = y[train_index], y[val_index]
    
    train = lgb.Dataset(X_train,y_train ,feature_name = "auto")
    val = lgb.Dataset(X_val ,y_val ,feature_name = "auto")
    clf = lgb.train(params,train,num_boost_round = 400,verbose_eval=True)
    
    total_preds_2 += clf.predict(test_df[columns_to_use])/nr_splits
    pred_oof = clf.predict(X_val)
    y_oof_2[val_index] = pred_oof
    print('Fold error', np.sqrt(mean_squared_error(y_val, pred_oof)))

print('Total error', np.sqrt(mean_squared_error(y, y_oof_2)))

# In[ ]:


params['max_depth'] = 6

y_oof_3 = np.zeros((y.shape[0]))
total_preds_3 = 0


kf = KFold(n_splits=nr_splits, shuffle=True, random_state=random_state)
for i, (train_index, val_index) in enumerate(kf.split(y)):
    print('Fitting fold', i+1, 'out of', nr_splits)
    X_train, X_val  = train_df[columns_to_use].iloc[train_index], train_df[columns_to_use].iloc[val_index]
    y_train, y_val = y[train_index], y[val_index]
    
    train = lgb.Dataset(X_train,y_train ,feature_name = "auto")
    val = lgb.Dataset(X_val ,y_val ,feature_name = "auto")
    clf = lgb.train(params,train,num_boost_round = 400,verbose_eval=True)
    
    total_preds_3 += clf.predict(test_df[columns_to_use])/nr_splits
    pred_oof = clf.predict(X_val)
    y_oof_3[val_index] = pred_oof
    print('Fold error', np.sqrt(mean_squared_error(y_val, pred_oof)))

print('Total error', np.sqrt(mean_squared_error(y, y_oof_3)))

# In[ ]:


params['max_depth'] = 7

y_oof_4 = np.zeros((y.shape[0]))
total_preds_4 = 0


kf = KFold(n_splits=nr_splits, shuffle=True, random_state=random_state)
for i, (train_index, val_index) in enumerate(kf.split(y)):
    print('Fitting fold', i+1, 'out of', nr_splits)
    X_train, X_val  = train_df[columns_to_use].iloc[train_index], train_df[columns_to_use].iloc[val_index]
    y_train, y_val = y[train_index], y[val_index]
    
    train = lgb.Dataset(X_train,y_train ,feature_name = "auto")
    val = lgb.Dataset(X_val ,y_val ,feature_name = "auto")
    clf = lgb.train(params,train,num_boost_round = 400,verbose_eval=True)
    
    total_preds_4 += clf.predict(test_df[columns_to_use])/nr_splits
    pred_oof = clf.predict(X_val)
    y_oof_4[val_index] = pred_oof
    print('Fold error', np.sqrt(mean_squared_error(y_val, pred_oof)))

print('Total error', np.sqrt(mean_squared_error(y, y_oof_4)))

# In[ ]:


params['max_depth'] = 8

y_oof_5 = np.zeros((y.shape[0]))
total_preds_5 = 0


kf = KFold(n_splits=nr_splits, shuffle=True, random_state=random_state)
for i, (train_index, val_index) in enumerate(kf.split(y)):
    print('Fitting fold', i+1, 'out of', nr_splits)
    X_train, X_val  = train_df[columns_to_use].iloc[train_index], train_df[columns_to_use].iloc[val_index]
    y_train, y_val = y[train_index], y[val_index]
    
    train = lgb.Dataset(X_train,y_train ,feature_name = "auto")
    val = lgb.Dataset(X_val ,y_val ,feature_name = "auto")
    clf = lgb.train(params,train,num_boost_round = 400,verbose_eval=True)
    
    total_preds_5 += clf.predict(test_df[columns_to_use])/nr_splits
    pred_oof = clf.predict(X_val)
    y_oof_5[val_index] = pred_oof
    print('Fold error', np.sqrt(mean_squared_error(y_val, pred_oof)))

print('Total error', np.sqrt(mean_squared_error(y, y_oof_5)))

# In[ ]:


params['max_depth'] = 10

y_oof_6 = np.zeros((y.shape[0]))
total_preds_6 = 0


kf = KFold(n_splits=nr_splits, shuffle=True, random_state=random_state)
for i, (train_index, val_index) in enumerate(kf.split(y)):
    print('Fitting fold', i+1, 'out of', nr_splits)
    X_train, X_val  = train_df[columns_to_use].iloc[train_index], train_df[columns_to_use].iloc[val_index]
    y_train, y_val = y[train_index], y[val_index]
    
    train = lgb.Dataset(X_train,y_train ,feature_name = "auto")
    val = lgb.Dataset(X_val ,y_val ,feature_name = "auto")
    clf = lgb.train(params,train,num_boost_round = 400,verbose_eval=True)
    
    total_preds_6 += clf.predict(test_df[columns_to_use])/nr_splits
    pred_oof = clf.predict(X_val)
    y_oof_6[val_index] = pred_oof
    print('Fold error', np.sqrt(mean_squared_error(y_val, pred_oof)))

print('Total error', np.sqrt(mean_squared_error(y, y_oof_6)))

# In[ ]:


params['max_depth'] = 12

y_oof_7 = np.zeros((y.shape[0]))
total_preds_7 = 0


kf = KFold(n_splits=nr_splits, shuffle=True, random_state=random_state)
for i, (train_index, val_index) in enumerate(kf.split(y)):
    print('Fitting fold', i+1, 'out of', nr_splits)
    X_train, X_val  = train_df[columns_to_use].iloc[train_index], train_df[columns_to_use].iloc[val_index]
    y_train, y_val = y[train_index], y[val_index]
    
    train = lgb.Dataset(X_train,y_train ,feature_name = "auto")
    val = lgb.Dataset(X_val ,y_val ,feature_name = "auto")
    clf = lgb.train(params,train,num_boost_round = 400,verbose_eval=True)
    
    total_preds_7 += clf.predict(test_df[columns_to_use])/nr_splits
    pred_oof = clf.predict(X_val)
    y_oof_7[val_index] = pred_oof
    print('Fold error', np.sqrt(mean_squared_error(y_val, pred_oof)))

print('Total error', np.sqrt(mean_squared_error(y, y_oof_7)))

# In[ ]:


print('Total error', np.sqrt(mean_squared_error(y, 1.4*(1.6*y_oof_7-0.6*y_oof_6)-0.4*y_oof_5)))

# In[ ]:


print('Total error', np.sqrt(mean_squared_error(y, -0.5*y_oof-0.5*y_oof_2-y_oof_3
                                                +3*y_oof_4)))

# In[ ]:


print('Total error', np.sqrt(mean_squared_error(y, 0.75*(1.4*(1.6*y_oof_7-0.6*y_oof_6)-0.4*y_oof_5)+
                                                0.25*(-0.5*y_oof-0.5*y_oof_2-y_oof_3
                                                +3*y_oof_4))))

# In[ ]:


sub_preds = (0.75*(1.4*(1.6*total_preds_7-0.6*total_preds_6)-0.4*total_preds_5)+
                                                0.25*(-0.5*total_preds-0.5*total_preds_2-total_preds_3
                                                +3*total_preds_4))
#sub_preds = (-0.5*total_preds-0.5*total_preds_2-total_preds_3+3*total_preds_4)
sample_submission.target = np.exp(sub_preds)-1
sample_submission.to_csv('blended_submission_2.csv', index=False)
sample_submission.head()

# In[ ]:


params = {'objective': 'reg:linear', 
          'eval_metric': 'rmse',
          'eta': 0.01,
          'max_depth': 10, 
          'subsample': 0.6, 
          'colsample_bytree': 0.6,
          'alpha':0.001,
          'random_state': 42, 
          'silent': True}

y_oof_8 = np.zeros((y.shape[0]))
total_preds_8 = 0

dtest = xgb.DMatrix(test_df[columns_to_use])

kf = KFold(n_splits=nr_splits, shuffle=True, random_state=random_state)
for i, (train_index, val_index) in enumerate(kf.split(y)):
    print('Fitting fold', i+1, 'out of', nr_splits)
    X_train, X_val  = train_df[columns_to_use].iloc[train_index], train_df[columns_to_use].iloc[val_index]
    y_train, y_val = y[train_index], y[val_index]
    
    
    train = xgb.DMatrix(X_train, y_train)
    val = xgb.DMatrix(X_val, y_val)
    
    watchlist = [(train, 'train'), (val, 'val')]
    
    clf = xgb.train(params, train, 1000, watchlist, 
                          maximize=False, early_stopping_rounds = 60, verbose_eval=100)

    
    total_preds_8 += clf.predict(dtest, ntree_limit=clf.best_ntree_limit)/nr_splits
    pred_oof = clf.predict(val, ntree_limit=clf.best_ntree_limit)
    y_oof_8[val_index] = pred_oof
    print('Fold error', np.sqrt(mean_squared_error(y_val, pred_oof)))

print('Total error', np.sqrt(mean_squared_error(y, y_oof_8)))

# ### To do:
# 
# 1. Meke some plots
# 2. Build a few models
# 3. Do feature importance analysis
# 
# ## To be continued ...

# In[ ]:


print('Total error', np.sqrt(mean_squared_error(y, 0.7*(0.75*(1.4*(1.6*y_oof_7-0.6*y_oof_6)-0.4*y_oof_5)+
                                                0.25*(-0.5*y_oof-0.5*y_oof_2-y_oof_3
                                                +3*y_oof_4))+0.3*y_oof_8)))

# In[ ]:


sub_preds = (0.7*(0.75*(1.4*(1.6*total_preds_7-0.6*total_preds_6)-0.4*total_preds_5)+
                                                0.25*(-0.5*total_preds-0.5*total_preds_2-total_preds_3
                                                +3*total_preds_4))+0.3*total_preds_8)
#sub_preds = (-0.5*total_preds-0.5*total_preds_2-total_preds_3+3*total_preds_4)
sample_submission.target = np.exp(sub_preds)-1
sample_submission.to_csv('blended_submission_3.csv', index=False)
sample_submission.head()

# In[ ]:


feature_importances = clf.get_fscore()

# In[ ]:


importance = sorted(feature_importances.items(), key=operator.itemgetter(1))

# In[ ]:


best_2500 = importance[::-1][:2500]

# In[ ]:


best_2500 =[ x[0] for x in best_2500]

# In[ ]:




# In[ ]:


params = {'objective': 'reg:linear', 
          'eval_metric': 'rmse',
          'eta': 0.01,
          'max_depth': 10, 
          'subsample': 0.6, 
          'colsample_bytree': 0.6,
          'alpha':0.001,
          'random_state': 42, 
          'silent': True}

y_oof_9 = np.zeros((y.shape[0]))
total_preds_9 = 0

dtest = xgb.DMatrix(test_df[best_2500])

kf = KFold(n_splits=nr_splits, shuffle=True, random_state=random_state)
for i, (train_index, val_index) in enumerate(kf.split(y)):
    print('Fitting fold', i+1, 'out of', nr_splits)
    X_train, X_val  = train_df[best_2500].iloc[train_index], train_df[best_2500].iloc[val_index]
    y_train, y_val = y[train_index], y[val_index]
    
    
    train = xgb.DMatrix(X_train, y_train)
    val = xgb.DMatrix(X_val, y_val)
    
    watchlist = [(train, 'train'), (val, 'val')]
    
    clf = xgb.train(params, train, 1500, watchlist, 
                          maximize=False, early_stopping_rounds = 60, verbose_eval=100)

    
    total_preds_9 += clf.predict(dtest, ntree_limit=clf.best_ntree_limit)/nr_splits
    pred_oof = clf.predict(val, ntree_limit=clf.best_ntree_limit)
    y_oof_9[val_index] = pred_oof
    print('Fold error', np.sqrt(mean_squared_error(y_val, pred_oof)))

print('Total error', np.sqrt(mean_squared_error(y, y_oof_9)))

# In[ ]:


print('Total error', np.sqrt(mean_squared_error(y, 0.5*y_oof_9+0.5*(0.7*(0.75*(1.4*(1.6*y_oof_7-0.6*y_oof_6)-0.4*y_oof_5)+
                                                0.25*(-0.5*y_oof-0.5*y_oof_2-y_oof_3
                                                +3*y_oof_4))+0.3*y_oof_8))))

sub_preds = 0.5*total_preds_9+0.5*(0.7*(0.75*(1.4*(1.6*total_preds_7-0.6*total_preds_6)-0.4*total_preds_5)+
                                                0.25*(-0.5*total_preds-0.5*total_preds_2-total_preds_3
                                                +3*total_preds_4))+0.3*total_preds_8)
#sub_preds = (-0.5*total_preds-0.5*total_preds_2-total_preds_3+3*total_preds_4)
sample_submission.target = np.exp(sub_preds)-1
sample_submission.to_csv('blended_submission_4.csv', index=False)
sample_submission.head()
