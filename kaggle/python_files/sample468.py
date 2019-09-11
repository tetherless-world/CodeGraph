#!/usr/bin/env python
# coding: utf-8

# # Bayesian global optimization with gaussian processes for finding (sub-)optimal parameters of LightGBM
# 
# As many of fellow kaggler asking how did I get LightGBM parameters for the kernel [Customer Transaction Prediction](https://www.kaggle.com/fayzur/customer-transaction-prediction) I published. So, I decided to publish a kernel to optimize parameters. 
# 
# 
# 
# In this kernel I use Bayesian global optimization with gaussian processes for finding optimal parameters. This optimization attempts to find the maximum value of an black box function in as few iterations as possible. In our case the black box function will be a function that I will write to optimize (maximize) the evaluation function (AUC) so that parameters get maximize AUC in training and validation, and expect to do good in the private. The final prediction will be **rank average on 5 fold cross validation predictions**.
# 
# Continue to the end of this kernel and **upvote it if you find it is interesting**.
# 
# ![image.jpg](https://i.imgur.com/XKS1oqU.jpg)
# 
# Image taken from : https://github.com/fmfn/BayesianOptimization

# ## Notebook  Content
# 0. [Installing Bayesian global optimization library](#0) <br>    
# 1. [Loading the data](#1)
# 2. [Black box function to be optimized (LightGBM)](#2)
# 3. [Training LightGBM model](#3)
# 4. [Rank averaging](#4)
# 5. [Submission](#5)

# <a id="0"></a> <br>
# ## 0. Installing Bayesian global optimization library
# 
# Let's install the latest release from pip

# In[ ]:



# <a id="1"></a> <br>
# ## 1. Loading the data

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from scipy.stats import rankdata
import lightgbm as lgb
from sklearn import metrics
import gc
import warnings

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


target = 'target'
predictors = train_df.columns.values.tolist()[2:]

# In[ ]:


train_df.target.value_counts()

# The problem is unbalanced! 

# In this kernel I will be using **50% Stratified rows** as holdout rows for the validation-set to get optimal parameters. Later I will use 5 fold cross validation in the final model fit.

# In[ ]:


bayesian_tr_index, bayesian_val_index  = list(StratifiedKFold(n_splits=2, shuffle=True, random_state=1).split(train_df, train_df.target.values))[0]

# These `bayesian_tr_index` and `bayesian_val_index` indexes will be used for the bayesian optimization as training and validation index of training dataset.

# <a id="2"></a> <br>
# ## 2. Black box function to be optimized (LightGBM)

# As data is loaded, let's create the black box function for LightGBM to find parameters.

# In[ ]:


def LGB_bayesian(
    num_leaves,  # int
    min_data_in_leaf,  # int
    learning_rate,
    min_sum_hessian_in_leaf,    # int  
    feature_fraction,
    lambda_l1,
    lambda_l2,
    min_gain_to_split,
    max_depth):
    
    # LightGBM expects next three parameters need to be integer. So we make them integer
    num_leaves = int(num_leaves)
    min_data_in_leaf = int(min_data_in_leaf)
    max_depth = int(max_depth)

    assert type(num_leaves) == int
    assert type(min_data_in_leaf) == int
    assert type(max_depth) == int

    param = {
        'num_leaves': num_leaves,
        'max_bin': 63,
        'min_data_in_leaf': min_data_in_leaf,
        'learning_rate': learning_rate,
        'min_sum_hessian_in_leaf': min_sum_hessian_in_leaf,
        'bagging_fraction': 1.0,
        'bagging_freq': 5,
        'feature_fraction': feature_fraction,
        'lambda_l1': lambda_l1,
        'lambda_l2': lambda_l2,
        'min_gain_to_split': min_gain_to_split,
        'max_depth': max_depth,
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
        'boost_from_average': False,   

    }    
    
    
    xg_train = lgb.Dataset(train_df.iloc[bayesian_tr_index][predictors].values,
                           label=train_df.iloc[bayesian_tr_index][target].values,
                           feature_name=predictors,
                           free_raw_data = False
                           )
    xg_valid = lgb.Dataset(train_df.iloc[bayesian_val_index][predictors].values,
                           label=train_df.iloc[bayesian_val_index][target].values,
                           feature_name=predictors,
                           free_raw_data = False
                           )   

    num_round = 5000
    clf = lgb.train(param, xg_train, num_round, valid_sets = [xg_valid], verbose_eval=250, early_stopping_rounds = 50)
    
    predictions = clf.predict(train_df.iloc[bayesian_val_index][predictors].values, num_iteration=clf.best_iteration)   
    
    score = metrics.roc_auc_score(train_df.iloc[bayesian_val_index][target].values, predictions)
    
    return score

# The above `LGB_bayesian` function will act as black box function for Bayesian optimization. I already defined the the trainng and validation dataset for LightGBM inside the `LGB_bayesian` function. 
# 
# The `LGB_bayesian` function takes values for `num_leaves`, `min_data_in_leaf`, `learning_rate`, `min_sum_hessian_in_leaf`, `feature_fraction`, `lambda_l1`, `lambda_l2`, `min_gain_to_split`, `max_depth` from Bayesian optimization framework. Keep in mind that `num_leaves`, `min_data_in_leaf`, and `max_depth` should be integer for LightGBM. But Bayesian Optimization sends continous vales to function. So I force them to be integer. I am only going to find optimal parameter values of them. The reader may increase or decrease number of parameters to optimize.

# Now I need to give bounds for these parameters, so that Bayesian optimization only search inside the bounds.

# In[ ]:


# Bounded region of parameter space
bounds_LGB = {
    'num_leaves': (5, 20), 
    'min_data_in_leaf': (5, 20),  
    'learning_rate': (0.01, 0.3),
    'min_sum_hessian_in_leaf': (0.00001, 0.01),    
    'feature_fraction': (0.05, 0.5),
    'lambda_l1': (0, 5.0), 
    'lambda_l2': (0, 5.0), 
    'min_gain_to_split': (0, 1.0),
    'max_depth':(3,15),
}

# Let's put all of them in BayesianOptimization object

# In[ ]:


from bayes_opt import BayesianOptimization

# In[ ]:


LGB_BO = BayesianOptimization(LGB_bayesian, bounds_LGB, random_state=13)

# Now, let's the the key space (parameters) we are going to optimize:

# In[ ]:


print(LGB_BO.space.keys)

# I have created the BayesianOptimization object (`LGB_BO`), it will not work until I call maximize. Before calling it, I want to explain two parameters of BayesianOptimization object (`LGB_BO`) which we can pass to maximize:
# - `init_points`: How many initial random runs of **random** exploration we want to perform. In our case `LGB_bayesian` will be called `n_iter` times.
# - `n_iter`: How many runs of bayesian optimization we want to perform after number of `init_points` runs. 

# Now, it's time to call the function from Bayesian optimization framework to maximize. I allow `LGB_BO` object to run for 5 `init_points` (exploration) and 5 `n_iter` (exploitation).

# In[ ]:


init_points = 5
n_iter = 5

# In[ ]:


print('-' * 130)

with warnings.catch_warnings():
    warnings.filterwarnings('ignore')
    LGB_BO.maximize(init_points=init_points, n_iter=n_iter, acq='ucb', xi=0.0, alpha=1e-6)

# As the optimization is done, let's see what is the maximum value we have got.

# In[ ]:


LGB_BO.max['target']

# The validation AUC for parameters is 0.89 ! Let's see parameters is responsible for this score :)

# In[ ]:


LGB_BO.max['params']

# Now we can use these parameters to our final model!

# Wait, I want to show one more cool option from BayesianOptimization library. You can probe the `LGB_bayesian` function, if you have an idea of the optimal parameters or it you get **parameters from other kernel** like mine [mine](https://www.kaggle.com/fayzur/customer-transaction-prediction). I will copy and paste parameters from my other kernel here. You can probe as folowing:

# In[ ]:


# parameters from version 2 of
#https://www.kaggle.com/fayzur/customer-transaction-prediction?scriptVersionId=10522231

LGB_BO.probe(
    params={'feature_fraction': 0.1403, 
            'lambda_l1': 4.218, 
            'lambda_l2': 1.734, 
            'learning_rate': 0.07, 
            'max_depth': 14, 
            'min_data_in_leaf': 17, 
            'min_gain_to_split': 0.1501, 
            'min_sum_hessian_in_leaf': 0.000446, 
            'num_leaves': 6},
    lazy=True, # 
)

# OK, by default these will be explored lazily (lazy=True), meaning these points will be evaluated only the next time you call maximize. Let's do a maximize call of `LGB_BO` object.

# In[ ]:


LGB_BO.maximize(init_points=0, n_iter=0) # remember no init_points or n_iter

# Finally, the list of all parameters probed and their corresponding target values is available via the property LGB_BO.res.

# In[ ]:


for i, res in enumerate(LGB_BO.res):
    print("Iteration {}: \n\t{}".format(i, res))

# We have got a better validation score in the probe! As previously I ran `LGB_BO` only for 10 runs. In practice I increase it to arround 100.

# In[ ]:


LGB_BO.max['target']

# In[ ]:


LGB_BO.max['params']

# Let's build a model together use therse parameters ;)

# <a id="3"></a> <br>
# ## 3. Training LightGBM model

# In[ ]:


param_lgb = {
        'num_leaves': int(LGB_BO.max['params']['num_leaves']), # remember to int here
        'max_bin': 63,
        'min_data_in_leaf': int(LGB_BO.max['params']['min_data_in_leaf']), # remember to int here
        'learning_rate': LGB_BO.max['params']['learning_rate'],
        'min_sum_hessian_in_leaf': LGB_BO.max['params']['min_sum_hessian_in_leaf'],
        'bagging_fraction': 1.0, 
        'bagging_freq': 5, 
        'feature_fraction': LGB_BO.max['params']['feature_fraction'],
        'lambda_l1': LGB_BO.max['params']['lambda_l1'],
        'lambda_l2': LGB_BO.max['params']['lambda_l2'],
        'min_gain_to_split': LGB_BO.max['params']['min_gain_to_split'],
        'max_depth': int(LGB_BO.max['params']['max_depth']), # remember to int here
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
        'boost_from_average': False,
    }

# As you see, I assined `LGB_BO`'s optimal parameters to the `param_lgb` dictionary and they will be used to train a model with 5 fold.

# Number of Kfolds:

# In[ ]:


nfold = 5

# In[ ]:


gc.collect()

# In[ ]:


skf = StratifiedKFold(n_splits=nfold, shuffle=True, random_state=2019)

# In[ ]:


oof = np.zeros(len(train_df))
predictions = np.zeros((len(test_df),nfold))

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

    
    clf = lgb.train(param_lgb, xg_train, 5000, valid_sets = [xg_valid], verbose_eval=250, early_stopping_rounds = 50)
    oof[valid_index] = clf.predict(train_df.iloc[valid_index][predictors].values, num_iteration=clf.best_iteration) 
    
    predictions[:,i-1] += clf.predict(test_df[predictors], num_iteration=clf.best_iteration)
    i = i + 1

print("\n\nCV AUC: {:<0.2f}".format(metrics.roc_auc_score(train_df.target.values, oof)))

# So we got 0.90 AUC in 5 fold cross validation. And 5 fold prediction look like:

# In[ ]:


predictions

# If you are still reading, bare with me. I will not take much of your time. :D We are almost done. Let's do a rank averaging on 5 fold predictions.

# <a id="4"></a> <br>
# ## 4. Rank averaging

# In[ ]:


print("Rank averaging on", nfold, "fold predictions")
rank_predictions = np.zeros((predictions.shape[0],1))
for i in range(nfold):
    rank_predictions[:, 0] = np.add(rank_predictions[:, 0], rankdata(predictions[:, i].reshape(-1,1))/rank_predictions.shape[0]) 

rank_predictions /= nfold

# Let's submit prediction to Kaggle.

# <a id="5"></a> <br>
# ## 5. Submission

# In[ ]:


sub_df = pd.DataFrame({"ID_code": test_df.ID_code.values})
sub_df["target"] = rank_predictions
sub_df[:10]

# In[ ]:


sub_df.to_csv("Customer_Transaction_rank_predictions.csv", index=False)

# Do not forget to upvote :) Also fork and modify for your own use. ;)

# In[ ]:



