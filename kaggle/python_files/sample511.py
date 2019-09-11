#!/usr/bin/env python
# coding: utf-8

# # 2.  Quick study: LGBM, XGB and Catboost
# 
# &nbsp;
# 
# Hi, and welcome!  In this short kernel, we will: 1) run  baseline versions of `LightGBM`, `XGBoost` and `Catboost`  over the [Google Analytics Customer Revenue Prediction](https://www.kaggle.com/c/google-analytics-customer-revenue-prediction) Challenge dataset, 2)  present the offline `rmse`, the running time and the `public score` of each of them and 3) create a trivial linear ensemble obtaining a `1.6677` score in the public leaderboard.
# 
# 
# &nbsp;
# 
# | Model        | Rounds | Train RMSE           | Validation RMSE | Train time | Public Score|
# | ------------- |------:|-----:|-----:| -----:| -----:|
# | `LightGBM`      | 5000| 1.505 | <span style='color:green'>1.60372 </span> | 7min 48s | <span style='color:green'>1.6717</span> |
# | `XGBoost`      | 2000| 1.568 | 1.64924 | <span style='color:red'>54min 54s </span> | 1.6946 |
# | `Catboost`      | 1000| 1.52184 | 1.61231  | <span style='color:green'>2min 24s</span> | 1.6722 |
# | `Ensemble`      | -- | --| -- | -- | <span style='color:green'>1.6677</span>|
# Result table from [Conclusions](#conclusions) section.
# 
# 
# This kernel is strongly based on these previous work:
# * [LGBM (RF) starter [LB: 1.70]](https://www.kaggle.com/fabiendaniel/lgbm-rf-starter-lb-1-70) - Preprocessing is taken *as-is* from this awesome kernel by [FabienDaniel](https://www.kaggle.com/fabiendaniel/).
# * [LightGBM + XGBoost + Catboost](https://www.kaggle.com/samratp/lightgbm-xgboost-catboost) - LGBM, XGBoost and Catboost functions *taken and ligerely adapted* from this other awesome kernel by [Samrat P](https://www.kaggle.com/samratp).
# 
# 
# 
# 
# The notebook has the following sections:
# 
# 1. [Preprocessing](#preprocessing)
# 2. [Models](#models)
#   - 2.1. [LightGBM](#lightgbm)
#   - 2.2. [XGBoost](#xgboost)
#   - 2.3. [Catboost](#catboost)
# 3. [Ensemble and submissions](#ensemble)
# 4. [Conclusions](#conclusions)
# 5. [References](#references)

# <a id='preprocessing'></a>
# ## 1. Preprocessing
# 
# &nbsp;
# 
# The preprocessing was taken as-is from [LGBM (RF) starter [LB: 1.70]](https://www.kaggle.com/fabiendaniel/lgbm-rf-starter-lb-1-70),  we just gathered it in one function. It consists of the following: drop columns with no information, label-encode the categorical columns with `pd.factorize()` and create few time-related columns. Refer to that already well known kernel for a detailed explanation of the preprocessing! 
# 
# Here, we will step over this section, hidding the code. The function `preprocess()` reads the original csvs (files pointed by variables `INPUT_TRAIN` and `INPUT_TEST`) and generates three processed ones: `TRAIN`, `TEST` and `Y`, where `Y` is separated from `TRAIN` for convenience. The resulting `TRAIN` and `TEST` files have 32 columns and `Y` has the logarithm of the `transactionRevenue` of the `TRAIN`.

# In[ ]:


INPUT_TRAIN = "../input/train.csv"
INPUT_TEST = "../input/test.csv"

TRAIN='train-processed.csv'
TEST='test-processed.csv'
Y='y.csv'

# In[ ]:


import os
import gc
import json
import time
from datetime import datetime
import numpy as np
import pandas as pd
from pandas.io.json import json_normalize

import warnings
warnings.filterwarnings('ignore')

# Reference: https://www.kaggle.com/julian3833/1-quick-start-read-csv-and-flatten-json-fields
def load_df(csv_path=INPUT_TRAIN, nrows=None):
    print(f"Loading {csv_path}")
    JSON_COLUMNS = ['device', 'geoNetwork', 'totals', 'trafficSource']
    
    df = pd.read_csv(csv_path, 
                     converters={column: json.loads for column in JSON_COLUMNS}, 
                     dtype={'fullVisitorId': 'str'}, # Important!!
                     nrows=nrows)
    for column in JSON_COLUMNS:
        column_as_df = json_normalize(df[column])
        column_as_df.columns = [f"{column}.{subcolumn}" for subcolumn in column_as_df.columns]
        df = df.drop(column, axis=1).merge(column_as_df, right_index=True, left_index=True)
    print(f"Loaded {os.path.basename(csv_path)}. Shape: {df.shape}")
    return df


# This function is just a packaged version of this kernel:
# https://www.kaggle.com/fabiendaniel/lgbm-rf-starter-lb-1-70
def process_dfs(train_df, test_df):
    print("Processing dfs...")
    print("Dropping repeated columns...")
    columns = [col for col in train_df.columns if train_df[col].nunique() > 1]
    
    train_df = train_df[columns]
    test_df = test_df[columns]

    trn_len = train_df.shape[0]
    merged_df = pd.concat([train_df, test_df])

    merged_df['diff_visitId_time'] = merged_df['visitId'] - merged_df['visitStartTime']
    merged_df['diff_visitId_time'] = (merged_df['diff_visitId_time'] != 0).astype(int)
    del merged_df['visitId']

    del merged_df['sessionId']

    print("Generating date columns...")
    format_str = '%Y%m%d' 
    merged_df['formated_date'] = merged_df['date'].apply(lambda x: datetime.strptime(str(x), format_str))
    merged_df['WoY'] = merged_df['formated_date'].apply(lambda x: x.isocalendar()[1])
    merged_df['month'] = merged_df['formated_date'].apply(lambda x:x.month)
    merged_df['quarter_month'] = merged_df['formated_date'].apply(lambda x:x.day//8)
    merged_df['weekday'] = merged_df['formated_date'].apply(lambda x:x.weekday())

    del merged_df['date']
    del merged_df['formated_date']

    merged_df['formated_visitStartTime'] = merged_df['visitStartTime'].apply(
        lambda x: time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(x)))
    merged_df['formated_visitStartTime'] = pd.to_datetime(merged_df['formated_visitStartTime'])
    merged_df['visit_hour'] = merged_df['formated_visitStartTime'].apply(lambda x: x.hour)

    del merged_df['visitStartTime']
    del merged_df['formated_visitStartTime']

    print("Encoding columns with pd.factorize()")
    for col in merged_df.columns:
        if col in ['fullVisitorId', 'month', 'quarter_month', 'weekday', 'visit_hour', 'WoY']: continue
        if merged_df[col].dtypes == object or merged_df[col].dtypes == bool:
            merged_df[col], indexer = pd.factorize(merged_df[col])

    print("Splitting back...")
    train_df = merged_df[:trn_len]
    test_df = merged_df[trn_len:]
    return train_df, test_df

def preprocess():
    train_df = load_df()
    test_df = load_df(INPUT_TEST)

    target = train_df['totals.transactionRevenue'].fillna(0).astype(float)
    target = target.apply(lambda x: np.log1p(x))
    del train_df['totals.transactionRevenue']

    train_df, test_df = process_dfs(train_df, test_df)
    train_df.to_csv(TRAIN, index=False)
    test_df.to_csv(TEST, index=False)
    target.to_csv(Y, index=False)


# In[ ]:


preprocess()

# <a id='models'></a>
# ## 2. Models
# 
# &nbsp;
# 
# Before jumping into the models, we need some imports and auxiliary code. The most relevant things to note in this section are the imports of the three different libraries:
# 
# ```python
# import lightgbm as lgb
# import xgboost as xgb
# from catboost import CatBoostRegressor
# ```
# 
# Besides of that, we define the `rmse` evaluation metric based on sklearn's `mean_squared_error` and an auxiliary function for loading the preprocessed dataframes of the previous step,  `load_preprocessed_dfs()`
# 

# In[ ]:


import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def rmse(y_true, y_pred):
    return round(np.sqrt(mean_squared_error(y_true, y_pred)), 5)

def load_preprocessed_dfs(drop_full_visitor_id=True):
    """
    Loads files `TRAIN`, `TEST` and `Y` generated by preprocess() into variables
    """
    X_train = pd.read_csv(TRAIN, converters={'fullVisitorId': str})
    X_test = pd.read_csv(TEST, converters={'fullVisitorId': str})
    y_train = pd.read_csv(Y, names=['LogRevenue']).T.squeeze()
    
    # This is the only `object` column, we drop it for train and evaluation
    if drop_full_visitor_id: 
        X_train = X_train.drop(['fullVisitorId'], axis=1)
        X_test = X_test.drop(['fullVisitorId'], axis=1)
    return X_train, y_train, X_test

# In the cell below we load the dataframes `X`, `y` and `X_test` and generate a `train`, `validation` from the train data.  We drop `fullVisitorId`, so the dataframes have 31 columns each, while the `train` data has 768,000 rows, the `validation` 135,000 and the `test` set has 800,000.

# In[ ]:


X, y, X_test = load_preprocessed_dfs()
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=1)

print(f"Train shape: {X_train.shape}")
print(f"Validation shape: {X_val.shape}")
print(f"Test (submit) shape: {X_test.shape}")

# <a id='lightgbm'></a>
# ### 2.1. LightGBM
# 
# &nbsp;
# 
# The following code is an adaption of [LightGBM + XGBoost + Catboost](https://www.kaggle.com/samratp/lightgbm-xgboost-catboost) to this particular competition: it's pretty simple, in fact, so the best thing to do is to just look at `run_lgb()` below.  Here, some notes which may help to read the code:
# * lightgbm is installed with `pip install lightgbm` and typically imported as `lgb`.
# * `lgb` defines a `Dataset` object, which can be a tuple `(X, y)` or just a `X` (you'd typically have `train`, `validation` and `test` Datasets).
# * `create Dataset` $\rightarrow$ `call train` is the most common workflow with `lgb`, but there is a [Scikit-learn API](https://lightgbm.readthedocs.io/en/latest/Python-API.html#scikit-learn-api) (which allows to do: `create model` $\rightarrow$ `fit`  $\rightarrow$ `predict`).
# * `lgb` has  **a lot** of [parameters](https://lightgbm.readthedocs.io/en/latest/Parameters.html).
# * `lgb` is *good out of the box* , *fast* and *it can handle categorical values*.

# In[ ]:


def run_lgb(X_train, y_train, X_val, y_val, X_test):
    
    params = {
        "objective" : "regression",
        "metric" : "rmse",
        "num_leaves" : 40,
        "learning_rate" : 0.005,
        "bagging_fraction" : 0.6,
        "feature_fraction" : 0.6,
        "bagging_frequency" : 6,
        "bagging_seed" : 42,
        "verbosity" : -1,
        "seed": 42
    }
    
    lgb_train_data = lgb.Dataset(X_train, label=y_train)
    lgb_val_data = lgb.Dataset(X_val, label=y_val)

    model = lgb.train(params, lgb_train_data, 
                      num_boost_round=5000,
                      valid_sets=[lgb_train_data, lgb_val_data],
                      early_stopping_rounds=100,
                      verbose_eval=500)

    y_pred_train = model.predict(X_train, num_iteration=model.best_iteration)
    y_pred_val = model.predict(X_val, num_iteration=model.best_iteration)
    y_pred_submit = model.predict(X_test, num_iteration=model.best_iteration)

    print(f"LGBM: RMSE val: {rmse(y_val, y_pred_val)}  - RMSE train: {rmse(y_train, y_pred_train)}")
    return y_pred_submit, model

# In[ ]:


# Train LGBM and generate predictions
lgb_preds, lgb_model = run_lgb(X_train, y_train, X_val, y_val, X_test)

# In[ ]:


print("LightGBM features importance...")
gain = lgb_model.feature_importance('gain')
featureimp = pd.DataFrame({'feature': lgb_model.feature_name(), 
                   'split': lgb_model.feature_importance('split'), 
                   'gain': 100 * gain / gain.sum()}).sort_values('gain', ascending=False)
print(featureimp[:10])

# Further readings about `LightGBM`:
# * [What is LightGBM, How to implement it? How to fine tune the parameters?](https://medium.com/@pushkarmandot/https-medium-com-pushkarmandot-what-is-lightgbm-how-to-implement-it-how-to-fine-tune-the-parameters-60347819b7fc)
# * [Documentation](https://lightgbm.readthedocs.io/en/latest/).  In particular:
#    - [Python Quick Start](https://lightgbm.readthedocs.io/en/latest/Python-Intro.html)
#    - [Python API](https://lightgbm.readthedocs.io/en/latest/Python-API.html)
#    - [Parameters section](https://lightgbm.readthedocs.io/en/latest/Parameters.html)

# <a id='xgboost'></a>
# ### 2.2. XGBoost
# 
# &nbsp;
# 
# `XGBoost` stands for `Extreme Grandient Boosting`, which is a kind of `sklearn.ensemble`'s `GradientBoostingRegressor` on steroids.
# 
# 
# > XGBoost is an optimized distributed gradient boosting library designed to be highly efficient, flexible and portable. 
# 
# 
# In concrete, the library is installed with `pip install xgboost`, its typically imported as `xgb` and,  once imported, it works similarly to `lgb`: we will create a custom dataset object (in this case, the `DMatrix`) and we will call the `train()` function of the module with some parameters and some `DMatrices`. 

# In[ ]:


def run_xgb(X_train, y_train, X_val, y_val, X_test):
    params = {'objective': 'reg:linear',
              'eval_metric': 'rmse',
              'eta': 0.001,
              'max_depth': 10,
              'subsample': 0.6,
              'colsample_bytree': 0.6,
              'alpha':0.001,
              'random_state': 42,
              'silent': True}

    xgb_train_data = xgb.DMatrix(X_train, y_train)
    xgb_val_data = xgb.DMatrix(X_val, y_val)
    xgb_submit_data = xgb.DMatrix(X_test)

    model = xgb.train(params, xgb_train_data, 
                      num_boost_round=2000, 
                      evals= [(xgb_train_data, 'train'), (xgb_val_data, 'valid')],
                      early_stopping_rounds=100, 
                      verbose_eval=500
                     )

    y_pred_train = model.predict(xgb_train_data, ntree_limit=model.best_ntree_limit)
    y_pred_val = model.predict(xgb_val_data, ntree_limit=model.best_ntree_limit)
    y_pred_submit = model.predict(xgb_submit_data, ntree_limit=model.best_ntree_limit)

    print(f"XGB : RMSE val: {rmse(y_val, y_pred_val)}  - RMSE train: {rmse(y_train, y_pred_train)}")
    return y_pred_submit, model

# In[ ]:


xgb_preds, xgb_model = run_xgb(X_train, y_train, X_val, y_val, X_test)

# Further readings about `XGBoost`:
# * [Documentation](https://xgboost.readthedocs.io/en/latest/), in particular:
#  - [Python Intro](https://xgboost.readthedocs.io/en/latest/python/python_intro.html)
# * [A Gentle Introduction to XGBoost for Applied Machine Learning](https://machinelearningmastery.com/gentle-introduction-xgboost-applied-machine-learning/)

# <a id='catboost'></a>
# ### 2.3. Catboost
# 
# &nbsp;
# 
# > `CatBoost` is a state-of-the-art open-source gradient boosting on decision trees library.
# 
# CatBoost is a very good, fast and trendy boosting model, which handles categorical parameters (that's the `cat` in `catboost`!). It's installed with `pip install catboost` and proposes a scikit-learn like workflow with a Classifier or a Regressor and a `create` - `fit` -  `predict` cycle.
# 
# Note that both `lgb` and `xgb` offer a "Scikit-learn API" option too: check out [here](https://lightgbm.readthedocs.io/en/latest/Python-API.html#scikit-learn-api) for `LightGBM` and [here](https://xgboost.readthedocs.io/en/latest/python/python_api.html#module-xgboost.sklearn) for `XGBoost`.

# In[ ]:


def run_catboost(X_train, y_train, X_val, y_val, X_test):
    model = CatBoostRegressor(iterations=1000,
                             learning_rate=0.05,
                             depth=10,
                             eval_metric='RMSE',
                             random_seed = 42,
                             bagging_temperature = 0.2,
                             od_type='Iter',
                             metric_period = 50,
                             od_wait=20)
    model.fit(X_train, y_train,
              eval_set=(X_val, y_val),
              use_best_model=True,
              verbose=True)
    
    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)
    y_pred_submit = model.predict(X_test)

    print(f"CatB: RMSE val: {rmse(y_val, y_pred_val)}  - RMSE train: {rmse(y_train, y_pred_train)}")
    return y_pred_submit, model

# In[ ]:


# Train Catboost and generate predictions
cat_preds, cat_model = run_catboost(X_train, y_train, X_val, y_val,  X_test)

# Further readings on `Catboost`:
# * [Documentation](https://tech.yandex.com/catboost/doc/dg/concepts/about-docpage/), in particular:
#   - [Python Quickstart](https://tech.yandex.com/catboost/doc/dg/concepts/python-quickstart-docpage/)
# * [CatBoost: A machine learning library to handle categorical (CAT) data automatically](https://www.analyticsvidhya.com/blog/2017/08/catboost-automated-categorical-data/)

# <a id='ensemble'></a>
# ## 3. Ensemble and submissions

# In the previous section we trained baseline versions of `LightGBM`, `XGBoost` and `Catboost`.  
# 
# In this section we will create a trivial linear ensemble using hardcoded coefficients (70/30/0). We will create the submissions for the 3 baseline models and for the new ensemble as well.

# In[ ]:


# Note: this is currently being reconstructed!
ensemble_preds_70_30_00 = 0.7 * lgb_preds + 0.3 * cat_preds + 0.0 * xgb_preds 
ensemble_preds_70_25_05 = 0.7 * lgb_preds + 0.25 * cat_preds + 0.05 * xgb_preds 

# In[ ]:


def submit(predictions, filename='submit.csv'):
    """
    Takes a (804684,) 1d-array of predictions and generates a submission file named filename
    """
    _, _, X_submit = load_preprocessed_dfs(drop_full_visitor_id=False)
    submission = X_submit[['fullVisitorId']].copy()
    
    submission.loc[:, 'PredictedLogRevenue'] = predictions
    grouped_test = submission[['fullVisitorId', 'PredictedLogRevenue']].groupby('fullVisitorId').sum().reset_index()
    grouped_test.to_csv(filename,index=False)

submit(lgb_preds, "submit-lgb.csv")
# Note: I disabled XGB to make the notebook run faster
submit(xgb_preds, "submit-xgb.csv")
submit(cat_preds, "submit-cat.csv")
submit(ensemble_preds_70_30_00, "submit-ensemble-70_30_00.csv")
submit(ensemble_preds_70_25_05, "submit-ensemble-70_25_05.csv")

ensemble_preds_70_30_00_pos = np.where(ensemble_preds_70_30_00 < 0, 0, ensemble_preds_70_30_00)
submit(ensemble_preds_70_30_00_pos, "submit-ensemble-70_30_00-positive.csv")

ensemble_preds_70_25_05_pos = np.where(ensemble_preds_70_25_05 < 0, 0, ensemble_preds_70_25_05)
submit(ensemble_preds_70_25_05_pos, "submit-ensemble-70_25_05-positive.csv")

# <a id='conclusions'></a>
# ## 4. Conclusions
# 
# <!-- Before going to the conclusions, some disclaimers about the scope of this notebooks:
# * The `RMSE` we are calculating is `per visit`, while the online `RMSE` is `per visitor`. 
# * The `lgb` submit contains negative results. This can be trivially fixed after the prediction or, I think, setting some parameters. 
# * We didn't hold-out a `test` split of the dataset, so we don't have a test score. This was done in order to simplify the code and a more advanced version of it should consider that split.
# * `LightGBM` or `Catboost` can handle categorical values, but we are not doing it here.
# -->
# 
# With a baseline parameter configuration, the following results were obtained:
# 
# | Model        | Rounds | Train RMSE           | Validation RMSE | Train time | Submit Score|
# | ------------- |------:|-----:|-----:| -----:| -----:|
# | `LightGBM`      | 5000| 1.505 | <span style='color:green'>1.60372 </span> | 7min 48s | 1.6717 |
# | `XGBoost`      | 2000| 1.568 | 1.64924 | <span style='color:red'>54min 54s </span> | 1.6946|
# | `Catboost`      | 1000| 1.52184 | 1.61231  | <span style='color:green'>2min 24s</span> | 1.6722|
# | `Ensemble`      | -- | --| -- | -- | 1.6677|
# 
# &nbsp;
# 
# `LightGBM` achieved the best results in train, validation and public score, while `Catboost` obtained the best train-time and an extremely competitive score as well.  `XGBoost`, on the other hand, took much more time (from minutes to hours), and didn't perform well in score either.
# 
# Of course, we are not trying to prove any general point, this notebook is just a very narrow and educational comparison experiment of very baseline usages of these three trendy libraries!
# 
# Well, that's it, those are some baseline implementations of the 3 boosting kings (?) applied to the GA challenge. 
# 
# Comments/ideas/improvements are welcomed!

# <a id='references'></a>
# ## 5. References
# 
# #### Kernels
# * [LGBM (RF) starter [LB: 1.70]](https://www.kaggle.com/fabiendaniel/lgbm-rf-starter-lb-1-70)
# * [LightGBM + XGBoost + Catboost](https://www.kaggle.com/samratp/lightgbm-xgboost-catboost) 
# 
# #### Articles
# * [What is LightGBM, How to implement it? How to fine tune the parameters?](https://medium.com/@pushkarmandot/https-medium-com-pushkarmandot-what-is-lightgbm-how-to-implement-it-how-to-fine-tune-the-parameters-60347819b7fc)
# * [A Gentle Introduction to XGBoost for Applied Machine Learning](https://machinelearningmastery.com/gentle-introduction-xgboost-applied-machine-learning/)
# * [CatBoost: A machine learning library to handle categorical (CAT) data automatically](https://www.analyticsvidhya.com/blog/2017/08/catboost-automated-categorical-data/)
# * [CatBoost vs. Light GBM vs. XGBoost](https://towardsdatascience.com/catboost-vs-light-gbm-vs-xgboost-5f93620723db?gi=4e06b8e37886)
# * [Machine Learning Challenge Winning Solutions](https://github.com/Microsoft/LightGBM/blob/master/examples/README.md#machine-learning-challenge-winning-solutions) - a list of challenges won by some version of LightGBM.
# 
# #### Documentation
# *  [LightGBM Documentation: Python Quick Start](https://lightgbm.readthedocs.io/en/latest/Python-Intro.html)
# *  [LightGBM Documentation: Python API](https://lightgbm.readthedocs.io/en/latest/Python-API.html)
# * [LightGBM Documentation: Parameters section](https://lightgbm.readthedocs.io/en/latest/Parameters.html)
# * [XGBoost Documentation: Python Intro](https://xgboost.readthedocs.io/en/latest/python/python_intro.html)
# * [Catboost Documentation: Python Quickstart](https://tech.yandex.com/catboost/doc/dg/concepts/python-quickstart-docpage/)
# * [LightGBM Documentation: Scikit-learn API](https://lightgbm.readthedocs.io/en/latest/Python-API.html#scikit-learn-api)
# * [XGBoost Documentation: Scikit-learn API](https://xgboost.readthedocs.io/en/latest/python/python_api.html#module-xgboost.sklearn)
# 

# In[ ]:


# Delete the files created by catboost. 
# TODO: check if `allow_writing_files` params works or find another better way to do this
