#!/usr/bin/env python
# coding: utf-8

# <h2>1. About this notebook</h2>
# 
# In this notebook I try a few different models to predict the time to failure during earthquake simulations (LANL competition). Every model is implemented trough a Scikit-learn pipeline and compared using cross-validation scores and visualizations.
# 
# For more details about LANL competition you can check my [previous kernel](https://www.kaggle.com/jsaguiar/seismic-data-exploration).

# In[1]:


import os
import time
import random
import numpy as np
import pandas as pd
from tqdm import tqdm_notebook
# Visualizations
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
# Sklearn utilities
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV, KFold, RandomizedSearchCV
from sklearn.feature_selection import RFECV, SelectFromModel
# Models
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import NuSVR, SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
import lightgbm as lgb
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
sns.set()
init_notebook_mode(connected=True)

# This competition has a single feature (seismic signal) and we must predict the time to failure:

# In[2]:


data_type = {'acoustic_data': np.int16, 'time_to_failure': np.float32}
train = pd.read_csv('../input/train.csv', dtype=data_type)
train.head()

# <h2>2. Feature Engineering</h2>
# 
# The idea is to group the acoustic data in chunks and extract the following features:
# 
# * Aggregations: min, max, mean and std
# * Absolute features: max, mean and std
# * Quantile features
# * Trend features
# * Rolling features
# * Ratios

# In[3]:


def add_trend_feature(arr, abs_values=False):
    """Fit a univariate linear regression and return the coefficient."""
    idx = np.array(range(len(arr)))
    if abs_values:
        arr = np.abs(arr)
    lr = LinearRegression()
    lr.fit(idx.reshape(-1, 1), arr)
    return lr.coef_[0]

def extract_features_from_segment(x):
    """Returns a dictionary with the features for the given segment of acoustic data."""
    features = {}
    
    features['ave'] = x.values.mean()
    features['std'] = x.values.std()
    features['max'] = x.values.max()
    features['min'] = x.values.min()
    features['q90'] = np.quantile(x.values, 0.90)
    features['q95'] = np.quantile(x.values, 0.95)
    features['q99'] = np.quantile(x.values, 0.99)
    features['q05'] = np.quantile(x.values, 0.05)
    features['q10'] = np.quantile(x.values, 0.10)
    features['q01'] = np.quantile(x.values, 0.01)
    features['std_to_mean'] = features['std'] / features['ave']
    
    features['abs_max'] = np.abs(x.values).max()
    features['abs_mean'] = np.abs(x.values).mean()
    features['abs_std'] = np.abs(x.values).std()
    features['trend'] = add_trend_feature(x.values)
    features['abs_trend'] = add_trend_feature(x.values, abs_values=True)
    
    # New features - rolling features
    for w in [10, 50, 100, 1000]:
        x_roll_abs_mean = x.abs().rolling(w).mean().dropna().values
        x_roll_mean = x.rolling(w).mean().dropna().values
        x_roll_std = x.rolling(w).std().dropna().values
        x_roll_min = x.rolling(w).min().dropna().values
        x_roll_max = x.rolling(w).max().dropna().values
        
        features['ave_roll_std_' + str(w)] = x_roll_std.mean()
        features['std_roll_std_' + str(w)] = x_roll_std.std()
        features['max_roll_std_' + str(w)] = x_roll_std.max()
        features['min_roll_std_' + str(w)] = x_roll_std.min()
        features['q01_roll_std_' + str(w)] = np.quantile(x_roll_std, 0.01)
        features['q05_roll_std_' + str(w)] = np.quantile(x_roll_std, 0.05)
        features['q10_roll_std_' + str(w)] = np.quantile(x_roll_std, 0.10)
        features['q95_roll_std_' + str(w)] = np.quantile(x_roll_std, 0.95)
        features['q99_roll_std_' + str(w)] = np.quantile(x_roll_std, 0.99)
        
        features['ave_roll_mean_' + str(w)] = x_roll_mean.mean()
        features['std_roll_mean_' + str(w)] = x_roll_mean.std()
        features['max_roll_mean_' + str(w)] = x_roll_mean.max()
        features['min_roll_mean_' + str(w)] = x_roll_mean.min()
        features['q05_roll_mean_' + str(w)] = np.quantile(x_roll_mean, 0.05)
        features['q95_roll_mean_' + str(w)] = np.quantile(x_roll_mean, 0.95)
        
        features['ave_roll_abs_mean_' + str(w)] = x_roll_abs_mean.mean()
        features['std_roll_abs_mean_' + str(w)] = x_roll_abs_mean.std()
        features['q05_roll_abs_mean_' + str(w)] = np.quantile(x_roll_abs_mean, 0.05)
        features['q95_roll_abs_mean_' + str(w)] = np.quantile(x_roll_abs_mean, 0.95)
        
        features['std_roll_min_' + str(w)] = x_roll_min.std()
        features['max_roll_min_' + str(w)] = x_roll_min.max()
        features['q05_roll_min_' + str(w)] = np.quantile(x_roll_min, 0.05)
        features['q95_roll_min_' + str(w)] = np.quantile(x_roll_min, 0.95)

        features['std_roll_max_' + str(w)] = x_roll_max.std()
        features['min_roll_max_' + str(w)] = x_roll_max.min()
        features['q05_roll_max_' + str(w)] = np.quantile(x_roll_max, 0.05)
        features['q95_roll_max_' + str(w)] = np.quantile(x_roll_max, 0.95)
    return features

# Functions for extracting features and creating dataframes. Make train also returns one Series with the target variable and another one with the earthquake number for each segment. 

# In[4]:


def make_train(train_data, size=150000, skip=150000):
    num_segments = int(np.floor((train_data.shape[0] - size) / skip)) + 1
    features_list = []
    target_list = []
    quake_num = []
    quake_count = 0
    
    for index in tqdm_notebook(range(num_segments)):
        seg = train_data.iloc[index*skip:index*skip + size]
        
        target_list.append(seg.time_to_failure.values[-1])
        features_list.append(extract_features_from_segment(seg.acoustic_data))
        
        # From which quake does the segment come from
        quake_num.append(quake_count)
        if any(seg.time_to_failure.diff() > 5):
            quake_count += 1
    return pd.DataFrame(features_list), pd.Series(target_list), pd.Series(quake_num)


def make_test():
    submission = pd.read_csv('../input/sample_submission.csv', index_col='seg_id')
    X_test = pd.DataFrame(index=submission.index, dtype=np.float64)
    features_list = []
    
    for seg_id in tqdm_notebook(submission.index):
        seg = pd.read_csv('../input/test/' + seg_id + '.csv')
        features_list.append(extract_features_from_segment(seg.acoustic_data))
    return pd.DataFrame(features_list)

# In[5]:


X_train, target, quake = make_train(train, skip=150000)
print("Train shape:", X_train.shape)
X_train.head(3)

# <h2>3. Cross-validation strategy</h2>
# 
# Since there are only four thousand samples, it is better to use cross-validation instead of a simple split for validation. This can be implemented trough a generator, so it is easy to change our strategy. I am using KFold, but you can try a [Group KFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html) with the quake series or [Stratified KFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html) if you prefer.

# In[6]:


def fold_generator(x, y, groups=None, num_folds=10, shuffle=True, seed=2019):
    folds = KFold(num_folds, shuffle=shuffle, random_state=seed)
    for train_index, test_index in folds.split(x, y, groups):
        yield train_index, test_index

# <h2>4. Feature Selection</h2>
# 
# Work in progress...

# <h2>5. Making a Pipeline</h2>
# 
# We will be using a Sklearn Pipeline to perform hyperparameter search and to make predictions. The advantage of using a pipeline is that we are not leaking information from the training to the validation set.
# 
# The feature selection could also be moved to this pipeline, but it would take too long to perform the grid search.

# In[7]:


def make_pipeline(estimator):
    pipeline = Pipeline([
        # Each item is a tuple with a name and a transformer or estimator
        ('scaler', StandardScaler()),
        ('model', estimator)
    ])
    return pipeline

# Now let's create two functions: one for searching the best hyperparameters and another for making predictions and ploting.

# In[8]:


def search_cv(x, y, pipeline, grid, max_iter=None, num_folds=10, shuffle=True):
    """Search hyperparameters and returns a estimator with the best combination found."""
    t0 = time.time()
    
    cv = fold_generator(x, y, num_folds=num_folds)
    if max_iter is None:
        search = GridSearchCV(pipeline, grid, cv=cv,
                              scoring='neg_mean_absolute_error')
    else:
        search = RandomizedSearchCV(pipeline, grid, n_iter=max_iter, cv=cv,
                                    scoring='neg_mean_absolute_error')
    search.fit(x, y)
    
    t0 = time.time() - t0
    print("Best CV score: {:.4f}, time: {:.1f}s".format(-search.best_score_, t0))
    print(search.best_params_)
    return search.best_estimator_


def make_predictions(x, y, pipeline, num_folds=10, shuffle=True, test=None, plot=True):
    """Train, make predictions (oof and test data) and plot."""
    if test is not None:
        sub_prediction = np.zeros(test.shape[0])
        
    oof_prediction = np.zeros(x.shape[0])
    for tr_idx, val_idx in fold_generator(x, y, num_folds=num_folds):
        pipeline.fit(x.iloc[tr_idx], y.iloc[tr_idx])
        oof_prediction[val_idx] = pipeline.predict(x.iloc[val_idx])

        if test is not None:
            sub_prediction += pipeline.predict(test) / num_folds
    
    if plot:
        plot_predictions(y, oof_prediction)
    if test is None:
        return oof_prediction
    else:
        return oof_prediction, sub_prediction

# In[9]:


def plot_predictions(y, oof_predictions):
    """Plot out-of-fold predictions vs actual values."""
    fig, axis = plt.subplots(1, 2, figsize=(14, 6))
    ax1, ax2 = axis
    ax1.set_xlabel('actual')
    ax1.set_ylabel('predicted')
    ax1.set_ylim([-5, 20])
    ax2.set_xlabel('train index')
    ax2.set_ylabel('time to failure')
    ax2.set_ylim([-2, 18])
    ax1.scatter(y, oof_predictions, color='brown')
    ax1.plot([(0, 0), (20, 20)], [(0, 0), (20, 20)], color='blue')
    ax2.plot(y, color='blue', label='y_train')
    ax2.plot(oof_predictions, color='orange')

# <h2>6. Testing Models</h2>
# 
# The predicted values in the following plots are using a out-of-fold scheme.
# 
# <h3>Ridge Regression</h3>
# 
# The first model will be a linear regression with L2 regularization.
# 
# 

# In[10]:


grid = {'model__alpha': np.concatenate([np.linspace(0.001, 1, 200),
                                        np.linspace(1, 100, 500)])}


ridge_pipe = make_pipeline(Ridge(random_state=2019))
ridge_pipe = search_cv(X_train, target, ridge_pipe, grid)
ridge_oof = make_predictions(X_train, target, ridge_pipe)

# There are some negative predictions when using a linear model. We can try to change negative values for zeros:

# In[ ]:


ridge_oof[ridge_oof < 0] = 0
print("Mean error: {:.4f}".format(mean_absolute_error(target, ridge_oof)))

# <h3>Kernel Ridge</h3>
# 
# This model combines regularized linear regression with a given kernel (radial basis in this case).

# In[ ]:


grid = {'model__gamma': np.linspace(1e-8, 0.1, 100),
        'model__alpha': np.linspace(1e-6, 1, 100)}
kr_pipe = make_pipeline(KernelRidge(kernel='rbf'))
kr_pipe = search_cv(X_train, target, kr_pipe, grid, max_iter=40)
kr_oof = make_predictions(X_train, target, kr_pipe)

# <h3>SVM</h3>
# Support vector machine with radial basis function kernel.

# In[ ]:


grid = {'model__epsilon': np.linspace(0.01, 0.5, 10),
        'model__C': np.linspace(0.01, 5, 100)}
svm_pipe = make_pipeline(SVR(kernel='rbf', gamma='scale'))
svm_pipe = search_cv(X_train, target, svm_pipe, grid, max_iter=40)
svm_oof = make_predictions(X_train, target, svm_pipe)

# <h3>Random Forests</h3>
# 
# This regressor fits several decision trees with a different subset of the original data for each tree. Predictions are the average between trees.

# In[ ]:


grid = {
    'model__max_depth': [4, 6, 8, 10, 12],
    'model__max_features': ['auto', 'sqrt', 'log2'],
    'model__min_samples_leaf': [2, 4, 8, 12, 14, 16, 20],
    'model__min_samples_split': [2, 4, 6, 8, 12, 16, 20],
}
rf_pipe = make_pipeline(RandomForestRegressor(criterion='mae', n_estimators=50))
rf_pipe = search_cv(X_train, target, rf_pipe, grid, max_iter=10)
rf_oof = make_predictions(X_train, target, rf_pipe)

# <h3>Ada Boost</h3>
# 
# AdaBoost begins by fitting a base estimator on the original dataset and then fits additional copies on the same dataset. At each iteration (estimator), the weights of instances are adjusted according to the error of the last prediction. It's similar to the next model, but gradient boosting fits additional estimator copies on the current error and not on the original dataset.

# In[ ]:


grid = {'model__learning_rate': np.linspace(1e-5, 0.9, 100)}
#base = DecisionTreeRegressor(max_depth=5)
base = Ridge(alpha=2)

ada_pipe = make_pipeline(AdaBoostRegressor(base_estimator=base, n_estimators=200))
ada_pipe = search_cv(X_train, target, ada_pipe, grid, max_iter=30)
ada_oof = make_predictions(X_train, target, ada_pipe)

# <h3>Gradient Boosting</h3>
# 
# The last model is a gradient boosting decision tree. It's not possible to use GridSearchCV with early stopping (lightgbm), so I am using a custom function for random search.

# In[ ]:


fixed_params = {
    'objective': 'regression_l1',
    'boosting': 'gbdt',
    'verbosity': -1,
    'random_seed': 19,
    'n_estimators': 20000,
}

param_grid = {
    'learning_rate': [0.1, 0.05, 0.01, 0.005],
    'num_leaves': list(range(2, 60, 2)),
    'max_depth': [4, 6, 8, 12, 16, -1],
    'feature_fraction': [0.8, 0.85, 0.9, 0.95, 1],
    'subsample': [0.8, 0.9, 0.95, 1],
    'lambda_l1': [0, 0.1, 0.2, 0.4, 0.6, 0.8],
    'lambda_l2': [0, 0.1, 0.2, 0.4, 0.6, 0.8],
    'min_data_in_leaf': [10, 20, 40, 60, 100],
    'min_gain_to_split': [0, 0.001, 0.01, 0.1],
}

best_score = 999
dataset = lgb.Dataset(X_train, label=target)  # no need to scale features

for i in range(500):
    params = {k: random.choice(v) for k, v in param_grid.items()}
    params.update(fixed_params)
    result = lgb.cv(params, dataset, nfold=5, early_stopping_rounds=100,
                    stratified=False)
    
    if result['l1-mean'][-1] < best_score:
        best_score = result['l1-mean'][-1]
        best_params = params
        best_nrounds = len(result['l1-mean'])

# In[ ]:


print("Best mean score: {:.4f}, num rounds: {}".format(best_score, best_nrounds))
print(best_params)
gb_pipe = make_pipeline(lgb.LGBMRegressor(**best_params))
gb_oof = make_predictions(X_train, target, gb_pipe)

# Now let's have a look at the <b>feature importance</b>:

# In[ ]:


def plot_feature_importance(x, y, columns):
    importance_frame = pd.DataFrame()
    for (train_index, valid_index) in fold_generator(x, y):
        reg = lgb.LGBMRegressor(**best_params)
        reg.fit(x.iloc[train_index], y.iloc[train_index],
                early_stopping_rounds=100, verbose=False,
                eval_set=[(x.iloc[train_index], y.iloc[train_index]),
                          (x.iloc[valid_index], y.iloc[valid_index])])
        fold_importance = pd.DataFrame()
        fold_importance["feature"] = columns
        fold_importance["gain"] = reg.booster_.feature_importance(importance_type='gain')
        #fold_importance["split"] = reg.booster_.feature_importance(importance_type='split')
        importance_frame = pd.concat([importance_frame, fold_importance], axis=0)
        
    mean_importance = importance_frame.groupby('feature').mean().reset_index()
    mean_importance.sort_values(by='gain', ascending=True, inplace=True)
    trace = go.Bar(y=mean_importance.feature, x=mean_importance.gain,
                   orientation='h', marker=dict(color='rgb(49,130,189)'))

    layout = go.Layout(
        title='Feature importance', height=1200, width=800,
        showlegend=False,
        xaxis=dict(
            title='Importance by gain',
            titlefont=dict(size=14, color='rgb(107, 107, 107)'),
            domain=[0.15, 1]
        ),
    )

    fig = go.Figure(data=[trace], layout=layout)
    iplot(fig)
    
plot_feature_importance(X_train, target, X_train.columns)

# <h2>7. Submission</h2>

# In[ ]:


X_test = make_test()
gb_oof, gb_sub = make_predictions(X_train, target, gb_pipe,
                                  test=X_test, plot=False)
submission = pd.read_csv('../input/sample_submission.csv')
submission['time_to_failure'] = gb_sub
submission.to_csv('submission_gb.csv', index=False)
submission.time_to_failure.describe()
