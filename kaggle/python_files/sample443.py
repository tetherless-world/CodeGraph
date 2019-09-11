#!/usr/bin/env python
# coding: utf-8

# I think its save to say that everyone taking part in this competition has had problems with validation. Local CV score does not at all represent the leaderboard score. Some people have even described a [negative correlation](https://www.kaggle.com/c/santander-value-prediction-challenge/discussion/59307#350685).
# 
# And how are we supposed to solve this problem without any idea if our model is good without submitting to the leaderboard first (and keep in mind that the public LB is evaluated only on 50% on the data too).
# 
# We have to find a better way of local validation. In this kernel I try a *different* technique. I wouldn't dare to call it better because I have not thoroughly evaluated it but it is certainly interesting.

# In[ ]:


import lightgbm as lgb
import xgboost as xgb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import KFold, BaseCrossValidator
from sklearn.decomposition import TruncatedSVD, FastICA, FactorAnalysis
from sklearn.random_projection import SparseRandomProjection
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import Ridge
from sklearn.preprocessing import scale
from scipy.stats import skew, kurtosis, gmean, ks_2samp
import gc
import psutil
from tqdm._tqdm_notebook import tqdm_notebook as tqdm

tqdm.pandas()
sns.set(style="white", color_codes=True)

# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train.head()

# # Feature engineering

# We will start by defining basic row aggregation features. These are used in most public kernels so I will not further elaborate on this part.

# In[ ]:


def aggregate_row(row):
    non_zero_values = row.iloc[row.nonzero()].astype(float)
    if non_zero_values.empty:
        aggs = {
            'non_zero_mean': np.nan,
            'non_zero_std': np.nan,
            'non_zero_max': np.nan,
            'non_zero_min': np.nan,
            'non_zero_sum': np.nan,
            'non_zero_skewness': np.nan,
            'non_zero_kurtosis': np.nan,
            'non_zero_median': np.nan,
            'non_zero_q1': np.nan,
            'non_zero_q3': np.nan,
            'non_zero_gmean': np.nan,
            'non_zero_log_mean': np.nan,
            'non_zero_log_std': np.nan,
            'non_zero_log_max': np.nan,
            'non_zero_log_min': np.nan,
            'non_zero_log_sum': np.nan,
            'non_zero_log_skewness': np.nan,
            'non_zero_log_kurtosis': np.nan,
            'non_zero_log_median': np.nan,
            'non_zero_log_q1': np.nan,
            'non_zero_log_q3': np.nan,
            'non_zero_log_gmean': np.nan,
            'non_zero_count': np.nan,
            'non_zero_fraction': np.nan
        }
    else:
        aggs = {
            'non_zero_mean': non_zero_values.mean(),
            'non_zero_std': non_zero_values.std(),
            'non_zero_max': non_zero_values.max(),
            'non_zero_min': non_zero_values.min(),
            'non_zero_sum': non_zero_values.sum(),
            'non_zero_skewness': skew(non_zero_values),
            'non_zero_kurtosis': kurtosis(non_zero_values),
            'non_zero_median': non_zero_values.median(),
            'non_zero_q1': np.percentile(non_zero_values, q=25),
            'non_zero_q3': np.percentile(non_zero_values, q=75),
            'non_zero_gmean': gmean(non_zero_values),
            'non_zero_log_mean': np.log1p(non_zero_values).mean(),
            'non_zero_log_std': np.log1p(non_zero_values).std(),
            'non_zero_log_max': np.log1p(non_zero_values).max(),
            'non_zero_log_min': np.log1p(non_zero_values).min(),
            'non_zero_log_sum': np.log1p(non_zero_values).sum(),
            'non_zero_log_skewness': skew(np.log1p(non_zero_values)),
            'non_zero_log_kurtosis': kurtosis(np.log1p(non_zero_values)),
            'non_zero_log_median': np.log1p(non_zero_values).median(),
            'non_zero_log_q1': np.percentile(np.log1p(non_zero_values), q=25),
            'non_zero_log_q3': np.percentile(np.log1p(non_zero_values), q=75),
            'non_zero_log_gmean': gmean(np.log1p(non_zero_values)),
            'non_zero_count': non_zero_values.count(),
            'non_zero_fraction': non_zero_values.count() / row.count()
        }
    return pd.Series(aggs, index=list(aggs.keys()))

# In[ ]:


eng_features = train.iloc[:, 2:].progress_apply(aggregate_row, axis=1)
eng_features_test = test.iloc[:, 1:].progress_apply(aggregate_row, axis=1)

# # Adversarial validation

# We will now perform adversarial validation using *only* the previously defined statistical features.

# In[ ]:


train_matrix = np.hstack([
    eng_features.values
])

test_matrix = np.hstack([
    eng_features_test.values
])

# In[ ]:


lgb_params = {
    'objective': 'binary',
    'metric': 'auc',
    'learning_rate': 0.1,
    'num_leaves': 16,
    'max_depth': -1,
    'min_child_samples': 1,
    'max_bin': 300,
    'subsample': 1.0,
    'subsample_freq': 1,
    'colsample_bytree': 0.5,
    'min_child_weight': 10,
    'reg_lambda': 0.1,
    'reg_alpha': 0.0,
    'scale_pos_weight': 1,
    'zero_as_missing': False,
    'num_threads': -1,
}

adversarial_x = np.vstack([
    train_matrix,
    test_matrix
])
adversarial_y = np.ones(len(adversarial_x))
adversarial_y[:len(train_matrix)] = 0

cv = KFold(n_splits=5, random_state=100, shuffle=True)
train_preds = np.zeros((len(adversarial_x)))

for i, (train_index, valid_index) in enumerate(cv.split(adversarial_y)):
    print(f'Fold {i}')
    
    dtrain = lgb.Dataset(adversarial_x[train_index], 
                         label=adversarial_y[train_index])
    dvalid = lgb.Dataset(adversarial_x[valid_index], 
                         label=adversarial_y[valid_index])
    
    evals_result = {}
    model = lgb.train(lgb_params, dtrain,
                      num_boost_round=10000, 
                      valid_sets=[dtrain, dvalid],
                      valid_names=['train', 'valid'],
                      early_stopping_rounds=100, 
                      verbose_eval=2000, 
                      evals_result=evals_result)

    valid_preds = model.predict(adversarial_x[valid_index])
    train_preds[valid_index] = valid_preds
    
print('Overall ROC AUC', roc_auc_score(adversarial_y, train_preds))

# This yields a horrible score of 97% ROC AUC. And that is only with statistical features! But at least it can not perfectly distinguish between the sets. Let's plot the predictions of the model against the train set.
# 
# If the train set is a time series, we might see more incorrectly classified samples in the end of the train series.

# In[ ]:


predictions = pd.Series(train_preds[:len(train)])
predictions_sample = predictions.sample(frac=1)
sns.jointplot(predictions_sample.index, predictions_sample, size=10, stat_func=None,
              marginal_kws=dict(bins=15), joint_kws=dict(s=3))

# That does not seem to be the case. But there are quite some samples which the model mistakes as from the test set. With these weights we can define a weighted metric function. That is weighted RMSLE in our case. I will also save the weights so you can conveniently download them and try it in your own model.

# In[ ]:


np.save('weights.npy', predictions.values)

# In[ ]:


weights = predictions + 0.1
weights = weights / weights.mean()

def weighted_rmsle(y_true, y_pred, index):
    errors = (np.log1p(y_true) - np.log1p(y_pred)) ** 2
    errors = errors * weights[index]
    
    return np.sqrt(np.mean(errors))

def rmsle(y_true, y_pred):
    errors = (np.log1p(y_true) - np.log1p(y_pred)) ** 2

    return np.sqrt(np.mean(errors))

# What we do here is first add 0.1 to the predictions (so that samples are never completely ignored by our metric function, even if the adversarial classifier marks them as "clearly from the train set"). 
# 
# Afterwards, we divide the weights by their mean to center them around 1, so that samples are weighted by 1 on average. This is done to keep the weighted RMSLE on the same scale as regular RMSLE. 
# 
# Finally, we define the weighted metric by multiplying the error of each sample by the corresponding weight before averaging it. Recall that RMSLE is defined as
# 
# $$\text{RMSLE} = \sqrt{\frac{1}{n}\sum_{i=0}^{n}(\log(1+\hat{y}_i)-\log(1+y_i))^2}$$
# 
# then weighted RMSLE is defined as 
# 
# $$\text{weighted RMSLE} = \sqrt{\frac{1}{n}\sum_{i=0}^{n}(\log(1+\hat{y}_i)-\log(1+y_i))^2\cdot w_i}$$ 
# 
# where $w$ is a vector of the (processed) outputs of our adversarial validation model which should have a mean of 1.

# # Training a regular regressor

# We will now train a regular regressor and see how well our new validaton method fares. Regarding feature selection I did:
# - dropping of duplicate columns
# - dropping of columns with zero variance

# In[ ]:


x_train = train.iloc[:, 2:].values
x_test = test.iloc[:, 1:].values
y_train = np.log(train['target'])

_, unique_indices = np.unique(x_train, return_index=True, axis=1)
variance_greater_zero = x_train.var(axis=0) > 0

mask = np.zeros(x_train.shape[1], dtype=bool)
mask[unique_indices] = True
mask[variance_greater_zero] = True

x_train = x_train[:, mask]
x_test = x_test[:, mask]

x_train.shape, x_test.shape

# There is also some very basic feature engineering with decompositon features using SVD and ICA to keep the regressor (more or less) competitive.

# In[ ]:


decomposer = FeatureUnion([
    ('svd', TruncatedSVD(n_components=50, random_state=100)),
    ('ica', FastICA(n_components=20, random_state=100))
])

decomposed_train = decomposer.fit_transform(x_train)
decomposed_test = decomposer.transform(x_test)

# In[ ]:


train_matrix = np.hstack([
    eng_features.values,
    decomposed_train
])
test_matrix = np.hstack([
    eng_features_test.values,
    decomposed_test
])

# Hyperparameters are the same as with the adversarial validation model and barely tuned on this input.

# In[ ]:


lgb_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'learning_rate': 0.01,
    'num_leaves': 16,
    'max_depth': -1,
    'min_child_samples': 1,
    'max_bin': 300,
    'subsample': 1.0,
    'subsample_freq': 1,
    'colsample_bytree': 0.5,
    'min_child_weight': 10,
    'reg_lambda': 0.1,
    'reg_alpha': 0.0,
    'scale_pos_weight': 1,
    'zero_as_missing': False,
    'num_threads': -1,
}

# I also use a spin on KFold which splits the training set based on target value. This somewhat decreases the error differences between multiple folds, but I didn't notice it that much. The implementation is from [here](https://www.kaggle.com/c/santander-value-prediction-challenge/discussion/59299).

# In[ ]:


class KFoldByTargetValue(BaseCrossValidator):
    def __init__(self, n_splits=3, shuffle=False, random_state=None):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def _iter_test_indices(self, X, y=None, groups=None):
        n_samples = X.shape[0]
        indices = np.arange(n_samples)

        sorted_idx_vals = sorted(zip(indices, X), key=lambda x: x[1])
        indices = [idx for idx, val in sorted_idx_vals]

        for split_start in range(self.n_splits):
            split_indeces = indices[split_start::self.n_splits]
            yield split_indeces

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_split

# Finally, train the model. We will measure regular RMSLE and weighted RMSLE. I will also submit the predictions to the leaderboard and compare scores.
# 
# Note that you should *not* tune parameters like the 0.1 minimum prediction so that the weighted RMSLE is equal to the leaderboard score. Or if you do, be very careful with it. You will overfit to the behaviour of this specific models difference from training set to test set. Which might make the metric a bad measurement for other models.
# 
# It is only supposed to be a rough estimate of the LB score which regards which samples from the training distribution are similar to those from the test distribution and weights them accordingly higher.

# In[ ]:


cv = KFoldByTargetValue(n_splits=5, shuffle=True, random_state=100)

train_preds = np.zeros((len(train)))
test_preds = np.zeros((len(test)))

for i, (train_index, valid_index) in enumerate(cv.split(y_train)):
    print(f'Fold {i}')
    
    dtrain = lgb.Dataset(train_matrix[train_index], 
                         label=y_train[train_index])
    dvalid = lgb.Dataset(train_matrix[valid_index], 
                         label=y_train[valid_index])
    
    evals_result = {}
    model = lgb.train(lgb_params, dtrain,
                      num_boost_round=10000, 
                      valid_sets=[dtrain, dvalid],
                      valid_names=['train', 'valid'],
                      early_stopping_rounds=100, 
                      verbose_eval=2000, 
                      evals_result=evals_result)
    
    valid_preds = np.exp(model.predict(train_matrix[valid_index]))
    test_preds += np.exp(model.predict(test_matrix)) / cv.n_splits
    
    train_preds[valid_index] = valid_preds
    
    print('RMSLE: ', rmsle(np.exp(y_train[valid_index]), valid_preds))
    print('Weighted RMSLE: ', weighted_rmsle(np.exp(y_train[valid_index]), valid_preds, valid_index))
print()
print('Overall RMSLE: ', rmsle(np.exp(y_train), train_preds))
print('Overall Weighted RMSLE: ', weighted_rmsle(np.exp(y_train), train_preds, np.arange(len(train_preds))))

# Notice that the weighted RMSLE is significantly higher than regular RMSLE. This makes sense, because samples that are similar to the distribution are obviously harder to predict than samples which have properties similar to the other samples in the train distribution.

# In[ ]:


submission = pd.DataFrame()
submission['ID'] = test['ID']
submission['target'] = test_preds
submission.to_csv('submission.csv', index=False)

# Finally, submit the predictions and hope the weighted RMSLE is a good approximation of the LB score.

# # Where to go from here
# 
# In this kernel, I showed a different kind of validation. To be honest, I am not sure if this is a good way of validation at all. Please discuss with me in the comments :) I will definitely try this scheme in more elaborate models and see how it compares to the leaderboard score there. And again: *Without* tuning this score for a specific model. That defeats the purpose.
# 
# One way to probably improve the score is by weighting samples more conservatively e. g. start from 1 and add / substract some value based on the predictions of the adversarial classifier like
# 
# $$\text{weights} = 1 + (\text{predictions} - 0.5) * \alpha$$ 
# 
# where 0 < $\alpha$ < 2 and the smaller $\alpha$, the less noticeable is the adversarial weighting.
