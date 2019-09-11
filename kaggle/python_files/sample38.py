#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Import libraries and set desired options
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

# In[ ]:


# A helper function for writing predictions to a file
def write_to_submission_file(predicted_labels, out_file,
                             target='target', index_label="session_id"):
    predicted_df = pd.DataFrame(predicted_labels,
                                index = np.arange(1, predicted_labels.shape[0] + 1),
                                columns=[target])
    predicted_df.to_csv(out_file, index_label=index_label)

# **Read training and test sets, sort train set by session start time.**

# In[ ]:


train_df = pd.read_csv('../input/train_sessions.csv',
                       index_col='session_id', parse_dates=['time1'])
test_df = pd.read_csv('../input/test_sessions.csv',
                      index_col='session_id', parse_dates=['time1'])

# Sort the data by time
train_df = train_df.sort_values(by='time1')

# Look at the first rows of the training set
train_df.head()

# **Transform data into format which can be fed into `CountVectorizer`**

# In[ ]:


sites = ['site%s' % i for i in range(1, 11)]
train_df[sites].fillna(0).astype('int').to_csv('train_sessions_text.txt', 
                                               sep=' ', 
                       index=None, header=None)
test_df[sites].fillna(0).astype('int').to_csv('test_sessions_text.txt', 
                                              sep=' ', 
                       index=None, header=None)

# In[ ]:



# **Fit `CountVectorizer` and transfrom data with it.**

# In[ ]:


cv = CountVectorizer(ngram_range=(1, 3), max_features=50000)
with open('train_sessions_text.txt') as inp_train_file:
    X_train = cv.fit_transform(inp_train_file)
with open('test_sessions_text.txt') as inp_test_file:
    X_test = cv.transform(inp_test_file)
X_train.shape, X_test.shape

# **Save train targets into a separate vector.**

# In[ ]:


y_train = train_df['target'].astype('int').values

# **We'll be performing time series cross-validation, see `sklearn` [TimeSeriesSplit](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html) and [this dicussion](https://stats.stackexchange.com/questions/14099/using-k-fold-cross-validation-for-time-series-model-selection) on StackOverflow.**

# In[ ]:


time_split = TimeSeriesSplit(n_splits=10)

# <img src="https://habrastorage.org/webt/8i/5k/vx/8i5kvxrehatyvf-l3glz_-ymhtw.png" />

# In[ ]:


[(el[0].shape, el[1].shape) for el in time_split.split(X_train)]

# **Perform time series cross-validation with logistic regression.**

# In[ ]:


logit = LogisticRegression(C=1, random_state=17, solver='liblinear')

# In[ ]:



cv_scores = cross_val_score(logit, X_train, y_train, cv=time_split, 
                            scoring='roc_auc', n_jobs=1) # hangs with n_jobs > 1, and locally this runs much faster

# In[ ]:


cv_scores, cv_scores.mean()

# **Train logistic regression with all training data, make predictions for test set and form a submission file.**

# In[ ]:


logit.fit(X_train, y_train)

# In[ ]:


logit_test_pred = logit.predict_proba(X_test)[:, 1]
write_to_submission_file(logit_test_pred, 'subm1.csv') # 0.91288

# **Now we'll add some time features: indicators of morning, day, evening and night.**

# In[ ]:


def add_time_features(df, X_sparse):
    hour = df['time1'].apply(lambda ts: ts.hour)
    morning = ((hour >= 7) & (hour <= 11)).astype('int')
    day = ((hour >= 12) & (hour <= 18)).astype('int')
    evening = ((hour >= 19) & (hour <= 23)).astype('int')
    night = ((hour >= 0) & (hour <= 6)).astype('int')
    X = hstack([X_sparse, morning.values.reshape(-1, 1), 
                day.values.reshape(-1, 1), evening.values.reshape(-1, 1), 
                night.values.reshape(-1, 1)])
    return X

# In[ ]:


X_train_new = add_time_features(train_df.fillna(0), X_train)
X_test_new = add_time_features(test_df.fillna(0), X_test)

# In[ ]:


X_train_new.shape, X_test_new.shape

# **Performing time series cross-validation, we see an improvement in ROC AUC.**

# In[ ]:


cv_scores = cross_val_score(logit, X_train_new, y_train, cv=time_split, 
                            scoring='roc_auc', n_jobs=1) # hangs with n_jobs > 1, and locally this runs much faster

# In[ ]:


cv_scores, cv_scores.mean()

# **Making a new submission, we notice a leaderboard score improvement as well (0.91288 ->  0.93843). Correlated CV and LB improvements is a good justifications for added features being useful and CV scheme being correct.**

# In[ ]:


logit.fit(X_train_new, y_train)

# In[ ]:


logit_test_pred2 = logit.predict_proba(X_test_new)[:, 1]
write_to_submission_file(logit_test_pred2, 'subm2.csv') # 0.93843

# **Now we tune regularization parameter `C`.**

# In[ ]:


c_values = np.logspace(-2, 2, 10)

logit_grid_searcher = GridSearchCV(estimator=logit, param_grid={'C': c_values},
                                  scoring='roc_auc', n_jobs=1, cv=time_split, verbose=1)

# In[ ]:


logit_grid_searcher.fit(X_train_new, y_train) # WTF? Locally, it's 3min 30s

# In[ ]:


logit_grid_searcher.best_score_, logit_grid_searcher.best_params_

# In[ ]:


logit_test_pred3 = logit_grid_searcher.predict_proba(X_test_new)[:, 1]
write_to_submission_file(logit_test_pred3, 'subm3.csv') # 0.94242

# **Again, we notice an improvement in both cross-validation score and LB score. Now that you've settled a correct cross-validation scheme, go on with feature engineering! Good luck!**
