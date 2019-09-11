#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import matplotlib.pyplot as plt

import seaborn as sns

from IPython.display import HTML

import os
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
# Any results you write to the current directory are saved as output.

# ### Reading the whole data
# Note: LARGE file, decrease the bit of float to 32 to avoid crush.
# (It took couple minutes to load the training file alone)

# In[ ]:


train = pd.read_csv('../input/train.csv',dtype={'acoustic_data': np.int16, 'time_to_failure': np.float32})
train.head()

# ### A look at the whole data
# - There are only about 16 "earthquakes" happened in the training data. 
# - Each "earthquake" seems to happend right after the unusually large activity in the acoustic data. 

# In[ ]:


measurement = train['acoustic_data'].values[::100]
ttf = train['time_to_failure'].values[::100]
fig, ax1 = plt.subplots(figsize=(12, 8))
# plt.title(title)
plt.plot(measurement, color='black')
ax1.set_ylabel('acoustic data', color='black')
plt.legend(['acoustic data'], loc=(0.01, 0.95))
ax2 = ax1.twinx()
plt.plot(ttf, color='blue')
ax2.set_ylabel('time to failure', color='blue')
plt.legend(['time to failure'], loc=(0.01, 0.9))
plt.grid(True)

del measurement
del ttf

# ### First ~10% of training data

# In[ ]:


measurement = train['acoustic_data'].values[:150000]
ttf = train['time_to_failure'].values[:150000]
fig, ax1 = plt.subplots(figsize=(12, 8))
# plt.title(title)
plt.plot(measurement, color='black')
ax1.set_ylabel('acoustic data', color='black')
plt.legend(['acoustic data'], loc=(0.01, 0.95))
ax2 = ax1.twinx()
plt.plot(ttf, color='blue')
ax2.set_ylabel('time to failure', color='blue')
plt.legend(['time to failure'], loc=(0.01, 0.9))
plt.grid(True)

del measurement
del ttf

# ## Basic Idea
# 1. Divide the training data into segments the same length as test files (which all have 150,000 rows). 
# 2. Do feature engineering on both training segments and testing segments.
# 3. The feature to predict for each segment is the "time-to-failure" at the last row of each segment.

# In[ ]:


tests = os.listdir("../input/test/")
test1 = pd.read_csv("../input/test/"+tests[2])
test1.shape

# In[ ]:


print('There are %d test segments in the data.' %len(tests))

# There are 4194 training segments if we divide the training data back-to-back. But there's no need to keep to a back-to-back dividing, so we can randomly select the starting point and generate more segments as needed. But keep in mind if we divide the training segments randomly, these segments are NOT independent.

# In[ ]:


np.floor(train.shape[0]/test1.shape[0])

# In[ ]:


rows = 150000
segments = int(np.floor(train.shape[0]/rows))
train_X = pd.DataFrame(dtype=np.float64)
#train_y = pd.DataFrame(index=range(segments), dtype=np.float64, columns=['time_to_failure'])

# In[ ]:


def build_features(seg_id,seg,train_X):
    from scipy.stats import kurtosis
    data = seg['acoustic_data']
    features = {}
    features['mean'] = seg['acoustic_data'].mean()
    features['abs_mean']= seg['acoustic_data'].abs().mean()
    features['std'] = seg['acoustic_data'].std()
    features['min'] = seg['acoustic_data'].min()
    features['max'] = seg['acoustic_data'].max()
    features['range'] = features['max'] - features['min']
    features['abs_min'] = seg['acoustic_data'].abs().min()
    features['abs_max'] = seg['acoustic_data'].abs().max()
    features['argmax'] = seg['acoustic_data'].argmax()
    features['argmin'] = seg['acoustic_data'].argmin()
    features['distance'] = np.abs(features['argmax'] - features['argmin'])
    features['skew'] = seg['acoustic_data'].skew()
    features['kurtosis'] = seg['acoustic_data'].kurt()
    ffts = np.fft.fft(seg['acoustic_data'].values)
    reals = np.array([x.real for x in ffts])
    imags = np.array([x.imag for x in ffts])
    features['real_fft_mean'] = np.mean(reals)
    features['real_fft_abs_mean']= np.mean(np.abs(reals))
    features['real_fft_std'] = np.std(reals)
    features['real_fft_min'] = np.min(reals)
    features['real_fft_max'] = np.max(reals)
    features['real_fft_range'] = features['real_fft_max'] - features['real_fft_min']
    features['real_fft_abs_min'] = np.min(np.abs(reals))
    features['real_fft_abs_max'] = np.max(np.abs(reals))
    features['real_fft_argmax'] = np.argmax(reals)
    features['real_fft_argmin'] = np.argmin(reals)
    features['real_fft_distance'] = np.abs(features['real_fft_argmax'] - features['real_fft_argmin'])
    features['real_fft_skew'] = np.skew(reals)
    features['real_fft_kurtosis'] = kurtosis(reals)
    features['imag_fft_mean'] = np.mean(imags)
    features['imag_fft_abs_mean']= np.mean(np.abs(imags))
    features['imag_fft_std'] = np.std(imags)
    features['imag_fft_min'] = np.min(imags)
    features['imag_fft_max'] = np.max(imags)
    features['imag_fft_range'] = features['imag_fft_max'] - features['imag_fft_min']
    features['imag_fft_abs_min'] = np.min(np.abs(imags))
    features['imag_fft_abs_max'] = np.max(np.abs(imags))
    features['imag_fft_argmax'] = np.argmax(imags)
    features['imag_fft_argmin'] = np.argmin(imags)
    features['imag_fft_distance'] = np.abs(features['imag_fft_argmax'] - features['imag_fft_argmin'])
    features['imag_fft_skew'] = np.skew(imags)
    features['imag_fft_kurtosis'] = kurtosis(imags)
    f = pd.DataFrame(list(features.values()))
    f = f.T
    f.columns = list(features.keys())
    train_X = train_X.append(f)
    return train_X

# In[ ]:


train_y = []
for seg_id in range(segments):
    seg = train.iloc[seg_id*rows:seg_id*rows+rows,:]
    train_X = build_features(seg_id, seg, train_X)
    train_y.append(seg['time_to_failure'].values[-1])

train_X = train_X.reset_index(drop=True)

# In[ ]:


i = 0
test_X = pd.DataFrame(dtype=np.float64)
seg_ids = []
from re import sub
for test_file in tests:
    test_data = pd.read_csv("../input/test/"+test_file)
    seg_ids.append(sub('.csv','',test_file))
    test_X = build_features(i, test_data, test_X)
    i += 1
test_X = test_X.reset_index(drop=True)

# In[ ]:


from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=500, criterion='mae', max_depth=20, bootstrap=True, n_jobs=-1)
kfold = GroupKFold(n_splits=10).get_n_splits(train_X, train_y)
results = -1*cross_val_score(model, train_X, train_y, cv=kfold, n_jobs=-1,scoring='neg_mean_absolute_error')
print("Cross validation mae is: %.2f (%.2f)" % (results.mean(), results.std()))

# In[ ]:


model.fit(train_X,train_y)
predictions = model.predict(test_X)
submission = pd.DataFrame({'seg_id':seg_ids,'time_to_failure':predictions})
submission.to_csv('submission.csv',index=False)
