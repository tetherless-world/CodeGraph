#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from sklearn.model_selection import KFold
import warnings
import gc
import time
import sys
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')
from sklearn import metrics

plt.style.use('seaborn')
sns.set(font_scale=2)
pd.set_option('display.max_columns', 500)

# In[ ]:


DEBUG = False

if DEBUG:
    NROWS = 100000
else:
    NROWS = None

# # 1. Read dataset

# In[ ]:


train = pd.read_csv('../input/train.csv', nrows=NROWS)

# ## 1.1 Target  check

# In[ ]:


train['target'].value_counts().plot.bar()

# - This competiiton is Imbalanced target competition.
# - You can check similar competitions, Porto, Homecredit competition.
# - Specially, Porto also gave use anonymized dataset. 
# - https://www.kaggle.com/c/home-credit-default-risk
# - https://www.kaggle.com/c/porto-seguro-safe-driver-prediction

# ## 1.2 Null data check

# In[ ]:


# checking missing data
total = train.isnull().sum().sort_values(ascending = False)
percent = (train.isnull().sum()/train.isnull().count()*100).sort_values(ascending = False)
missing_train_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

# In[ ]:


missing_train_data

# - There is no missing values.
# - Because we don't know the exact meaning of variables, we need to check some values as null value.

# # 2. Exploratory Data Analysis

# - Before EDA, let's group the features into category and non-category based on the number of uniqueness.

# In[ ]:


for col in train.columns[2:]:
    print("Number of unique values of {} : {}".format(col, train[col].nunique()))

# - Oh, Most features have more than thousands of values for each variable except var_68 (435)

# - Let's see var 68

# In[ ]:


train['var_68'].value_counts()

# - It also has float numbers. 
# - Uncovering these values will be intersting job!
# - Multiplying and dividing with some values can make the hidden categories, See Radder work(https://www.kaggle.com/raddar/target-true-meaning-revealed)

# # 2.1 Correlation

# In[ ]:


corr = train.corr()

# In[ ]:


abs(corr['target']).sort_values(ascending=False)

# - The largest correlation value is 0.08
# - Actually, the target is binary and variables are continous, so correlation is not enough to judge. Let's see the distribution!

# # 2.2 Distribution regarding to target

# In[ ]:


target_mask = train['target'] == 1
non_target_mask = train['target'] == 0 

# In[ ]:


from scipy.stats import ks_2samp

# In[ ]:


statistic, pvalue = ks_2samp(train.loc[non_target_mask, col], train.loc[target_mask, col])

# In[ ]:


statistics_array = []
for col in train.columns[2:]:
    statistic, pvalue = ks_2samp(train.loc[non_target_mask, col], train.loc[target_mask, col])
    statistics_array.append(statistic)
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    sns.kdeplot(train.loc[non_target_mask, col], ax=ax, label='Target == 0')
    sns.kdeplot(train.loc[target_mask, col], ax=ax, label='Target == 1')

    ax.set_title('name: {}, statistics: {:.5f}, pvalue: {:5f}'.format(col, statistic, pvalue))
    plt.show()

# # TODO

# - As you know, Santander hosted other competition 6 month before.
# - So you can check this competition. https://www.kaggle.com/c/santander-value-prediction-challenge

# - I will do time series analysis for this dataset.

# In[ ]:




# In[ ]:




# In[ ]:



