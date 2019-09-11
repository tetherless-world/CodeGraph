#!/usr/bin/env python
# coding: utf-8

# **Thank you everyone for showing your appreciation and support. It's my first gold medal in kernels and I hope to publish far better kernels than this in near future.**

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from functools import reduce
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

# ## Blend with one rank weighted submission [0.8 LB]

# In[ ]:


M1 = pd.read_csv('../input/diversity/LGBM.798.csv')
M2 = pd.read_csv('../input/ingredients/WEIGHT_AVERAGE_RANK2.csv')
M3 = pd.read_csv('../input/neural/sub_nn.csv')
M4 = pd.read_csv('../input/genetic/pure_submission.csv')
M5 = pd.read_csv('../input/diversity/xgb.796.csv')

# In[ ]:


# Function for merging dataframes efficiently 
def merge_dataframes(dfs, merge_keys):
    dfs_merged = reduce(lambda left,right: pd.merge(left, right, on=merge_keys), dfs)
    return dfs_merged

# In[ ]:


dfs = [M1,M2,M3,M4,M5]
merge_keys=['SK_ID_CURR']
df = merge_dataframes(dfs, merge_keys=merge_keys)

# In[ ]:


df.columns = ['SK_ID_CURR','T1','T2','T3','T4','T5']
df.head()

# In[ ]:


pred_prob = 0.5 * df['T2'] + 0.5 * df['T1']
pred_prob.head()

# In[ ]:


sub = pd.DataFrame()
sub['SK_ID_CURR'] = df['SK_ID_CURR']
sub['target']= pred_prob

# In[ ]:


sub.to_csv('ldit.csv', index=False)

# ## Diversified blend [0.799 LB]
# 
# 
# **The blending ingredients are taken from three different type of models.**

# In[ ]:


B_prob = 0.6 * df['T1'] + 0.2 * df['T3'] + 0.2 * df['T4']

# In[ ]:


B_prob.head()

# In[ ]:


SUB = pd.DataFrame()
SUB['SK_ID_CURR'] = df['SK_ID_CURR']
SUB['TARGET'] = B_prob
SUB.to_csv('Blendss.csv', index=False)

# ## Blending lowest correlated models

# In[ ]:


df_c = df.copy()
df_c = df.drop(['SK_ID_CURR'],axis=1)
Corr_Mat = df_c.corr()
print(Corr_Mat) # Correlation matrix of five submission files
sns.heatmap(Corr_Mat)

# In[ ]:


corr_pred = 0.6 * df['T2'] + 0.05 * df['T3'] + 0.05 * df['T4'] + 0.1 * df['T5'] + 0.2 * df['T1']
corr_pred.head()

# In[ ]:


SuB = pd.DataFrame()
SuB['SK_ID_CURR'] = df['SK_ID_CURR']
SuB['TARGET'] = corr_pred
SuB.to_csv('corr_blend.csv', index=False)

# 

# In[ ]:



