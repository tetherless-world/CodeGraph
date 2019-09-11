#!/usr/bin/env python
# coding: utf-8

# Thanks for Vladislav and his kernel https://www.kaggle.com/speedwagon/are-magics-interconnected  
# Just wanna dive deeper into this interconnection thing.

# ## 1. Interconnection between magics

# In[8]:


import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from tqdm import tqdm_notebook

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")

train_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')

# In[2]:


train_data.head()

# In[4]:


cols = [c for c in train_data.columns if c not in ['id', 'target', 'wheezy-copper-turtle-magic']]

magic_idx = []
magic_pred = []
magic_auc = []

for i in tqdm_notebook(range(512)):
    train2 = train_data[train_data['wheezy-copper-turtle-magic'] == i]
    train2.reset_index(drop=True, inplace=True)

    clf = LogisticRegression(solver='liblinear', penalty='l1', C=0.05)
    clf.fit(train2[cols], train2['target'])

    for j in range(0, 512):
        val = train_data[train_data['wheezy-copper-turtle-magic'] == j]
        preds = clf.predict_proba(val[cols])[:, 1]
        auc = roc_auc_score(val['target'], preds)
        magic_idx.append(i)
        magic_pred.append(j)
        magic_auc.append(auc)

# In[5]:


magic_mx = pd.DataFrame({'magic_fit':magic_idx, 'magic_pred':magic_pred, 'auc':magic_auc})
magic_mx = magic_mx[['magic_fit', 'magic_pred', 'auc']]
magic_mx.head()

# This illustrates the auc when we train the model on magic_fit and predict magic_pred.  
# We can plot that as heatmap shown as following:

# In[9]:


magic_mx_pt = pd.pivot_table(magic_mx, index='magic_fit', columns='magic_pred', values='auc')

plt.style.use({'figure.figsize':(18, 15), 'font.size':15}) # set the size of plots
sns.heatmap(magic_mx_pt)

# In[10]:


sns.heatmap(magic_mx_pt > 0.6) # higher correlated

# ## 2. Feature and Target correlation under different magics

# In senkin13's EDA https://www.kaggle.com/senkin13/eda-starter (Thanks to him), we saw that the correlation between the features and target were very low. But how will it be under different magics? Let's see.

# ### Highly correlated

# In[11]:


cols = [c for c in train_data.columns if c not in ['id', 'wheezy-copper-turtle-magic']]

magic_num = []
col_name = []
corr_ls = []
for i in tqdm_notebook(range(512)):
    tmp = train_data[train_data['wheezy-copper-turtle-magic'] == i]
    correlations = tmp[cols].corr().abs().unstack().sort_values(kind="quicksort", ascending=False).reset_index()
    correlations = correlations[correlations['level_0'] != correlations['level_1']]
    corr = correlations[correlations['level_0'] == 'target']
    magic_num.append(i)
    col_name.append(corr['level_1'].iloc[0])
    corr_ls.append(corr[0].iloc[0])

# I just put the most highly correlated feature in the table here. (either positive or negative correlated)

# In[12]:


corr_under_magic = pd.DataFrame({'magic':magic_num, 'feature':col_name, 'corr':corr_ls})
corr_under_magic.head()

# In[14]:


corr_under_magic['feature'].nunique()

# In[17]:


corr_under_magic['feature'].value_counts()

# Well, looks like there might be something with "beady-lilac-hornet-expert"?

# ### Uncorrelated ones

# In[20]:


cols = [c for c in train_data.columns if c not in ['id', 'wheezy-copper-turtle-magic']]

magic_num = []
col_name = []
corr_ls = []
for i in tqdm_notebook(range(512)):
    tmp = train_data[train_data['wheezy-copper-turtle-magic'] == i]
    correlations = tmp[cols].corr().abs().unstack().sort_values(kind="quicksort", ascending=True).reset_index()
    correlations = correlations[correlations['level_0'] != correlations['level_1']]
    corr = correlations[correlations['level_0'] == 'target']
    magic_num.append(i)
    col_name.append(corr['level_1'].iloc[0])
    corr_ls.append(corr[0].iloc[0])

# In[21]:


corr_under_magic = pd.DataFrame({'magic':magic_num, 'feature':col_name, 'corr':corr_ls})
corr_under_magic.head()

# In[22]:


corr_under_magic['feature'].nunique()

# In[23]:


corr_under_magic['feature'].value_counts()

# Looks like there is something here, but I just can't name it. Maybe I'm just not clever enough to find it out. Hope it could give you guys some insight. :)

# In[ ]:



