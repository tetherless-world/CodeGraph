#!/usr/bin/env python
# coding: utf-8

# # LOFO Feature Importance
# https://github.com/aerdem4/lofo-importance

# Considering @cdeotte's model which trains different Logistic Regression models for each wheezy-copper-turtle-magic value, we see that we get different feature importances for each wheezy-copper-turtle-magic split.

# In[1]:



# In[2]:


import numpy as np
import pandas as pd

df = pd.read_csv("../input/train.csv", index_col='id')
df['wheezy-copper-turtle-magic'] = df['wheezy-copper-turtle-magic'].astype('category')
df.shape

# ### Top 20 Features for wheezy-copper-turtle-magic = 0

# In[3]:


from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
from lofo import LOFOImportance, FLOFOImportance, plot_importance


features = [c for c in df.columns if c not in ['id', 'target', 'wheezy-copper-turtle-magic']]

lofo_imp = LOFOImportance(df[df['wheezy-copper-turtle-magic'] == 0], features=features, target="target", 
                          cv=StratifiedKFold(n_splits=5, random_state=42, shuffle=True), scoring="roc_auc",
                          model=LogisticRegression(solver='liblinear',penalty='l2',C=1.0), n_jobs=1)
importance_df0 = lofo_imp.get_importance()
plot_importance(importance_df0.head(20), figsize=(12, 12))

# ### Top 20 Features for wheezy-copper-turtle-magic = 1

# In[4]:


lofo_imp = LOFOImportance(df[df['wheezy-copper-turtle-magic'] == 1], features=features, target="target", 
                          cv=StratifiedKFold(n_splits=5, random_state=42, shuffle=True), scoring="roc_auc",
                          model=LogisticRegression(solver='liblinear',penalty='l2',C=1.0), n_jobs=1)
importance_df1 = lofo_imp.get_importance()
plot_importance(importance_df1.head(20), figsize=(12, 12))

# ### Top 20 Features for wheezy-copper-turtle-magic = 2

# In[5]:


lofo_imp = LOFOImportance(df[df['wheezy-copper-turtle-magic'] == 2], features=features, target="target", 
                          cv=StratifiedKFold(n_splits=5, random_state=42, shuffle=True), scoring="roc_auc",
                          model=LogisticRegression(solver='liblinear',penalty='l2',C=1.0), n_jobs=1)
importance_df2 = lofo_imp.get_importance()
plot_importance(importance_df2.head(20), figsize=(12, 12))

# In[ ]:



