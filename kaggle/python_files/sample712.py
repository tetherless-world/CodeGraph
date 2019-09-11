#!/usr/bin/env python
# coding: utf-8

# The purpose of this kernel is to show that the findings in Scirpus' kernel (https://www.kaggle.com/scirpus/weird-data-structure) is the result of the dataset being limited in range and rounded to 4 decimal places and as a result the "diagonals" can be replicated with completely random data with similar properties. 

# In[ ]:


import pandas as pd
import numpy as np
from pathlib import Path


# In[ ]:


train = pd.DataFrame(np.round(np.random.uniform(-40, 40, (200000, 200)), 4), columns=["var_" + str(i) for i in range(200)])
test = pd.DataFrame(np.round(np.random.uniform(-40, 40, (200000, 200)), 4), columns=["var_" + str(i) for i in range(200)])

# In[ ]:


x = train['var_0'].value_counts()
x = x[x==1].reset_index(drop=False)
x.head()

# In[ ]:


candidates = []
for c in train.columns[1:-1]:
    if(train[train[c] == x['index'][0]].shape[0]==1):
        candidates.append(c)
indexes = []
for c in candidates:
    indexes.append(train[train[c] == x['index'][0]].index.values[0])
y = train.iloc[indexes][candidates]
y.head(y.shape[0])

# In[ ]:


candidates = []
for c in test.columns[1:-1]:
    if(test[test[c] == x['index'][0]].shape[0]==1):
        candidates.append(c)
indexes = []
for c in candidates:
    indexes.append(test[test[c] == x['index'][0]].index.values[0])
y = test.iloc[indexes][candidates]
y.head(y.shape[0])

# In[ ]:


candidates = []
for c in train.columns[1:-1]:
    if(train[train[c] == x['index'][1]].shape[0]==1):
        candidates.append(c)
indexes = []
for c in candidates:
    indexes.append(train[train[c] == x['index'][1]].index.values[0])
y = train.iloc[indexes][candidates]
y.head(y.shape[0])

# In[ ]:


candidates = []
for c in test.columns[1:-1]:
    if(test[test[c] == x['index'][1]].shape[0]==1):
        candidates.append(c)
indexes = []
for c in candidates:
    indexes.append(test[test[c] == x['index'][1]].index.values[0])
y = test.iloc[indexes][candidates]
y.head(y.shape[0])
