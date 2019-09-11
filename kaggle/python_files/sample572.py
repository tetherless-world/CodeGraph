#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

# In[19]:


df = pd.read_csv('../input/train.csv')

# In[20]:


df.head()

# In[21]:


test = pd.read_csv('../input/test.csv')

# In[22]:


test.head()

# In[23]:


df.info()

# In[24]:


df.shape, test.shape

# In[25]:


df = pd.concat([df, test], axis=0)

# In[26]:


for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].astype('category').cat.codes

# In[27]:


df

# In[28]:


df.dtypes

# In[29]:


df.shape

# In[30]:


df.head()

# In[31]:


df

# In[35]:


df.loc[2]

# In[33]:


df.reset_index(inplace=True)

# In[38]:


df.head()

# In[40]:




# In[42]:


df.fillna(value=-1, inplace=True)

# In[43]:


df.head()

# In[46]:


df['nota_mat'] = np.where(df['nota_mat']==-1,np.nan, df['nota_mat'])

# In[47]:


df

# In[48]:


test = df[df['nota_mat'].isnull()]
df = df[~df['nota_mat'].isnull()]

# In[49]:


df.shape, test.shape

# In[50]:


from sklearn.model_selection import train_test_split



# In[51]:


train, valid = train_test_split(df, test_size=0.20, random_state=42)

# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



