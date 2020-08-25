
# coding: utf-8

# In[85]:


import features
import numpy as np
import pandas as pd

rawdata = pd.read_csv('../kaggle_datasets/flight-delays/flights.csv', encoding='latin-1', error_bad_lines=False)
name = "flight-delays"
rawdata.info()
rawdata.isnull().sum()


# In[86]:


drop = "AIRLINE_DELAY"
encoded = rawdata.copy(deep = True)
#encoded = encoded.fillna(0)
#encoded.dropna(axis=0, inplace=True)
for x in encoded:
    if encoded[x].dtype == "object":
        encoded[x] = encoded[x].astype('category').cat.codes
if encoded.shape[0] > 25000:
    encoded = encoded.sample(n = 25000, axis=0) 
encoded.info()


# In[87]:


Y = encoded[drop]
X = encoded.drop([drop], axis = 1)

Y = Y.to_numpy()
X = X.to_numpy()
vec = features.getFeatures(X, Y, name)
features.serialize(name, vec)
vec

