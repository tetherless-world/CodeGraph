#!/usr/bin/env python
# coding: utf-8

# # Five thirty Eight Comic Characters

# As per the data description , we have two datasets - Marvel Wikia and DC Wikia . The corresponding story that was published in five thirty eight [blog](https://fivethirtyeight.com/features/women-in-comic-books/) had an comprehensive analysis about the representation of women in the comic world.

# In[ ]:


## importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')
os.listdir()

# In[ ]:


## import dataset:
kaggle=1
if kaggle==1:
    dc=pd.read_csv('../input/dc-wikia-data.csv')
    marvel=pd.read_csv('../input/marvel-wikia-data.csv')
else:
    dc=pd.read_csv('dc-wikia-data.csv')
    marvel=pd.read_csv('marvel-wikia-data.csv')
    

# In[ ]:


## Glimpse at the data:
dc.head()

# In[ ]:


marvel.head()

# For our analysis purpose we join both these datasets.Before that ,we create a separate column in each of the datasets so that we can identify the characters .

# In[ ]:


marvel['WORLD']='Marvel'
dc['WORLD']='DC'

# In[ ]:


marvel.info()

# In[ ]:


dc.info()

# In[ ]:


print(f'There are {dc.page_id.nunique()} DC Characters and {marvel.page_id.nunique()} marvel characters')

# In[ ]:


data=pd.concat([marvel,dc])

# In[ ]:


data.shape

# Lets check which colums have null values,

# In[ ]:


data.isnull().sum().sort_values(ascending=False)

# Lets check the distribution of the characters based on gender.

# In[ ]:


sex=data['SEX'].value_counts()
print('% Distribution of the characters based on gender')
print(sex/len(data['SEX'])*100)

# 70 % of the characters represented in the data are male whereas 24 % are female.

# Lets check whether these characters are good/bad/neutral.

# **work in progress**
