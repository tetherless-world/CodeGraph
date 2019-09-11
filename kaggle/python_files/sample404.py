#!/usr/bin/env python
# coding: utf-8

# ***Background***
# 
# - As you know that, The size of the data is large and it takes a long time and memory error occurs. 
# - So I share a study of how to save time and how to reduce memory. 
# - This is my first field of study. So if there is a mistake or something to add, please add it as a comment.

# ***OUTLINE***
# 
# - Deleting unused variables and gc.collect()  
# - Presetting the datatypes  
# - Importing selected rows of the a file. 
# - Importing just selected columns  
# - Using debug mode 
# - Lightgbm: prevent RAM spike (explode) at the init training

# ***source***
# - https://www.kaggle.com/yuliagm/how-to-work-with-big-datasets-on-16g-ram-dask
# - https://www.kaggle.com/gemartin/load-data-reduce-memory-usage
# - https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection/discussion/53773
# 
# For reference, the materials here are tailored to PBUG based on the material at https://www.kaggle.com/yuliagm/how-to-work-with-big-datasets-on-16g-ram-dask. So if you do upvote, please upvote the article above.

# In[ ]:


import pandas as pd #Analysis 
import matplotlib.pyplot as plt #Visulization
import seaborn as sns #Visulization
import numpy as np #Analysis 
from scipy.stats import norm #Analysis 
from sklearn.preprocessing import StandardScaler #Analysis 
from scipy import stats #Analysis 
import warnings 
warnings.filterwarnings('ignore')
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import plotly.figure_factory as ff
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
import datetime
import os
import time
import gc

# ### 1. Deleting unused variables and gc.collect()
# 
# Unlike other languages, Python does not efficiently utilize memory. Variables that we do not use, or that we use or discard, also occupy memory. So we have to keep in mind two things.
# 
# 1. Unused variables are deleted using del.
# 
# 2. After del deleting it, it is surely removed from memory through the command gc.collect()

# In[ ]:


df_train = pd.read_csv('../input/train.csv')
df_test  = pd.read_csv('../input/test.csv')

df_train_sample = df_train.copy()
del df_train_sample
gc.collect()

# ### 2. Presetting the datatypes 
# 
# Python automatically reads the data type, which causes a lot of memory waste. So if we know in advance the memory we will set up, we can use it much more effectively.

# In[ ]:


df_train.head()

# In[ ]:


df_train.tail()

# In[ ]:


df_train.shape

# In[ ]:


dtypes = {
        'Id'                : 'uint32',
        'groupId'           : 'uint32',
        'matchId'           : 'uint16',
        'assists'           : 'uint8',
        'boosts'            : 'uint8',
        'damageDealt'       : 'float16',
        'DBNOs'             : 'uint8',
        'headshotKills'     : 'uint8', 
        'heals'             : 'uint8',    
        'killPlace'         : 'uint8',    
        'killPoints'        : 'uint16',    
        'kills'             : 'uint8',    
        'killStreaks'       : 'uint8',    
        'longestKill'       : 'float16',    
        'maxPlace'          : 'uint8',    
        'numGroups'         : 'uint8',    
        'revives'           : 'uint8',    
        'rideDistance'      : 'float16',    
        'roadKills'         : 'uint8',    
        'swimDistance'      : 'float16',    
        'teamKills'         : 'uint8',    
        'vehicleDestroys'   : 'uint8',    
        'walkDistance'      : 'float16',    
        'weaponsAcquired'   : 'uint8',    
        'winPoints'         : 'uint8', 
        'winPlacePerc'      : 'float16' 
}

# In[ ]:


train_dtypes = pd.read_csv('../input/train.csv', dtype=dtypes)
df_train = pd.read_csv('../input/train.csv')

#check datatypes:
train_dtypes.info()

# In[ ]:


#check datatypes:
df_train.info()

# You saved almost five times the memory.
# **864.3 MB -> 162.1 MB**

# If you do not want to do the above, it's a good idea to use kaggler's code.

# In[ ]:


# Memory saving function credit to https://www.kaggle.com/gemartin/load-data-reduce-memory-usage
def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    #start_mem = df.memory_usage().sum() / 1024**2
    #print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    #end_mem = df.memory_usage().sum() / 1024**2
    #print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    #print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df

# In[ ]:


df_train = reduce_mem_usage(df_train)
df_train.info()

# You saved almost five times the memory. **864.3 MB -> 178.7  MB** . almost similiar

# ### 3. Importing selected rows of the a file. 
# If the size of the data is large as in this competition, you can try sampling. If you check code working well, use selected rows not all rows. ( it is called debug )

# a) Select number of rows to import

# In[ ]:


train_dtypes = pd.read_csv('../input/train.csv',nrows=10000 , dtype=dtypes)

# In[ ]:


train_dtypes.head()

# b) Simple row skip

# In[ ]:


train = pd.read_csv('../input/train.csv', skiprows=range(1, 3000000), nrows=10000, dtype=dtypes)

# In[ ]:


train.head()

# In[ ]:


del train; del train_dtypes;
gc.collect()

# ### 4. Importing just selected columns  
# If you want to analyze just some specific feature, you can import just the selected columns. 

# In[ ]:


columns = ['Id', 'groupId', 'matchId','killPlace','killPoints','kills','killStreaks','longestKill','winPlacePerc']

dtypes = {
        'Id'                : 'uint32',
        'groupId'           : 'uint32',
        'matchId'           : 'uint16',   
        'killPlace'         : 'uint8',    
        'killPoints'        : 'uint8',    
        'kills'             : 'uint8',    
        'killStreaks'       : 'uint8',    
        'longestKill'       : 'float16',    
        'winPlacePerc'      : 'float16' 
}
example = pd.read_csv('../input/train.csv', usecols=columns, dtype=dtypes)

# In[ ]:


example.head()

# ### 5. Using debug mode
# Many people try to make feature engineering and predict pipelines. However, if the size of the data is large, it takes too long to create a variable or training a model. In this case, we can save time and effort by drawing a sample in advance as metioned above.

# In[ ]:


debug = True
if debug:
    df_train = pd.read_csv('../input/train.csv',nrows=10000 , dtype=dtypes)
    df_test  = pd.read_csv('../input/test.csv', dtype=dtypes)
else:
    df_train = pd.read_csv('../input/train.csv', dtype=dtypes)
    df_test  = pd.read_csv('../input/test.csv', dtype=dtypes)

# ### 6. Lightgbm: prevent RAM spike (explode) at the init training
# 
# I've been looking for a discussion on a similar contest called TalkingdataADtracking. 
# This competition also had a lot of memory errors due to the large data size. 
# The relevant document is as below but I have not confirmed it yet.
# 
# - [LightGBM Faster Training](https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection/discussion/56158)
# - [Lightgbm: prevent RAM spike (explode) at the init training
# ](https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection/discussion/53773)
# - [Reducing Lightgbm RAM spike using 1 difference](https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection/discussion/55325)
# 
# - I have not tested this part yet, so I will check the results and rewrite it.

# ### Results
# - Using these methods, you can reduce ram spikes and time effort, memory. 
# - If you have a good tip or method that you know, please ask me for a comment.
