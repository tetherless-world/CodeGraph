#!/usr/bin/env python
# coding: utf-8

# # How to load all  55,423,855 rows of data into one single DataFrame about 2 minutes, and reload it the same next time under 5 seconds (your mileage may vary)

# Updates:
# - 2 August 2018 - Using a neat trick from this [kernel ](https://www.kaggle.com/btyuhas/bayesian-optimization-with-xgboost) by @btyuhas, the CSV loading time has improved from about 5 minutes to 2 minutes.
# - 9 August 2018 - @danlester suggested a much faster way (10x) to count the number of lines of a CSV file using the `wc` (word count) unix command.
# - 12 August 2018 - @jpmiller suggested using dask to read the csv faster. Added **Using Dask** at the bottom section.

# In[ ]:


import pandas as pd 
import dask.dataframe as dd
import os
from tqdm import tqdm

TRAIN_PATH = '../input/train.csv'

# In[ ]:


# Assume we only know that the csv file is somehow large, but not the exact size
# we want to know the exact number of rows

# Method 1, using file.readlines. Takes about 20 seconds.
with open(TRAIN_PATH) as file:
    n_rows = len(file.readlines())

print (f'Exact number of rows: {n_rows}')

# In[ ]:



# Method 2 by @danlester, using wc unix command. Takes only 3 seconds!
s = !wc -l {TRAIN_PATH}

# add one because the file isn't quite correctly formatted as a CSV, should have a final newline char
n_rows = int(s[0].split(' ')[0])+1

print (f'Exact number of rows: {n_rows}')

# In[ ]:



# Same method but more 'pythonic'
import subprocess

def file_len(fname):
    p = subprocess.Popen(['wc', '-l', fname], stdout=subprocess.PIPE, 
                                              stderr=subprocess.PIPE)
    result, err = p.communicate()
    if p.returncode != 0:
        raise IOError(err)
    return int(result.strip().split()[0])+1

n_rows = file_len(TRAIN_PATH)
print (f'Exact number of rows: {n_rows}')

# In[ ]:


# Peep at the training file header
df_tmp = pd.read_csv(TRAIN_PATH, nrows=5)
df_tmp.head()

# In[ ]:


df_tmp.info()

# We might not need float64 (16 decimal places) for the longitude and latitude values. float32 (7 decimal places) might be just enough.
# 
# See https://gis.stackexchange.com/questions/8650/measuring-accuracy-of-latitude-and-longitude/8674#8674

# In[ ]:


# Set columns to most suitable type to optimize for memory usage
traintypes = {'fare_amount': 'float32',
              'pickup_datetime': 'str', 
              'pickup_longitude': 'float32',
              'pickup_latitude': 'float32',
              'dropoff_longitude': 'float32',
              'dropoff_latitude': 'float32',
              'passenger_count': 'uint8'}

cols = list(traintypes.keys())

# In[ ]:


chunksize = 5_000_000 # 5 million rows at one go. Or try 10 million

# In[ ]:


df_list = [] # list to hold the batch dataframe

for df_chunk in tqdm(pd.read_csv(TRAIN_PATH, usecols=cols, dtype=traintypes, chunksize=chunksize)):
     
    # Neat trick from https://www.kaggle.com/btyuhas/bayesian-optimization-with-xgboost
    # Using parse_dates would be much slower!
    df_chunk['pickup_datetime'] = df_chunk['pickup_datetime'].str.slice(0, 16)
    df_chunk['pickup_datetime'] = pd.to_datetime(df_chunk['pickup_datetime'], utc=True, format='%Y-%m-%d %H:%M')
    
    # Can process each chunk of dataframe here
    # clean_data(), feature_engineer(),fit()
    
    # Alternatively, append the chunk to list and merge all
    df_list.append(df_chunk) 

# In[ ]:


# Merge all dataframes into one dataframe
train_df = pd.concat(df_list)

# Delete the dataframe list to release memory
del df_list

# See what we have loaded
train_df.info()

# In[ ]:


display(train_df.head())
display(train_df.tail())

# In[ ]:


# Save into feather format, about 1.5Gb. 
train_df.to_feather('nyc_taxi_data_raw.feather')

# In[ ]:


# load the same dataframe next time directly, without reading the csv file again!
train_df_new = pd.read_feather('nyc_taxi_data_raw.feather')

# In[ ]:


# print the dataframe info to verify we have indeed loaded the saved dataframe of 55 million rows
train_df_new.info()

# Notice that it only takes **less than 5 seconds** to load from the feather file the next time you want to import the dataframe.
# 
# Of course, it doesn't mean that we can all use the entire 55 million rows directly (unless your machine has tons of RAM) as the memory usage will increase as we do our processing on the dataframe. 
# 
# However, it would be useful as we can do some EDA on the full training data instead of subset. 

# # Using Dask
# 
# As suggested by @jpmiller

# With Dask and its dataframe construct, you set up the dataframe must like you would in pandas but rather than loading the data into pandas, this approach keeps the dataframe as a sort of ‘pointer’ to the data file and doesn’t load anything until you specifically tell it to do so.
# 
# Source: http://pythondata.com/dask-large-csv-python/

# In[ ]:



# dask's read_csv takes no time at all!
ddf = dd.read_csv(TRAIN_PATH, usecols=cols, dtype=traintypes)

# In[ ]:


# no info?
ddf.info()

# In[ ]:


# nothing to describe?
ddf.describe()

# In[ ]:



# dask is lazy. It only works when it is asked explicitly with compute()
ddf.describe().compute()

# In[ ]:



# Again, it only works when it is asked :)
len(ddf)

# In[ ]:


del ddf

# ## Pandas read_csv vs Dask read_csv

# In[ ]:



# using panda read_csv to read the entire file in one shot
df = pd.read_csv(TRAIN_PATH, usecols=cols, dtype=traintypes)

# In[ ]:


df['pickup_datetime'] = df['pickup_datetime'].str.slice(0, 16)
df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'], utc=True, format='%Y-%m-%d %H:%M')

# In[ ]:


df.info()

# In[ ]:


del df

# In[ ]:



# using dask read_csv followed by compute() to create a panda dataframe
ddf_pd = dd.read_csv(TRAIN_PATH, usecols=cols, dtype=traintypes).compute()

# RangeIndex is an optimized version of Int64Index that can represent a monotonic ordered set
# Source: https://pandas-docs.github.io/pandas-docs-travis/advanced.html#int64index-and-rangeindex
# Furthermore, without conversion, the resulting dataframe takes up more memory usage (1.9GB)
ddf_pd.index = pd.RangeIndex(start=0, stop=len(ddf_pd)) 

# In[ ]:


ddf_pd['pickup_datetime'] = ddf_pd['pickup_datetime'].str.slice(0, 16)
ddf_pd['pickup_datetime'] = pd.to_datetime(ddf_pd['pickup_datetime'], utc=True, format='%Y-%m-%d %H:%M')

# In[ ]:


ddf_pd.info()

# In[ ]:


del ddf_pd

# Based on my limited understanding, to perform task that involves iterating through the entire dataset serially,  (such as `describe()` and `len()` above), dask does not seem to perform better. The RAM usage is also very high. I feel like it's easier to just load all the data in memory using panda's DataFrame to perform computation rather than depending on dask's lazy loading. That is of course when your memory capacity allows you to do so. 
# 
# On the other hand, I do realize the advantage that it can be very useful for selective operations that can be done in parallel such as filtering when you cannot fit all the data in memory. Imagine if we have 55 billions (instead of 55 million) rows of data now and we need to find all the taxi rides with exactly 4 passengers in 2010 March with fare amount less than $20.
# 
# The other advantage of dask is multi processing.  dask's `read_csv` (around 50+s) clearly performs better than pandas' `read_csv` (around 1min 20s to 1min30+s) and I believe it is due to this feature.
