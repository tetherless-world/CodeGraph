#!/usr/bin/env python
# coding: utf-8

# # 1. Quick start: read csv and flatten json fields
# 
# Hi! This notebook just loads the data and flattens the json fields. 
# I have put the code in one function so you can copy it easily.
# 
# Also, I dumped the processed dataframes to disk so they are easily importable from within another kernel.

# In[ ]:


import os
import json
import numpy as np
import pandas as pd
from pandas.io.json import json_normalize

def load_df(csv_path='../input/train.csv', nrows=None):
    JSON_COLUMNS = ['device', 'geoNetwork', 'totals', 'trafficSource']
    
    df = pd.read_csv(csv_path, 
                     converters={column: json.loads for column in JSON_COLUMNS}, 
                     dtype={'fullVisitorId': 'str'}, # Important!!
                     nrows=nrows)
    
    for column in JSON_COLUMNS:
        column_as_df = json_normalize(df[column])
        column_as_df.columns = [f"{column}.{subcolumn}" for subcolumn in column_as_df.columns]
        df = df.drop(column, axis=1).merge(column_as_df, right_index=True, left_index=True)
    print(f"Loaded {os.path.basename(csv_path)}. Shape: {df.shape}")
    return df

print(os.listdir("../input"))

# In[ ]:


df_small = load_df(nrows=20000)

# In[ ]:


df_train = load_df()
df_test = load_df("../input/test.csv")

# In[ ]:


df_train.to_csv("train-flattened.csv", index=False)
df_test.to_csv("test-flattened.csv", index=False)
