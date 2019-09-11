#!/usr/bin/env python
# coding: utf-8

# Thanks to this discussion for the observation: https://www.kaggle.com/c/santander-customer-transaction-prediction/discussion/84450
# 
# In this notebook, I transform this column and re-order the train dataset using this column, and see what 
# happens.

# In[ ]:


import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

# In[ ]:


train_df = pd.read_csv('../input/train.csv')

# # Before you read

# This exploration is an attempt to discover some hidden things behind the annonymization.
# Nothing is certain of course and this is so far specualative. 
# Use this knowledge accordingly. 

# # Preliminary work

# Two things to observe: 
#     
# - data has been annonymized
# - it comes from a business setting
# 
# Thus, it is most likely (but not 100% sure) that some of the features
# contain date-like information (and also categorical features but that's 
# for another day). 
# 
# How to find potential columns? Let's try to sort the columns using the number of unique 
# values. What's the heurestic behind this choice? 
# Well there shouldn't be a lot of dates, maybe few thousand top.

# In[ ]:


train_df.drop(['ID_code', 'target'], axis=1).nunique().sort_values()

# ==> `var_68` has the least number of uniques, thus it **might** be a date-like column
# (it could also be a categorical column).
# There is also a possibility that this small number of uniques is a coincidence due to the rounding to 4 decimal numbers (bonus question: could you compute the probability of this event?)

# In[ ]:


f"Min: {train_df['var_68'].min()} and max: {train_df['var_68'].max()}"

# So how to extract a date?
# Well, first, get ride of the decimal values.
# Then transform to a datetime object supposing that it is an ordinal datetime.
# Try different offsets until you get a meaningful date range.
# That's it. Let's see this in action.

# In[ ]:


epoch_datetime = pd.datetime(1900, 1, 1)
trf_var_68_s = (train_df['var_68']*10000 - 7000 + epoch_datetime.toordinal()).astype(int)
date_s = trf_var_68_s.map(datetime.fromordinal)
train_df['date'] = date_s
sorted_train_df = train_df.drop('var_68', axis=1).sort_values('date')

# # Some plots 

# In[ ]:


fig, ax = plt.subplots(1, 1, figsize=(20, 8))
sorted_train_df.set_index('date')['var_0'].plot(ax=ax)

# In[ ]:


fig, ax = plt.subplots(1, 1, figsize=(20, 8))
sorted_train_df.set_index('date')['var_1'].plot(ax=ax)

# In[ ]:


fig, ax = plt.subplots(1, 1, figsize=(20, 8))
sorted_train_df.set_index('date')['var_2'].plot(ax=ax)

# In[ ]:


fig, ax = plt.subplots(1, 1, figsize=(20, 8))
sorted_train_df.set_index('date')['target'].plot(ax=ax)

# # Date column exploration

# Alright, let's now explore this newly created column.

# In[ ]:


date_s.nunique()

# => I will thus use the `date` column to group rows. 

# In[ ]:


f"Train starts: {date_s.min()}, ends: {date_s.max()}"

# In[ ]:


sorted_train_df['date'].dt.month.value_counts()

# In[ ]:


sorted_train_df['date'].dt.month.value_counts().plot(kind='bar')

# In[ ]:


sorted_train_df['date'].dt.year.value_counts()

# In[ ]:


sorted_train_df['date'].dt.year.value_counts().plot(kind='bar')

# In[ ]:


sorted_train_df['date'].dt.dayofweek.value_counts()

# In[ ]:


sorted_train_df['date'].dt.dayofweek.value_counts().plot(kind='bar')

# ==> Uniform day of week distribution. That's a good sign!

# In[ ]:


fig, ax = plt.subplots(1, 1, figsize=(20, 8))
sorted_train_df.groupby('date')['target'].agg(['std', 'mean', 'max', 'min']).plot(ax=ax)

# In[ ]:


# In another cell signs the count is much bigger than the other statistics
fig, ax = plt.subplots(1, 1, figsize=(20, 8))
sorted_train_df.groupby('date')['target'].agg(['count']).plot(ax=ax)

# # What about the test?

# Let's see if our observation transfers well to the test dataset.

# In[ ]:


test_df = pd.read_csv('../input/test.csv')
epoch_datetime = pd.datetime(1900, 1, 1)
s = (test_df['var_68']*10000 - 7000 + epoch_datetime.toordinal()).astype(int)
test_df['date'] = s.map(datetime.fromordinal)
sorted_test_df = test_df.drop('var_68', axis=1).sort_values('date')

# In[ ]:


f"Test starts: {test_df['date'].min()} and ends: {test_df['date'].max()}"

# In[ ]:


test_df['date'].dt.year.value_counts().plot(kind='bar')

# In[ ]:


test_df['date'].dt.month.value_counts().plot(kind='bar')

# In[ ]:


test_df['date'].dt.dayofweek.value_counts().plot(kind='bar')

# In[ ]:


fig, ax = plt.subplots(1, 1, figsize=(20, 8))
sorted_test_df.groupby('date')['var_1'].agg(['count']).plot(ax=ax)

# # Test and train date column comparaison

# In[ ]:


len(set(sorted_train_df['date']))

# In[ ]:


len(set(sorted_test_df['date']))

# In[ ]:


len(set(sorted_train_df['date']) & set(sorted_test_df['date']))

# In[ ]:


len(set(sorted_train_df['date']) - set(sorted_test_df['date']))

# In[ ]:


len(set(sorted_test_df['date']) - set(sorted_train_df['date']))

# In[ ]:


set(sorted_test_df['date']) - set(sorted_train_df['date'])

# ==> Most of the dates overlap. 

# In[ ]:


fig, ax = plt.subplots(1, 1, figsize=(12, 8))
sorted_train_df.groupby('date')['var_91'].count().plot(ax=ax, label="train")
sorted_test_df.groupby('date')['var_91'].count().plot(ax=ax, label="test")
ax.legend()

# In[ ]:


# Zoom on 2018
fig, ax = plt.subplots(1, 1, figsize=(12, 8))
(sorted_train_df.loc[lambda df: df.date.dt.year == 2018]
               .groupby('date')['var_91']
               .count()
               .plot(ax=ax, label="train"))
(sorted_test_df.loc[lambda df: df.date.dt.year == 2018]
               .groupby('date')['var_91']
               .count()
               .plot(ax=ax, label="test"))
ax.legend()

# In[ ]:


# Zoom on 2018-1
fig, ax = plt.subplots(1, 1, figsize=(12, 8))
(sorted_train_df.loc[lambda df: (df.date.dt.year == 2018) & (df.date.dt.month == 1)]
               .groupby('date')['var_91']
               .count()
               .plot(ax=ax, label="train"))
(sorted_test_df.loc[lambda df: (df.date.dt.year == 2018) & (df.date.dt.month == 1)]
               .groupby('date')['var_91']
               .count()
               .plot(ax=ax, label="test"))
ax.legend()

# In[ ]:


# Zoom on 2018-1
fig, ax = plt.subplots(1, 1, figsize=(12, 8))
(sorted_train_df.loc[lambda df: (df.date.dt.year == 2018) & (df.date.dt.month == 1)]
               .groupby('date')['var_91']
               .mean()
               .plot(ax=ax, label="train"))
(sorted_test_df.loc[lambda df: (df.date.dt.year == 2018) & (df.date.dt.month == 1)]
               .groupby('date')['var_91']
               .mean()
               .plot(ax=ax, label="test"))
ax.legend()

# Idea to try: predict the mean of the target (using the date 
# for grouping) for the overlapping dates. 

# In[ ]:


overlapping_dates = set(sorted_train_df['date']) & set(sorted_test_df['date'])

# In[ ]:


grouped_df = (sorted_train_df.loc[lambda df: df.date.isin(overlapping_dates)]
                             .groupby('date')['target']
                             .mean())

# In[ ]:


grouped_df.plot(kind='hist', bins=100)

# In[ ]:


grouped_df.to_csv('grouped_df.csv', index=False)

# # What to do now?

# Some of the things I will try to do: 
# - Use this transformed column for a better temporal CV. Some ideas I have tried: stratification using years, day of weeks, and so on.
# - Transform other columns using this new one
# 
# Stay tuned for more insights. :)
