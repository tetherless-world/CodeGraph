#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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

# # my first eda 
# Please tell me if I make mistake  
# ## Contents
# * [metadata_train/test.csv](#metadata_[train/test].csv)  
#     * [Overview](#Overview)
#     * [check null](#checknull)
#     * [check target](#checktarget)
#     * [check metadata](#checkmetadata)
# * [train/test.parquet](#[train/test].parquet)
#     * [train.parquet](#train.parquet)
#         * [Overview](#p-Overview)
#         * [check waves](#checkwaves)
#     * [test.parquet](#test.parquet)
# 

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
import pyarrow.parquet as pq
import gc

# In[ ]:


train_meta_df = pd.read_csv("../input/metadata_train.csv")
test_meta_df = pd.read_csv("../input/metadata_test.csv")

# <a name="metadata_[train/test].csv"></a>
# # metadata_[train/test].csv
# * id_measurement: the ID code for a trio of signals recorded at the same time.
# * signal_id: the foreign key for the signal data.
# * phase: the phase ID code within the signal trio. The phases may or may not all be impacted by a fault on the line.
# * target: 0 if the power line is undamaged, 1 if there is a fault.  (only train)
# <a name="Overview"></a>
# ## Overview

# In[ ]:


print("metadata_train shape is {}".format(train_meta_df.shape))
print("metadata_test shape is {}".format(test_meta_df.shape))

# In[ ]:


train_meta_df.head(6)

# In[ ]:


test_meta_df.head()

# <a name="checknull"></a>
# ## check null

# In[ ]:


train_meta_df.isnull().sum()

# In[ ]:


test_meta_df.isnull().sum()

# they have **no** nulls

# <a name="checktarget"></a>
# ## check target
# target: 0 if the power line is undamaged, 1 if there is a fault.

# In[ ]:


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
sns.countplot(x="target", data=train_meta_df, ax=ax1)
sns.countplot(x="target", data=train_meta_df, hue="phase", ax=ax2)

# In[ ]:


target_count = train_meta_df.target.value_counts()
print("negative(target=0) target: {}".format(target_count[0]))
print("positive(target=1) target: {}".format(target_count[1]))
print("positive data {:.3}".format((target_count[1]/(target_count[0]+target_count[1]))*100))

# <font color="red">  fault data is too small </font>  
# Target is almost uniformly distributed in all phases  

# In[ ]:


miss = train_meta_df.groupby(["id_measurement"]).sum().query("target != 3 & target != 0")
print("not all postive or negative num: {}".format(miss.shape[0]))
miss

# **Data with the same id is not always all positive or negative.**

# <a name="checkmetadata"></a>
# ## check metadata
# ### id_measurement

# In[ ]:


print("id_measurement have {} uniques in train".format(train_meta_df.id_measurement.nunique()))
print("id_measurement have {} uniques in test".format(test_meta_df.id_measurement.nunique()))

# In[ ]:


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
g = sns.catplot(x="id_measurement", data=train_meta_df, ax=ax1, kind="count")
label = list(range(train_meta_df.id_measurement.min(), train_meta_df.id_measurement.max(), 1000))
ax1.set_xticks(label, [str(i) for i in label])
ax1.patch.set_facecolor('green')
ax1.patch.set_alpha(0.2)
plt.close(g.fig)
g = sns.catplot(x="id_measurement", data=test_meta_df, ax=ax2, kind="count")
label = list(range(test_meta_df.id_measurement.min(), test_meta_df.id_measurement.max(), 1000))
ax2.set_xticks(label, [str(i) for i in label])
ax2.patch.set_facecolor('yellow')
ax2.patch.set_alpha(0.2)
plt.close(g.fig)

# In[ ]:


train_meta_df.id_measurement.value_counts().describe()

# In[ ]:


test_meta_df.id_measurement.value_counts().describe()

# id_measurement:  3 data per one unique id  
# because  electric transmission lines have three-phase alternating current(maybe)  
# ### phase

# In[ ]:


print("phase have {} uniques in train".format(train_meta_df.phase.unique()))
print("phase have {} uniques in test".format(test_meta_df.phase.unique()))
print("they are phase numbering")

# # Let's look parquet data
# how to read parquet files  
# ref: https://www.kaggle.com/sohier/reading-the-data-with-python

# In[ ]:


gc.collect()
subset_train_df = pq.read_pandas('../input/train.parquet').to_pandas()

# <a name="[train/test].parquet"></a>
# # [train/test].parquet
# The signal data. Each **<font color="red">column</font>** contains one signal; 800,000 int8 measurements as exported with pyarrow.parquet version 0.11.  
# <font color="red">Please note that this is different than our usual data orientation of one row per observation; </font>  
# the switch makes it possible loading a subset of the signals efficiently.   
# If you haven't worked with Apache Parquet before, please refer to either the Python data loading starter kernel.  
# 
# <a name="train.parquet"></a>
# # train.parquet
# <a name="p-Overview"></a>
# ## Overview

# In[ ]:


nan = 0
for col in range(len(subset_train_df.columns)):
    nan += np.count_nonzero(subset_train_df.loc[col, :].isnull())
print("train.parquet have {} nulls".format(nan))
print("train.parquet shape is {}".format(subset_train_df.shape))

# In[ ]:


subset_train_df.head()

# please care Each **<font color="red">column</font>**  contains one signal !  
# so I do transpose this data  

# In[ ]:


subset_train_df = subset_train_df.T

# In[ ]:


print("train shape is {}".format(subset_train_df.shape))

# <a name="checkwaves"></a>
# ## check waves

# In[ ]:


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 5), sharey=True)
for i in range(3):
    sns.lineplot(x=subset_train_df.columns, y=subset_train_df.iloc[i, :], ax=ax1, label=["phase:"+str(train_meta_df.iloc[i, :].phase)])
ax1.set_xlabel("example of undamaged signal", fontsize=18)
ax1.set_ylabel("amp", fontsize=18)
ax1.patch.set_facecolor('blue')
ax1.patch.set_alpha(0.2)
for i in range(3, 6):
    sns.lineplot(x=subset_train_df.columns, y=subset_train_df.iloc[i, :], ax=ax2, label=["phase:"+str(train_meta_df.iloc[i, :].phase)])
ax2.set_xlabel("example of damaged signal", fontsize=18)
ax2.set_ylabel("amp", fontsize=18)
ax2.patch.set_facecolor('red')
ax2.patch.set_alpha(0.2)

# we can see three-phase and some noise  
# I don't see big difference  
# Is this difference is big noise?  
# so let's look more data  

# In[ ]:


neg_index = train_meta_df.query("target == 0 & phase == 0").head(9).index.values
pos_index = train_meta_df.query("target == 1 & phase == 0").head(9).index.values

# In[ ]:


fig, axes = plt.subplots(3, 3, figsize=(20, 12), sharex=True, sharey=True)
fig.suptitle("Undamaged examples", size=18)
for x, index in enumerate(neg_index):
    for phase in range(3):
        sns.lineplot(x=subset_train_df.columns, y=subset_train_df.iloc[index+phase, :], ax=axes[x//3, x%3])
    axes[x//3, x%3].patch.set_facecolor('blue')
    axes[x//3, x%3].patch.set_alpha(0.2)

# In[ ]:


fig, axes = plt.subplots(3, 3, figsize=(20, 12), sharex=True, sharey=True)
fig.suptitle("Damaged examples", size=18)
for x, index in enumerate(pos_index):
    for phase in range(3):
        sns.lineplot(x=subset_train_df.columns, y=subset_train_df.iloc[index+phase, :], ax=axes[x//3, x%3])
        axes[x//3, x%3].patch.set_facecolor('red')
        axes[x//3, x%3].patch.set_alpha(0.2)

# I cannot tell them apart.  
# Is this noise or dameged?  
# Did I maked a mistake in plot.....?  

# In[ ]:


del subset_train_df, fig, axes
gc.collect()

# <a name="test.parquet"></a>
# ## test.parquet
# test.parquet is too big (20337, 800000)  
# so I will read test data in 6 parts  
# ### 1/6 (0～3389 columns)

# In[ ]:


INPUT_NUM = 3390
TRAIN_NUM = 8712
shapes = []
nulls = 0

# In[ ]:


subset_test_df = pq.read_pandas('../input/test.parquet', columns=[str(i) for i in range(TRAIN_NUM + INPUT_NUM)]).to_pandas()
nan = 0
for col in range(len(subset_test_df.columns)):
    nan += np.count_nonzero(subset_test_df.loc[col, :].isnull())
shapes.append(subset_test_df.shape)
nulls += nan
print("1st of the six test.parquet shape is {}".format(subset_test_df.shape))
print("1st of the six test.parquet have {} nulls".format(nan))

# In[ ]:


del subset_test_df
gc.collect()

# ### 2/6 (3390～6779 columns)

# In[ ]:


subset_test_df = pq.read_pandas('../input/test.parquet', columns=[str(i) for i in range(TRAIN_NUM + INPUT_NUM, TRAIN_NUM + INPUT_NUM*2)]).to_pandas()
nan = 0
for col in range(len(subset_test_df.columns)):
    nan += np.count_nonzero(subset_test_df.loc[col, :].isnull())
shapes.append(subset_test_df.shape)
nulls += nan
print("2nd of the six test.parquet shape is {}".format(subset_test_df.shape))
print("2nd of the six test.parquet have {} nulls".format(nan))


# In[ ]:


del subset_test_df
gc.collect()

# ### 3/6 (6780～10169 columns)

# In[ ]:


subset_test_df = pq.read_pandas('../input/test.parquet', columns=[str(i) for i in range(TRAIN_NUM + INPUT_NUM*2, TRAIN_NUM + INPUT_NUM*3)]).to_pandas()
nan = 0
for col in range(len(subset_test_df.columns)):
    nan += np.count_nonzero(subset_test_df.loc[col, :].isnull())
shapes.append(subset_test_df.shape)
nulls += nan
print("3rd of the six test.parquet shape is {}".format(subset_test_df.shape))
print("3rd of the six test.parquet have {} nulls".format(nan))


# In[ ]:


del subset_test_df
gc.collect()

# ### 4/6 (10169～13559 columns)

# In[ ]:


subset_test_df = pq.read_pandas('../input/test.parquet', columns=[str(i) for i in range(TRAIN_NUM + INPUT_NUM*3, TRAIN_NUM + INPUT_NUM*4)]).to_pandas()
nan = 0
for col in range(len(subset_test_df.columns)):
    nan += np.count_nonzero(subset_test_df.loc[col, :].isnull())
shapes.append(subset_test_df.shape)
nulls += nan
print("4th of the six test.parquet shape is {}".format(subset_test_df.shape))
print("4th of the six test.parquet have {} nulls".format(nan))


# In[ ]:


del subset_test_df
gc.collect()

# ### 5/6 (13560～16949 columns)

# In[ ]:


subset_test_df = pq.read_pandas('../input/test.parquet', columns=[str(i) for i in range(TRAIN_NUM + INPUT_NUM*4, TRAIN_NUM + INPUT_NUM*5)]).to_pandas()
nan = 0
for col in range(len(subset_test_df.columns)):
    nan += np.count_nonzero(subset_test_df.loc[col, :].isnull())
shapes.append(subset_test_df.shape)
nulls += nan
print("5th of the six test.parquet shape is {}".format(subset_test_df.shape))
print("5th of the six test.parquet have {} nulls".format(nan))


# In[ ]:


del subset_test_df
gc.collect()

# ### 6/6 (16950～20336 columns)

# In[ ]:


subset_test_df = pq.read_pandas('../input/test.parquet', columns=[str(i) for i in range(TRAIN_NUM + INPUT_NUM*5, TRAIN_NUM + 20337)]).to_pandas()
nan = 0
for col in range(len(subset_test_df.columns)):
    nan += np.count_nonzero(subset_test_df.loc[col, :].isnull())
shapes.append(subset_test_df.shape)
nulls += nan
print("6th of the six test.parquet shape is {}".format(subset_test_df.shape))
print("6th of the six test.parquet have {} nulls".format(nan))

# In[ ]:


print("train.parquet have {} nulls".format(nulls))
index = 0
for shape in shapes:
    index += shape[1]
print("train.parquet shape is ({}, {})".format(index, shapes[0][0]))

# **Thank you for watching!**  
# ## In Progress
# * Make simple model
# * serch more effective feature

# In[ ]:


print("test")

# In[ ]:



