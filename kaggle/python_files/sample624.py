#!/usr/bin/env python
# coding: utf-8

# I will update this kernel regularly! 
# 
# 
# Currently this kernel contains:
# 1. Defining the metadata of silly column names
#     - useful to select specific variables for analysis, visualization, modelling etc.
# 2. Distributions of target 0,1 depending on columns
#     - Discovers weird column : 'wheezy-copper-turtle-magic'
# 3. Look over statistics of the other columns sharing partial name of 'wheezy-copper-turtle-magic'

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# visualization
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

# In[ ]:


train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
print("train set : ",train.shape)
print("test set : ",test.shape)

# In[ ]:


train.head()

# In[ ]:


test.head()

# # Make metadata of the column names 
# - I got this idea to make a metadata from this kernel : https://www.kaggle.com/bertcarremans/data-preparation-exploration

# In[ ]:


data = []
for col in test.columns:
    # parse column names
    if col not in ['id', 'target']:
        col_split_list = col.split("-")
        
        # Initialize keep to True
#         keep = True
    
        # Create dictionary that contains all the metadata for each columns
        col_dict = {
            'feature_name' : col,
            'col_name_1' : col_split_list[0],
            'col_name_2' : col_split_list[1],
            'col_name_3' : col_split_list[2],
            'col_name_4' : col_split_list[3]
#             'keep' : keep
        }
        data.append(col_dict)
    
# meta = pd.DataFrame(data, columns = ['feature_name', 'col_name_1', 'col_name_2', 'col_name_3', 'col_name_4', 'keep'])
    
meta = pd.DataFrame(data, columns = ['feature_name', 'col_name_1', 'col_name_2', 'col_name_3', 'col_name_4'])
meta.set_index('feature_name', inplace=True)

# In[ ]:


meta

# # unique value counts of each column name parts

# In[ ]:


print('length of unique values of each part of column names:', 
      '\n', 'col_name_1 :',len(meta.col_name_1.unique()), 
      '\n', 'col_name_2 :',len(meta.col_name_2.unique()), 
      '\n', 'col_name_3 :',len(meta.col_name_3.unique()), 
      '\n', 'col_name_4 :',len(meta.col_name_4.unique()))

# In[ ]:


meta.col_name_1.value_counts().head(10)

# In[ ]:


meta.col_name_2.value_counts().head(10)

# In[ ]:


meta.col_name_3.value_counts().head(10)

# In[ ]:


meta.col_name_4.value_counts().head(15)

# # unique list of each column name parts

# In[ ]:


meta['col_name_1'].unique()

# In[ ]:


meta['col_name_2'].unique()

# In[ ]:


meta['col_name_3'].unique()

# In[ ]:


meta['col_name_4'].unique()

# # Example to extract each type of columns

# In[ ]:


meta[(meta.col_name_4 == 'important')]

# In[ ]:


pd.DataFrame({'count': meta.groupby(['col_name_4', 'col_name_1'])['col_name_4'].size()}).reset_index()

# # Target distribution of group of column names

# In[ ]:


meta[(meta.col_name_4 == 'important')]

# In[ ]:


train_4_important = train[['ugly-tangerine-chihuahua-important','muggy-turquoise-donkey-important'] ]

# In[ ]:


temp = meta[(meta.col_name_4 == 'important')].index.tolist() + ['target']

# In[ ]:


train_4_important = train[temp]

# In[ ]:


train_4_important.shape

# In[ ]:


train_4_important.head()

# # Visualization

# ## Target distribution

# In[ ]:


sns.countplot(train['target'], palette='Set3')

# - 0,1 distribution plots of train data for each column
# - code from : https://www.kaggle.com/senkin13/eda-starter, figsize modified

# In[ ]:


feats = [f for f in train.columns if f not in ['id','target']]
def plot_feature_distribution(df1, df2, label1, label2, features, row, col):
    i = 0
    sns.set_style('whitegrid')
    plt.figure()
    ratio = int(row/col/2)
    fig, ax = plt.subplots(row,col,figsize=(15,15*(ratio+1)))

    for feature in features:
        i += 1
        plt.subplot(row,col,i)
        sns.distplot(df1[feature], hist=False,label=label1)
        sns.distplot(df2[feature], hist=False,label=label2)
        plt.xlabel(feature, fontsize=9)
        locs, labels = plt.xticks()
        plt.tick_params(axis='x', which='major', labelsize=6, pad=-6)
        plt.tick_params(axis='y', which='major', labelsize=6)
    plt.show();

# In[ ]:


t0 = train[feats].loc[train['target'] == 0]
t1 = train[feats].loc[train['target'] == 1]
features = train[feats].columns.values
plot_feature_distribution(t0, t1, '0', '1', features, 64,4)   

# - notice that 'wheezy-copper-turtle-magic' shows different pattern from other features

# - statistics of the columns starting with 'wheezy'

# In[ ]:


meta[(meta.col_name_1 == 'wheezy')]

# In[ ]:


train[meta[(meta.col_name_1 == 'wheezy')].index].describe()

# In[ ]:


test[meta[(meta.col_name_1 == 'wheezy')].index].describe()

# In[ ]:


t0 = train[feats].loc[train['target'] == 0]
t1 = train[feats].loc[train['target'] == 1]
features = meta[(meta.col_name_1 == 'wheezy')].index
plot_feature_distribution(t0, t1, '0', '1', features,3,2) 

# - statistics with columns which col_name_2==copper

# In[ ]:


meta[(meta.col_name_2 == 'copper')]

# In[ ]:


train[meta[(meta.col_name_2 == 'copper')].index].describe()

# In[ ]:


test[meta[(meta.col_name_2 == 'copper')].index].describe()

# In[ ]:


t0 = train[feats].loc[train['target'] == 0]
t1 = train[feats].loc[train['target'] == 1]
features = meta[(meta.col_name_2 == 'copper')].index
plot_feature_distribution(t0, t1, '0', '1', features,3,2) 

# - Only one column has 3rd part ==turtle or 4th part ==magic 

# In[ ]:


meta[(meta.col_name_3 == 'turtle')]

# In[ ]:


meta[(meta.col_name_4 == 'magic')]

# In[ ]:


meta[(meta.col_name_4 == 'important')]

# In[ ]:


train[meta[(meta.col_name_4 == 'important')].index].describe()

# In[ ]:


test[meta[(meta.col_name_4 == 'important')].index].describe()

# In[ ]:


t0 = train[feats].loc[train['target'] == 0]
t1 = train[feats].loc[train['target'] == 1]
features = meta[(meta.col_name_4 == 'important')].index
plot_feature_distribution(t0, t1, '0', '1', features,7,3) 

# In[ ]:


meta[(meta.col_name_4 == 'hint')]

# In[ ]:


train[meta[(meta.col_name_4 == 'hint')].index].describe()

# In[ ]:


test[meta[(meta.col_name_4 == 'hint')].index].describe()

# In[ ]:


t0 = train[feats].loc[train['target'] == 0]
t1 = train[feats].loc[train['target'] == 1]
features = meta[(meta.col_name_4 == 'hint')].index
plot_feature_distribution(t0, t1, '0', '1', features,4,3) 

# ## heatmap

# In[ ]:


plt.figure(figsize = [16,9])
sns.heatmap(train.corr())

# In[ ]:


sns.heatmap(train[meta[(meta.col_name_1 == 'wheezy')].index].corr())

# ## Scatter Plots by features

# In[ ]:


color = sns.color_palette()
plt.figure(figsize=(6,6))
plt.scatter(train[feats[0]], train[feats[1]], c=train.target, alpha = 0.6)

# In[ ]:


color = sns.color_palette()
plt.figure(figsize=(6,6))
plt.scatter(train['wheezy-copper-turtle-magic'], train['wheezy-harlequin-earwig-gaussian'], c=train.target, alpha = 0.6)

# In[ ]:


plt.figure(figsize=(6,6))
plt.scatter(train['wheezy-harlequin-earwig-gaussian'], train['wheezy-red-iguana-entropy'], c=train.target)

# In[ ]:


plt.figure(figsize=(6,6))
plt.scatter(train['wheezy-copper-turtle-magic'], train['wheezy-red-iguana-entropy'], c=train.target, alpha = 0.6)

# - It seems like the magic feature is categorical

# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



