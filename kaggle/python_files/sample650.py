#!/usr/bin/env python
# coding: utf-8

# ## Giba's Property
# 
# - https://www.kaggle.com/titericz/the-property-by-giba (kernel)
# - https://www.kaggle.com/c/santander-value-prediction-challenge/discussion/61329 (post)
# 
# #### This kernel is just to extend giba's result in a *stupid* brute-force way
# ### The updated part is based on [S D's comment](https://www.kaggle.com/johnfarrell/giba-s-property-extended-result/notebook#358945)

# In[ ]:


import os
import gc
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
DATA_DIR = '../input/'

hex2dec = lambda x: int(x, 16)

# In[ ]:


train = pd.read_csv(DATA_DIR+'train.csv')

# In[ ]:


cols = [
    "f190486d6","58e2e02e6","eeb9cd3aa","9fd594eec","6eef030c1","15ace8c9f",
    "fb0f5dbfe","58e056e12","20aa07010","024c577b9","d6bb78916",
    "b43a7cfd5","58232a6fb"
]
rows = np.array([2072,3493,379,2972,2367,4415,2791,3980,194,1190,3517,811,4444])-1

# ### The original result

# In[ ]:


tmp = train.loc[rows, ["ID","target"]+cols]
print('original shape', tmp.shape)
tmp

# ### Generate a subset candidates with same rows to search extendable columns

# In[ ]:


df_cand_col = train.loc[rows, :]
df_cand_col = df_cand_col.iloc[:, 2:]
df_cand_col

# In[ ]:


df_new = train.loc[rows, cols]

# ### Search in a brute-force way

# In[ ]:


def bf_search(df_new, df_cand):
    cnt = 0
    head_curr = df_new.values[1:, 0]
    tail_curr = df_new.values[:-1, -1]
    while True:
        for c in df_cand.columns:
            if c in df_new:
                continue
            elif np.all(
                df_cand[c].iloc[:-1].values==head_curr
            ) and len(df_cand[c].unique())>1:
                df_new.insert(0, c, df_cand[c].values)
                head_curr = df_new.values[1:, 0]
                print(c, 'found head!', 'new shape', df_new.shape)
                cnt += 1
                break
            elif np.all(
                df_cand[c].iloc[1:].values==tail_curr
            ) and len(df_cand[c].unique())>1:
                df_new[c] = df_cand[c].values
                tail_curr = df_new.values[:-1, -1]
                print(c, 'found tail!', 'new shape', df_new.shape)
                cnt += 1
                break
            else:
                continue
        if cnt==0:
            break
        else:
            cnt = 0
            continue
    return df_new

# In[ ]:


print('Column searching ...')
df_new = bf_search(df_new, df_cand_col)

# In[ ]:


df_new

# ### Transpose the new result 
# ### and use the same method to search rows

# In[ ]:


df_new = df_new.T.copy()
df_new

# In[ ]:


df_cand_row = train[df_new.index].T.copy()
df_cand_row.head()

# In[ ]:


print('Row searching ...')
df_new = bf_search(df_new, df_cand_row)

# In[ ]:


df_new = df_new.T.copy()
df_new

# In[ ]:


df_cand_col = train.loc[df_new.index, :]
df_cand_col = df_cand_col.iloc[:, 2:]
df_cand_col

# In[ ]:


print('Column searching (second time) ...')
df_new = bf_search(df_new, df_cand_col)

# In[ ]:


print('new shape', df_new.shape)
train.loc[df_new.index, ["ID","target"]+df_new.columns.tolist()]

# In[ ]:


print(f'Row indexes({df_new.shape[0]})\n', df_new.index.values.tolist())
print(f'Column indexes({df_new.shape[1]})\n', df_new.columns.values.tolist())

# ## There's a strange long tail of number ***1563411.76*** pointed out by [S D](https://www.kaggle.com/johnfarrell/giba-s-property-extended-result/comments#358945)
# ## Thanks [S D](https://www.kaggle.com/sdoria) for this new idea!
# ## Let's check it

# In[ ]:


for i, c in enumerate(df_new.columns):
    print(
        'No.', i, 'Column Name', c, 
        'subset count',
        (df_new[c].values==1563411.76).sum(), 
        'train count',
        (train[c].values==1563411.76).sum()
    )

# In[ ]:


res_cnt = dict((c, (train[c].values==1563411.76).sum()) for c in train.columns[2:])
res_cnt = pd.DataFrame.from_dict(res_cnt, orient='index', columns=['strange_number_cnt'])
res_cnt = res_cnt.sort_values('strange_number_cnt', 0, False)
res_cnt.head(50).T

# In[ ]:


res_cnt.head(10)

# ## Sad... there are *2* columns with the same count 30!
# - 91f701ba2
# - fc99f9426
# 
# ## What about the row-wise?

# In[ ]:


for i, c in enumerate(df_new.T.columns):
    print(
        'No.', i, 'Row Name', c, 
        'subset count',
        (df_new.T[c].values==1563411.76).sum(), 
        'train count',
        (train.T[c].values==1563411.76).sum()
    )

# In[ ]:


tmp = train.iloc[:, 2:].values
res_t_cnt = dict((idx, (tmp[i, :]==1563411.76).sum()) for i,idx in enumerate(train.index))
res_t_cnt = pd.DataFrame.from_dict(res_t_cnt, orient='index', columns=['strange_number_cnt'])
res_t_cnt = res_t_cnt.sort_values('strange_number_cnt', 0, False)
res_t_cnt.head(50).T

# ## Hmm, seems no same values. OK, we start from rows!

# In[ ]:


head_row_indexes = res_t_cnt[res_t_cnt['strange_number_cnt']>24].index.tolist()
head_row_indexes

# In[ ]:


mask = res_t_cnt['strange_number_cnt']>0 
mask&=res_t_cnt['strange_number_cnt']<8
tail_row_indexes = res_t_cnt.loc[mask].index.tolist()
tail_row_indexes

# In[ ]:


pd.concat([
    train.loc[head_row_indexes, ['target']+df_new.columns.tolist()], 
    train.loc[df_new.index, ['target']+df_new.columns.tolist()],
    train.loc[tail_row_indexes, ['target']+df_new.columns.tolist()], 
])

# ## Good! This is our new df_new and we got to search once more!

# In[ ]:


df_new = pd.concat([
    train.loc[head_row_indexes, df_new.columns.tolist()], 
    train.loc[df_new.index, df_new.columns.tolist()],
    train.loc[tail_row_indexes, df_new.columns.tolist()], 
])

# In[ ]:


def row_bf_search(df_new):
    df_new = df_new.T.copy()
    df_cand_row = train[df_new.index].T.copy()
    print('Row searching ...')
    df_new = bf_search(df_new, df_cand_row)
    df_new = df_new.T.copy()
    return df_new
def column_bf_search(df_new):
    df_cand_col = train.loc[df_new.index, :]
    df_cand_col = df_cand_col.iloc[:, 2:]
    print('Column searching ...')
    df_new = bf_search(df_new, df_cand_col)
    return df_new

# In[ ]:


df_new = column_bf_search(df_new)

# In[ ]:


df_new = row_bf_search(df_new)

# ## No good... But wait! Let's check it!
# ### Here, we still can't identify the true order of the columns with the same count 30
# ### Just for example take a look at 'fc99f9426','91f701ba2' at first
# ### We add 'fc99f9426','91f701ba2' and the first 6 columns with count 31~36

# In[ ]:


res_cnt[:10]

# In[ ]:


train.loc[df_new.index, ['target']+df_new.columns.tolist()+['fc99f9426','91f701ba2'] + res_cnt.index.values[:6].tolist()[::-1]]

# ## Hmm, seems something is wrong
# ### at column: ['1db387535', 'fc99f9426','91f701ba2']
# ### from ~last 10 rows

# In[ ]:


train.loc[df_new.index[-10:], df_new.columns.tolist()[-2:]+['fc99f9426', '91f701ba2'] + res_cnt.index.values[:6].tolist()[::-1]]

# ### Check the table above
# ### Seems 540000.00 disappeared from column '1db387535'
# ###      And 1015000.00	 disappeared from column 'fc99f9426'
# ### Thus the time series is becoming not *serious* from here...
# ### Another question, what about the order of 'fc99f9426','91f701ba2'? Which's first?

# In[ ]:


train.loc[df_new.index[:8], ['f190486d6', '58e2e02e6']].T

# In[ ]:


train.loc[df_new.index[-2:], df_new.columns.tolist()[-2:]+['fc99f9426', '91f701ba2']+res_cnt.index.values[:6].tolist()[::-1]]

# ### From row values (in reverse order), it seems 'fc99f9426', '91f701ba2' is consistent with other columns 
# ### Thus, in our limited analysis, the new df_new is

# In[ ]:


df_new_new = train.loc[df_new.index, df_new.columns.tolist()+['fc99f9426','91f701ba2'] + res_cnt.index.values[:6].tolist()[::-1]]
df_new_new.shape

# In[ ]:


print(f'Row indexes({df_new_new.shape[0]})\n', df_new_new.index.values.tolist())
print(f'Column indexes({df_new_new.shape[1]})\n', df_new_new.columns.values.tolist())

# In[ ]:


train.loc[df_new_new.index, ['ID', 'target']+df_new_new.columns.tolist()]

# ### For the final result consist many uncertains and hypothesis
# ### Be careful of using this !!!
# ### Any comments and advices are welcome !!!
# ### Please point out if there's something wrong!!!

# In[ ]:




# In[ ]:



