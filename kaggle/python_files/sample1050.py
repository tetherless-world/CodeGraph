#!/usr/bin/env python
# coding: utf-8

# # Different time points for card observations

# I saw most of the kernels using *magic* 2018-02 month as a reference point for calculating date features. However, when I looked at the data it was obvious that this doesn't hold for each `card_id`. 
# 
# Let's try to extract the correct reference point (let's call it `observation_date`) for each `card_id`.

# In[ ]:


import numpy as np
import pandas as pd
pd.set_option('display.float_format', '{:.10f}'.format)

# read the data
train = pd.read_csv('../input/train.csv')
historical_transactions = pd.read_csv('../input/historical_transactions.csv')
new_merchant_transactions = pd.read_csv('../input/new_merchant_transactions.csv')

# In[ ]:


# fast way to get last historic transaction / first new transaction
last_hist_transaction = historical_transactions.groupby('card_id').agg({'month_lag' : 'max', 'purchase_date' : 'max'}).reset_index()
last_hist_transaction.columns = ['card_id', 'hist_month_lag', 'hist_purchase_date']
first_new_transaction = new_merchant_transactions.groupby('card_id').agg({'month_lag' : 'min', 'purchase_date' : 'min'}).reset_index()
first_new_transaction.columns = ['card_id', 'new_month_lag', 'new_purchase_date']

# In[ ]:


# converting to datetime
last_hist_transaction['hist_purchase_date'] = pd.to_datetime(last_hist_transaction['hist_purchase_date']) 
first_new_transaction['new_purchase_date'] = pd.to_datetime(first_new_transaction['new_purchase_date']) 

# In[ ]:


# substracting month_lag for each row
last_hist_transaction['observation_date'] = \
    last_hist_transaction.apply(lambda x: x['hist_purchase_date']  - pd.DateOffset(months=x['hist_month_lag']), axis=1)

first_new_transaction['observation_date'] = \
    first_new_transaction.apply(lambda x: x['new_purchase_date']  - pd.DateOffset(months=x['new_month_lag']-1), axis=1)

# At this point we just reversed month lag function to get a rought estimate of the `observation_date` to be used for specific `card_id`. As you can see below, the `observation_date` is already different for many cards!

# In[ ]:


last_hist_transaction.head(20)

# In[ ]:


first_new_transaction.head(20)

# First thing you may notice is that historical transactions tend to be at the end of month, and for future transactions tend to be close to the start of the month. At this point it is safe to assume, that observation date is in format 2017-10-01, 2017-11-01, 2018-02-01, etc.
# 
# Let's do that!

# In[ ]:


last_hist_transaction['observation_date'] = last_hist_transaction['observation_date'].dt.to_period('M').dt.to_timestamp() + pd.DateOffset(months=1)
first_new_transaction['observation_date'] = first_new_transaction['observation_date'].dt.to_period('M').dt.to_timestamp()

# In[ ]:


last_hist_transaction.head(20)

# In[ ]:


first_new_transaction.head(20)

# All we need to do is to validate if `observation_date` matches both `historical_transactions` and `new_merchant_transactions` table information:

# In[ ]:


validate = last_hist_transaction.merge(first_new_transaction, on = 'card_id')
all(validate['observation_date_x'] == validate['observation_date_y'])

# So using two different tables we were able to reach the same observation date for each `card_id`. Now let's take a look how `observation_date` behaves together with `target` information:

# In[ ]:


train = train.merge(last_hist_transaction, on = 'card_id')

# In[ ]:


train.groupby('observation_date').agg({'target': ['mean','count']})

# So... there is definetly different behavior for how cards behave in different observation dates! 
# 
# Now there is some interesting brainstorming you can do... the lowest target ratio is for `observation_date` at 2017-11/12 - right before Christmas season. So it seems people are spending more money on presents from new merchants, therefore the "loyal" merchants suffered from that during Christmas shopping season...

# `observation_date` could be a good feature candidate... or maybe not, as kernels are already exploiting dates too much... :)

# ## Thanks for reading, hope you find this useful!

# 
