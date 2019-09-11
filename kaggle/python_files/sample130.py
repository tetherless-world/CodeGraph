#!/usr/bin/env python
# coding: utf-8

# # This is going to be epic... sit back, relax and enjoy!

# Following my previous kernel (https://www.kaggle.com/raddar/towards-de-anonymizing-the-data-some-insights) I was able to reveal that `target` is transformed using log function, and the raw target can be reversed with `2**target` transformation. Moreover, I speculated that the true target is a ratio of `product_sum`. 
# 
# This kernel is all about explaining, what kind of ratio we are working with, and why the problem is so hard!
# 
# And this is going to blow your mind!

# In[ ]:


import numpy as np
import pandas as pd
pd.set_option('display.float_format', '{:.10f}'.format)

train = pd.read_csv('../input/train.csv')
historical_transactions = pd.read_csv('../input/historical_transactions.csv')
new_merchant_transactions = pd.read_csv('../input/new_merchant_transactions.csv')

# Let's apply transformations as in previous kernel:

# In[ ]:


new_merchant_transactions['purchase_amount_new'] = np.round(new_merchant_transactions['purchase_amount'] / 0.00150265118 + 497.06,2)
historical_transactions['purchase_amount_new'] = np.round(historical_transactions['purchase_amount'] / 0.00150265118 + 497.06,2)
train['target_raw'] = 2**train['target']

# Now in order to prove that target is a meaningful ratio, I was looking for some easy cases, such as `card_id` with only one `merchant_id`, with no transactions in`new_merchant_transaction_table`. This helps to isolate the problem and analyze it more thoroughly. 
# 
# With some manual exploration, I found that `merchant_id=M_ID_fc7d7969c3` is a perfect candidate to work with.

# In[ ]:


merchant_counts = historical_transactions.groupby(['card_id'])['merchant_id'].nunique().reset_index(name = 'merchant_n')
one_merchant = merchant_counts[merchant_counts['merchant_n']==1].reset_index(drop=True)
dat = historical_transactions.loc[historical_transactions['card_id'].isin(one_merchant['card_id'])]
dat = dat.loc[~dat['card_id'].isin(new_merchant_transactions['card_id'])]
dat = dat.loc[dat.merchant_id=='M_ID_fc7d7969c3'].reset_index(drop=True)

# Let's take a look at a random `card_id` to inspect the transactional history:

# In[ ]:


historical_transactions[historical_transactions.card_id=='C_ID_b3c7ff9e19'].sort_values('purchase_date')

# In[ ]:


new_merchant_transactions[new_merchant_transactions.card_id=='C_ID_b3c7ff9e19'].sort_values('purchase_date')

# The payments are happeninig roughly on the same day of month (maybe recurring credit card payments). We also observe a `purchase_amount` increase from 29.9 to 37.9.

# So why the `merchant_id=M_ID_fc7d7969c3` is so special? It turns out it is subscribtion based merchant - most likely internet payments???The `purchase_amount` is distributed in a few very distinct categories:

# In[ ]:


historical_transactions.loc[historical_transactions.merchant_id=='M_ID_fc7d7969c3'].groupby('purchase_amount_new')['card_id'].count()

# This is a very useful information - remember - the hypothesis was that target is based on some kind of ratios, and the `purchase_amount` ratio seems like a very reasonable candidate.

# At this point let's summarise what we know:
# 
# - merchant is selling subscribtion based products
# - card_id is automatically charged monthly
# - the price for the subscribtion can change (most likely due to upselling)
# - there are 5 products with prices of 19.90, 22.90, 27.90, 29.90 and 37.90; the 1.00 is probably subscribtion activation fee

# Now let's try to make some correlations with train `target`:

# In[ ]:


dat = dat.merge(train, on = 'card_id')

# In[ ]:


dat.head()

# In[ ]:


dat.groupby('target_raw')['card_id'].count().reset_index(name='n')

# It seems most of the target values for this specific `card_id` group is concentrated in `target_raw=1`. This is great news, because we were able to somehow cluster this specific group of cards just by some ad hoc rules (single merchant, specific merchant, no new merchant transactions).
# 
# Let's take a look at `card_id``s, which `target_raw` is equal to 1:

# In[ ]:


dat.loc[dat['target_raw']==1,['card_id','purchase_date','purchase_amount_new','target_raw']].sort_values(['card_id','purchase_date'])

# So there is some fluctuation in `purchase_amount`, therefore it is still unclear why the `target_raw==1`.

# However, the most interesting part lies not in `target_raw=1`, but actually in other float numbers!
# 
# Let's take a step back first and revisit the `purchase_amount` values we extracted earlier, and let's calculate possible ratios based on these values:

# In[ ]:


prices = [19.90, 22.90, 27.90, 29.90, 37.90]
sorted({ i/j for j in prices for i in prices})

# These numbers are not telling anything at the moment.
# 
# But there is the catch - let's revisit the previous table, were we calculated unique `target_raw` values (compare the list above with the table below).
# 
# Hint: if you looked closely you may find that some of the list values overlap!

# In[ ]:


dat.groupby('target_raw')['card_id'].count().reset_index(name='n')

# You will find the values like 0.713261, 1.21834, 1.35842 appearing in both objects!

# This is pretty amazing find and at this point I am 100% sure that the target is a ratio of change in `purchase_amount`.
# 
# However, there is still a question to ask - how exactly this ratio is calculated?
# 
# Let's take this `card_id` for example:

# In[ ]:


dat[dat.card_id=='C_ID_2c8d99614f'].sort_values('purchase_date')

# If we excluded the subscribtion activation fees (1.00), you would expect the `target_raw` to be 1 (as `purchase_amount` is constant...)
# 
# However the `target_raw` is 1.2183406082, which in fact is equivalent to `27.90/22.90`.
# 
# Amazing! Now we definetly know that the `purchase_amount` for this `card_id` has changed, and it is very likely happened in the future (`month_lag = 1` or `month_lag = 2`). This information we do not observe in `new_merchant_transactions` table, because organizers have excluded that on purpose in data preparation stage - if we would have this information, we could easily track down how target has been calculated, and 0.000 RMSE score would be possible.
# 
# What does this all mean for `target` calculation?
# 
# The answer is simple:
# ## target is equal to the ratio of money spent in the future divided by money spent in the past!
# 
# I guess this transfers to all merchants and to all cards in the dataset, but the calculations are a bit harder, as more merchants are involved.

# # Bonus
# 
# Now as we know that we are working with future/history ratios, the outlier value (-33.xxxxx) is actually the meaning of a user not spending a single cent in the future on the historical merchants - mystery solved!
# 
# If i was an organizer, I would have made it to be ~ -10 or so, as RMSE is so sensitive regarding this outler'ish value...
# 
# Sadly, this also means that these outliers cannot be predicted with the given historical information...
# 
# 
# 
# Knowing all this I can say for sure that the data we have is in fact **real data** and not **simulated one**! In case I am wrong on this one - well done Elo - you did a marvelous job in simulation part (would be hard to make a clean generator like that!)

# # Things that are still not clear...

# Now as we definetly know that target is a ratio of future and past transaction amounts, there are still some unaswered questions, mainly:
# 
# - which months are taken into calculation? `month_lag=1` / `month_lag=0`, `month_lag=2` / `month_lag=0`, `month_lag=1` / `min(month_lag=0, month_lag=-1)`, etc. All these are all valid options...
# - does `new_merchant_transactions` have any influence for the `target`?

# # What's next?
# 
# So what you can do with this kind of information. Firstly, the rate of `purchase_amount` monthly change for (`card_id, merchant_id`) tuple should be the key features in your models, i.e. average change ratio of (5->4, 4->3, 3->2, 2->1, 1->0); Or features such as `sum(future_purchase_amount)/sum(purchase_amount_lag_-1)`, etc.
# 
# This also allows building new intermediate models, like predicting the historic merchant performance in `month_lag=1` and `month_lag=2` and stacking its predictions to your main model.
# 
# These are the most impactful things I can think of right now, but there could always be more!

# # Thank you!
# 
# if you liked the content don't forget to upvote!

# 

# 

# 

# 

# 
