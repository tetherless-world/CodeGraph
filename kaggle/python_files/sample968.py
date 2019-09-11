#!/usr/bin/env python
# coding: utf-8

# # Digging deeper into anonymization
# 
# It is clear that some features, and the target itself is somehow obfuscated. Well, guess what - I really like working with these kind of puzzles - lookup BNP Paribas competition in Kaggle :). And it was quite obvious for me that same tricks could be applied here as well (at least at some level). I am going to share what I found in this competition so far.

# In[ ]:


import numpy as np
import pandas as pd

train = pd.read_csv('../input/train.csv')
merchants = pd.read_csv('../input/merchants.csv')
new_merchant_transactions = pd.read_csv('../input/new_merchant_transactions.csv')

# ## ~~ purchase_amount ~~

# For simplicity, I am going to ignore historical_transactions, but all the findings below transfers to this table as well.
# 
# The most interesting feature in transactions table is `purchase_amount`. And it should be natural to expect some integers in it's distribution, and the values to be positive - which is obviously not the case here!

# In[ ]:


new_merchant_transactions.head()

# In[ ]:


new_merchant_transactions['purchase_amount_integer'] = new_merchant_transactions.purchase_amount.apply(lambda x: x == np.round(x))
print(new_merchant_transactions.groupby('purchase_amount_integer')['card_id'].count())

# So imagine yourself in the shoes of the person who had the task to anonymize this column. One of the most straightforward ways to do that is apply some kind of demeaning and scaling. And with that assumption in mind it is pretty straightforward to reverse engineer that! So after some optimization and manual checking you can find the mean and scale parameters, so that the values would make much more sense. 
# 
# This is the magic formula that I was able to find:

# In[ ]:


new_merchant_transactions['purchase_amount_new'] = np.round(new_merchant_transactions['purchase_amount'] / 0.00150265118 + 497.06,8)

# In[ ]:


new_merchant_transactions.purchase_amount_new.head(100)

# Pretty awesome! it seems we can safely round up to 2 decimal points and get a nice looking `purchase_amount` values!

# In[ ]:


new_merchant_transactions['purchase_amount_new'] = np.round(new_merchant_transactions['purchase_amount'] / 0.00150265118 + 497.06,2)

# Now check how many integers we got now...

# In[ ]:


new_merchant_transactions['purchase_amount_integer'] = new_merchant_transactions.purchase_amount_new.apply(lambda x: x == np.round(x))
print(new_merchant_transactions.groupby('purchase_amount_integer')['card_id'].count())

# That is roughly 40% - which is too high to be it a coincidence - do you agree?:)
# 
# We can actually investigate it a little bit further to find more meanining to this column:

# In[ ]:


new_merchant_transactions.groupby('purchase_amount_new')['card_id'].count().reset_index(name='count').sort_values('count',ascending=False).head(100)

# So we both got some nice looking integer values and x.99 ones which you would expect a price to have. So it seems the `purhcase_amount` could actually mean how much money was spent on each transaction! Now this reveals some interesting opportunities to make more meaningful features!

# ## ~~ numerical_1 & numerical_2 ~~

# In[ ]:


merchants.head(10)

# Now as we know that `purchase_amount` was anonymized in lazy fashion, you could expect something similar in these features as well - and in fact it turns out to be true:

# In[ ]:


merchants['numerical_1'] = np.round(merchants['numerical_1'] / 0.009914905 + 5.79639, 0)
merchants['numerical_2'] = np.round(merchants['numerical_2'] / 0.009914905 + 5.79639, 0)

# In[ ]:


merchants.groupby('numerical_1')['merchant_id'].count().head(10)

# In[ ]:


merchants.groupby('numerical_2')['merchant_id'].count().head(10)

# If you were to investigate further, all values are integers! However, I still strugle to find the meaning of these columns - one of my hypothesis was that it could represent how many "loyalty" products were purchased of this merchant in last 3/6/12 months? This could be stong feature on aggregated level, I guess...
# 
# Other features (avg sales and such) are still a mystery for me, but maybe they are not of that importance. Morever, as you could notice, so far I have exploited high numeric precision of float numbers - and sales features are rounded up - so it is practically impossible to reverse engineer something here.

# ## ~~ target ~~

# Now this is probably the most interesting part! There were some hints in the forums about log10(2) being used in the outlier value of -33.xxxxx; 
# 
# But one thing people missed out was that the same log10(2) is used for all the target values as well!

# In[ ]:


train['target_new'] = 10**(train['target']*np.log10(2))

# In[ ]:


train.head(10)

# In[ ]:


train['target_new'].describe()

# You might say, so what? there are no integers or something like that... 
# 
# However, there are no negative values! AND, if you looked closely, it is actually ratios what we are dealing with!
# 
# If you tried putting some of these values into https://www.wolframalpha.com/ , it does a good job trying to explain how one could get this kind of float value.
# 
# There is a couple I was able to find reasonably fast:

# In[ ]:


print(train['target_new'][2], 29/18)
print(train['target_new'][29823], 973/300)

# There are a lot "clean" looking divisions, and it is enough to conclude that the customer loyalty is in fact measured by some kind of ratio. I believe, this ratio represents some kind of aggregate transaction `purchase_amount` ratios in the future. But I have never personally worked with loyalty prediction; I think there are more people who would understand right away what these numbers mean.

# ## tl;dr 

# Anonymous values are demeaned, unscaled or log transformed. However, it is pretty easy to reverse that and try to guess the meaning of anonymous features.
# 
# The target column itself is a ratio of some kind - probably sum of transactions compared to historical... 
# 
# Need more input on this one from you guys!

# ## Thank you for reading, happy hunting!

# 

# 

# 

# 

# 
