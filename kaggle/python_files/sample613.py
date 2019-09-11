#!/usr/bin/env python
# coding: utf-8

# Some exploratory data analysis of the two sigma training dataset, focusing on portfolio or market returns

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm


with pd.HDFStore('../input/train.h5') as train:
    df = train.get('train')

# In[ ]:


print('Shape : {}'.format(df.shape))

# In[ ]:


df.head()

# In[ ]:


len(df.id.unique()) # how many assets (instruments) are we tracking?

# In[ ]:


len(df.timestamp.unique()) # how many periods?

# Looks like we have 1,424 assets that we are tracking across 1,813 time periods. We can't make any assumptions about the time period length - it could be days, hours, minutes, etc. as long as the period is uniform.
# 
# The set of assets could be considered as the market portfolio. It would be interesting to see if these assets could be grouped into classes based on the observed data and features. For example, asset classes may be equities, bonds, etc. 
# 
# One approach may be to determine market return for a specific time period, and based on that predict the expected return of each asset based on autocorrelation and on how the asset returns correlate to market returns, given an asset class and other features.
# 
# For now let's try to visualize the market return over the time period.

# In[ ]:


market_df = df[['timestamp', 'y']].groupby('timestamp').agg([np.mean, np.std, len]).reset_index()
market_df.head()

# In[ ]:


t      = market_df['timestamp']
y_mean = np.array(market_df['y']['mean'])
y_std  = np.array(market_df['y']['std'])
n      = np.array(market_df['y']['len'])

plt.figure()
plt.plot(t, y_mean, '.')
plt.xlabel('timestamp')
plt.ylabel('mean of y')

plt.figure()
plt.plot(t, y_std, '.')
plt.xlabel('timestamp')
plt.ylabel('std of y')

plt.figure()
plt.plot(t, n, '.')
plt.xlabel('timestamp')
plt.ylabel('portfolio size')

# Looks like two periods of high variance that are correlated with rapid increases in the number of assets. The number of assets being tracked increases from 750 in the first timestamp to just under 1100 in the last. 
# 
# The total number of assets across all timestamps is 1424, so some assets are being dropped as well. It looks like assets are added to the portfolio periodically (see the gaps in the chart), and sold off more slowly.

# Let's derive a price chart for these returns. We can take the log of the periodic mean returns and get a cumulative sum for each time period to derive a fairly good approximation of a price chart for the portfolio.

# In[ ]:


simple_ret = y_mean # this is a vector of the mean of asset returns for each timestamp
cum_ret = np.log(1+simple_ret).cumsum()

# In[ ]:


portfolio_mean = np.mean(cum_ret)
portfolio_std = np.std(cum_ret)
print("portfolio mean periodic return: " + str(portfolio_mean))
print("portfolio std dev of periodic returns: " + str(portfolio_std))

# In[ ]:


plt.figure()
plt.plot(t, cum_ret)
plt.xlabel('timestamp')
plt.ylabel('portfolio value')

# Taking the log returns and adding them up is a good approximation for the compounding of returns.
# 
# It would be interesting to see how the returns of individual assets correlate to the portfolio returns (determine alpha and beta for each asset with respect to the portfolio). Maybe we can use the portfolio return as a proxy for the market return. 
# 
# Recall that the simple regression model for the return of an individual asset, using the market (or index) return as a feature is:
# 
# return[asset i] = alpha[asset i] + beta * return[market]
# 
# Alpha represents the component of asset returns that cannot be attributed to the market portfolio returns (and may reflect the skill - or luck - of the portfolio manager). Beta represents the asset volatility with respect to the market portfolio. For example, an asset with a beta of 1.5 will rise or fall on average 1.5 times the value of the market portfolio (or index).
# 
# Let's take a look at some individual assets:

# In[ ]:


assets_df = df.groupby('id')['y'].agg(['mean','std',len]).reset_index()
assets_df.head()

# In[ ]:


assets_df = assets_df.sort_values(by='mean')
assets_df.head()

# In[ ]:


assets_df.tail()

# In[ ]:


assets_df.describe()

# Looks like individual asset returns range from a min of -0.035077 to a high of 0.010827, with a mean return of 0.000186 and a std dev of 0.001884.
# 
# Assets have a mean holding period of roughly 1201 periods with a std dev of 646 periods, with a min holding period of 2 and a max of 1813 (all periods).

# In[ ]:


sns.distplot(assets_df['mean'], rug=True, hist=False)

# Checking for correlations between asset return, std and holding period.

# In[ ]:


assets_df.corr()

# In[ ]:


g = sns.PairGrid(assets_df, vars=["mean", "std", "len"])
g = g.map_diag(plt.hist)
g = g.map_offdiag(plt.scatter)

# There seem to be some interesting relationships here. Notably, mean asset returns and holding period are negatively correlated with the std dev of returns.

# Thanks for visiting! Next I'm going to look at time series of individual asset return and correlation with the portfolio returns...

# In[ ]:



