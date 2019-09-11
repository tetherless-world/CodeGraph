#!/usr/bin/env python
# coding: utf-8

# In this notebook I'll try to explore the basic information about the dataset to help us build our models / features.
# 
# 
# # Data description
# > Each asset is identified by an `assetCode` *(note that a single company may have multiple assetCodes)*. Depending on what you wish to do, you may use the `assetCode`, `assetName`, or `time` as a way to join the market data to news data.
# >
# > The marketdata contains a variety of returns calculated over different timespans. All of the returns in this set of marketdata have these properties:
# * Returns are always calculated either open-to-open (from the opening time of one trading day to the open of another) or close-to-close (from the closing time of one trading day to the open of another).
# * Returns are either raw, meaning that the data is not adjusted against any benchmark, or market-residualized (Mktres), meaning that the movement of the market as a whole has been accounted for, leaving only movements inherent to the instrument.
# * Returns can be calculated over any arbitrary interval. Provided here are 1 day and 10 day horizons.
# * Returns are tagged with 'Prev' if they are backwards looking in time, or 'Next' if forwards looking.

# Let's start with importing the necessary libraries.

# In[ ]:


import numpy as np
import pandas as pd

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

from kaggle.competitions import twosigmanews

pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 999

# Get 2Sigma environment
env = twosigmanews.make_env()

# In[ ]:


# Get the data
mt_df, nt_df = env.get_training_data()

# # Market data

# In[ ]:


mt_df.head()

# In[ ]:


print("We have {:,} market samples in the training dataset.".format(mt_df.shape[0]))

# **dtype**

# In[ ]:


mt_df.dtypes

# According to the data description:
# > The data is stored and retrieved as Pandas dataframes in the Kernels environment. Columns types are optimized to minimize space in memory.
# 
# I think the `assetCode` could be `category` dtype too.
# > *universe(float64)* - a boolean indicating whether or not the instrument on that day will be included in scoring. This value is not provided outside of the training data time period. The trading universe on a given date is the set of instruments that are avilable for trading (the scoring function will not consider instruments that are not in the trading universe). The trading universe changes daily.
# 
# Why we need a `float64` for a boolean? Do I missing something?

# **NaN**

# In[ ]:


mt_df.isna().sum()

# **Number of unique values**

# In[ ]:


mt_df.nunique()

# # Example Asset - Apple Inc

# In[ ]:


asset1Code = 'AAPL.O'
asset1_df = mt_df[(mt_df['assetCode'] == asset1Code) & (mt_df['time'] > '2015-01-01') & (mt_df['time'] < '2017-01-01')]

# In[ ]:


# Create a trace
trace1 = go.Scatter(
    x = asset1_df['time'].dt.strftime(date_format='%Y-%m-%d').values,
    y = asset1_df['close'].values
)

layout = dict(title = "Closing prices of {}".format(asset1Code),
              xaxis = dict(title = 'Month'),
              yaxis = dict(title = 'Price (USD)'),
              )
data = [trace1]

py.iplot(dict(data=data, layout=layout), filename='basic-line')

# In[ ]:


asset1_df['high'] = asset1_df['open']
asset1_df['low'] = asset1_df['close']

for ind, row in asset1_df.iterrows():
    if row['close'] > row['open']:
        asset1_df.loc[ind, 'high'] = row['close']
        asset1_df.loc[ind, 'low'] = row['open']

trace1 = go.Candlestick(
    x = asset1_df['time'].dt.strftime(date_format='%Y-%m-%d').values,
    open = asset1_df['open'].values,
    low = asset1_df['low'].values,
    high = asset1_df['high'].values,
    close = asset1_df['close'].values
)

layout = dict(title = "Candlestick chart for {}".format(asset1Code),
              xaxis = dict(
                  title = 'Month',
                  rangeslider = dict(visible = False)
              ),
              yaxis = dict(title = 'Price (USD)')
             )
data = [trace1]

py.iplot(dict(data=data, layout=layout), filename='basic-line')

# *Please note: we don't have `high` and `low` data in the dataset*

# # `time` column

# In[ ]:


mt_df['time'].dt.date.describe()

# In[ ]:


print("There are {} missing values in the `time` column".format(mt_df['time'].isna().sum()))

# According to the competition's data description:
# > all rows are taken at 22:00 UTC
# 
# Let's see...

# In[ ]:


mt_df['time'].dt.time.describe()

# In[ ]:


assetsByTradingDay = mt_df.groupby(mt_df['time'].dt.date)['assetCode'].nunique()

# In[ ]:


# Create a trace
trace1 = go.Bar(
    x = assetsByTradingDay.index, # asset1_df['time'].dt.strftime(date_format='%Y-%m-%d').values,
    y = assetsByTradingDay.values
)

layout = dict(title = "# of assets by trading days",
              xaxis = dict(title = 'Year'),
              yaxis = dict(title = '# of assets'),
              )
data = [trace1]

py.iplot(dict(data=data, layout=layout), filename='basic-line')

# # `assetCode` column
# > a unique id of an asset

# In[ ]:


print("There are {:,} unique assets in the training set".format(mt_df['assetCode'].nunique()))

# In[ ]:


print("There are {} missing values in the `assetCode` column".format(mt_df['time'].isna().sum()))

# In[ ]:


volumeByAssets = mt_df.groupby(mt_df['assetCode'])['volume'].sum()
highestVolumes = volumeByAssets.sort_values(ascending=False)[0:10]

# In[ ]:


# Create a trace
trace1 = go.Pie(
    labels = highestVolumes.index,
    values = highestVolumes.values
)

layout = dict(title = "Highest trading volumes")
data = [trace1]

py.iplot(dict(data=data, layout=layout), filename='basic-line')

# # `assetName` column
# > the name that corresponds to a group of `assetCodes`. These may be "Unknown" if the corresponding `assetCode` does not have any rows in the news data.

# In[ ]:


mt_df['assetName'].describe()

# In[ ]:


print("There are {:,} records with assetName = `Unknown` in the training set".format(mt_df[mt_df['assetName'] == 'Unknown'].size))

# In[ ]:


assetNameGB = mt_df[mt_df['assetName'] == 'Unknown'].groupby('assetCode')
unknownAssets = assetNameGB.size().reset_index('assetCode')

# In[ ]:


print("There are {} unique assets without assetName in the training set".format(unknownAssets.shape[0]))

# In[ ]:


unknownAssets

# # `universe` column
# > a boolean indicating whether or not the instrument on that day will be included in scoring. This value is not provided outside of the training data time period. The trading universe on a given date is the set of instruments that are avilable for trading (the scoring function will not consider instruments that are not in the trading universe). The trading universe changes daily.

# In[ ]:


mt_df['universe'].nunique()

# In[ ]:


print("There are {:,} missing values in the `universe` column".format(mt_df['universe'].isna().sum()))

# # `volume` column
# > trading volume in shares for the day

# In[ ]:


print("There are {:,} missing values in the `volume` column".format(mt_df['volume'].isna().sum()))

# In[ ]:


mt_df['volume'].describe()

# There are 0 trading volumes, let's examine those.

# In[ ]:


zeroVolume = mt_df[mt_df['volume'] == 0]

# In[ ]:


print("There are {:,} sample in the training set with zero trading volumes".format(len(zeroVolume)))

# How many of them are included in the scoring calculation?

# In[ ]:


print("The scoring function will consider {:,} out of {:,} 'zero trading' training samples".format(len(zeroVolume[zeroVolume['universe'] == 1]), len(zeroVolume)))

# In[ ]:


volumesByTradingDay = mt_df.groupby(mt_df['time'].dt.date)['volume'].sum()

# In[ ]:


# Create a trace
trace1 = go.Bar(
    x = volumesByTradingDay.index,
    y = volumesByTradingDay.values
)

layout = dict(title = "Trading volumes by date",
              xaxis = dict(title = 'Year'),
              yaxis = dict(title = 'Volume'),
              )
data = [trace1]

py.iplot(dict(data=data, layout=layout), filename='basic-line')

# I just realized that the trading volumes are shares not currency values, so the chart above shows the trading volumes per day in shares. Trading volume in USD would be more interesting. I'll fix it later.

# # `open` column
# > the open price for the day (not adjusted for splits or dividends)

# In[ ]:


print("There are {:,} missing values in the `open` column".format(mt_df['open'].isna().sum()))

# In[ ]:


mt_df['open'].describe()

# # `close` column
# > the close price for the day (not adjusted for splits or dividends)

# In[ ]:


print("There are {:,} missing values in the `close` column".format(mt_df['close'].isna().sum()))

# In[ ]:


mt_df['close'].describe()

# # `returns` columns
# > The marketdata contains a variety of returns calculated over different timespans. All of the returns in this set of marketdata have these properties:
# >
# > * Returns are always calculated either open-to-open (from the opening time of one trading day to the open of another) or close-to-close (from the closing time of one trading day to the open of another).
# * Returns are either raw, meaning that the data is not adjusted against any benchmark, or market-residualized (Mktres), meaning that the movement of the market as a whole has been accounted for, leaving only movements inherent to the instrument.
# * Returns can be calculated over any arbitrary interval. Provided here are 1 day and 10 day horizons.
# * Returns are tagged with 'Prev' if they are backwards looking in time, or 'Next' if forwards looking.

# **returnsOpenNextMktres10**
# Market-residualized open-to-open returns in the next 10 days.
# > This is the target variable used in competition scoring. The market data **has been filtered** such that `returnsOpenNextMktres10` is **always not null**.

# In[ ]:


print("There are {} missing `returnsOpenNextMktres10` values in the training set.".format(mt_df['returnsOpenNextMktres10'].isna().sum()))

# In[ ]:


# No growth, no decrease
print(len(mt_df[mt_df['returnsOpenNextMktres10'] == 0]))

# In[ ]:


mt_df['returnsOpenNextMktres10'].describe()

# Ok, so we have more than 4M samples. The mean is 0.014 and the std is 7.24 but we have -1375 and 9761 minimum and maximum values. We should examine those outliers.

# In[ ]:


outliers = mt_df[(mt_df['returnsOpenNextMktres10'] > 1) |  (mt_df['returnsOpenNextMktres10'] < -1)]
outliers['returnsOpenNextMktres10'].describe()

# In[ ]:


# returnsOpenNextMktres10 data without outliers
woOutliers = mt_df[(mt_df['returnsOpenNextMktres10'] < 1) &  (mt_df['returnsOpenNextMktres10'] > -1)]
woOutliers['returnsOpenNextMktres10'].describe()

# In[ ]:


# Create a trace
trace1 = go.Histogram(
    x = woOutliers.sample(n=10000)['returnsOpenNextMktres10'].values
)

layout = dict(title = "returnsOpenNextMktres10 (random 10.000 sample; without outliers)")
data = [trace1]

py.iplot(dict(data=data, layout=layout), filename='basic-line')

# ------------------
# **More to come. Stay tuned!**
# 
# Please upvote if you like the notebook :)
