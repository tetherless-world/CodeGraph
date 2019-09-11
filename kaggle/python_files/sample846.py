#!/usr/bin/env python
# coding: utf-8

# # Background

# - I'm not quant trader. But, I know some simple indexes to analyze the charts. 
# - Ta-lib is very good and very helpful library for calculating various indexes, but kernel doesn't support.
# - So, I introduce some indexes and short scripts to obtain them.

# In[ ]:


import pandas as pd
import numpy as np
import time
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn')
sns.set(font_scale=2)

import warnings 
warnings.filterwarnings('ignore')
import os

# In[ ]:


from kaggle.competitions import twosigmanews
# You can only call make_env() once, so don't lose it!
env = twosigmanews.make_env()
print('Done!')

# In[ ]:


(market_train_df, news_train_df) = env.get_training_data()

# In[ ]:


for i, df in market_train_df.groupby('assetCode'):
    break

# # Moving average

# > An example of two moving average curves
# In statistics, a moving average (rolling average or running average) is a calculation to analyze data points by creating series of averages of different subsets of the full data set. It is also called a moving mean (MM)[1] or rolling mean and is a type of finite impulse response filter.
# 
# ref. https://en.wikipedia.org/wiki/Moving_average

# ## Moving average

# - Moving average is so simple

# In[ ]:


df['MA_7MA'] = df['close'].rolling(window=7).mean()
df['MA_15MA'] = df['close'].rolling(window=15).mean()
df['MA_30MA'] = df['close'].rolling(window=30).mean()
df['MA_60MA'] = df['close'].rolling(window=60).mean()

# ## Exponential Moving Average

# > An exponential moving average (EMA), also known as an exponentially weighted moving average (EWMA),[5] is a first-order infinite impulse response filter that applies weighting factors which decrease exponentially.
# 
# ref. https://en.wikipedia.org/wiki/Moving_average#Exponential_moving_average

# In[ ]:


ewma = pd.Series.ewm

# In[ ]:


df['close_30EMA'] = ewma(df["close"], span=30).mean()

# In[ ]:


plt.figure(figsize=(10, 10))
plt.plot(df['close'].values)
plt.plot(df['MA_7MA'].values)
plt.plot(df['MA_60MA'].values)
plt.plot(df['close_30EMA'].values)
plt.legend(['Close', 'MA_7MA', 'MA_60MA', 'close_30EMA'])
plt.show()

# # MACD
# - MACD: (12-day EMA - 26-day EMA)

# > Moving average convergence divergence (MACD) is a trend-following momentum indicator that shows the relationship between two moving averages of prices. The MACD is calculated by subtracting the 26-day exponential moving average (EMA) from the 12-day EMA
# 
# ref. https://www.investopedia.com/terms/m/macd.asp

# In[ ]:


df['close_26EMA'] = ewma(df["close"], span=26).mean()
df['close_12EMA'] = ewma(df["close"], span=12).mean()

df['MACD'] = df['close_12EMA'] - df['close_26EMA']

# ## Bollinger Band

# > Bollinger Bands are a type of statistical chart characterizing the prices and volatility over time of a financial instrument or commodity, using a formulaic method propounded by John Bollinger in the 1980s. Financial traders employ these charts as a methodical tool to inform trading decisions, control automated trading systems, or as a component of technical analysis. Bollinger Bands display a graphical band (the envelope maximum and minimum of moving averages, similar to Keltner or Donchian channels) and volatility (expressed by the width of the envelope) in one two-dimensional chart.
# 
# ref. https://en.wikipedia.org/wiki/Bollinger_Bands

# In[ ]:


no_of_std = 2

df['MA_7MA'] = df['close'].rolling(window=7).mean()
df['MA_7MA_std'] = df['close'].rolling(window=7).std() 
df['MA_7MA_BB_high'] = df['MA_7MA'] + no_of_std * df['MA_7MA_std']
df['MA_7MA_BB_low'] = df['MA_7MA'] - no_of_std * df['MA_7MA_std']

plt.figure(figsize=(10, 10))
plt.plot(df['close'].values)
plt.plot(df['MA_7MA'].values)
plt.plot(df['MA_7MA_BB_high'].values)
plt.plot(df['MA_7MA_BB_low'].values)
plt.legend(['Close', 'MA_7MA', 'MA_7MA_BB_high', 'MA_7MA_BB_low'])
plt.xlim(2200, 2500)
plt.ylim(30, 50)
plt.show()

# In[ ]:


no_of_std = 2

df['MA_15MA'] = df['close'].rolling(window=15).mean()
df['MA_15MA_std'] = df['close'].rolling(window=15).std() 
df['MA_15MA_BB_high'] = df['MA_15MA'] + no_of_std * df['MA_15MA_std']
df['MA_15MA_BB_low'] = df['MA_15MA'] - no_of_std * df['MA_15MA_std']

plt.figure(figsize=(10, 10))
plt.plot(df['close'].values)
plt.plot(df['MA_15MA'].values)
plt.plot(df['MA_15MA_BB_high'].values)
plt.plot(df['MA_15MA_BB_low'].values)
plt.legend(['Close', 'MA_15MA', 'MA_15MA_BB_high', 'MA_15MA_BB_low'])
plt.xlim(2000, 2500)
plt.show()

# In[ ]:


no_of_std = 2

df['MA_30MA'] = df['close'].rolling(window=30).mean()
df['MA_30MA_std'] = df['close'].rolling(window=30).std() 
df['MA_30MA_BB_high'] = df['MA_30MA'] + no_of_std * df['MA_30MA_std']
df['MA_30MA_BB_low'] = df['MA_30MA'] - no_of_std * df['MA_30MA_std']

plt.figure(figsize=(10, 10))
plt.plot(df['close'].values)
plt.plot(df['MA_30MA'].values)
plt.plot(df['MA_30MA_BB_high'].values)
plt.plot(df['MA_30MA_BB_low'].values)
plt.legend(['Close', 'MA_30MA', 'MA_30MA_BB_high', 'MA_30MA_BB_low'])
plt.xlim(2000, 2500)
plt.show()

# # RSI

# > The Relative Strength Index (RSI), developed by J. Welles Wilder, is a momentum oscillator that measures the speed and change of price movements. The RSI oscillates between zero and 100. Traditionally the RSI is considered overbought when above 70 and oversold when below 30. Signals can be generated by looking for divergences and failure swings. RSI can also be used to identify the general trend.
# 
# ref. https://www.fidelity.com/learning-center/trading-investing/technical-analysis/technical-indicator-guide/RSI

# In[ ]:


def rsiFunc(prices, n=14):
    deltas = np.diff(prices)
    seed = deltas[:n+1]
    up = seed[seed>=0].sum()/n
    down = -seed[seed<0].sum()/n
    rs = up/down
    rsi = np.zeros_like(prices)
    rsi[:n] = 100. - 100./(1.+rs)

    for i in range(n, len(prices)):
        delta = deltas[i-1] # cause the diff is 1 shorter

        if delta>0:
            upval = delta
            downval = 0.
        else:
            upval = 0.
            downval = -delta

        up = (up*(n-1) + upval)/n
        down = (down*(n-1) + downval)/n

        rs = up/down
        rsi[i] = 100. - 100./(1.+rs)

    return rsi

# In[ ]:


rsi_6 = rsiFunc(df['close'].values, 6)
rsi_14 = rsiFunc(df['close'].values, 14)
rsi_20 = rsiFunc(df['close'].values, 20)

# In[ ]:


fig, ax = plt.subplots(2, 1, figsize=(10, 10))

ax[0].plot(df['MA_15MA'].values)
ax[0].plot(df['MA_15MA_BB_high'].values)
ax[0].plot(df['MA_15MA_BB_low'].values)
ax[1].plot(rsi_6)
ax[1].plot(rsi_14)
ax[1].plot(rsi_20)

ax[0].set_xlim([1500, 2000])
ax[0].legend(['MA_15MA', 'MA_15MA_BB_high', 'MA_15MA_BB_low'])
ax[1].set_xlim([1500, 2000])
ax[1].legend(['rsi_6', 'rsi_14', 'rsi_20'])
plt.show()

# # Volume moving avreage

# > A Volume Moving Average is the simplest volume-based technical indicator. Similar to a price moving average, a VMA is an average volume of a security (stock), commodity, index or exchange over a selected period of time. Volume Moving Averages are used in charts and in technical analysis to smooth and describe a volume trend by filtering short term spikes and gaps.
# 
# ref. https://www.marketvolume.com/analysis/volume_ma.asp

# In[ ]:


df['VMA_7MA'] = df['volume'].rolling(window=7).mean()
df['VMA_15MA'] = df['volume'].rolling(window=15).mean()
df['VMA_30MA'] = df['volume'].rolling(window=30).mean()
df['VMA_60MA'] = df['volume'].rolling(window=60).mean()

# In[ ]:


plt.figure(figsize=(10, 5))
# plt.plot(df['close'].values)
plt.plot(df['VMA_7MA'].values)
plt.plot(df['VMA_15MA'].values)
plt.plot(df['VMA_30MA'].values)
plt.plot(df['VMA_60MA'].values)
plt.legend(['Close', 'VMA_7MA', 'VMA_15MA', 'VMA_30MA', 'VMA_60MA'])
plt.xlim([1500, 2000])
plt.show()

# # Total

# In[ ]:


fig, ax = plt.subplots(3, 1, figsize=(10, 20))

ax[0].plot(df['MA_15MA'].values)
ax[0].plot(df['MA_15MA_BB_high'].values)
ax[0].plot(df['MA_15MA_BB_low'].values)
ax[1].plot(rsi_6)
ax[1].plot(rsi_14)
ax[1].plot(rsi_20)
ax[2].plot(df['VMA_15MA'].values)
ax[2].plot(df['VMA_30MA'].values)


ax[0].set_xlim([1500, 2000])
ax[0].legend(['MA_15MA', 'MA_15MA_BB_high', 'MA_15MA_BB_low'])
ax[1].set_xlim([1500, 2000])
ax[1].legend(['rsi_6', 'rsi_14', 'rsi_20'])
ax[2].set_xlim([1500, 2000])
ax[2].legend(['VMA_15MA', 'VMA_30MA'])
plt.show()

# # To to
# - There are many indexes. I will update!

# In[ ]:



