#!/usr/bin/env python
# coding: utf-8

# ![](https://choco9966.github.io/Team-EDA/image/TwoSigma_Logo_RGB.jpg)

# # 1. Introduction
# 
# Here is an Exploratory Data Analysis for the Two Sigma: Using News to Predict Stock Movements 
# within the Python environment. In this competition, you must predict a signed confidence value, y^ti∈[−1,1] , which is multiplied by the market-adjusted return of a given `assetCode` over a ten day window. If you expect a stock to have a large positive return--compared to the broad market--over the next ten days, you might assign it a large, positive `confidenceValue` (near 1.0). If you expect a stock to have a negative return, you might assign it a large, negative `confidenceValue` (near -1.0). If unsure, you might assign it a value near zero.
# 
# For each day in the evaluation time period, we calculate:
# 
# $$x_{t} = \sum_{i}\hat y_{ti}r_{ti}u_{ti}$$
# 
# where rti is the 10-day market-adjusted leading return for day t for instrument i, and uti is a 0/1 `universe` variable (see the data description for details) that controls whether a particular asset is included in scoring on a particular day.
# 
# Your submission score is then calculated as the mean divided by the standard deviation of your daily xt values:
# 
# $$score = \frac{\bar x_{t}}{\sigma(x_{t})}.$$
# 
# If the standard deviation of predictions is 0, the score is defined as 0.
# 
# **Note :** According to the discussion below, it seems that R does not work yet.   
# https://www.kaggle.com/c/two-sigma-financial-news/discussion/66831

# # 2. Preparations
# ## 2.1 Load libraries

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

# In the data file description,
# `About this file This is just a sample of the market data. You should not use this data directly.
# Instead, call env.get_training_data() from the twosigmanews package to get the full training sets in your Kernel.`
# 
# **So you download directly below**. I using DJ sterling kernel(https://www.kaggle.com/dster/two-sigma-news-official-getting-started-kernel) thnaks 

# In[ ]:


from kaggle.competitions import twosigmanews
# You can only call make_env() once, so don't lose it!
env = twosigmanews.make_env()
print('Done!')

# ## 2.2 Load train_data

# In[ ]:


(market_train_df, news_train_df) = env.get_training_data()

# In[ ]:


print("In total: ", market_train_df.shape)

# In[ ]:


market_train_df.head()

# In[ ]:


print("In total: ", news_train_df.shape)

# In[ ]:


news_train_df.head()

# ## 2.3 Load test_data

# In[ ]:


days = env.get_prediction_days()
(market_obs_df, news_obs_df, predictions_template_df) = next(days)

# In[ ]:


print("In total: ", market_obs_df.shape)

# In[ ]:


market_obs_df.head()

# In[ ]:


print("In total: ", news_obs_df.shape)

# In[ ]:


news_obs_df.head()

# In[ ]:


predictions_template_df.head()

# In[ ]:


print("In market_train_df: ", market_train_df.shape);print("In market_obs_df: ", market_obs_df.shape);
print("In news_train_df: ", news_train_df.shape);print("In news_obs_df: ", news_obs_df.shape)

# - `market_train data` is about **40.7 million( 2007-02-01 ~ 2016-12-30 )** , `market_test` is about **1800  ( 2017-01-03 )**. Similarly, news is similar.  
# - The other two variables above are missing `assetName` and` universe`.

# ## 2.4 Data Description
# ### Market data
# Market data
# The data includes a subset of US-listed instruments. The set of included instruments changes daily and is determined based on the amount traded and the availability of information. This means that there may be instruments that enter and leave this subset of data. There may therefore be gaps in the data provided, and this does not necessarily imply that that data does not exist (those rows are likely not included due to the selection criteria).
# 
# The marketdata contains a variety of returns calculated over different timespans. All of the returns in this set of marketdata have these properties:
# 
# - Returns are always calculated either open-to-open (from the opening time of one trading day to the open of another) or close-to-close (from the closing time of one trading day to the open of another).
# - Returns are either raw, meaning that the data is not adjusted against any benchmark, or market-residualized (Mktres), meaning that the movement of the market as a whole has been accounted for, leaving only movements inherent to the instrument.
# - Returns can be calculated over any arbitrary interval. Provided here are 1 day and 10 day horizons.
# - Returns are tagged with 'Prev' if they are backwards looking in time, or 'Next' if forwards looking.
# 
# Within the marketdata, you will find the following columns:
# 
# - time(datetime64[ns, UTC]) - the current time (in marketdata, all rows are taken at 22:00 UTC)
# - assetCode(object) - a unique id of an asset
# - assetName(category) - the name that corresponds to a group of assetCodes. These may be "Unknown" if the corresponding assetCode does not have any rows in the news data.
# - universe(float64) - a boolean indicating whether or not the instrument on that day will be included in scoring. This value is not provided outside of the training data time period. The trading universe on a given date is the set of instruments that are avilable for trading (the scoring function will not consider instruments that are not in the trading universe). The trading universe changes daily.
# - volume(float64) - trading volume in shares for the day
# - close(float64) - the close price for the day (not adjusted for splits or dividends)
# - open(float64) - the open price for the day (not adjusted for splits or dividends)
# - returnsClosePrevRaw1(float64) - see returns explanation above
# - returnsOpenPrevRaw1(float64) - see returns explanation above
# - returnsClosePrevMktres1(float64) - see returns explanation above
# - returnsOpenPrevMktres1(float64) - see returns explanation above
# - returnsClosePrevRaw10(float64) - see returns explanation above
# - returnsOpenPrevRaw10(float64) - see returns explanation above
# - returnsClosePrevMktres10(float64) - see returns explanation above
# - returnsOpenPrevMktres10(float64) - see returns explanation above
# - returnsOpenNextMktres10(float64) - 10 day, market-residualized return. This is the target variable used in competition scoring. The market data has been filtered such that returnsOpenNextMktres10 is always not null.
# 
# ### News data
# The news data contains information at both the news article level and asset level (in other words, the table is intentionally not normalized).
# 
# - time(datetime64[ns, UTC]) - UTC timestamp showing when the data was available on the feed (second precision)
# - sourceTimestamp(datetime64[ns, UTC]) - UTC timestamp of this news item when it was created
# - firstCreated(datetime64[ns, UTC]) - UTC timestamp for the first version of the item
# - sourceId(object) - an Id for each news item
# - headline(object) - the item's headline
# - urgency(int8) - differentiates story types (1: alert, 3: article)
# - takeSequence(int16) - the take sequence number of the news item, starting at 1. For a given story, alerts and articles have separate sequences.
# - provider(category) - identifier for the organization which provided the news item (e.g. RTRS for Reuters News, BSW for Business Wire)
# - subjects(category) - topic codes and company identifiers that relate to this news item. Topic codes describe the news item's subject matter. These can cover asset classes, geographies, events, industries/sectors, and other types.
# - audiences(category) - identifies which desktop news product(s) the news item belongs to. They are typically tailored to specific audiences. (e.g. "M" for Money International News Service and "FB" for French General News Service)
# - bodySize(int32) - the size of the current version of the story body in characters
# - companyCount(int8) - the number of companies explicitly listed in the news item in the subjects field
# - headlineTag(object) - the Thomson Reuters headline tag for the news item
# - marketCommentary(bool) - boolean indicator that the item is discussing general market conditions, such as "After the Bell" summaries
# - sentenceCount(int16) - the total number of sentences in the news item. Can be used in conjunction with firstMentionSentence to determine the relative position of the first mention in the item.
# - wordCount(int32) - the total number of lexical tokens (words and punctuation) in the news item
# - assetCodes(category) - list of assets mentioned in the item
# - assetName(category) - name of the asset
# - firstMentionSentence(int16) - the first sentence, starting with the headline, in which the scored asset is mentioned.
#     -  1: headline
#     - 2: first sentence of the story body
#     - 3: second sentence of the body, etc
#     - 0: the asset being scored was not found in the news item's headline or body text. As a result, the entire news item's text (headline + body) will be used to determine the sentiment score.
# - relevance(float32) - a decimal number indicating the relevance of the news item to the asset. It ranges from 0 to 1. If the asset is mentioned in the headline, the relevance is set to 1. When the item is an alert (urgency == 1), relevance should be gauged by firstMentionSentence instead.
# - sentimentClass(int8) - indicates the predominant sentiment class for this news item with respect to the asset. The indicated class is the one with the highest probability.
# - sentimentNegative(float32) - probability that the sentiment of the news item was negative for the asset
# - sentimentNeutral(float32) - probability that the sentiment of the news item was neutral for the asset
# - sentimentPositive(float32) - probability that the sentiment of the news item was positive for the asset
# - sentimentWordCount(int32) - the number of lexical tokens in the sections of the item text that are deemed relevant to the asset. This can be used in conjunction with wordCount to determine the proportion of the news item discussing the asset.
# - noveltyCount12H(int16) - The 12 hour novelty of the content within a news item on a particular asset. It is calculated by comparing it with the asset-specific text over a cache of previous news items that contain the asset.
# - noveltyCount24H(int16) - same as above, but for 24 hours
# - noveltyCount3D(int16) - same as above, but for 3 days
# - noveltyCount5D(int16) - same as above, but for 5 days
# - noveltyCount7D(int16) - same as above, but for 7 days
# - volumeCounts12H(int16) - the 12 hour volume of news for each asset. A cache of previous news items is maintained and the number of news items that mention the asset within each of five historical periods is calculated.
# - volumeCounts24H(int16) - same as above, but for 24 hours
# - volumeCounts3D(int16) - same as above, but for 3 days
# - volumeCounts5D(int16) - same as above, but for 5 days
# - volumeCounts7D(int16) - same as above, but for 7 days

# # 3. Simple Exploration
# ## 3.1 Check null data
# - ### Market

# In[ ]:


percent = (100 * market_train_df.isnull().sum() / market_train_df.shape[0]).sort_values(ascending=False)
percent.plot(kind="bar", figsize = (20,10), fontsize = 20)
plt.xlabel("Columns", fontsize = 20)
plt.ylabel("Value Percent(%)", fontsize = 20)
plt.title("Total Missing Value by market_obs_df", fontsize = 20)

# In[ ]:


percent1 = (100 * market_obs_df.isnull().sum() / market_obs_df.shape[0]).sort_values(ascending=False)
percent1.plot(kind="bar", figsize = (20,10), fontsize = 20)
plt.xlabel("Columns", fontsize = 20)
plt.ylabel("Value Percent(%)", fontsize = 20)
plt.title("Total Missing Value by market_obs_df", fontsize = 20)

# The types of missing values are the same, but the percentage is slightly different.
# - **market_train_df** : { returnsOpenPrevMktres10 : 2.284680 , returnsClosePrevMktres10 : 2.283599, returnsOpenPrevMktres1 : 0.392540 , returnsClosePrevMktres1 : 0.392344 }
# 
# - **market_obs_df** : { returnsClosePrevMktres10 : 2.029622, returnsOpenPrevMktres10 : 2.029622, returnsOpenPrevMktres1 : 0.658256, returnsClosePrevMktres1 : 0.658256 }

# - ### news

# In[ ]:


news_train_df['headlineTag'].unique()[0:5]

# As shown above,, **`''`** is recognized as object. So we have to change  these values as missing.

# In[ ]:


# '' convert to NA
for i in news_train_df.columns.values.tolist():
    # Does NaN means no numbers, can '' be replaced with nan? I do not know this part.
    news_train_df[i] = news_train_df[i].replace('', np.nan)  
news_train_df['headlineTag'].unique()[0:5]
# I think it would be faster if you just replace object and categorical variables(not int,float). How do I fix the code?

# In[ ]:


percent = (100 * news_train_df.isnull().sum() / news_train_df.shape[0]).sort_values(ascending=False)
percent.plot(kind="bar", figsize = (20,10), fontsize = 20)
plt.xlabel("Columns", fontsize = 20)
plt.ylabel("Value Percent(%)", fontsize = 20)
plt.title("Total Missing Value by news_train_df", fontsize = 20)

# In[ ]:


# '' convert to NA
for i in news_obs_df.columns.values.tolist():
    # Does NaN means no numbers, can '' be replaced with nan? I do not know this part.
    news_obs_df[i] = news_obs_df[i].replace('', np.nan)

# In[ ]:


percent1 = (100 * news_obs_df.isnull().sum() / news_obs_df.shape[0]).sort_values(ascending=False)
percent1.plot(kind="bar", figsize = (20,10), fontsize = 20)
plt.xlabel("Columns", fontsize = 20)
plt.ylabel("Value Percent(%)", fontsize = 20)
plt.title("Total Missing Value by news_obs_df", fontsize = 20)

# - `headlineTag`, both the train and the test are close to 70% missing. 
# - `headline` has some missing values, but not the test.

# ## 3.2 Number of unique values

# In[ ]:


percent2 = (market_train_df.nunique()).sort_values(ascending=False)
percent2.plot(kind="bar", figsize = (20,10), fontsize = 20)
plt.xlabel("Columns", fontsize = 20)
plt.ylabel("Unique Number", fontsize = 20)
plt.title("Unique Number by market_train_df", fontsize = 20)

# In[ ]:


market_train_df.nunique()

# In[ ]:


percent2 = (market_obs_df.nunique()).sort_values(ascending=False)
percent2.plot(kind="bar", figsize = (20,10), fontsize = 20)
plt.xlabel("Columns", fontsize = 20)
plt.ylabel("Unique Number", fontsize = 20)
plt.title("Unique Number by market_obs_df", fontsize = 20)

# In[ ]:


market_obs_df.nunique()

# In[ ]:


percent2 = (news_train_df.nunique()).sort_values(ascending=False)
percent2.plot(kind="bar", figsize = (20,10), fontsize = 20)
plt.xlabel("Columns", fontsize = 20)
plt.ylabel("Unique Number", fontsize = 20)
plt.title("Unique Number by news_train_df", fontsize = 20)

# In[ ]:


news_train_df.nunique()

# In[ ]:


percent2 = (news_obs_df.nunique()).sort_values(ascending=False)
percent2.plot(kind="bar", figsize = (20,10), fontsize = 20)
plt.xlabel("Columns", fontsize = 20)
plt.ylabel("Unique Number", fontsize = 20)
plt.title("Unique Number by news_obs_df", fontsize = 20)

# In[ ]:


news_obs_df.nunique()

# ## 3.3 Object features

# In[ ]:


features_object = [col for col in market_train_df.columns if market_train_df[col].dtype == 'object']
features_object

# - assetCode(object) - a unique id of an asset

# In[ ]:


market_train_df['assetCode'].value_counts()

# In[ ]:


features_object = [col for col in news_train_df.columns if news_train_df[col].dtype == 'object']
features_object

# - sourceId(object) - an Id for each news item

# In[ ]:


news_train_df['sourceId'].value_counts()

# - headline(object) - the item's headline

# In[ ]:


news_train_df['headline'].value_counts()

# - headlineTag(object) - the Thomson Reuters headline tag for the news item

# In[ ]:


news_train_df['headlineTag'].value_counts()

# ## 3.4 Categorical features
# ### Market
# - assetName(category) - the name that corresponds to a group of assetCodes. These may be "Unknown" if the corresponding assetCode does not have any rows in the news data.
# 
# ### News
# - provider(category) - identifier for the organization which provided the news item (e.g. RTRS for Reuters News, BSW for Business Wire)
# - subjects(category) - topic codes and company identifiers that relate to this news item. Topic codes describe the news item's subject matter. These can cover asset classes, geographies, events, industries/sectors, and other types.
# - audiences(category) - identifies which desktop news product(s) the news item belongs to. They are typically tailored to specific audiences. (e.g. "M" for Money International News Service and "FB" for French General News Service)
# - assetCodes(category) - list of assets mentioned in the item
# - assetName(category) - name of the asset

# In[ ]:


news_train_df['provider'].value_counts()

# In[ ]:


news_train_df['subjects'].value_counts()

# In[ ]:


news_train_df['audiences'].value_counts()

# ## 3.5 Numeric features
# ### Market
# - universe(float64) - a boolean indicating whether or not the instrument on that day will be included in scoring. This value is not provided outside of the training data time period. The trading universe on a given date is the set of instruments that are avilable for trading (the scoring function will not consider instruments that are not in the trading universe). The trading universe changes daily.
# - volume(float64) - trading volume in shares for the day
# - close(float64) - the close price for the day (not adjusted for splits or dividends)
# - open(float64) - the open price for the day (not adjusted for splits or dividends)
# - returnsClosePrevRaw1(float64) - see returns explanation above
# - returnsOpenPrevRaw1(float64) - see returns explanation above
# - returnsClosePrevMktres1(float64) - see returns explanation above
# - returnsOpenPrevMktres1(float64) - see returns explanation above
# - returnsClosePrevRaw10(float64) - see returns explanation above
# - returnsOpenPrevRaw10(float64) - see returns explanation above
# - returnsClosePrevMktres10(float64) - see returns explanation above
# - returnsOpenPrevMktres10(float64) - see returns explanation above
# - returnsOpenNextMktres10(float64) - 10 day, market-residualized return. This is the target variable used in competition scoring. The market data has been filtered such that returnsOpenNextMktres10 is always not null.
# 
# 
# ### News
# - urgency(int8) - differentiates story types (1: alert, 3: article)
# - takeSequence(int16) - the take sequence number of the news item, starting at 1. For a given story, alerts and articles have separate sequences.
# - bodySize(int32) - the size of the current version of the story body in characters
# - companyCount(int8) - the number of companies explicitly listed in the news item in the subjects field
# - sentenceCount(int16) - the total number of sentences in the news item. Can be used in conjunction with firstMentionSentence to determine the relative position of the first mention in the item.
# - wordCount(int32) - the total number of lexical tokens (words and punctuation) in the news item
# - firstMentionSentence(int16) - the first sentence, starting with the headline, in which the scored asset is mentioned.
#     -  1: headline
#     - 2: first sentence of the story body
#     - 3: second sentence of the body, etc
#     - 0: the asset being scored was not found in the news item's headline or body text. As a result, the entire news item's text (headline + body) will be used to determine the sentiment score.
# - relevance(float32) - a decimal number indicating the relevance of the news item to the asset. It ranges from 0 to 1. If the asset is mentioned in the headline, the relevance is set to 1. When the item is an alert (urgency == 1), relevance should be gauged by firstMentionSentence instead.
# - sentimentClass(int8) - indicates the predominant sentiment class for this news item with respect to the asset. The indicated class is the one with the highest probability.
# - sentimentNegative(float32) - probability that the sentiment of the news item was negative for the asset
# - sentimentNeutral(float32) - probability that the sentiment of the news item was neutral for the asset
# - sentimentPositive(float32) - probability that the sentiment of the news item was positive for the asset
# - sentimentWordCount(int32) - the number of lexical tokens in the sections of the item text that are deemed relevant to the asset. This can be used in conjunction with wordCount to determine the proportion of the news item discussing the asset.
# - noveltyCount12H(int16) - The 12 hour novelty of the content within a news item on a particular asset. It is calculated by comparing it with the asset-specific text over a cache of previous news items that contain the asset.
# - noveltyCount24H(int16) - same as above, but for 24 hours
# - noveltyCount3D(int16) - same as above, but for 3 days
# - noveltyCount5D(int16) - same as above, but for 5 days
# - noveltyCount7D(int16) - same as above, but for 7 days
# - volumeCounts12H(int16) - the 12 hour volume of news for each asset. A cache of previous news items is maintained and the number of news items that mention the asset within each of five historical periods is calculated.
# - volumeCounts24H(int16) - same as above, but for 24 hours
# - volumeCounts3D(int16) - same as above, but for 3 days
# - volumeCounts5D(int16) - same as above, but for 5 days
# - volumeCounts7D(int16) - same as above, but for 7 days

# In[ ]:


(market_train_df['universe']).describe()

# In[ ]:


market_train_df['universe'].plot.hist(title = 'universe Histogram');
plt.xlabel('universe');

# ## 3.6 Simple NLP

# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer

news_train_df.head()

# ## headline
# - ### CountVectorizer

# In[ ]:


list(news_train_df['headline'])[0:5]

# In[ ]:


# CountVectorizer() env
news_train_df['headline'] = news_train_df['headline'].replace(np.nan, '')
news_train_df['headlineTag'] = news_train_df['headlineTag'].replace(np.nan, '')

# In[ ]:


vect = CountVectorizer()
vect.fit(list(news_train_df['headline']))

# In[ ]:


list((vect.vocabulary_).items())[0:10]

# In[ ]:


vect.vocabulary_ = sorted(vect.vocabulary_.items(), key=lambda x: x[1], reverse=True)

# In[ ]:


(vect.vocabulary_)[0:10]

# - ### n-gram

# In[ ]:


vect1 = CountVectorizer(ngram_range=(2, 2))
vect1.fit(list(news_train_df['headline']))

# In[ ]:


list((vect1.vocabulary_).items())[0:10]

# In[ ]:


vect1.vocabulary_ = sorted(vect1.vocabulary_.items(), key=lambda x: x[1], reverse=True)

# In[ ]:


(vect1.vocabulary_)[0:10]

# - NLP is the first time I'll study a little more and upload the kernel.

# # 4. Time-Series Analysis some Stock

# **train_df :** 2007-02-01 ~ 2016-12-30
# 
# **test_df:** 2017-01-03

# In[ ]:


market_train_df.time.head()

# In[ ]:


market_train_df.time.tail()

# In[ ]:


market_obs_df.time.head()

# In[ ]:


market_obs_df.time.tail()

# In[ ]:


def change_date_to_datetime(x):
    str_time = str(x)
    date = '{}-{}-{}'.format(str_time[:4], str_time[5:7], str_time[8:10])
    return date

market_train_df['date'] = market_train_df['time'].apply(change_date_to_datetime)

# In[ ]:


def add_time_feature(data):
    data['date'] = pd.to_datetime(data['date'])
    data['Year'] = data.date.dt.year
    data['Month'] = data.date.dt.month
    data['Day'] = data.date.dt.day
    data['WeekOfYear'] = data.date.dt.weekofyear
    return data

market_train_df = add_time_feature(market_train_df)

# ### Top 10 Largest Assets code by Open, Close, Volume value

# In[ ]:


best_asset_open = market_train_df.groupby("assetCode")["open"].count().to_frame().sort_values(by=['open'],ascending= False)
best_asset_open = best_asset_open.sort_values(by=['open'])
largest_by_open = list(best_asset_open.nlargest(10, ['open']).index)

best_asset_close = market_train_df.groupby("assetCode")["close"].count().to_frame().sort_values(by=['close'],ascending= False)
best_asset_close = best_asset_close.sort_values(by=['close'])
largest_by_close = list(best_asset_close.nlargest(10, ['close']).index)

best_asset_volume = market_train_df.groupby("assetCode")["volume"].count().to_frame().sort_values(by=['volume'],ascending= False)
best_asset_volume = best_asset_volume.sort_values(by=['volume'])
largest_by_volume = list(best_asset_volume.nlargest(10, ['volume']).index)

# In[ ]:


print(largest_by_open)
print(largest_by_close)
print(largest_by_volume)

# - Top 10 open, close, volume same.
# - Top 10 assetName : 
# *Cardinal Health Inc, Conagra Brands Inc, UnitedHealth Group Inc, Unilever PLC, Cameco Corp, Tim Participacoes SA, Universal Health Services Inc, 	Tyson Foods Inc, CMS Energy Corp*
# 

# ### Asset - Conagra Brands Inc, Apple Inc
# The code that implements the candle chart refers to this kernel. (https://www.kaggle.com/pestipeti/simple-eda-two-sigma)

# In[ ]:


import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

# In[ ]:


asset1Code = 'CAH.N'
asset1_df = market_train_df[(market_train_df['assetCode'] == asset1Code) & (market_train_df['time'] > '2015-01-01') & (market_train_df['time'] < '2017-01-01')]

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

# In[ ]:


asset1_df = market_train_df[(market_train_df['assetCode'] == 'CAH.N') & (market_train_df['time'] > '2015-01-01') & (market_train_df['time'] < '2017-01-01')]
    # Create a trace
trace1 = go.Scatter(
        x = asset1_df['time'].dt.strftime(date_format='%Y-%m-%d').values,
        y = asset1_df['open'].values
    )

layout = dict(title = "Open prices of CAH.N",
                  xaxis = dict(title = 'Month'),
                  yaxis = dict(title = 'Price (USD)'),
                  )
data = [trace1]
py.iplot(dict(data=data, layout=layout), filename='basic-line')

# In[ ]:


asset1_df = market_train_df[(market_train_df['assetCode'] == 'CAH.N') & (market_train_df['time'] > '2015-01-01') & (market_train_df['time'] < '2017-01-01')]
    # Create a trace
trace1 = go.Scatter(
        x = asset1_df['time'].dt.strftime(date_format='%Y-%m-%d').values,
        y = asset1_df['close'].values
    )

layout = dict(title = "Closing prices of CAH.N",
                  xaxis = dict(title = 'Month'),
                  yaxis = dict(title = 'Price (USD)'),
                  )
data = [trace1]
py.iplot(dict(data=data, layout=layout), filename='basic-line')

# In[ ]:


asset1_df = market_train_df[(market_train_df['assetCode'] == 'CAH.N') & (market_train_df['time'] > '2015-01-01') & (market_train_df['time'] < '2017-01-01')]
    # Create a trace
trace1 = go.Scatter(
        x = asset1_df['time'].dt.strftime(date_format='%Y-%m-%d').values,
        y = asset1_df['volume'].values
    )

layout = dict(title = "Volume of CAH.N",
                  xaxis = dict(title = 'Month'),
                  yaxis = dict(title = 'Price (USD)'),
                  )
data = [trace1]
py.iplot(dict(data=data, layout=layout), filename='basic-line')

# In[ ]:


asset1Code = 'AAPL.O'
asset1_df = market_train_df[(market_train_df['assetCode'] == asset1Code) & (market_train_df['time'] > '2015-01-01') & (market_train_df['time'] < '2017-01-01')]

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

# In[ ]:


asset1_df = market_train_df[(market_train_df['assetCode'] == 'AAPL.O') & (market_train_df['time'] > '2015-01-01') & (market_train_df['time'] < '2017-01-01')]
    # Create a trace
trace1 = go.Scatter(
        x = asset1_df['time'].dt.strftime(date_format='%Y-%m-%d').values,
        y = asset1_df['open'].values
    )

layout = dict(title = "Open prices of AAPL.O",
                  xaxis = dict(title = 'Month'),
                  yaxis = dict(title = 'Price (USD)'),
                  )
data = [trace1]
py.iplot(dict(data=data, layout=layout), filename='basic-line')

# In[ ]:


asset1_df = market_train_df[(market_train_df['assetCode'] == 'AAPL.O') & (market_train_df['time'] > '2015-01-01') & (market_train_df['time'] < '2017-01-01')]
    # Create a trace
trace1 = go.Scatter(
        x = asset1_df['time'].dt.strftime(date_format='%Y-%m-%d').values,
        y = asset1_df['close'].values
    )

layout = dict(title = "Closing prices of AAPL.O",
                  xaxis = dict(title = 'Month'),
                  yaxis = dict(title = 'Price (USD)'),
                  )
data = [trace1]
py.iplot(dict(data=data, layout=layout), filename='basic-line')

# In[ ]:


asset1_df = market_train_df[(market_train_df['assetCode'] == 'AAPL.O') & (market_train_df['time'] > '2015-01-01') & (market_train_df['time'] < '2017-01-01')]
    # Create a trace
trace1 = go.Scatter(
        x = asset1_df['time'].dt.strftime(date_format='%Y-%m-%d').values,
        y = asset1_df['volume'].values
    )

layout = dict(title = "Volume of AAPL.O",
                  xaxis = dict(title = 'Month'),
                  yaxis = dict(title = 'Price (USD)'),
                  )
data = [trace1]
py.iplot(dict(data=data, layout=layout), filename='basic-line')

# ### ECDF: empirical cumulative distribution function
# 
# To get the first impression about continious variables in the data we can plot ECDF.

# In[ ]:


# data visualization
import matplotlib.pyplot as plt
import seaborn as sns # advanced vizs

# statistics
from statsmodels.distributions.empirical_distribution import ECDF

# time series analysis
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# prophet by Facebook
from fbprophet import Prophet

# In[ ]:


sns.set(style = "ticks")# to format into seaborn 
c = '#386B7F' # basic color for plots
plt.figure(figsize = (12, 6))

plt.subplot(311)
cdf = ECDF(market_train_df['volume'])
plt.plot(cdf.x, cdf.y, label = "statmodels", color = c);
plt.xlabel(''); plt.ylabel('ECDF');

# plot second ECDF  
plt.subplot(312)
cdf = ECDF(market_train_df['open'])
plt.plot(cdf.x, cdf.y, label = "statmodels", color = c);
plt.xlabel('');plt.ylabel('ECDF');

# plot second ECDF  
plt.subplot(313)
cdf = ECDF(market_train_df['close'])
plt.plot(cdf.x, cdf.y, label = "statmodels", color = c);
plt.xlabel('');plt.ylabel('ECDF');

# In[ ]:


market_train_df[(market_train_df.assetCode=='A.N')].head()

# ### Seasonality

# In[ ]:


voluem_an = np.transpose(pd.DataFrame([(market_train_df[(market_train_df.assetCode=='A.N')]).set_index('date')['volume']]))

# In[ ]:


voluem_an = np.transpose(pd.DataFrame([(market_train_df[(market_train_df.assetCode=='A.N')]).set_index('date')['volume']]))
open_an = np.transpose(pd.DataFrame([(market_train_df[(market_train_df.assetCode=='A.N')]).set_index('date')['open']]))
close_an = np.transpose(pd.DataFrame([(market_train_df[(market_train_df.assetCode=='A.N')]).set_index('date')['close']]))

f, (ax1, ax2, ax3) = plt.subplots(3, figsize = (12, 13))

# store types
voluem_an.resample('W').sum().plot(color = c, ax = ax1)
open_an.resample('W').sum().plot(color = c, ax = ax2)
close_an.resample('W').sum().plot(color = c, ax = ax3)

# ### Yearly

# In[ ]:


f, (ax1, ax2, ax3) = plt.subplots(3, figsize = (12, 13))

# monthly
decomposition_a = seasonal_decompose(voluem_an, model = 'additive', freq = 365)
decomposition_a.trend.plot(color = c, ax = ax1)

decomposition_b = seasonal_decompose(open_an, model = 'additive', freq = 365)
decomposition_b.trend.plot(color = c, ax = ax2)

decomposition_c = seasonal_decompose(close_an, model = 'additive', freq = 365)
decomposition_c.trend.plot(color = c, ax = ax3)

# ### Autocorrelaion

# In[ ]:


# figure for subplots
plt.figure(figsize = (12, 8))

# acf and pacf for volume
plt.subplot(321); plot_acf(voluem_an, lags = 50, ax = plt.gca(), color = c)
plt.subplot(322); plot_pacf(voluem_an, lags = 50, ax = plt.gca(), color = c)

# acf and pacf for open
plt.subplot(323); plot_acf(open_an, lags = 50, ax = plt.gca(), color = c)
plt.subplot(324); plot_pacf(open_an, lags = 50, ax = plt.gca(), color = c)

# acf and pacf for close
plt.subplot(325); plot_acf(close_an, lags = 50, ax = plt.gca(), color = c)
plt.subplot(326); plot_pacf(close_an, lags = 50, ax = plt.gca(), color = c)

plt.show()

# ### Time Series Analysis and Forecasting with Prophet
# 
# The Core Data Science team at Facebook recently published a new procedure for forecasting time series data called Prophet. It is based on an additive model where non-linear trends are fit with yearly and weekly seasonality, plus holidays. It enables performing automated forecasting which are already implemented in R at scale in Python 3.

# In[ ]:


df = market_train_df[(market_train_df["assetCode"] == 'A.N')]

volume = df.loc[:, ['date', 'volume']]

# reverse to the order: from 2013 to 2015
volume = volume.sort_index(ascending = True)

# to datetime64
volume['date'] = pd.DatetimeIndex(volume['date'])
volume.dtypes

# from the prophet documentation every variables should have specific names
volume = volume.rename(columns = {'date': 'ds',
                                'volume': 'y'})
volume.head()

# In[ ]:


# plot daily sales
ax = volume.set_index('ds').plot(figsize = (12, 4), color = c)
ax.set_ylabel('Daily volume of A.N')
ax.set_xlabel('Date')
plt.show()

# In[ ]:


# set the uncertainty interval to 95% (the Prophet default is 80%)
my_model = Prophet(interval_width = 0.95)
my_model.fit(volume)

# dataframe that extends into future 6 weeks 
future_dates = my_model.make_future_dataframe(periods = 1)

print("First day to forecast.")
future_dates

# predictions
forecast = my_model.predict(future_dates)

# preditions for last week
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

# In[ ]:


fc = forecast[['ds', 'yhat']].rename(columns = {'Date': 'ds', 'Forecast': 'yhat'})

# In[ ]:


my_model.plot(forecast);

# In[ ]:


my_model.plot_components(forecast);

# **Note :** I am still a student and there are many deficiencies. If there is a part that you think is insufficient or you think that you want to add, please give advice. Note that I will update the kernel. If this kernel is helpful, please upvote.
# 
