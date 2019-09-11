#!/usr/bin/env python
# coding: utf-8

# # Aim
# Since my first week on this platform, I have been fascinated by the topic of **time series analysis.** This kernel is prepared to be a container of many broad topics in the field of time series analysis. My motive is to make this the ultimate reference to time series analysis for beginners and experienced people alike.
# 
# # Some important things
# 1. This kernel **is a work in progress so every time you see on your home feed and open it, you will surely find fresh content.**
# 2. I am doing this only after completing various courses in this field. I continue to study more advanced concepts to provide more knowledge and content.
# 3. If there is any suggestion or any specific topic you would like me to cover, kindly mention that in the comments.
# 4. **If you like my work, be sure to upvote**(press the like button) this kernel so it looks more relevant and meaningful to the community.

# In[ ]:


# Importing libraries
import os
import warnings
warnings.filterwarnings('ignore')
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight') 
# Above is a special style template for matplotlib, highly useful for visualizing time series data
from pylab import rcParams
from plotly import tools
import plotly.plotly as py
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.figure_factory as ff
import statsmodels.api as sm
from numpy.random import normal, seed
from scipy.stats import norm
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.tsa.arima_model import ARIMA
import math
from sklearn.metrics import mean_squared_error
print(os.listdir("../input"))

# - <a href='#1'>1. Introduction to date and time</a>
#     - <a href='#1.1'>1.1 Importing time series data</a>
#     - <a href='#1.2'>1.2 Cleaning and preparing time series data</a>
#     - <a href='#1.3'>1.3 Visualizing the datasets</a>
#     - <a href='#1.4'>1.4 Timestamps and Periods</a>
#     - <a href='#1.5'>1.5 Using date_range</a>
#     - <a href='#1.6'>1.6 Using to_datetime</a>
#     - <a href='#1.7'>1.7 Shifting and lags</a>
#     - <a href='#1.8'>1.8 Resampling</a>
# - <a href='#2'>2. Finance and Statistics</a>
#     - <a href='#2.1'>2.1 Percent change</a>
#     - <a href='#2.2'>2.2 Stock returns</a>
#     - <a href='#2.3'>2.3 Absolute change in successive rows</a>
#     - <a href='#2.4'>2.4 Comaring two or more time series</a>
#     - <a href='#2.5'>2.5 Window functions</a>
#     - <a href='#2.6'>2.6 OHLC charts</a>
#     - <a href='#2.7'>2.7 Candlestick charts</a>
#     - <a href='#2.8'>2.8 Autocorrelation and Partial Autocorrelation</a>
# - <a href='#3'>3. Time series decomposition and Random Walks</a>
#     - <a href='#3.1'>3.1 Trends, Seasonality and Noise</a>
#     - <a href='#3.2'>3.2 White Noise</a>
#     - <a href='#3.3'>3.3 Random Walk</a>
#     - <a href='#3.4'>3.4 Stationarity</a>
# - <a href='#4'>4. Modelling using statsmodels</a>
#     - <a href='#4.1'>4.1 AR models</a>
#     - <a href='#4.2'>4.2 MA models</a>
#     - <a href='#4.3'>4.3 ARMA models</a>
#     - <a href='#4.4'>4.4 ARIMA models</a>
#     - <a href='#4.5'>4.5 VAR models</a>
#     - <a href='#4.6'>4.6 State space methods</a>
#         - <a href='#4.6.1'>4.6.1 SARIMA models</a>
#         - <a href='#4.6.2'>4.6.2 Unobserved components</a>
#         - <a href='#4.6.3'>4.6.3 Dynamic Factor models</a>

# # <a id='1'>1. Introduction to date and time</a>

# ## <a id='1.1'>1.1 Importing time series data</a>

# ### How to import data?
# First, we import all the datasets needed for this kernel. The required time series column is imported as a datetime column using **parse_dates** parameter and is also selected as index of the dataframe using **index_col** parameter. 
# #### Data being used:-
# 1. Google  Stocks Data
# 2. Humidity in different world cities
# 3. Microsoft  Stocks Data
# 3. Pressure in different world cities

# In[ ]:


google = pd.read_csv('../input/stock-time-series-20050101-to-20171231/GOOGL_2006-01-01_to_2018-01-01.csv', index_col='Date', parse_dates=['Date'])
google.head()

# In[ ]:


humidity = pd.read_csv('../input/historical-hourly-weather-data/humidity.csv', index_col='datetime', parse_dates=['datetime'])
humidity.tail()

# ## <a id='1.2'>1.2 Cleaning and preparing time series data</a>

# ### How to prepare data?
# Google stocks data doesn't have any missing values but humidity data does have its fair share of missing values. It is cleaned using **fillna()** method with **ffill** parameter which propagates last valid observation to fill gaps

# In[ ]:


humidity = humidity.iloc[1:]
humidity = humidity.fillna(method='ffill')
humidity.head()

# ## <a id='1.3'>1.3 Visualizing the datasets</a>

# In[ ]:


humidity["Kansas City"].asfreq('M').plot() # asfreq method is used to convert a time series to a specified frequency. Here it is monthly frequency.
plt.title('Humidity in Kansas City over time(Monthly frequency)')
plt.show()

# In[ ]:


google['2008':'2010'].plot(subplots=True, figsize=(10,12))
plt.title('Google stock attributes from 2008 to 2010')
plt.savefig('stocks.png')
plt.show()

# ## <a id='1.4'>1.4 Timestamps and Periods</a>

# ### What are timestamps and periods and how are they useful?
# Timestamps are used to represent a point in time. Periods represent an interval in time. Periods can used to check if a specific event in the given period. They can also be converted to each other's form.

# In[ ]:


# Creating a Timestamp
timestamp = pd.Timestamp(2017, 1, 1, 12)
timestamp

# In[ ]:


# Creating a period
period = pd.Period('2017-01-01')
period

# In[ ]:


# Checking if the given timestamp exists in the given period
period.start_time < timestamp < period.end_time

# In[ ]:


# Converting timestamp to period
new_period = timestamp.to_period(freq='H')
new_period

# In[ ]:


# Converting period to timestamp
new_timestamp = period.to_timestamp(freq='H', how='start')
new_timestamp

# ## <a id='1.5'>1.5 Using date_range</a>

# ### What is date_range and how is it useful?
# **date_range** is a method that returns  a fixed frequency datetimeindex. It is quite useful when creating your own time series attribute for pre-existing data or arranging the whole data around the time series attribute created by you.

# In[ ]:


# Creating a datetimeindex with daily frequency
dr1 = pd.date_range(start='1/1/18', end='1/9/18')
dr1

# In[ ]:


# Creating a datetimeindex with monthly frequency
dr2 = pd.date_range(start='1/1/18', end='1/1/19', freq='M')
dr2

# In[ ]:


# Creating a datetimeindex without specifying start date and using periods
dr3 = pd.date_range(end='1/4/2014', periods=8)
dr3

# In[ ]:


# Creating a datetimeindex specifying start date , end date and periods
dr4 = pd.date_range(start='2013-04-24', end='2014-11-27', periods=3)
dr4

# ## <a id='1.6'>1.6 Using to_datetime</a> 

# pandas.to_datetime() is used for converting arguments to datetime. Here, a DataFrame is converted to a datetime series.

# In[ ]:


df = pd.DataFrame({'year': [2015, 2016], 'month': [2, 3], 'day': [4, 5]})
df

# In[ ]:


df = pd.to_datetime(df)
df

# In[ ]:


df = pd.to_datetime('01-01-2017')
df

# ## <a id='1.7'>1.7 Shifting and lags</a>

# We can shift index by desired number of periods with an optional time frequency. This is useful when comparing the time series with a past of itself

# In[ ]:


humidity["Vancouver"].asfreq('M').plot(legend=True)
shifted = humidity["Vancouver"].asfreq('M').shift(10).plot(legend=True)
shifted.legend(['Vancouver','Vancouver_lagged'])
plt.show()

# ## <a id='#1.8'>1.8 Resampling</a>
# **Upsampling** - Time series is resampled from low frequency to high frequency(Monthly to daily frequency). It involves filling or interpolating missing data
# 
# **Downsampling** - Time series is resampled from high frequency to low frequency(Weekly to monthly frequency). It involves aggregation of existing data.
# 

# In[ ]:


# Let's use pressure data to demonstrate this
pressure = pd.read_csv('../input/historical-hourly-weather-data/pressure.csv', index_col='datetime', parse_dates=['datetime'])
pressure.tail()

# Sigh! A lot of cleaning is required.

# In[ ]:


pressure = pressure.iloc[1:]
pressure = pressure.fillna(method='ffill')
pressure.tail()

# In[ ]:


pressure = pressure.fillna(method='bfill')
pressure.head()

# First, we used **ffill** parameter which propagates last valid observation to fill gaps. Then we use **bfill** to propogate next valid observation to fill gaps.

# In[ ]:


# Shape before resampling(downsampling)
pressure.shape

# In[ ]:


# We downsample from hourly to 3 day frequency aggregated using mean
pressure = pressure.resample('3D').mean()
pressure.head()

# In[ ]:


# Shape after resampling(downsampling)
pressure.shape

# Much less rows are left. Now, we will upsample from 3 day frequency to daily frequency

# In[ ]:


pressure = pressure.resample('D').pad()
pressure.head()

# In[ ]:


# Shape after resampling(upsampling)
pressure.shape

# Again an increase in number of rows. Resampling is cool when used properly.

# # <a id='2'>2. Finance and statistics</a>

# ## <a id='2.1'>2.1 Percent change</a>

# In[ ]:


google['Change'] = google.High.div(google.High.shift())
google['Change'].plot(figsize=(20,8))

# ## <a id='2.2'>2.2 Stock returns</a> 

# In[ ]:


google['Return'] = google.Change.sub(1).mul(100)
google['Return'].plot(figsize=(20,8))

# In[ ]:


google.High.pct_change().mul(100).plot(figsize=(20,6)) # Another way to calculate returns

# ## <a id='2.3'>2.3 Absolute change in successive rows</a>

# In[ ]:


google.High.diff().plot(figsize=(20,6))

# ## <a id='2.4'>2.4 Comaring two or more time series</a> 

# We will compare 2 time series by normalizing them. This is achieved by dividing each time series element of all time series by the first element. This way both series start at the same point and can be easily compared.

# In[ ]:


# We choose microsoft stocks to compare them with google
microsoft = pd.read_csv('../input/stock-time-series-20050101-to-20171231/MSFT_2006-01-01_to_2018-01-01.csv', index_col='Date', parse_dates=['Date'])

# In[ ]:


# Plotting before normalization
google.High.plot()
microsoft.High.plot()
plt.legend(['Google','Microsoft'])
plt.show()

# In[ ]:


# Normalizing and comparison
# Both stocks start from 100
normalized_google = google.High.div(google.High.iloc[0]).mul(100)
normalized_microsoft = microsoft.High.div(microsoft.High.iloc[0]).mul(100)
normalized_google.plot()
normalized_microsoft.plot()
plt.legend(['Google','Microsoft'])
plt.show()

# You can clearly see how google outperforms microsoft over time.

# ## <a id='2.5'>2.5 Window functions</a>
# Window functions are used to identify sub periods, calculates sub-metrics of sub-periods.
# 
# **Rolling** - Same size and sliding
# 
# **Expanding** - Contains all prior values

# In[ ]:


# Rolling window functions
rolling_google = google.High.rolling('90D').mean()
google.High.plot()
rolling_google.plot()
plt.legend(['High','Rolling Mean'])
# Plotting a rolling mean of 90 day window with original High attribute of google stocks
plt.show()

# Now, observe that rolling mean plot is a smoother version of the original plot.

# In[ ]:


# Expanding window functions
microsoft_mean = microsoft.High.expanding().mean()
microsoft_std = microsoft.High.expanding().std()
microsoft.High.plot()
microsoft_mean.plot()
microsoft_std.plot()
plt.legend(['High','Expanding Mean','Expanding Standard Deviation'])
plt.show()

# ## <a id='2.6'>2.6 OHLC charts</a>
# An OHLC chart is any type of price chart that shows the open, high, low and close price of a certain time period. Open-high-low-close Charts (or OHLC Charts) are used as a trading tool to visualise and analyse the price changes over time for securities, currencies, stocks, bonds, commodities, etc. OHLC Charts are useful for interpreting the day-to-day sentiment of the market and forecasting any future price changes through the patterns produced.
# 
# The y-axis on an OHLC Chart is used for the price scale, while the x-axis is the timescale. On each single time period, an OHLC Charts plots a symbol that represents two ranges: the highest and lowest prices traded, and also the opening and closing price on that single time period (for example in a day). On the range symbol, the high and low price ranges are represented by the length of the main vertical line. The open and close prices are represented by the vertical positioning of tick-marks that appear on the left (representing the open price) and on right (representing the close price) sides of the high-low vertical line.
# 
# Colour can be assigned to each OHLC Chart symbol, to distinguish whether the market is "bullish" (the closing price is higher then it opened) or "bearish" (the closing price is lower then it opened).
# 
# <img src="https://datavizcatalogue.com/methods/images/anatomy/SVG/OHLC_chart.svg">
# 
# Source: [Datavizcatalogue](https://datavizcatalogue.com/methods/OHLC_chart.html)

# In[ ]:


# OHLC chart of June 2008
trace = go.Ohlc(x=google['06-2008'].index,
                open=google['06-2008'].Open,
                high=google['06-2008'].High,
                low=google['06-2008'].Low,
                close=google['06-2008'].Close)
data = [trace]
iplot(data, filename='simple_ohlc')

# In[ ]:


# OHLC chart of 2008
trace = go.Ohlc(x=google['2008'].index,
                open=google['2008'].Open,
                high=google['2008'].High,
                low=google['2008'].Low,
                close=google['2008'].Close)
data = [trace]
iplot(data, filename='simple_ohlc')

# In[ ]:


# OHLC chart of 2008
trace = go.Ohlc(x=google.index,
                open=google.Open,
                high=google.High,
                low=google.Low,
                close=google.Close)
data = [trace]
iplot(data, filename='simple_ohlc')

#  ## <a id='2.7'>2.7 Candlestick charts</a>
# This type of chart is used as a trading tool to visualise and analyse the price movements over time for securities, derivatives, currencies, stocks, bonds, commodities, etc. Although the symbols used in Candlestick Charts resemble a Box Plot, they function differently and therefore, are not to be confused with one another.
# 
# Candlestick Charts display multiple bits of price information such as the open price, close price, highest price and lowest price through the use of candlestick-like symbols. Each symbol represents the compressed trading activity for a single time period (a minute, hour, day, month, etc). Each Candlestick symbol is plotted along a time scale on the x-axis, to show the trading activity over time.
# 
# The main rectangle in the symbol is known as the real body, which is used to display the range between the open and close price of that time period. While the lines extending from the bottom and top of the real body is known as the lower and upper shadows (or wick). Each shadow represents the highest or lowest price traded during the time period represented. When the market is Bullish (the closing price is higher than it opened), then the body is coloured typically white or green. But when the market is Bearish (the closing price is lower than it opened), then the body is usually coloured either black or red.
# 
# <img src="https://datavizcatalogue.com/methods/images/anatomy/SVG/candlestick_chart.svg">
# 
# Candlestick Charts are great for detecting and predicting market trends over time and are useful for interpreting the day-to-day sentiment of the market, through each candlestick symbol's colouring and shape. For example, the longer the body is, the more intense the selling or buying pressure is. While, a very short body, would indicate that there is very little price movement in that time period and represents consolidation.
# 
# Candlestick Charts help reveal the market psychology (the fear and greed experienced by sellers and buyers) through the various indicators, such as shape and colour, but also by the many identifiable patterns that can be found in Candlestick Charts. In total, there are 42 recognised patterns that are divided into simple and complex patterns. These patterns found in Candlestick Charts are useful for displaying price relationships and can be used for predicting the possible future movement of the market. You can find a list and description of each pattern here.
# 
# Please bear in mind, that Candlestick Charts don't express the events taking place between the open and close price - only the relationship between the two prices. So you can't tell how volatile trading was within that single time period.
# 
# Source: [Datavizcatalogue](https://datavizcatalogue.com/methods/candlestick_chart.html)

# In[ ]:


# Candlestick chart of march 2008
trace = go.Candlestick(x=google['03-2008'].index,
                open=google['03-2008'].Open,
                high=google['03-2008'].High,
                low=google['03-2008'].Low,
                close=google['03-2008'].Close)
data = [trace]
iplot(data, filename='simple_candlestick')

# In[ ]:


# Candlestick chart of 2008
trace = go.Candlestick(x=google['2008'].index,
                open=google['2008'].Open,
                high=google['2008'].High,
                low=google['2008'].Low,
                close=google['2008'].Close)
data = [trace]
iplot(data, filename='simple_candlestick')

# In[ ]:


# Candlestick chart of 2006-2018
trace = go.Candlestick(x=google.index,
                open=google.Open,
                high=google.High,
                low=google.Low,
                close=google.Close)
data = [trace]
iplot(data, filename='simple_candlestick')

# ## <a id='2.8'>2.8 Autocorrelation and Partial Autocorrelation</a>
# * Autocorrelation - The autocorrelation function (ACF) measures how a series is correlated with itself at different lags.
# * Partial Autocorrelation - The partial autocorrelation function can be interpreted as a regression of the series against its past lags.  The terms can be interpreted the same way as a standard  linear regression, that is the contribution of a change in that particular lag while holding others constant. 
# 
# Source: [Quora](https://www.quora.com/What-is-the-difference-among-auto-correlation-partial-auto-correlation-and-inverse-auto-correlation-while-modelling-an-ARIMA-series)

# ## Autocorrelation

# In[ ]:


# Autocorrelation of humidity of San Diego
plot_acf(humidity["San Diego"],lags=25,title="San Diego")
plt.show()

#  As all lags are either close to 1 or at least greater than the confidence interval, they are statistically significant.

# ## Partial Autocorrelation

# In[ ]:


# Partial Autocorrelation of humidity of San Diego
plot_pacf(humidity["San Diego"],lags=25)
plt.show()

# Though it is statistically signficant, partial autocorrelation after first 2 lags is very low.

# In[ ]:


# Partial Autocorrelation of closing price of microsoft stocks
plot_pacf(microsoft["Close"],lags=25)
plt.show()

# Here, only 0th, 1st and 20th lag are statistically significant.

# # <a id='3'>3. Time series decomposition and Random walks</a>

# ## <a id='3.1'>3.1. Trends, seasonality and noise</a>
# These are the components of a time series
# * Trend - Consistent upwards or downwards slope of a time series
# * Seasonality - Clear periodic pattern of a time series(like sine funtion)
# * Noise - Outliers or missing values

# In[ ]:


# Let's take Google stocks High for this
google["High"].plot(figsize=(16,8))

# In[ ]:


# Now, for decomposition...
rcParams['figure.figsize'] = 11, 9
decomposed_google_volume = sm.tsa.seasonal_decompose(google["High"],freq=360) # The frequncy is annual
figure = decomposed_google_volume.plot()
plt.show()

# * There is clearly an upward trend in the above plot.
# * You can also see the uniform seasonal change.
# * Non-uniform noise that represent outliers and missing values

# ## <a id='3.2'>3.2. White noise</a>
# White noise has...
# * Constant mean
# * Constant variance
# * Zero auto-correlation at all lags

# In[ ]:


# Plotting white noise
rcParams['figure.figsize'] = 16, 6
white_noise = np.random.normal(loc=0, scale=1, size=1000)
# loc is mean, scale is variance
plt.plot(white_noise)

# In[ ]:


# Plotting autocorrelation of white noise
plot_acf(white_noise,lags=20)
plt.show()

# See how all lags are statistically insigficant as they lie inside the confidence interval(shaded portion).

# ## <a id='3.3'>3.3. Random Walk</a>
# A random walk is a mathematical object, known as a stochastic or random process, that describes a path that consists of a succession of random steps on some mathematical space such as the integers. 
# 
# In general if we talk about stocks, Today's Price = Yesterday's Price + Noise
# 
# # P<sub>t</sub> = P<sub>t-1</sub> + ε<sub>t</sub> 
# 
# Random walks can't be forecasted because well, noise is random.
# 
# Random Walk with Drift(drift(μ) is zero-mean)
# #### P<sub>t</sub> - P<sub>t-1</sub> = μ + ε<sub>t</sub> 
# <br>
# Regression test for random walk 
# #### P<sub>t</sub> = α + βP<sub>t-1</sub> + ε<sub>t</sub>
# #### Equivalent to  P<sub>t</sub> - P<sub>t-1</sub> = α + βP<sub>t-1</sub> + ε<sub>t</sub>
# <br>
# Test:
# #### H<sub>0</sub>: β = 1 (This is a random walk)
# #### H<sub>1</sub>: β < 1 (This is not a random walk)
# <br>
# Dickey-Fuller Test:
# #### H<sub>0</sub>: β = 0 (This is a random walk)
# #### H<sub>1</sub>: β < 0 (This is not a random walk)

# ### Augmented Dickey-Fuller test
# An augmented Dickey–Fuller test (ADF) tests the null hypothesis that a unit root is present in a time series sample. It is basically Dickey-Fuller test with more lagged changes on RHS.

# In[ ]:


# Augmented Dickey-Fuller test on volume of google and microsoft stocks 
adf = adfuller(microsoft["Volume"])
print("p-value of microsoft: {}".format(float(adf[1])))
adf = adfuller(google["Volume"])
print("p-value of google: {}".format(float(adf[1])))

# ##### As microsoft has p-value 0.0003201525 which is less than 0.05, null hypothesis is rejected and this is not a random walk.
# ##### Now google has p-value 0.0000006510 which is more than 0.05, null hypothesis is rejected and this is not a  random walk.

# ### Generating a random walk

# In[ ]:


seed(42)
rcParams['figure.figsize'] = 16, 6
random_walk = normal(loc=0, scale=0.01, size=1000)
plt.plot(random_walk)
plt.show()

# In[ ]:


fig = ff.create_distplot([random_walk],['Random Walk'],bin_size=0.001)
iplot(fig, filename='Basic Distplot')

# ## <a id='3.4'>3.4 Stationarity</a>
# A stationary time series is one whose statistical properties such as mean, variance, autocorrelation, etc. are all constant over time.
# * Strong stationarity:  is a stochastic process whose unconditional joint probability distribution does not change when shifted in time. Consequently, parameters such as mean and variance also do not change over time.
# * Weak stationarity: is a process where mean, variance, autocorrelation are constant throughout the time
# 
# Stationarity is important as  non-stationary series that depend on time have too many parameters to account for when modelling the time series. diff() method can easily convert a non-stationary series to a stationary series.
# 
# We will try to decompose seasonal component of the above decomposed time series.

# In[ ]:


# The original non-stationary plot
decomposed_google_volume.trend.plot()

# In[ ]:


# The new stationary plot
decomposed_google_volume.trend.diff().plot()

# # <a id='4'>4. Modelling using statstools</a>

# ## <a id='4.1'>4.1 AR models</a>
#  An autoregressive (AR) model is a representation of a type of random process; as such, it is used to describe certain time-varying processes in nature, economics, etc. The autoregressive model specifies that the output variable depends linearly on its own previous values and on a stochastic term (an imperfectly predictable term); thus the model is in the form of a stochastic difference equation.
#  #### AR(1) model
#  R<sub>t</sub> = μ + ϕR<sub>t-1</sub> + ε<sub>t</sub>
#  ##### As RHS has only one lagged value(R<sub>t-1</sub>)this is called AR model of order 1 where μ is mean and ε is noise at time t
#  If ϕ = 1, it is random walk. Else if ϕ = 0, it is white noise. Else if -1 < ϕ < 1, it is stationary. If ϕ is -ve, there is men reversion. If ϕ is +ve, there is momentum.
#  #### AR(2) model
#  R<sub>t</sub> = μ + ϕ<sub>1</sub>R<sub>t-1</sub> + ϕ<sub>2</sub>R<sub>t-2</sub> + ε<sub>t</sub>
#  #### AR(3) model
#  R<sub>t</sub> = μ + ϕ<sub>1</sub>R<sub>t-1</sub> + ϕ<sub>2</sub>R<sub>t-2</sub> + ϕ<sub>3</sub>R<sub>t-3</sub> + ε<sub>t</sub>

# ## Simulating AR(1) model

# In[ ]:


# AR(1) MA(1) model:AR parameter = +0.9
rcParams['figure.figsize'] = 16, 12
plt.subplot(4,1,1)
ar1 = np.array([1, -0.9]) # We choose -0.9 as AR parameter is +0.9
ma1 = np.array([1])
AR1 = ArmaProcess(ar1, ma1)
sim1 = AR1.generate_sample(nsample=1000)
plt.title('AR(1) model: AR parameter = +0.9')
plt.plot(sim1)
# We will take care of MA model later
# AR(1) MA(1) AR parameter = -0.9
plt.subplot(4,1,2)
ar2 = np.array([1, 0.9]) # We choose +0.9 as AR parameter is -0.9
ma2 = np.array([1])
AR2 = ArmaProcess(ar2, ma2)
sim2 = AR2.generate_sample(nsample=1000)
plt.title('AR(1) model: AR parameter = -0.9')
plt.plot(sim2)
# AR(2) MA(1) AR parameter = 0.9
plt.subplot(4,1,3)
ar3 = np.array([2, -0.9]) # We choose -0.9 as AR parameter is +0.9
ma3 = np.array([1])
AR3 = ArmaProcess(ar3, ma3)
sim3 = AR3.generate_sample(nsample=1000)
plt.title('AR(2) model: AR parameter = +0.9')
plt.plot(sim3)
# AR(2) MA(1) AR parameter = -0.9
plt.subplot(4,1,4)
ar4 = np.array([2, 0.9]) # We choose +0.9 as AR parameter is -0.9
ma4 = np.array([1])
AR4 = ArmaProcess(ar4, ma4)
sim4 = AR4.generate_sample(nsample=1000)
plt.title('AR(2) model: AR parameter = -0.9')
plt.plot(sim4)
plt.show()

# ## Forecasting a simulated model

# In[ ]:


model = ARMA(sim1, order=(1,0))
result = model.fit()
print(result.summary())
print("μ={} ,ϕ={}".format(result.params[0],result.params[1]))


# ϕ is around 0.9 which is what we chose as AR parameter in our first simulated model.

# ## Predicting the models

# In[ ]:


# Predicting simulated AR(1) model 
result.plot_predict(start=900, end=1010)
plt.show()

# In[ ]:


rmse = math.sqrt(mean_squared_error(sim1[900:1011], result.predict(start=900,end=999)))
print("The root mean squared error is {}.".format(rmse))

# y is predicted plot. Quite neat!

# In[ ]:


# Predicting humidity level of Montreal
humid = ARMA(humidity["Montreal"].diff().iloc[1:].values, order=(1,0))
res = humid.fit()
res.plot_predict(start=1000, end=1100)
plt.show()

# In[ ]:


rmse = math.sqrt(mean_squared_error(humidity["Montreal"].diff().iloc[900:1000].values, result.predict(start=900,end=999)))
print("The root mean squared error is {}.".format(rmse))

#  Not quite impressive. But let's try google stocks. 

# In[ ]:


# Predicting closing prices of google
humid = ARMA(google["Close"].diff().iloc[1:].values, order=(1,0))
res = humid.fit()
res.plot_predict(start=900, end=1010)
plt.show()

# There are always better models.

# ## <a id='4.2'>4.2 MA models</a>
# The moving-average (MA) model is a common approach for modeling univariate time series. The moving-average model specifies that the output variable depends linearly on the current and various past values of a stochastic (imperfectly predictable) term.
# #### MA(1) model
# R<sub>t</sub> = μ  +  ϵ<sub>t</sub>1 + θϵ<sub>t-1</sub>
# 
# It translates to Today's returns = mean + today's noise + yesterday's noise
# ##### As there is only 1 lagged value in RHS, it is an MA model of order 1

# ## Simulating MA(1) model

# In[ ]:


rcParams['figure.figsize'] = 16, 6
ar1 = np.array([1])
ma1 = np.array([1, -0.5])
MA1 = ArmaProcess(ar1, ma1)
sim1 = MA1.generate_sample(nsample=1000)
plt.plot(sim1)

# ## Forecasting the simulated MA model

# In[ ]:


model = ARMA(sim1, order=(0,1))
result = model.fit()
print(result.summary())
print("μ={} ,θ={}".format(result.params[0],result.params[1]))

# ## Prediction using MA models

# In[ ]:


# Forecasting and predicting montreal humidity
model = ARMA(humidity["Montreal"].diff().iloc[1:].values, order=(0,3))
result = model.fit()
print(result.summary())
print("μ={} ,θ={}".format(result.params[0],result.params[1]))
result.plot_predict(start=1000, end=1100)
plt.show()

# In[ ]:


rmse = math.sqrt(mean_squared_error(humidity["Montreal"].diff().iloc[1000:1101].values, result.predict(start=1000,end=1100)))
print("The root mean squared error is {}.".format(rmse))

# Now, for ARMA models.
# 
# ## <a id='4.3'>4.3 ARMA models</a>
# Autoregressive–moving-average (ARMA) models provide a parsimonious description of a (weakly) stationary stochastic process in terms of two polynomials, one for the autoregression and the second for the moving average. It's the fusion of AR and MA models.
# ### ARMA(1,1) model
# ### R<sub>t</sub> = μ + ϕR<sub>t-1</sub> + ϵ<sub>t</sub> + θϵ<sub>t-1</sub> 
# Basically, Today's return = mean + Yesterday's return + noise + yesterday's noise.

# ## Prediction using ARMA models
# I am not simulating any model because it's quite similar to AR and MA models. Just  forecasting and predictions for this one.

# In[ ]:


# Forecasting and predicting microsoft stocks volume
model = ARMA(microsoft["Volume"].diff().iloc[1:].values, order=(3,3))
result = model.fit()
print(result.summary())
print("μ={}, ϕ={}, θ={}".format(result.params[0],result.params[1],result.params[2]))
result.plot_predict(start=1000, end=1100)
plt.show()

# In[ ]:


rmse = math.sqrt(mean_squared_error(microsoft["Volume"].diff().iloc[1000:1101].values, result.predict(start=1000,end=1100)))
print("The root mean squared error is {}.".format(rmse))

# ARMA model shows much better results than AR and MA models.

# ## <a id='4.4'>4.4 ARIMA models</a>
# An autoregressive integrated moving average (ARIMA) model is a generalization of an autoregressive moving average (ARMA) model. Both of these models are fitted to time series data either to better understand the data or to predict future points in the series (forecasting). ARIMA models are applied in some cases where data show evidence of non-stationarity, where an initial differencing step (corresponding to the "integrated" part of the model) can be applied one or more times to eliminate the non-stationarity.
# ARIMA model is of the form: ARIMA(p,d,q): p is AR parameter, d is differential parameter, q is MA parameter
# ### ARIMA(1,0,0)
# **y<sub>t</sub> = a<sub>1</sub>y<sub>t-1</sub> + ϵ<sub>t</sub>**
# ### ARIMA(1,0,1)
# **y<sub>t</sub> = a<sub>1</sub>y<sub>t-1</sub> + ϵ<sub>t</sub> + b<sub>1</sub>ϵ<sub>t-1</sub>**
# ### ARIMA(1,1,1)
# **Δy<sub>t</sub> = a<sub>1</sub>Δy<sub>t-1</sub> + ϵ<sub>t</sub> + b<sub>1</sub>ϵ<sub>t-1</sub>** where **Δy<sub>t</sub> = y<sub>t</sub> - y<sub>t-1</sub>**

# ## Prediction using ARIMA model

# In[ ]:


# Predicting the microsoft stocks volume
rcParams['figure.figsize'] = 16, 6
model = ARIMA(microsoft["Volume"].diff().iloc[1:].values, order=(2,1,0))
result = model.fit()
print(result.summary())
result.plot_predict(start=700, end=1000)
plt.show()

# In[ ]:


rmse = math.sqrt(mean_squared_error(microsoft["Volume"].diff().iloc[700:1001].values, result.predict(start=700,end=1000)))
print("The root mean squared error is {}.".format(rmse))

# Taking the slight lag into account, this is a fine model.

# ## <a id='4.5'>4.5 VAR models</a>
# Vector autoregression (VAR) is a stochastic process model used to capture the linear interdependencies among multiple time series. VAR models generalize the univariate autoregressive model (AR model) by allowing for more than one evolving variable. All variables in a VAR enter the model in the same way: each variable has an equation explaining its evolution based on its own lagged values, the lagged values of the other model variables, and an error term. VAR modeling does not require as much knowledge about the forces influencing a variable as do structural models with simultaneous equations: The only prior knowledge required is a list of variables which can be hypothesized to affect each other intertemporally.
# 
# <img src="http://gdurl.com/VvRm">

# In[ ]:


# Predicting closing price of Google and microsoft
train_sample = pd.concat([google["Close"].diff().iloc[1:],microsoft["Close"].diff().iloc[1:]],axis=1)
model = sm.tsa.VARMAX(train_sample,order=(2,1),trend='c')
result = model.fit(maxiter=1000,disp=False)
print(result.summary())
predicted_result = result.predict(start=0, end=1000)
result.plot_diagnostics()
# calculating error
rmse = math.sqrt(mean_squared_error(train_sample.iloc[1:1002].values, predicted_result.values))
print("The root mean squared error is {}.".format(rmse))

# ## <a id='4.6'>4.6 State Space methods</a>
# 
# A general state space model is of the form
# 
# y<sub>t</sub>=Z<sub>t</sub>α<sub>t</sub>+d<sub>t</sub>+ε<sub>t</sub>
# 
# α<sub>t</sub>=T<sub>t</sub>α<sub>t</sub>−1+c<sub>t</sub>+R<sub>t</sub>η<sub>t</sub>
# 
# where y<sub>t</sub> refers to the observation vector at time t, α<sub>t</sub> refers to the (unobserved) state vector at time t, and where the irregular components are defined as
# 
# ε<sub>t</sub>∼N(0,H<sub>t</sub>)
# 
# η<sub>t</sub>∼N(0,Q<sub>t</sub>)
# 
# The remaining variables (Z<sub>t</sub>,d<sub>t</sub>,H<sub>t</sub>,T<sub>t</sub>,c<sub>t</sub>,R<sub>t</sub>,Q<sub>t</sub>) in the equations are matrices describing the process. Their variable names and dimensions are as follows
# 
# Z : design (k_endog×k_states×nobs)
# 
# d : obs_intercept (k_endog×nobs)
# 
# H : obs_cov (k_endog×k_endog×nobs)
# 
# T : transition (k_states×k_states×nobs)
# 
# c : state_intercept (k_states×nobs)
# 
# R : selection (k_states×k_posdef×nobs)
# 
# Q : state_cov (k_posdef×k_posdef×nobs)
# 
# In the case that one of the matrices is time-invariant (so that, for example, Z<sub>t</sub>=Z<sub>t</sub>+1 ∀ <sub>t</sub>), its last dimension may be of size 1 rather than size nobs.
# 
# This generic form encapsulates many of the most popular linear time series models (see below) and is very flexible, allowing estimation with missing observations, forecasting, impulse response functions, and much more.
# 
# Source: [statsmodels](https://www.statsmodels.org/dev/statespace.html)

# ## <a id='4.6.1'>4.6.1 SARIMA models</a>
# SARIMA models are useful for modeling seasonal time series, in which the mean and other statistics for a given season are not stationary across the years. The SARIMA model defined constitutes a straightforward extension of the nonseasonal autoregressive-moving average (ARMA) and autoregressive integrated moving average (ARIMA) models presented

# In[ ]:


# Predicting closing price of Google'
train_sample = google["Close"].diff().iloc[1:].values
model = sm.tsa.SARIMAX(train_sample,order=(4,0,4),trend='c')
result = model.fit(maxiter=1000,disp=False)
print(result.summary())
predicted_result = result.predict(start=0, end=500)
result.plot_diagnostics()
# calculating error
rmse = math.sqrt(mean_squared_error(train_sample[1:502], predicted_result))
print("The root mean squared error is {}.".format(rmse))

# In[ ]:


plt.plot(train_sample[1:502],color='red')
plt.plot(predicted_result,color='blue')
plt.legend(['Actual','Predicted'])
plt.title('Google Closing prices')
plt.show()

# ## <a id='4.6.2'>4.6.2 Unobserved components</a>
# A UCM decomposes the response series into components such as trend, seasons, cycles, and the regression effects due to predictor series. The following model shows a possible scenario:
# <img src="http://support.sas.com/documentation/cdl/en/etsug/66840/HTML/default/images/etsug_ucm0134.png">
# Source: [http://support.sas.com/documentation/cdl/en/etsug/66840/HTML/default/viewer.htm#etsug_ucm_details01.htm](http://support.sas.com/documentation/cdl/en/etsug/66840/HTML/default/viewer.htm#etsug_ucm_details01.htm)

# In[ ]:


# Predicting closing price of Google'
train_sample = google["Close"].diff().iloc[1:].values
model = sm.tsa.UnobservedComponents(train_sample,'local level')
result = model.fit(maxiter=1000,disp=False)
print(result.summary())
predicted_result = result.predict(start=0, end=500)
result.plot_diagnostics()
# calculating error
rmse = math.sqrt(mean_squared_error(train_sample[1:502], predicted_result))
print("The root mean squared error is {}.".format(rmse))

# In[ ]:


plt.plot(train_sample[1:502],color='red')
plt.plot(predicted_result,color='blue')
plt.legend(['Actual','Predicted'])
plt.title('Google Closing prices')
plt.show()

# ## <a id='4.6.3'>4.6.3 Dynamic Factor models</a>
# Dynamic-factor models are flexible models for multivariate time series in which the observed endogenous variables are linear functions of exogenous covariates and unobserved factors, which have a vector autoregressive structure. The unobserved factors may also be a function of exogenous covariates. The disturbances in the equations for the dependent variables may be autocorrelated.

# In[ ]:


# Predicting closing price of Google and microsoft
train_sample = pd.concat([google["Close"].diff().iloc[1:],microsoft["Close"].diff().iloc[1:]],axis=1)
model = sm.tsa.DynamicFactor(train_sample, k_factors=1, factor_order=2)
result = model.fit(maxiter=1000,disp=False)
print(result.summary())
predicted_result = result.predict(start=0, end=1000)
result.plot_diagnostics()
# calculating error
rmse = math.sqrt(mean_squared_error(train_sample.iloc[1:1002].values, predicted_result.values))
print("The root mean squared error is {}.".format(rmse))

# I may add more regression models soon and there is much more stuff to cover. **But in my experience, the best models for time series forecasting are LSTM based Recurrent Neural Networks. I have prepared a detailed tutorial for that. Here is the link: <u>https://www.kaggle.com/thebrownviking20/intro-to-recurrent-neural-networks-lstm-gru</u>**

# References and influences(These have more in-depth content and explanations): 
# * [Manipulating Time Series Data in Python](https://www.datacamp.com/courses/manipulating-time-series-data-in-python)
# * [Introduction to Time Series Analysis in Python](https://www.datacamp.com/courses/introduction-to-time-series-analysis-in-python)
# * [Visualizing Time Series Data in Python](https://www.datacamp.com/courses/visualizing-time-series-data-in-python)
# * [VAR models and LSTM](https://www.youtube.com/watch?v=_vQ0W_qXMxk)
# * [State space models](https://www.statsmodels.org/dev/statespace.html)
# 
# Stay tuned for more! And don't forget to upvote and comment.
