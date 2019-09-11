#!/usr/bin/env python
# coding: utf-8

# # Cleaning up market data
# ---

# **This is a list of errors and issues found in the market data.  Will update if more are found.  Detailed walkthrough (in progress) is below.**
# 1. `open`, `close`, and `volume` errors on 2016-07-06 for BBBY.O, DISH.O, FLEX.O, MAT.O, NDAQ.O, PCAR.O, PZZA.O, SHLD.O, & ZNGA.O
# 2. Towers Watson (TW.N) `open` price error & incorrect assetCode (WW.N)
# 3. Bad `open` prices of 0.01 and 999.99 & filtering out data from 2007 & 2008
# 4. PGN.N errors in `raw` return data on 2011-12-23 & 2012-02-15 causing broader issues from Oct '11 to Mar '12
# 5. QRVO.O bad `raw` and `mktres` return data in Jan and Feb 2015
# 6. TECD.O bad `open` price on 2015-03-16
# 7. EBR.N bad `raw` and `mktres` return data in Oct 2016
# 8. HGSI.O bad `mktres` return data in 2009
# 9. MNST.O bad `raw` and `mktres` return data on and around 2015-06-15
# 10. ASH.N bad `raw` and `mktres` return data on and around 2016-09-20
# 11. PGH.N bad `raw` and `mktres` return data on and around 2011-01-03
# 12. ERF.N bad `raw` and `mktres` return data on and around 2011-01-03
# 13. PWE.N bad `raw` and `mktres` return data on and around 2011-01-03
# 14. AMT.N bad `raw` and `mktres` return data on and around 2012-01-03
# 15. TSU.N bad `raw` and `mktres` return data on and around 2011-08-08
# 16. GRA.N bad `raw` and `mktres` return data on and around 2016-02-04
# 17. HSNI.O bad `raw` and `mktres` return data on and around 2015-02-05 & 2015-06-01
# 18. CYH.N bad `raw` and `mktres` return data on and around 2016-05-02
# 19. AWI.N bad `raw` and `mktres` return data on and around 2016-04-04
# 20. AES.N bad `raw` and `mktres` return data on and around 2009-04-09
# 21. Dealing with Stock Splits
# ---

# In[ ]:


import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import cufflinks as cf
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.graph_objs as go
init_notebook_mode(connected=True)
cf.go_offline()

# In[ ]:


from kaggle.competitions import twosigmanews
env = twosigmanews.make_env()

# In[ ]:


(market_train_df, news_train_df) = env.get_training_data()
market_train_df['date'] = market_train_df['time'].dt.strftime('%Y-%m-%d')

# Starting out by looking at summary statistics for each of the numeric columns.  There are certainly some outliers and extreme values to check out.

# In[ ]:


market_train_df.describe().round(3)

# For example, the `open` price column has a maximum value of nearly 10,000 while the highest `close` price is only about 1578, so that's a clear red flag.  Looking at the `returnsClosePrevRaw1` column, there are assets that supposedly increased by around 4600% or lost nearly 100% their value in a single day.  Even more extreme day-over-day returns are seen in the `returnsOpenPrevRaw1` column.

# First I'm taking a look at assets that are showing very large drop in day-over-day close prices.  Below are all the observations showing `close` prices dropping by more than 70% day-over-day...

# In[ ]:


market_train_df[market_train_df['returnsClosePrevRaw1'] < -.7]

# There are 12 records showing a > 70% drop in daily `close` price.  The first record above from 2008 shows Bear Sterns stock dropping by 84%.  This jumps out as probably being accurate.  If you recall, Bear Stearns was an investment bank that failed due to losses largely associated with subprime mortgage backed securities.  Their failure occurred shortly before the broader global financial crisis during 2008 and 2009.

# ### Errors on 2016-07-06
# ---
# After a brief review of the data surrounding the 12 records above, it appears only four of these high negative `returnsClosePrevRaw1` values are due to data errors.  Each of the errors is for a record on 2016-07-06.  The assets `FLEX.O`, `MAT.O`, `SHLD.O`, and `ZNGA.O` are affected.  Below is a closer look at the errors...

# In[ ]:


someAssetsWithBadData = ['FLEX.O','MAT.O','SHLD.O','ZNGA.O']
someMarketData = market_train_df[(market_train_df['assetCode'].isin(someAssetsWithBadData)) 
                & (market_train_df['time'] >= '2016-07-05')
                & (market_train_df['time'] < '2016-07-08')].sort_values('assetCode')
someMarketData

# It appears that each of the columns `volume`, `close`, and `open` are inaccurate for the four records above on 2016-07-06.  I looked up what the correct historical volume and open/close prices were to confirm the errors.
# 
# Below is a graph showing the `close` price for each of these four stocks over the last couple years in the `market_train_df` dataset.  It's a little tough to see because the graphs overlap each other, but if you hover your cursor over the spike, you can see all four asset's `close` price shoots up to around 123.5 on 2016-07-06 (and then returns back to normal).

# In[ ]:


selectedAssets = market_train_df[(market_train_df['assetCode'].isin(someAssetsWithBadData))
                                 & (market_train_df['time'] >= '2015-01-01')]
selectedAssetsPivot = selectedAssets.pivot(index='date',columns='assetCode',values='close')

flex = go.Scatter(x = selectedAssetsPivot.index,y = selectedAssetsPivot['FLEX.O'],mode = 'lines',name = 'FLEX.O')
mat = go.Scatter(x = selectedAssetsPivot.index,y = selectedAssetsPivot['MAT.O'],mode = 'lines',name = 'MAT.O')
shld = go.Scatter(x = selectedAssetsPivot.index,y = selectedAssetsPivot['SHLD.O'],mode = 'lines',name = 'SHLD.O')
znga = go.Scatter(x = selectedAssetsPivot.index,y = selectedAssetsPivot['ZNGA.O'],mode = 'lines',name = 'ZNGA.O')
data = [flex,mat,shld,znga]
layout = go.Layout(
    title = 'assets with error on 2016-07-06',
    yaxis = dict(title='close price')
)
fig = go.Figure(data=data, layout=layout)
iplot(fig)

# Given each of these stocks have a similar incorrect `close` price of 123.45 or 123.47, I decided to look and see if there were any more stocks with similar issues on 2016-07-06.  Searching for those specific `close` prices on that date revealed five more stocks with similar errors... `BBBY.O`, `DISH.O`, `NDAQ.O`, `PCAR.O`, & `PZZA.O`.

# In[ ]:


market_train_df[(market_train_df['date'] == '2016-07-06')
                & ((market_train_df['close'] == 123.45) 
                   | (market_train_df['close'] == 123.47))]

# Because the `close` and `open` prices are wrong, this has also affected many of the return columns for these stocks around the same time frame.  Both `raw` and `mktres` columns have been affected.
# 
# Some of the `mktres` return data that was skewed because of the innacurate close/open prices I wouldn't have expected.  For example, the `returnsClosePrevMktres1` column shows odd values for up to 20 days after the bad data row on 2016-07-06.  Looking at this return metric for Zynga, it has values of 249% on 2016-08-02, and -124% on 2016-08-03 which are obviously incorrect (see graph below).

# In[ ]:


sampleZynga = market_train_df[(market_train_df['assetCode'] == 'ZNGA.O')
                              & (market_train_df['time'] >= '2016-06-01')
                              & (market_train_df['time'] < '2016-09-01')]

sampleZyngaReturn = go.Scatter(
    x = sampleZynga['date'],
    y = sampleZynga['returnsClosePrevMktres1'],
    mode = 'lines',
    name = 'ZNGA.O'
)
data = [sampleZyngaReturn]
layout = go.Layout(
    title = 'ZNGA.O returnsClosePrevMktres1 metric<br>(errors in return data linger for 20 days after 2016-07-06 price error)',
    yaxis = dict(
        title='return (%)',
        tickformat = ',.0%',
    )
)
fig = go.Figure(data=data, layout=layout)
iplot(fig)

# After reviewing the data around these nine errors, each instance appeared to have the same pattern of bad data and affected rows and columns.  In each of the nine errors, I found that 32 rows of return data had at least one column that needed to be fixed.  So I created a few functions to help fix the bad data for these errors.
# 
# It isn't entirely clear to me how to properly 'fix' the `mktres` data, since this metric is calculated behind the scenes based on the overall market and an individual stock's performance.  While I haven't had much of a chance to review it, this kernel looks into what the `mktres` return actual is and suggests it is based on a stock's beta coefficient... https://www.kaggle.com/marketneutral/eda-what-does-mktres-mean.
# 
# For now, I've opted to look at a individual stock's `raw` vs. `mktres` values for a particular return metric across a sample of days, and use a linear best fit line to approximate the `mktres` value I want to update.  Certainly far from ideal, but it's a relatively easy fix and should be good enough since the 'bad' `mktres` data I'm fixing isn't a huge part of the market training dataset anyways.  Just to show an example, here's a scatter plot showing the roughly linear relationship between `returnsClosePrevRaw1` and `returnsClosePrevMktres1` for a sample of Zynga data unaffected by the data error...

# In[ ]:


quickZyngaSample = market_train_df[(market_train_df['assetCode'] == 'ZNGA.O')
                                   & ((market_train_df['time'] >= '2016-06-01') & (market_train_df['time'] < '2016-07-06')
                                      | (market_train_df['time'] >= '2016-08-04') & (market_train_df['time'] < '2016-09-01'))]

quickZyngaSample['ones'] = 1
A = quickZyngaSample[['returnsClosePrevRaw1','ones']]
y = quickZyngaSample['returnsClosePrevMktres1']
m, c = np.linalg.lstsq(A,y,rcond=-1)[0]
zyngaFitLine = m*np.array(quickZyngaSample['returnsClosePrevRaw1'])+c

quickZyngaSampleReturnsComparison = go.Scatter(
    x = quickZyngaSample['returnsClosePrevRaw1'],
    y = quickZyngaSample['returnsClosePrevMktres1'],
    mode = 'markers',
    name = 'ZNGA.O raw vs. mktres return'
)
quickZyngaSampleReturnsComparisonBestFitLine = go.Scatter(
    x = quickZyngaSample['returnsClosePrevRaw1'],
    y = zyngaFitLine,
    mode = 'lines',
    name = 'best fit line'
)
data = [quickZyngaSampleReturnsComparison,quickZyngaSampleReturnsComparisonBestFitLine]
layout = go.Layout(
    title = 'ZNGA.O sample of returnsClosePrevRaw1 vs. returnsClosePrevMktres1',
    xaxis = dict(
        title='returnsClosePrevRaw1 (%)',
        tickformat = ',.0%',
    ),
    yaxis = dict(
        title='returnsClosePrevMktres1 (%)',
        tickformat = ',.0%',
    )
)
fig = go.Figure(data=data, layout=layout)
iplot(fig)

# Finally, after the affected `PrevMktres` values were estimated, the bad `returnsOpenNextMktres10` values were updated based on the newly estimated `returnsOpenPrevMktres10` values.

# ### Fixing the 2016-07-06 Errors
# ---
# The helper functions and the fixes for the nine 2016-07-06 errors are below (helper functions have been hidden, but you can click on the "Code" buttons to view).

# In[ ]:


def sampleAssetData(assetCode, date, numDays):
    d = datetime.strptime(date,'%Y-%m-%d')
    start = d - timedelta(days=numDays)
    end = d + timedelta(days=numDays)
    return market_train_df[(market_train_df['assetCode'] == assetCode)
                             & (market_train_df['time'] >= start.strftime('%Y-%m-%d'))
                             & (market_train_df['time'] <= end.strftime('%Y-%m-%d'))].copy()

# In[ ]:


def updateRawReturns(assetData, indices):
    rowsToUpdate1 = assetData[(assetData.index >= indices[0][0]) & (assetData.index <= indices[0][1])]
    for index, row in rowsToUpdate1.iterrows():
        market_train_df.loc[[index],['returnsClosePrevRaw1']] = assetData['close'].pct_change()
        market_train_df.loc[[index],['returnsOpenPrevRaw1']] = assetData['open'].pct_change()
    rowsToUpdate2 = assetData[(assetData.index >= indices[1][0]) & (assetData.index <= indices[1][1])]
    for index, row in rowsToUpdate2.iterrows():
        market_train_df.loc[[index],['returnsClosePrevRaw10']] = assetData['close'].pct_change(periods=10)
        market_train_df.loc[[index],['returnsOpenPrevRaw10']] = assetData['open'].pct_change(periods=10)

# In[ ]:


def estimateMktresReturn(sampleData, mktresCol, index):
    sampleData['ones'] = 1
    sampleData.dropna(inplace=True)
    rawCol = mktresCol.replace('Mktres','Raw')
    A = sampleData[[rawCol,'ones']]
    y = sampleData[mktresCol]
    m, c = np.linalg.lstsq(A,y,rcond=-1)[0]
    return c + m * market_train_df.loc[index,rawCol]

# In[ ]:


def updateMktresReturns(assetCode, assetData, indices):
    # update range of values for returnsClosePrevMktres1 & returnsOpenPrevMktres1
    sample1 = assetData[(assetData.index < indices[2][0]) | (assetData.index > indices[2][1])]
    rowsToUpdate1 = assetData[(assetData.index >= indices[2][0]) & (assetData.index <= indices[2][1])]
    for index, row in rowsToUpdate1.iterrows():
        market_train_df.loc[[index],['returnsClosePrevMktres1']] = estimateMktresReturn(sample1,'returnsClosePrevMktres1',index)
        market_train_df.loc[[index],['returnsOpenPrevMktres1']] = estimateMktresReturn(sample1,'returnsOpenPrevMktres1',index)
    # update range of values for returnsClosePrevMktres10 & returnsOpenPrevMktres10
    sample2 = assetData[(assetData.index < indices[3][0]) | (assetData.index > indices[3][1])]
    rowsToUpdate2 = assetData[(assetData.index >= indices[3][0]) & (assetData.index <= indices[3][1])]
    l = []
    for index, row in rowsToUpdate2.iterrows():
        market_train_df.loc[[index],['returnsClosePrevMktres10']] = estimateMktresReturn(sample2,'returnsClosePrevMktres10',index)
        est = estimateMktresReturn(sample2,'returnsOpenPrevMktres10',index)
        l.append(est)
        market_train_df.loc[[index],['returnsOpenPrevMktres10']] = est
    # update range of values for returnsOpenNextMktres10
    rowsToUpdate3 = assetData[(assetData.index >= indices[4][0]) & (assetData.index <= indices[4][1])]
    i = 0
    for index, row in rowsToUpdate3.iterrows():
        market_train_df.loc[[index],['returnsOpenNextMktres10']] = l[i]
        i += 1

# In[ ]:


def fixBadReturnData(assetCode, badDate, badIndex, badReturnDataRanges, dayWindow):
    # store copy of bad data window
    badDataWindow = sampleAssetData(assetCode,badDate,dayWindow)
    badDataWindow.reset_index(inplace=True)
    # store indices needed to update raw and mktres return data
    newIdx = badDataWindow[badDataWindow['index'] == badIndex].index[0]
    indices = [
        # range of bad data for... returnsClosePrevRaw1 & returnsOpenPrevRaw1
        [badIndex,badDataWindow.loc[newIdx+badReturnDataRanges[0],'index']],
        # returnsClosePrevRaw10 & returnsOpenPrevRaw10
        [badIndex,badDataWindow.loc[newIdx+badReturnDataRanges[1],'index']],
        # returnsClosePrevMktres1 & returnsOpenPrevMktres1
        [badIndex,badDataWindow.loc[newIdx+badReturnDataRanges[2],'index']],
        # returnsClosePrevMktres10 & returnsOpenPrevMktres10
        [badIndex,badDataWindow.loc[newIdx+badReturnDataRanges[3],'index']],
        # returnsOpenNextMktres10
        [badDataWindow.loc[newIdx+badReturnDataRanges[4],'index'],badDataWindow.loc[newIdx+badReturnDataRanges[5],'index']]
    ]
    badDataWindow.set_index('index',inplace=True)
    # correct bad raw return data
    updateRawReturns(badDataWindow,indices)
    # estimate affected mktres return data
    updateMktresReturns(assetCode,badDataWindow,indices)

# In[ ]:


# bad volume, open, and close for ZNGA.O on 2016-07-06
assetCode = 'ZNGA.O'
badDate = '2016-07-06'
badIndex = market_train_df[(market_train_df['assetCode'] == assetCode) & (market_train_df['date'] == badDate)].index[0]
# correct bad data
market_train_df.loc[[badIndex],['volume']] = 19213100
market_train_df.loc[[badIndex],['open']] = 2.64
market_train_df.loc[[badIndex],['close']] = 2.75
# ranges of affected return data
#   integers specify how many rows of data need to be updated for different return columns, in reference to "badDate" row
badReturnDataRanges = [1,10,20,20,-11,9]
# sample data window (on either side of "badDate")
sampleWindow = 45
# fix bad return data in market_train_df
fixBadReturnData(assetCode,badDate,badIndex,badReturnDataRanges,sampleWindow)

# In[ ]:


# bad volume, open, and close for FLEX.O on 2016-07-06
assetCode = 'FLEX.O'
badDate = '2016-07-06'
badIndex = market_train_df[(market_train_df['assetCode'] == assetCode) & (market_train_df['date'] == badDate)].index[0]
# correct bad data
market_train_df.loc[[badIndex],['volume']] = 5406600
market_train_df.loc[[badIndex],['open']] = 11.580
market_train_df.loc[[badIndex],['close']] = 11.750
# ranges of affected return data
badReturnDataRanges = [1,10,20,20,-11,9]
# sample data window (on either side of "badDate")
sampleWindow = 45
# fix bad return data in market_train_df
fixBadReturnData(assetCode,badDate,badIndex,badReturnDataRanges,sampleWindow)

# In[ ]:


# bad volume, open, and close for SHLD.O on 2016-07-06
assetCode = 'SHLD.O'
badDate = '2016-07-06'
badIndex = market_train_df[(market_train_df['assetCode'] == assetCode) & (market_train_df['date'] == badDate)].index[0]
# correct bad data
market_train_df.loc[[badIndex],['volume']] = 279300
market_train_df.loc[[badIndex],['open']] = 12.8900
market_train_df.loc[[badIndex],['close']] = 13.1400
# ranges of affected return data
badReturnDataRanges = [1,10,20,20,-11,9]
# sample data window (on either side of "badDate")
sampleWindow = 45
# fix bad return data in market_train_df
fixBadReturnData(assetCode,badDate,badIndex,badReturnDataRanges,sampleWindow)

# In[ ]:


# bad volume, open, and close for MAT.O on 2016-07-06
assetCode = 'MAT.O'
badDate = '2016-07-06'
badIndex = market_train_df[(market_train_df['assetCode'] == assetCode) & (market_train_df['date'] == badDate)].index[0]
# correct bad data
market_train_df.loc[[badIndex],['volume']] = 3242100
market_train_df.loc[[badIndex],['open']] = 32.13
market_train_df.loc[[badIndex],['close']] = 31.52
# ranges of affected return data
badReturnDataRanges = [1,10,20,20,-11,9]
# sample data window (on either side of "badDate")
sampleWindow = 45
# fix bad return data in market_train_df
fixBadReturnData(assetCode,badDate,badIndex,badReturnDataRanges,sampleWindow)

# In[ ]:


# bad volume, open, and close for BBBY.O on 2016-07-06
assetCode = 'BBBY.O'
badDate = '2016-07-06'
badIndex = market_train_df[(market_train_df['assetCode'] == assetCode) & (market_train_df['date'] == badDate)].index[0]
# correct bad data
market_train_df.loc[[badIndex],['volume']] = 4205500
market_train_df.loc[[badIndex],['open']] = 42.23
market_train_df.loc[[badIndex],['close']] = 43.55
# ranges of affected return data
badReturnDataRanges = [1,10,20,20,-11,9]
# sample data window (on either side of "badDate")
sampleWindow = 45
# fix bad return data in market_train_df
fixBadReturnData(assetCode,badDate,badIndex,badReturnDataRanges,sampleWindow)

# In[ ]:


# bad volume, open, and close for DISH.O on 2016-07-06
assetCode = 'DISH.O'
badDate = '2016-07-06'
badIndex = market_train_df[(market_train_df['assetCode'] == assetCode) & (market_train_df['date'] == badDate)].index[0]
# correct bad data
market_train_df.loc[[badIndex],['volume']] = 2303300
market_train_df.loc[[badIndex],['open']] = 50.06
market_train_df.loc[[badIndex],['close']] = 51.33
# ranges of affected return data
badReturnDataRanges = [1,10,20,20,-11,9]
# sample data window (on either side of "badDate")
sampleWindow = 45
# fix bad return data in market_train_df
fixBadReturnData(assetCode,badDate,badIndex,badReturnDataRanges,sampleWindow)

# In[ ]:


# bad volume, open, and close for NDAQ.O on 2016-07-06
assetCode = 'NDAQ.O'
badDate = '2016-07-06'
badIndex = market_train_df[(market_train_df['assetCode'] == assetCode) & (market_train_df['date'] == badDate)].index[0]
# correct bad data
market_train_df.loc[[badIndex],['volume']] = 733400
market_train_df.loc[[badIndex],['open']] = 64.64
market_train_df.loc[[badIndex],['close']] = 64.74
# ranges of affected return data
badReturnDataRanges = [1,10,20,20,-11,9]
# sample data window (on either side of "badDate")
sampleWindow = 45
# fix bad return data in market_train_df
fixBadReturnData(assetCode,badDate,badIndex,badReturnDataRanges,sampleWindow)

# In[ ]:


# bad volume, open, and close for PCAR.O on 2016-07-06
assetCode = 'PCAR.O'
badDate = '2016-07-06'
badIndex = market_train_df[(market_train_df['assetCode'] == assetCode) & (market_train_df['date'] == badDate)].index[0]
# correct bad data
market_train_df.loc[[badIndex],['volume']] = 2394300
market_train_df.loc[[badIndex],['open']] = 50.16
market_train_df.loc[[badIndex],['close']] = 50.79
# ranges of affected return data
badReturnDataRanges = [1,10,20,20,-11,9]
# sample data window (on either side of "badDate")
sampleWindow = 45
# fix bad return data in market_train_df
fixBadReturnData(assetCode,badDate,badIndex,badReturnDataRanges,sampleWindow)

# In[ ]:


# bad volume, open, and close for PZZA.O on 2016-07-06
assetCode = 'PZZA.O'
badDate = '2016-07-06'
badIndex = market_train_df[(market_train_df['assetCode'] == assetCode) & (market_train_df['date'] == badDate)].index[0]
# correct bad data
market_train_df.loc[[badIndex],['volume']] = 185100
market_train_df.loc[[badIndex],['open']] = 67.86
market_train_df.loc[[badIndex],['close']] = 67.91
# ranges of affected return data
badReturnDataRanges = [1,10,20,20,-11,9]
# sample data window (on either side of "badDate")
sampleWindow = 45
# fix bad return data in market_train_df
fixBadReturnData(assetCode,badDate,badIndex,badReturnDataRanges,sampleWindow)

# After running the code above (which fixed the errors, corrected the `raw` returns, and estimated the affected `mktres` return values), the data around these errors looks much more reasonable. Revisiting the summary statistics, we've removed some of the extreme values previously seen in the `returnsClosePrevRaw1` column.

# In[ ]:


market_train_df.describe().round(3)

# ### Towers Watson (TW.N) open price error & incorrect assetCode (WW.N)
# ---
# Now turning to the very high `open` price of 9998.99...

# In[ ]:


market_train_df[market_train_df['open'] > 2000]

# The record above for Towers Watson & Co appears to be an obvious error with an `open` price of 9998 and a `close` price of 50.  There's a similar error for Bank of New York Mellon Corp, however, it's likely I won't be using market data from 2007 and 2008 when modeling, so for now I'm going to ignore this error.
# 
# I didn't notice it initially, but there's another `assetCode` for Towers Watson, "WW.N" (in addition to the "TW.N" code).  While a single `assetName` can have multiple `assetCode`s, based on some quick searching online I don't think that is the case here.  Instead, it appears to be that the "WW.N" `assetCode` is an error.  So I went ahead and updated the "WW.N" tickers to "TW.N" to match the rest of the Towers Watson data.

# In[ ]:


market_train_df.loc[market_train_df['assetCode'] == 'WW.N','assetCode'] = 'TW.N'

# After reviewing the first few years of the updated Towers Watson data ("TW.N" + "WW.N"), we can see there are a few time periods where the available market data is pretty spotty.  See the histogram below for a monthly count of records for Towers Watson.

# In[ ]:


twSample = market_train_df[(market_train_df['assetCode'] == 'TW.N') & (market_train_df['time'] < '2012-01-01')]
twSample['month'] = twSample['time'].dt.strftime('%Y') + '-' + twSample['time'].dt.strftime('%m')
twSample['month'].iplot(kind='hist',bins=48,layout=dict(title='monthly TW.N (and WW.N) records in market training dataset'))

# The record with the outlier ~10k `open` price is from Jan 2010.  This price error has caused issues with the returns data back to 2009-12-16, and up to 2010-01-07.  Given the data in Dec 2009 and Jan 2010 is very isolated from the rest of the Towers Watson dataset (next closest data on either side is July 2009 and June 2010), I opted to remove all the affected records due to the price error.  So records from 2009-12-16 to 2010-01-07 were removed.

# In[ ]:


# dropping Towers Watson data affected by 2010-01-04 open price error
market_train_df = market_train_df[~((market_train_df['assetCode'] == 'TW.N')
                                  & (market_train_df['time'] >= '2009-12-16')
                                  & (market_train_df['time'] < '2010-01-08'))]

# ### Bad open prices of 0.01 and 999.99
# ---
# Returning to the summary statistics table, the `returnsOpenPrevRaw1` column has a ridiculously high maximum value; implying a 920,900% day-over-day increase.  This is undoubtedly an error.

# In[ ]:


market_train_df.describe().round(3)

# Looking at records with a greater than 1000% `returnsOpenPrevRaw1` metric...

# In[ ]:


market_train_df[market_train_df['returnsOpenPrevRaw1'] > 10]

# There are 22 records with a `returnsOpenPrevRaw1` value over 1000%.  All of these records are from 2007 or 2008, and we can see a large batch of them from 2007-03-23.  Some of the records have obvious issues.  For example, three of them have an `open` price of 999.99.  After looking into the data around these records, several have a inaccurate `open` price of 0.01 the day prior.  Below is a quick look at a few examples of those bad 0.01 prices...

# In[ ]:


market_train_df[((market_train_df['assetCode'] == 'PBRa.N') & (market_train_df['time'] >= '2007-05-02') & (market_train_df['time'] < '2007-05-05'))
                | ((market_train_df['assetCode'] == 'EXH.N') & (market_train_df['time'] >= '2007-08-22') & (market_train_df['time'] < '2007-08-25'))
                | ((market_train_df['assetCode'] == 'ATPG.O') & (market_train_df['time'] >= '2007-10-29') & (market_train_df['time'] < '2007-11-01'))
                | ((market_train_df['assetCode'] == 'TEO.N') & (market_train_df['time'] >= '2007-02-26') & (market_train_df['time'] < '2007-03-01'))
               ].sort_values('assetCode')

# Again, at the moment I'm probably not going to leverage the market data from 2007 and 2008, so I'm not going to worry about fixing errors during that time frame.  But if you are going to use market data from these years, you should probably look into fixing errors like those seen above.
# 
# Looking again at the summary statistics, but this time filtering for years 2009 and after, we can see a much nicer looking set of statistics with fewer extreme/outlier values across the board...

# In[ ]:


market_train_df = market_train_df[market_train_df['time'].dt.year >= 2009]
market_train_df.describe().round(3)

# ### Paragon Offshore (PGN.N) bad return data from Oct 2011 to Mar 2012
# ---
# The summary table above still shows there are records with abnormally large day-over-day returns, with a max value of around 700%.  Looking at records that imply an increase by more than 300% in 24 hours...

# In[ ]:


market_train_df[(market_train_df['returnsOpenPrevRaw1'] > 3)]

# There are four records fitting this criteria, including two records for Paragon Offshore PLC.  After reviewing the surrounding data it appears there are data quality issues with all three of these stocks.
# 
# Starting with Paragon, I took a zoomed out  look at the `open` and `close` prices for PGN.N around these two return spikes...

# In[ ]:


paragonSample = market_train_df[(market_train_df['assetCode'] == 'PGN.N')
                                & (market_train_df['time'] >= '2011-08-01')
                                & (market_train_df['time'] < '2012-06-01')]

paragonOpen = go.Scatter(x = paragonSample['date'],y = paragonSample['open'],mode = 'lines',name = 'open price')
paragonClose = go.Scatter(x = paragonSample['date'],y = paragonSample['close'],mode = 'lines',name = 'close price')
paragonOpenPrevRaw1 = go.Scatter(x = paragonSample['date'],y = paragonSample['returnsOpenPrevRaw1'],mode = 'lines',name = 'returnsOpenPrevRaw1',yaxis='y2')
data = [paragonOpen,paragonClose,paragonOpenPrevRaw1]
layout = go.Layout(
    title = 'PGN.N stock price vs. OpenPrevRaw1 return',
    yaxis = dict(
        title='Stock Price',
        showgrid=False,
        zeroline=False
    ),
    yaxis2 = dict(
        title = 'return (%)',
        overlaying = 'y',
        showgrid=False,
        zeroline=False,
        side = 'right',
        tickformat = ',.0%'
    ),
    legend=dict(orientation="h",x=.25)
)
fig = go.Figure(data=data, layout=layout)
iplot(fig)

# From the graph above, we can see Paragon Offshore was a very volatile stock during this window.  There is a large spike in price from 0.09 to 0.736 on 2011-11-02.  So, the first spike in `returnsOpenPrevRaw1` does appear to be accurate.  However, there is another spike implying a 559% increase in `open` price on 2012-02-15 that is clearly an error.  While not as obvious, there is also an error in the raw return data on 2011-12-23.
# 
# After further review, the raw return data errors for PGN.N on 2011-12-23 and 2012-02-15 have cause broader issues with return data in the surrounding months.  After identifying the affected rows (which spanned from 2011-10-19 to 2012-03-28), I leveraged the helper functions from earlier to correct the `raw` return data and estimate new values for the `mktres` return data.

# In[ ]:


# bad return data for PGN.N from 2011-10-19 to 2012-03-28
assetCode = 'PGN.N'
# fix first section of incorrect raw returns
badDate1 = '2011-12-23'
badIndex1 = market_train_df[(market_train_df['assetCode'] == assetCode) & (market_train_df['date'] == badDate1)].index[0]
dayWindow1 = 20
badReturnDataRanges1 = [0,9]
badDataWindow1 = sampleAssetData(assetCode,badDate1,dayWindow1)
badDataWindow1.reset_index(inplace=True)
newIdx1 = badDataWindow1[badDataWindow1['index'] == badIndex1].index[0]
rawIndices1 = [
    [badIndex1,badDataWindow1.loc[newIdx1+badReturnDataRanges1[0],'index']],
    [badIndex1,badDataWindow1.loc[newIdx1+badReturnDataRanges1[1],'index']]
]
badDataWindow1.set_index('index',inplace=True)
updateRawReturns(badDataWindow1,rawIndices1)
# fix second section of incorrect raw returns
badDate2 = '2012-02-15'
badIndex2 = market_train_df[(market_train_df['assetCode'] == assetCode) & (market_train_df['date'] == badDate2)].index[0]
dayWindow2 = 20
badReturnDataRanges2 = [0,9]
badDataWindow2 = sampleAssetData(assetCode,badDate2,dayWindow2)
badDataWindow2.reset_index(inplace=True)
newIdx2 = badDataWindow2[badDataWindow2['index'] == badIndex2].index[0]
rawIndices2 = [
    [badIndex2,badDataWindow2.loc[newIdx2+badReturnDataRanges2[0],'index']],
    [badIndex2,badDataWindow2.loc[newIdx2+badReturnDataRanges2[1],'index']]
]
badDataWindow2.set_index('index',inplace=True)
updateRawReturns(badDataWindow2,rawIndices2)
# fix mktres returns
badDate3 = '2012-01-09'
badIndex3 = market_train_df[(market_train_df['assetCode'] == assetCode) & (market_train_df['date'] == badDate3)].index[0]
dayWindow3 = 120
badReturnDataRanges3 = [-44,46,55,-55,44]
badDataWindow3 = sampleAssetData(assetCode,badDate3,dayWindow3)
badDataWindow3.reset_index(inplace=True)
newIdx3 = badDataWindow3[badDataWindow3['index'] == badIndex3].index[0]
indices3 = [[],[],
    [badDataWindow3.loc[newIdx3+badReturnDataRanges3[0],'index'],badDataWindow3.loc[newIdx3+badReturnDataRanges3[1],'index']],
    [badDataWindow3.loc[newIdx3+badReturnDataRanges3[0],'index'],badDataWindow3.loc[newIdx3+badReturnDataRanges3[2],'index']],
    [badDataWindow3.loc[newIdx3+badReturnDataRanges3[3],'index'],badDataWindow3.loc[newIdx3+badReturnDataRanges3[4],'index']]
]
badDataWindow3.set_index('index',inplace=True)
updateMktresReturns(assetCode,badDataWindow3,indices3)

# Now looking again at the returns metrics during this time, we see much cleaner values...

# In[ ]:


paragonSample = market_train_df[(market_train_df['assetCode'] == 'PGN.N')
                                & (market_train_df['time'] >= '2011-08-01')
                                & (market_train_df['time'] < '2012-06-01')]

paragonOpen = go.Scatter(x = paragonSample['date'],y = paragonSample['open'],mode = 'lines',name = 'open price')
paragonClose = go.Scatter(x = paragonSample['date'],y = paragonSample['close'],mode = 'lines',name = 'close price')
paragonOpenPrevRaw1 = go.Scatter(x = paragonSample['date'],y = paragonSample['returnsOpenPrevRaw1'],mode = 'lines',name = 'returnsOpenPrevRaw1',yaxis='y2')
paragonClosePrevMktres10 = go.Scatter(x = paragonSample['date'],y = paragonSample['returnsClosePrevMktres10'],mode = 'lines',name = 'returnsClosePrevMktres10',yaxis='y2')
paragonOpenNextMktres10 = go.Scatter(x = paragonSample['date'],y = paragonSample['returnsOpenNextMktres10'],mode = 'lines',name = 'returnsOpenNextMktres10',yaxis='y2')
data = [paragonOpen,paragonClose,paragonOpenPrevRaw1,paragonClosePrevMktres10,paragonOpenNextMktres10]
layout = go.Layout(
    title = 'PGN.N stock price vs. selected return metrics',
    yaxis = dict(
        title='Stock Price',
        showgrid=False
    ),
    yaxis2 = dict(
        title = 'return (%)',
        overlaying = 'y',
        showgrid=False,
        side = 'right',
        tickformat = ',.0%'
    ),
    legend=dict(orientation="h",x=.15)
)
fig = go.Figure(data=data, layout=layout)
iplot(fig)

# ### Qorvo (QRVO.O) bad return data in Jan and Feb 2015
# ---
# The record with bad return data for Qorvo on 2015-01-02 is the first day of data available in the training set.  Looking elsewhere online also shows no historical stock price data available for Qorvo prior to 2015-01-02.
# 
# The graph below shows Qorvo's stock price vs. return metrics, and we can see in addition to the bad return data on 2015-01-02, return data continues to fluctuate wildly.  For example, some of the 10-day return metrics show values implying near 100% increases weeks after 2015-01-02 that are are obviously wrong when looking at the actual price data.

# In[ ]:


qorvo = market_train_df[(market_train_df['assetCode'] == 'QRVO.O')]

qorvoOpen = go.Scatter(x = qorvo['date'],y = qorvo['open'],mode = 'lines',name = 'open price',yaxis= 'y2')
qorvoClose = go.Scatter(x = qorvo['date'],y = qorvo['close'],mode = 'lines',name = 'close price',yaxis= 'y2')
qorvoOpenPrevRaw10 = go.Scatter(x = qorvo['date'],y = qorvo['returnsOpenPrevRaw10'],mode = 'lines',name = 'returnsOpenPrevRaw10')
qorvoOpenPrevMktres10 = go.Scatter(x = qorvo['date'],y = qorvo['returnsOpenPrevMktres10'],mode = 'lines',name = 'returnsOpenPrevMktres10')
qorvoOpenNextMktres10 = go.Scatter(x = qorvo['date'],y = qorvo['returnsOpenNextMktres10'],mode = 'lines',name = 'returnsOpenNextMktres10')
data = [qorvoOpen,qorvoClose,qorvoOpenPrevRaw10,qorvoOpenPrevMktres10,qorvoOpenNextMktres10]
layout = go.Layout(
    title = 'QRVO.O stock price vs. returns<br>(notice the bad returns data through 2015-02-13)',
    yaxis = dict(
        title='return metrics (%)',
        tickformat = ',.0%',
        showgrid=False,
        zeroline=False
    ),
    yaxis2 = dict(
        title = 'Stock Price',
        overlaying = 'y',
        showgrid=False,
        zeroline=False,
        side = 'right'
    )
)
fig = go.Figure(data=data, layout=layout)
iplot(fig)

# The inaccurate returns data for Qorvo are found from 2015-01-02 to 2015-02-13.  Given that the affected data is at the beginning of the available historical data for Qorvo (and some of these returns should actual just be "N/A"), I've decided to delete the Qorvo data during this time frame, as opposed to trying to come up with a fix.

# In[ ]:


# dropping Qorvo data through 2015-02-13
market_train_df = market_train_df[~((market_train_df['assetCode'] == 'QRVO.O')
                                  & (market_train_df['time'] < '2015-02-14'))]

# Looking back at the Qorvo data, now the remaining returns data is in a reasonable range with no extreme outliers...

# In[ ]:


qorvo = market_train_df[(market_train_df['assetCode'] == 'QRVO.O')]

qorvoOpenPrevRaw10 = go.Scatter(x = qorvo['date'],y = qorvo['returnsOpenPrevRaw10'],mode = 'lines',name = 'returnsOpenPrevRaw10')
qorvoOpenPrevMktres10 = go.Scatter(x = qorvo['date'],y = qorvo['returnsOpenPrevMktres10'],mode = 'lines',name = 'returnsOpenPrevMktres10')
qorvoOpenNextMktres10 = go.Scatter(x = qorvo['date'],y = qorvo['returnsOpenNextMktres10'],mode = 'lines',name = 'returnsOpenNextMktres10')
data = [qorvoOpenPrevRaw10,qorvoOpenPrevMktres10,qorvoOpenNextMktres10]
layout = go.Layout(
    title = 'QRVO.O returns',
    yaxis = dict(
        title='return (%)',
        tickformat = ',.0%',
        showgrid=False,
        zeroline=False
    )
)
fig = go.Figure(data=data, layout=layout)
iplot(fig)

# ### Tech Data (TECD.O) bad open price on 2015-03-16
# ---
# Now we look at Tech Data Corp's record on 2015-03-16 which shows a `returnsOpenPrevRaw1` value of 387%.  A quick review of Tech Data market data reveals an error in the `open` price on this date.  This error is confirmed by taking a quick look online for TECD.0's historical stock price.  The actual `open` price that day was 56.18 (not 263.8).  Using the helper functions from earlier, we can go ahead and fix the bad value, along with the returns data affected by this error.

# In[ ]:


# bad volume, open, and close for TECD.O on 2015-03-16
assetCode = 'TECD.O'
badDate = '2015-03-16'
badIndex = market_train_df[(market_train_df['assetCode'] == assetCode) & (market_train_df['date'] == badDate)].index[0]
# correct bad data
market_train_df.loc[[badIndex],['open']] = 56.18
# ranges of affected return data
badReturnDataRanges = [1,10,20,20,-11,9]
# sample data window (on either side of "badDate")
sampleWindow = 45
# fix bad return data in market_train_df
fixBadReturnData(assetCode,badDate,badIndex,badReturnDataRanges, sampleWindow)

# ### Centrais Eletricas Brasileiras SA (EBR.N) bad return data in Oct 2016
# ---
# As I continued looking for outliers and bad data, I came across some return metrics that didn't make sense for the stock EBR.N.  The graph below shows the last couple years worth of data available for this stock in the training data..

# In[ ]:


ebrnSample = market_train_df[(market_train_df['assetCode'] == 'EBR.N')
                             & (market_train_df['time'] >= '2014-01-01')]

ebrnClose = go.Scatter(x = ebrnSample['date'],y = ebrnSample['close'],mode = 'lines',name = 'close price',yaxis= 'y2')
ebrnClosePrevRaw1 = go.Scatter(x = ebrnSample['date'],y = ebrnSample['returnsClosePrevRaw1'],mode = 'lines',name = 'returnsClosePrevRaw1')
ebrnOpenNextMktres10 = go.Scatter(x = ebrnSample['date'],y = ebrnSample['returnsOpenNextMktres10'],mode = 'lines',name = 'returnsOpenNextMktres10')
data = [ebrnClose,ebrnClosePrevRaw1,ebrnOpenNextMktres10]
layout = go.Layout(
    title = 'EBR.N close price vs. returns<br>(notice gap in data from May to Oct 2016 and bad returns data in Oct 2016)',
    yaxis = dict(
        title='return metrics (%)',
        tickformat = ',.0%',
        showgrid=False,
        zeroline=False
    ),
    yaxis2 = dict(
        title = 'Close Price',
        overlaying = 'y',
        showgrid=False,
        zeroline=False,
        side = 'right'
    )
)
fig = go.Figure(data=data, layout=layout)
iplot(fig)

# From the graph we can see there is a large gap in data from 2016-05-17 to 2016-10-13.  This gap appears have caused issues with the returns data.  There is a value of 240% on 2016-10-13 for the `returnsClosePrevRaw1` metric (as if the stock price jumped over night, when in fact it took 5 months to climb that high).  We can also see odd values for the `returnsOpenNextMktres10` metric, including -123% on 2016-10-18 which makes no sense.  Given these handful of observations in Oct 2016 are isolated from the rest of the data for EBR.N, I opted to delete these from the training set, rather than trying to fix them or fill in the months of missing data between May and Oct 2016.

# In[ ]:


# dropping EBR.N data in Oct 2016
market_train_df = market_train_df[~((market_train_df['assetCode'] == 'EBR.N')
                                  & (market_train_df['time'] >= '2016-10-01'))]

# With the Oct 2016 data removed, we can see that the remaining returns data for EBR.N is in a much more reasonable range...

# In[ ]:


ebrn = market_train_df[(market_train_df['assetCode'] == 'EBR.N')]
ebrn.iplot(kind='line',x='date',y=['returnsClosePrevRaw1','returnsOpenNextMktres10'],layout=dict(title='EBR.N selected return metrics',yaxis=dict(tickformat = ',.0%')))

# ### Human Genome Sciences Inc (HGSI.O) bad return data in 2009
# ---
# If you return to the summary statistics table and scroll to the far right, the minimum value for the `returnsClosePrevMktres10` and `returnsOpenPrevMktres10` metrics is less than -1.  This to me implies a decrease in price of more than 100% which doesn't make sense.

# In[ ]:


market_train_df.describe().round(3)

# Looking at records with less than -100% value for the `returnsClosePrevMktres10` metric, we see there are three stocks that fit this criteria: HGSI.O, IDIX.O, and HTZ.N.

# In[ ]:


market_train_df[market_train_df['returnsClosePrevMktres10'] < -1]

# Starting with HGSI.O, the `raw` returns data looked fine, but there were obvious issues with the `mktres` returns data.  The graph below shows the stock's price and the 10-day `PrevMktres` returns around the affected time period.

# In[ ]:


hgsi = market_train_df[(market_train_df['assetCode'] == 'HGSI.O')
                       & (market_train_df['time'] < '2010-07-01')]

hgsiClose = go.Scatter(x = hgsi['date'],y = hgsi['close'],mode = 'lines',name = 'close price',yaxis= 'y2')
hgsiOpenPrevMktres10 = go.Scatter(x = hgsi['date'],y = hgsi['returnsOpenPrevMktres10'],mode = 'lines',name = 'returnsOpenPrevMktres10')
hgsiClosePrevMktres10 = go.Scatter(x = hgsi['date'],y = hgsi['returnsClosePrevMktres10'],mode = 'lines',name = 'returnsClosePrevMktres10')
data = [hgsiClose,hgsiOpenPrevMktres10,hgsiClosePrevMktres10]
layout = go.Layout(
    title = 'HGSI.O stock price vs. 10-day Prev returns<br>(notice the large swings and incorrect return values in Aug 2009)',
    yaxis = dict(
        title='return metrics (%)',
        tickformat = ',.0%',
        showgrid=False,
        zeroline=False
    ),
    yaxis2 = dict(
        title = 'Stock Price',
        overlaying = 'y',
        showgrid=False,
        zeroline=False,
        side = 'right'
    )
)
fig = go.Figure(data=data, layout=layout)
iplot(fig)

# There are wild fluctuations for the 10-day `PrevMktres` returns metrics (including large negative values) that don't align with the price data.  It's tough to say what is causing this, but it is likely a byproduct of the large spike in price around 2009-06-20 (or perhaps the large set of missing data between March and June).  Regardless of what caused the inaccurate values, I decided to estimate the affected return data around this time.
# 
# After some review, there also appeared to be issues with return data in Feb and Mar 2009.  Given these 13 records in early 2009 are completely isolated from the rest of HGSI.O's historical data, it would be impossible to accurately estimate new values for the prediction column `returnsOpenNextMktres10`.  So, I opted to remove these handful of data points.

# In[ ]:


# bad return data for HGSI.O around June to August 2009
assetCode = 'HGSI.O'
badDate1 = '2009-08-03'
badIndex1 = market_train_df[(market_train_df['assetCode'] == assetCode) & (market_train_df['date'] == badDate1)].index[0]
dayWindow1 = 120
badReturnDataRanges1 = [-7,11,-14,19,-25,8]
badDataWindow1 = sampleAssetData(assetCode,badDate1,dayWindow1)
badDataWindow1.reset_index(inplace=True)
newIdx1 = badDataWindow1[badDataWindow1['index'] == badIndex1].index[0]
indices1 = [[],[],
    [badDataWindow1.loc[newIdx1+badReturnDataRanges1[0],'index'],badDataWindow1.loc[newIdx1+badReturnDataRanges1[1],'index']],
    [badDataWindow1.loc[newIdx1+badReturnDataRanges1[2],'index'],badDataWindow1.loc[newIdx1+badReturnDataRanges1[3],'index']],
    [badDataWindow1.loc[newIdx1+badReturnDataRanges1[4],'index'],badDataWindow1.loc[newIdx1+badReturnDataRanges1[5],'index']]
]
badDataWindow1.set_index('index',inplace=True)
updateMktresReturns(assetCode,badDataWindow1,indices1)
# dropping HGSI.O data in Feb and Mar 2016
market_train_df = market_train_df[~((market_train_df['assetCode'] == 'HGSI.O')
                                  & (market_train_df['time'] < '2009-04-01'))]

# Taking another look at the HGSI.O data, we see much improved looking return data in 2009...

# In[ ]:


hgsi = market_train_df[(market_train_df['assetCode'] == 'HGSI.O')
                       & (market_train_df['time'] < '2010-07-01')]

hgsiClose = go.Scatter(x = hgsi['date'],y = hgsi['close'],mode = 'lines',name = 'close price',yaxis= 'y2')
hgsiOpenPrevMktres10 = go.Scatter(x = hgsi['date'],y = hgsi['returnsOpenPrevMktres10'],mode = 'lines',name = 'returnsOpenPrevMktres10')
hgsiClosePrevMktres10 = go.Scatter(x = hgsi['date'],y = hgsi['returnsClosePrevMktres10'],mode = 'lines',name = 'returnsClosePrevMktres10')
data = [hgsiClose,hgsiOpenPrevMktres10,hgsiClosePrevMktres10]
layout = go.Layout(
    title = 'HGSI.O stock price vs. 10-day Prev returns',
    yaxis = dict(
        title='return metrics (%)',
        tickformat = ',.0%',
        showgrid=False,
        zeroline=False
    ),
    yaxis2 = dict(
        title = 'Stock Price',
        overlaying = 'y',
        showgrid=False,
        zeroline=False,
        side = 'right'
    )
)
fig = go.Figure(data=data, layout=layout)
iplot(fig)

# ### Stock Splits
# ---
# While not data errors, if you want to use the `open` and `close` columns to generate new features, you should be aware of potential issues arising from stock splits.
# 
# Using Apple's stock as an example, below is a graph of the `open` price for the `assetCode`='AAPL.O' in the `market_train_df`...

# In[ ]:


apple = market_train_df[market_train_df['assetCode'] == 'AAPL.O']
apple.iplot(kind='line',x='date',y='open')

# From the graph it looks like Apple's stock plummeted in June 2014, but acutally the stock just split.  There is commentary about this event in the news data...

# In[ ]:


appleNews = news_train_df[news_train_df['assetName'] == 'Apple Inc']
list(appleNews[(appleNews['headline'].str.contains('stock split')) & (appleNews['relevance'] >= 0.6)].head()['headline'])

# Apple's 7-to-1 stock split occurred on 2014-06-09.  It's worth noting that while the `open` and `close` columns don't take into account the stock split, the returns columns do (see excerpt below).

# In[ ]:


apple[(apple['time'] > '2014-06-01') & (apple['time'] < '2014-06-16')]

# Since it appears there's no issues with the return columns, if you're not planning on calculating any new features using the `open` and `close` columns, then you shouldn't need to worry about stock splits.
# 
# However, I am interested in trying to create some new features from these columns (e.g. moving averages), so I'm looking at adjusting historical stock prices like Apple's to account for splits.  Below is a graph showing an adjusted view of Apple's `open` price along with some handy moving averages.

# In[ ]:


apple['adjOpen'] = np.where(apple['time'] < '2014-06-09',apple['open']/7.0,apple['open'])
apple['MA10'] = apple['adjOpen'].rolling(window=10).mean()
apple['MA50'] = apple['adjOpen'].rolling(window=50).mean()
apple['MA200'] = apple['adjOpen'].rolling(window=200).mean()
apple.iplot(kind='line',x='date',y=['adjOpen','MA10','MA50','MA200'])

# ---
# ---
# I'm continuing to explore and clean up the market (and news) data prior to modeling and predicting.
# 
# In general, the training data appears to be in pretty good shape.  But as I find issues, I'll update this kernel.  Hope this helps!
