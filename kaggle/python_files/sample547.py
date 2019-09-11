#!/usr/bin/env python
# coding: utf-8

# DSE200x Mini Project
# # Q: What Does Influence Life Expectancy the Most?
# What does help us to live longer or to die earlier?
# Among hundreds of indicators in the data set, let's find the most titghly bound to the life expectancy.
# 
# For the Life Expectancy - I took `Life expectancy at birth, total (years)` indicator, average for both genders.<br>
# For the measure of statistical relationship I took determination coefficien, by countries and years, no aggregation. <br>
# 
# For each of 1344 indicators available I calculated as follows:
# 1. Filter `country-year-value` data for the life expectancy
# 1. Filter `country-year-value` data for the indicator being tested
# 1. Inner-joined both by equality of country and year
# 1. Calculated correlation
# 1. Stored the indicator name, as long as its correlation is at least 0.5 in absolute value. See the full list in the bottom.
# 1. TODO: Perform Ridge or Lasso multi-vriate analysis with test-set cross validation
# 
# Then I listed the indicators sorted by absolute value of correlation, large to small.
# The calculation processsing takes roughly half an hour.

# In[15]:


# Soft part:
#  imported libraries
#  helper data processing functions
#  helper visualization functions

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from math import copysign

pd.set_option('display.max_rows', 500)
pd.options.display.max_rows = 200
pd.options.display.max_colwidth = 200

ROWS, COLUMNS = 0, 1

def countries_coorelation(df, iX, iY):
    corrs = pd.DataFrame(
        data = {'Correlation':float('nan'), 'Count':0}
        , index=df.index.get_level_values('CountryCode'))
    for country in df.index.get_level_values('CountryCode'):
        country_data = df.loc[ country ]
        if country_data.shape[0] > 3 and np.var(country_data[iX])>0:
            correlation = country_data[iX].corr(country_data[iY])
            corrs.loc[country] = (correlation, len(country_data))
    return corrs
    
#def years_coorelation(df, iX, iY):
#    corrs = pd.DataFrame(columns=['Correlation', 'Year', 'Count'])
#    data = get_two_indicators(df, iX, iY)
#    i = 0
#    for year in set(data.Year):
#        year_data = data[ data.Year == Year ]
#        if len(year_data) > 2:
#            correlation = year_data.Value1.corr(year_data.Value2)
#            i += 1
#            corrs.loc[i] = (correlation, year, len(year_data))
#    return corrs

def scatter_color(df, iX, iY):
    d = pvt[[iX, iY]].dropna(ROWS)
    X = d[iX]
    Y = d[iY]
    Year = d.reset_index('Year')['Year']

    plt.figure(1).set_size_inches(14,5)
    plot = plt.scatter(X, Y, c=Year, marker=',', s=6, alpha=0.25)
    plt.xlabel( ser.loc[iX].IndicatorName )
    plt.ylabel( ser.loc[iY].IndicatorName )
    plt.colorbar(plot)
    
    m, b = np.polyfit(X, Y, 1)
    fitX = np.array([min(X), max(X)])
    plt.plot(fitX, fitX*m + b, 'r-', linewidth=4)
    
    r_sq = (X*m + b).var() / Y.var()
    plt.title( 'Determination {:.0f}% , corr={:.2f}'.format( r_sq*100, X.corr(Y) ), loc='right' )

    plt.grid()
    
    return plot

def demo_indicators(iX, iY='SP.DYN.LE00.IN'):
    def signed_sqare(x):
        # converts correlation into R-sqared, but of the same sign
        return copysign( x**2, x)
    scatter_color(ind, iX, iY)
    plt.show()
    by_country = countries_coorelation(pvt, iX, iY)
    plt.hist(by_country.Correlation.dropna().apply(signed_sqare) * 100,
            bins = 20)
    plt.xlim(-100, 100)
    plt.xlabel('determination, %')
    plt.ylabel('# conutries')
    plt.grid()
    plt.show()

# In[16]:


# load data
ind = pd.read_csv('../input/Indicators.csv', usecols=['CountryCode','IndicatorCode','Year','Value'])
ser = pd.read_csv('../input/Series.csv', index_col='SeriesCode')
countries = pd.read_csv('../input/Country.csv', index_col='CountryCode')

pvt = ind.pivot_table(index=['CountryCode','Year'], columns='IndicatorCode', values='Value')
#pvt.loc[('USA',1995):('USA',1998)].iloc[:,4:8]

# In[17]:


correlations = pd.DataFrame(
    np.full_like(pvt.columns, np.nan, dtype=float)
    , columns = ['correlation']
    , index = pvt.columns)
for column in pvt.columns:
    tmpDat = pvt[[column,'SP.DYN.LE00.IN']].dropna()
    if tmpDat.size > 0 and np.var(tmpDat.iloc[:,0]) != 0:
        correlations.loc[column] = np.corrcoef(
            x = tmpDat
            , rowvar = False
        )[0,1]
correlations = correlations[['correlation']].join(
    ser[['Topic','IndicatorName','ShortDefinition','LongDefinition']]
)
correlations = correlations[correlations.Topic != 'Health: Mortality']
correlations['abscorr'] = np.abs(correlations.correlation)
#correlations.nlargest(20, 'abscorr').drop('abscorr',COLUMNS)

# # The Best Matchers
# |   | Corr | Detrmination | Indicator Code * | Indicator Name |
# |---|--------------|-----|--------------------|--------------------|
# | <img src="https://image.flaticon.com/icons/svg/190/190205.svg" height=30 width=30 /> | -0.88 | 77% | SP.DYN.CBRT.IN | Birth rate, crude (per 1,000 people) |
# | <img src="https://image.flaticon.com/icons/svg/0/422.svg" height=30 width=30 /> | +0.87 | 76% | SE.SEC.NENR | School enrollment, secondary (% net) |
# | <img src="https://image.flaticon.com/icons/svg/498/498227.svg" height=30 width=30 /> | -0.75 | 56% | NV.AGR.TOTL.ZS | Agriculture, value added (% of GDP) |
# | <img src="https://image.flaticon.com/icons/svg/263/263924.svg" height=30 width=30 />  | -0.75 | 56% | EG.USE.CRNW.ZS | Combustible renewables and waste (% of total energy) |
# | <img src="https://image.flaticon.com/icons/svg/134/134954.svg" height=30 width=30>  | +0.72 | 52% | IT.MLT.MAIN.P2 | Fixed telephone subscriptions (per 100 people) |
# | <img src="https://image.flaticon.com/icons/svg/138/138281.svg" height=30 width=30> | +0.64 | 41% | NE.CON.PRVT.PC.KD | Household final consumption expenditure per capita (constant 2005 US$) |
# and more
# 
# Note the first five indicators correlate better than wealth, the sixth row.<br>
# Wealth only explains 41% of difference in life expectancy.
# 
# \* Bornigly correlating indicators sorted out (such as  life expectancy by categories, mortality rates, etc.) <br>
# \* Icons made by Freepik, Roundicons, Smashicons from [www.flaticon.com](http://www.flaticon.com) </img>

# ## Birth rate, crude (per 1,000 people)
# Surprizingly and sad, birth rate negatively correlates to life expectancy nearly as stron as various mortality rates (see the full list in the bottom)
# 
# Histgram of correlations is also imressive. Out of 247 countries:
# * 100 countries fit nearly perfect, 80% to 100%
# * majority of others are 50% to 80%
# * a handful of countries have weak positive correlation, below 30%
# * there are no countries with positive moderate or strong correlation
# 
# Both domain and range are wide, making the tendency even more cinvincive.
# 
# Open questions for possible futher investigation:
# * what does make exceptional countries with moderate positive relationship?
# * for each birth rate value still there is 20 years worth gap between top and bottom. What does make those different?

# In[18]:


demo_indicators('SP.DYN.CBRT.IN')

# ## School enrollment, secondary (% net)
# The biggest surprise of the paper. Top of the top indicators among those which cannot be attributed to sickness/mortality things neither directly nor indirectly.<br>
# There is a number of similar indicators fitting about the same, e.g. different level school enrollments. I will describe only this one.
# 
# The indicator is available for 203 countries only. Out of these:
# * 110, the majority fit perfect 80% to 100%
# * most other fit at about 75%
# * about 20 have weak positive or negative relationship, -25% to +25%
# * few have negative correlation
# 
# Gives a fresh look at the life-long learning perspective.<br>
# Admission level of 80% and more assures at least 65 years life expectancy, nomatter is it 1970ies or2010s.<br>
# And vice versa - admission of 20% or less guarantees for expectancy under 65.

# In[19]:


print( "Countries involved: {}".format(len(set(ind[ind.IndicatorCode == 'SE.SEC.NENR'].CountryCode))) )

# In[20]:


demo_indicators('SE.SEC.NENR')

# ## Agriculture, value added (% of GDP)
# Looking at the spread width and not-as-good correlation,
#  I suggest this is merely a coincidence of two global trends:
#  1. steady reduction of agriculture fractin in world GDP 
#  1. steady increase in the worldwide average of life expectany
#  
#  As both are naturally changing slow, they are doomed to statistically fit.
#  
# Relative abundance of old years dots in the lower-left prompts the smaller contribution of agriculture is no guarantee for a better life expectancy.<br>
# 50% of agriculture and more assures life expectancy under 55, however in recent years there is a handful of countries reaching 50% (see yelow dots)<br>
# On the other hand lower agriculture contribution says little - expectancy can be anywhere between 40 and 75 years.

# In[21]:


demo_indicators('NV.AGR.TOTL.ZS')

# ## Combustive renewables and waste (% of total energy)
# Wide spread of 45 to 75 years at almost entire domain of indicator values.<br>
# Country histogram is dreadful - comparable number of countries have strong negative and strong positive correlation.
# 
# Overall the indicator is not that definitive, and mostly controversial.<br>
# This is rather an artifact of several global trends coinciding.
# 
# Or, need further investigation for the criterion to tell positivey related countries form negatively related.

# In[22]:


demo_indicators('EG.USE.CRNW.ZS')

# ## Fixed telephone subscriptions (per 100 people)
# Graph and histogram say indicator is positively and strongly bound to the life expectancy,
#  however it is far from linear.
#  
#  Roughly there all the countries with at least 10 phone landlines per 100 people have at least 65 years life expectancy.<br>
#  Less then 10 lines tell nothing - can be anywhere between 40 and 80 years.
#  
#  Interestingly enough the tendency is still there in 2010s (yellow dots)<br>
#  Did anyone hear about the age of wireless phones?
#  
#  Open question: is it only about calling for ambulance in time?

# In[23]:


demo_indicators('IT.MLT.MAIN.P2')

# ## Wholesale Price Index (2010 = 100)
# Correlation and country histogram look good

# In[24]:


demo_indicators('FP.WPI.TOTL')

# ## Household final consumption expenditure per capita (constant 2005 US \$ )
# Love is not the only thing money cannot buy you :)
# 
# Almost anything above 3,000\$ per capita annually is all the same game - between 70 and 80 years.<br>
# Also yellow/purple points in the high wage area shows it is years passing rather than wealth level that improves the result.
# 
# On the other hand area of under 3,000\$ and over 70 years is densly populated.<br>
# Sort of being poor is not being condemned.

# In[25]:


demo_indicators('NE.CON.PRVT.PC.KD')

# Please take a look below at the per-country examples for how diverse the dependency is.<br>
# Though country-wise histogram looks good enough.

# In[26]:


# row and column sharing
f, sub_plots = plt.subplots(4, 4, sharex='all', sharey='all')
f.set_size_inches(14,14)
sub_plots = [ x for a in sub_plots for x in a ]

for cList in [
    ['WLD'],['ARE'],['TUR'],['SYR'],
    ['BGR'],['BRB'],['BGD'],['BHR'],
    ['AUS'],['USA'],['UKR'],['CHE'],
    ['GBR'],['MKD'],['RUS'],['JPN']]:
    d = pvt.loc[cList]
    p = sub_plots.pop(0)
    p.scatter(d['NE.CON.PRVT.PC.KD'], d['SP.DYN.LE00.IN'], c=d.reset_index('Year').Year, marker=',', s=4)
    p.set_title("+".join(countries.loc[cList].ShortName), loc='left')

# # All Indicators Correlating at Least 50%
# Either positively or negatively

# In[27]:


correlations[correlations.abscorr>0.5].sort_values('abscorr', ascending=False)[
    ['correlation', 'IndicatorName']
]

# # The End
# Thank you for reading!
