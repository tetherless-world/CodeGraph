#!/usr/bin/env python
# coding: utf-8

# ## Probability of winning in US green card lottery
# ### In this kernel, we will estimate the probability of winning in the green card lottery for participants from different countries and visualize results on the map. We will also estimate the number of years one needs to participate in order to win with 0.95 probability.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import math

import geopandas as gpd
import geoplot.crs as gcrs
import geoplot as gplt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

# In[ ]:


# Adjust display settings
pd.set_option('display.max_rows', 500)

# ## Data preparation
# ### Read the lottery data

# In[ ]:


df = pd.read_csv('../input/DV_lottery_dataset.csv', sep=';')
df.fillna(0, inplace = True)
#print(df.info())

# ### Read geospatial data (we will use it later for data visualization)

# In[ ]:


path = gpd.datasets.get_path('naturalearth_lowres')
gdf = gpd.read_file(path)
gdf = gdf[gdf.name!="Antarctica"]
print(gdf.info())

# ### Merge the geospatial data and lottery data

# In[ ]:


# We will merge it on 'name' column
df['name'] = df['Foreign State']

# Fixing discrepancy in the country names
df.loc[df['Foreign State']== 'French Southern and Antarctic Lands', ['name']] = 'Fr. S. Antarctic Lands'
df.loc[df['Foreign State']== 'Bosnia and Herzegovina', ['name']] = 'Bosnia and Herz.'
df.loc[df['Foreign State']== 'Cote d\'Ivoire', ['name']] = 'Côte d\'Ivoire'
df.loc[df['Foreign State']== 'Congo, Democratic Republic of The', ['name']] = 'Dem. Rep. Congo'
df.loc[df['Foreign State']== 'Congo, Republic of The', ['name']] = 'Congo'
df.loc[df['Foreign State']== 'Czech Republic', ['name']] = 'Czech Rep.'
df.loc[df['Foreign State']== 'Equatorial Guinea', ['name']] = 'Eq. Guinea'
df.loc[df['Foreign State']== 'North Korea', ['name']] = 'Dem. Rep. Korea'
df.loc[df['Foreign State']== 'Western Sahara', ['name']] = 'W. Sahara'
df.loc[df['Foreign State']== 'South Sudan', ['name']] = 'S. Sudan'
df.loc[df['Foreign State']== 'Solomon Islands', ['name']] = 'Solomon Is.'
df.loc[df['Foreign State']== 'Laos', ['name']] = 'Lao PDR'
df.loc[df['Foreign State']== 'Central African Republic', ['name']] = 'Central African Rep.'
df.loc[df['Foreign State']== 'Solomon Islands', ['name']] = 'Solomon Is.'


# In[ ]:


gdf = gdf.merge(df, on='name', how="left")
#print(gdf.info())

# ## Let's explore the total number of participants
# ### 2018 year:

# In[ ]:


df[['Foreign State', '2018_Total']].sort_values(by='2018_Total',ascending= False)

# In[ ]:


gplt.choropleth(gdf, hue=gdf['2018_Total'],projection=gcrs.Robinson(),
                cmap='Purples', linewidth=0.5, edgecolor='gray', 
                k=None, legend=True, figsize=(20, 8))
plt.title("Total number of participants in 2018")

# ### 2007 for comparision:

# In[ ]:


gplt.choropleth(gdf, hue=gdf['2007_Total'],projection=gcrs.Robinson(),
                cmap='Purples', linewidth=0.5, edgecolor='gray', 
                k=None, legend=True, figsize=(20, 8))
plt.title("Total number of participants in 2007")

# ## Number of winners
# ### 2017:

# In[ ]:


df[['Foreign State', '2017_visas']].sort_values(by='2017_visas',ascending= False)

# In[ ]:


gplt.choropleth(gdf, hue=gdf['2017_visas'],projection=gcrs.Robinson(),
                cmap='Purples', linewidth=0.5, edgecolor='grey', 
                k=None, legend=True, figsize=(20, 8))
plt.title("Number of winners in 2017")

# ### 2008 for comparision:

# In[ ]:


gplt.choropleth(gdf, hue=gdf['2008_visas'],projection=gcrs.Robinson(),
                cmap='Purples', linewidth=0.5, edgecolor='gray', 
                k=None, legend=True, figsize=(20, 8))
plt.title("Number of winners in 2008")

# ## Let's calculate the probability of winning in a single year
# ### The probability for each country will be the number of winners divided by the number of participants. 
# ### For 2017:

# In[ ]:


df['2017_p'] = df['2017_visas']/df['2017_Total']
df['2017_p'].describe()

# In[ ]:


df[['Foreign State', '2017_p']].sort_values(by='2017_p',ascending= False)

# In[ ]:


gdf['2017_p'] = gdf['2017_visas']/gdf['2017_Total']
gplt.choropleth(gdf, hue=gdf['2017_p'],projection=gcrs.Robinson(),
                cmap='Purples', linewidth=0.5, edgecolor='gray', 
                k=None, legend=True, figsize=(20, 8))
plt.title("Probability of winning in 2017")

# ### The probability of winning in a single year (2017) for participants from all countries (except Tuvalu) is below 2%.
# ### 2010 for comparision:

# In[ ]:


gdf['2010_p'] = gdf['2010_visas']/gdf['2010_Total']
gplt.choropleth(gdf, hue=gdf['2010_p'],projection=gcrs.Robinson(),
                cmap='Purples', linewidth=0.5, edgecolor='gray', 
                k=None, legend=True, figsize=(20, 8))
plt.title("Probability of winning in 2010")

# ## Calculate number of years
# 
# ### Perhaps the probability value is too abstract, let's convert probability to a number which is more tangible. 
# 
# ### We will estimate the number of years one needs to participate in the lottery in order to win with high confidence.
# For calculations we will use Binomial distrubution formula:
# \\(P(x) = \frac{n!}{(n-x)!(x)!}p^xq^{(n-x)}\\), 
# 
# where \\(n\\) is the number of trials/years, \\(x\\) - number of wins, \\(p\\) - probability of winning in one trial/year, \\(q = 1-p\\) is probability of not winning in one trial/year.
# 
# It is easier to calculate \\(P(0)\\) (probability of not winning) and to find probability of winning as \\(1 - P(0)\\).
# 
# \\(P(0) = q^n = (1-p)^n\\)
# 
# By taking logarithm of the equation, we can get the formula for number of trials/years \\(n\\):
# 
# \\(n = \frac{ln(P(0))}{ln(1-p)}\\)
# 
# We will use this formula for estimation of the number of years
# 
# The number of years one needs to participate in the lottery in order to win with 0.95 probability:

# In[ ]:


def nestim(x, win_prob):
    if (x>0) and (x<=1):
        return math.log(1-win_prob)/math.log(1-x)
    else: return np.nan

df['2017_Nyears'] = df['2017_p'].apply(lambda x: nestim(x, 0.95)) 
print(df['2017_Nyears'].describe())

# In[ ]:


df[['Foreign State', '2017_Nyears']].sort_values(by='2017_Nyears',ascending= False)

# In[ ]:


gdf['2017_Nyears'] = gdf['2017_p'].apply(lambda x: nestim(x, 0.95)) 
gplt.choropleth(gdf, hue=gdf['2017_Nyears'],projection=gcrs.Robinson(),
                cmap='Purples', linewidth=0.5, edgecolor='gray', 
                k=None, legend=True, figsize=(20, 8))
plt.title("Number of years one needs to play to win with 0.95 probability")

# ### The result is not very promising.
# ### The estimated number of years for all countries (except Tuvalu) is higher than the current life expectancy. Of course, everyone can participate [applies to eligible countries] since it is free, but I would not recommend anyone to seriously bet on winning. 

# ## P.S. We will cross-check our calculation results using statistical simulations
# 
# ### For selected countries, we'll calculate the probability of winning over \\(n\\) years period using newly obtained number of years.
# 
# ### For Russia:

# In[ ]:


# outcome: 1 - win, 0 -lose
p = 0.005457
q = 1 - p
outcome, probability, n = [0,1], [q, p], 548
results = []
for i in range(100000):
    outcomes = np.random.choice(outcome, size=n, p=probability) 
    if 1 in outcomes: results.append(1)
    else: results.append(0)
print('Probability of winning in {0} trials is {1}'.format(n, sum(results)/len(results)))

# ### For Australia:

# In[ ]:


# outcome: 1 - win, 0 -lose
p = 0.014904
q = 1 - p
outcome, probability, n = [0,1], [q, p], 200
results = []
for i in range(100000):
    outcomes = np.random.choice(outcome, size=n, p=probability) 
    if 1 in outcomes: results.append(1)
    else: results.append(0)
print('Probability of winning in {0} trials is {1}'.format(n, sum(results)/len(results)))

# In[ ]:



