#!/usr/bin/env python
# coding: utf-8

# # Google Play Store Visualization

# ![imglink](https://cdn.wccftech.com/wp-content/uploads/2017/09/download-google-play-store-1.png)
# 

# # Introduction
# 
# From google playstore dataset, I decide to make some simple notebook for Exploratory Data Analysis<br>
# and do some wrangling data for put it on my prediction model in future <br>
# 
# 
# In this kernel,
# 
#  - Simple Exploratory Data Analysis
#  - Data preprocessing
#  - ...

# ## Import libraries

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # this is used for the plot the graph 
import seaborn as sns # used for plot interactive graph.

import matplotlib.pyplot as plt
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go

import warnings
warnings.filterwarnings('ignore')
from pylab import rcParams
# figure size in inches


# ### Data exploration

# In[ ]:


## Read file

data = pd.read_csv("../input/googleplaystore.csv")

# In[ ]:


data.head()

# In[ ]:


print(data.shape)

#     

# In[ ]:


#missing data
total = data.isnull().sum().sort_values(ascending=False)
percent = (data.isnull().sum()/data.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(6)

# In[ ]:


data.dropna(how ='any', inplace = True)

# In[ ]:


#missing data
total = data.isnull().sum().sort_values(ascending=False)
percent = (data.isnull().sum()/data.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(6)

# In[ ]:


print(data.shape)

# **<font color='tomato'> Finding</font>**
# 
#     After remove missing data,
#     our data contain 9,360 records with 13 fields

# ## Rating

# In[ ]:


data['Rating'].describe()

# In[ ]:


# rating distibution 
rcParams['figure.figsize'] = 11.7,8.27
g = sns.kdeplot(data.Rating, color="Red", shade = True)
g.set_xlabel("Rating")
g.set_ylabel("Frequency")
plt.title('Distribution of Rating',size = 20)

# **<font color='tomato'> Finding</font>**
# 
#     Average of rating of application in store is around 4 which is very high

# ### Category

# In[ ]:


print( len(data['Category'].unique()) , "categories")

print("\n", data['Category'].unique())

# In[ ]:


g = sns.countplot(x="Category",data=data, palette = "Set1")
g.set_xticklabels(g.get_xticklabels(), rotation=90, ha="right")
g 
plt.title('Count of app in each category',size = 20)

# **<font color='tomato'> Finding</font>**
# 
#     Game and Family category are the most appearances for application in store

# In[ ]:


g = sns.catplot(x="Category",y="Rating",data=data, kind="box", height = 10 ,
palette = "Set1")
g.despine(left=True)
g.set_xticklabels(rotation=90)
g.set( xticks=range(0,34))
g = g.set_ylabels("Rating")
plt.title('Boxplot of Rating VS Category',size = 20)

# **<font color='tomato'> Finding</font>**
# 
#     Rating of application in each category is not different too much 

# ### Reviews

# In[ ]:


data['Reviews'].head()

# **<font color='tomato'> Finding</font>**
# 
#     Data still in object type, we need to convert to int

# In[ ]:


# convert to int

data['Reviews'] = data['Reviews'].apply(lambda x: int(x))

# In[ ]:


# rating distibution 
rcParams['figure.figsize'] = 11.7,8.27
g = sns.kdeplot(data.Reviews, color="Green", shade = True)
g.set_xlabel("Reviews")
g.set_ylabel("Frequency")
plt.title('Distribution of Reveiw',size = 20)

# In[ ]:


data[data.Reviews > 5000000].head()

# **<font color='tomato'> Finding</font>**
# 
#     Most of application in this store have less than 1M in reviews.
#     Obviously, well-known applictions have a lot of reviews

# ![imglink](http://emojimyface.com/wp-content/uploads/2015/01/msg-apps-icons-trans.png)
# 

# In[ ]:


plt.figure(figsize = (10,10))
g = sns.jointplot(x="Reviews", y="Rating",color = 'orange', data=data,size = 8);

# In[ ]:


plt.figure(figsize = (10,10))
sns.regplot(x="Reviews", y="Rating", color = 'darkorange',data=data[data['Reviews']<1000000]);
plt.title('Rating VS Reveiws',size = 20)

# **<font color='tomato'> Finding</font>**
# 
#     Seem like well-known application will get a good rating

# ## Size

# In[ ]:


data['Size'].head()

# In[ ]:


data['Size'].unique()

# **<font color='tomato'> Finding</font>**
# 
#     Data still in object type, and contain the metric symbol for the perefix,
#     and special size which are "Varies with device" that we need to deal with it.

# In[ ]:


len(data[data.Size == 'Varies with device'])

# In[ ]:


# change it to NA first

data['Size'].replace('Varies with device', np.nan, inplace = True ) 

# In[ ]:


data.Size = (data.Size.replace(r'[kM]+$', '', regex=True).astype(float) * \
             data.Size.str.extract(r'[\d\.]+([KM]+)', expand=False)
            .fillna(1)
            .replace(['k','M'], [10**3, 10**6]).astype(int))

# In[ ]:


data['Size'].fillna(data.groupby('Category')['Size'].transform('mean'),inplace = True)

# **<font color='forestgreen'> Note </font>**
# 
#     I decide to fill "Varies with device" with mean of size in each category

# In[ ]:


plt.figure(figsize = (10,10))
g = sns.jointplot(x="Size", y="Rating",color = 'orangered', data=data, size = 8);

# ## Installs

# In[ ]:


data['Installs'].head()

# **<font color='tomato'> Finding</font>**
# 
#     Data still in object type, and contain the plus sign

# In[ ]:


data['Installs'].unique()

# **<font color='forestgreen'> Note </font>**
# 
#     I encode it by order of size;
#     ex: 0 refer to 1+
#         1 refer to 5+
#         2 refer  to 10+
#         .
#         .
#         .

# In[ ]:


data.Installs = data.Installs.apply(lambda x: x.replace(',',''))
data.Installs = data.Installs.apply(lambda x: x.replace('+',''))
data.Installs = data.Installs.apply(lambda x: int(x))

# In[ ]:


data['Installs'].unique()

# In[ ]:


Sorted_value = sorted(list(data['Installs'].unique()))

# In[ ]:


data['Installs'].replace(Sorted_value,range(0,len(Sorted_value),1), inplace = True )

# In[ ]:


data['Installs'].head()

# In[ ]:


plt.figure(figsize = (10,10))
sns.regplot(x="Installs", y="Rating", color = 'teal',data=data);
plt.title('Rating VS Installs',size = 20)

# **<font color='tomato'> Finding</font>**
# 
#     Seem like number of install affect to rating

# ## Type

# In[ ]:


data['Type'].unique()

# In[ ]:


# Data to plot
labels =data['Type'].value_counts(sort = True).index
sizes = data['Type'].value_counts(sort = True)


colors = ["palegreen","orangered"]
explode = (0.1,0)  # explode 1st slice
 
rcParams['figure.figsize'] = 8,8
# Plot
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=270,)

plt.title('Percent of Free App in store',size = 20)
plt.show()

# **<font color='tomato'> Finding</font>**
# 
#     Most of application in this store are free (93.1%).

# **<font color='forestgreen'> Note </font>**
# 
#     Because string can't enter to model, I need to change format a little bit

# In[ ]:


data['Free'] = data['Type'].map(lambda s :1  if s =='Free' else 0)
data.drop(['Type'], axis=1, inplace=True)

# ## Price

# In[ ]:


data['Price'].head()

# In[ ]:


data.Price.unique()

# **<font color='tomato'> Finding</font>**
# 
#     Data is in object type, in format of dollar sign.

# In[ ]:


data['Price'].value_counts().head(30)

# In[ ]:


data.Price = data.Price.apply(lambda x: x.replace('$',''))
data['Price'] = data['Price'].apply(lambda x: float(x))

# In[ ]:


data['Price'].describe()

# **<font color='tomato'> Finding</font>**
# 
#     Average of price is around 0.96, but most of them are free (8715 from 9360).
#     The most expensive app is 400 dollar !!!.

# ![imglink](https://lh3.googleusercontent.com/j1tsG7eaKkVidSRk6eE25bX7sQASNGdfrUof50UM7rKu-HV_Qg7dLeKyHPSxkx-myQ=s180)
# 

# In[ ]:


data[data['Price'] == 400]

# In[ ]:


plt.figure(figsize = (10,10))
sns.regplot(x="Price", y="Rating", color = 'darkorange',data=data[data['Reviews']<1000000]);
plt.title('Scatter plot Rating VS Price',size = 20)

# **<font color='tomato'> Finding</font>**
# 
#     Higher price application may make customer disappointed, if they are not good enough.

# **<font color='forestgreen'> Note </font>**
# 
#     Try to visualize in type of band, it may easier to understand

# In[ ]:


data.loc[ data['Price'] == 0, 'PriceBand'] = '0 Free'
data.loc[(data['Price'] > 0) & (data['Price'] <= 0.99), 'PriceBand'] = '1 cheap'
data.loc[(data['Price'] > 0.99) & (data['Price'] <= 2.99), 'PriceBand']   = '2 not cheap'
data.loc[(data['Price'] > 2.99) & (data['Price'] <= 4.99), 'PriceBand']   = '3 normal'
data.loc[(data['Price'] > 4.99) & (data['Price'] <= 14.99), 'PriceBand']   = '4 expensive'
data.loc[(data['Price'] > 14.99) & (data['Price'] <= 29.99), 'PriceBand']   = '5 too expensive'
data.loc[(data['Price'] > 29.99), 'PriceBand']  = '6 FXXXing expensive'


# In[ ]:


data[['PriceBand', 'Rating']].groupby(['PriceBand'], as_index=False).mean()

# In[ ]:


g = sns.catplot(x="PriceBand",y="Rating",data=data, kind="boxen", height = 10 ,palette = "Pastel1")
g.despine(left=True)
g.set_xticklabels(rotation=90)
g = g.set_ylabels("Rating")
plt.title('Boxen plot Rating VS PriceBand',size = 20)

# **<font color='tomato'> Finding</font>**
# 
#     Price are not effect to rating ,but if it is very expensive, it might get low rating

# In[ ]:


import random

def generate_color():
    color = '#{:02x}{:02x}{:02x}'.format(*map(lambda x: random.randint(0, 255), range(3)))
    return color

# In[ ]:


# Create palette for categories

flatui = []
for i in range(0,len(data['Category'].unique()),1):
    flatui.append(generate_color())

# In[ ]:


g = sns.catplot(x="PriceBand", y="Rating", hue="Category", kind="swarm", data=data,palette = flatui,size = 10)
g.despine(left=True)
g.set_xticklabels(rotation=90)
plt.title('Category in each Priceband VS Rating',size = 20)

# ## Content Rating

# In[ ]:


data['Content Rating'].unique()

# In[ ]:


g = sns.catplot(x="Content Rating",y="Rating",data=data, kind="box", height = 10 ,palette = "Paired")
g.despine(left=True)
g.set_xticklabels(rotation=90)
g = g.set_ylabels("Rating")
plt.title('Box plot Rating VS Content Rating',size = 20)

# **<font color='tomato'> Finding</font>**
# 
#     Content Rating not effect too much to rating, 
#     but in Mature applications ,look like they get lower rating than other.

# In[ ]:


data[data['Content Rating']=='Unrated']

# In[ ]:


data = data[data['Content Rating'] != 'Unrated']

# **<font color='forestgreen'> Note </font>**
# 
#     I decide to drop 'Unrated' rating because it contain only 1 record

# In[ ]:


data = pd.get_dummies(data, columns= ["Content Rating"])

# ## Genres

# In[ ]:


print( len(data['Genres'].unique()) , "genres")

print("\n", data['Genres'].unique())

# In[ ]:


data.Genres.value_counts().head(10)

# In[ ]:


data.Genres.value_counts().tail(10)

# **<font color='forestgreen'> Note </font>**
# 
#     Many genre contain only few record, it may make a bias.
#     Then, I decide to group it to bigger genre by ignore sub-genre (after " ; " sign)

# In[ ]:


data['Genres'] = data['Genres'].str.split(';').str[0]

# In[ ]:


print( len(data['Genres'].unique()) , "genres")

print("\n", data['Genres'].unique())


# In[ ]:


data.Genres.value_counts().tail(10)

# **<font color='forestgreen'> Note </font>**
# 
#     Group Music & Audio  as  Music

# In[ ]:


data['Genres'].replace('Music & Audio', 'Music',inplace = True)

# In[ ]:


data[['Genres', 'Rating']].groupby(['Genres'], as_index=False).mean().describe()

# In[ ]:


data[['Genres', 'Rating']].groupby(['Genres'], as_index=False).mean().sort_values('Rating').head(1)

# In[ ]:


data[['Genres', 'Rating']].groupby(['Genres'], as_index=False).mean().sort_values('Rating').tail(1)

# **<font color='tomato'> Finding</font>**
# 
#     Observing from Standard Deviation, seem like genre is not effect too much to rating.
#     The lowest of an average rating on genres (Dating) is 3.97 
#     while the highest (Events) is 4.43
#    

# In[ ]:


g = sns.catplot(x="Genres",y="Rating",data=data, kind="boxen", height = 10 ,palette = "Paired")
g.despine(left=True)
g.set_xticklabels(rotation=90)
g = g.set_ylabels("Rating")
plt.title('Boxenplot of Rating VS Genres',size = 20)

# ## Last Updated

# In[ ]:


data['Last Updated'].head()

# **<font color='tomato'> Finding</font>**
# 
#     "Last Updated" feature still in String format, we need to change it in someway for plot or put it in model

# **<font color='forestgreen'> Note </font>**
# 
#     I decide to change it to "datetime" format but it still can't use in raw,
#     I create new feature which called "lastupdate",
#     "lastupdate" will contain how long is this app got update in last time (... days ago).
#     I assume that today is the day which the latest Update of application in this dataset (2018 - 08 -08)

# In[ ]:


data['new'] = pd.to_datetime(data['Last Updated'])
data['new'].describe()

# In[ ]:


# new format of  Last Updated
data['new'].max() 

# In[ ]:


# Example of finding difference between two dates in pandas
data['new'][0] -  data['new'].max()

# In[ ]:


data['lastupdate'] = (data['new'] -  data['new'].max()).dt.days
data['lastupdate'].head()

# In[ ]:


plt.figure(figsize = (10,10))
sns.regplot(x="lastupdate", y="Rating", color = 'lightpink',data=data );
plt.title('Rating  VS Last Update( days ago )',size = 20)

# In[ ]:




# In[ ]:


data.head()

#     Thanks you for read til the end.
#     If I made some mistake, please tell me in comment.
#     and...
#     
#     Vote up this kernel if you like it :)
#     
#     Next version will update soon...
