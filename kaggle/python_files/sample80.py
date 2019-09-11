#!/usr/bin/env python
# coding: utf-8

# ## New York City Taxi Fare Prediction
# <br>
# ![New York City Taxi Fare Prediction](https://kaggle2.blob.core.windows.net/competitions/kaggle/10170/logos/header.png?t=2018-07-12-22-07-30)
# <br>
# In this playground competition, hosted in partnership with Google Cloud and Coursera, you are tasked with predicting the fare amount (inclusive of tolls) for a taxi ride in New York City given the pickup and dropoff locations. This notebook focuses on the** Exploratory Data Analysis** of the following dataset. Currently the plots are implemented with plotly and custom map styles are possible with Mapbox.  
# 
# > Maps and the EDA will be updated twice a week. 

# **Setup training data**
# First let's read in our training data. Kernels do not yet support enough memory to load the whole dataset at once, at least using pd.read_csv. The entire dataset is about 55M rows, so we're skipping a good portion of the data, but it's certainly possible to build a model using all the data.

# In[ ]:


import numpy as np 
import pandas as pd
import warnings
warnings.simplefilter("ignore")
import numpy as np
import pandas as pd
from scipy.special import boxcox
import seaborn as sns
import matplotlib.pyplot as plt
#PLOTLY
import plotly
import plotly.plotly as py
import plotly.offline as offline
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
import cufflinks as cf
from plotly.graph_objs import Scatter, Figure, Layout
cf.set_config_file(offline=True)

# In[ ]:


train = pd.read_csv("../input/train.csv", nrows = 30_000)
test = pd.read_csv("../input/test.csv",nrows = 30_000)
print(">>  Data Loaded")

# ## Checking the head of the dataframes

# **Train and Test headframes**

# In[ ]:


train.head()

# In[ ]:


test.head()

# Well, we see a key, pickup_datetime and latitude and longitude information. Let's take look into each of the following features. But, first lets check the data types of our features and decide they needed to be converted into a specific format or not. You can start checking the data type with ** df.dtypes** a pandas inbuilt function

# In[ ]:


print(train.dtypes)

# There you see! The datetime objects are just referred as objects. This is hard for interpretation and creating other date time features. We first need to convert these columns in **datetime objects**. You can easily do in online with ** pandas.to_datetime()** function

# In[ ]:


train['key'] = pd.to_datetime(train['key'])
train['pickup_datetime'] = pd.to_datetime(train['pickup_datetime'])

# Great! Now, lets check the datatypes again.

# In[ ]:


print(train.dtypes)

# That's good! See now we have datetime objects. The advantage now with this format is that we can create features on Year, Day, Month, Hour of the day, Day of Week for a particular date, quarter of the year, week of the year, is_month_start, is_month_end and other interaction features which would have been a hectic task in raw format. Thanks to pandas for such handy functionality!

# ## Find that Nulls!

# In[ ]:


print(f"Numer of Missing values in train: ", train.isnull().sum().sum())
print(f"Number of Missing values in test: ", test.isnull().sum().sum())

# Note that we are doing this analysis on only first 30,000 rows. The whole NaN might be different when you run the same operation on the whole dataset.

# Checking the shape again!

# In[ ]:


print("Train shape {}".format(train.shape))
print("Test shape {}".format(test.shape))

# ## Fare Value Distribution

# In[ ]:


target = train.fare_amount
data = [go.Histogram(x=target)]
layout = go.Layout(title = "Fare Amount Histogram")
fig = go.Figure(data=data, layout=layout)
iplot(fig)

# ## Checking Max and Min available dates 

# In[ ]:


train.head()

# In[ ]:


print(f">> Data Available since {train.key.min()}")
print(f">> Data Available upto {train.key.max()}")

# *Note that this again in in context to 30,000 rows only*

# ## A Introduction for plotting GeoPlots
# Geoplots are interesting to visualize. They are quite beautiful too! Let's visualize some of them with Plotly and Mapbox

# In[ ]:


train.head()

# ## Visualizing Pickup locations in NewYork

# In[ ]:


data = [go.Scattermapbox(
            lat= train['pickup_latitude'] ,
            lon= train['pickup_longitude'],
            customdata = train['key'],
            mode='markers',
            marker=dict(
                size= 4,
                color = 'gold',
                opacity = .8,
            ),
          )]
layout = go.Layout(autosize=False,
                   mapbox= dict(accesstoken="pk.eyJ1Ijoic2hhejEzIiwiYSI6ImNqYXA3NjhmeDR4d3Iyd2w5M2phM3E2djQifQ.yyxsAzT94VGYYEEOhxy87w",
                                bearing=10,
                                pitch=60,
                                zoom=13,
                                center= dict(
                                         lat=40.721319,
                                         lon=-73.987130),
                                style= "mapbox://styles/shaz13/cjiog1iqa1vkd2soeu5eocy4i"),
                    width=900,
                    height=600, title = "Pick up Locations in NewYork")

# In[ ]:


fig = dict(data=data, layout=layout)
iplot(fig)

# That's amazing to look at! With plotly you can also interactively zoom in and out. Try it out and explore the bird eye view of the data points. To get familiar with the plots again, lets plot the drop off location using another styled map. MapBox provides many styles, you can also makes custom maps in their studio platform. 

# ## Visualizing Dropoff locations in NewYork

# In[ ]:


data = [go.Scattermapbox(
            lat= train['dropoff_latitude'] ,
            lon= train['dropoff_longitude'],
            customdata = train['key'],
            mode='markers',
            marker=dict(
                size= 4,
                color = 'cyan',
                opacity = .8,
            ),
          )]
layout = go.Layout(autosize=False,
                   mapbox= dict(accesstoken="pk.eyJ1Ijoic2hhejEzIiwiYSI6ImNqYXA3NjhmeDR4d3Iyd2w5M2phM3E2djQifQ.yyxsAzT94VGYYEEOhxy87w",
                                bearing=10,
                                pitch=60,
                                zoom=13,
                                center= dict(
                                         lat=40.721319,
                                         lon=-73.987130),
                                style= "mapbox://styles/shaz13/cjk4wlc1s02bm2smsqd7qtjhs"),
                    width=900,
                    height=600, title = "Drop off locations in Newyork")
fig = dict(data=data, layout=layout)
iplot(fig)

# The style is Odyssey. For other styles have a look at this [link](https://www.mapbox.com/designer-maps/) 

# ## Feature Engineering for advanced plots

# In order to plot more advanced plots. We first have to get new features from the datetime features. For the demostration lets focus on feature engineering on **pickup_datetime** for the context of this kernel. The datetime features can be easily extracted with **pandas.to_datetime()** and **year, month and date** data can be extracted with respective timedate functions.

# In[ ]:


train.head(2)

# In[ ]:


train['pickup_datetime_month'] = train['pickup_datetime'].dt.month
train['pickup_datetime_year'] = train['pickup_datetime'].dt.year
train['pickup_datetime_day_of_week_name'] = train['pickup_datetime'].dt.weekday_name
train['pickup_datetime_day_of_week'] = train['pickup_datetime'].dt.weekday
train['pickup_datetime_day_of_hour'] = train['pickup_datetime'].dt.hour

# In[ ]:


train.head(7)

# Great! Now we have year, hour, day, month and week day name information with us. Let's spy what the people in Newyork are upto .. :P

# ## Travels on Business days
# Typical business day starts from Monday. So, lets segment our data on weekday basis. The **pickup_datetime_day_of_week** is numerical representation of **pickup_datetime_day_of_week_name**. Starting with Monday with 0. 

# In[ ]:


business_train = train[train['pickup_datetime_day_of_week'] < 5 ]
business_train.head(5)

# The usual plot is same as the previous ones. Let's filter the early morning and evening times pickups

# In[ ]:


early_business_hours = business_train[business_train['pickup_datetime_day_of_hour'] < 10]
late_business_hours = business_train[business_train['pickup_datetime_day_of_hour'] > 6]

# In[ ]:


data = [go.Scattermapbox(
            lat= early_business_hours['dropoff_latitude'] ,
            lon= early_business_hours['dropoff_longitude'],
            customdata = early_business_hours['key'],
            mode='markers',
            marker=dict(
                size= 5,
                color = 'gold',
                opacity = .8),
            name ='early_business_hours'
          ),
        go.Scattermapbox(
            lat= late_business_hours['dropoff_latitude'] ,
            lon= late_business_hours['dropoff_longitude'],
            customdata = late_business_hours['key'],
            mode='markers',
            marker=dict(
                size= 5,
                color = 'cyan',
                opacity = .8),
            name ='late_business_hours'
          )]
layout = go.Layout(autosize=False,
                   mapbox= dict(accesstoken="pk.eyJ1Ijoic2hhejEzIiwiYSI6ImNqYXA3NjhmeDR4d3Iyd2w5M2phM3E2djQifQ.yyxsAzT94VGYYEEOhxy87w",
                                bearing=10,
                                pitch=60,
                                zoom=13,
                                center= dict(
                                         lat=40.721319,
                                         lon=-73.987130),
                                style= "mapbox://styles/shaz13/cjiog1iqa1vkd2soeu5eocy4i"),
                    width=900,
                    height=600, title = "Early vs. Late Business Days Pickup Locations")
fig = dict(data=data, layout=layout)
iplot(fig)

# Great. Many of these **might be offices or work locations**. GIven that people travel to work during these hours. It will be interesting to see same plot vs. on weekend days

# In[ ]:


weekend_train  = train[train['pickup_datetime_day_of_week'] >= 5 ]
early_weekend_hours = weekend_train[weekend_train['pickup_datetime_day_of_hour'] < 10]
late_weekend_hours = weekend_train[weekend_train['pickup_datetime_day_of_hour'] > 6]

# In[ ]:


data = [go.Scattermapbox(
            lat= early_weekend_hours['dropoff_latitude'] ,
            lon= early_weekend_hours['dropoff_longitude'],
            customdata = early_weekend_hours['key'],
            mode='markers',
            marker=dict(
                size= 5,
                color = 'violet',
                opacity = .8),
            name ='early_weekend_hours'
          ),
        go.Scattermapbox(
            lat= late_weekend_hours['dropoff_latitude'] ,
            lon= late_weekend_hours['dropoff_longitude'],
            customdata = late_weekend_hours['key'],
            mode='markers',
            marker=dict(
                size= 5,
                color = 'orange',
                opacity = .8),
            name ='late_weekend_hours'
          )]
layout = go.Layout(autosize=False,
                   mapbox= dict(accesstoken="pk.eyJ1Ijoic2hhejEzIiwiYSI6ImNqYXA3NjhmeDR4d3Iyd2w5M2phM3E2djQifQ.yyxsAzT94VGYYEEOhxy87w",
                                bearing=10,
                                pitch=60,
                                zoom=13,
                                center= dict(
                                         lat=40.721319,
                                         lon=-73.987130),
                                style= "mapbox://styles/shaz13/cjiog1iqa1vkd2soeu5eocy4i"),
                    width=900,
                    height=600, title = "Early vs. Late Weekend Days Pickup Locations")
fig = dict(data=data, layout=layout)
iplot(fig)

# ## High fare locations

# In[ ]:


high_fares = train[train['fare_amount'] > train.fare_amount.mean() + 3* train.fare_amount.std()]

# In[ ]:


high_fares.head()

data = [go.Scattermapbox(
            lat= high_fares['pickup_latitude'] ,
            lon= high_fares['pickup_longitude'],
            customdata = high_fares['key'],
            mode='markers',
            marker=dict(
                size= 8,
                color = 'violet',
                opacity = .8),
            name ='high_fares_pick_up'
          ),
        go.Scattermapbox(
            lat= high_fares['dropoff_latitude'] ,
            lon= high_fares['dropoff_longitude'],
            customdata = high_fares['key'],
            mode='markers',
            marker=dict(
                size= 8,
                color = 'gold',
                opacity = .8),
            name ='high_fares_drop_off'
          )]
layout = go.Layout(autosize=False,
                   mapbox= dict(accesstoken="pk.eyJ1Ijoic2hhejEzIiwiYSI6ImNqYXA3NjhmeDR4d3Iyd2w5M2phM3E2djQifQ.yyxsAzT94VGYYEEOhxy87w",
                                bearing=10,
                                pitch=60,
                                zoom=13,
                                center= dict(
                                         lat=40.721319,
                                         lon=-73.987130),
                                style= "mapbox://styles/shaz13/cjk4wlc1s02bm2smsqd7qtjhs"),
                    width=900,
                    height=600, title = "High Fare Locations")
fig = dict(data=data, layout=layout)
iplot(fig)

# Feel free to provide feedbacks and also other plotting ideas besides the Todos.  Have a great day!
