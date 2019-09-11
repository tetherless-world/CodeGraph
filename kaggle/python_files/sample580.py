#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Import important packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


import plotly.plotly as py
import plotly.graph_objs as go

from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)

# Model
## GLM part
from glm.glm import GLM
from glm.families import Gaussian, Bernoulli, Poisson, Exponential

## linear model
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from matplotlib import pyplot 

# In[ ]:


# Import file
data = pd.read_csv("../input/nypd-motor-vehicle-collisions.csv")

# In[ ]:


# Exploratory dataset in General
## Look at the structure of dataset
data.info()

## Look at sample data
data.tail(10)

# Look at missing value
data.isnull().sum()


# In[ ]:


# Exploratory dataset in specific variable

## create function looking at frequency table for each variable
def freq(data, var):
    tmp_freq = pd.crosstab(index = data[var], columns = 'count')
    return tmp_freq

#freq(data, 'ZIP CODE')
#freq(data, 'NUMBER OF PERSONS INJURED')
#freq(data, 'NUMBER OF PERSONS KILLED')
#freq(data, 'CONTRIBUTING FACTOR VEHICLE 1')
#freq(data, 'BOROUGH')
freq(data, 'TIME_GRP')

# In[ ]:


# Data manipulation
## Rename columns
data.rename(columns = {'ZIP CODE'          : 'ZIP_CODE',
                       'ON STREET NAME'    : 'STREET_ON',
                       'CROSS STREET NAME' : 'STREET_CROSS',
                       'OFF STREET NAME'   : 'STREET_OFF',
                       'NUMBER OF PERSONS INJURED'     : 'NUM_PER_INJUR',
                       'NUMBER OF PERSONS KILLED'      : 'NUM_PER_KILL',
                       'NUMBER OF PEDESTRIANS INJURED' : 'NUM_PED_INJUR',
                       'NUMBER OF PEDESTRIANS KILLED'  : 'NUM_PED_KILL',
                       'NUMBER OF CYCLIST INJURED'     : 'NUM_CYC_INJUR',
                       'NUMBER OF CYCLIST KILLED'      : 'NUM_CYC_KILL',
                       'NUMBER OF MOTORIST INJURED'    : 'NUM_MOTOR_INJUR',
                       'NUMBER OF MOTORIST KILLED'     : 'NUM_MOTOR_KILL',
                       'CONTRIBUTING FACTOR VEHICLE 1' : 'VEH_FACTOR_1',
                       'CONTRIBUTING FACTOR VEHICLE 2' : 'VEH_FACTOR_2',
                       'CONTRIBUTING FACTOR VEHICLE 3' : 'VEH_FACTOR_3',
                       'CONTRIBUTING FACTOR VEHICLE 4' : 'VEH_FACTOR_4',
                       'CONTRIBUTING FACTOR VEHICLE 5' : 'VEH_FACTOR_5',
                       'UNIQUE KEY' : 'UNIQUE_KEY',
                       'VEHICLE TYPE CODE 1' : 'VEH_TYPE_1',
                       'VEHICLE TYPE CODE 2' : 'VEH_TYPE_2',
                       'VEHICLE TYPE CODE 3' : 'VEH_TYPE_3',
                       'VEHICLE TYPE CODE 4' : 'VEH_TYPE_4',
                       'VEHICLE TYPE CODE 5' : 'VEH_TYPE_5'},
           inplace = True) 

# Create variables
data['DATE_YEAR'] = pd.to_datetime(data['DATE']).dt.year
data['DATE_MTH']  = pd.to_datetime(data['DATE']).dt.month

time = data['TIME']

#split_time  = lambda data: len(data['TIME'].split(".")) -1
data['TIME_O'] = data['TIME'].apply(lambda time: time.split(':')[0])

time_dict = {'0' : 'A 0 O Clock', '1' : 'B 1 O Clock', '2' : 'C 2 O Clock',
             '3' : 'D 3 O Clock', '4' : 'E 4 O Clock', '5' : 'F 5 O Clock',
             '6' : 'G 6 O Clock', '7' : 'H 7 O Clock', '8' : 'I 8 O Clock',
             '9' : 'J 9 O Clock', '10' : 'K 10 O Clock', '11' : 'L 11 O Clock',
             '12' : 'M 12 O Clock', '13' : 'N 13 O Clock', '14' : 'O 14 Clock',
             '15' : 'P 15 O Clock', '16' : 'Q 16 O Clock', '17' : 'R 17 O Clock',
             '18' : 'S 18 O Clock', '19' : 'T 19 O Clock', '20' : 'U 20 O Clock',
             '21' : 'V 21 O Clock', '22' : 'W 22 O Clock', '23' : 'X 23 O Clock' }
        
data['TIME_GRP'] = data['TIME_O'].map({value : key for value, key in time_dict.items()})
    
# Clean up na value 
data['NUM_PER_INJUR'].fillna = 0
data['NUM_PER_KILL'].fillna = 0

# Recheck columns
data.info()

# In[ ]:


# Preliminary analysis
## Create bar plot by year

plt.figure().set_figheight(5)
#plt.figure().set_figwidth(30)
#plt.figure().subplots_adjust(left = None, bottom = None, right = None, top = None, wspace=0.5, hspace=4)

# Looking at frequency of person injury by year
plt.subplot(4, 2 ,1)
data.groupby('DATE_YEAR').NUM_PER_INJUR.sum().plot.bar().set_title('Number of person injury')

# Looking at frequency of person kill by year
plt.subplot(4, 2, 2)
data.groupby('DATE_YEAR').NUM_PER_KILL.sum().plot.bar().set_title('Number of person kill')

# Looking at frequency of PEDESTRIANS injury by year
plt.subplot(4, 2, 3)
data.groupby('DATE_YEAR').NUM_PED_INJUR.sum().plot.bar().set_title('Number of pedestrians injury')

# Looking at frequency of pedesstrians kill by year
plt.subplot(4, 2, 4)
data.groupby('DATE_YEAR').NUM_PED_INJUR.sum().plot.bar().set_title('Number of pedestrians kill')

# Looking at frequency of pedestrians kill by year
plt.subplot(4, 2, 5)
data.groupby('DATE_YEAR').NUM_MOTOR_INJUR.sum().plot.bar().set_title('Number of motorist injury')

# Looking at frequency of pedestrians kill by year
plt.subplot(4, 2, 6)
data.groupby('DATE_YEAR').NUM_MOTOR_KILL.sum().plot.bar().set_title('Number of motorist kill')

# Looking at frequency of pedesstrians kill by year
plt.subplot(4, 2, 7)
data.groupby('DATE_YEAR').NUM_CYC_INJUR.sum().plot.bar().set_title('Number of cyclist injury')

# Looking at frequency of pedesstrians kill by year
plt.subplot(4, 2, 8)
data.groupby('DATE_YEAR').NUM_CYC_KILL.sum().plot.bar().set_title('Number of cyclist kill')



# In[ ]:


# Preliminary analysis
## Create bar plot by month

plt.figure().set_figheight(5)
#plt.figure().set_figwidth(30)
#plt.figure().subplots_adjust(left = None, bottom = None, right = None, top = None, wspace=0.5, hspace=4)

# Looking at frequency of person injury by year
plt.subplot(4, 2 ,1)
data.groupby('DATE_MTH').NUM_PER_INJUR.sum().plot.bar().set_title('Number of person injury')

# Looking at frequency of person kill by year
plt.subplot(4, 2, 2)
data.groupby('DATE_MTH').NUM_PER_KILL.sum().plot.bar().set_title('Number of person kill')

# Looking at frequency of PEDESTRIANS injury by year
plt.subplot(4, 2, 3)
data.groupby('DATE_MTH').NUM_PED_INJUR.sum().plot.bar().set_title('Number of pedestrians injury')

# Looking at frequency of pedesstrians kill by year
plt.subplot(4, 2, 4)
data.groupby('DATE_MTH').NUM_PED_INJUR.sum().plot.bar().set_title('Number of pedestrians kill')

# Looking at frequency of pedestrians kill by year
plt.subplot(4, 2, 5)
data.groupby('DATE_MTH').NUM_MOTOR_INJUR.sum().plot.bar().set_title('Number of motorist injury')

# Looking at frequency of pedestrians kill by year
plt.subplot(4, 2, 6)
data.groupby('DATE_MTH').NUM_MOTOR_KILL.sum().plot.bar().set_title('Number of motorist kill')

# Looking at frequency of pedesstrians kill by year
plt.subplot(4, 2, 7)
data.groupby('DATE_MTH').NUM_CYC_INJUR.sum().plot.bar().set_title('Number of cyclist injury')

# Looking at frequency of pedesstrians kill by year
plt.subplot(4, 2, 8)
data.groupby('DATE_MTH').NUM_CYC_KILL.sum().plot.bar().set_title('Number of cyclist kill')


# In[ ]:


# Preliminary analysis
## Create bar plot by time in a day

plt.figure().set_figheight(5)
#plt.figure().set_figwidth(30)
#plt.figure().subplots_adjust(left = None, bottom = None, right = None, top = None, wspace=0.5, hspace=4)

# Looking at frequency of person injury by year
plt.subplot(4, 2 ,1)
data.groupby('TIME_GRP').NUM_PER_INJUR.sum().plot.bar().set_title('Number of person injury')

# Looking at frequency of person kill by year
plt.subplot(4, 2, 2)
data.groupby('TIME_GRP').NUM_PER_KILL.sum().plot.bar().set_title('Number of person kill')

# Looking at frequency of PEDESTRIANS injury by year
plt.subplot(4, 2, 3)
data.groupby('TIME_GRP').NUM_PED_INJUR.sum().plot.bar().set_title('Number of pedestrians injury')

# Looking at frequency of pedesstrians kill by year
plt.subplot(4, 2, 4)
data.groupby('TIME_GRP').NUM_PED_INJUR.sum().plot.bar().set_title('Number of pedestrians kill')

# Looking at frequency of pedestrians kill by year
plt.subplot(4, 2, 5)
data.groupby('TIME_GRP').NUM_MOTOR_INJUR.sum().plot.bar().set_title('Number of motorist injury')

# Looking at frequency of pedestrians kill by year
plt.subplot(4, 2, 6)
data.groupby('TIME_GRP').NUM_MOTOR_KILL.sum().plot.bar().set_title('Number of motorist kill')

# Looking at frequency of pedesstrians kill by year
plt.subplot(4, 2, 7)
data.groupby('TIME_GRP').NUM_CYC_INJUR.sum().plot.bar().set_title('Number of cyclist injury')

# Looking at frequency of pedesstrians kill by year
plt.subplot(4, 2, 8)
data.groupby('TIME_GRP').NUM_CYC_KILL.sum().plot.bar().set_title('Number of cyclist kill')

# Create summary table by time in a day
data.groupby('TIME_GRP').sum()[["NUM_PER_INJUR", "NUM_PER_KILL", "NUM_PED_INJUR", "NUM_PED_KILL", 
                                "NUM_MOTOR_INJUR", "NUM_MOTOR_KILL", "NUM_CYC_INJUR", "NUM_CYC_KILL"]]


# In[ ]:


# Create variable before doing geo plot

data['LAB_NUMPERINJUR'] = 'INJURY PERSON ' + data['NUM_PER_INJUR'].astype(str) + ' KILL PERSON ' + data['NUM_PER_KILL'].astype(str)
data2 = data[:10000]
data2 = data2[(~data2['LATITUDE'].isnull()) | (~data2['LONGITUDE'].isnull()) ]

#data.info()

# In[ ]:


# Preliminary analysis
## Create frequency map by using lat long data

import warnings
warnings.filterwarnings('ignore')

mapbox_style = 'mapbox://styles/teeradol/cjvvz389101a81co5hqfdbvsi'
mapbox_access_token = 'pk.eyJ1IjoidGVlcmFkb2wiLCJhIjoiY2p2dnoybWpmNDdjYjN5cW92ejZldmxqYiJ9.v2TRrGbjGqiQqQkDwgzQ-A'


data = [go.Scattermapbox(
    lat=data2['LATITUDE'],
    lon=data2['LONGITUDE'],
    mode='markers',
    marker=dict(
        size=4,
        opacity=0.8
    ),
    text= data2['LAB_NUMPERINJUR'] ,
    name='locations'
)]

layout = dict(
    title='Motor Vehicle Collision',
    autosize=True,
    hovermode='closest',
    mapbox=dict(
        accesstoken = mapbox_access_token,
        bearing=0,
        center=dict(
            lat=40.7,
            lon=-73.9
        ),
        pitch=0,
        zoom=8.5,
        style=mapbox_style,
    ),
    xaxis = dict(
        domain = [0.6, 1]
    ),
)

fig = dict(data=data, layout=layout)

iplot(fig)


# In[1]:


# Multivariate analysis & Modeling part
## Import library for modeling part
# Library for spliting data into training and testing dataset 
from sklearn.model_selection import train_test_split

# Library for h2o cloud
import h2o
h2o.remove_all  # clean slate, in case cluster was already running
h2o.init(max_mem_size = "16g")

# Library for doiung 
from h2o.estimators.glm import H2OGeneralizedLinearEstimator
