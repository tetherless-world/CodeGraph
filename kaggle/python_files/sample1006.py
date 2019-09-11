#!/usr/bin/env python
# coding: utf-8

# ## Introduction
# * Lets look at relationships between **gun violance, execution,  airbnb prices, depression number and mental health issue in Texas between 2013 and 2016.**
# * Content:
#     1. [Gun Violance](#1)
#     1. [Execution](#2)
#     1. [Airbnb Prices](#3)
#     1. [Depression Number](#4)

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import plotly.plotly as py
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

# In[ ]:


# import data
data = pd.read_csv("../input/gun-violence-data/gun-violence-data_01-2013_03-2018.csv")

# <a id="1"></a> <br>
# ## Gun Violance
# * Look at total attacks in Texas between 2013 and 2016.

# In[ ]:


# total attack(killed + injured) 
data["total_attack"] = data.n_killed + data.n_injured

# In[ ]:


# We will explore data according to years
data["year"] = [each.split("-")[0] for each in data.date]

# In[ ]:


# quick check of data
texas = data[data.state == "Texas"]
texas.head()


# In[ ]:


# total number of attack according to years. Example in 2014, there are 2251 attack(killed + injured) 
total_attack_list = []
for i in texas.year.unique():
    total_attack_list.append(sum(texas.total_attack[texas.year == i]))

# In[ ]:



# Creating trace1
trace1 = go.Scatter(
                    x = np.array(["2013","2014","2015","2016"]),
                    y = total_attack_list,
                    mode = "lines+markers",
                    name = "",
                    marker = dict(color = 'rgba(80, 26, 80, 0.8)'),
                    text= "")

data = [trace1]
layout = dict(title = 'Number of total attack (killed and injured) in Texas between 2013 and 2016',
              xaxis= dict(title= 'Year',ticklen= 5,zeroline= False),
              yaxis= dict(title= 'Number of Attack',ticklen= 5,zeroline= False)
             )
fig = dict(data = data, layout = layout)
iplot(fig)

# * **The question is that why there is an increase in attack(killed and injured) between 2013 and 2016 in Texas?**

# <a id="2"></a> <br>
# ## Execution
# * Lets start with checking  *Executions in the United States, 1976-2016* data 
#     - This data includes number of executions in USA.
#     - Even if there is not enough data to explore Texas state, number of execution in USA also gives comparable information with number of total attack.

# In[ ]:


# import Insightful & Vast USA Statistics data 
execution = pd.read_csv("../input/execution-database/database.csv", encoding='ISO-8859-1' )

# In[ ]:


# create year feature
execution["year"] = [each.split("/")[2] for each in execution.Date]

# In[ ]:


# according to years create total number of execution list
total_execution_list = []
for each in ["2013","2014","2015","2016"]:
    df  = execution[execution["year"] == each]
    total_execution_list.append(sum(df['Victim Count']))

# In[ ]:


# Creating trace1
total_attack_list_normalized = np.array(total_attack_list)/100 # In order to visualize properly I scale total_attack list
trace1 = go.Scatter(
                    x = np.array(["2013","2014","2015","2016"]),
                    y = total_attack_list_normalized,
                    mode = "lines+markers",
                    name = "Total attack(killed + injured)",
                    marker = dict(color = 'rgba(80, 26, 80, 0.8)'),
                    text= "")
# Creating trace2
trace2 = go.Scatter(
                    x = np.array(["2013","2014","2015","2016"]),
                    y = total_execution_list,
                    mode = "lines+markers",
                    name = "Number of Execution",
                    marker = dict(color = 'rgba(12, 222, 154, 0.8)'),
                    text= "")
data = [trace1,trace2]
layout = dict(title = 'Number of total attack (killed and injured) in Texas and execution in USA between 2013 and 2016',
              xaxis= dict(title= 'Year',ticklen= 5,zeroline= False),
              yaxis= dict(title= 'Number of Attack and Execution',ticklen= 5,zeroline= False)
             )
fig = dict(data = data, layout = layout)
iplot(fig)

# * As a result, when number of execution is decreased in USA, number of total attack is increased.
#   

# <a id="3"></a> <br>
# ## Airbnb Price
# * Lets check another data set that is "Airbnb Property Data from Texas"
#     - It includes average rate per night for years between 2013 and 2016

# In[ ]:


# import data
airbnb = pd.read_csv("../input/airbnb-property-data-from-texas/Airbnb_Texas_Rentals.csv")

# In[ ]:


# creater year column
airbnb["year"] = [each.split()[1] for each in airbnb.date_of_listing]
airbnb.dropna(inplace = True)
airbnb["average_rate_per_night_dollar"] = [ int(each[1:]) for each in airbnb.average_rate_per_night]

# In[ ]:


# store average rate per night
total_average_rate_per_night = []
for each in ["2013","2014","2015","2016"]:
    df  = airbnb[airbnb["year"] == each]
    total_average_rate_per_night.append(sum(df['average_rate_per_night_dollar'])/len(df['average_rate_per_night_dollar']))

# In[ ]:


# Creating trace1
total_attack_list_normalized = np.array(total_attack_list)/100 # In order to visualize properly I scale total_attack list
total_average_rate_per_night_normalized = np.array(total_average_rate_per_night)/2 # In order to visualize properly I scale total_attack list
trace1 = go.Scatter(
                    x = np.array(["2013","2014","2015","2016"]),
                    y = total_attack_list_normalized,
                    mode = "lines+markers",
                    name = "Total attack(killed + injured)",
                    marker = dict(color = 'rgba(80, 26, 80, 0.8)'),
                    text= "")
# Creating trace2
trace2 = go.Scatter(
                    x = np.array(["2013","2014","2015","2016"]),
                    y = total_execution_list,
                    mode = "lines+markers",
                    name = "Number of Execution",
                    marker = dict(color = 'rgba(12, 222, 154, 0.8)'),
                    text= "")
# Creating trace3
trace3 = go.Scatter(
                    x = np.array(["2013","2014","2015","2016"]),
                    y = total_average_rate_per_night_normalized,
                    mode = "lines+markers",
                    name = "Number of average rate per night in airbnb",
                    marker = dict(color = 'rgba(1, 1, 200, 0.8)'),
                    text= "")
data = [trace1,trace2,trace3]
layout = dict(title = 'Number of total attack (killed and injured), airbnb average price per night in Texas and execution in USA between 2013 and 2016',
              xaxis= dict(title= 'Year',ticklen= 5,zeroline= False),
              yaxis= dict(title= 'Number of Attack and Execution',ticklen= 5,zeroline= False)
             )
fig = dict(data = data, layout = layout)
iplot(fig)

# * As a result, when the airbnb price per night is increased, number of total attack is increased.

# <a id="4"></a> <br>
# ## Depression
# * Lets check other dataset that is Health searches by US Metropolitan Area, 2005-2017
#     - It includes depression number in Texas state between 2013 and 2016.
#     - We are looking for number of depression from 2013 to 2016.

# In[ ]:


# import data
health = pd.read_csv("../input/health-searches-us-county/RegionalInterestByConditionOverTime.csv")

# In[ ]:


# according to years(from 2013 to 2016) total number of depression in Texas
depression2013 = 0
depression2014 = 0
depression2015 = 0
depression2016 = 0
for each in health.dma.unique():
    if("TX" in each):
        df = health[health.dma == each]
        depression2013 = depression2013 +(df.loc[:,"2013+depression"].values[0])
        depression2014 = depression2014 +(df.loc[:,"2014+depression"].values[0])
        depression2015 = depression2015 +(df.loc[:,"2015+depression"].values[0])
        depression2016 = depression2016 +(df.loc[:,"2016+depression"].values[0])


# In[ ]:


# Creating trace1
total_attack_list_normalized = np.array(total_attack_list)/100 # In order to visualize properly I scale total_attack list
total_average_rate_per_night_normalized = np.array(total_average_rate_per_night)/2 # In order to visualize properly I scale total_attack list
total_depression_number_normalized = np.array([depression2013,depression2014,depression2015,depression2016])/100
trace1 = go.Scatter(
                    x = np.array(["2013","2014","2015","2016"]),
                    y = total_attack_list_normalized,
                    mode = "lines+markers",
                    name = "Total attack(killed + injured)",
                    marker = dict(color = 'rgba(80, 26, 80, 0.8)'),
                    text= "")
# Creating trace2
trace2 = go.Scatter(
                    x = np.array(["2013","2014","2015","2016"]),
                    y = total_execution_list,
                    mode = "lines+markers",
                    name = "Number of Execution",
                    marker = dict(color = 'rgba(12, 222, 154, 0.8)'),
                    text= "")
# Creating trace3
trace3 = go.Scatter(
                    x = np.array(["2013","2014","2015","2016"]),
                    y = total_average_rate_per_night_normalized,
                    mode = "lines+markers",
                    name = "Number of average rate per night in airbnb",
                    marker = dict(color = 'rgba(1, 1, 200, 0.8)'),
                    text= "")
# Creating trace4
trace4 = go.Scatter(
                    x = np.array(["2013","2014","2015","2016"]),
                    y = total_depression_number_normalized,
                    mode = "lines+markers",
                    name = "Depression number",
                    marker = dict(color = 'rgba(100, 100, 1, 0.8)'),
                    text= "")
data = [trace1,trace2,trace3,trace4]
layout = dict(title = 'Number of total attack, airbnb average price per night, depression number in Texas and execution in USA between 2013 and 2016',
              xaxis= dict(title= 'Year',ticklen= 5,zeroline= False),
              yaxis= dict(title= 'Number of Attack and Execution',ticklen= 5,zeroline= False)
             )
fig = dict(data = data, layout = layout)
plt.savefig('graph.png')
iplot(fig)

# * Chance in depression numbers look like constant.
# * Therefore, it does not have any effect on changes of total attack(killed and infured).

# ## Conclusion
# * As a result, total attack(killed and infured) has positively related with number of average rate per night in airbnb.(I think this result is weird)
# * Total attack(killed and infured) has negatively related with number of execution in USA.
# * On ther other hand, depression number does not have any effect on it in Texas state between 2013 and 2016.
# * If you have any question, feedback or suggestion, I will be happy yo hear it.
# * **Thank you for your comments and upvotes.**

# In[ ]:



