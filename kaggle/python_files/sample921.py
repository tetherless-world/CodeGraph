#!/usr/bin/env python
# coding: utf-8

# ## Customers Demographic Analysis
# We will analyze sales by demographic Analysis of customers eg city, age, gender .. The goal of this process is to give more information about our data so that the marketing team prepares to intensify the efficiency based on the data and information we will provide ! <br>
# At the end of this intention there is a challenge or duty you have to complete the end you have to understand all the facts that have already occurred and all the facts that I did not mention some and listed as a story you have to tell every part even if simple Ithakk to make a full page as a full report on this subject, From this core is teaching you how exploratory analysis is how you tell stories how to make information in your mind
# 
# * <a href="#gender">Gender</a>
# * <a href="#age">Age</a>
# * <a href="#city_category">City</a>
# * <a href="#stability">Stability</a>
# * <a href="#occupation">Occupation</a>
# * <a href="#products">Products</a>

# In[ ]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt # visualizing data
import seaborn as sns 
from collections import Counter
import plotly.plotly as py
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
import plotly.figure_factory as ff
import os
print(os.listdir("../input"))
import plotly.plotly as py
import plotly.graph_objs as go
import seaborn as sns


# In[ ]:


df = pd.read_csv('../input/BlackFriday.csv')
df.shape

# In[ ]:


#df.info()

# In[ ]:


df.describe()


# In[ ]:


df.describe()
df.head()

# # <div id="gender"> 1- Gender<div/>

# In[ ]:


explode = (0.1,0)  
fig1, ax1 = plt.subplots(figsize=(12,7))
ax1.pie(df['Gender'].value_counts(), explode=explode,labels=['Male','Female'], autopct='%1.1f%%',
        shadow=True, startangle=90)
# Equal aspect ratio ensures that pie is drawn as a circle
ax1.axis('equal')  
plt.tight_layout()
plt.legend()
plt.show()

# In[ ]:


def plot(group,column,plot):
    ax=plt.figure(figsize=(12,6))
    df.groupby(group)[column].sum().sort_values().plot(plot)
    
plot('Gender','Purchase','bar')

# **Men's purchasing power is greater than women's purchasing power, even in normal circumstances. This is likely to affect the owner of the money, but there has been a high turnout of men in the store. About 75% of the customers have made sales of men of all ages, The men are generally heading toward products at  8000 -  12,000, we have probably made sales worth more than  4 billion in men and more than 1 billion in ladies **

# # <div id="age"> 2-Age<div/>

# In[ ]:


fig1, ax1 = plt.subplots(figsize=(12,7))
sns.countplot(df['Age'],hue=df['Gender'])


# In[ ]:


plot('Age','Purchase','bar')

# **Obviously, we can consider that the target age group of our stores is the age group of 26-35 years, we have achieved sales of more than  3 billion in the age group of 26-45 years**

# # <div id="city_category">3-City <div/>

# In[ ]:


explode = (0.1, 0, 0)
fig1, ax1 = plt.subplots(figsize=(12,7))
ax1.pie(df['City_Category'].value_counts(),explode=explode, labels=df['City_Category'].unique(), autopct='%1.1f%%',
        shadow=True, startangle=90)
# Equal aspect ratio ensures that pie is drawn as a circle
ax1.axis('equal')  
plt.tight_layout()
plt.legend()
plt.show()

# In[ ]:


explode = (0.1, 0, 0)
fig1, ax1 = plt.subplots(figsize=(12,7))
ax1.pie(df.groupby('City_Category')['Purchase'].sum(),explode=explode, labels=df['City_Category'].unique(), autopct='%1.1f%%',
        shadow=True, startangle=90)
# Equal aspect ratio ensures that pie is drawn as a circle
ax1.axis('equal')  
plt.tight_layout()
plt.legend()
plt.show()

# In[ ]:


fig1, ax1 = plt.subplots(figsize=(12,7))
sns.countplot(df['City_Category'],hue=df['Age'])


# **Unexpectedly, the highest sales do not come in the number of purchases, people from Area B have a greater purchasing power than others, and greater sales gained from people from Area C**

# Look at the comments, sometimes we have to identify the appropriate wording  and not mix with it as I did here. I would like to point out that sales do not reflect purchasing power, but the number of attendees reflects purchasing power because the data are individual sales. sales in the city, it reflects the purchasing power ..

# In[ ]:


#label=['Underage 0-17','Retired +55','Middleage 26-35','46-50 y/o','Oldman 51-55','Middleage+ 36-45','Youth']
explode = (0.1, 0)
fig1, ax1 = plt.subplots(figsize=(12,7))
ax1.pie(df['Marital_Status'].value_counts(),explode=explode, labels=['Yes','No'], autopct='%1.1f%%',
        shadow=True, startangle=90)
# Equal aspect ratio ensures that pie is drawn as a circle
ax1.axis('equal')  
plt.tight_layout()
plt.legend()
plt.show()

# **City A is the most cities followed by B and then C, the distribution of ages on the procurement map is very close, we have to focus on that category of work averages of 36-45**

# **Most of our customers are more than 60% married, I see that the strategy of targeting families to ensure more clients succeed**

# # <div id="stability">4-Stability<div/>

# In[ ]:


labels=['First Year','Second Year','Third Year','More Than Four Years','Geust']
explode = (0.1, 0.1,0,0,0)
fig1, ax1 = plt.subplots(figsize=(12,7))
ax1.pie(df.groupby('Stay_In_Current_City_Years')['Purchase'].sum(),explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
# Equal aspect ratio ensures that pie is drawn as a circle
ax1.axis('equal')  
plt.tight_layout()
plt.legend()
plt.show()

# In[ ]:


labels=['First Year','Second Year','Third Year','More Than Four Years','Geust']
#label=['Underage 0-17','Retired +55','Middleage 26-35','46-50 y/o','Oldman 51-55','Middleage+ 36-45','Youth']
explode = (0.1, 0.1,0,0,0)
fig1, ax1 = plt.subplots(figsize=(12,7))
ax1.pie(df['Stay_In_Current_City_Years'].value_counts(),explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
# Equal aspect ratio ensures that pie is drawn as a circle
ax1.axis('equal')  
plt.tight_layout()
plt.legend()
plt.show()

# In[ ]:


plot('Stay_In_Current_City_Years','Purchase','bar')

# **We have worked hard in the past two years and have achieved a large percentage of sales from the new population of cities, but these figures indicate that the older city dwellers have less passion for our products. I do not know in fact look at it for yourselves why old city dwellers did not achieve higher sales of the population New visitors or visitors from outside the city?<BR>
# We have almost gained about 1.75 billion new city residents only!**

# # <div id="occupation">5-Occupation<div/>

# In[ ]:


fig1, ax1 = plt.subplots(figsize=(12,7))
df['Occupation'].value_counts().sort_values().plot('bar')


# In[ ]:


plot('Occupation','Purchase','bar')

# **We also note here that purchasing power is closely related to the Occupation in some cases as the first class of the table but there are some differences we will notice when checking the number of purchases and the value of those purchases total**

# ## <div id="products">6-Products and Catiegories </div>

# In[ ]:



plot('Product_Category_1','Purchase','barh')

# In[ ]:



plot('Product_Category_2','Purchase','barh')

# In[ ]:


plot('Product_Category_3','Purchase','barh')

# In[ ]:


fig1, ax1 = plt.subplots(figsize=(12,7))
df.groupby('Product_ID')['Purchase'].count().nlargest(10).sort_values().plot('barh')

# In[ ]:


fig1, ax1 = plt.subplots(figsize=(12,7))
df.groupby('Product_ID')['Purchase'].sum().nlargest(10).sort_values().plot('barh')

# 
# **Well, now we have the top 10 products for the top 10 profits , and first 10 category for each products .<br> Remember that I may have been explored, but this is where your next task begins. The goal of this kernel is to teach you how to analyze how to compare things to each other, How to add information you can tell yourself and tell the stakeholders.
# **

# <div id="end"><div/>
# The task is to complete this kernel and modify it and look at it again and rebuild everything even if you own your own view does not matter<br>
# You have to write a comment to describe the story that happened in the store Imagine a story What the purpose of this exercise is to compile all the conclusions and put them in the form of a story, for example :<br>
# **" On the Black Friday Day, people from different cities went to a shop and Kanu were quiet at x, y, z, and their average age was 19-45 and we gained sales by x billoin dollar   and so on ..."** until the end of the report.<br>
# Thank you please put your comments down , loai abdalslam - Pin-offer
#  Thanks guys for the gread feed back , it's awesome now

# In[ ]:



