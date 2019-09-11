#!/usr/bin/env python
# coding: utf-8

# ## Impact of Game of Thrones on US Baby Names
# 
# #### Table of contents
# 1. [Introduction](#introduction)
# 2. [Importing required modules](#2)
# 3. [Importing the data](#3)
# 4. [Viewing a sample of the data](#4)
# 5. [Some basic summary statistics](#5)
# 6. [Trend in number of yearly applicants in the last 20 years](#6)
# 7. [Function to plot the yearly number of applicants with a particular name](#7)
# 8. [Trend in popularity of Game of Thrones character names](#8)
#     *     [Daenerys Targaryen](#i)
#     *     [Arya Stark](#ii)
#     *     [Sansa Stark](#iii)
#     *     [Tyrion Lannister](#iv)
#     *     [Brienne Tarth](#v)
#     *     [Lyanna Stark](#vi)
#     *      [Meera Reed](#vii)
#     *      [Other important characters](#viii)
# 9. [Next step](#9)
# 10. [References](#10)
# 
# 
# ![](http://theunknows.e-monsite.com/medias/images/got.jpg?fx=r_1200_800)

# ### 1.Introduction <a name="introduction"></a>
# 
# #### Objective:
# * Popular culture has always impacted society in interesting ways
# * In this kernel, we explore how Game of Thrones has inspired baby names in recent years
# * We look at some manually chosen Game of Thrones characters, and how the popularity of their names has risen along with their popularity on the show
# 
# #### Background:
# * ** Game of Thrones**
#     * Game of Thrones is an American fantasy drama television series acclaimed by critics and audiences
#     * The show has a broad, active fan base. In 2012 Vulture.com ranked the series' fans as the most devoted in popular culture, more so than Lady Gaga's, Justin Bieber's, Harry Potter's or Star Wars'
# * **About the data**
#     * This public dataset was created by the Social Security Administration and contains all names from Social Security card applications for births that occurred in the United States after 1879
#     * All data are from a 100% sample of records on Social Security card applications
#     * To safeguard privacy, the Social Security Administration restricts names to those with at least 5 occurrences

# ### 2.Importing required modules <a name="2"></a>

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from tabulate import tabulate
import bq_helper
import os

# ### 3.Importing the data <a name="3"></a>
# 
# (The below query has been taken from [Salil Gautam](https://www.kaggle.com/salil007)'s [kernel](https://www.kaggle.com/salil007/a-very-extensive-exploratory-analysis-usa-names))

# In[ ]:


#https://www.kaggle.com/salil007/a-very-extensive-exploratory-analysis-usa-names
usa_names = bq_helper.BigQueryHelper(active_project="bigquery-public-data", dataset_name="usa_names")
query = """SELECT year, gender, name, sum(number) as count FROM `bigquery-public-data.usa_names.usa_1910_current` GROUP BY year, gender, name"""
data = usa_names.query_to_pandas_safe(query)
data.to_csv("usa_names_data.csv")

# ### 4.Viewing a sample of the data <a name="4"></a>

# In[ ]:


data.sample(5)

# ### 5.Some basic summary statistics <a name="5"></a>

# In[ ]:


print("Number of rows and columns in the data: ",data.shape,"\n")
print("Number and Range of years available: ",len(data["year"].unique()),"years between ",data["year"].min(), "to ",data["year"].max())

print("Total number of applicants in the dataset: ",sum(data["count"]))
print("% of male applicants : ","{0:.2f}".format(sum(data["count"][data["gender"]=="M"])/sum(data["count"])))
print("% of female applicants :","{0:.2f}".format(sum(data["count"][data["gender"]=="F"])/sum(data["count"])),"\n")

print("Total number of unique names in the data set: ",len(data["name"].unique()))
print("Total number of unique male names in the data set: ",len(data["name"][data["gender"]=="M"].unique()))
print("Total number of unique female names in the data set: ",len(data["name"][data["gender"]=="F"].unique()),"\n")

print("\n Most popular male names of all time")
print(tabulate(data[data["gender"]=="M"].groupby('name', as_index=False).agg({"count": "sum"}).sort_values("count",ascending=False).reset_index(drop=True).head(5),headers='keys', tablefmt='psql'))

print("\n Most popular female names of all time")
print(tabulate(data[data["gender"]=="F"].groupby('name', as_index=False).agg({"count": "sum"}).sort_values("count",ascending=False).reset_index(drop=True).head(5),headers='keys', tablefmt='psql'))

# ### 6.Trend in number of yearly applicants in the last 20 years <a name="6"></a>
# 
# Let us first plot a bar chart of the number of social security applicants each year in the past 20 years.

# In[ ]:


data=data[data["year"]>=1998]
data_agg=data.groupby(["year"],as_index=False).agg({"count": "sum"})
ax=data_agg.plot('year', 'count', kind='bar', figsize=(17,5), color='#86bf91', zorder=2, width=0.85)
ax.set_xlabel("Year", labelpad=20, size=12)
# Set y-axis label
ax.set_ylabel("# of Applicants", labelpad=20, size=12)
ax.legend_.remove()

# We observe that the total number of applicants each year has fluctuated between 3M to 3.5M in the last 20 years

# ### 7.Function to plot the yearly number of applicants with a particular name  <a name="7"></a>

# In[ ]:


def plot_yearly_count(character_name):
    data_agg=data[data["name"]==character_name].groupby(["year"],as_index=False).agg({"count": "sum"})
    if len(data_agg)==0:
        print("No data available")
    else:
        year_df=pd.DataFrame()
        year_df["year"]=data["year"].unique()
        data_agg["key"]=1
        data_agg=pd.merge(year_df,data_agg,on=["year"],how="left")
        data_agg=data_agg.sort_values("year",ascending=True)
        ax=data_agg.plot('year', 'count', kind='bar', figsize=(17,5), color='#86bf91', zorder=2, width=0.85)
        # Switch off ticks
        ax.tick_params(axis="both", which="both", bottom=False, top=False, labelbottom=True, left=False, right=False, labelleft=True)
        # Set x-axis label
        ax.set_xlabel("Year", labelpad=20, size=12)
        # Set y-axis label
        ax.set_ylabel("# of Applicants", labelpad=20, size=12)
        # Set title
        ax.set_title("Popularity of the name "+str(character_name)+" in the past 20 years")
        ax.legend_.remove()

# ### 8.Trend in popularity of Game of Thrones character names  <a name="8"></a>
# 
# * Game of Thrones premiered on HBO in the United States on April 17, 2011 
# * For reference, below is the year that each season was released:
#     * Season 1:  2011
#     * Season 2: 2012
#     * Season 3: 2013
#     * Season 4: 2014
#     * Season 5: 2015
#     * Season 6: 2016
#     * Season 7: 2017
# * If we find an uptick in the number of applicants with a certain character name the same year that the character was introduced, or got a lot of screentime, this would be an indicator that some parents named their babies after the character

# ### Daenerys Targaryen <a name="i"></a>
# 
# ![](https://imagesvc.timeincapp.com/v3/fan/image?url=https%3A%2F%2Fwinteriscoming.net%2Ffiles%2F2017%2F04%2FGoT-Sn7_FirstLook_11.jpg&c=sc&w=850&h=560)
# 
# 

# In[ ]:


plot_yearly_count("Daenerys")

# * We observe above that about 20 childen were named Daenerys in 2013
# * The name has never appeared in the data base before, clearly indicating that the parents were inspired by the Game of Thrones character, who emerges  as a clear protagonist on the show in season 3, released in 2013. 
# * It is also interesting to observe that about 60 children were named Daenerys in 2016 and 2017
# * Let us take a look at the popularity of her other name "Khaleesi", across the years

# In[ ]:


plot_yearly_count("Khaleesi")

# Again, it is very clear from the above that the name "Khaleesi" has been inspired by Game of Thrones, growing in popularity with the popularity of the show from 2011 to 2017, when a whopping ~400 children were named Khaleesi

# ###  Arya Stark  <a name="ii"></a>
# 
# ![](http://images6.fanpop.com/image/photos/39600000/arya-stark-arya-stark-39631351-1280-1019.jpg)

# In[ ]:


plot_yearly_count("Arya")

# Observe above how the popularity of the name Arya shoots up between 2011 to 2017, during which the character on the show transforms from a water dancer to a faceless woman

# ### Sansa Stark <a name="iii"></a>
# 
# ![](https://wallpapersite.com/images/pages/pic_h/7345.jpg)

# In[ ]:


plot_yearly_count("Sansa")

# While in the intitial few seasons, audiences found the character Sansa Stark largely annoying, she is established as a legitimate player in the game by seasons 5-6

# ### Tyrion Lannister  <a name="iv"></a>
# 
# ![](https://resources.stuff.co.nz/content/dam/images/1/9/q/i/3/h/image.related.StuffLandscapeSixteenByNine.710x400.1v1obu.png/1557565141381.jpg)

# In[ ]:


plot_yearly_count("Tyrion")

# The above plot shows another obvious case of parents naming thier children after their favorite Game of Thrones character.

# > ### Brienne Tarth  <a name="v"></a>
# 
# ![](http://cdn.collider.com/wp-content/uploads/game-of-thrones-season-three-28.jpg)

# In[ ]:


plot_yearly_count("Brienne")

# The name "Brienne" appears in the database for the first time in 2016, with 5 children named Brienne in 2016 and 2017

# ### Lyanna Stark  <a name="vi"></a>
# 
# ![](http://watchersonthewall.com/wp-content/uploads/2017/08/Lyanna-Stark.jpg)

# In[ ]:


plot_yearly_count("Lyanna")

# The name Lyanna shoots up in popularity in the year 2016. This is the same year that season 6 was aired, which ends with a possible  revelation of Jon Snow's true heritage. Season 7, that aired in 2017, confirms the revelation through Bran's vision of Rhaegar and Lyanna's wedding.

# ### Meera Reed  <a name="vii"></a>
# 
# ![](http://www.blackfilm.com/read/wp-content/uploads/2016/04/Game-Of-Thrones-S6-Ep2-Ellie-Kendrick.jpg)

# In[ ]:


plot_yearly_count("Meera")

# The name Meera shows an increase in popularity from 2013 onwards. 2013 happens to be the year when season 3 was aired, and Meera Reed's character was first introduced onscreen.

# ### 8.Other Important Characters  <a name="vii"></a>
# 
# Let us look at the popularity of the names of other characters such as Jon Stark, Catelyn Stark, Jaime Lannister

# In[ ]:


plot_yearly_count("Jon")
plot_yearly_count("Catelyn")
plot_yearly_count("Jaime")

# From the above graphs, I was not able to descern any noticable impact of the show on the popularity of the names.

# ### Next Steps: <a name="9"></a>
# *  It would be interesting to analyize the impact of other popular culture phenomenon such as Harry Potter on the popularity of names
# * The Game of Thrones characters above were chosen from my memory, and I may have missed important characters that show a trend. To avoid this, instead of manually looking for character names, we could use the [Game of Thrones data set](https://www.kaggle.com/mylesoneill/game-of-thrones) available on Kaggle, to search for the character names in this dataset

# ### References <a name="10"></a>
# * https://gameofthrones.fandom.com/wiki/Daenerys_Targaryen
# * https://gameofthrones.fandom.com/wiki/Arya_Stark
# * https://gameofthrones.fandom.com/wiki/Sansa_Stark
# * https://gameofthrones.fandom.com/wiki/Tyrion_Lannister
# * https://gameofthrones.fandom.com/wiki/Brienne_of_Tarth
# * https://gameofthrones.fandom.com/wiki/Lyanna_Stark
# * https://gameofthrones.fandom.com/wiki/Meera_Reed
# * https://closeronline.co.uk/family/news/game-thrones-baby-boy-name-baby-girl-name-meaning/
# 
