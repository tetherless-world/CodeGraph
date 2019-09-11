#!/usr/bin/env python
# coding: utf-8

# In[4]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

# In[48]:


movies = pd.read_csv("../input/movies_metadata.csv");
ratings = pd.read_csv("../input/ratings_small.csv");
keywords_plot = pd.read_csv("../input/keywords.csv");
credits = pd.read_csv("../input/credits.csv");
rate = pd.read_csv("../input/ratings.csv");
links = pd.read_csv("../input/links.csv");

# In[49]:


#let's vizualise MOVIES Data Frame and clean this data set to find Top 3 Revenue Movies.
movies

# In[50]:


#Column-Wise
movies.isnull().sum()

# In[51]:


# checking the percentage of null values
round(100*(movies.isnull().sum()/len(movies.index)),2)

# In[52]:


# We wiil drop the Columns which have null values more than 0.1%
## These Columns are : 'belongs_to_collection','homepage','overview','release_date','tagline','runtime','status'
# We will even drop Poster Path and Overview as these are not required columns
movies = movies.drop(['belongs_to_collection','homepage','overview','release_date','tagline','poster_path','overview','runtime','status'],axis=1)
movies

# In[53]:


# there might be some null values still, so let us check the % of null values present still
round(100*(movies.isnull().sum()/len(movies.index)),2)

# In[54]:


movies.shape

# In[55]:


# Now we will drop the NULL Values in the Rows
movies = movies[~np.isnan(movies.revenue)]
movies

# In[56]:


movies.shape

# In[57]:


round(100*(movies.isnull().sum()/len(movies.index)),2)

# In[58]:


movies = movies[movies.isnull().sum(axis=1)<=5]
movies

# In[59]:


movies.original_language.describe()

# In[60]:


# So we set all the missing values in the Data frame with en
movies.loc[pd.isnull(movies['original_language']),['original_language']] = 'en'
movies

# In[61]:


movies.isnull().sum()

# In[62]:


movies.imdb_id.describe()

# In[63]:


#we replace the NULL Values with the most frequent 'tt1180333', but we should not do that. So either we drop that column or we replace it with most frequent ID
#movies.loc[pd.isnull(movies['imdb_id']),['imdb_id']] = 'tt1180333'
#movies
#Instead of Imputing Values, we try to delete the Rows where imdb_id is a Null Value
movies = movies[movies['imdb_id'].notnull()]

# In[64]:


movies

# In[65]:


# Now let's check wether we have any null value or not
movies.isnull().sum()

# In[66]:


# % for null values
round(100*(movies.isnull().sum()/len(movies.index)),2)

# In[67]:


# Now the Data is Clean of any Missing Values.
# So, Top 3 revenues movies:
movies = movies.sort_values(by = 'revenue',ascending=False)
movies

# In[68]:


#Top 3 Revenue Movie
top3revenue = movies.loc[:3,]
top3revenue

# In[69]:



movies.drop_duplicates(subset = None,keep='first',inplace=True)
movies


# In[70]:


movies.set_index(['id'])

# In[71]:


#Top 10 movies according to Revenue.
movies.iloc[:10,]

# Hope you like It!

# **Now We will Look onto Ratings Data Frame and try to clean the Data Set to find the Top 10 rated Movies****

# In[72]:


ratings

# In[73]:


ratings.isnull().sum()
#We see that It is one of the cleaned data frames.
#Next Step we think of is to Merge/Concat the Data Frame. What do you think....should we Merge or Concat??

# In[74]:


top10 = ratings.sort_values(by='rating',ascending=False)
top10 = top10.iloc[:10,]
top10

# In[75]:


#top 10 movies
top10 = top10.drop(['timestamp'],axis=1)
top10

# In[76]:


#Now let's dive into other Data Frames too!
keywords_plot

# In[77]:


keywords_plot.isnull().sum()

# In[78]:


round(100*(keywords_plot.isnull().sum()/len(keywords_plot.index)),2)

# In[79]:


#Let's split the data and Apply a ML Model to predict the imdb_score for the testing data!

# In[80]:


movies.drop(['vote_count'],axis=1)

# In[81]:


X = movies.loc[:,:'video'].as_matrix()
y = movies['vote_average']

# In[82]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=2)
print(X_train.shape,X_test.shape)
print(y_train.shape,y_test.shape)

# # One thing to observe is that it is a Categorical Data and, So we either we Use Hot Encoding to to convert the dtype() or use a more Complexed behaviour Model.
