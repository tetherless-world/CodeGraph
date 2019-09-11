#!/usr/bin/env python
# coding: utf-8

# In[36]:


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

# In[37]:


df2 = pd.read_csv("../input/deliveries.csv")
df1 = pd.read_csv("../input/matches.csv")


# In[38]:


df1.head()

# In[39]:


df1.isnull().sum()

# In[40]:


df1=df1.drop(columns={"umpire3","id"})

# In[41]:


df1.isnull().sum()

# In[42]:


df1 = df1.fillna(method="ffill")

# In[43]:


df1.columns

# In[44]:


df1.dtypes

# In[45]:


df1_cat=df1.select_dtypes(include=['object'])
df1.columns

# In[49]:


print("city::\n",df1["city"].unique(),"\n")
print("season::\n",df1["season"].unique(),"\n")
print("Team1::\n",df1["team1"].unique(),"\n")
print("Team2::\n",df1["team2"].unique(),"\n")
print("toss_winner::\n",df1["toss_winner"].unique(),"\n")
print("Toss_decision::\n",df1["toss_decision"].unique(),"\n")
print("result::\n",df1["result"].unique(),"\n")
print("dl_applied::\n",df1["dl_applied"].unique(),"\n")
print("winner::\n",df1["winner"].unique(),"\n")
print("win_by_runs::\n",df1["win_by_runs"].unique(),"\n")
print("win_by_wickets::\n",df1["win_by_wickets"].unique(),"\n")
print("player_of_match::\n",df1["player_of_match"].unique(),"\n")
print("Venue::\n",df1["venue"].unique(),"\n")

# In[53]:


df1["team1"]=df1["team1"].replace("Rising Pune Supergiant","Rising Pune Supergiants")
df1["team2"]=df1["team2"].replace("Rising Pune Supergiant","Rising Pune Supergiants")
df1["winner"]=df1["winner"].replace("Rising Pune Supergiant","Rising Pune Supergiants")
df1["toss_winner"]=df1["toss_winner"].replace("Rising Pune Supergiant","Rising Pune Supergiants")

# In[55]:


print("Team1::\n",df1["team1"].unique(),"\n")
print("Team2::\n",df1["team2"].unique(),"\n")
print("toss_winner::\n",df1["toss_winner"].unique(),"\n")

# In[62]:


df1.to_csv("IPL_matches1.csv",index=False)

# In[ ]:



