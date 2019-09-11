#!/usr/bin/env python
# coding: utf-8

# **[Pandas Micro-Course Home Page](https://www.kaggle.com/learn/pandas)**
# 
# ---
# 

# ## Intro
# Data comes from many places, and you'll frequently need to combine multiple Series into a single DataFrame. Keeping everything organized when you combine series can require choosing the appropriate name for each series. 
# 
# You'll learn both renaming and combining in this lesson. You'll see it all with the familiar with Wine reviews dataset.

# In[ ]:


import pandas as pd
pd.set_option('max_rows', 5)
reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
reviews

# ## Renaming
# 
# Data can come with crazy column names or conventions. You'll use `pandas` renaming utility functions to change the names of the offending entries to something better.
# 
# You can do this with the `rename` method. For example, you can change the `points` column to `score` like this:

# In[ ]:


reviews.rename(columns={'points': 'score'})

# `rename` lets you rename index _or_ column values by specifying a `index` or `column` keyword parameter, respectively. It supports a variety of input formats, but I usually find a Python `dict` to be the most convenient one. Here is an example using it to rename some elements on the index.

# In[ ]:


reviews.rename(index={0: 'firstEntry', 1: 'secondEntry'})

# You'll probably rename columns very often, but rename index values very rarely.  For that, `set_index` is usually more convenient.
# 
# Both the row index and the column index can have their own `name` attribute. The complimentary `rename_axis` method may be used to change these names. For example:

# In[ ]:


reviews.rename_axis("wines", axis='rows').rename_axis("fields", axis='columns')

# ## Combining

# When performing operations on a dataset we will sometimes need to combine different `DataFrame` and/or `Series` in non-trivial ways. There are three core methods for doing this. In order of increasing complexity, these are `concat`, `join`, and `merge`. We will focus on the first two functions here.
# 
# The simplest combining method is `concat`. Given a list of elements, it will smashes those elements together along an axis.
# 
# This is useful when we have data in different `DataFrame` or `Series` objects with the same fields (columns). One example: the [YouTube Videos dataset](https://www.kaggle.com/datasnaek/youtube-new), which splits the data up based on country of origin (e.g. Canada and the UK, in this example). If we want to study multiple countries simultaneously, we can use `concat` to combine them together:

# In[ ]:


canadian_youtube = pd.read_csv("../input/youtube-new/CAvideos.csv")
british_youtube = pd.read_csv("../input/youtube-new/GBvideos.csv")

pd.concat([canadian_youtube, british_youtube])

# The `join` function combines `DataFrame` objects with a common index. For example, to pull down videos that happened to be trending on the same day in _both_ Canada and the UK, you would write:

# In[ ]:


left = canadian_youtube.set_index(['title', 'trending_date'])
right = british_youtube.set_index(['title', 'trending_date'])

left.join(right, lsuffix='_CAN', rsuffix='_UK')

# The `lsuffix` and `rsuffix` parameters are necessary here because the data has the same column names in both British and Canadian datasets. If this wasn't true (because, say, we'd renamed them beforehand) we wouldn't need them.

# ---
# **[Pandas Micro-Course Home Page](https://www.kaggle.com/learn/pandas)**
# 
# 
