#!/usr/bin/env python
# coding: utf-8

# **[Pandas Micro-Course Home Page](https://www.kaggle.com/learn/pandas)**
# 
# ---
# 

# You'll probably be shocked how useful `groupby` is as you work with data in practical applications. This lesson will show you how to use it with examples from the wine review dataset.

# In[ ]:


import pandas as pd
reviews = pd.read_csv("../input/winemag-data-130k-v2.csv", index_col=0)
pd.set_option("display.max_rows", 5)

# ## Grouping
# 
# Grouping can be used for many purposes, but it may help to start with a single example to see what it does. You've already used the `value_counts` function. We can replicate what `value_counts` does using `groupby` by doing the following:

# In[ ]:


reviews.groupby('points').points.count()

# `groupby` created groups of reviews with the same point values. Then, for each group, it counted the size of the group.
# 
# In this case, `count` was the "aggregation" for each group. But there are others. For example, to get the cheapest wine in each point value category, we can do the following:

# In[ ]:


reviews.groupby('points').price.min()

# In[ ]:


reviews.head()

# You can think of each group we generate as being a slice of our `DataFrame` containing only data with values that match. This `DataFrame` is accessible to us directly using the `apply` method, and we can then manipulate the data in any way we see fit. For example, here's one way of selecting the name of the first wine reviewed from each winery in the dataset:

# In[ ]:


reviews.groupby('winery').apply(lambda df: df.title.iloc[0])

# For even more fine-grained control, you can also group by more than one column. For an example, here's how we would pick out the best wine by country _and_ province:

# In[ ]:


reviews.groupby(['country', 'province']).apply(lambda df: df.loc[df.points.argmax()])

# Another `groupby` method worth mentioning is `agg`, which lets you run a bunch of different functions on your `DataFrame` simultaneously. For example, we can generate a simple statistical summary of the dataset as follows:

# In[ ]:


reviews.groupby(['country']).price.agg([len, min, max])

# Effective use of `groupby` will allow you to do lots of really powerful things with your dataset.

# ## Multi-indexes
# 
# The examples you've seen so far worked with `DataFrame` or `Series` objects with a single-label index. `groupby` is slightly different because it will sometimes result in a "multi-index."
# 
# A multi-index has multiple levels. For example:

# In[ ]:


countries_reviewed = reviews.groupby(['country', 'province']).description.agg([len])
countries_reviewed

# Multi-indices have several methods for dealing with their tiered structure which are absent for single-level indices. They also require two levels of labels to retrieve a value, an operation that looks something like this. Dealing with multi-index output is a common "gotcha" for users new to `pandas`.
# 
# The use cases for a `MultiIndex` are detailed alongside detailed instructions on using them in the [MultiIndex / Advanced Selection](https://pandas.pydata.org/pandas-docs/stable/advanced.html) section of the `pandas` documentation.
# 
# But the `MultiIndex` method you will use most often is the one for converting back to a regular index, the `reset_index` method:

# In[ ]:


countries_reviewed.reset_index()

# ## Sorting
# 
# Grouping returns data in index order, not in value order. That is to say, when outputting the result of a `groupby`, the order of the rows is dependent on the values in the index, not the data.
# 
# But you can sort the data with the `sort_values` method.

# In[ ]:


countries_reviewed = countries_reviewed.reset_index()
countries_reviewed.sort_values(by='len')

# `sort_values` defaults to an ascending sort, where the lowest values go first. Most of the time we want a descending sort however, where the higher numbers go first. That goes thusly:

# In[ ]:


countries_reviewed.sort_values(by='len', ascending=False)

# To sort by index values, use the companion method `sort_index`. This method has the same arguments and default order:

# In[ ]:


countries_reviewed.sort_index()

# Finally, know that you can sort by more than one column at a time:

# In[ ]:


countries_reviewed.sort_values(by=['country', 'len'])

# ---
# **[Pandas Micro-Course Home Page](https://www.kaggle.com/learn/pandas)**
# 
# 
