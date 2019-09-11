#!/usr/bin/env python
# coding: utf-8

# **[Pandas Micro-Course Home Page](https://www.kaggle.com/learn/pandas)**
# 
# ---
# 

# In[ ]:




# # Intro
# 
# You'll select specific values of a pandas `DataFrame` or `Series` to work on in most data operations, so it's a foundational skill for data science.
# 
# You will explore the [Wine Reviews dataset](https://www.kaggle.com/zynicide/wine-reviews) while practicing this skill.
# 
# # Relevant Resources
# * **[Quickstart to indexing and selecting data](https://www.kaggle.com/residentmario/indexing-and-selecting-data/)** 
# * [Indexing and Selecting Data](https://pandas.pydata.org/pandas-docs/stable/indexing.html) section of pandas documentation
# * [Pandas Cheat Sheet](https://assets.datacamp.com/blog_assets/PandasPythonForDataScience.pdf)
# 
# 
# 

# # Set Up
# 
# Run the following cell to load your data and some utility functions (including code to check your answers).

# In[ ]:


import pandas as pd

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
pd.set_option("display.max_rows", 5)

from learntools.core import binder; binder.bind(globals())
from learntools.pandas.indexing_selecting_and_assigning import *
print("Setup complete.")

# Look at an overview of your data by running the following line

# In[ ]:


reviews.head()

# # Exercises

# ## 1.
# 
# Select the `description` column from `reviews` and assign the result to the variable `desc`.

# In[ ]:


# Your code here
desc = ____

q1.check()

# Follow-up question: what type of object is `desc`? If you're not sure, you can check by calling Python's `type` function: `type(desc)`.

# In[ ]:


#q1.hint()
#q1.solution()

# ## 2.
# 
# Select the first value from the description column of `reviews`, assigning it to variable `first_description`.

# In[ ]:


first_description = ____

q2.check()
first_description

# In[ ]:


#q2.hint()
#q2.solution()

# ## 3. 
# 
# Select the first row of data (the first record) from `reviews`, assigning it to the variable `first_row`.

# In[ ]:


first_row = ____

q3.check()
first_row

# In[ ]:


#q3.hint()
#q3.solution()

# ## 4.
# 
# Select the first 10 values from the `description` column in `reviews`, assigning the result to variable `first_descriptions`.
# 
# Hint: format your output as a `pandas` `Series`.

# In[ ]:


first_descriptions = ____

q4.check()
first_descriptions

# In[ ]:


#q4.hint()
#q4.solution()

# ## 5.
# 
# Select the records with index labels `1`, `2`, `3`, `5`, and `8`, assigning the result to the variable `sample_reviews`.
# 
# In other words, generate the following DataFrame:
# 
# ![](https://i.imgur.com/sHZvI1O.png)

# In[ ]:


sample_reviews = ____

q5.check()
sample_reviews

# In[ ]:


#q5.hint()
#q5.solution()

# ## 6.
# 
# Create a variable `df` containing the `country`, `province`, `region_1`, and `region_2` columns of the records with the index labels `0`, `1`, `10`, and `100`. In other words, generate the following `DataFrame`:
# 
# ![](https://i.imgur.com/FUCGiKP.png)

# In[ ]:


df = ____

q6.check()
df

# In[ ]:


#q6.hint()
#q6.solution()

# ## 7.
# 
# Create a variable `df` containing the `country` and `variety` columns of the first 100 records. 
# 
# Hint: you may use `loc` or `iloc`. When working on the answer this question and the several of the ones that follow, keep the following "gotcha" described in the [reference](https://www.kaggle.com/residentmario/indexing-selecting-assigning-reference) for this tutorial section:
# 
# > `iloc` uses the Python stdlib indexing scheme, where the first element of the range is included and the last one excluded. So `0:10` will select entries `0,...,9`. `loc`, meanwhile, indexes inclusively. So `0:10` will select entries `0,...,10`.
# 
# > [...]
# 
# > ...[consider] when the DataFrame index is a simple numerical list, e.g. `0,...,1000`. In this case `reviews.iloc[0:1000]` will return 1000 entries, while `reviews.loc[0:1000]` return 1001 of them! To get 1000 elements using `iloc`, you will need to go one higher and ask for `reviews.iloc[0:1001]`.

# In[ ]:


df = ____

q7.check()
df

# In[ ]:


#q7.hint()
#q7.solution()

# ## 8.
# 
# Create a DataFrame `italian_wines` containing reviews of wines made in `Italy`. Hint: `reviews.country` equals what?

# In[ ]:


italian_wines = ____

q8.check()

# In[ ]:


#q8.hint()
#q8.solution()

# ## 9.
# 
# Create a DataFrame `top_oceania_wines` containing all reviews with at least 95 points (out of 100) for wines from Australia or New Zealand.

# In[ ]:


top_oceania_wines = ____

q9.check()
top_oceania_wines

# In[ ]:


#q9.hint()
#q9.solution()

# ## Keep going
# 
# Great job. Next, learn about **[Summary functions and Map functions](https://www.kaggle.com/residentmario/summary-functions-and-maps-reference)** to get high-level insights and statistics about your data.

# ---
# **[Pandas Micro-Course Home Page](https://www.kaggle.com/learn/pandas)**
# 
# 
