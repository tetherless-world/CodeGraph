#!/usr/bin/env python
# coding: utf-8

# **[Pandas Micro-Course Home Page](https://www.kaggle.com/learn/pandas)**
# 
# ---
# 

# # Intro
# Welcome to the **[Learn Pandas](https://www.kaggle.com/learn/pandas)** micro-course. 
# 
# You will go through a series of hands-on exercises. If you already know some Pandas, you can try the exercises without the reference materials or tutorials. But most people find it useful to open the pages listed as `relevant resources` to help you as you go through the exercise questions
# 
# The first step in most data analytics projects is reading the data file. So you will start there.
# 
# # Relevant Resources
# * **[Creating, Reading and Writing Reference](https://www.kaggle.com/residentmario/creating-reading-and-writing-reference)**
# * [General Pandas Cheat Sheet](https://assets.datacamp.com/blog_assets/PandasPythonForDataScience.pdf)
# 
# # Set Up
# 
# Run the code cell below to load libraries you will need (including code to check your answers).

# In[ ]:


import pandas as pd
pd.set_option('max_rows', 5)
from learntools.core import binder; binder.bind(globals())
from learntools.pandas.creating_reading_and_writing import *
print("Setup complete.")

# # Exercises

# ## 1.
# 
# In the cell below, create a DataFrame `fruits` that looks like this:
# 
# ![](https://i.imgur.com/Ax3pp2A.png)

# In[ ]:


# Your code goes here. Create a dataframe matching the above diagram and assign it to the variable fruits.
fruits = ____

q1.check()
fruits

# In[ ]:


#q1.hint()
#q1.solution()

# ## 2.
# 
# Create a dataframe `fruit_sales` that matches the diagram below:
# 
# ![](https://i.imgur.com/CHPn7ZF.png)

# In[ ]:


# Your code goes here. Create a dataframe matching the above diagram and assign it to the variable fruit_sales.
fruit_sales = ____

q2.check()
fruit_sales

# In[ ]:


#q2.hint()
#q2.solution()

# ## 3.
# 
# Create a variable `ingredients` with a `pd.Series` that looks like:
# 
# ```
# Flour     4 cups
# Milk       1 cup
# Eggs     2 large
# Spam       1 can
# Name: Dinner, dtype: object
# ```

# In[ ]:


ingredients = ____

q3.check()
ingredients

# In[ ]:


#q3.hint()
#q3.solution()

# ## 4.
# 
# Read the following csv dataset of wine reviews into a DataFrame called `reviews`:
# 
# ![](https://i.imgur.com/74RCZtU.png)
# 
# The filepath to the csv file is `../input/wine-reviews/winemag-data_first150k.csv`. The first few lines look like:
# 
# ```
# ,country,description,designation,points,price,province,region_1,region_2,variety,winery
# 0,US,"This tremendous 100% varietal wine[...]",Martha's Vineyard,96,235.0,California,Napa Valley,Napa,Cabernet Sauvignon,Heitz
# 1,Spain,"Ripe aromas of fig, blackberry and[...]",Carodorum Selección Especial Reserva,96,110.0,Northern Spain,Toro,,Tinta de Toro,Bodega Carmen Rodríguez
# ```

# In[ ]:


reviews = ____

q4.check()
reviews

# In[ ]:


#q4.hint()
#q4.solution()

# ## 5.
# 
# Run the cell below to create and display a DataFrame called `animals`:

# In[ ]:


animals = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])
animals

# In the cell below, write code to save this DataFrame to disk as a csv file with the name `cows_and_goats.csv`.

# In[ ]:


# Your code goes here

q5.check()

# In[ ]:


#q5.hint()
#q5.solution()

# ## Keep going
# 
# Move on to learn **[indexing, selecting and assigning](https://www.kaggle.com/residentmario/indexing-selecting-assigning-reference)**, which are probably the parts of Pandas you will use most frequently.

# ---
# **[Pandas Micro-Course Home Page](https://www.kaggle.com/learn/pandas)**
# 
# 
