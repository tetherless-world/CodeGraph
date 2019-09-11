#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # data visualization
#sns.set(style="ticks", color_codes=True)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input/dsb_2019/"))

# Any results you write to the current directory are saved as output.

# ## Pandas
# Pandas is an open-source Python Library providing high-performance data manipulation and analysis tool using its powerful data structures. The name Pandas is derived from the word Panel Data – an Econometrics from Multidimensional data.
# 
# 
# - [Python TutorialsPoint](https://www.tutorialspoint.com/python_pandas/)
# - [Pandas Official Website](https://pandas.pydata.org/)
# - [10 Mins to Pandas](https://pandas.pydata.org/pandas-docs/version/0.22/10min.html)
# - [Daniel Chen | Introduction to Pandas | PyData 2016](https://www.youtube.com/watch?v=dye7rDktJ2E)
# - [A Visual Guide To Pandas | Jason Wirth](https://www.youtube.com/watch?v=9d5-Ti6onew)
# 
# 
# ### Key Features of Pandas
# - Fast and efficient DataFrame object with default and customized indexing.
# - Tools for loading data into in-memory data objects from different file formats.
# - Data alignment and integrated handling of missing data.
# - Reshaping and pivoting of date sets.
# - Label-based slicing, indexing and subsetting of large data sets.
# - Columns from a data structure can be deleted or inserted.
# - Group by data for aggregation and transformations.
# - High performance merging and joining of data.
# - Time Series functionality.
# 
# 

# ### Reading a dataset

# In[ ]:


PATH = '../input/dsb_2019/'

product_master = pd.read_csv(PATH+'product_master.csv')
training_data = pd.read_csv(PATH+'training_data.csv')
customer_master = pd.read_csv(PATH+'customer_master.csv')
sample_submission = pd.read_csv(PATH+'sample_submission_file.csv')

# Displaying a dataset
display(product_master.head())
display(training_data.head())
display(customer_master.head())
display(sample_submission.head())

# #### Basic Panda Functionality

# In[ ]:


print("customer_master.empty : ", customer_master.empty) # Return a boolean variable. True if the dataset is empty, else False.
print("customer_master.ndim : ", customer_master.ndim) #Returns the number of dimensions.
print("customer_master.shape : ", customer_master.shape) #Returns the shape of the dataset in (rows, columns) form.
print("customer_master.axes : ", customer_master.axes) #Returns the list of the labels of the series.

# #### Dataset Summary

# In[ ]:


display(product_master.describe())
display(training_data.describe())
display(customer_master.describe())

# In[ ]:


customer_master.head()

# #### Data Cleaning
# - Substituting NA/NaN values - Using the **isna()** function to check for any Null records.
# - Making new and _cleaner_ columns - 
# - Dropping columns

# In[ ]:


print(sum(customer_master['emailDomain'].isna())) #Boolean to check number of NaN records in the emailDomain column

# In[ ]:


customer_master['clean_emailDomain'] = customer_master['emailDomain'].fillna("BLANK") #Handling NaN values by substitution
print("Number of NaN records in the emailDomain column: ", sum(customer_master['clean_emailDomain'].isna())) # All the records with NaN values have been substituted with '_' literal.

# In[ ]:


display(customer_master.drop(columns='emailDomain').head())
display(customer_master.head())

# Frequency Count for a column 

# In[ ]:


customer_master['gender'].value_counts().plot(kind='bar')

# In[ ]:


# Substituting wrong entry values 
customer_master[customer_master['gender'] == '-1'].head()
customer_master['gender'][customer_master['gender'] == '-1'] = 'Unknown' 
customer_master.head()

# **Aggregation Operations**

# In[ ]:


customer_master[['Customer_Id', 'gender']].groupby(['gender'], sort=True).count().plot(kind='bar',legend=False)

# In[ ]:


customer_master[['Customer_Id', 'gender','anon_loyalty']].groupby(['gender','anon_loyalty']).count().plot(kind='bar',)

# In[ ]:


#customer_master[['DOJ_month','DOJ_day']].plot.barh(stacked=True)

# In[ ]:


#customer_master['emailDomain'] = customer_master['emailDomain'].astype('category')

# In[ ]:




# In[ ]:



