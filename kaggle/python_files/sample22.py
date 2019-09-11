#!/usr/bin/env python
# coding: utf-8

# # <div style="text-align: center">The Data Scientist’s Toolbox Tutorial - 2</div>
# 
# ### <div style="text-align: center">CLEAR DATA. MADE MODEL.</div>
# <div style="text-align:center">last update: <b>30/12/2018</b></div>
# 
# 
# >###### Before starting to read this kernel, It is a good idea review first step: 
# 1. [The Data Scientist’s Toolbox Tutorial - 1](https://www.kaggle.com/mjbahmani/the-data-scientist-s-toolbox-tutorial-1)
# 2. <font color="red">You are in the second step</font>
# 3. [Mathematics and Linear Algebra](https://www.kaggle.com/mjbahmani/linear-algebra-for-data-scientists)
# 4. [Programming &amp; Analysis Tools](https://www.kaggle.com/mjbahmani/20-ml-algorithms-15-plot-for-beginners)
# 5. [Big Data](https://www.kaggle.com/mjbahmani/a-data-science-framework-for-quora)
# 6. [Data visualization](https://www.kaggle.com/mjbahmani/top-5-data-visualization-libraries-tutorial)
# 7. [Data Cleaning](https://www.kaggle.com/mjbahmani/machine-learning-workflow-for-house-prices)
# 8. [How to solve a Problem?](https://www.kaggle.com/mjbahmani/the-data-scientist-s-toolbox-tutorial-2)
# 9. [Machine Learning](https://www.kaggle.com/mjbahmani/a-comprehensive-ml-workflow-with-python)
# 10. [Deep Learning](https://www.kaggle.com/mjbahmani/top-5-deep-learning-frameworks-tutorial)
# 
# ---------------------------------------------------------------------
# You can Fork and Run this kernel on Github:
# > ###### [ GitHub](https://github.com/mjbahmani/10-steps-to-become-a-data-scientist)
# 
# -------------------------------------------------------------------------------------------------------------
# 
#  **I hope you find this kernel helpful and some <font color="red"><b>UPVOTES</b></font> would be very much appreciated**
#  
#  -----------

#  <a id="top"></a> <br>
# ## Notebook  Content
# 1. [Introduction](#1)
#     1. [Import](#2)
#     1. [Version](#3)
# 1. [NumPy](#2)
#     1. [Creating Arrays](#21)
#     1. [How We Can Combining Arrays?](#22)
#     1. [Operations](#23)
#     1. [How to use Sklearn Data Set? ](#24)
#     1. [Loading external data](#25)
#     1. [Model Deployment](#26)
#     1. [Families of ML algorithms](#27)
#     1. [Prepare Features & Targets](#28)
#     1. [Accuracy and precision](#29)
#     1. [Estimators](#210)
#     1. [Predictors](#211)
# 1. [Pandas](#3)
#     1. [DataFrame  ](#31)
#     1. [Missing values](#32)
#     1. [Merging Dataframes](#33)
#     1. [Making Code Pandorable](#34)
#     1. [Group by](#35)
#     1. [Scales](#36)
# 1. [Sklearn](#4)
#     1. [Algorithms ](#41)
# 1. [conclusion](#5)
# 1. [References](#6)

# <a id="1"></a> <br>
# # 1-Introduction
# 
# This Kernel is mostly for **beginners**, and of course, all **professionals** who think they need to review  their  knowledge.
# Also, this is  the second version for (  [The Data Scientist’s Toolbox Tutorial - 1](https://www.kaggle.com/mjbahmani/the-data-scientist-s-toolbox-tutorial-1) ) and we will continue with other important packages in this kernel.keep following!
# In this section of the tutorial, we introduce two other functional libraries that are required for each specialist.
# 1. Numpy
# 1. Pandas
# 
# This kernels is based on following perfect tutorails and I want to give them credits:
# 1. [Coursera-data-science-python](https://www.coursera.org/specializations/data-science-python)
# 1. [Sklearn](https://scikit-learn.org)
# 1. [Feature Scaling with scikit-learn](http://benalexkeen.com/feature-scaling-with-scikit-learn/)
# 1. [https://docs.scipy.org/doc/numpy/user/quickstart.html](https://docs.scipy.org/doc/numpy/user/quickstart.html)
# 1. [https://pandas.pydata.org/](https://pandas.pydata.org/)
# 1. [https://www.tutorialspoint.com/numpy](https://www.tutorialspoint.com/numpy)
# 1. [python-numpy-tutorial](https://www.datacamp.com/community/tutorials/python-numpy-tutorial)

# <a id="11"></a> <br>
# ##   1-1 Import

# In[ ]:


from pandas import get_dummies
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib
import warnings
import scipy
import numpy
import json
import sys
import csv
import os

# <a id="12"></a> <br>
# ## 1-2 Version

# In[ ]:


print('scipy: {}'.format(scipy.__version__))
print('seaborn: {}'.format(sns.__version__))
print('pandas: {}'.format(pd.__version__))
print('numpy: {}'.format(np.__version__))
print('Python: {}'.format(sys.version))

# <a id="13"></a> <br>
# ## 1-3 Setup
# 
# A few tiny adjustments for better **code readability**

# In[ ]:


warnings.filterwarnings('ignore')

# <a id="14"></a> <br>
# ## 1-4 Import DataSets

# In[ ]:


    hp_train=pd.read_csv('../input/melb_data.csv')

# <a id="2"></a> <br>
# # 2- NumPy
# Numpy is an open source library that you can with it do a lot of math operation. for checking it, you can give a visit in this [page](http://www.numpy.org/)
# 
# <img src='https://scipy-lectures.org/_images/numpy_indexing.png' width=400 heght=400>
# [**Image Credit**](https://scipy-lectures.org/intro/numpy/array_object.html)
# 
# Some of its most important features include:
# 1. stands for Numerical Python
# 1. Use for mathematical and logical operations
# 1. Operations related to linear algebra
# 1. Numpy is good  for  indexing and slicing
# 
# For a fast start you can check this [cheatsheat](https://www.datacamp.com/community/blog/python-numpy-cheat-sheet) too.

# In[ ]:


import numpy as np

# <a id="21"></a> <br>
# ## 2-1 How can I Create  Arrays?

# for Creating an Array using following command:

# In[ ]:


mylist = [1, 2, 3]
myarray = np.array(mylist)
myarray.shape

# <img src='http://community.datacamp.com.s3.amazonaws.com/community/production/ckeditor_assets/pictures/332/content_arrays-axes.png' width=500 heght=500>
# [**Image Credit**](https://www.datacamp.com/community/tutorials/python-numpy-tutorial)

# In[ ]:


myarray.shape

# <br>
# `resize` changes the shape and size of array in-place.

# In[ ]:


myarray.resize(3, 3)
myarray

# <br>
# `ones` returns a new array of given shape and type, filled with ones.

# In[ ]:


np.ones((3, 2))

# <br>
# `zeros` returns a new array of given shape and type, filled with zeros.

# In[ ]:


np.zeros((2, 3))

# <br>
# `eye` returns a 2-D array with ones on the diagonal and zeros elsewhere.

# In[ ]:


np.eye(3)

# <br>
# `diag` extracts a diagonal or constructs a diagonal array.

# In[ ]:


np.diag(myarray)

# <br>
# Create an array using repeating list (or see `np.tile`)

# In[ ]:


np.array([1, 2, 3] * 3)

# <br>
# Repeat elements of an array using `repeat`.

# In[ ]:


np.repeat([1, 2, 3], 3)

# <a id="22"></a> <br>
# ## 2-2 How We Can Combining Arrays?
# [docs.scipy.org](https://docs.scipy.org/doc/numpy/reference/generated/numpy.concatenate.html)
# ###### [Go to top](#top)

# In[ ]:


p = np.ones([2, 3], int)
p

# <br>
# Use `vstack` to stack arrays in sequence vertically (row wise).

# In[ ]:


np.vstack([p, 2*p])

# <br>
# Use `hstack` to stack arrays in sequence horizontally (column wise).

# In[ ]:


np.hstack([p, 2*p])

# <a id="23"></a> <br>
# ## 2-3 Operations
# for learning numpy operator, this good idea to get a visit in this [page](http://scipy-lectures.org/intro/numpy/operations.html)
# <img src='http://scipy-lectures.org/_images/numpy_broadcasting.png'>
# [Image Credit](http://scipy-lectures.org/intro/numpy/operations.html)
# ###### [Go to top](#top)

# Use `+`, `-`, `*`, `/` and `**` to perform element wise addition, subtraction, multiplication, division and power.

# In[ ]:


x=np.array([1, 2, 3])
y=np.array([4, 5, 6])

# In[ ]:


print(x + y) # elementwise addition     [1 2 3] + [4 5 6] = [5  7  9]
print(x - y) # elementwise subtraction  [1 2 3] - [4 5 6] = [-3 -3 -3]

# In[ ]:


print(x * y) # elementwise multiplication  [1 2 3] * [4 5 6] = [4  10  18]
print(x / y) # elementwise divison         [1 2 3] / [4 5 6] = [0.25  0.4  0.5]

# In[ ]:


print(x**2) # elementwise power  [1 2 3] ^2 =  [1 4 9]

# <br>
# **Dot Product:**  
# 
# $ \begin{bmatrix}x_1 \ x_2 \ x_3\end{bmatrix}
# \cdot
# \begin{bmatrix}y_1 \\ y_2 \\ y_3\end{bmatrix}
# = x_1 y_1 + x_2 y_2 + x_3 y_3$

# In[ ]:


x.dot(y) # dot product  1*4 + 2*5 + 3*6

# In[ ]:


z = np.array([y, y**2])
print(len(z)) # number of rows of array

# <br>
# Let's look at transposing arrays. Transposing permutes the dimensions of the array.

# In[ ]:


z = np.array([y, y**2])
z

# <br>
# The shape of array `z` is `(2,3)` before transposing.

# In[ ]:


z.shape

# <br>
# Use `.T` to get the transpose.

# In[ ]:


z.T

# <br>
# The number of rows has swapped with the number of columns.

# In[ ]:


z.T.shape

# <br>
# Use `.dtype` to see the data type of the elements in the array.

# In[ ]:


z.dtype

# <br>
# Use `.astype` to cast to a specific type.

# In[ ]:


z = z.astype('f')
z.dtype

# <a id="24"></a> <br>
# ## 2-4 Math Functions
# For learning numpy math function, this good idea to get a visit in this [page](https://www.geeksforgeeks.org/numpy-mathematical-function/)
# <img src='http://s8.picofile.com/file/8353147492/numpy_math.png'>
# [Image Credit](https://www.geeksforgeeks.org/numpy-mathematical-function/)
# 
# ###### [Go to top](#top)

# Numpy has many built in math functions that can be performed on arrays.

# In[ ]:


myarray = np.array([-4, -2, 1, 3, 5])

# In[ ]:


myarray.sum()

# In[ ]:


myarray.max()

# In[ ]:


myarray.min()

# In[ ]:


myarray.mean()

# In[ ]:


myarray.std()

# <br>
# `argmax` and `argmin` return the index of the maximum and minimum values in the array.

# In[ ]:


myarray.argmax()

# In[ ]:


myarray.argmin()

# <a id="25"></a> <br>
# 
# ## 2-5 Indexing / Slicing
# For learning numpy Indexing / Slicing , this good idea to get a visit in this [page](https://www.stechies.com/numpy-indexing-slicing/)
# <img src='http://s8.picofile.com/file/8353147750/numpy_math2.png'>
# [Image Credit](https://www.stechies.com/numpy-indexing-slicing/)
# ###### [Go to top](#top)

# In[ ]:


myarray = np.arange(13)**2
myarray

# <br>
# Use bracket notation to get the value at a specific index. Remember that indexing starts at 0.

# In[ ]:


myarray[0], myarray[4], myarray[-1]

# <br>
# Use `:` to indicate a range. `array[start:stop]`
# 
# 
# Leaving `start` or `stop` empty will default to the beginning/end of the array.

# In[ ]:


myarray[1:5]

# <br>
# Use negatives to count from the back.

# In[ ]:


myarray[-4:]

# <br>
# A second `:` can be used to indicate step-size. `array[start:stop:stepsize]`
# 
# Here we are starting 5th element from the end, and counting backwards by 2 until the beginning of the array is reached.

# In[ ]:


myarray[-5::-2]

# <br>
# Let's look at a multidimensional array.

# In[ ]:


r = np.arange(36)
r.resize((6, 6))
r

# <br>
# Use bracket notation to slice: `array[row, column]`.

# In[ ]:


r[2, 2]

# <br>
# And use : to select a range of rows or columns.

# In[ ]:


r[3, 3:6]

# <br>
# Here we are selecting all the rows up to (and not including) row 2, and all the columns up to (and not including) the last column.

# In[ ]:


r[:2, :-1]

# <br>
# This is a slice of the last row, and only every other element.

# In[ ]:


r[-1, ::2]

# <br>
# We can also perform conditional indexing. Here we are selecting values from the array that are greater than 30. (Also see `np.where`)

# In[ ]:


r[r > 30]

# <br>
# Here we are assigning all values in the array that are greater than 30 to the value of 30.
# ###### [Go to top](#top)

# In[ ]:


r[r > 30] = 30
r

# <a id="26"></a> <br>
# ## 2-6 Copying Data

# Be careful with copying and modifying arrays in NumPy!
# 
# 
# `r2` is a slice of `r`

# In[ ]:


r2 = r[:3,:3]
r2

# <br>
# Set this slice's values to zero ([:] selects the entire array)

# In[ ]:


r2[:] = 0
r2

# <br>
# `r` has also been changed!

# In[ ]:


r

# <br>
# To avoid this, use `r.copy` to create a copy that will not affect the original array

# In[ ]:


r_copy = r.copy()
r_copy

# <br>
# Now when r_copy is modified, r will not be changed.

# In[ ]:


r_copy[:] = 10
print(r_copy, '\n')
print(r)

# <a id="27"></a> <br>
# ## 2-7 Iterating Over Arrays

# Let's create a new 4 by 3 array of random numbers 0-9.

# In[ ]:


test = np.random.randint(0, 10, (4,3))
test

# <br>
# Iterate by row:

# In[ ]:


for row in test:
    print(row)

# <br>
# Iterate by index:

# In[ ]:


for i in range(len(test)):
    print(test[i])

# <br>
# Iterate by row and index:

# In[ ]:


for i, row in enumerate(test):
    print('row', i, 'is', row)

# <br>
# Use `zip` to iterate over multiple iterables.

# In[ ]:


test2 = test**2
test2

# In[ ]:


for i, j in zip(test, test2):
    print(i,'+',j,'=',i+j)

# <a id="28"></a> <br>
# ## 2-8 The Series Data Structure
# One-dimensional ndarray with axis labels (including time series)
# For learning Series Data Structure , this good idea to get a visit in this [page](https://www.kdnuggets.com/2017/01/pandas-cheat-sheet.html)
# <img src='https://www.kdnuggets.com/wp-content/uploads/pandas-02.png'>
# [Image Credit](https://www.kdnuggets.com/2017/01/pandas-cheat-sheet.html)

# In[ ]:


animals = ['Tiger', 'Bear', 'Moose']
pd.Series(animals)

# In[ ]:


numbers = [1, 2, 3]
pd.Series(numbers)

# In[ ]:


animals = ['Tiger', 'Bear', None]
pd.Series(animals)

# In[ ]:


numbers = [1, 2, None]
pd.Series(numbers)

# In[ ]:


import numpy as np
np.nan == None

# In[ ]:


np.nan == np.nan

# In[ ]:


np.isnan(np.nan)

# In[ ]:


sports = {'Archery': 'Bhutan',
          'Golf': 'Scotland',
          'Sumo': 'Japan',
          'Taekwondo': 'South Korea'}
s = pd.Series(sports)
s

# In[ ]:


s.index

# In[ ]:


s = pd.Series(['Tiger', 'Bear', 'Moose'], index=['India', 'America', 'Canada'])
s

# In[ ]:


sports = {'Archery': 'Bhutan',
          'Golf': 'Scotland',
          'Sumo': 'Japan',
          'Taekwondo': 'South Korea'}
s = pd.Series(sports, index=['Golf', 'Sumo', 'Hockey'])
s

# <a id="29"></a> <br>
# # 2-9 Querying a Series

# In[ ]:


sports = {'Archery': 'Bhutan',
          'Golf': 'Scotland',
          'Sumo': 'Japan',
          'Taekwondo': 'South Korea'}
s = pd.Series(sports)
s

# In[ ]:


s.iloc[3]

# In[ ]:


s.loc['Golf']

# In[ ]:


s[3]

# In[ ]:


s['Golf']

# In[ ]:


sports = {99: 'Bhutan',
          100: 'Scotland',
          101: 'Japan',
          102: 'South Korea'}
s = pd.Series(sports)

# In[ ]:


s = pd.Series([100.00, 120.00, 101.00, 3.00])
s

# In[ ]:


total = 0
for item in s:
    total+=item
print(total)

# In[ ]:


total = np.sum(s)
print(total)

# In[ ]:


#this creates a big series of random numbers
s = pd.Series(np.random.randint(0,1000,10000))
s.head()

# In[ ]:


len(s)

# In[ ]:


summary = 0
for item in s:
    summary+=item

# In[ ]:


summary = np.sum(s)

# In[ ]:


s+=2 #adds two to each item in s using broadcasting
s.head()

# In[ ]:


for label, value in s.iteritems():
    s.set_value(label, value+2)
s.head()

# In[ ]:


s = pd.Series(np.random.randint(0,1000,100))
for label, value in s.iteritems():
    s.loc[label]= value+2

# In[ ]:


s = pd.Series(np.random.randint(0,1000,100))
s+=2


# In[ ]:


s = pd.Series([1, 2, 3])
s.loc['Animal'] = 'Bears'
s

# <a id="210"></a> <br>
# ## 2-10 Distributions in Numpy
# ###### [Go to top](#top)

# In[ ]:


np.random.binomial(1, 0.5)

# In[ ]:


np.random.binomial(1000, 0.5)/1000

# In[ ]:


chance_of_tornado = 0.01/100
np.random.binomial(100000, chance_of_tornado)

# In[ ]:


chance_of_tornado = 0.01

tornado_events = np.random.binomial(1, chance_of_tornado, 1000000)
    
two_days_in_a_row = 0
for j in range(1,len(tornado_events)-1):
    if tornado_events[j]==1 and tornado_events[j-1]==1:
        two_days_in_a_row+=1

print('{} tornadoes back to back in {} years'.format(two_days_in_a_row, 1000000/365))

# In[ ]:


np.random.uniform(0, 1)

# In[ ]:


np.random.normal(0.75)

# In[ ]:


distribution = np.random.normal(0.75,size=1000)

np.sqrt(np.sum((np.mean(distribution)-distribution)**2)/len(distribution))

# In[ ]:


np.std(distribution)

# <a id="3"></a> <br>
# ## 3- Pandas
# Pandas is capable of many tasks including  [Based on this [page](https://medium.com/dunder-data/how-to-learn-pandas-108905ab4955)]:
# 
# 1. Reading/writing many different data formats
# 1. Selecting subsets of data
# 1. Calculating across rows and down columns
# 1. Finding and filling missing data
# 1. Applying operations to independent groups within the data
# 1. Reshaping data into different forms
# 1. Combing multiple datasets together
# 1. Advanced time-series functionality
# 1. Visualization through matplotlib and seaborn
# 
# ###### [Go to top](#top)

# In[ ]:



purchase_1 = pd.Series({'Name': 'Chris',
                        'Item Purchased': 'Dog Food',
                        'Cost': 22.50})
purchase_2 = pd.Series({'Name': 'Kevyn',
                        'Item Purchased': 'Kitty Litter',
                        'Cost': 2.50})
purchase_3 = pd.Series({'Name': 'Vinod',
                        'Item Purchased': 'Bird Seed',
                        'Cost': 5.00})
df = pd.DataFrame([purchase_1, purchase_2, purchase_3], index=['Store 1', 'Store 1', 'Store 2'])
df.head()

# In[ ]:


df.loc['Store 2']

# In[ ]:


type(df.loc['Store 2'])

# In[ ]:


df.loc['Store 1']

# In[ ]:


df.loc['Store 1', 'Cost']

# In[ ]:


df.T

# In[ ]:


df.T.loc['Cost']

# In[ ]:


df['Cost']

# In[ ]:


df.loc['Store 1']['Cost']

# In[ ]:


df.loc[:,['Name', 'Cost']]

# In[ ]:


df.drop('Store 1')

# In[ ]:


df

# In[ ]:


copy_df = df.copy()
copy_df = copy_df.drop('Store 1')
copy_df

# In[ ]:


copy_df.drop

# In[ ]:


del copy_df['Name']
copy_df

# In[ ]:


df['Location'] = None
df

# In[ ]:


costs = df['Cost']
costs

# In[ ]:


costs+=2
costs

# In[ ]:


df

# <a id="31"></a> <br>
# # 3-1 Dataframe
# 
# As a Data Scientist, you'll often find that the data you need is not in a single file. It may be spread across a number of text files, spreadsheets, or databases. You want to be able to import the data of interest as a collection of DataFrames and figure out how to combine them to answer your central questions.
# ###### [Go to top](#top)

# In[ ]:


df = pd.read_csv('../input/melb_data.csv')
df.head()

# In[ ]:


df.columns

# In[ ]:


# Querying a DataFrame

# In[ ]:


df['Price'] > 10000000

# In[ ]:


only_SalePrice = df.where(df['Price'] > 0)
only_SalePrice.head()

# In[ ]:


only_SalePrice['Price'].count()

# In[ ]:


df['Price'].count()

# In[ ]:


only_SalePrice = only_SalePrice.dropna()
only_SalePrice.head()

# In[ ]:


only_SalePrice = df[df['Price'] > 0]
only_SalePrice.head()

# In[ ]:


len(df[(df['Price'] > 0) | (df['Price'] > 0)])

# In[ ]:


df[(df['Price'] > 0) & (df['Price'] == 0)]

# <a id="311"></a> <br>
# ## 3-1-1 Dataframes

# In[ ]:


df.head()

# In[ ]:


df['SalePrice'] = df.index
df = df.set_index('SalePrice')
df.head()

# In[ ]:



df = df.reset_index()
df.head()

# <a id="32"></a> <br>
# # 3-2 Missing values
# 

# In[ ]:


df = pd.read_csv('../input/melb_data.csv')

# In[ ]:


df.fillna

# In[ ]:


df = df.fillna(method='ffill')
df.head()

# <a id="33"></a> <br>
# # 3-3 Merging Dataframes
# For learning Merging Dataframes , this is a good idea to give a visit in this [page](https://www.ryanbaumann.com/blog/2016/4/30/python-pandas-tosql-only-insert-new-rows)
# <img src='https://static1.squarespace.com/static/54bb1957e4b04c160a32f928/t/5724fd0bf699bb5ad6432150/1462041871236/?format=750w'>
# [Image Credit](https://www.ryanbaumann.com/blog/2016/4/30/python-pandas-tosql-only-insert-new-rows)
# 

# In[ ]:


df = pd.DataFrame([{'Name': 'MJ', 'Item Purchased': 'Sponge', 'Cost': 22.50},
                   {'Name': 'Kevyn', 'Item Purchased': 'Kitty Litter', 'Cost': 2.50},
                   {'Name': 'Filip', 'Item Purchased': 'Spoon', 'Cost': 5.00}],
                  index=['Store 1', 'Store 1', 'Store 2'])
df

# In[ ]:


df['Date'] = ['December 1', 'January 1', 'mid-May']
df

# In[ ]:


df['Delivered'] = True
df

# In[ ]:


df['Feedback'] = ['Positive', None, 'Negative']
df

# In[ ]:


adf = df.reset_index()
adf['Date'] = pd.Series({0: 'December 1', 2: 'mid-May'})
adf

# <a id="34"></a> <br>
# # 3-4 Making Code Pandorable
# based on this amazing **[Article](https://www.datacamp.com/community/tutorials/pandas-idiomatic)**
# 1. Indexing with the help of **loc** and **iloc**, and a short introduction to querying your DataFrame with query();
# 1. Method Chaining, with the help of the pipe() function as an alternative to nested functions;
# 1. Memory Optimization, which you can achieve through setting data types;
# 1. groupby operation, in the naive and Pandas way; and
# 1. Visualization of your DataFrames with Matplotlib and Seaborn.

# In[ ]:


df = pd.read_csv('../input/melb_data.csv')

# In[ ]:


df.head()

# <a id="35"></a> <br>
# ## 3-5 Group by

# In[ ]:


df = df[df['Price']>500000]
df

# <a id="36"></a> <br>
# ## 3-6 Scales
# 

# In[ ]:


df = pd.DataFrame(['A+', 'A', 'A-', 'B+', 'B', 'B-', 'C+', 'C', 'C-', 'D+', 'D'],
                  index=['excellent', 'excellent', 'excellent', 'good', 'good', 'good', 'ok', 'ok', 'ok', 'poor', 'poor'])
df.rename(columns={0: 'Grades'}, inplace=True)
df

# In[ ]:


df['Grades'].astype('category').head()

# In[ ]:


grades = df['Grades'].astype('category',
                             categories=['D', 'D+', 'C-', 'C', 'C+', 'B-', 'B', 'B+', 'A-', 'A', 'A+'],
                             ordered=True)
grades.head()

# In[ ]:


grades > 'C'

# <a id="361"></a> <br>
# ## 3-6-1 Select

# To select rows whose column value equals a scalar, some_value, use ==:

# In[ ]:


df.loc[df['Grades'] == 'A+']


# To select rows whose column value is in an iterable, some_values, use **isin**:

# In[ ]:


df_test = pd.DataFrame({'A': [1, 2, 3], 'B': [1, 4, 7]})
df_test.isin({'A': [1, 3], 'B': [4, 7, 12]})

# Combine multiple conditions with &:

# In[ ]:


df.loc[(df['Grades'] == 'A+') & (df['Grades'] == 'D')]


# To select rows whose column value does not equal some_value, use !=:
# 

# In[ ]:



df.loc[df['Grades'] != 'B+']


# isin returns a boolean Series, so to select rows whose value is not in some_values, negate the boolean Series using ~:
# 

# In[ ]:


df_test = pd.DataFrame({'A': [1, 2, 3], 'B': [1, 4, 7]})

# In[ ]:


df_test.loc[~df_test['A'].isin({'A': [1, 3], 'B': [4, 7, 12]})]

# <a id="37"></a> <br>
# ## 3-7 Date Functionality
# ###### [Go to top](#top)

# <a id="371"></a> <br>
# ### 3-7-1 Timestamp

# In[ ]:


pd.Timestamp('9/1/2016 10:05AM')

# <a id="372"></a> <br>
# ### 3-7-2 Period

# In[ ]:


pd.Period('1/2016')

# In[ ]:


pd.Period('3/5/2016')

# <a id="373"></a> <br>
# ### 3-7-3 DatetimeIndex

# In[ ]:


t1 = pd.Series(list('abc'), [pd.Timestamp('2016-09-01'), pd.Timestamp('2016-09-02'), pd.Timestamp('2016-09-03')])
t1

# In[ ]:


type(t1.index)

# <a id="374"></a> <br>
# ### 3-7-4 PeriodIndex

# In[ ]:


t2 = pd.Series(list('def'), [pd.Period('2016-09'), pd.Period('2016-10'), pd.Period('2016-11')])
t2

# In[ ]:


type(t2.index)

# <a id="38"></a> <br>
# ## 3-8 Converting to Datetime

# In[ ]:


d1 = ['2 June 2013', 'Aug 29, 2014', '2015-06-26', '7/12/16']
ts3 = pd.DataFrame(np.random.randint(10, 100, (4,2)), index=d1, columns=list('ab'))
ts3

# In[ ]:


ts3.index = pd.to_datetime(ts3.index)
ts3

# In[ ]:


pd.to_datetime('4.7.12', dayfirst=True)

# In[ ]:


pd.Timestamp('9/3/2016')-pd.Timestamp('9/1/2016')

# <a id="381"></a> <br>
# ### 3-8-1 Timedeltas

# In[ ]:


pd.Timestamp('9/3/2016')-pd.Timestamp('9/1/2016')

# In[ ]:


pd.Timestamp('9/2/2016 8:10AM') + pd.Timedelta('12D 3H')

# <a id="382"></a> <br>
# ### 3-8-2 Working with Dates in a Dataframe
# 

# In[ ]:


dates = pd.date_range('10-01-2016', periods=9, freq='2W-SUN')
dates

# In[ ]:


df.index.ravel

# 
# 

# <a id="4"></a> <br>
# # 4- Sklearn 
# [sklearn has following feature](https://scikit-learn.org/stable/):
# 1. Simple and efficient tools for data mining and data analysis
# 1. Accessible to everybody, and reusable in various contexts
# 1. Built on NumPy, SciPy, and matplotlib
# 1. Open source, commercially usable - BSD license

# <a id="41"></a> <br>
# ## 4-1 Algorithms
# 
# **Supervised learning**:
# 
# 1. Linear models (Ridge, Lasso, Elastic Net, ...)
# 1. Support Vector Machines
# 1. Tree-based methods (Random Forests, Bagging, GBRT, ...)
# 1. Nearest neighbors 
# 1. Neural networks (basics)
# 1. Gaussian Processes
# 1. Feature selection

# **Unsupervised learning**:
# 
# 1. Clustering (KMeans, Ward, ...)
# 1. Matrix decomposition (PCA, ICA, ...)
# 1. Density estimation
# 1. Outlier detection

# __Model selection and evaluation:__
# 
# 1. Cross-validation
# 1. Grid-search
# 1. Lots of metrics
# 
# _... and many more!_ (See our [Reference](http://scikit-learn.org/dev/modules/classes.html))

# For learning this section please give a visit on [this kernel](https://www.kaggle.com/mjbahmani/20-ml-algorithms-15-plot-for-beginners)

# <a id="7"></a> <br>
# # 7- conclusion
# After the first version of this kernel, in the second edition, we introduced Numpy & Pandas. in addition, we examined each one in detail. This kernel is finished due to the large size and you can follow the discussion in the my other kernel.

# >###### you may  be interested have a look at it: [**10-steps-to-become-a-data-scientist**](https://github.com/mjbahmani/10-steps-to-become-a-data-scientist)
# 
# 
# ---------------------------------------------------------------------
# you can Fork and Run this kernel on Github:
# > ###### [ GitHub](https://github.com/mjbahmani/10-steps-to-become-a-data-scientist)
# 
# -------------------------------------------------------------------------------------------------------------
# 
#  **I hope you find this kernel helpful and some <font color="red"><b>UPVOTES</b></font> would be very much appreciated**
#  
#  -----------

# <a id="8"></a> <br>
# # 8- References & Credits
# 1. [Coursera](https://www.coursera.org/specializations/data-science-python)
# 1. [Sklearn](https://scikit-learn.org)
# 1. [Feature Scaling with scikit-learn](http://benalexkeen.com/feature-scaling-with-scikit-learn/)
# 1. [https://docs.scipy.org/doc/numpy/user/quickstart.html](https://docs.scipy.org/doc/numpy/user/quickstart.html)
# 1. [https://pandas.pydata.org/](https://pandas.pydata.org/)
# 1. [https://www.stechies.com/numpy-indexing-slicing/](https://www.stechies.com/numpy-indexing-slicing/)
# 1. [python_pandas_dataframe](https://www.tutorialspoint.com/python_pandas/python_pandas_dataframe.htm)
# ###### [Go to top](#top)

# **you may be interested have a look at it: [Course Home Page](https://www.kaggle.com/mjbahmani/10-steps-to-become-a-data-scientist)**
