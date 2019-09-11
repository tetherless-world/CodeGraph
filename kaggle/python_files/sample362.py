#!/usr/bin/env python
# coding: utf-8

# <center>
# <img src="https://habrastorage.org/files/fd4/502/43d/fd450243dd604b81b9713213a247aa20.jpg">
#     
# ## [mlcourse.ai](mlcourse.ai) – Open Machine Learning Course 
# 
# <center>Author: [Yury Kashnitskiy](http://yorko.github.io) <br>
# Translated and edited by [Sergey Isaev](https://www.linkedin.com/in/isvforall/), [Artem Trunov](https://www.linkedin.com/in/datamove/), [Anastasia Manokhina](https://www.linkedin.com/in/anastasiamanokhina/), and [Yuanyuan Pao](https://www.linkedin.com/in/yuanyuanpao/) <br>All content is distributed under the [Creative Commons CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) license.

# # <center> Assignment #1 (demo)
# ## <center>  Exploratory data analysis with Pandas
# 

# **In this task you should use Pandas to answer a few questions about the [Adult](https://archive.ics.uci.edu/ml/datasets/Adult) dataset. (You don't have to download the data – it's already here). Choose the answers in the [web-form](https://docs.google.com/forms/d/1uY7MpI2trKx6FLWZte0uVh3ULV4Cm_tDud0VDFGCOKg). This is a demo version of an assignment, so by submitting the form, you'll see a link to the solution .ipynb file.**

# Unique values of all features (for more information, please see the links above):
# - `age`: continuous.
# - `workclass`: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.
# - `fnlwgt`: continuous.
# - `education`: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
# - `education-num`: continuous.
# - `marital-status`: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.
# - `occupation`: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces.
# - `relationship`: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
# - `race`: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
# - `sex`: Female, Male.
# - `capital-gain`: continuous.
# - `capital-loss`: continuous.
# - `hours-per-week`: continuous.
# - `native-country`: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.   
# - `salary`: >50K,<=50K

# In[ ]:


import pandas as pd

# In[ ]:


data = pd.read_csv('../input/adult.data.csv')
data.head()

# **1. How many men and women (*sex* feature) are represented in this dataset?** 

# In[ ]:


# You code here

# **2. What is the average age (*age* feature) of women?**

# In[ ]:


# You code here

# **3. What is the percentage of German citizens (*native-country* feature)?**

# In[ ]:


# You code here

# **4-5. What are the mean and standard deviation of age for those who earn more than 50K per year (*salary* feature) and those who earn less than 50K per year? **

# In[ ]:


# You code here

# **6. Is it true that people who earn more than 50K have at least high school education? (*education – Bachelors, Prof-school, Assoc-acdm, Assoc-voc, Masters* or *Doctorate* feature)**

# In[ ]:


# You code here

# **7. Display age statistics for each race (*race* feature) and each gender (*sex* feature). Use *groupby()* and *describe()*. Find the maximum age of men of *Amer-Indian-Eskimo* race.**

# In[ ]:


# You code here

# **8. Among whom is the proportion of those who earn a lot (>50K) greater: married or single men (*marital-status* feature)? Consider as married those who have a *marital-status* starting with *Married* (Married-civ-spouse, Married-spouse-absent or Married-AF-spouse), the rest are considered bachelors.**

# In[ ]:


# You code here

# **9. What is the maximum number of hours a person works per week (*hours-per-week* feature)? How many people work such a number of hours, and what is the percentage of those who earn a lot (>50K) among them?**

# In[ ]:


# You code here

# **10. Count the average time of work (*hours-per-week*) for those who earn a little and a lot (*salary*) for each country (*native-country*). What will these be for Japan?**

# In[ ]:


# You code here
