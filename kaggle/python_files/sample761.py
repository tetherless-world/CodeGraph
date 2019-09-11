#!/usr/bin/env python
# coding: utf-8

# ![app](https://i.imgur.com/jKtMJgg.jpg)
# 
# The missing data within a dataset can often provide insight into the issue at hand. We can look at the structure of the missing values - which features are affected, which records are affected, and differences between groups. We can also use the missing data as a feature itself by counting missing values or transforming them. In this report I explore some patterns and suggest one way to improve your predictive.
# 
# ## Patterns
# First let's look at the overall pattern of missing data. The [missingno](https://github.com/ResidentMario/missingno) package by [Aleksey Bilogur](https://www.kaggle.com/residentmario) is the perfect tool here. Looking at a sample of data for all columns we see a group of columns where the missing values appear correlated.
# 
# 

# In[ ]:


import numpy as np
import pandas as pd
import missingno as msno

train = pd.read_csv('../input/application_train.csv')
msno.matrix(train.sample(500), inline=True, sparkline=True, figsize=(20,10), sort=None, color=(0.25, 0.45, 0.6))

# Zooming in on the middle columns we see they deal mostly with information about the building where the client lives. It appears there are many applicants who leave blank the information for their housing. We can think about why that might be the case or how it might inform our model.
# 
# I'll sort the data this time to better see the proportions.

# In[ ]:


msno.matrix(train.iloc[0:100, 40:94], inline=True, sparkline=True, figsize=(20,10), sort='ascending', fontsize=12, labels=True, color=(0.25, 0.45, 0.6))

# The dendrogram view shows how missing values are related across columns by using hierarchical clustering. Pretty cool! 

# In[ ]:


msno.dendrogram(train, inline=True, fontsize=12, figsize=(20,30))

# ## Comparison of Completed Applications
# 
# With an idea of the overall picture, let's now focus on the large group of applications with missing house data. Is there a difference in mean default rates between those with house information and those without?

# In[ ]:


train['incomplete'] = 1
train.loc[train.isnull().sum(axis=1) < 35, 'incomplete'] = 0

mean_c = np.mean(train.loc[train['incomplete'] == 0, 'TARGET'].values)
mean_i = np.mean(train.loc[train['incomplete'] == 1, 'TARGET'].values)
print('default ratio for more complete: {:.1%} \ndefault ratio for less complete: {:.1%}'.format(mean_c, mean_i))

# There appears to be a difference. Viewed one way, borrowers with incomplete applications are ~30% more likely to default. You may want to include this information in your model. I found it helpful to add a binary feature called 'no_housing_info'. The application is flagged if it has more than 45 blanks. You could also create three classes to account for the applications with some housing data (which may denote apartment dwellers). 
# 
# 
# ## Statistical Significance
# To be thorough, I looked at statistical significance of the difference in default rates between groups. I used a [G-test](https://en.wikipedia.org/wiki/G-test) which is similar to Pearson's chi-squared test. Either one should work in this case, but I generally prefer the G-test.

# In[ ]:


from scipy.stats import chi2_contingency

props = pd.crosstab(train.incomplete, train.TARGET)
c = chi2_contingency(props, lambda_="log-likelihood")
print(props, "\n p-value= ", c[1])

# "If p is low, the null must go." The p-value here is 1e-114 which is pretty much 0. So we can reject the null hypothesis with only a small probability of [Type 1 error](https://en.wikipedia.org/wiki/Type_I_and_type_II_errors). In other words, the difference in default ratios between the two groups is not due to random chance. 
# 
# Good luck!

# In[ ]:



