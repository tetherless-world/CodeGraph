#!/usr/bin/env python
# coding: utf-8

# # Students performance in exams
# #### Marks secured by the students in college
# 
# ## Aim
# #### To understand the influence of various factors like economic, personal and social on the students performance 
# 
# ## Inferences would be : 
# #### 1. How to imporve the students performance in each test ?
# #### 2. What are the major factors influencing the test scores ?
# #### 3. Effectiveness of test preparation course?
# #### 4. Other inferences 

# 

# #### Import the required libraries

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# #### Let us initialize the required values ( we will use them later in the program )
# #### we will set the minimum marks to 40 to pass in a exam

# In[ ]:


passmark = 40

# #### Let us read the data from the csv file

# In[ ]:


df = pd.read_csv("../input/StudentsPerformance.csv")

# #### We will print top few rows to understand about the various data columns

# In[ ]:


df.head()

# #### Size of data frame

# In[ ]:


print (df.shape)

# #### Let us understand about the basic information of the data, like min, max, mean and standard deviation etc.

# In[ ]:


df.describe()

# #### Let us check for any missing values

# In[ ]:


df.isnull().sum()

# ##### As seen above, there are no missing ( null ) values in this dataframe but in real scenarios we need work on dataset with a lot of missing values  

# ####  Let us explore the Math Score first

# In[ ]:


p = sns.countplot(x="math score", data = df, palette="muted")
_ = plt.setp(p.get_xticklabels(), rotation=90) 

# #### How many students passed in Math exam ?

# In[ ]:


df['Math_PassStatus'] = np.where(df['math score']<passmark, 'F', 'P')
df.Math_PassStatus.value_counts()

# In[ ]:


p = sns.countplot(x='parental level of education', data = df, hue='Math_PassStatus', palette='bright')
_ = plt.setp(p.get_xticklabels(), rotation=90) 

# #### Let us explore the Reading score

# In[ ]:


sns.countplot(x="reading score", data = df, palette="muted")
plt.show()

# #### How many studends passed in reading ?

# In[ ]:


df['Reading_PassStatus'] = np.where(df['reading score']<passmark, 'F', 'P')
df.Reading_PassStatus.value_counts()

# In[ ]:


p = sns.countplot(x='parental level of education', data = df, hue='Reading_PassStatus', palette='bright')
_ = plt.setp(p.get_xticklabels(), rotation=90) 

# #### Let us explore writing score

# In[ ]:


p = sns.countplot(x="writing score", data = df, palette="muted")
_ = plt.setp(p.get_xticklabels(), rotation=90) 

# #### How many students passed writing ?

# In[ ]:


df['Writing_PassStatus'] = np.where(df['writing score']<passmark, 'F', 'P')
df.Writing_PassStatus.value_counts()

# In[ ]:


p = sns.countplot(x='parental level of education', data = df, hue='Writing_PassStatus', palette='bright')
_ = plt.setp(p.get_xticklabels(), rotation=90) 

# #### Iet us check "How many students passed in all the subjects ?"

# In[ ]:


df['OverAll_PassStatus'] = df.apply(lambda x : 'F' if x['Math_PassStatus'] == 'F' or 
                                    x['Reading_PassStatus'] == 'F' or x['Writing_PassStatus'] == 'F' else 'P', axis =1)

df.OverAll_PassStatus.value_counts()

# In[ ]:


p = sns.countplot(x='parental level of education', data = df, hue='OverAll_PassStatus', palette='bright')
_ = plt.setp(p.get_xticklabels(), rotation=90) 

# #### Find the percentage of marks

# In[ ]:


df['Total_Marks'] = df['math score']+df['reading score']+df['writing score']
df['Percentage'] = df['Total_Marks']/3

# In[ ]:


p = sns.countplot(x="Percentage", data = df, palette="muted")
_ = plt.setp(p.get_xticklabels(), rotation=0) 

# 

# #### Let us assign the grades
# 
# ### Grading 
# ####    above 80 = A Grade
# ####      70 to 80 = B Grade
# ####      60 to 70 = C Grade
# ####      50 to 60 = D Grade
# ####      40 to 50 = E Grade
# ####    below 40 = F Grade  ( means Fail )
# 

# In[ ]:


def GetGrade(Percentage, OverAll_PassStatus):
    if ( OverAll_PassStatus == 'F'):
        return 'F'    
    if ( Percentage >= 80 ):
        return 'A'
    if ( Percentage >= 70):
        return 'B'
    if ( Percentage >= 60):
        return 'C'
    if ( Percentage >= 50):
        return 'D'
    if ( Percentage >= 40):
        return 'E'
    else: 
        return 'F'

df['Grade'] = df.apply(lambda x : GetGrade(x['Percentage'], x['OverAll_PassStatus']), axis=1)

df.Grade.value_counts()

# #### we will plot the grades obtained in a order

# In[ ]:


sns.countplot(x="Grade", data = df, order=['A','B','C','D','E','F'],  palette="muted")
plt.show()

# In[ ]:


p = sns.countplot(x='parental level of education', data = df, hue='Grade', palette='bright')
_ = plt.setp(p.get_xticklabels(), rotation=90) 

# In[ ]:



