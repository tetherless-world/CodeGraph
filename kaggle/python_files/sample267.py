#!/usr/bin/env python
# coding: utf-8

# # I CHOOSE KAGGLE + DATA SCIENCE.

# <img src='https://drive.google.com/uc?id=1jLLEThALBvFGa01gvpQkNnbvlJUml1g2' width=800 >

# <img src='https://drive.google.com/uc?id=1rUzGRQ__HvXBUGK1CH4DuxCCH-NxXXuf' width=800>

# ## All the graphs are interactive

# ## ABOUT NOTEBOOK:-
# * After survey conducted by kaggle and I am as an enthusiat to learn Data Science, started thinking to do analysis on their dataset.
# * This thing encouraged me to make this notebook.
# * In this notebook I am analysing kaggle survey dataset by forming questions and then answering them.
# * The questions which I am analysing are given below:-
# #### Basic analysis:-
# * Q1). Ratio of gender in survey.
# * Q2). Ages of maximum of respondants.
# * Q3). Top 10 countries participating in survey.
# * Q4). Top 5 formal educations of respondants.
# * Q5). Top 10 Current Job Title.
# * Q6). Top 10 Industries of respondants.
# * Q7). Average years of experience of people who know about data science.
# * Q8). Current Yearly compensation of respondants.
# * Q9). Important Activities.
# ## Data Science Analysis:-
# * Q10). Current Employers incorporate Machine Learning or not.
# * Q11). Top 10 IDE.
# * Q12). Online Notebooks which are highly used people.
# *  Q13). Top 10 Programming Languages.
# *  Q14). Top 10 machine Learning libraries.
# * Q15). Top 10 Libraries for Visualization.
# * Q16). How long have people been writing code to analysis the data.
# * Q17). Top 10 Machine Learning products that are used at work.
# * Q18). Top 10 Big Data and Analytics Products.
# * Q19). Top 10 types of Data people currently interact.
# * Q20). Top 10 places to find Pubic Datasets. 
# * Q21). Top 10  Online Platforms to learn Data Science.
# * Q22). Top 10 Websites for Data Science News.
# * Q23). People view on the quality of Online learning and  bootcamp.
# * Q24). Academic Achievement VS Independent projects for Data Science.
# 

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from plotly.offline import init_notebook_mode, iplot, download_plotlyjs
import plotly.graph_objs as go
import cufflinks as cf
init_notebook_mode(connected=True)
cf.go_offline()
warnings.filterwarnings('ignore')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

# In[2]:


# Reading dataset in this form, so that I can use it on my laptop also.
try:
    df_mcr = pd.read_csv('../input/multipleChoiceResponses.csv')
    df_schema = pd.read_csv('../input/SurveySchema.csv')
except Exception as e:
    pass

# In[3]:


# Basic information about Dataset.
print(df_mcr.shape)
print("As one can see there are 395 columns.")

# In[4]:


# Let's see the details about the null values.
null = df_mcr.isnull().sum()
print(null[:30])


# * As one can see there are so many null values in columns but we do not have to remove them, as they  are 
# * part of multiple choice questions.

# In[5]:


df_mcr.head()

# # BASIC ANALYSIS

# ### Q1). Ratio of gender in survey.

# In[6]:


gender = df_mcr['Q1'].value_counts().reset_index()
gender.iplot(kind='pie', labels='index', values='Q1',title='Ration of Gender', pull=0.2, hole=0.2 )

# * More than 80% of respondences are male.

# #### ====================================================================================================

# ### Q2). Ages of maximum of respondants.
# 

# In[7]:


age = df_mcr['Q2'].value_counts().reset_index()
age.iplot(kind='bar', x='index', y='Q2', title='Age of respondants', xTitle='Age',
          yTitle='Number of responses', colors='deepskyblue')

# * Maximum number of people are in age group of 25 to 35.
# * As this is our demographic dividents.

# #### ===================================================================================================

# ### Q3). Top 10 countries participating in survey.

# In[8]:


country = df_mcr['Q3'].value_counts().reset_index()[:10]
country.iplot(kind='bar', x='index', y='Q3', 
              title='Top 10 Countries participated in survey', xTitle='Country', yTitle='Number of Respondants')
country.drop([3], axis=0, inplace=True)
country
values = country['Q3'].values
#print(values)
name = country['index'].values
#print(name)
code = ['USA','IND','CHN','RUS','BRA','DEU','GBR','CAN','FRA']
data = dict(
        type = 'choropleth',
        locations = code,
        z = values,
        text = name,
        colorbar = {'title' : 'Number of Participants'},
      ) 
layout = dict(
    title = 'Country Wise Users',
    geo = dict(
        showframe = False,
        projection = {'type':'natural earth'}
    )
)
choromap = go.Figure(data = [data],layout = layout)
iplot(choromap)

# #### Observations:-
# * Maximum number of responses are from USA and India.
# * It indicates that maximum number of people who know Kaggle platforms are from USA and India.
# * Maximum number of people who are learning Data Science are from USA and India.
# * It's good to see India at this place.

# #### ==================================================================================================

# ### Q4). Top 5 formal educations of respondants.

# In[9]:


degree = df_mcr['Q4'].value_counts().reset_index()
degree.iplot(kind='bar',x='index', y='Q4', title='Top formal educations', xTitle='Degree', 
             yTitle='Frequency', colors='deepskyblue')

# #### ===================================================================================================

# ### Q5). Top 10 Current Job Title.

# In[10]:


titles = df_mcr['Q6'].value_counts().reset_index()[:10]
titles.iplot(kind='bar',x='index', y='Q6', title='Top Current Titles', 
             xTitle='Title', yTitle='Frequency',colors='green')

# #### =================================================================================================

# ### Q6). Top 10 Industries of respondants.

# In[11]:


industry = df_mcr['Q7'].value_counts().reset_index()[:10]
industry.iplot(kind='bar',x='index', y='Q7', title='Top 10 Current industry', 
             xTitle='Industry', yTitle='Frequency',colors='indigo')

# #### Observations:-
# * Many respondants from students are also there.
# * Till now data science is widely used by computer, education, or finanace industries.
# * Government Sectors are far away from using data science, they should start to use data science otherwise they will lack behind from other Industries.****

# #### ==================================================================================================

# ### Q7). Average years of experience of people who know about data science.

# In[12]:


experience = df_mcr['Q8'].value_counts().reset_index()[:10]
experience.iplot(kind='bar',x='index', y='Q8', title='Average Years of experience', 
             xTitle='Years', yTitle='Frequency',colors='indianred')

# #### Observations:-
# * Maximum number of people are new in this field with experience of 1 to 3 years.
# * I think I choose data science at a correct time.

# #### ====================================================================================================

# ### Q8). Current Yearly compensation of respondants.

# In[13]:


compensation = df_mcr['Q9'].value_counts().reset_index()[:10]
compensation.iplot(kind='bar',x='index', y='Q9', title='Average compensation', 
             xTitle='Amount of compensation', yTitle='Frequency',colors='indianred')


# #### =====================================================================================================

# ### Q9). Important Activities.

# In[14]:


temp_df = df_mcr.drop([0],axis=0)
temp_df.head()

p1,p2,p3 = temp_df.loc[:,'Q11_Part_1'],temp_df.loc[:,'Q11_Part_2'],temp_df.loc[:,'Q11_Part_3'],
p4,p5  = temp_df.loc[:,'Q11_Part_4'],temp_df.loc[:,'Q11_Part_5'],
p6,p7 = temp_df.loc[:,'Q11_Part_6'],temp_df.loc[:,'Q11_Part_7']

new_df = pd.concat([p1,p2,p3,p4,p5,p6,p7]).reset_index()      # Concating 7 columns of Q7.
new_df.dropna(inplace=True)
new_df = new_df[0].value_counts().reset_index()
new_df.columns = ['a','b']
new_df.iplot(kind='bar', x='a', y='b', title='Important Activities', xTitle='Activity', colors='brown')    


# #### ====================================================================================================

# # DATA SCIENCE ANALYSIS

# ### Q10). Current Employers incorporate Machine Learning or not.

# In[15]:


ml = df_mcr['Q10'].value_counts().reset_index()[:10]
ml.iplot(kind='bar',x='index', y='Q10', title='Use of ML ', 
             xTitle='Whether Ml is used or not', yTitle='Frequency',colors='indigo')


# * ML is highly used in industries.
# * Maximum are exploring ML, means they are new to ML and they want to use ML.

# #### ====================================================================================================

# ### Q11). Top 10 IDE.

# In[16]:


temp_df = df_mcr.drop([0],axis=0)
temp_df.head()

p1,p2,p3 = temp_df.loc[:,'Q13_Part_1'],temp_df.loc[:,'Q13_Part_2'],temp_df.loc[:,'Q13_Part_3'],
p4,p5  = temp_df.loc[:,'Q13_Part_4'],temp_df.loc[:,'Q13_Part_5'],
p6,p7,p8 = temp_df.loc[:,'Q13_Part_6'],temp_df.loc[:,'Q13_Part_7'],temp_df.loc[:,'Q13_Part_8']
p9,p10,p11 = temp_df.loc[:,'Q13_Part_9'],temp_df.loc[:,'Q13_Part_10'],temp_df.loc[:,'Q13_Part_11']
p12,p13,p14 = temp_df.loc[:,'Q13_Part_12'],temp_df.loc[:,'Q13_Part_13'],temp_df.loc[:,'Q13_Part_14']
p15 = temp_df.loc[:,'Q13_Part_15']

new_df = pd.concat([p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14,p15]).reset_index()      # Concating 7 columns of Q7.
new_df.dropna(inplace=True)
new_df = new_df[0].value_counts().reset_index()
new_df.columns = ['a','b']
new_df.iplot(kind='bar', x='a', y='b', title='TOP IDE', xTitle='IDE', colors='deepskyblue')    


# #### Observations:-
# * Jupyter notebook is highly used by people follwed by RStudio.
# * Notebook++ is also in list.
# * PyCharm is mainly used backhand programmers. 

# #### =====================================================================================================

# #### =====================================================================================================

# ### Q12). Online Notebooks which are highly used people.

# In[17]:


temp_df = df_mcr.drop([0],axis=0)
temp_df.head()

p1,p2,p3 = temp_df.loc[:,'Q14_Part_1'],temp_df.loc[:,'Q14_Part_2'],temp_df.loc[:,'Q14_Part_3'],
p4,p5  = temp_df.loc[:,'Q14_Part_4'],temp_df.loc[:,'Q14_Part_5'],
p6,p7,p8 = temp_df.loc[:,'Q14_Part_6'],temp_df.loc[:,'Q14_Part_7'],temp_df.loc[:,'Q14_Part_8']
p9,p10,p11 = temp_df.loc[:,'Q14_Part_9'],temp_df.loc[:,'Q14_Part_10'],temp_df.loc[:,'Q14_Part_11']
#p12,p13,p14 = temp_df.loc[:,'Q13_Part_12'],temp_df.loc[:,'Q13_Part_13'],temp_df.loc[:,'Q13_Part_14']
#p15 = temp_df.loc[:,'Q13_Part_15']

new_df = pd.concat([p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11]).reset_index()      # Concating 7 columns of Q7.
new_df.dropna(inplace=True)
new_df = new_df[0].value_counts().reset_index()
new_df.columns = ['a','b']
#new_df.iplot(kind='bar', x='a', y='b', title='TOP IDE', xTitle='IDE', colors='deepskyblue')    
new_df.iplot(kind='pie', labels='a', values='b', title='Top ONLINE NOTEBOOKS', pull=0.2, hole=0.2)

# #### Observations:-
# * Kaggle kernels, they are highly used Data science people.
# * Followed by Jupyter Hub.
# * This is the reason I choose Kaggle + Data Science.
# * 28% people use no online notebook as they prefer to work on thier own machine.
# 
# 

# #### ====================================================================================================

# ### Q13). Top 10 Programming Languages.

# In[18]:


temp_df = df_mcr.drop([0],axis=0)
temp_df.head()

p1,p2,p3 = temp_df.loc[:,'Q16_Part_1'],temp_df.loc[:,'Q16_Part_2'],temp_df.loc[:,'Q16_Part_3'],
p4,p5  = temp_df.loc[:,'Q16_Part_4'],temp_df.loc[:,'Q16_Part_5'],
p6,p7,p8 = temp_df.loc[:,'Q16_Part_6'],temp_df.loc[:,'Q16_Part_7'],temp_df.loc[:,'Q16_Part_8']
p9,p10,p11 = temp_df.loc[:,'Q16_Part_9'],temp_df.loc[:,'Q16_Part_10'],temp_df.loc[:,'Q16_Part_11']
p12,p13,p14 = temp_df.loc[:,'Q16_Part_12'],temp_df.loc[:,'Q16_Part_13'],temp_df.loc[:,'Q16_Part_14']
p15,p16,p17 = temp_df.loc[:,'Q16_Part_15'], temp_df.loc[:,'Q16_Part_16'], temp_df.loc[:,'Q16_Part_17']
p18 = temp_df.loc[:,'Q16_Part_18']

new_df = pd.concat([p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14,p15,p16,p17,p18]).reset_index()      

new_df.dropna(inplace=True)
new_df = new_df[0].value_counts().reset_index()[:10]
new_df.columns = ['a','b']
#new_df.iplot(kind='bar', x='a', y='b', title='TOP IDE', xTitle='IDE', colors='deepskyblue')    
new_df.iplot(kind='pie', labels='a', values='b', title='Top 10 Programming Languages', pull=0.2, hole=0.2)

# #### Observations:-
# * As aspected python, sql and R at the top.
# <img src='https://drive.google.com/uc?id=1iObz8QN2D3OS83tXeJQQ51VV3d2vXFUx' width=500>
# * In future python will eat maximum languages.
# * 31% data science people use python.
# * SQL is also neccesary to connect with data base, which is at 16%.

# #### ====================================================================================================

# ### Q14). Top 10 machine Learning libraries.

# In[19]:


temp_df = df_mcr.drop([0],axis=0)
temp_df.head()

p1,p2,p3 = temp_df.loc[:,'Q19_Part_1'],temp_df.loc[:,'Q19_Part_2'],temp_df.loc[:,'Q19_Part_3'],
p4,p5  = temp_df.loc[:,'Q19_Part_4'],temp_df.loc[:,'Q19_Part_5'],
p6,p7,p8 = temp_df.loc[:,'Q19_Part_6'],temp_df.loc[:,'Q19_Part_7'],temp_df.loc[:,'Q19_Part_8']
p9,p10,p11 = temp_df.loc[:,'Q19_Part_9'],temp_df.loc[:,'Q19_Part_10'],temp_df.loc[:,'Q19_Part_11']
p12,p13,p14 = temp_df.loc[:,'Q19_Part_12'],temp_df.loc[:,'Q19_Part_13'],temp_df.loc[:,'Q19_Part_14']
p15,p16,p17 = temp_df.loc[:,'Q19_Part_15'], temp_df.loc[:,'Q19_Part_16'], temp_df.loc[:,'Q19_Part_17']
p18 = temp_df.loc[:,'Q19_Part_18']

new_df = pd.concat([p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14,p15,p16,p17,p18]).reset_index()      

new_df.dropna(inplace=True)
new_df = new_df[0].value_counts().reset_index()[:10]
new_df.columns = ['a','b']
#new_df.iplot(kind='bar', x='a', y='b', title='TOP IDE', xTitle='IDE', colors='deepskyblue')    
new_df.iplot(kind='pie', labels='a', values='b', title='Top 10 Machine Learning Libraries', pull=0.2, hole=0.2)

# ### Observations:-
# * Scikit-learn topped the list followed by tensorflow and by keras.
# * 22% people use Scikit-learn.
# * 18% people use Tensorflow.
# * 14% people use Keras.

# #### ===================================================================================================

# ### Q15). Top 10 Libraries for Visualization.

# In[20]:


temp_df = df_mcr.drop([0],axis=0)
temp_df.head()

p1,p2,p3 = temp_df.loc[:,'Q21_Part_1'],temp_df.loc[:,'Q21_Part_2'],temp_df.loc[:,'Q21_Part_3'],
p4,p5  = temp_df.loc[:,'Q21_Part_4'],temp_df.loc[:,'Q21_Part_5'],
p6,p7,p8 = temp_df.loc[:,'Q21_Part_6'],temp_df.loc[:,'Q21_Part_7'],temp_df.loc[:,'Q21_Part_8']
p9,p10,p11 = temp_df.loc[:,'Q21_Part_9'],temp_df.loc[:,'Q21_Part_10'],temp_df.loc[:,'Q21_Part_11']
p12,p13 = temp_df.loc[:,'Q21_Part_12'],temp_df.loc[:,'Q21_Part_13']
#p15,p16,p17 = temp_df.loc[:,'Q16_Part_15'], temp_df.loc[:,'Q16_Part_16'], temp_df.loc[:,'Q16_Part_17']
#p18 = temp_df.loc[:,'Q16_Part_18']

new_df = pd.concat([p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,]).reset_index()      

new_df.dropna(inplace=True)
new_df = new_df[0].value_counts().reset_index()[:10]
new_df.columns = ['a','b']
#new_df.iplot(kind='bar', x='a', y='b', title='TOP IDE', xTitle='IDE', colors='deepskyblue')    
new_df.iplot(kind='pie', labels='a', values='b', title='Top 10 Libraries for Visualization', pull=0.2, hole=0.2)

# #### Observations:-
# * Matplotlib,the basic of visualization library, topped the list.
# * Seaborn,more advance than matplotlib and it is based on maplotlib, is at 2nd rank.
# * Plotly, which I am using right now, is at 4th place.
# * Feeling happy to see this list.

# #### ====================================================================================================

# ### Q16). How long have people been writing code to analysis the data.

# In[21]:


time = df_mcr['Q24'].value_counts().reset_index()
time.iplot(kind='pie', labels='index', values='Q24',title='Time in Year ', pull=0.2, hole=0.2 )

# #### Observations:-
# * Mainly people are using it for 1 to 5 year.
# * Totaly new field and expanding at a high pace.
# * Maybe my step to choose Data Science is correct, let's see in future.
# 

# #### ===================================================================================================

# ### Q17). Top 10 Machine Learning products that are used at work.

# In[22]:


temp_df = df_mcr.drop([0],axis=0)
temp_df.head()

p1,p2,p3 = temp_df.loc[:,'Q28_Part_1'],temp_df.loc[:,'Q28_Part_2'],temp_df.loc[:,'Q28_Part_3'],
p4,p5  = temp_df.loc[:,'Q28_Part_4'],temp_df.loc[:,'Q28_Part_5'],
p6,p7,p8 = temp_df.loc[:,'Q28_Part_6'],temp_df.loc[:,'Q28_Part_7'],temp_df.loc[:,'Q28_Part_8']
p9,p10,p11 = temp_df.loc[:,'Q28_Part_9'],temp_df.loc[:,'Q28_Part_10'],temp_df.loc[:,'Q28_Part_11']
p12,p13,p14 = temp_df.loc[:,'Q28_Part_12'],temp_df.loc[:,'Q28_Part_13'],temp_df.loc[:,'Q28_Part_14']
p15,p16,p17 = temp_df.loc[:,'Q28_Part_15'], temp_df.loc[:,'Q28_Part_16'], temp_df.loc[:,'Q28_Part_17']
p18 = temp_df.loc[:,'Q28_Part_18']

new_df = pd.concat([p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14,p15,p16,p17,p18]).reset_index()      

new_df.dropna(inplace=True)
new_df = new_df[0].value_counts().reset_index()[:10]
new_df.columns = ['a','b']
#new_df.iplot(kind='bar', x='a', y='b', title='TOP IDE', xTitle='IDE', colors='deepskyblue')    
new_df.iplot(kind='pie', labels='a', values='b', title='Top 10 Machine Learning Products', pull=0.2, hole=0.2)

# #### ====================================================================================================

# ### Q18). Top 10 Big Data and Analytics Products.

# In[25]:


temp_df = df_mcr.drop([0],axis=0)
temp_df.head()

p1,p2,p3 = temp_df.loc[:,'Q30_Part_1'],temp_df.loc[:,'Q30_Part_2'],temp_df.loc[:,'Q30_Part_3'],
p4,p5  = temp_df.loc[:,'Q30_Part_4'],temp_df.loc[:,'Q30_Part_5'],
p6,p7,p8 = temp_df.loc[:,'Q30_Part_6'],temp_df.loc[:,'Q30_Part_7'],temp_df.loc[:,'Q30_Part_8']
p9,p10,p11 = temp_df.loc[:,'Q30_Part_9'],temp_df.loc[:,'Q30_Part_10'],temp_df.loc[:,'Q30_Part_11']
p12,p13,p14 = temp_df.loc[:,'Q30_Part_12'],temp_df.loc[:,'Q30_Part_13'],temp_df.loc[:,'Q30_Part_14']
p15,p16,p17 = temp_df.loc[:,'Q30_Part_15'], temp_df.loc[:,'Q30_Part_16'], temp_df.loc[:,'Q30_Part_17']
p18 = temp_df.loc[:,'Q30_Part_18']

new_df = pd.concat([p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14,p15,p16,p17,p18]).reset_index()      

new_df.dropna(inplace=True)
new_df = new_df[0].value_counts().reset_index()[:10]
new_df.columns = ['a','b']
#new_df.iplot(kind='bar', x='a', y='b', title='TOP IDE', xTitle='IDE', colors='deepskyblue')    
new_df.iplot(kind='pie', labels='a', values='b', title='Top 10 Big Data and Analytic Products', pull=0.2, hole=0.2)

# 
# #### ================================================================================================

# ### Q19). Top 10 types of Data people currently interact.

# In[26]:


temp_df = df_mcr.drop([0],axis=0)
temp_df.head()

p1,p2,p3 = temp_df.loc[:,'Q31_Part_1'],temp_df.loc[:,'Q31_Part_2'],temp_df.loc[:,'Q31_Part_3'],
p4,p5  = temp_df.loc[:,'Q31_Part_4'],temp_df.loc[:,'Q31_Part_5'],
p6,p7,p8 = temp_df.loc[:,'Q31_Part_6'],temp_df.loc[:,'Q31_Part_7'],temp_df.loc[:,'Q31_Part_8']
p9,p10,p11 = temp_df.loc[:,'Q31_Part_9'],temp_df.loc[:,'Q31_Part_10'],temp_df.loc[:,'Q31_Part_11']
p12 = temp_df.loc[:,'Q31_Part_12']
#p15,p16,p17 = temp_df.loc[:,'Q30_Part_15'], temp_df.loc[:,'Q30_Part_16'], temp_df.loc[:,'Q30_Part_17']
#p18 = temp_df.loc[:,'Q30_Part_18']

new_df = pd.concat([p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12]).reset_index()      

new_df.dropna(inplace=True)
new_df = new_df[0].value_counts().reset_index()[:10]
new_df.columns = ['a','b']
#new_df.iplot(kind='bar', x='a', y='b', title='TOP IDE', xTitle='IDE', colors='deepskyblue')    
new_df.iplot(kind='pie', labels='a', values='b', title='Top 10 Type of Data', pull=0.2, hole=0.2)

# * Mainly people interact with numerical data then by text data and then by categorical data.
# * So practice on these type of data will help.
# 

# #### =====================================================================================================

# ### Q20). Top 10 places to find Pubic Datasets.

# In[27]:


temp_df = df_mcr.drop([0],axis=0)
temp_df.head()

p1,p2,p3 = temp_df.loc[:,'Q33_Part_1'],temp_df.loc[:,'Q33_Part_2'],temp_df.loc[:,'Q33_Part_3'],
p4,p5  = temp_df.loc[:,'Q33_Part_4'],temp_df.loc[:,'Q33_Part_5'],
p6,p7,p8 = temp_df.loc[:,'Q33_Part_6'],temp_df.loc[:,'Q33_Part_7'],temp_df.loc[:,'Q33_Part_8']
p9,p10,p11 = temp_df.loc[:,'Q33_Part_9'],temp_df.loc[:,'Q33_Part_10'],temp_df.loc[:,'Q33_Part_11']
#p12 = temp_df.loc[:,'Q31_Part_12']
#p15,p16,p17 = temp_df.loc[:,'Q30_Part_15'], temp_df.loc[:,'Q30_Part_16'], temp_df.loc[:,'Q30_Part_17']
#p18 = temp_df.loc[:,'Q30_Part_18']

new_df = pd.concat([p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11]).reset_index()      

new_df.dropna(inplace=True)
new_df = new_df[0].value_counts().reset_index()[:10]
new_df.columns = ['a','b']
new_df.iplot(kind='bar', x='a', y='b', title='TOP Places for Dataset', xTitle='Place', colors='deepskyblue')    
#new_df.iplot(kind='pie', labels='a', values='b', title='Top 10 Type of Data', pull=0.2, hole=0.2)

# * Maximum people get their dataset from kaggle.
# * As expected kaggle lead the race.
# * The reason I choose kaggle.
# 

# #### =====================================================================================================

# ### Q21). Top 10  Online Platforms to learn Data Science.

# In[28]:


online = df_mcr['Q37'].value_counts().reset_index()[:10]
online.iplot(kind='pie', labels='index', values='Q37',title='Top 10 Online Platforms to learn Data Science ',
             pull=0.2, hole=0.2 )

# #### Observations:- 
# * 40% people are learning from Coursera.
# * Datacamp, udemy are at 2nd and 3rd places with 12% each.
# * Mainly people are learning Data Science from online courses.
# * Online courses are very flexible, so they are in high demnad.

# #### ==================================================================================================

# ### Q22). Top 10 Websites for Data Science News.

# In[29]:


temp_df = df_mcr.drop([0],axis=0)
temp_df.head()

p1,p2,p3 = temp_df.loc[:,'Q38_Part_1'],temp_df.loc[:,'Q38_Part_2'],temp_df.loc[:,'Q38_Part_3'],
p4,p5  = temp_df.loc[:,'Q38_Part_4'],temp_df.loc[:,'Q38_Part_5'],
p6,p7,p8 = temp_df.loc[:,'Q38_Part_6'],temp_df.loc[:,'Q38_Part_7'],temp_df.loc[:,'Q38_Part_8']
p9,p10,p11 = temp_df.loc[:,'Q38_Part_9'],temp_df.loc[:,'Q38_Part_10'],temp_df.loc[:,'Q38_Part_11']
p12,p13,p14 = temp_df.loc[:,'Q38_Part_12'],temp_df.loc[:,'Q38_Part_13'],temp_df.loc[:,'Q38_Part_14']
p15,p16,p17 = temp_df.loc[:,'Q38_Part_15'], temp_df.loc[:,'Q38_Part_16'], temp_df.loc[:,'Q38_Part_17']
p18 = temp_df.loc[:,'Q38_Part_18']

new_df = pd.concat([p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14,p15,p16,p17,p18]).reset_index()      

new_df.dropna(inplace=True)
new_df = new_df[0].value_counts().reset_index()[:10]
new_df.columns = ['a','b']
#new_df.iplot(kind='bar', x='a', y='b', title='TOP IDE', xTitle='IDE', colors='deepskyblue')    
new_df.iplot(kind='pie', labels='a', values='b', title='Top 10 Websites for Data Science News', pull=0.2, hole=0.2)

# #### Observations:-
# * Kaggleform is at top place with 18%.
# * It is followed by mediumblog post.
# * Kaggle is again at the top.
# * Many things I am learning from kaggle platform.
# 

#  ### Q23). People view on the quality of Online learning and  bootcamp.

# In[30]:


temp_df = df_mcr.drop([0],axis=0)
temp_df.head()

p1,p2 = temp_df.loc[:,'Q39_Part_1'],temp_df.loc[:,'Q39_Part_2']


new_df = pd.concat([p1,p2]).reset_index()      

new_df.dropna(inplace=True)
new_df = new_df[0].value_counts().reset_index()[:10]
new_df.columns = ['a','b']
#new_df.iplot(kind='bar', x='a', y='b', title='TOP IDE', xTitle='IDE', colors='deepskyblue')    
new_df.iplot(kind='pie', labels='a', values='b', title='View on quality of OnlineLearning and Bootcamp',
             pull=0.2, hole=0.2)

# #### Observations:-
# * 50% people love to learn from online sources or from bootcamp.
# * This is a new type of education revolution.
# 

# #### ===================================================================================================

# ### Q24). Academic Achievement VS Independent projects for Data Science.

# In[31]:


online = df_mcr['Q40'].value_counts().reset_index()[:10]
online.iplot(kind='bar', x='index', y='Q40',title='Academic Achievement VS Independent projects for Data Science ',
             xTitle='Comparision', colors='deepskyblue' )

# * Independent projects are more important than academic achievement.

# #### ===================================================================================================
# #### ===================================================================================================
# #### ===================================================================================================

# # IF YOU LIKE THIS KERNEL,THEN PLEASE UPVOTE.
# <img src='https://drive.google.com/uc?id=1snfO_T6LnQAgj4ps3whAvcfkipEOEORj' width=600>

# In[ ]:



