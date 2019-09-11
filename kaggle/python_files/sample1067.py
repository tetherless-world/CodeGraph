#!/usr/bin/env python
# coding: utf-8

# ![](https://static.wixstatic.com/media/80a58d_adc900710a474cd091d5dae9649734f9~mv2.png/v1/fill/w_812,h_353,al_c,lg_1/80a58d_adc900710a474cd091d5dae9649734f9~mv2.png)

# # More To Come. Stay Tuned. !!
# If there are any suggestions/changes you would like to see in the Kernel please let me know :). Appreciate every ounce of help!
# 
# **This notebook will always be a work in progress.** Please leave any comments about further improvements to the notebook! Any feedback or constructive criticism is greatly appreciated!. **If you like it or it helps you , you can upvote and/or leave a comment :).**

# - <a href='#intro'>1. Introduction</a>  
# - <a href='#rtd'>2. Retrieving the Data</a>
#      - <a href='#ll'>2.1 Load libraries</a>
#      - <a href='#rrtd'>2.2 Read the Data</a>
# - <a href='#god'>3. Glimpse of Data</a>
#      - <a href='#oot'>3.1 Overview of tables</a>
#      - <a href='#sootd'>3.2 Statistical overview of the Data</a>
# - <a href='#dp'>4. Data preparation</a>
#      - <a href='#cfmd'> 4.1 Check for missing data</a>
# - <a href='#de'>5. Data Exploration</a>
#      - <a href='#ppan'>5.1 Project proposal is Approved or not ?</a>
#      - <a href='#d'>5.2 Distribution</a>
#          - <a href='#doss'>5.2.a Distribution of School states</a>
#          - <a href='#dpgc'>5.2.b Distribution of project_grade_category (school grade levels (PreK-2, 3-5, 6-8, and 9-12))</a>
#          - <a href='#dcotp'>5.2.c Distribution of category of the project</a>
#          - <a href='#dnppast'>5.2.d Distribution of number of previously posted applications by the submitting teacher</a>
#          - <a href='#dsotp'>5.2.e Distribution of subcategory of the project</a>
#          - <a href='#dopt'>5.2.f Distribution of Project titles</a>
#          - <a href='#doporr'>5.2.g Distribution of price of resource requested</a>
#          - <a href='#doqorr'>5.2.h Distribution of quantity of resource requested</a>
#          - <a href='#tpd'>5.2.i Teacher prefix Distribution</a>
#      - <a href='#trnfp'>5.3 Top resources needed for the project</a> 
#      - <a href='#wcrr'>5.4 Word Cloud of resources requested</a>
#      - <a href='#vpppan'>5.5 Various popularities in terms of project acceptance rate and project rejection rate</a>
#          - <a href='#psstpan'>5.5.a Popular School states in terms of project acceptance rate and project rejection rate</a>
#          - <a href='#ptptppa'>5.5.b Popular Teacher Prefix in terms of project acceptance rate and project rejection rate</a>
#          - <a href='#psgltpp'>5.5.c Popular school grade levels in terms of project acceptance rate and project rejection rate</a>
#          - <a href='#pcoptppa'>5.5.d Popular category of the project in terms of project acceptance rate and project rejection rate</a>
#          - <a href='#psotpppan'>5.5.e Popular subcategory of the project in terms of project acceptance rate and project rejection rate</a>
#          - <a href='#ppttpan'>5.5.f Popular project titles in terms of project acceptance rate and project rejection rate</a>
#      - <a href='#ppuss'>5.6 Project Proposals by US States</a>
#      - <a href='#ppmaruss'>5.7 Project Proposals Mean Acceptance Rate by US States</a>
#      - <a href='#cmhmtd'>5.8 Correlation Matrix and HeatMap of training data</a>
#          - <a href='#tppiaic'>5.8.a Teacher_prefix and project_is_approved Intervals Correlation</a>
#          - <a href='#tnppppaic'>5.8.b Teacher_number_of_previously_posted_projects and project_is_approved Intervals Correlation</a>
#          - <a href='#cmaht'>5.8.c Correlation Matrix and Heatmap of training data</a>
#      - <a href='#psta'>5.9 Project Submission Time Analysis</a>
#          - <a href='#psma'>5.9.a Project Submission Month Analysis</a>
#          - <a href='#pswa'>5.9.b Project Submission Weekday Analysis</a>
#          - <a href='#psda'>5.9.c Project Submission Date Analysis</a>
#          - <a href='#psha'>5.9.d Project Submission Hour Analysis</a>
#      - <a href='#tkips1'>5.10 Top Keywords in project_essay_1</a>
#      - <a href='#tkipe2'>5.11 Top Keywords in project_essay_2</a>
#      - <a href='#tkinprs'>5.12 Top Keywords in project_resource_summary</a>
#      - <a href='#qvp'>5.13 Quantity V.S. Price</a>
#      - <a href='#gapc'>5.14 Gender Analysis</a>
#      - <a href='#mwdppes'>5.15 Month wise distribution of number of projects proposal submitted in each state</a>
#      - <a href='#prfrd'>5.16 Price requested for resources distribution</a> [I commented the code becuase of excessive rendering time but i wrote the results]
#          - <a href='#prfrbds'>5.16.a Price requested for resources distribution by different states</a>
#          - <a href='#prfrbtp'>5.16.b Price requested for resources distribution by Teacher prefixes</a>
#          - <a href='#prfrddga'>5.16.c Price requested for resources distribution by different Genders</a>
#          - <a href='#prfrddpgc'>5.16.d Price requested for resources distribution by different project_grade_category</a>
#      - <a href='#ca'>5.17 CA(California)</a>
#          - <a href='#potpic'>5.17.a Popularities of Teacher prefixes in California</a>
#          - <a href='#posglic'>5.17.b Popularities of school grade levels in California</a>
#          - <a href='#tptic'>5.17.c Top project titles in California</a>
#          - <a href='#topstic'>5.17.d Trend of project submission time in California</a>
#      - <a href='#TX'>5.18 TX(Texas)</a>
#          - <a href='#potpit'>5.18.a Popularities of Teacher prefixes in Texas</a>
#          - <a href='#posglit'>5.18.b Popularities of school grade levels in Texas</a>
#          - <a href='#tptit'>5.18.c Top project titles in Texas</a>
#          - <a href='#topstit'>5.18.d Trend of project submission time in Texas</a>
# - <a href='#fea'>6. Feature Engineering</a>
#     - <a href='#fcfpsd'>6.1 Features Created from project_submitted_datetime</a>
#     - <a href='#fcfpppr'>6.2 Features created from project_essay_1, project_essay_2, project_title, project_resource_summary</a>
#     - <a href='#fracttr'>6.3 Features created after combining the training data, test data with resource file</a>
#     - <a href='#gfc'>6.4 Gender feature creation</a>
#     - <a href='#ccftnf'>6.5 Converting Categorical featues to numericals features</a>
#     - <a href='#ruf'>6.6 Removing Unwanted features</a>
# - <a href='#bmlm'>7. Building Machine Learning Models</a>
#     - <a href='#lgbm'>7.1 Light GBM</a>
#     - <a href='#xgbst'>7.2 XGBoost</a>
# - <a href='#feva'>8. Further Evaluation</a>
#     - <a href='#rocaucc'>8.1 ROC AUC Curve</a>
#     - <a href='#prcur'>8.2 Precision Recall curve</a>
#     - <a href='#pmdtlgb'>8.3 Ploting Metrics during training of Light GBM</a>
#     - <a href='#fiblgm'>8.4 Feature importances by LightGBM</a>
# - <a href='#bsc'>9. Brief Summary/Conclusion :</a>

# ## <a id='intro'>1. Intoduction</a>

# **About DonorsChoose:**
# 
# DonorsChoose.org is a United States–based 501(c)(3) nonprofit organization that allows individuals to donate directly to public school classroom projects. Founded in 2000 by former public school teacher Charles Best, DonorsChoose.org was among the first civic crowdfunding platforms of its kind. The organization has been given Charity Navigator’s highest rating every year since 2005. In January 2018, they announced that 1 million projects had been funded. In 77% of public schools in the United States, at least one project has been requested on DonorsChoose.org. Schools from wealthy areas are more likely to make technology requests, while schools from less affluent areas are more likely to request basic supplies. It's been noted that repeat donors on DonorsChoose typically donate to projects they have no prior relationship with, and most often fund projects serving financially challenged students.
# 
# 
# **Objective of this Notebook:**
# 
# In this Notebook i will do Exploratory Analysis.
# 
# **Objective of the competition:**
# 
# DonorsChoose.org receives hundreds of thousands of project proposals each year for classroom projects in need of funding. Right now, a large number of volunteers is needed to manually screen each submission before it's approved to be posted on the DonorsChoose.org website.The goal of the competition is to predict whether or not a DonorsChoose.org project proposal submitted by a teacher will be approved, using the text of project descriptions as well as additional metadata about the project, teacher, and school. DonorsChoose.org can then use this information to identify projects most likely to need further review before approval.
# 

# # <a id='rtd'>2. Retrieving the Data</a>

# ## <a id='ll'>2.1 Load libraries</a>

# In[ ]:


import pandas as pd # package for high-performance, easy-to-use data structures and data analysis
import numpy as np # fundamental package for scientific computing with Python
import matplotlib
import matplotlib.pyplot as plt # for plotting
import seaborn as sns # for making plots with seaborn
color = sns.color_palette()
import plotly.offline as py
py.init_notebook_mode(connected=True)
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.offline as offline
offline.init_notebook_mode()
import plotly.tools as tls
import squarify
from mpl_toolkits.basemap import Basemap
from numpy import array
from matplotlib import cm

# Supress unnecessary warnings so that presentation looks clean
import warnings
warnings.filterwarnings("ignore")

# Print all rows and columns
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

from nltk.corpus import stopwords
from textblob import TextBlob
import datetime as dt
import warnings
import string
import time
# stop_words = []
stop_words = list(set(stopwords.words('english')))
warnings.filterwarnings('ignore')
punctuation = string.punctuation

# In[ ]:


from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# ## <a id='rrtd'>2.2 Read tha Data</a>

# In[ ]:


train_data = pd.read_csv("../input/donorschoose-application-screening/train.csv")
test_data = pd.read_csv("../input/donorschoose-application-screening/test.csv")
resources_data = pd.read_csv("../input/donorschoose-application-screening/resources.csv")

## Merging with train and test data ##
train_resource = pd.merge(train_data, resources_data, on="id", how='left')
test_resource = pd.merge(test_data, resources_data, on="id", how='left')

# In[ ]:


print("Size of training data : ",train_data.shape)
print("Size of test data : ",test_data.shape)
print("Size of resource data : ",resources_data.shape)
print("Size of train_resource data : ",train_resource.shape)
print("Size of test_resource data : ",test_resource.shape)

# # <a id='god'>3. Glimpse of Data</a>

# ## <a id='oot'>3.1 Overview of tables</a>

# **Training Data**

# In[ ]:


train_data.head()

# **Test Data**

# In[ ]:


test_data.head()

# **Resource Data**

# In[ ]:


resources_data.head()

# **train_resource**

# In[ ]:


train_resource.head()

# **test_resource**

# In[ ]:


test_resource.head()

# ## <a id='sootd'>3.2 Statistical Overview of the Data</a>

# **Training Data some little info**

# In[ ]:


train_data.info()

# **Little description of training data for numerical features**
# 

# In[ ]:


train_data.describe()

# **Little description of training data for categorical features**
# 

# In[ ]:


train_data.describe(include=["O"])

# **Little description of train_resource data for numerical features**
# 

# In[ ]:


train_resource.describe()

# **Little description of train_resource data for categorical features**
# 

# In[ ]:


train_resource.describe(include=["O"])

# # <a id='dp'>4. Data preparation</a>

# ## <a id='cfmd'>4.1 Checking for missing data</a>

# **Missing data in train_data**

# In[ ]:


# checking missing data in training data 
total = train_data.isnull().sum().sort_values(ascending = False)
percent = (train_data.isnull().sum()/train_data.isnull().count()*100).sort_values(ascending = False)
missing_train_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_train_data.head()

# * In training data, we can **project_essay_4** and **project_essay_3** having 96 % null values. so during prediction, better remove these 2 columns.

# **Missing data in test_data**

# In[ ]:


# checking missing data in test data 
total = test_data.isnull().sum().sort_values(ascending = False)
percent = (test_data.isnull().sum()/test_data.isnull().count()*100).sort_values(ascending = False)
missing_test_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_test_data.head()

# * In test data, we can **project_essay_4** and **project_essay_3** having 96 % null values. so during prediction, better remove these 2 columns.

# **Missing data in resources_data**

# In[ ]:


# checking missing data in resource data 
total = resources_data.isnull().sum().sort_values(ascending = False)
percent = (resources_data.isnull().sum()/resources_data.isnull().count()*100).sort_values(ascending = False)
missing_resources_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_resources_data.head()

# * In resource data, only **description** column having few null values. So we can ignore these values.

# # <a id='de'>5. Data Exploration</a>

# ## <a id='ppan'>5.1 Project proposal is Approved or not ?</a>

# In[ ]:


temp = train_data['project_is_approved'].value_counts()
labels = temp.index
sizes = (temp / temp.sum())*100
trace = go.Pie(labels=labels, values=sizes, hoverinfo='label+percent')
layout = go.Layout(title='Project proposal is approved or not')
data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)

# * Training data is highly imbalanced that is approx. 85 % projetcs were approved and 15 % project were not approved. Majority imbalanced class is positive.

# ## <a id='d'>5.2 Distribution</a>

# ### <a id='doss'>5.2.a Distribution of School states</a>

# In[ ]:


temp = train_data["school_state"].value_counts()
#print("Total number of states : ",len(temp))
trace = go.Bar(
    x = temp.index,
    y = (temp / temp.sum())*100,
)
data = [trace]
layout = go.Layout(
    title = "Distribution of School states in % ",
    xaxis=dict(
        title='State Name',
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
    ),
    yaxis=dict(
        title='Count of project proposals submitted in %',
        titlefont=dict(
            size=16,
            color='rgb(107, 107, 107)'
        ),
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
)
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='schoolStateNames')

# * Out of 50 states, **California(CA)** having higher number of projects proposal submitted **approx. 14 %**  followed by **Texas(TX)(7 %)** and **Tennessee(NY)(7 %)**.

# ### <a id='dpgc'>5.2.b Distribution of project_grade_category (school grade levels (PreK-2, 3-5, 6-8, and 9-12))</a>

# In[ ]:


temp = train_data["project_grade_category"].value_counts()
print("Total number of project grade categories : ", len(temp))
trace = go.Bar(
    x = temp.index,
    y = (temp / temp.sum())*100,
)
data = [trace]
layout = go.Layout(
    title = "Distribution of project_grade_category (school grade levels) in %",
    xaxis=dict(
        title='school grade levels',
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
    ),
    yaxis=dict(
        title='Count of project proposals submitted in % ',
        titlefont=dict(
            size=16,
            color='rgb(107, 107, 107)'
        ),
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
)
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='schoolStateNames')

# * Out of 4 school grade levels, Project proposals submission in school grade levels is higher for **Grades Prek-2** which is approximately **41 %** followed by **Grades 3-5** which has approx. **34 %**.

# ### <a id='dcotp'>5.2.c Distribution of category of the project</a>

# In[ ]:


temp = train_data["project_subject_categories"].value_counts().head(10)
print("Total number of project subject categories : ", len(train_data["project_subject_categories"].value_counts()))
trace = go.Bar(
    x = temp.index,
    y = (temp / temp.sum())*100,
)
data = [trace]
layout = go.Layout(
    title = "Distribution of category of the project in %",
    xaxis=dict(
        title='category of the project',
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
    ),
    yaxis=dict(
        title='Count of project proposals submitted in %',
        titlefont=dict(
            size=16,
            color='rgb(107, 107, 107)'
        ),
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
)
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='schoolStateNames')

# * Out of 51 Project categories,  Project proposals submission for project categories is higher  for  **Literacy & Language** which is approx. **27 %** followed by **Math & Science** which has approx. **20 %**.

# ### <a id='dnppast'>5.2.d Distribution of number of previously posted applications by the submitting teacher</a>

# In[ ]:


plt.figure(figsize = (12, 8))

sns.distplot(train_data['teacher_number_of_previously_posted_projects'])
plt.xlabel('number of previously posted applications by the submitting teacher', fontsize=12)
plt.title("Histogram of number of previously posted applications by the submitting teacher")
plt.show() 
plt.figure(figsize = (12, 8))
plt.scatter(range(train_data.shape[0]), np.sort(train_data.teacher_number_of_previously_posted_projects.values))
plt.xlabel('number of previously posted applications by the submitting teacher', fontsize=12)
plt.title("Distribution of number of previously posted applications by the submitting teacher")
plt.show()

#    ### <a id='dsotp'>5.2.e Distribution of subcategory of the project</a>

# In[ ]:


temp = train_data["project_subject_subcategories"].value_counts().head(10)
print("Total sub-categories of the projects : ",len(train_data["project_subject_subcategories"]))
trace = go.Bar(
    x = temp.index,
    y = (temp / temp.sum())*100,
)
data = [trace]
layout = go.Layout(
    title = "Distribution of subcategory of the project in %",
    xaxis=dict(
        title='subcategory of the project',
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
    ),
    yaxis=dict(
        title='Count of project proposals submitted in %',
        titlefont=dict(
            size=16,
            color='rgb(107, 107, 107)'
        ),
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
)
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='schoolStateNames')

# * Out of 1,82,020 Project subcategories, Project proposals submission for project sub-categoriesis is higher  for **Literacy** which is approx. **16 % ** followed by **Literacy & Mathematics** which has approx. **16 %** .

# ### <a id='dopt'>5.2.f Distribution of Project titles</a>

# In[ ]:


temp = train_data["project_title"].value_counts().head(10)
print("Total project titles are : ", len(train_data["project_title"]))
trace = go.Bar(
    x = temp.index,
    y = (temp / temp.sum())*100,
)
data = [trace]
layout = go.Layout(
    title = "Distribution of Distribution of Project titles in %",
    xaxis=dict(
        title='Project Title',
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
    ),
    yaxis=dict(
        title='Count of project proposals submitted in %',
        titlefont=dict(
            size=16,
            color='rgb(107, 107, 107)'
        ),
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
)
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='schoolStateNames')

# * Out of 1,82,080 project titles, Project proposals submission for project titles is higher for **Flexible seating** which is approx. **27 %** followed by **Whiggle while your work** which has approx. **14 %**.

# ### <a id='doporr'>5.2.g Distribution of price of resource requested</a>

# In[ ]:


plt.figure(figsize = (12, 8))

sns.distplot(train_resource['price'])
plt.xlabel('Price', fontsize=12)
plt.title("Histogran of price of resource requested")
plt.show() 
plt.figure(figsize = (12, 8))
plt.scatter(range(train_resource.shape[0]), np.sort(train_resource.price.values))
plt.xlabel('price', fontsize=12)
plt.title("Distribution of price of resource requested")
plt.show()

# ### <a id='doqorr'>5.2.h Distribution of quantity of resource requested</a>

# In[ ]:


plt.figure(figsize = (12, 8))

sns.distplot(train_resource['price'])
plt.xlabel('quantity', fontsize=12)
plt.title("Histogran of quantity of resource requested")
plt.show() 
plt.figure(figsize = (12, 8))
plt.scatter(range(train_resource.shape[0]), np.sort(train_resource.quantity.values))
plt.xlabel('price', fontsize=12)
plt.title("Distribution of quantity of resource requested")
plt.show()

# ### <a id='tpd'>5.2.i Teacher prefix Distribution</a>

# In[ ]:


temp = train_data["teacher_prefix"].value_counts()
trace = go.Bar(
    x = temp.index,
    y = (temp / temp.sum())*100,
)
data = [trace]
layout = go.Layout(
    title = "Teacher prefix Distribution in %",
    xaxis=dict(
        title='Teacher prefix',
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
    ),
    yaxis=dict(
        title='Count of project proposals submitted in %',
        titlefont=dict(
            size=16,
            color='rgb(107, 107, 107)'
        ),
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
)
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)

# * Higher number of project proposal submitted by **married womens** which is approx. **53 %**  followed by **unmarried womens** which has approx. **37 %**.
# * Project proposal submitted by **Teacher** which is approx. **2 %** is vey low as compared to **Mrs., Ms., Mr**.

# ## <a id='trnfp'>5.3 Top resources needed for the project</a>

# In[ ]:


import re
from nltk.corpus import stopwords

REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))
def text_prepare(text):
    """
        text: a string
        
        return: modified initial string
    """
    text = text.lower()# lowercase text  
    text = REPLACE_BY_SPACE_RE.sub(' ',text)# replace REPLACE_BY_SPACE_RE symbols by space in text    
    text = BAD_SYMBOLS_RE.sub('',text)# delete symbols which are in BAD_SYMBOLS_RE from text    
    temp = [s.strip() for s in text.split() if s not in STOPWORDS]# delete stopwords from text
    new_text = ''
    for i in temp:
        new_text +=i+' '
    text = new_text
    return text.strip()

# In[ ]:


temp_data = train_data.dropna(subset=['project_resource_summary'])
# converting into lowercase
temp_data['project_resource_summary'] = temp_data['project_resource_summary'].apply(lambda x: " ".join(x.lower() for x in x.split()))
temp_data['project_resource_summary'] = temp_data['project_resource_summary'].map(text_prepare)


from wordcloud import WordCloud

wordcloud = WordCloud(max_font_size=50, width=600, height=300).generate(' '.join(temp_data['project_resource_summary'].values))
plt.figure(figsize=(15,8))
plt.imshow(wordcloud)
plt.title("Top resources needed for the project", fontsize=35)
plt.axis("off")
plt.show() 

# ## <a id='wcrr'>5.4 Word Cloud of resources requested</a>

# In[ ]:


temp_data = train_resource.dropna(subset=['description'])
# converting into lowercase
temp_data['description'] = temp_data['description'].apply(lambda x: " ".join(x.lower() for x in x.split()))
temp_data['description'] = temp_data['description'].map(text_prepare)


from wordcloud import WordCloud

wordcloud = WordCloud(max_font_size=50, width=600, height=300).generate(' '.join(temp_data['description'].values))
plt.figure(figsize=(15,8))
plt.imshow(wordcloud)
plt.title("Word Cloud of resources requested", fontsize=35)
plt.axis("off")
plt.show() 

# ## <a id='vpppan'>5.5 Various popularities in terms of project acceptance rate and project rejection rate</a>

# ### <a id='psstpan'>5.5.a Popular School states in terms of project acceptance rate and project rejection rate</a>

# In[ ]:


temp = train_data["school_state"].value_counts()
#print(temp.values)
temp_y0 = []
temp_y1 = []
for val in temp.index:
    temp_y1.append(np.sum(train_data["project_is_approved"][train_data["school_state"]==val] == 1))
    temp_y0.append(np.sum(train_data["project_is_approved"][train_data["school_state"]==val] == 0))    
trace1 = go.Bar(
    x = temp.index,
    y = temp_y1,
    name='Accepted Proposals'
)
trace2 = go.Bar(
    x = temp.index,
    y = temp_y0, 
    name='Rejected Proposals'
)

data = [trace1, trace2]
layout = go.Layout(
    title = "Popular School states in terms of project acceptance rate and project rejection rate",
    barmode='stack',
    width = 1000
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)

# ### <a id='ptptppa'>5.5.b Popular Teacher Prefix in terms of project acceptance rate and project rejection rate</a>

# In[ ]:


temp = train_data["teacher_prefix"].value_counts()
temp_y0 = []
temp_y1 = []
for val in temp.index:
    temp_y1.append(np.sum(train_data["project_is_approved"][train_data["teacher_prefix"]==val] == 1))
    temp_y0.append(np.sum(train_data["project_is_approved"][train_data["teacher_prefix"]==val] == 0))    
trace1 = go.Bar(
    x = temp.index,
    y = temp_y1,
    name='Accepted Proposals'
)
trace2 = go.Bar(
    x = temp.index,
    y = temp_y0, 
    name='Rejected Proposals'
)

data = [trace1, trace2]
layout = go.Layout(
    title = "Popular Teacher prefixes in terms of project acceptance rate and project rejection rate",
    barmode='stack',
    width = 1000
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)

# ### <a id='psgltpp'>5.5.c Popular school grade levels in terms of project acceptance rate and project rejection rate</a>

# In[ ]:


temp = train_data["project_grade_category"].value_counts()
#print(temp.values)
temp_y0 = []
temp_y1 = []
for val in temp.index:
    temp_y1.append(np.sum(train_data["project_is_approved"][train_data["project_grade_category"]==val] == 1))
    temp_y0.append(np.sum(train_data["project_is_approved"][train_data["project_grade_category"]==val] == 0))    
trace1 = go.Bar(
    x = temp.index,
    y = temp_y1,
    name='Accepted Proposals'
)
trace2 = go.Bar(
    x = temp.index,
    y = temp_y0, 
    name='Rejected Proposals'
)

data = [trace1, trace2]
layout = go.Layout(
    title = "Popular school grade levels in terms of project acceptance rate and project rejection rate",
    barmode='stack',
    width = 1000
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)

# ### <a id='pcoptppa'>5.5.d Popular category of the project in terms of project acceptance rate and project rejection rate</a>

# In[ ]:


temp = train_data["project_subject_categories"].value_counts()
#print(temp.values)
temp_y0 = []
temp_y1 = []
for val in temp.index:
    temp_y1.append(np.sum(train_data["project_is_approved"][train_data["project_subject_categories"]==val] == 1))
    temp_y0.append(np.sum(train_data["project_is_approved"][train_data["project_subject_categories"]==val] == 0))    
trace1 = go.Bar(
    x = temp.index,
    y = temp_y1,
    name='Accepted Proposals'
)
trace2 = go.Bar(
    x = temp.index,
    y = temp_y0, 
    name='Rejected Proposals'
)

data = [trace1, trace2]
layout = go.Layout(
    title = "Popular category of the project in terms of project acceptance rate and project rejection rate",
    barmode='stack',
    width = 1000
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)

# ### <a id='psotpppan'>5.5.e Popular subcategory of the project in terms of project acceptance rate and project rejection rate</a>

# In[ ]:


temp = train_data["project_subject_subcategories"].value_counts()
#print(temp.values)
temp_y0 = []
temp_y1 = []
for val in temp.index:
    temp_y1.append(np.sum(train_data["project_is_approved"][train_data["project_subject_subcategories"]==val] == 1))
    temp_y0.append(np.sum(train_data["project_is_approved"][train_data["project_subject_subcategories"]==val] == 0))    
trace1 = go.Bar(
    x = temp.index,
    y = temp_y1,
    name='Accepted Proposals'
)
trace2 = go.Bar(
    x = temp.index,
    y = temp_y0, 
    name='Rejected Proposals'
)

data = [trace1, trace2]
layout = go.Layout(
    title = "Popular subcategory of the project in terms of project acceptance rate and project rejection rate",
    barmode='stack',
    width = 1000
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)

# ### <a id='ppttpan'>5.5.f Popular project titles in terms of project acceptance rate and project rejection rate</a>

# In[ ]:


temp = train_data["project_title"].value_counts().head(20)
#print(temp.values)
temp_y0 = []
temp_y1 = []
for val in temp.index:
    temp_y1.append(np.sum(train_data["project_is_approved"][train_data["project_title"]==val] == 1))
    temp_y0.append(np.sum(train_data["project_is_approved"][train_data["project_title"]==val] == 0))    
trace1 = go.Bar(
    x = temp.index,
    y = temp_y1,
    name='Accepted Proposals'
)
trace2 = go.Bar(
    x = temp.index,
    y = temp_y0, 
    name='Rejected Proposals'
)

data = [trace1, trace2]
layout = go.Layout(
    title = "Popular project titles in terms of project acceptance rate and project rejection rate",
    barmode='stack',
    width = 1000
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)

# ## <a id='ppuss'>5.6 Project Proposals by US States</a>

# In[ ]:


temp = pd.DataFrame(train_data["school_state"].value_counts()).reset_index()
temp.columns = ['state_code', 'num_proposals']

data = [dict(
        type='choropleth',
        locations= temp['state_code'],
        locationmode='USA-states',
        z=temp['num_proposals'].astype(float),
        text=temp['state_code'],
        colorscale='Red',
        marker=dict(line=dict(width=0.7)),
        colorbar=dict(autotick=False, tickprefix='', title='Number of project proposals'),
)]
layout = dict(title = 'Project Proposals by US States',geo = dict(
            scope='usa',
            projection=dict( type='albers usa' ),
            showlakes = True,
            lakecolor = 'rgb(255, 255, 255)'),
             )
fig = dict(data=data, layout=layout)
py.iplot(fig, validate=False)

# ## <a id='ppmaruss'>5.7 Project Proposals Mean Acceptance Rate by US States</a>

# In[ ]:


temp = pd.DataFrame(train_data.groupby("school_state")["project_is_approved"].apply(np.mean)).reset_index()
temp.columns = ['state_code', 'num_proposals']

data = [dict(
        type='choropleth',
        locations= temp['state_code'],
        locationmode='USA-states',
        z=temp['num_proposals'].astype(float),
        text=temp['state_code'],
        colorscale='Red',
        marker=dict(line=dict(width=0.7)),
        colorbar=dict(autotick=False, tickprefix='', title='Number of project proposals'),
)]
layout = dict(title = 'Project Proposals Mean Acceptance Rate by US States',geo = dict(
            scope='usa',
            projection=dict( type='albers usa' ),
            showlakes = True,
            lakecolor = 'rgb(255, 255, 255)'),
             )
fig = dict(data=data, layout=layout)
py.iplot(fig, validate=False)

# ## <a id='cmhmtd'>5.8 Correlation Matrix and HeatMap of training data</a>

# ### <a id='tppiaic'>5.8.a Teacher_prefix and project_is_approved Intervals Correlation</a>

# In[ ]:


cols = ['teacher_prefix', 'project_is_approved']
cm = sns.light_palette("red", as_cmap=True)
pd.crosstab(train_data[cols[0]], train_data[cols[1]]).style.background_gradient(cmap = cm)

# ### <a id='tnppppaic'>5.8.b Teacher_number_of_previously_posted_projects and project_is_approved Intervals Correlation</a>

# In[ ]:


cols = ['teacher_number_of_previously_posted_projects', 'project_is_approved']
cm = sns.light_palette("red", as_cmap=True)
pd.crosstab(train_data[cols[0]], train_data[cols[1]]).style.background_gradient(cmap = cm)

# *  Number of previously posted applications by the submitting teacher was** Zero(0)** having more number of acceptance rate.

# ### <a id='cmaht'>5.8.c Correlation Matrix and Heatmap of training data</a>

# In[ ]:


#Correlation Matrix
corr = train_data.corr()
plt.figure(figsize=(12,12))
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values, annot=True, cmap='cubehelix', square=True)
plt.title('Correlation between different features')
corr

# ## <a id='psta'>5.9 Project Submission Time Analysis</a>

# In[ ]:


train_data["project_submitted_datetime"] = pd.to_datetime(train_data["project_submitted_datetime"])
train_data["month_created"] = train_data["project_submitted_datetime"].dt.month
train_data["weekday_created"] = train_data["project_submitted_datetime"].dt.weekday
train_data["date_created"] = train_data["project_submitted_datetime"].dt.date
train_data["hour_created"] = train_data["project_submitted_datetime"].dt.hour

# ### <a id='psma'>5.9.a Project Submission Month Analysis</a>

# In[ ]:


temp = train_data["month_created"].value_counts()
#print(temp.values)
temp_y0 = []
temp_y1 = []
for val in temp.index:
    temp_y1.append(np.sum(train_data["project_is_approved"][train_data["month_created"]==val] == 1))
    temp_y0.append(np.sum(train_data["project_is_approved"][train_data["month_created"]==val] == 0))
    
trace1 = go.Bar(
    x = temp.index,
    y = temp_y1,
    name='Accepted Proposals'
)
trace2 = go.Bar(
    x = temp.index,
    y = temp_y0, 
    name='Rejected Proposals'
)

data = [trace1, trace2]
layout = go.Layout(
    title = "Project Proposal Submission Month Distribution",
    barmode='stack',
    width = 1000
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)


# * **August month** has the second  number of proposals followed by **September month** .

# ### <a id='pswa'>5.9.b Project Submission Weekday Analysis</a>

# In[ ]:


temp = train_data["weekday_created"].value_counts()
#print(temp.values)
temp_y0 = []
temp_y1 = []
for val in temp.index:
    temp_y1.append(np.sum(train_data["project_is_approved"][train_data["weekday_created"]==val] == 1))
    temp_y0.append(np.sum(train_data["project_is_approved"][train_data["weekday_created"]==val] == 0))
 
temp.index = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
trace1 = go.Bar(
    x = temp.index,
    y = temp_y1,
    name='Accepted Proposals'
)
trace2 = go.Bar(
    x = temp.index,
    y = temp_y0, 
    name='Rejected Proposals'
)

data = [trace1, trace2]
layout = go.Layout(
    title = "Project Proposal Submission weekday Distribution",
    barmode='stack',
    width = 1000
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)

# * The number of proposals decreases as we move towards the end of the week.

# ### <a id='psda'>5.9.c Project Submission Date Analysis</a>

# In[ ]:


temp = train_data["date_created"].value_counts()
#print(temp.values)
temp_y0 = []
temp_y1 = []
for val in temp.index:
    temp_y1.append(np.sum(train_data["project_is_approved"][train_data["date_created"]==val] == 1))
    temp_y0.append(np.sum(train_data["project_is_approved"][train_data["date_created"]==val] == 0))
 
trace1 = go.Bar(
    x = temp.index,
    y = temp_y1,
    name='Accepted Proposals'
)
trace2 = go.Bar(
    x = temp.index,
    y = temp_y0, 
    name='Rejected Proposals'
)

data = [trace1, trace2]
layout = go.Layout(
    title = "Project Proposal Submission date Distribution",
    barmode='stack',
    width = 1000
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)

# * Looks like we have approximately one years' worth of data (May 2016 to April 2017) given in the training set.
# * There is a sudden spike on a single day (Sep 1, 2016) with respect to the number of proposals (may be some specific reason?)

# ### <a id='psha'>5.9.d Project Submission Hour Analysis</a>

# In[ ]:


temp = train_data["hour_created"].value_counts()
#print(temp.values)
temp_y0 = []
temp_y1 = []
for val in temp.index:
    temp_y1.append(np.sum(train_data["project_is_approved"][train_data["hour_created"]==val] == 1))
    temp_y0.append(np.sum(train_data["project_is_approved"][train_data["hour_created"]==val] == 0))
 
trace1 = go.Bar(
    x = temp.index,
    y = temp_y1,
    name='Accepted Proposals'
)
trace2 = go.Bar(
    x = temp.index,
    y = temp_y0, 
    name='Rejected Proposals'
)

data = [trace1, trace2]
layout = go.Layout(
    title = "Project Proposal Submission Hour Distribution",
    barmode='stack',
    width = 1000
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)

# * From Hours 03 to 05, number of proposals decreases.
# * Hours 06 to 14, number of proposals increases.
# * At Hour 14 has more number of proposals.

# ## <a id='tkips1'>5.10 Top Keywords in project_essay_1</a>

# In[ ]:


temp_data = train_data.dropna(subset=['project_essay_1'])
# converting into lowercase
temp_data['project_essay_1'] = temp_data['project_essay_1'].apply(lambda x: " ".join(x.lower() for x in x.split()))
temp_data['project_essay_1'] = temp_data['project_essay_1'].map(text_prepare)


from wordcloud import WordCloud

wordcloud = WordCloud(max_font_size=50, width=600, height=300).generate(' '.join(temp_data['project_essay_1'].values))
plt.figure(figsize=(15,8))
plt.imshow(wordcloud)
plt.title("Top Keywords in project_essay_1", fontsize=35)
plt.axis("off")
plt.show() 

# ## <a id='tkipe2'>5.11 Top keywords in project_essay_2</a>

# In[ ]:


temp_data = train_data.dropna(subset=['project_essay_2'])
# converting into lowercase
temp_data['project_essay_2'] = temp_data['project_essay_2'].apply(lambda x: " ".join(x.lower() for x in x.split()))
temp_data['project_essay_2'] = temp_data['project_essay_2'].map(text_prepare)


from wordcloud import WordCloud

wordcloud = WordCloud(max_font_size=50, width=600, height=300).generate(' '.join(temp_data['project_essay_2'].values))
plt.figure(figsize=(15,8))
plt.imshow(wordcloud)
plt.title("Top Keywords in project_essay_2", fontsize=35)
plt.axis("off")
plt.show() 

# ## <a id='tkinprs'>5.12 Top Keywords in project_resource_summary</a>

# In[ ]:


temp_data = train_data.dropna(subset=['project_resource_summary'])
# converting into lowercase
temp_data['project_resource_summary'] = temp_data['project_resource_summary'].apply(lambda x: " ".join(x.lower() for x in x.split()))
temp_data['project_resource_summary'] = temp_data['project_resource_summary'].map(text_prepare)


from wordcloud import WordCloud

wordcloud = WordCloud(max_font_size=50, width=600, height=300).generate(' '.join(temp_data['project_resource_summary'].values))
plt.figure(figsize=(15,8))
plt.imshow(wordcloud)
plt.title("Top Keywords in project_resource_summary2", fontsize=35)
plt.axis("off")
plt.show() 

# ##  <a id='qvp'>5.13 Quantity V.S. Price</a>

# In[ ]:


#iplot([go.Scatter(x=train_resource['quantity'], y=train_resource['price'], mode='markers')])
iplot([go.Histogram2dContour(x=train_resource.head(1000)['quantity'], 
                             y=train_resource.head(1000)['price'], 
                             contours=go.Contours(coloring='heatmap')),
       go.Scatter(x=train_resource.head(5000)['quantity'], y=train_resource.head(1000)['price'], mode='markers')])

# In[ ]:


populated_states = train_resource[:50]

data = [go.Scatter(
    y = populated_states['quantity'],
    x = populated_states['price'],
    mode='markers+text',
    marker=dict(
        size= np.log(populated_states.price) - 2,
        color=populated_states['quantity'],
        colorscale='Portland',
        showscale=True
    ),
    text=populated_states['school_state'],
    textposition=["top center"]
)]
layout = go.Layout(
    title='Quantity V.S. Price',
    xaxis= dict(title='price'),
    yaxis=dict(title='Quantity')
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)

# ## <a id ='gapc'> 5.14 Gender Analysis</a>

# We will create a new column from teacher_prefix and mapping follow as :
# 
# * Mrs, Ms --> Female
# * Mr. --> Male
# * Teacher, Dr --> Unknown

# In[ ]:


# Creating the gender column
gender_mapping = {"Mrs.": "Female", "Ms.":"Female", "Mr.":"Male", "Teacher":"Unknown", "Dr.":"Unknown", np.nan:"Unknown"  }
train_data["gender"] = train_data.teacher_prefix.map(gender_mapping)
test_data["gender"] = test_data.teacher_prefix.map(gender_mapping)
train_resource["gender"] = train_resource.teacher_prefix.map(gender_mapping)
test_resource["gender"] = test_resource.teacher_prefix.map(gender_mapping)

# In[ ]:


temp = train_data["gender"].value_counts()
trace = go.Bar(
    x = temp.index,
    y = (temp / temp.sum())*100,
)
data = [trace]
layout = go.Layout(
    title = "Gender in terms of projects proposals submitted in % ",
    xaxis=dict(
        title='Gender',
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
    ),
    yaxis=dict(
        title='Count of project proposals submitted in % ',
        titlefont=dict(
            size=16,
            color='rgb(107, 107, 107)'
        ),
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
)
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)

# * Female having more count which is approx. **88 %** than Male which has **10 %** in terms of projects proposals submissions.

# # <a id='mwdppes'>5.15 Month wise distribution of number of projects proposal submitted in each state</a>

# In[ ]:


#train_data['Month_wise'] = kiva_loans_data.date.dt.year
train_resource["project_submitted_datetime"] = pd.to_datetime(train_resource["project_submitted_datetime"])
train_resource["month_created"] = train_resource["project_submitted_datetime"].dt.month
loan = train_resource.groupby(['school_state', 'month_created'])['price'].mean().unstack()
loan = loan.sort_values([3], ascending=False)
f, ax = plt.subplots(figsize=(15, 20)) 
loan = loan.fillna(0)
temp = sns.heatmap(loan, cmap='Reds')
plt.show()

# * USA state **WY** was having more price requested for resources in **March** month than others.

# ## <a id='prfrd'>5.16 Price requested for resources distribution</a>

# ### <a id='prfrbds'>5.16.a Price requested for resources distribution by different states</a>

# In[ ]:


# trace = []
# for name, group in train_resource.groupby("school_state"):
#     trace.append ( 
#         go.Box(
#             x=group["price"].values,
#             name=name
#         )
#     )
# layout = go.Layout(
#     title='price requested for resources distributiom by different states ',
#     width = 800,
#     height = 800
# )
# #data = [trace0, trace1]
# fig = go.Figure(data=trace, layout=layout)
# py.iplot(fig)

# * As we can see most of the price requested for resources is between **0 to 2k dollar**.

# ### <a id='prfrbtp'>5.16.b Price requested for resources distribution by Teacher prefixes</a>

# In[ ]:


# trace = []
# for name, group in train_resource.groupby("teacher_prefix"):
#     trace.append ( 
#         go.Box(
#             x=group["price"].values,
#             name=name
#         )
#     )
# layout = go.Layout(
#     title='price requested for resources distributiom by techer_prefixes ',
#     width = 800,
#     height = 800
# )
# #data = [trace0, trace1]
# fig = go.Figure(data=trace, layout=layout)
# py.iplot(fig)

# * Mostly price requested for resources is 
#    * 0 to 2k Dollar by **teacher** prefix
#    * 0 to 4k Dolar by **Ms. , Mrs. and Mr.** prefixes 
#    * 0 to 500 Dollar by **Dr.** prefix.

# ### <a id='prfrddga'>5.16.c Price requested for resources distribution by different Genders</a>

# In[ ]:


# trace = []
# for name, group in train_resource.groupby("gender"):
#     trace.append ( 
#         go.Box(
#             x=group["price"].values,
#             name=name
#         )
#     )
# layout = go.Layout(
#     title='price requested for resources distributiom by different genders ',
#     width = 800,
#     height = 800
# )
# #data = [trace0, trace1]
# fig = go.Figure(data=trace, layout=layout)
# py.iplot(fig)

# * Mostly price requested for resources is 
#    * 0 to 2k Dollar by **Unknowns**
#    * 0 to 4k Dolar by **Males** 
#    * 0 to 5k Dollar by **Females**.

# ### <a id='prfrddpgc'>5.16.d Price requested for resources distribution by different project_grade_category</a>

# In[ ]:


# trace = []
# for name, group in train_resource.groupby("project_grade_category"):
#     trace.append ( 
#         go.Box(
#             x=group["price"].values,
#             name=name
#         )
#     )
# layout = go.Layout(
#     title='price requested for resources distributiom by different project_grade_categories ',
#     width = 800,
#     height = 800
# )
# #data = [trace0, trace1]
# fig = go.Figure(data=trace, layout=layout)
# py.iplot(fig)

# * Mostly price requested for resources is between approx. ** 0 to 4k** **Dollar**  for all type of project grade categories.

# ## <a id='ca'>5.17 CA(California)</a>

# ### <a id='potpic'>5.17.a Popularities of Teacher prefixes in California</a>

# In[ ]:


temp1 = pd.DataFrame(train_data[train_data["school_state"]=='CA'])
temp = temp1["teacher_prefix"].value_counts()
temp_y0 = []
temp_y1 = []
for val in temp.index:
    temp_y1.append(np.sum(train_data["project_is_approved"][train_data["teacher_prefix"]==val] == 1))
    temp_y0.append(np.sum(train_data["project_is_approved"][train_data["teacher_prefix"]==val] == 0))    
trace1 = go.Bar(
    x = temp.index,
    y = temp_y1,
    name='Accepted Proposals'
)
trace2 = go.Bar(
    x = temp.index,
    y = temp_y0, 
    name='Rejected Proposals'
)

data = [trace1, trace2]
layout = go.Layout(
    title = "Popular Teacher prefixes in terms of project proposal approved or not in California",
    barmode='stack',
    width = 1000
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)

# ### <a id='posglic'>5.17.b Popularities of school grade levels in California</a>

# In[ ]:


temp = temp1["project_grade_category"].value_counts()
#print(temp.values)
temp_y0 = []
temp_y1 = []
for val in temp.index:
    temp_y1.append(np.sum(train_data["project_is_approved"][train_data["project_grade_category"]==val] == 1))
    temp_y0.append(np.sum(train_data["project_is_approved"][train_data["project_grade_category"]==val] == 0))    
trace1 = go.Bar(
    x = temp.index,
    y = temp_y1,
    name='Accepted Proposals'
)
trace2 = go.Bar(
    x = temp.index,
    y = temp_y0, 
    name='Rejected Proposals'
)

data = [trace1, trace2]
layout = go.Layout(
    title = "Popular school grade levels in terms of project proposal approved or not in California",
    barmode='stack',
    width = 1000
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)

# ### <a id='tptic'>5.17.c Top project titles in California</a>

# In[ ]:


project_title_data = temp1['project_title']
percentages = round(project_title_data.value_counts() / len(project_title_data) * 100, 2)[:13]
trace = go.Pie(labels=percentages.keys(), values=percentages.values, hoverinfo='label+percent', 
                textfont=dict(size=18, color='#000000'))
data = [trace]
layout = go.Layout(width=800, height=800, title='Top project titles in California',titlefont= dict(size=20), 
                   legend=dict(x=0.1,y=-5))

fig = go.Figure(data=data, layout=layout)
offline.iplot(fig, show_link=False)

# ### <a id='topstic'>5.17.d Trend of project submission time in California</a>

# In[ ]:


temp = temp1["date_created"].value_counts()
#print(temp.values)
temp_y0 = []
temp_y1 = []
for val in temp.index:
    temp_y1.append(np.sum(train_data["project_is_approved"][train_data["date_created"]==val] == 1))
    temp_y0.append(np.sum(train_data["project_is_approved"][train_data["date_created"]==val] == 0))
    
trace1 = go.Bar(
    x = temp.index,
    y = temp_y1,
    name='Accepted Proposals'
)
trace2 = go.Bar(
    x = temp.index,
    y = temp_y0, 
    name='Rejected Proposals'
)

data = [trace1, trace2]
layout = go.Layout(
    title = "Project Proposal Submission Date Distribution in California",
    barmode='stack',
    width = 1000
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)


# ## <a id='TX'>5.18 TX(Texas)</a>

# ### <a id='potpit'>5.18.a Popularities of Teacher prefixes in Texas</a>

# In[ ]:


temp1 = pd.DataFrame(train_data[train_data["school_state"]=='TX'])
temp = temp1["teacher_prefix"].value_counts()
temp_y0 = []
temp_y1 = []
for val in temp.index:
    temp_y1.append(np.sum(train_data["project_is_approved"][train_data["teacher_prefix"]==val] == 1))
    temp_y0.append(np.sum(train_data["project_is_approved"][train_data["teacher_prefix"]==val] == 0))    
trace1 = go.Bar(
    x = temp.index,
    y = temp_y1,
    name='Accepted Proposals'
)
trace2 = go.Bar(
    x = temp.index,
    y = temp_y0, 
    name='Rejected Proposals'
)

data = [trace1, trace2]
layout = go.Layout(
    title = "Popular Teacher prefixes in terms of project proposal approved or not in Texas",
    barmode='stack',
    width = 1000
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)

# ### <a id='posglit'>5.18.b Popularities of school grade levels in Texas</a>

# In[ ]:


temp = temp1["project_grade_category"].value_counts()
#print(temp.values)
temp_y0 = []
temp_y1 = []
for val in temp.index:
    temp_y1.append(np.sum(train_data["project_is_approved"][train_data["project_grade_category"]==val] == 1))
    temp_y0.append(np.sum(train_data["project_is_approved"][train_data["project_grade_category"]==val] == 0))    
trace1 = go.Bar(
    x = temp.index,
    y = temp_y1,
    name='Accepted Proposals'
)
trace2 = go.Bar(
    x = temp.index,
    y = temp_y0, 
    name='Rejected Proposals'
)

data = [trace1, trace2]
layout = go.Layout(
    title = "Popular school grade levels in terms of project proposal approved or not in Texas",
    barmode='stack',
    width = 1000
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)

# ### <a id='tptit'>5.18.c Top project titles in Texas</a>

# In[ ]:


project_title_data = temp1['project_title']
percentages = round(project_title_data.value_counts() / len(project_title_data) * 100, 2)[:13]
trace = go.Pie(labels=percentages.keys(), values=percentages.values, hoverinfo='label+percent', 
                textfont=dict(size=18, color='#000000'))
data = [trace]
layout = go.Layout(width=800, height=800, title='Top project titles in Texas',titlefont= dict(size=20), 
                   legend=dict(x=0.1,y=-5))

fig = go.Figure(data=data, layout=layout)
offline.iplot(fig, show_link=False)

# ### <a id='topstit'>5.18.d Trend of project submission time in Texas</a>

# In[ ]:


temp = temp1["date_created"].value_counts()
#print(temp.values)
temp_y0 = []
temp_y1 = []
for val in temp.index:
    temp_y1.append(np.sum(train_data["project_is_approved"][train_data["date_created"]==val] == 1))
    temp_y0.append(np.sum(train_data["project_is_approved"][train_data["date_created"]==val] == 0))
    
trace1 = go.Bar(
    x = temp.index,
    y = temp_y1,
    name='Accepted Proposals'
)
trace2 = go.Bar(
    x = temp.index,
    y = temp_y0, 
    name='Rejected Proposals'
)

data = [trace1, trace2]
layout = go.Layout(
    title = "Project Proposal Submission Date Distribution in Texas",
    barmode='stack',
    width = 1000
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)

# # <a id='fea'>6. Feature Engineering</a>

# In[ ]:


cols_to_use = ['id','teacher_id','teacher_prefix','school_state','project_submitted_datetime','project_grade_category',
               'project_subject_categories','project_subject_subcategories','project_title','project_essay_1','project_essay_2',
               'project_resource_summary','teacher_number_of_previously_posted_projects']
y=train_data['project_is_approved']

# In[ ]:


train_data = train_data[cols_to_use]
test_data = test_data[cols_to_use]

# ## <a id='fcfpsd'>6.1 Features Created from project_submitted_datetime</a>

# Total 6 features created from project_submitted_datetime :
# * **year**
# * **month**
# * **weekday**
# * **hour**
# * **month_day**
# * **year_day**

# In[ ]:


start_time=time.time()
# *******train data *********
# extracting datetime features using project_submitted_datetime 
train_data['project_submitted_datetime'] = pd.to_datetime(train_data['project_submitted_datetime'])
test_data['project_submitted_datetime'] = pd.to_datetime(test_data['project_submitted_datetime'])

train_data["year"] = train_data["project_submitted_datetime"].dt.year
train_data["month"] = train_data["project_submitted_datetime"].dt.month
train_data['weekday'] = train_data['project_submitted_datetime'].dt.weekday
train_data["hour"] = train_data["project_submitted_datetime"].dt.hour
train_data["month_Day"] = train_data['project_submitted_datetime'].dt.day
train_data["year_Day"] = train_data['project_submitted_datetime'].dt.dayofyear
train_data = train_data.drop('project_submitted_datetime', axis=1)

# ****** test data *********
test_data["year"] = test_data["project_submitted_datetime"].dt.year
test_data["month"] = test_data["project_submitted_datetime"].dt.month
test_data['weekday'] = test_data['project_submitted_datetime'].dt.weekday
test_data["hour"] = test_data["project_submitted_datetime"].dt.hour
test_data["month_Day"] = test_data['project_submitted_datetime'].dt.day
test_data["year_Day"] = test_data['project_submitted_datetime'].dt.dayofyear
test_data = test_data.drop('project_submitted_datetime', axis=1)

end_time=time.time()
print("total time in the cuurent cell ",end_time-start_time,"s")

# ## <a id='fcfpppr'>6.2 Features created from project_essay_1, project_essay_2, project_title, project_resource_summary</a>

# Here we created total 11 features :
# * **e1_length**
# * **e2_length**
# * **project_title_len**
# * **project_resource_summary_len**
# * **char_count**
# * **word_count**
# * **word_density**
# * **punctuation_count**
# * **title_word_count**
# * **upper_case_word_count**
# * **stopword_count**

# In[ ]:


start_time=time.time()

# Thanks sban for such a great features engineering
# https://www.kaggle.com/shivamb/extensive-feature-engineering-updated/notebook
# *******train data *********
train_data['e1_length'] = train_data['project_essay_1'].apply(lambda x: len(str(x))) # Essay1 length
train_data['e2_length'] = train_data['project_essay_2'].apply(lambda x: len(str(x))) # Essay2 length
train_data['project_title_len'] = train_data['project_title'].apply(lambda x: len(str(x))) # Title length
train_data['project_resource_summary_len'] = train_data['project_resource_summary'].apply(lambda x: len(str(x))) # project resource summary length

# combine the project essays to create a complete essay text
train_data['text'] = train_data.apply(lambda row: ' '.join([str(row['project_essay_1']), 
                                            str(row['project_essay_2']), 
                                            ]), axis=1)

# extract features from text
train_data['char_count'] = train_data['text'].apply(len)
train_data['word_count'] = train_data['text'].apply(lambda x: len(x.split()))
train_data['word_density'] = train_data['char_count'] / (train_data['word_count']+1)
train_data['punctuation_count'] = train_data['text'].apply(lambda x: len("".join(_ for _ in x if _ in punctuation))) 
train_data['title_word_count'] = train_data['text'].apply(lambda x: len([wrd for wrd in x.split() if wrd.istitle()]))
train_data['upper_case_word_count'] = train_data['text'].apply(lambda x: len([wrd for wrd in x.split() if wrd.isupper()]))
train_data['stopword_count'] = train_data['text'].apply(lambda x: len([wrd for wrd in x.split() if wrd.lower() in stop_words]))

# ****** test data *********
test_data['e1_length'] = test_data['project_essay_1'].apply(lambda x: len(str(x))) # Essay1 length
test_data['e2_length'] = test_data['project_essay_2'].apply(lambda x: len(str(x))) # Essay2 length
test_data['project_title_len'] = test_data['project_title'].apply(lambda x: len(str(x))) # Title length
test_data['project_resource_summary_len'] = test_data['project_resource_summary'].apply(lambda x: len(str(x))) # project resource summary length

# combine the project essays to create a complete essay text
test_data['text'] = test_data.apply(lambda row: ' '.join([str(row['project_essay_1']), 
                                            str(row['project_essay_2']), 
                                            ]), axis=1)

# extract features from text
test_data['char_count'] = test_data['text'].apply(len)
test_data['word_count'] = test_data['text'].apply(lambda x: len(x.split()))
test_data['word_density'] = test_data['char_count'] / (test_data['word_count']+1)
test_data['punctuation_count'] = test_data['text'].apply(lambda x: len("".join(_ for _ in x if _ in punctuation))) 
test_data['title_word_count'] = test_data['text'].apply(lambda x: len([wrd for wrd in x.split() if wrd.istitle()]))
test_data['upper_case_word_count'] = test_data['text'].apply(lambda x: len([wrd for wrd in x.split() if wrd.isupper()]))
test_data['stopword_count'] = test_data['text'].apply(lambda x: len([wrd for wrd in x.split() if wrd.lower() in stop_words]))

end_time=time.time()
print("total time in the cuurent cell ",end_time-start_time,"s")

# ## <a id='fracttr'>6.3 Features created after combining the training data, test data with resource file</a>

# Here we created total 4 features :
# 
# * **resources_total**
# * **resources_total_mean**
# * **resources_quantity_sum**
# * **resources_quantity_count**

# In[ ]:


start_time=time.time()

# Combine resources file
# Thanks, the1owl! 
# https://www.kaggle.com/the1owl/the-choice-is-yours

resources_data['resources_total'] = resources_data['quantity'] * resources_data['price']

dfr_sum = resources_data.groupby(['id'], as_index=False)[['resources_total']].sum()
dfr_mean = resources_data.groupby(['id'], as_index=False)[['resources_total']].mean()
dfr_mean = dfr_mean.rename(columns={'resources_total':'resources_total_mean'})
dfq_sum = resources_data.groupby(['id'], as_index=False)[['quantity']].sum()
dfq_sum = dfq_sum.rename(columns={'quantity':'resources_quantity_sum'})
dfq_count = resources_data.groupby(['id'], as_index=False)[['quantity']].count()
dfq_count = dfq_count.rename(columns={'quantity':'resources_quantity_count'})

# ****** train data ************
train_data = pd.merge(train_data, dfr_sum, how='left', on='id').fillna(-1)
train_data = pd.merge(train_data, dfr_mean, how='left', on='id').fillna(-1)
train_data = pd.merge(train_data, dfq_sum, how='left', on='id').fillna(-1)
train_data = pd.merge(train_data, dfq_count, how='left', on='id').fillna(-1)

# ****** test data ***************
test_data = pd.merge(test_data, dfr_sum, how='left', on='id').fillna(-1)
test_data = pd.merge(test_data, dfr_mean, how='left', on='id').fillna(-1)
test_data = pd.merge(test_data, dfq_sum, how='left', on='id').fillna(-1)
test_data = pd.merge(test_data, dfq_count, how='left', on='id').fillna(-1)

end_time=time.time()
print("total time in the cuurent cell ",end_time-start_time,"s")

# ## <a id='gfc'>6.4 Gender feature creation</a>

# Here we created **gender** feature from **teacher_prefix** feature

# In[ ]:


# Creating the gender column
gender_mapping = {"Mrs.": "Female", "Ms.":"Female", "Mr.":"Male", "Teacher":"Unknown", "Dr.":"Unknown", np.nan:"Unknown"  }
train_data["gender"] = train_data.teacher_prefix.map(gender_mapping)
test_data["gender"] = test_data.teacher_prefix.map(gender_mapping)

# ## <a id='ccftnf'>6.5 Converting Categorical featues to numericals features</a>

# Here i have converted 7 Categorical featues to numericals features :
# 
# * **teacher_id**
# * **teacher_prefix**
# * **school_state**
# * **project_grade_category**
# * **project_subject_categories**
# * **project_subject_subcategories**
# * **gender**

# In[ ]:


from sklearn import preprocessing
def encode_features(df_train, df_test):
    features = [
    'teacher_id', 
    'teacher_prefix', 
    'school_state', 
    'project_grade_category',
    'project_subject_categories', 
    'project_subject_subcategories',
    'gender']
    df_combined = pd.concat([df_train[features], df_test[features]])
    
    for feature in features:
        le = preprocessing.LabelEncoder()
        le = le.fit(df_combined[feature].astype(str))
        df_train[feature] = le.transform(df_train[feature].astype(str))
        df_test[feature] = le.transform(df_test[feature].astype(str))
    return df_train, df_test
    
train_data, test_data = encode_features(train_data, test_data)

# ## <a id='ruf'>6.6 Removing Unwanted features</a>

# The unwanted featues are :
#  
# * **id**
# * **project_title**
# * **project_essay_1**
# * **project_essay_2**
# * **text**
# * **project_resource_summary**

# In[ ]:


cols_to_remove = ['id','project_title','project_essay_1','project_essay_2','text','project_resource_summary']

train_data = train_data.drop(cols_to_remove, axis = 1)
test_data = test_data.drop(cols_to_remove, axis = 1)
features = train_data.columns.values

# # <a id='bmlm'>7. Building Machine Learning Models</a>

# ## <a id='lgbm'>7.1 Light GBM</a>

# In[ ]:


print("Building model using Light GBM and finding AUC(Area Under Curve)")
from sklearn.model_selection import train_test_split
X_train_l, X_valid_l, y_train_l, y_valid_l = train_test_split(train_data, y, test_size=0.10, random_state=2018)
params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'auc',
        'max_depth': 7,
        'num_leaves': 32,
        'learning_rate': 0.02,
        'feature_fraction': 0.80,
        'bagging_fraction': 0.80,
        'bagging_freq': 5,
        'verbose': 0,
        'lambda_l2': 1,
    }  

import lightgbm as lgb
evals_result = {}  # to record eval results for plotting
model_lgb = lgb.train(
        params,
        lgb.Dataset(X_train_l, y_train_l),
        num_boost_round=10000,
        valid_sets=[lgb.Dataset(X_valid_l, y_valid_l)],
        early_stopping_rounds=100,
        evals_result=evals_result,
        verbose_eval=25)
from sklearn.metrics import roc_auc_score
valid_preds_lgb = model_lgb.predict(X_valid_l, num_iteration=model_lgb.best_iteration)
test_preds = model_lgb.predict(test_data, num_iteration=model_lgb.best_iteration)
auc = roc_auc_score(y_valid_l, valid_preds_lgb)
print('AUC:',auc)

# ## <a id='xgbst'>7.2 XGBoost</a>

# In[ ]:


print("Building model using XGBoost and finding AUC(Area Under Curve)")

import xgboost as xgb
xgb_params = {'eta': 0.2, 
                  'max_depth': 5, 
                  'subsample': 0.8, 
                  'colsample_bytree': 0.8, 
                  'objective': 'binary:logistic', 
                  'eval_metric': 'auc', 
                  'seed': 1234
                 }
X_train_x, X_valid_x, y_train_x, y_valid_x = train_test_split(train_data, y, test_size=0.33, random_state=2018)
d_train = xgb.DMatrix(X_train_x, y_train_x)
d_valid = xgb.DMatrix(X_valid_x, y_valid_x)
d_test = xgb.DMatrix(test_data)
watchlist = [(d_train, 'train'), (d_valid, 'valid')]
model_xgb = xgb.train(xgb_params, d_train, 500, watchlist, verbose_eval=50, early_stopping_rounds=20)
xgb_pred_test = model_xgb.predict(d_test)
xgb_pred_valid = model_xgb.predict(d_valid)
auc = roc_auc_score(y_valid_x, xgb_pred_valid)
print('AUC:',auc)

# # <a id='feva'>8. Further Evaluation</a>

# In[3]:


from IPython.display import Image
Image('../input/confusion-matrix/confusion matrix.PNG')

# ## <a id='rocaucc'>8.1 ROC AUC Curve</a>

# **The ROC AUC Score is the corresponding score to the ROC AUC Curve. It is simply computed by measuring the area under the curve, which is called AUC.**
# 
# **A classifiers that is 100% correct, would have a ROC AUC Score of 1 and a completely random classiffier would have a score of 0.5.**

# In[ ]:


from sklearn.metrics import roc_curve
from sklearn import metrics
fpr,tpr,thresholds=roc_curve(y_valid_l, valid_preds_lgb)
roc_auc = metrics.auc(fpr, tpr)

fpr_1,tpr_1,thresholds_1=roc_curve(y_valid_x,xgb_pred_valid)
roc_auc_1 = metrics.auc(fpr_1, tpr_1)

plt.figure(figsize=(15,8))

plt.title('Receiver Operating Characteristic for different ML algorithms')

plt.plot(fpr, tpr, 'b', label = 'LGBM-AUC = %0.2f' % roc_auc)
plt.plot(fpr_1, tpr_1, 'g', label = 'XGBoost-AUC = %0.2f' % roc_auc_1)

plt.legend(loc = 'upper left')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])

plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')

plt.show()

# ## <a id='prcur'>8.2 Precision Recall curve</a>

# Precision-Recall is a useful measure of success of prediction when the classes are very imbalanced. In information retrieval, precision is a measure of result relevancy, while recall is a measure of how many truly relevant results are returned.
# 
# The precision-recall curve shows the tradeoff between precision and recall for different threshold. A high area under the curve represents both high recall and high precision, where high precision relates to a low false positive rate, and high recall relates to a low false negative rate. High scores for both show that the classifier is returning accurate results (high precision), as well as returning a majority of all positive results (high recall).
# 
# A system with high recall but low precision returns many results, but most of its predicted labels are incorrect when compared to the training labels. A system with high precision but low recall is just the opposite, returning very few results, but most of its predicted labels are correct when compared to the training labels. An ideal system with high precision and high recall will return many results, with all results labeled correctly.

# In[ ]:


from sklearn.metrics import precision_recall_curve
precision, recall, threshold = precision_recall_curve(y_valid_l, valid_preds_lgb)

def plot_precision_and_recall(precision, recall, threshold):
    plt.plot(threshold, precision[:-1], "r-", label="precision", linewidth=5)
    plt.plot(threshold, recall[:-1], "b", label="recall", linewidth=5)
    plt.xlabel("threshold", fontsize=19)
    plt.legend(loc="upper right", fontsize=19)
    plt.ylim([0, 1])

plt.figure(figsize=(15, 8))
plot_precision_and_recall(precision, recall, threshold)
plt.show()

# In[ ]:


def plot_precision_vs_recall(precision, recall):
    plt.plot(recall, precision, "g--", linewidth=2.5)
    plt.ylabel("recall", fontsize=19)
    plt.xlabel("precision", fontsize=19)
    plt.axis([0, 1.5, 0, 1.5])

plt.figure(figsize=(15, 8))
plot_precision_vs_recall(precision, recall)
plt.show()

# ## <a id='pmdtlgb'>8.3 Ploting Metrics during training of Light GBM</a>

# In[ ]:


print('Plot metrics during training...')
ax = lgb.plot_metric(evals_result, metric='auc')
plt.figure(figsize=(15, 8))
plt.show()

# ## <a id='fiblgm'>8.4 Feature importances by LightGBM</a>

# In[ ]:


lgb.plot_importance(model_lgb, max_num_features=12)
plt.title("Feature importances by LightGBM")
plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
plt.show()

# # <a id='bsc'>9. Brief Summary/Conclusion :</a>
# --------------------------------------------------------------------------
# * I have done analysis only on training data.
# * This is only a brief summary if want more details please go through my Notebook.

# * Training data is highly imbalanced that is approx. 85 % projetcs were approved and 15 % project were not approved. Majority imbalanced class is positive.
# * Out of 50 states, **California(CA)** having higher number of projects proposal submitted **approx. 14 %**  followed by **Texas(TX)(7 %)** and **Tennessee(NY)(7 %)**.
# * Out of 4 school grade levels, Project proposals submission in school grade levels is higher for **Grades Prek-2** which is approximately **41 %** followed by **Grades 3-5** which has approx. **34 %**.
# * Out of 51 Project categories,  Project proposals submission for project categories is higher  for  **Literacy & Language** which is approx. **27 %** followed by **Math & Science** which has approx. **20 %**.
# * Out of 1,82,020 Project subcategories, Project proposals submission for project sub-categoriesis is higher  for **Literacy** which is approx. **16 % ** followed by **Literacy & Mathematics** which has approx. **16 %** .
# * Out of 1,82,080 project titles, Project proposals submission for project titles is higher for **Flexible seating** which is approx. **27 %** followed by **Whiggle while your work** which has approx. **14 %**.
# * Most of the price requested for resources is between **0 to 2k dollar**.
# * Mostly price requested for resources is 
#    * 0 to 2k Dollar by **teacher** prefix
#    * 0 to 4k Dolar by **Ms. , Mrs. and Mr.** prefixes 
#    * 0 to 500 Dollar by **Dr.** prefix.
# * Mostly price requested for resources is 
#    * 0 to 2k Dollar by **Unknowns**
#    * 0 to 4k Dolar by **Males** 
#    * 0 to 5k Dollar by **Females**.
# * Mostly price requested for resources is between approx. ** 0 to 4k** **Dollar**  for all type of project grade categories.   
# * Higher number of project proposal submitted by **married womens** which is approx. **53 %**  followed by **unmarried womens** which has approx. **37 %**.
# * Project proposal submitted by **Teacher** which is approx. **2 %** is vey low as compared to **Mrs., Ms., Mr**.
# *  Number of previously posted applications by the submitting teacher was** Zero(0)** having more number of acceptance rate.
# * Female having more count which is approx. **88 %** than Male which has **10 %** in terms of projects proposals submissions.
# * USA state **WY** was having more price requested for resources in **March** month than others.
# * Most of the price requested for resources is between 0 to 2k dollar.
# * If price per project is less then more chances of approval.
# * Projects with lesser number of quantity better chances of approval.
# * If project essay description having more number number of words better chances of approval.
# * **Project Submission Time Analysis** :
#    * **September month** has the second highest number of proposals .
#    * The number of proposals decreases as we move towards the end of the week.
#    * Looks like we have approximately one years' worth of data (May 2016 to April 2017) given in the training set.
#    * There is a sudden spike on a single day (Sep 1, 2016) with respect to the number of proposals (may be some specific reason?)
#    * From Hours 03 to 05, number of proposals decreases.
#    * Hours 06 to 14, number of proposals increases.
#    * At Hour 14 has more number of proposals.

# # More To Come. Stayed Tuned !!
