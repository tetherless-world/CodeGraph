#!/usr/bin/env python
# coding: utf-8

# ![](https://www.advancementcenterwts.org/wp-content/uploads/2016/05/Donors-Choose-Logo-and-Tagline.png)

# # More To Come. Stay Tuned. !!
# If there are any suggestions/changes you would like to see in the Kernel please let me know :). Appreciate every ounce of help!
# 
# **This notebook will always be a work in progress.** Please leave any comments about further improvements to the notebook! Any feedback or constructive criticism is greatly appreciated!. **If you like it or it helps you , you can upvote and/or leave a comment :).**

#  - <a href='#1'>1. Introduction</a>  
#  - <a href='#2'>2. Retrieving the Data</a>
#      - <a href='#2-1'>2.1 Load libraries</a>
#      - <a href='#2-2'>2.2 Read the Data</a>
#  - <a href='#3'>3. Glimpse of Data</a>
#      - <a href='#3-1'>3.1 Overview of table</a>
#      - <a href='#3-2'>3.2 Statistical overview of the Data</a>
#  - <a href='#4'>4. Data preparation</a>
#      - <a href='#4-1'> 4.1 Check for missing data</a>
#  - <a href='#5'>5. Data Exploration</a>
#      - <a href='#5-1'>5.1 Histogram and Distribution of Donation Amount</a>
#      - <a href='#5-2'>5.2 Top/Distribution</a>
#          - <a href='#5-2-1'>5.2.1 Top Donor Cities</a>
#          - <a href='#5-2-2'>5.2.2 Top Donor States</a>
#          - <a href='#5-2-3'>5.2.3 Top Donor checked out carts</a>
#          - <a href='#5-2-4'>5.2.4 Distribution of Project subject categories</a>
#          - <a href='#5-2-5'>5.2.5 Distribution of Project subject Sub-categories</a>
#          - <a href='#5-2-6'>5.2.6 Distribution of Project resource categories</a>
#          - <a href='#5-2.7'>5.2.7 Distribution of school Metro Type</a>
#          - <a href='#5-2-8'>5.2.8  Distribution of School cities</a>
#          - <a href='#5-2-9'>5.2.9 Distribution of School County</a>
#          - <a href='#5-2-10'>5.2.10 Distribution of Projects Grade Level Category</a>
#      - <a href='#5-3'>5.3 Donor is Teacher or not</a>
#      - <a href='#5.4'>5.4 Whether or not the donation included an optional donation.</a>
#      - <a href='#5-5'>5.5 Types of Projects</a>
#      - <a href='#5-6'>5.6 Projects were fully funded or not</a>
#      - <a href='#5-7'>5.7 Top Keywords from project Essay</a>
#      - <a href='#5-8'>5.8 Top Keywords from project Title</a>
#      - <a href='#5-9'>5.9 Donations Given by different States</a>
#      - <a href='#5-10'>5.10 Number of schools in different states</a>
#      - <a href='#5-11'>5.11 School Percentage Free Lunch for School Metro Type </a>
#      - <a href='#5-12'>5.12 Number of Donations given by donor</a>
#      - <a href='#5-13'>5.13 Gender Analysis</a>
#          - <a href='#5-13-1'>5.13.1 Distribution of Teachers Prefix</a>
#          - <a href='#5-13-2'>5.13.2 Males V.S. Female</a>
#  - <a href='#6'>6. Brief summary/Conclusions</a>

# # <a id='1'>1. Introduction</a>

# Founded in 2000 by a Bronx history teacher, DonorsChoose.org has raised $685 million for America's classrooms. Teachers at three-quarters of all the public schools in the U.S. have come to DonorsChoose.org to request what their students need, making DonorsChoose.org the leading platform for supporting public education.
# 
# To date, 3 million people and partners have funded 1.1 million DonorsChoose.org projects. But teachers still spend more than a billion dollars of their own money on classroom materials. To get students what they need to learn, the team at DonorsChoose.org needs to be able to connect donors with the projects that most inspire them.

# # <a id='2'>2. Retrieving the Data</a>

# ## <a id='2-1'>2.1 Load libraries</a>

# In[73]:


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
from plotly import tools
from mpl_toolkits.basemap import Basemap
from numpy import array
from matplotlib import cm

# import cufflinks and offline mode
import cufflinks as cf
cf.go_offline()

from sklearn import preprocessing
# Supress unnecessary warnings so that presentation looks clean
import warnings
warnings.filterwarnings("ignore")

# Print all rows and columns
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)



# ## <a id='2-2'>2.2 Read the Data</a>

# In[2]:


donations = pd.read_csv('../input/Donations.csv')
donors = pd.read_csv('../input/Donors.csv', low_memory=False)
schools = pd.read_csv('../input/Schools.csv', error_bad_lines=False)
teachers = pd.read_csv('../input/Teachers.csv', error_bad_lines=False)
projects = pd.read_csv('../input/Projects.csv', error_bad_lines=False, warn_bad_lines=False, parse_dates=["Project Posted Date","Project Fully Funded Date"])
resources = pd.read_csv('../input/Resources.csv', error_bad_lines=False, warn_bad_lines=False)

# In[3]:


# Merge donation data with donor data 
donors_donations = donations.merge(donors, on='Donor ID', how='inner')

# # <a id='3'>3. Glimpse of Data</a>

# ## <a id='3-1'>3.1 Overview of table</a>

# **Donation data**

# In[71]:


donations.head()

# **Donors Data**

# In[36]:


donors.head()

# **Schools Data**

# In[ ]:


schools.head()

# **Teachers data**

# In[ ]:


teachers.head()

# **Projects data**

# In[ ]:


projects.head()

# **Resources data**

# In[ ]:


resources.head()

# **donors_donations data**

# In[ ]:


donors_donations.head()

# **Projects_schools data**

# In[4]:


projects_schools = projects.merge(schools, on='School ID', how='inner')
projects_schools.head()

# ## <a id='3-2'>3.2 Statistical overview of the Data</a>

# **Donation Amount**

# In[5]:


donations["Donation Amount"].describe()

# * **Donation Amount :**
#   * Minimum amount given to the project by the donor : \$ 0
#   * Maximum amount given to the project by the donor : \$ 6000
#   * Mean  amount given to the project by the donor : \$ 60

# **School Percentage Free Lunch**

# In[6]:


schools['School Percentage Free Lunch'].describe()

# * Minimum number of students qualifying for free or reduced lunch in a particular school : 0 %
# * Maximum number of students qualifying for free or reduced lunch in a particular school : 100 %

# # <a id='4'>4. Data preparation</a>

# ## <a id='4-1'> 4.1 Check for missing data</a>

# **Checking missing data in donors dataset**

# In[7]:


# checking missing data in donors data 
total = donors.isnull().sum().sort_values(ascending = False)
percent = (donors.isnull().sum()/donors.isnull().count()*100).sort_values(ascending = False)
missing_donors_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_donors_data.head()

# **Checking missing data in donations dataset**

# In[8]:


# checking missing data in donations data 
total = donations.isnull().sum().sort_values(ascending = False)
percent = (donations.isnull().sum()/donations.isnull().count()*100).sort_values(ascending = False)
missing_donations_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_donations_data.head()

# **Checking missing data in schools dataset **

# In[9]:


# checking missing data in schools dataset 
total = schools.isnull().sum().sort_values(ascending = False)
percent = (schools.isnull().sum()/schools.isnull().count()*100).sort_values(ascending = False)
missing_schools_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_schools_data.head()

# **Checking missing data in teachers dataset **

# In[10]:


# checking missing data in teachers dataset 
total = teachers.isnull().sum().sort_values(ascending = False)
percent = (teachers.isnull().sum()/teachers.isnull().count()*100).sort_values(ascending = False)
missing_teachers_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_teachers_data.head()

# **Checking missing data in resources dataset**

# In[12]:


# checking missing data in resources dataset 
total = resources.isnull().sum().sort_values(ascending = False)
percent = (resources.isnull().sum()/resources.isnull().count()*100).sort_values(ascending = False)
missing_resources_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_resources_data.head()

# #  <a id='5'>5. Data Exploration</a>

# ## <a id='5-1'>5.1 Histogram and Distribution of Donation Amount</a>

# In[ ]:


plt.figure(figsize = (12, 8))

sns.distplot(donors_donations['Donation Amount'].dropna())
plt.xlabel('Donation Amount', fontsize=12)
plt.title("Histogram of Donation Amount")
plt.show() 

plt.figure(figsize = (12, 8))
plt.scatter(range(donors_donations.shape[0]), np.sort(donors_donations['Donation Amount'].values))
plt.xlabel('Donation Amount', fontsize=12)
plt.title("Distribution of Donation Amount")
plt.show()

# ## <a id='5-2'>5.2 Top/Distribution</a>

# ## <a id='5-2-1'>5.2.1 Top Donor Cities</a>

# In[72]:


temp = donors_donations["Donor City"].value_counts().head(30)
temp.iplot(kind='bar', xTitle = 'City name', yTitle = "Count", title = 'Top Donor cities')

# * **Top 5 Donor Cities :**
#   * Chicago
#   * New York
#   * Brooklyn
#   * Los Angeles
#   * San Francisco

# ## <a id='5-2-2'>5.2.2 Top Donor States</a>

# In[ ]:


temp = donors_donations["Donor State"].value_counts().head(30)
temp.iplot(kind='bar', xTitle = 'State name', yTitle = "Count", title = 'Top Donor States')

# * **Top 5 Donor States :**
#   * California
#   * New York
#   * Texas
#   * Florida
#   * Illnois
#   

# ## <a id='5-2-3'>5.2.3 Top Donor checked out carts</a>

# In[ ]:


temp = donors_donations['Donor Cart Sequence'].value_counts().head()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values', title='Top Donor checked out carts')

# * Top Donor checked out carts are 1, 2, 3, 4 and 5

# ## <a id='5-2-4'>5.2.4 Distribution of Project subject categories</a>

# In[ ]:


temp = projects_schools['Project Subject Category Tree'].value_counts().head(30)
temp.iplot(kind='bar', xTitle = 'Project Subject Category', yTitle = "Count", title = 'Distribution of Project subject categories')

# * Top Project sub-categories are : **Literacy & Language** , **Math & Science** , **Music & Arts**.

# ## <a id='5-2-5'>5.2.5 Distribution of Project subject Sub-categories</a>

# In[ ]:


temp = projects_schools['Project Subject Subcategory Tree'].value_counts().head(10)
temp.iplot(kind='bar', xTitle = 'Project Subject Sub-Category', yTitle = "Count", title = 'Distribution of Project subject Sub-categories')

# * Top Project subject Sub-categories are : **Literacy , Mathmatics & Writing**.

# ## <a id='5-2-6'>5.2.6 Distribution of Project resource categories</a>

# In[ ]:


# temp = projects_schools['Project Resource Category'].value_counts().head(30)
# temp.iplot(kind='bar', xTitle = 'Project Resource Category Name', yTitle = "Count", title = 'Distribution of Project Resource categories')
temp = projects_schools['Project Resource Category'].value_counts().head(10)
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values', title='Distribution of Project resource categories')

# * Top Project resource categories are : **supplies, Technology, Books, Others and Computer & Tablets.**

# ## <a id='5-2.7'>5.2.7 Distribution of school Metro Type</a>

# In[ ]:


temp = schools['School Metro Type'].value_counts()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values', title='Distribution of school Metro Type')

# * **4 Categories of Metro :**
#   * Suburban - having 31.5 % schools
#   * Urban - having 31.2 % schools
#   * Rural - having 17.8 % schools
#   * Town - having 8.38 % schools
#   
#   11.1 % schools are Unknown.

# ## <a id='5-2-8'>5.2.8  Distribution of School cities</a>

# In[ ]:


cnt_srs = projects_schools['School City'].value_counts().head(20)
trace = go.Bar(
    y=cnt_srs.index[::-1],
    x=cnt_srs.values[::-1],
    orientation = 'h',
    marker=dict(
        color=cnt_srs.values[::-1],
        colorscale = 'Blues',
        reversescale = True
    ),
)

layout = dict(
    title='Distribution of School cities',
    )
data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename="CitySchools")

# * **Top  cities of the school where teacher was teaching at at the time the project was posted are :**
#   * New York
#   * Chicago
#   * Log Angeles
#   * Houston

# ## <a id='5-2-9'>5.2.9 Distribution of School County</a>

# In[ ]:


cnt_srs = projects_schools['School County'].value_counts().head(20)
trace = go.Bar(
    y=cnt_srs.index[::-1],
    x=cnt_srs.values[::-1],
    orientation = 'h',
    marker=dict(
        color=cnt_srs.values[::-1],
        colorscale = 'Blues',
        reversescale = True
    ),
)

layout = dict(
    title='Distribution of School County',
    )
data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename="CountySchool")

# * **Top county of the school that the teacher was teaching at at the time the project was posted :**
#   * Los Angeles
#   * Cook
#   * Harris
#   * Orange
# 

# ## <a id='5-2-10'>5.2.10 Distribution of Projects Grade Level Category</a>

# In[18]:


temp = projects['Project Grade Level Category'].value_counts()
fig = {
  "data": [
    {
      "values": temp.values,
      "labels": temp.index,
      "domain": {"x": [0, .48]},
      "name": "Grade Level Category",
      #"hoverinfo":"label+percent+name",
      "hole": .4,
      "type": "pie"
    },
    
    ],
  "layout": {
        "title":"Distribution of Projects Grade Level Category",
        "annotations": [
            {
                "font": {
                    "size": 20
                },
                "showarrow": False,
                "text": "Grade Level Categories",
                "x": 0.15,
                "y": 0.5
            }
            
        ]
    }
}
iplot(fig, filename='donut')

# * **Top  Projects Grade Level Category :**
#   * Grade Prek-2 : 38.9 %
#   * Grade 3-5 : 32.9 %
#   * Grade 6-8 : 16.4 %
#   * Grade 9-12 : 11.8 %

# ## <a id='5-3'>5.3 Donor is Teacher or not</a>

# In[13]:


temp = donors['Donor Is Teacher'].value_counts()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values', title='Donor is Teacher or not')

# * Approx 10 % time Donor is Teacher and 90 % time Donor is not Teacher.

# ## <a id='5.4'>5.4 Whether or not the donation included an optional donation.</a>

# In[ ]:


temp = donors_donations['Donation Included Optional Donation'].value_counts()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values', title='Whether or not the donation included an optional donation.')

# * Approx. 82 % time donation included an optional donations and 18 % time not.

# ## <a id='5-5'>5.5 Types of Projects</a>

# In[ ]:


temp = projects_schools["Project Type"].dropna().value_counts().head(30)
temp.iplot(kind='bar', xTitle = 'Project name', yTitle = "Count", title = 'Types of Projects')

# * **Total 3 Types of projects are :**
#  1. Teacher-Led - count is 1M
#  2. Professional Development - count is 10K
#  3. Student-Led - count is 7710

# ## <a id='5-6'>5.6 Projects were fully funded or not</a>

# In[ ]:


temp = projects_schools['Project Current Status'].value_counts()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values', title='Projects were fully funded or not.')

# * Only 68 % projects are fully funded, 20 % are expired and 8.11 % are archieved.

# ## <a id='5-7'>5.7 Top Keywords from project Essay</a>

# In[ ]:


from wordcloud import WordCloud, STOPWORDS
names = projects_schools["Project Essay"][~pd.isnull(projects_schools["Project Essay"])].sample(10000)
#print(names)
wordcloud = WordCloud(max_font_size=50, width=600, height=300).generate(' '.join(names))
plt.figure(figsize=(15,8))
plt.imshow(wordcloud)
plt.title("Wordcloud for Top Keywords from project Essay", fontsize=35)
plt.axis("off")
plt.show() 

# ## <a id='5-8'>5.8 Top Keywords from project Title</a>

# In[ ]:


names = projects_schools["Project Title"][~pd.isnull(projects_schools["Project Title"])].sample(1000)
#print(names)
wordcloud = WordCloud(max_font_size=50, width=600, height=300).generate(' '.join(names))
plt.figure(figsize=(15,8))
plt.imshow(wordcloud)
plt.title("Wordcloud for Top Keywords from project Titles", fontsize=35)
plt.axis("off")
plt.show() 

# ## <a id='5-9'>5.9 Donations Given by different States</a>

# In[ ]:


state_wise = donors_donations.groupby('Donor State', as_index=False).agg({'Donation ID': 'count','Donation Amount':'sum'})   
state_wise.columns = ["State","Donation_num", "Donation_sum"]
state_wise["Donation_avg"]=state_wise["Donation_sum"]/state_wise["Donation_num"]
del state_wise['Donation_num']

# In[ ]:


for col in state_wise.columns:
    state_wise[col] = state_wise[col].astype(str)
state_wise['text'] = state_wise['State'] + '<br>' +\
    'Average amount per donation: $' + state_wise['Donation_avg']+ '<br>' +\
    'Total donation amount:  $' + state_wise['Donation_sum']
state_codes = {
    'District of Columbia' : 'DC','Mississippi': 'MS', 'Oklahoma': 'OK', 
    'Delaware': 'DE', 'Minnesota': 'MN', 'Illinois': 'IL', 'Arkansas': 'AR', 
    'New Mexico': 'NM', 'Indiana': 'IN', 'Maryland': 'MD', 'Louisiana': 'LA', 
    'Idaho': 'ID', 'Wyoming': 'WY', 'Tennessee': 'TN', 'Arizona': 'AZ', 
    'Iowa': 'IA', 'Michigan': 'MI', 'Kansas': 'KS', 'Utah': 'UT', 
    'Virginia': 'VA', 'Oregon': 'OR', 'Connecticut': 'CT', 'Montana': 'MT', 
    'California': 'CA', 'Massachusetts': 'MA', 'West Virginia': 'WV', 
    'South Carolina': 'SC', 'New Hampshire': 'NH', 'Wisconsin': 'WI',
    'Vermont': 'VT', 'Georgia': 'GA', 'North Dakota': 'ND', 
    'Pennsylvania': 'PA', 'Florida': 'FL', 'Alaska': 'AK', 'Kentucky': 'KY', 
    'Hawaii': 'HI', 'Nebraska': 'NE', 'Missouri': 'MO', 'Ohio': 'OH', 
    'Alabama': 'AL', 'Rhode Island': 'RI', 'South Dakota': 'SD', 
    'Colorado': 'CO', 'New Jersey': 'NJ', 'Washington': 'WA', 
    'North Carolina': 'NC', 'New York': 'NY', 'Texas': 'TX', 
    'Nevada': 'NV', 'Maine': 'ME', 'other': ''}

state_wise['code'] = state_wise['State'].map(state_codes)  

# In[ ]:


# https://plot.ly/python/choropleth-maps/
data = [ dict(
        type='choropleth',
        autocolorscale = True,
        locations = state_wise['code'], # The variable identifying state
        z = state_wise['Donation_sum'].astype(float), # The variable used to adjust map colors
        locationmode = 'USA-states', 
        text = state_wise['text'], # Text to show when mouse hovers on each state
        colorbar = dict(  
            title = "Donation in USD")  # Colorbar to show beside the map
        ) ]

layout = dict(
        title = 'Donation by State<br>(Hover for breakdown)',
        geo = dict(
            scope='usa',
            projection=dict( type='albers usa' ),
            showlakes = True,
            lakecolor = 'rgb(255, 255, 255)'),
             )
fig = dict(data=data, layout=layout)
iplot(fig)


# * **Highest donation given to the project in states :**
#   * California - Total donation amount Approx. 46 M
#   * New York  -  Total donation amount Approx.  24 M

# ## <a id='5-10'>5.10 Number of schools in different states</a>

# In[ ]:


school_count = schools['School State'].value_counts().reset_index()
school_count.columns = ['state', 'schools']
for col in school_count.columns:
    school_count[col] = school_count[col].astype(str)
school_count['text'] = school_count['state'] + '<br>' + '# of schools: ' + school_count['schools']

# In[ ]:


# https://plot.ly/python/choropleth-maps/
data = [ dict(
        type='choropleth',
        autocolorscale = True,
        locations = state_wise['code'], # The variable identifying state
        z = school_count['schools'].astype(float), # The variable used to adjust map colors
        locationmode = 'USA-states', 
        text = school_count['text'], # Text to show when mouse hovers on each state
        colorbar = dict(  
            title = "# of Schools")  # Colorbar to show beside the map
        ) ]

layout = dict(
        title = 'Number of schools in different states<br>(Hover for breakdown)',
        geo = dict(
            scope='usa',
            projection=dict( type='albers usa' ),
            showlakes = True,
            lakecolor = 'rgb(255, 255, 255)'),
             )
fig = dict(data=data, layout=layout)
iplot(fig)

# * **Top School states(The state of the school that the teacher was teaching at at the time the project was posted) :**
#   * California : No. of schools - 8360
#   * Texas : No. of schools - 6406

# ## <a id='5-11'>5.11 School Percentage Free Lunch for School Metro Type </a>

# In[19]:


schools.groupby('School Metro Type')['School Percentage Free Lunch'].describe()

# * **School Percentage Free Lunch(percentage of students qualifying for free or reduced lunch, obtained from NCES data) for School Metro Type :**
#   * rural : \# times - 12929, Min. - 0 % , Max. - 100 %
#   * suburban : \# times - 22965, Min. - 0 % , Max. - 100 %
#   * town : \# times - 6105, Min. - 0 % , Max. - 99 %
#   * urban : \# times - 22607, Min. - 0 % , Max. - 100 %

# ## <a id='5-12'>5.12 Number of Donations given by donor</a>

# In[32]:


donations_per_donor = donations.groupby('Donor ID')['Donor Cart Sequence'].max()
donations_per_donor = (donations_per_donor == 1).mean() *100
print("Only one time donation is given by : "+ str(donations_per_donor) +" % donors")

# ## <a id='5-13'>5.13 Gender wise  Analysis</a>

# ## <a id='5-13-1'>5.13.1 Distribution of Teachers Prefix</a>

# In[35]:


temp = teachers['Teacher Prefix'].value_counts()
fig = {
  "data": [
    {
      "values": temp.values,
      "labels": temp.index,
      "domain": {"x": [0, .48]},
      "name": "Teachers Prefix",
      #"hoverinfo":"label+percent+name",
      "hole": .4,
      "type": "pie"
    },
    
    ],
  "layout": {
        "title":"Distribution of Teachers Prefix",
        "annotations": [
            {
                "font": {
                    "size": 20
                },
                "showarrow": False,
                "text": "Teachers Prefix",
                "x": 0.17,
                "y": 0.5
            }
            
        ]
    }
}
iplot(fig, filename='donut')

# * **Keywords chosen by teachers during account creation :**
#   * Mrs. - 50.2 % time
#   * Ms. 36.2 % time
#   * Mr. - 11.8 % time
#   * Teacher(gender neutral option) : 1.84 % time

# ## <a id='5-13-2'>5.13.2 Males V.S. Female</a>

# Here we are doing some mapping  :
# 
#   * Mrs, Ms --> Female
#   * Mr. --> Male
#   * Teacher, Dr., Mx --> Unknown

# In[37]:


# Creating the gender column
gender_mapping = {"Mrs.": "Female", "Ms.":"Female", "Mr.":"Male", "Teacher":"Unknown", "Dr.":"Unknown", np.nan:"Unknown", "Mx.":"Unknown" }
teachers["gender"] = teachers['Teacher Prefix'].map(gender_mapping)

# In[40]:


temp = teachers['gender'].value_counts()
fig = {
  "data": [
    {
      "values": temp.values,
      "labels": temp.index,
      "domain": {"x": [0, .48]},
      "name": "Gender",
      #"hoverinfo":"label+percent+name",
      "hole": .4,
      "type": "pie"
    },
    
    ],
  "layout": {
        "title":"Males V.S. Females",
        "annotations": [
            {
                "font": {
                    "size": 20
                },
                "showarrow": False,
                "text": "Gender",
                "x": 0.20,
                "y": 0.5
            }
            
        ]
    }
}
iplot(fig, filename='donut')

# * **Teachers posted projects on website :**
#   * Females : 86.3 %
#   * Male : 11.8 %
#   * Unknown : 1.87 %

# ## 5.14 Resource data

# ## 5.14.1 Top requested items

# In[44]:


temp = resources["Resource Item Name"].dropna().value_counts().head(20)
temp.iplot(kind='bar', xTitle = 'Resource Item Name', yTitle = "Count", title = 'Top requested items')

# ## 5.14.2 Top Resource Vendor Name

# In[46]:


temp = resources["Resource Vendor Name"].dropna().value_counts().head(20)
temp.iplot(kind='bar', xTitle = 'Resource Vendor Name', yTitle = "Count", title = 'Top Resource Vendor Name')

# *  **Top 3 Resource Vendor Name :**
#   * Amazon Business
#   * Lakeshore Learning Materials
#   * AKJ Education

# ## 5.14.3 Distribution of resources price

# In[49]:


resources['total_price'] = resources['Resource Quantity'] * resources['Resource Unit Price']
plt.figure(figsize = (12, 8))
plt.scatter(range(resources.shape[0]), np.sort(resources['total_price'].values))
plt.xlabel('Price', fontsize=12)
plt.title("Distribution of resources price")
plt.show()

# ## 5.15 Time Series Analysis

# In[53]:


teachers['Teacher First Project Posted Date'] = pd.to_datetime(teachers['Teacher First Project Posted Date'])
teachers['weekdays'] = teachers['Teacher First Project Posted Date'].dt.dayofweek
teachers['month'] = teachers['Teacher First Project Posted Date'].dt.month 
teachers['year'] = teachers['Teacher First Project Posted Date'].dt.year

dmap = {0:'Monday',1:'Tueday',2:'Wednesday',3:'Thursday',4:'Friday',5:'Saturday',6:'Sunday'}
teachers['weekdays'] = teachers['weekdays'].map(dmap)

month_dict = {1 :"Jan",2 :"Feb",3 :"Mar",4 :"Apr",5 : "May",6 : "Jun",7 : "Jul",8 :"Aug",9 :"Sep",10 :"Oct",11 :"Nov",12 :"Dec"}
teachers['month'] = teachers['month'].map(month_dict)
teachers.head()

# ## 5.15.1 week day wise first project posted by teachers

# In[58]:


temp = teachers['weekdays'].value_counts()
temp = temp.reset_index()
temp.columns = ['weekdays', 'count']
#temp.head()
temp[['weekdays','count']].set_index('weekdays').iplot(kind = 'bar', xTitle = 'Day of Week', yTitle = "# of projects posted", title ="week day wise first project posted by teachers")

# * **Top week days when teachers posted thier first project :**
#   * Sunday - Approx. 73 K 
#   * Saturday - Approx. 66 K
#   * Monday - Approx. 61 K

# ## 5.15.2 Month wise first project posted by teachers

# In[59]:


temp = teachers['month'].value_counts()
temp = temp.reset_index()
temp.columns = ['month', 'count']
#temp.head()
temp[['month','count']].set_index('month').iplot(kind = 'bar', xTitle = 'Month', yTitle = "# of projects posted", title ="Month wise first project posted by teachers")

# * **Top Months when teachers posted thier first project :**
#   * September -Approx. 59 K
#   * August - Approx 52 K
#   * October - Approx. 48 K

# ## 5.15.3 Trend  of teachers who posted their project first time(2002 to 2017)

# In[74]:


temp = teachers.groupby('year').agg({'Teacher ID' : 'count'}).reset_index()
year2002_2017 = temp[~temp.year.isin([2018])] 
year2002_2017 = year2002_2017.sort_values('year').set_index("year")
#temp.head()
# temp = teachers['year'].value_counts()
# temp = temp.reset_index()
# temp.columns = ['year', 'count']
year2002_2017.iplot(kind = 'scatter', xTitle='Year 2002 to Year 2017',  yTitle = "# of teachers first project posted", title ="Trend of teachers who posted their project first time(2002 to 2017)")

# * Growth of number of teachers posted thier first project **increase** from **2002(# projects : 13) to 2016 (# of projects : Approx. 80 K)** then **decreases** in **2017 (# projects posed : Approx. 76 K)**

# ## 5.15.4 Trend of Teachers who posted their project first time in 2018

# In[69]:


#temp = teachers.groupby('year').agg({'Teacher ID' : 'count'}).reset_index()
year2018 = temp[temp.year.isin([2018])] 
year2018 = year2018.sort_values('year').set_index("year")
year2018.iplot(kind = 'scatter', xTitle='Year 2018',  yTitle = "# of teachers first project posted", title ="Trend of teachers who posted their project first time(2002 to 2017)")

# # <a id='6'>6. Brief summary/Conclusions</a>

# **This is only brief summary. If you want more detail please go through my notebook.**

# * **Donation Amount :**
#   * Minimum amount given to the project by the donor : \$ 0
#   * Maximum amount given to the project by the donor : \$ 6000
#   * Mean  amount given to the project by the donor : \$ 60
# * Minimum number of students qualifying for free or reduced lunch in a particular school : 0 %
# * Maximum number of students qualifying for free or reduced lunch in a particular school : 100 %  
# * **Top 5 Donor Cities :**
#   * Chicago
#   * New York
#   * Brooklyn
#   * Los Angeles
#   * San Francisco
# * **Top 5 Donor States :**
#   * California
#   * New York
#   * Texas
#   * Florida
#   * Illnois
# * Approx 10 % time Donor is Teacher and 90 % time Donor is not Teacher.
# * Approx. 82 % time donation included an optional donations and 18 % time not.
# * Top Donor checked out carts are 1, 2, 3, 4 and 5
# * Top Project sub-categories are : **Literacy & Language** , **Math & Science** , **Music & Arts**.
# * Top Project subject Sub-categories are : **Literacy , Mathmatics & Writing**.
# * Top Project resource categories are : **supplies, Technology, Books, Others and Computer & Tablets.**
# * **Top  Projects Grade Level Category :**
#   * Grade Prek-2 : 38.9 %
#   * Grade 3-5 : 32.9 %
#   * Grade 6-8 : 16.4 %
#   * Grade 9-12 : 11.8 %
#  
# * **Total 3 Types of projects are :**
#  1. Teacher-Led - count is 1M
#  2. Professional Development - count is 10K
#  3. Student-Led - count is 7710
# * **4 Categories of Metro :**
#    * Suburban - having 31.5 % schools
#    * Urban - having 31.2 % schools
#    * Rural - having 17.8 % schools
#    * Town - having 8.38 % schools
#   
#   11.1 % schools are Unknown.
# * **Top  cities of the school where teacher was teaching at at the time the project was posted are :**
#   * New York
#   * Chicago
#   * Log Angeles
#   * Houston 
# * **Top county of the school that the teacher was teaching at at the time the project was posted :**
#   * Los Angeles
#   * Cook
#   * Harris
#   * Orange
# *  **Top 3 Resource Vendor Name :**
#   * Amazon Business
#   * Lakeshore Learning Materials
#   * AKJ Education
# * **Teachers posted projects on website :**
#   * Females : 86.3 %
#   * Male : 11.8 %
#   * Unknown : 1.87 %
# * **Highest donation given to the project in states :**
#   * California - Total donation amount Approx. 46 M
#   * New York  -  Total donation amount Approx.  24 M
#  
# * Only one time donation is given by : 69.74380959559794 % donors
# * Only 68 % projects are fully funded, 20 % are expired and 8.11 % are archieved.
# * **Top week days when teachers posted thier first project :**
#   * Sunday - Approx. 73 K 
#   * Saturday - Approx. 66 K
#   * Monday - Approx. 61 K
# * **Top Months when teachers posted thier first project :**
#   * September -Approx. 59 K
#   * August - Approx 52 K
#   * October - Approx. 48 K
# * Growth of number of teachers posted thier first project **increase** from **2002 ( # projects : 13) to 2016 (  # of projects : Approx. 80 K)** then **decreases** in **2017 ( # projects posed : Approx. 76 K)**  

# # More To Come. Styed Tuned !!
