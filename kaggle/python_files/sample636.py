#!/usr/bin/env python
# coding: utf-8

# ![](http://www.lauracandler.com/filecabinet/misc/dcimages/DonorsChooseWebinar.png)

# ![](https://www.google.org/impactchallenge/disabilities/img/donors/donors-logo.png)

# ## Table of Content
# - <a href='#1'>1. Introduction</a>  
# - <a href='#2'>2. Retrieving the Data</a>  
#     - <a href='#2-1'>2.1 Load libraries</a>
#     - <a href='#2-2'>2.2 Read the Data</a>
# - <a href='#3'>3. Glimpse of Data</a>
#     - <a href='#3-1'>3.1 Tabular View of Data </a>
#     - <a href='#3-2'>3.2 Statistical overview of the Data </a>
# - <a href='#4'>4. Data Exploration </a>
#     - <a href='#4-1'>4.1 Top 5 States With Maximum Number of Donor Cities </a>
#     - <a href='#4-2'>4.2 Locality of schools with their counts </a>
#     - <a href='#4-3'>4.3 Top Donor State Which Donated Highest Money </a>
#     - <a href='#4-4'>4.4  Nber of Donations Made by a Particular State </a>
#     - <a href='#4-5'>4.5 Average amount funded by Top 5 State in terms of number of projects </a>
#     - <a href='#4-6'>4.6 Percentage of Donor as Teachers </a>
#     - <a href='#4-7'>4.7 Top Donor Checked Out Carts </a>
#     - <a href='#4-8'>4.8 Average Percentage of free lunch based on Metro Type </a>
#     - <a href='#4-9'>4.9 Unique Project Types </a>
#     - <a href='#4-10'>4.10 Top 5 projects with their count </a>
#     - <a href='#4-11'>4.11 Project Count According to School Metro Type </a>
#     - <a href='#4-12'>4.12 Current  Status of Projects </a>
#     - <a href='#4-13'> 4.13 Time Series Analysis </a>
#         - <a href='#4-13-1'> 4.13.1 Average Number of days needed for each project resosurce category to get fund approval </a>
#         - <a href='#4-13-2'>4.13.2  Projects Posted VS  Projects Funded at a perticular year </a>
#         - <a href='#4-13-3'>4.13.3 Number of Project Posted Month Wise. </a>
# - <a href='#5'> 5. Brief EDA Summmary </a>
# - <a href='#6'> 6.Building a Recommender </a>
#     - <a href='#6-1'>6.1 Content based Filtering</a>
#     - <a href='#6-2'>6.2 Collaborative Filtering(CF) Method</a>
# 
# 

# # <a id='1'>1. Introduction</a>
# ## What is DonorsChoose?
# 
# A crowdfunding platform for K-12 teachers serving in US schools, a nonprofit organization that allows individuals to donate directly to public school classroom projects. and is founded by **Charles Best**(former public school teacher).
# 
# To date, 3 million people and partners have funded 1.1 million DonorsChoose.org projects. But teachers still spend more than a billion dollars of their own money on classroom materials. To get students what they need to learn, the team at DonorsChoose.org needs to be able to connect donors with the projects that most inspire them.
# 
# In this second Kaggle Data Science for Good challenge, DonorsChoose.org, in partnership with Google.org, is inviting the community to help them pair up donors to the classroom requests that will most motivate them to make an additional gift. To support this challenge, DonorsChoose.org has supplied anonymized data on donor giving from the past five years. The winning methods will be implemented in DonorsChoose.org email marketing campaigns.
# 
# So, I start with my analysis of this challenge by starting from basic overview and going deep down showing the insights.

# # <a id='2'>2. Retrieving Data</a>

#  # <a id='2-1'>2. Loading Libraries</a>
# ## Importing Libraries

# In[ ]:


import pandas as pd
import numpy as np # package for linear algebra
from keras.preprocessing import text, sequence

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
import calendar
from mpl_toolkits.basemap import Basemap
from numpy import array
from matplotlib import cm


# import cufflinks and offline mode
import cufflinks as cf
cf.go_offline()

import os
print(os.listdir("../input"))
from sklearn import preprocessing
# Supress unnecessary warnings so that presentation looks clean
import warnings
warnings.filterwarnings("ignore")

# Print all rows and columns
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# # <a id='2-2'>2.2 Reading Data</a>
# # Loading Data

# In[ ]:


path='../input/io/'
dtypes_donations = {'Project ID' : 'object','Donation ID':'object','device': 'object','Donor ID' : 'object','Donation Included Optional Donation': 'object','Donation Amount': 'float64','Donor Cart Sequence' : 'int64'}
donations = pd.read_csv(path+'Donations.csv',dtype=dtypes_donations)

dtypes_donors = {'Donor ID' : 'object','Donor City': 'object','Donor State': 'object','Donor Is Teacher' : 'object','Donor Zip':'object'}
donors = pd.read_csv(path+'Donors.csv', low_memory=False,dtype=dtypes_donors)

dtypes_schools = {'School ID':'object','School Name':'object','School Metro Type':'object','School Percentage Free Lunch':'float64','School State':'object','School Zip':'int64','School City':'object','School County':'object','School District':'object'}
schools = pd.read_csv(path+'Schools.csv', dtype=dtypes_schools)#error_bad_lines=False

dtypes_teachers = {'Teacher ID':'object','Teacher Prefix':'object','Teacher First Project Posted Date':'object'}
teachers = pd.read_csv(path+'Teachers.csv', dtype=dtypes_teachers)#error_bad_lines=False,
                   
dtypes_projects = {'Project ID' : 'object','School ID' : 'object','Teacher ID': 'object','Teacher Project Posted Sequence':'int64','Project Type': 'object','Project Title':'object','Project Essay':'object','Project Subject Category Tree':'object','Project Subject Subcategory Tree':'object','Project Grade Level Category':'object','Project Resource Category':'object','Project Cost':'object','Project Posted Date':'object','Project Current Status':'object','Project Fully Funded Date':'object'}
projects = pd.read_csv(path+'Projects.csv',parse_dates=['Project Posted Date','Project Fully Funded Date'], dtype=dtypes_projects)#error_bad_lines=False, warn_bad_lines=False,

dtypes_resources = {'Project ID' : 'object','Resource Item Name' : 'object','Resource Quantity': 'float64','Resource Unit Price' : 'float64','Resource Vendor Name': 'object'}
resources = pd.read_csv(path+'Resources.csv', dtype=dtypes_resources)#

# # <a id='3'>3. Glimpse of Data</a>

# # <a id='3-1'>3.1 Tabular View of Data</a>
# # Checking out the Data

# #### View of Donations Data

# In[ ]:


donations.head()[:3]

# #### View of Donors data

# In[ ]:


donors.head()[:3]

# #### View of Schools  data

# In[ ]:


schools.head()[:3]

# #### View of Teachers data

# In[ ]:


teachers.head()[:3]

# #### View of Projects data

# In[ ]:


projects.head()[:3]

# #### View of Resources data

# In[ ]:


resources.head()[:3]

# ## Create new table by combining Donors with Donation as they are mainly supporting classroom requests

# In[ ]:


donors_donations = donations.merge(donors, on='Donor ID', how='inner')
donors_donations.head()[:4]

# # <a id='3-2'>3. 2  Statistical View of Data</a>
# # Statistical View of Data

# In[ ]:


donations["Donation Amount"].describe().apply(lambda x: format(x, 'f'))

# 
# * Max Amount Donated : 60000
# *  Min Amount Donated : 0
# *  Mean Amount : 60.65126
# *  Meadian Amount : 25 
# *  Standard Deviation : 166.88
# * As You can See Mean is Greater than Median ..Donation Amount is Rightly Skewed.
# 

# In[ ]:


schools['School Percentage Free Lunch'].describe()

# * Max School Percentage Free Lunch : 100
# * Min School Percentage Free Lunch : 0
# * Mean School Percentage Free Lunch : 58.55
# * Median School Percentage Free Lunch : 61
# * Mean is less tha Median but not too less ..we can say Mean == Meadian . Data is Symmetric.

# # <a id='4'>4.  Data Exploration </a>

# ## Visualizing some insights from data

# # <a id='4-1'>4.1 Top 5 States With Maximum Number of Donor Cities </a>

# In[ ]:


df = donors.groupby("Donor State")['Donor City'].nunique().to_frame().reset_index()
X = df['Donor State'].tolist()
Y = df['Donor City'].apply(float).tolist()
Z = [x for _,x in sorted(zip(Y,X))]

data = pd.DataFrame({'Donor State' : Z[-5:][::-1], 'Count of Donor Cities' : sorted(Y)[-5:][::-1] })
sns.barplot(x="Donor State",y="Count of Donor Cities",data=data)

# # <a id='4-2'>4.2 Locality of schools with their counts </a>

# In[ ]:


plt.rcParams["figure.figsize"] = [12,6]
sns.countplot(x='School Metro Type',data = schools)

# # <a id='4-3'>4.3 Top Donor State Which Donated Highest Money</a>

# In[ ]:


donors_state_amount=donors_donations.groupby('Donor State')['Donation Amount'].sum().reset_index()
donors_state_amount['Donation Amount']=donors_state_amount['Donation Amount'].apply(lambda x: format(x, 'f'))

df = donors_state_amount[['Donor State','Donation Amount']]
X = df['Donor State'].tolist()
Y = df['Donation Amount'].apply(float).tolist()
Z = [x for _,x in sorted(zip(Y,X))]

data = pd.DataFrame({'Donor State' : Z[-5:][::-1], 'Total Donation Amount' : sorted(Y)[-5:][::-1] })
sns.barplot(x="Donor State",y="Total Donation Amount",data=data)

# ### Top 5 Donor States
# * **California (46+ Million)**
# * **New York (24+ Million)**
# * **Texas (17+ Million)**
# * **Illinois (14+ Million)**
# * **Florida (13+ Million)**

# # <a id='4-4'>4.4 Total Number of Donations Made by a Particular State</a>

# In[ ]:


temp = donors_donations["Donor State"].value_counts().head(25)
temp.iplot(kind='bar', xTitle = 'State name', yTitle = "Count", title = 'Top Donor States')

# Top States who funded Project
# California(723 k) funded most Projects followed by New York(365 k) then Texas(287 k)

# # <a id='4-5'>4.5 Average amount funded by top 5 states in terms of number of projects</a>

# In[ ]:


state_count = temp.to_frame(name="number_of_projects").reset_index()
state_count = state_count.rename(columns= {'index': 'Donor State'})
# merging states with projects and amount funded
donor_state_amount_project = state_count.merge(donors_state_amount, on='Donor State', how='inner')

val = [x/y for x, y in zip(donor_state_amount_project['Donation Amount'].apply(float).tolist(),donor_state_amount_project['number_of_projects'].tolist())]
state_average_funding = pd.DataFrame({'Donor State':donor_state_amount_project['Donor State'][-5:][::-1],'Average Funding':val[-5:][::-1]})
sns.barplot(x="Donor State",y="Average Funding",data=state_average_funding)

# # <a id='4-6'>4.6 Percentage of donors as teachers</a>

# In[ ]:


per_teacher_as_donor = donors['Donor Is Teacher'].value_counts().to_frame().reset_index()
per_teacher_as_donor = per_teacher_as_donor.rename(columns= {'index': 'Types'})
labels = ['Donor is a Teacher','Donor is not a Teacher']
values = per_teacher_as_donor['Donor Is Teacher'].tolist()
colors = ['#96D38C', '#E1396C']
trace = go.Pie(labels=labels, values=values,
               hoverinfo='label+percent', textinfo='value',
               textfont=dict(size=20),
               marker=dict(colors=colors,
                           line=dict(color='#000000', width=2)))
py.iplot([trace], filename='styled_pie_chart')

# # <a id='4-7'>4.7 Top Donor Checked Out carts</a>

# In[ ]:


temp = donors_donations['Donor Cart Sequence'].value_counts().head()
df = pd.DataFrame({'labels': temp.index,'values': temp.values})
labels = df['labels'].tolist()
values = df['values'].tolist()
colors = ['#96D38C', '#E1396C','#C0C0C0','#FF0000','#F08080']
trace = go.Pie(labels=labels, values=values,
               hoverinfo='label+percent', textinfo='value',
               textfont=dict(size=20),
               marker=dict(colors=colors,
                           line=dict(color='#000000', width=2)))

py.iplot([trace], filename='styled_pie_charimport calendart')

# # <a id='4-8'>4.8 Average Percentage of free lunch based on Metro Type</a>

# In[ ]:


schools.groupby('School Metro Type')['School Percentage Free Lunch'].describe()

# ### Inference from above table
# * Rural Schools Average of free Lunch 55.87 % with Deviation of 21%.
# * Suburban Schools Average of free Lunch 49.33% with Deviation of 27%.
# * Town Schools Average of free Lunch 58.33% with Deviation of 19.65%.
# * Urban Schools Average of free Lunch 68.33% with Deviation of 24%.
# * Unknown Schools Average of free Lunch 62.34% with Deviation of 22%.

# ## Merging projects table and resources table on "Project ID" and analyzing insights from them.

# In[ ]:


projects_resources = projects.merge(resources, on='Project ID', how='inner')
projects_resources.head()[:4]

# #Top Resource Vendor

# In[ ]:


resource_vendor_name=projects_resources['Resource Vendor Name'].value_counts()
resource_vendor_name.iplot(kind='bar', xTitle = 'Vendor Name', yTitle = "Count", title = 'Resource Vendor',color='green')

# **Clearly Top Resource Vendor Provider is Amazon Buisness**

# # <a id='4-9'>4.9 Unique Project Types</a>

# ### Checking for null values in project title and filling up with dummy value

# In[ ]:


projects_resources['Project Title'].fillna('Blank',inplace=True)

# # <a id='4-10'>4.10 Top 5 projects with their count</a>

# In[ ]:


project_title = projects_resources['Project Title'].value_counts().to_frame().reset_index()[:5]
project_title = project_title.rename(columns= {'index': 'Project Title','Project Title':'Count'})
sns.barplot(x="Count",y="Project Title",data=project_title).set_title('Unique project title')

# ## Merging schools and projects to derive insights from them

# In[ ]:


school_project = schools.merge(projects, on='School ID', how='inner')
school_project[:4]

# # <a id='4-11'>4.11 Project Count According to School Metro Type</a>

# ### Visualize Projects count according to School Metro Type

# In[ ]:


school_project_count = school_project.groupby('School Metro Type')['Project ID'].count().reset_index()
school_project_count = pd.DataFrame({'School Metro Type':school_project_count['School Metro Type'],'Project Count':school_project_count['Project ID']})
sns.barplot(x="School Metro Type",y="Project Count",data=school_project_count)

# #### Inference:- A lot of projects are allocated to urban areas. 

# # <a id='4-12'>4.12 Current  Status of Projects</a>

# In[ ]:


temp = school_project['Project Current Status'].value_counts().head()
df = pd.DataFrame({'labels': temp.index,'values': temp.values})
labels = df['labels'].tolist()
values = df['values'].tolist()
colors = ['#96D38C', '#E1396C','#C0C0C0','#FF0000']
trace = go.Pie(labels=labels, values=values,
               hoverinfo='label+percent', textinfo='value',
               textfont=dict(size=20),
               marker=dict(colors=colors,
                           line=dict(color='#000000', width=2)))

py.iplot([trace], filename='styled_pie_chart')

# In[ ]:


project_open_close=school_project[['Project Resource Category','Project Posted Date','Project Fully Funded Date']]
project_open_close['Project Posted Date'] = pd.to_datetime(project_open_close['Project Posted Date'])
project_open_close['Project Fully Funded Date'] = pd.to_datetime(project_open_close['Project Fully Funded Date'])

time_gap = []
for i in range(school_project['School ID'].count()):
    if school_project['Project Current Status'][i] =='Fully Funded':
        time_gap.append(abs(project_open_close['Project Fully Funded Date'][i]-project_open_close['Project Posted Date'][i]).days)
    else:
        time_gap.append(-1)

project_open_close['Time Duration(days)'] = time_gap
project_open_close.head()

# # <a id='4-13'>4.13 Time Series Analysis </a>

# # <a id='4-13-1'>4.13.1 Average Number of days needed for each project resosurce category to get fund approval </a>

# In[ ]:


project_open_close_resource=project_open_close.groupby('Project Resource Category')['Time Duration(days)'].mean().reset_index()
df = project_open_close_resource[['Project Resource Category','Time Duration(days)']]
X = df['Project Resource Category'].tolist()
Y = df['Time Duration(days)'].apply(int).tolist()
Z = [x for _,x in sorted(zip(Y,X))]

data = pd.DataFrame({'Project Resource Category' : Z[0:5], 'Total Time Duration(days)' : sorted(Y)[0:5] })
sns.barplot(x="Total Time Duration(days)",y="Project Resource Category",data=data)

# In[ ]:


project_open_close.head()

# #### Inference:- So when the project with resource category as "Food Clothing & Hygiene" gets funded much ealy compared to other project categories.

# In[ ]:


school_project["Project Posted Date"] = pd.to_datetime(school_project["Project Posted Date"])
school_project['Project Posted year']=school_project['Project Posted Date'].dt.year
# school_project['Project Posted year']
# school_project.head()

# In[ ]:


school_project["Project Fully Funded Date"] = pd.to_datetime(school_project["Project Fully Funded Date"])
school_project['Project Funded year']=school_project['Project Fully Funded Date'].dt.year
# school_project.head()

# In[ ]:


temp=school_project['Project Posted year'].value_counts()
temp1=school_project['Project Funded year'].value_counts()

# # <a id='4-13-2'>4.13.2  Projects Posted VS  Projects Funded at a perticular year </a>

# In[ ]:


temp.iplot(kind='bar', xTitle = 'Year', yTitle = "Count", title = 'Project Posted In a Year')

# In[ ]:


temp1.iplot(kind='bar', xTitle = 'Year', yTitle = "Count", title = 'Project Funded In a Year',color='blue')

# # <a id='4-13-3'>4.13.3  Projects Posted Month Wise </a>

# In[ ]:


school_project['Project Posted Month']=school_project['Project Posted Date'].dt.month
school_project['Project Posted Month'] = school_project['Project Posted Month'].apply(lambda x: calendar.month_abbr[x])
month_count=school_project['Project Posted Month'].value_counts()
month_count.iplot(kind='bar', xTitle = 'Month', yTitle = "Count", title = 'Project Posted Month Wise')

# * Highest Number of Projects are Posted in the Month of September followed by August.
# * Lowest Number of Projects are posted in Month of June.

# # Project Funded Month Wise

# In[ ]:


# school_project["Project Fully Funded Date"] = pd.to_datetime(school_project["Project Fully Funded Date"])
# school_project['Project Funded Month']=school_project['Project Funded Date'].dt.month
# school_project['Project Funded  Month'] = school_project['Project Posted Month'].apply(lambda x: calendar.month_abbr[x])

# # Will Be Adding More Time Analysis Soon ...Please Upvote..:)

# # <a id='5'>5 Brief EDA Summary Till Now </a>

# ## Brief Summary
# 
# ### Top 5 Donor Cities :
# * Chicago
# * New York
# * Brooklyn
# * Los Angeles
# * San Francisco
# 
# ### Top 5 Donor States :
# * California (46+ Million)
# * New York (24+ Million)
# * Texas (17+ Million)
# * Illinois (14+ Million)
# * Florida (13+ Million)
# 
# 
# * Approx 28 % time Donor is Teacher and 73 % Donor is not Teacher.
# * Urban school types are given 68% mean free lunch but rural areas are given 55% time free lunch. But rural areas must get more free lunch.
# * A lot of projects with title "Classroom Library" exist in the data.
# * A lot of projects are allocated to urban areas. 
# * So when the project with resource category as "Food Clothing & Hygiene" gets funded much early compared to other project categories.
# * Maximum Number of Project Posted in Year 2017 301.24 K and Funded project 206.978 k
# * Highest Number of Projects are Posted in Month of September 165.886 k
# * Lowest Number of Projects are posted in June. 55.23 k

# # <a id='6'>6 Building a Recommender</a>

# ### What is a Recommender System?

# It is to estimate a **utility** function that automatically **predicts** how a user will like the item.
# It is based on following factors:-
# * Past behavior
# * Relations to other users
# * Item Similarity
# * Context
# * .....

# ### Types and Traditional techniques for creating recommenders

# 1. **Content based**: Recommend based on item features.
# 
# 2. **Collaborative filtering**: Recommend items based only on the users past behavior
#     * **User based**: Find similar users to me and recommend what they liked.
#     * **Item based**: Find similar items to those that I have previously liked.

# ### Novel methods

# 1. Personalized **learning to rank**: Treating recommendations as ranking problem like used in Google, Quora, Facebook, Spotify, Amazon, Netflix, Pandora, etc.
# 2. **Demographics**: Recommend based on user features like region, age, gender, race, ethnicity, etc.
# 3. **Social Recommendations**: Trust based recommendations like if top renowned people tend to follow particular trend then most of the people follow that trend.
# 4. **Hybrid**: Combine any one of the above techniques.

# ## Let's go to some practical stuff

# ### Need to recommend classroom requests to the donors

# In[ ]:


donations = donations.merge(donors, on="Donor ID", how="left")
df = donations.merge(projects,on="Project ID", how="left")

# ### Checking how much amount a particular donor has donated

# In[ ]:


donor_amount_df = df.groupby(['Donor ID', 'Project ID'])['Donation Amount'].sum().reset_index()
donor_amount_df.head()

# # <a id='6-1'>6.1 Content based filtering-------- Deep Learning based solution </a>

# In[ ]:


# data preprocessing
features = ['Project Subject Category Tree','Project Title','Project Essay']
for col in features:
    projects[col] = projects[col].astype(str).fillna('fillna')
    projects[col] = projects[col].str.lower()

# tokenizing text
final_projects = projects[features]    
tok=text.Tokenizer(num_words=1000,lower=True)
tok.fit_on_texts(list(final_projects))
final_projects=tok.texts_to_sequences(final_projects)
final_projects_train=sequence.pad_sequences(final_projects,maxlen=150)

# ### Loading word embeddings

# In[ ]:


EMBEDDING_FILE = '../input/glove840b300dtxt/glove.840B.300d.txt'
embeddings_index = {}
with open(EMBEDDING_FILE,encoding='utf8') as f:
    for line in f:
        values = line.rstrip().rsplit(' ')
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

# In[ ]:


max_features = 100000
embed_size=300

word_index = tok.word_index
#prepare embedding matrix

num_words = min(max_features, len(word_index) + 1)
embedding_matrix = np.zeros((num_words, embed_size))
for word, i in word_index.items():
    if i >= max_features:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

# #### Now, use this embedding matrix to get word embeddings, use it with donors and then  try to come up with the relations between the donors based on their likes,interests.

# ### Context based recommendations
# * Matrix factorization
# * SVD

# ## Deep Learning based approaches:
# Assume the data is normally distributed, we need to find out maximize log likelihood of the users.

# # <a id='6-2'>6.2 Collaborative Filtering(CF) Method </a>
# These models adopt donors or projects previous history of interactions, such as the funding given to the project by the donor, which have attracted more attention due to their better prediction performance than the *content-based method*. 
# Two  effective CF-based methods are:-
# 1. Matrix factorization (MF)
# 2. Restricted Boltzmann machine (RBM)
# 
# MF directly learns the latent vectors of the donors and the projects from the donor-project funding matrix and captures the interaction between the donor and the project. Complex interactions cannot be captured by MF since its estimated funding matrix is produced by the simple inner product between corresponding latent vectors of donor and project.
# 
# The RBM-like methods explicitly make recommendation from either donor or project side via constructing independent models for donors or projects, respectively. But this also has a drawback, in this model the correlation can only be considered from a single side, that is project–project or donor–donor, thus ignoring the other completely. 
# 
# ### To handle above drawbacks:-
# We can create feed-forward neural networks that will take the information of both given donor and project with their corresponding historical information into consideration. 
# This can be thought of a **sequence modeling** task.

# ### Questions to ponder:-
# * Will **RMSE** work here as an evaluation metric?
# * Do we need any extra information about donors or projects for better recommendations?

# ### STAY TUNED..I will be adding more exploration and finally the model... :) Please Upvote.. :)

# In[ ]:



