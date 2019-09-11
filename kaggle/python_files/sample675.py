#!/usr/bin/env python
# coding: utf-8

# ## Introduction
# In the recent times, Data Scientist / Machine Learning Engineer has become one of the most sought after profession. Even HBR regarded Data Scientist as the 'sexiest job of the 21st century'.
# 
# 

# ![Data Scientist ](https://datastoriesthailand.files.wordpress.com/2015/02/screenshot-2015-02-02-21-18-24.png)

# So there is a lot of interest among people to learn about Data Science. Also Data science is a multi-disciplinary domain which requires knowledge of multiple subjects. The following picture will give an idea. 

# ![DS_Subjects](https://cdn-images-1.medium.com/max/1600/1*tMm8qCW59DCK7G0hb-HevQ.png)

# Given this situation, one main question that comes to people's mind is that where can I learn about DS and ML. An answer to this question will be very helpful for people to get started in this field. So in this notebook, let us explore the different options where people learn about DS / ML skills. 
# 
# **Most of the plots are interactive. So please feel free to hover over the plots, zoom in / out, rotate them as needed**
# 
# Firstly, I would like to thank Kaggle for conducting this DS / ML survey again this year and making the data available for people like us to use.

# In[ ]:


import numpy as np 
import pandas as pd 
import os
import math
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing

from plotly import tools
from IPython.core import display as ICD
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
import warnings
warnings.filterwarnings("ignore")
init_notebook_mode(connected=True)
pd.set_option('display.max_columns', 5000)

# ## Learning Category:
# 
# In this intial phase of the analysis, let us take up question 35 in the survey. The question is:
# 
# *What percentage of your machine learning/ data science training falls under each category?*
# 
# The choices given for this question are
#  1. Self-taught 
#  2. Online courses like Coursera, Udemy, edX etc
#  3. Work
#  4. University
#  5. Kaggle Competiitons
#  6. Other - Free text field
#  
# We need to give a percentage value for each of these learning categories and the total should sum up to 100%.
# 
# Overall there are 23,858 responses in this survey and let us check the number of respondents for each of these training categories (percentage of the category is greater than 0).

# In[ ]:


base_dir = '../input/kaggle-survey-2018/'
fileName = 'multipleChoiceResponses.csv'
filePath = os.path.join(base_dir,fileName)
survey_df = pd.read_csv(filePath) 
responses_df = survey_df[1:]
responses_df_orig = responses_df.copy()

# In[ ]:


responses_df = responses_df[~pd.isnull(responses_df['Q35_Part_1'])]

# In[ ]:


count_dict = {
    'Self-taught' : (responses_df['Q35_Part_1'].astype(float)>0).sum(),
    'Online courses (Coursera, Udemy, edX, etc.)' : (responses_df['Q35_Part_2'].astype(float)>0).sum(),
    'Work' : (responses_df['Q35_Part_3'].astype(float)>0).sum(),
    'University' : (responses_df['Q35_Part_4'].astype(float)>0).sum(),
    'Kaggle competitions' : (responses_df['Q35_Part_5'].astype(float)>0).sum(),
    'Other' : (responses_df['Q35_Part_6'].astype(float)>0).sum()
}

cnt_srs = pd.Series(count_dict)

trace = go.Bar(
    x=cnt_srs.index,
    y=cnt_srs.values,
    marker=dict(
        color='blue',
    #    colorscale = 'Picnic',
    #    reversescale = True
    ),
)

layout = go.Layout(
    title='Number of respondents for each learning category'
)

data = [trace]
fig = go.Figure(data=data, layout=layout)
iplot(fig, filename="learningcategory")

# **Observations:**
#  * There are some missing values for this question and after dropping them we have 15,745 responses in total.
#  * 'Self taught is the category with most number of respondents having percentage of learning greater than 0.
#  * With the recent explosion of MOOC courses, 'Online courses' come in second 
#  * Learning as part of work is third and traditional way of learning - 'University' is fourth
#  * Though Kaggle competitions take the fifth spot, the number of respondents is not much lesser than third and forth place.
#  
#  
#  ### Percentage Contribution of Learning Categories:
#  
#  Now let us see, how much percentage each of the learning categories contribute to the learning process.

# In[ ]:


responses_df['Q35_Part_5'].loc[responses_df['Q35_Part_5'].astype(float)<0] = 0   

ys = [responses_df['Q35_Part_1'].values, 
      responses_df['Q35_Part_2'].values,
      responses_df['Q35_Part_3'].values,
      responses_df['Q35_Part_4'].values,
      responses_df['Q35_Part_5'].values,
      responses_df['Q35_Part_6'].values
     ]
names = ["Self-taught",
         'Online courses (Coursera, Udemy, edX, etc.)',
         'Work',
         'University',
         'Kaggle competitions',
         'Other'
        ]
#colors = ['rgba(93, 164, 214, 0.5)', 'rgba(255, 144, 14, 0.5)', 'rgba(44, 160, 101, 0.5)', 'rgba(255, 65, 54, 0.5)', 'rgba(207, 114, 255, 0.5)', 'rgba(127, 96, 0, 0.5)']
colors = ["#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71","#df6a84"]

trace = []

for i in range(6):
    trace.append ( 
        go.Box(
            y=ys[i],
            name=names[i],
            marker = dict(
                color=colors[i],
            )
        )
    )
layout = go.Layout(
    title='Box plots on % contribution of each ML / DS training category'
)

fig = go.Figure(data=trace, layout=layout)
iplot(fig, filename="TimeSpent")

# **Observations:**
#  
#  * Looking at the median of each of the learning categories, it seems there is no one category that completely dominated the learning process of ML / DS
#  * Self-taught seems to have higher percentage of share in the learning process compared to others. 
#  * Only less than half of the respondents have the percentage share of 'University' as greater than 0 
#  
#  
#  ### Distribution of DS / ML Learning Category at different Countries:
#  
#  Now let us have a look at how these learning categories are distributed across the top countries. We will take the respondents from top 10 countries and do the analysis.

# In[ ]:


def get_trace(country_name):
    responses_df = responses_df_orig.copy()
    responses_df = responses_df[responses_df["Q3"]==country_name]
    responses_df['Q35_Part_5'].loc[responses_df['Q35_Part_5'].astype(float)<0] = 0   
    ys = [responses_df['Q35_Part_1'].values, 
      responses_df['Q35_Part_2'].values,
      responses_df['Q35_Part_3'].values,
      responses_df['Q35_Part_4'].values,
      responses_df['Q35_Part_5'].values,
      responses_df['Q35_Part_6'].values
     ]
    names = ["Self-taught",
         'Online courses (Coursera, Udemy, edX, etc.)',
         'Work',
         'University',
         'Kaggle competitions',
         'Other'
        ]
    colors = ["#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71","#df6a84"]

    trace = []

    for i in range(6):
        trace.append ( 
            go.Box(
                y=ys[i],
                name=names[i],
                marker = dict(
                    color=colors[i],
                )
            )
        )
    return trace
    
traces_list = []
country_names = ["United States of America", "India", "China", "Russia", "Brazil", "Germany",
                "United Kingdom of Great Britain and Northern Ireland", "Canada", "France", "Japan"]
for country_name in country_names:
    traces_list.append(get_trace(country_name))
    
# Creating two subplots
fig = tools.make_subplots(rows=int(len(country_names)/2), cols=2, vertical_spacing=0.1, 
                          subplot_titles=country_names)

for ind, traces in enumerate(traces_list):
    for trace in traces:
        fig.append_trace(trace, int(np.floor(ind/2)+1), int((ind%2) + 1))

fig['layout'].update(height=2000, width=800, paper_bgcolor='rgb(233,233,233)', title="% Contribution of DS / ML training by Country", showlegend=False)
iplot(fig, filename='mldscountry')


# **Observations:**
#  * Compared to other top countries, the percentage contribution of universities in USA is much higher. I think this might be because there are multiple universities in US offering courses in DS / ML when compared with other countries.
#  * In countries like India, Russia & Japan, the role of universities in Learning DS / ML is much lesser compared to other categories. 
#  * Also if we look at **Asian countries** in the list (India, China, Russia and Japan), **median percentage of Kaggle contribution** for learning DS / ML is greater than 0 while it is zero for other countries. 
#  * Contribution from learning at Work is more in Russia, France and Japan
#  * In Brazil, the contribution of MOOC courses seem to be more than other learning categories
#  * I personally think these plots directly represent the comfortable ways to acquire knowledge at these corresponding regions.
#  
#  
#  ### Distribution of Learning Category By Profession:
#  
#  In this section, let us look at the how the learning categories vary based on the profession.

# In[ ]:


def get_trace(country_name):
    responses_df = responses_df_orig.copy()
    responses_df = responses_df[responses_df["Q6"]==country_name]
    responses_df['Q35_Part_5'].loc[responses_df['Q35_Part_5'].astype(float)<0] = 0   
    ys = [responses_df['Q35_Part_1'].values, 
      responses_df['Q35_Part_2'].values,
      responses_df['Q35_Part_3'].values,
      responses_df['Q35_Part_4'].values,
      responses_df['Q35_Part_5'].values,
      responses_df['Q35_Part_6'].values
     ]
    names = ["Self-taught",
         'Online courses (Coursera, Udemy, edX, etc.)',
         'Work',
         'University',
         'Kaggle competitions',
         'Other'
        ]
    colors = ["#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71","#df6a84"]

    trace = []

    for i in range(6):
        trace.append ( 
            go.Box(
                y=ys[i],
                name=names[i],
                marker = dict(
                    color=colors[i],
                )
            )
        )
    return trace
    
traces_list = []
country_names = ["Student", "Data Scientist", "Software Engineer", "Data Analyst"]
for country_name in country_names:
    traces_list.append(get_trace(country_name))
    
# Creating two subplots
fig = tools.make_subplots(rows=int(len(country_names)/2), cols=2, vertical_spacing=0.2, 
                          subplot_titles=country_names)

for ind, traces in enumerate(traces_list):
    for trace in traces:
        fig.append_trace(trace, int(np.floor(ind/2)+1), int((ind%2) + 1))

fig['layout'].update(height=800, width=800, paper_bgcolor='rgb(233,233,233)', title="% Contribution of DS / ML training by Profession", showlegend=False)
iplot(fig, filename='mldscountry')


# **Observations:**
#  
#  * As expected, University plays a major role in imparting DS / ML knowledge among students and 'Work' has the least contribution
#  * In case of Data Scientist, most of the respondents have mentioned that 'Work' plays a major role in learning the concepts. Self-learning also plays an equally important role.
#  * For 'Software Enginner' who are learning DS / ML, 'Self-taught' and 'Online courses' are the ways to acquire knowledge compared to other means.
#  * Also respondents with title 'Software Engineer' mentioned that Kaggle competitions share a higher percentage in learning DS / ML compared to other professions (looking at the third quartile)
#  
#  
#  ### Distribution of Learning Category by Degree Attained:
#  
#  Now let us see how the learning categories vary based on the highest degree attained.

# In[ ]:


def get_trace(country_name):
    responses_df = responses_df_orig.copy()
    responses_df = responses_df[responses_df["Q4"]==country_name]
    responses_df['Q35_Part_5'].loc[responses_df['Q35_Part_5'].astype(float)<0] = 0   
    ys = [responses_df['Q35_Part_1'].values, 
      responses_df['Q35_Part_2'].values,
      responses_df['Q35_Part_3'].values,
      responses_df['Q35_Part_4'].values,
      responses_df['Q35_Part_5'].values,
      responses_df['Q35_Part_6'].values
     ]
    names = ["Self-taught",
         'Online courses (Coursera, Udemy, edX, etc.)',
         'Work',
         'University',
         'Kaggle competitions',
         'Other'
        ]
    colors = ["#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71","#df6a84"]

    trace = []

    for i in range(6):
        trace.append ( 
            go.Box(
                y=ys[i],
                name=names[i],
                marker = dict(
                    color=colors[i],
                )
            )
        )
    return trace
    
traces_list = []
country_names = ["Master’s degree", "Bachelor’s degree", "Doctoral degree", "Professional degree"]
for country_name in country_names:
    traces_list.append(get_trace(country_name))
    
# Creating two subplots
fig = tools.make_subplots(rows=int(len(country_names)/2), cols=2, vertical_spacing=0.2, 
                          subplot_titles=country_names)

for ind, traces in enumerate(traces_list):
    for trace in traces:
        fig.append_trace(trace, int(np.floor(ind/2)+1), int((ind%2) + 1))

fig['layout'].update(height=800, width=800, paper_bgcolor='rgb(233,233,233)', title="% Contribution of DS / ML training by Degree", showlegend=False)
iplot(fig, filename='mldscountry')


# **Observations:**
#  * Respondents having a Masters or Doctoral degree have a higher contribution for learning DS / ML from university compared to other two sections (looking at the third quartile of university)
#  * Respondents with Bachelors degree have higher contribution of learning from self taught courses and online courses (looking at the median of different learning categories of Bachelors degree)
#  * Learning from Kaggle competitions seem to have a fairly stable contribution across all sections.
#  
# ### Distribution of Learning Category by Gender

# In[ ]:


def get_trace(country_name):
    responses_df = responses_df_orig.copy()
    responses_df = responses_df[responses_df["Q1"]==country_name]
    responses_df['Q35_Part_5'].loc[responses_df['Q35_Part_5'].astype(float)<0] = 0   
    ys = [responses_df['Q35_Part_1'].values, 
      responses_df['Q35_Part_2'].values,
      responses_df['Q35_Part_3'].values,
      responses_df['Q35_Part_4'].values,
      responses_df['Q35_Part_5'].values,
      responses_df['Q35_Part_6'].values
     ]
    names = ["Self-taught",
         'Online courses (Coursera, Udemy, edX, etc.)',
         'Work',
         'University',
         'Kaggle competitions',
         'Other'
        ]
    colors = ["#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71","#df6a84"]

    trace = []

    for i in range(6):
        trace.append ( 
            go.Box(
                y=ys[i],
                name=names[i],
                marker = dict(
                    color=colors[i],
                )
            )
        )
    return trace
    
traces_list = []
country_names = ["Male", "Female"]
for country_name in country_names:
    traces_list.append(get_trace(country_name))
    
# Creating two subplots
fig = tools.make_subplots(rows=int(len(country_names)/2), cols=2, vertical_spacing=0.2, 
                          subplot_titles=country_names)

for ind, traces in enumerate(traces_list):
    for trace in traces:
        fig.append_trace(trace, int(np.floor(ind/2)+1), int((ind%2) + 1))

fig['layout'].update(height=400, width=800, paper_bgcolor='rgb(233,233,233)', title="% Contribution of DS / ML training by Gender", showlegend=False)
iplot(fig, filename='mldscountry')

# **Observations:**
# 
# * Looking at the third quartile of all the categories, 'Self-taught' has a higher contribution for Male while 'University' has the higher contribution for Female. 
# 
# ### Distrbution of Learning Categories by Age:
#  
#  In this section, let us see how the percentage contribution each of the learning categories change based on age.

# In[ ]:


name_dict = {
    'Q35_Part_1' : "Self-taught",
    'Q35_Part_2' : "Online Courses",
    'Q35_Part_3' : "Work",
    'Q35_Part_4' : "University",
    'Q35_Part_5' : "Kaggle competitions"
}
colors = ["#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71","#df6a84"]
def get_trace(country_name, color):
    responses_df = responses_df_orig.copy()
    #responses_df = responses_df[responses_df["Q4"]==country_name]
    responses_df['Q35_Part_5'].loc[responses_df['Q35_Part_5'].astype(float)<0] = 0  
    responses_df = responses_df.sort_values(by="Q2")
    trace = go.Box(
        y = responses_df[country_name].values,
        x = responses_df['Q2'].values,
        name = name_dict[country_name],
        marker = dict(
                    color=color,
                )
    )
    return trace

traces_list = []
country_names = ["Q35_Part_1", "Q35_Part_2", "Q35_Part_3", "Q35_Part_4", "Q35_Part_5"]
for ind, country_name in enumerate(country_names):
    traces_list.append(get_trace(country_name, colors[ind]))
    
# Creating two subplots
fig = tools.make_subplots(rows=int(len(country_names)), cols=1, vertical_spacing=0.05, 
                          subplot_titles=[name_dict[cn] for cn in country_names])

for ind, trace in enumerate(traces_list):
        fig.append_trace(trace, ind+1, 1)

fig['layout'].update(height=1600, width=800, paper_bgcolor='rgb(233,233,233)', title="% Contribution of DS / ML training category by Age", showlegend=True)
iplot(fig, filename='mldscountry')


# **Observations:**
#  * 'Self-taught' category contributes more for respondents aged more than 35 compared to respondents aged less than 35
#  * Looking at the distribution of online courses by age, we can see that respondents aged more than 60 seem to have a lower median score compared to other age groups
#  * Median contribution of 'Work' as learning category is high for middle aged people compared to younger and older ones
#  * Median contribution of 'University' as learning category is high for people less than 30 years of age
#  * Contribution of Kaggle as a learning category is fairly consistent across age groups with a slght higher third quartile for people aged less than 21. Looks like Kaggle is quite popular with the younger bunch ;)
#  
# 
# ### Other ML / DS Learning Category - FreeForm Text
#  
#  There is also a free form text column that contains the responses for the DS / ML learning category apart from the choices given. Let us look at them.

# In[ ]:


survey_freeform_df = pd.read_csv(base_dir + "freeFormResponses.csv").loc[1:,:]
col_name = "Q35_OTHER_TEXT"

def preprocess(x):
    if str(x) != "nan":
        x = str(x).lower()
        if x[-1] == "s":
            x = x[:-1]
    return x

survey_freeform_df[col_name] = survey_freeform_df[col_name].apply(lambda x: preprocess(x))

cnt_srs = survey_freeform_df[col_name].value_counts().head(20)

trace = go.Bar(
    x=cnt_srs.index,
    y=cnt_srs.values,
    marker=dict(
        color='blue',
    ),
)

layout = go.Layout(
    title='Count of other DS / ML learning category - free form text'
)

data = [trace]
fig = go.Figure(data=data, layout=layout)
iplot(fig, filename="platform")

# **Observations:**
# 
# * Apart from the given choices, bootcamps and books seem to be the next popular choices of learning DS / ML.
#  
#  
#  ## Online Course Platforms
# 
# As we could see from the previous analysis, online platforms play a major role in imparting DS / ML education. So now let us focus on the different online plarforms. There are multiple online platforms available from which we can learn DS. Some of them can be seen below.

# ![dsonline](https://qph.fs.quoracdn.net/main-qimg-462634f264ae54c975af145e518fe801)

# ### Number of Respondents for each online platform
# 
# First let us look at the number of respondents for each online platform.

# In[ ]:


responses_df = responses_df_orig.copy()

count_dict = {
    'Coursera' : (responses_df['Q36_Part_2'].count()),
    'Udemy' : (responses_df['Q36_Part_9'].count()),
    'DataCamp' : (responses_df['Q36_Part_4'].count()),
    'Kaggle Learn' : (responses_df['Q36_Part_6'].count()),
    'Udacity' : (responses_df['Q36_Part_1'].count()),
    'edX' : (responses_df['Q36_Part_3'].count()),
    'Online University Courses' : (responses_df['Q36_Part_11'].count()),
    'Fast.AI' : (responses_df['Q36_Part_7'].count()),
    'Developers.google.com' : (responses_df['Q36_Part_8'].count()),
    'DataQuest' : (responses_df['Q36_Part_5'].count()),
    'The School of AI' : (responses_df['Q36_Part_10'].count())
}


cnt_srs = pd.Series(count_dict)

trace = go.Bar(
    x=cnt_srs.index,
    y=cnt_srs.values,
    marker=dict(
        color='green',
    ),
)

layout = go.Layout(
    title='Number of Respondents for each online platform'
)

data = [trace]
fig = go.Figure(data=data, layout=layout)
iplot(fig, filename="platform")

# We also have a free form text column for this question, to add any other sources apart from the one mentioned. Let us look at them now.

# In[ ]:


col_name = "Q36_OTHER_TEXT"

def preprocess(x):
    if str(x) != "nan":
        x = str(x).lower()
    return x

survey_freeform_df[col_name] = survey_freeform_df[col_name].apply(lambda x: preprocess(x))

cnt_srs = survey_freeform_df[col_name].value_counts().head(20)

trace = go.Bar(
    x=cnt_srs.index,
    y=cnt_srs.values,
    marker=dict(
        color='red',
    ),
)

layout = go.Layout(
    title='Number of respondents for other online platforms - free form text'
)

data = [trace]
fig = go.Figure(data=data, layout=layout)
iplot(fig, filename="platform")

# 
# ### Most Used Online Platform to Learn DS
# 
# Next let us look at the online platforms where the people had spent most of their time.

# In[ ]:


responses_df = responses_df_orig.copy()
temp_series = responses_df['Q37'].value_counts()
labels = (np.array(temp_series.index))
sizes = (np.array((temp_series / temp_series.sum())*100))

trace = go.Pie(labels=labels, values=sizes)
layout = go.Layout(
    title='Most Used Online Platform distribution'
)
data = [trace]
fig = go.Figure(data=data, layout=layout)
iplot(fig, filename="onlinecourse")

# **Observations:** 
#  * Coursera tops the list with about 39% of the respondents mentioning that they spent most of their time there.
#  * DataCamp and Udemy are neck to neck with each other with about 12% share
#  * Udacity and edX are about 8.5% and 8% respectively
#  * Though Kaggle Learn is relatively new, it is preferred by about 7% of respondents
#  
#  
#   ### Most used Online Platform by Country:
#   
#   The world map plots are interactive. Please rotate them to have a better view of the countries you would like to see.

# In[ ]:


responses_df = responses_df_orig.copy()

from sklearn import preprocessing
lbl = preprocessing.LabelEncoder()

gdf = responses_df.groupby(['Q3', 'Q37']).size().reset_index()
gdf.columns = ['country', 'platform', 'count']
gdf = gdf.sort_values(by=['country','count'])
gdf = gdf.drop_duplicates(subset='country', keep='last')
gdf['count'] = lbl.fit_transform(gdf['platform'].values)

colorscale = [[0, 'rgb(102,194,165)'], [0.33, 'rgb(253,174,97)'], [0.66, 'rgb(213,62,79)'], [1.0, 'rgb(158,1,66)']]
data = [ dict(
        type = 'choropleth',
        autocolorscale = False,
        colorscale = colorscale,
        showscale = False,
        locations = gdf.country,
        z = gdf['count'].values,
        locationmode = 'country names',
        text = gdf.platform,
        marker = dict(
            line = dict(color = '#fff', width = 2)) )           ]

layout = dict(
    height=800,
    title = 'Most Used Online Platform by Country',
    geo = dict(
        showframe = True,
        showocean = True,
        oceancolor = '#222',
        projection = dict(
        type = 'orthographic',
            rotation = dict(
                    lon = 60,
                    lat = 10),
        ),
        lonaxis =  dict(
                showgrid = False,
                gridcolor = 'rgb(102, 102, 102)'
            ),
        lataxis = dict(
                showgrid = False,
                gridcolor = 'rgb(102, 102, 102)'
                )
            ),
        )
fig = dict(data=data, layout=layout)
iplot(fig)

# **Observations:**
#  * As we could see from the map above, coursera is the most commonly used platform in almost all countries
#  * In Turkey, South Africa and Philippines, it is Udemy.
#  * DataCamp is more popular in Indonesia and edX is the most popular in Nigeria
#  
#  
#  ### Second Most Used Online Platform by Country

# In[ ]:


responses_df = responses_df_orig.copy()
lbl = preprocessing.LabelEncoder()

gdf = responses_df.groupby(['Q3', 'Q37']).size().reset_index()
gdf.columns = ['country', 'platform', 'count']
gdf = gdf.sort_values(by=['country','count'], ascending=False)
gdf = gdf.groupby(['country']).nth(2).reset_index()
gdf['count'] = lbl.fit_transform(gdf['platform'].values)

colorscale = [[0, 'rgb(102,194,165)'], [0.33, 'rgb(253,174,97)'], [0.66, 'rgb(213,62,79)'], [1.0, 'rgb(158,1,66)']]
data = [ dict(
        type = 'choropleth',
        autocolorscale = False,
        colorscale = 'picnic',
        showscale = False,
        locations = gdf.country,
        z = gdf['count'].values,
        locationmode = 'country names',
        text = gdf.platform,
        marker = dict(
            line = dict(color = '#fff', width = 2)) )           ]

layout = dict(
    height=800,
    title = 'Second Most Used Online Platform by Country',
    geo = dict(
        showframe = True,
        showocean = True,
        oceancolor = '#222',
        projection = dict(
        type = 'orthographic',
            rotation = dict(
                    lon = 60,
                    lat = 10),
        ),
        lonaxis =  dict(
                showgrid = False,
                gridcolor = 'rgb(102, 102, 102)'
            ),
        lataxis = dict(
                showgrid = False,
                gridcolor = 'rgb(102, 102, 102)'
                )
            ),
        )
fig = dict(data=data, layout=layout)
iplot(fig)

# **Observations:**
#  * Udemy is the second most used online platform to learn about DS / ML at US, UK etc
#  * In India, Canada, Spain etc, DataCamp is the second most used online platform for learning
#  * In EU countries like Germany, Italy, Portugal, Sweden etc Udacity is the second most popular platform
#  * edX is the second widely used platform in Australia, Argentina etc
#  * Kaggle Learn is second most used platform to learn in Russia, France, Ukraine etc
#  

# ### Online Courses Vs Brick & Mortar
# 
# We have seen that online courses have gained a lot of interest in the previous sections. Now let us see what people perceive about the quality of online courses compared to traditional brick and mortar ones.

# In[ ]:


responses_df = responses_df_orig.copy()

temp_series = responses_df['Q39_Part_1'].value_counts()
labels = (np.array(temp_series.index))
sizes = (np.array((temp_series / temp_series.sum())*100))

trace = go.Pie(labels=labels, values=sizes, hole=0.4)
layout = go.Layout(
    height = 700,
    width = 700,
    title='How are Online Courses compared to Brick & Mortar Courses',
)
data = [trace]
fig = go.Figure(data=data, layout=layout)
iplot(fig, filename="onlinecourse")

# **Observations:**
#  * About **53%** of the respondents feel that **online courses are better** than traditional brick and mortar courses
#  * About 33% of the respondents are neutral
#  * Overall, people seem to be more satisfied with online courses.
#  
#  
# 
#  
#  

# ## Coursera Data Science Courses Review Dataset
# 
# Now that we got an idea about the perception of people about online courses, let us check the perception from some other place. Thankfully we also have a [coursera course review dataset](https://www.kaggle.com/septa97/100k-courseras-course-reviews-dataset#reviews_by_course.csv) in Kaggle datasets. So in this section let us use this dataset to make some plots and see if they also give similar results.

# In[ ]:


coursera_df = pd.read_csv("../input/100k-courseras-course-reviews-dataset/reviews_by_course.csv")
courses = ["machine-learning", "python-data", "r-programming", "data-scientists-tools", "ml-foundations", "python-data-analysis"]
coursera_df = coursera_df[coursera_df["CourseId"].isin(courses)]

cnt_srs = coursera_df["CourseId"].value_counts()

trace = go.Bar(
    x=cnt_srs.index,
    y=cnt_srs.values,
    marker=dict(
        color='yellow',
    ),
)

layout = go.Layout(
    title='Number of Reviews for each coursera course'
)

data = [trace]
fig = go.Figure(data=data, layout=layout)
iplot(fig, filename="platform")


# **Observations:**
#  * Looks like the "machine learning" course in coursera is one of the widely popular courses and it has the highest number of reviews
#  * "python-data" is the second one with most number of reviews followed by "data-scientist-tools"

# In[ ]:


names = []
ones = []
twos = []
threes = []
fours = []
fives = []
for col in courses[::-1]:
    tmp_df = coursera_df[coursera_df["CourseId"]==col]
    cnt_srs= tmp_df["Label"].value_counts()
    cnt_srs_sum = float(cnt_srs.values.sum()) 
    names.append(col)
    ones.append(cnt_srs[1] / cnt_srs_sum * 100)
    twos.append(cnt_srs[2] / cnt_srs_sum * 100)
    threes.append(cnt_srs[3] / cnt_srs_sum * 100)
    fours.append(cnt_srs[4] / cnt_srs_sum * 100)
    fives.append(cnt_srs[5] / cnt_srs_sum * 100)

trace1 = go.Bar(
    y=names,
    x=ones,
    orientation = 'h',
    name = "Very Negative"
)
trace2 = go.Bar(
    y=names,
    x=twos,
    orientation = 'h',
    name = "Negative"
)
trace3 = go.Bar(
    y=names,
    x=threes,
    orientation = 'h',
    name = "Neutral"
)
trace4 = go.Bar(
    y=names,
    x=fours,
    orientation = 'h',
    name = "Positive"
)
trace5 = go.Bar(
    y=names,
    x=fives,
    orientation = 'h',
    name = "Very Positive"
)

layout = go.Layout(
    title='Coursera Course Reviews in Percentage',
    barmode='stack',
    width = 800,
    height = 800,
    #yaxis=dict(tickangle=-45),
)

data = [trace5, trace4, trace3, trace2, trace1]
fig = go.Figure(data=data, layout=layout)
iplot(fig, filename="CourseraReviews")

# **Observations:**
#  * Machine Learning in coursera is not just the most popular course by count, it is the course with most percentage of "very positive" reviews as well. I think this is one of the very popular starting courses for people wanting to learn DS / ML
#  * Overall, the positive reviews are higher than the negative reviews for all the courses and is inline with Kaggle survey results as well. 
#  * Now you know which course to start first ;)

# ## In-person Bootcamps
# 
# Apart from online courses, one another recent addition that has gained popularity to learn data science is in-person bootcamps. More information about the bootcamps can be seen [here](https://www.cio.com/article/3051124/careers-staffing/10-boot-camps-to-kick-start-your-data-science-career.html) and [here](https://www.switchup.org/rankings/best-data-science-bootcamps).
# 
# ![BootCamps](https://www.kdnuggets.com/images/nycdsa-data-science-bootcamp.jpg)
# 
# In this section, let us see how people perceive bootcamps compared to traditional brick & mortar courses.

# In[ ]:


responses_df = responses_df_orig.copy()

temp_series = responses_df['Q39_Part_2'].value_counts()
labels = (np.array(temp_series.index))
sizes = (np.array((temp_series / temp_series.sum())*100))

trace = go.Pie(labels=labels, values=sizes, hole=0.4)
layout = go.Layout(
    height = 700,
    width = 700,
    title='How are In-person bootcamps compared to Tradational Brick & Mortar Institutions',
)
data = [trace]
fig = go.Figure(data=data, layout=layout)
iplot(fig, filename="bootcamps")

# **Observations:**
#  * Seems like a good part of the respondents are not aware of the bootcamps. About 33% of the people took a "No opinion / I do not know" stance
#  * About 39.5% of the people feel that they are better than traditional institutions
#  * About 18.8% of the respondents are neutral on their views

#  ## Media Sources for Data Science
#  
#  There are quite a few media sources available for us to keep a tab on the happenings in Data Science space. In this section, let us explore the data science media sources.
#  
#  ![Media Sources](https://i1.wp.com/blog.udacity.com/wp-content/uploads/2014/12/24-DA-resources.png?resize=640%2C1333)

#  ### Favorite Media Sources

# In[ ]:


responses_df = responses_df_orig.copy()

count_dict = {
    'Kaggle Forums' : (responses_df['Q38_Part_4'].count()),
    'Medium Blog Posts' : (responses_df['Q38_Part_18'].count()),
    'ArXiv & Preprints' : (responses_df['Q38_Part_11'].count()),
    'Twitter' : (responses_df['Q38_Part_1'].count()),
    'None / I do not know' : (responses_df['Q38_Part_21'].count()),
    'r/machinelearning' : (responses_df['Q38_Part_3'].count()),
    'KDNuggets' : (responses_df['Q38_Part_14'].count()),
    'Journal Publications' : (responses_df['Q38_Part_12'].count()),
    'Siraj Raval Youtube' : (responses_df['Q38_Part_6'].count()),
    'HackerNews' : (responses_df['Q38_Part_2'].count()),
    'FiveThirtyEight.com' : (responses_df['Q38_Part_10'].count())
}


cnt_srs = pd.Series(count_dict)

trace = go.Bar(
    x=cnt_srs.index,
    y=cnt_srs.values,
    marker=dict(
        color='red',
    ),
)

layout = go.Layout(
    title='Favorite Media Source on DS topics'
)

data = [trace]
fig = go.Figure(data=data, layout=layout)
iplot(fig, filename="platform")

# ### Favorite Media Sources By Country
# 
#   The below plot is interactive. Please rotate them to have a better view of the countries you would like to see.

# In[ ]:


responses_df = responses_df_orig.copy()

map_dict = {
    'Q38_Part_4' : 'Kaggle Forums',
    'Q38_Part_18' : 'Medium Blog Posts',
    'Q38_Part_11' : 'ArXiv & Preprints',
    'Q38_Part_1' : 'Twitter', 
    'Q38_Part_21' : 'None / I do not know',
    'Q38_Part_3' : 'r/machinelearning',
    'Q38_Part_14' : 'KDNuggets',
    'Q38_Part_12' : 'Journal Publications',
    'Q38_Part_6' : 'Siraj Raval Youtube',
    'Q38_Part_2' : 'HackerNews',
    'Q38_Part_10' : 'FiveThirtyEight.com'
}

fdf = pd.DataFrame()
for key, item in map_dict.items():
    tdf = responses_df.groupby('Q3')[key].count().reset_index()
    tdf.columns = ['country', 'cnt']
    tdf['source'] = item
    fdf = pd.concat([fdf, tdf])

fdf = fdf.sort_values(by=['country','cnt'])
fdf = fdf.drop_duplicates(subset='country', keep='last')
lbl = preprocessing.LabelEncoder()
fdf['source_lbl'] = lbl.fit_transform(fdf['source'].values)
    
data = [ dict(
        type = 'choropleth',
        autocolorscale = False,
        colorscale = 'Jet',
        showscale = False,
        locations = fdf.country,
        z = fdf['source_lbl'].values,
        locationmode = 'country names',
        text = fdf.source,
        marker = dict(
            line = dict(color = '#fff', width = 2)) )           ]

layout = dict(
    height=800,
    title = 'Favorite media sources by Country',
    geo = dict(
        showframe = True,
        showocean = True,
        oceancolor = '#222',
        projection = dict(
        type = 'orthographic',
            rotation = dict(
                    lon = 270,
                    lat = 10),
        ),
        lonaxis =  dict(
                showgrid = False,
                gridcolor = 'rgb(102, 102, 102)'
            ),
        lataxis = dict(
                showgrid = False,
                gridcolor = 'rgb(102, 102, 102)'
                )
            ),
        )
fig = dict(data=data, layout=layout)
iplot(fig)

# ### Other Media Sources : Free Form Text
# 
# We also have a free form text column for other media sources. So let us look at this column to get an idea of the other favorite ML / DS sources.

# In[ ]:


from wordcloud import WordCloud, STOPWORDS

survey_freeform_df = pd.read_csv(base_dir + "freeFormResponses.csv").loc[1:,:]
col_name = "Q38_OTHER_TEXT"

text = ''
text_array = survey_freeform_df[col_name].values
for ind, t in enumerate(text_array):
    if str(t) != "nan":
        text = " ".join([text, "".join(t.lower())])
text = text.strip()
    
plt.figure(figsize=(24.0,16.0))
wordcloud = WordCloud(background_color='black', width=800, height=400, max_font_size=100, max_words=100).generate(text)
wordcloud.recolor(random_state=ind*312)
plt.imshow(wordcloud)
plt.title("Other online resources for ML / DS", fontsize=40)
plt.axis("off")
#plt.show()
plt.tight_layout() 


# **Observations:**
#  * ods.ai seem to be one another popular media source 
#  * We could also see some social media channels like linkedin, facebook, youtube, quora etc
#  * Being from India, I could also see Analytics Vidhya mentioned in few places ( which means it needs some better cleaning ;) )

# 
# Thank you & Happy Learning.!
#  
#  **References:**
#   1. https://www.kaggle.com/shivamb/exploratory-analysis-ga-customer-revenue
# 
