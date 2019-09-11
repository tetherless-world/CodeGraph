#!/usr/bin/env python
# coding: utf-8

# ## General information
# 
# Some time ago Kaggle launched a big online survey for kagglers and now this data is public. There were multiple choice questions and some forms for open answers. Survey received 23,859 usable respondents from 147 countries and territories. As a result we have a big dataset with rich information on data scientists using Kaggle.
# 
# In this kernel I'll try to analyze this data and provide various insights. Main tools for this will be Python and seaborn + plotly.
# 
# ***Scroll down to find new interactive visualizations where you can choose any country and the graph will be shown for it***
# 
# ![](https://www.kaggle.com/static/images/host-home/host-home-business.png)
# 
# ### Main idea: comparing countries
# 
# While there are many ways to analyse the data, I have decided to perform the analysis based on the country of the responders. You'll see more about this lower.

# In[ ]:


# libraries
import numpy as np 
import pandas as pd 
import os
import math
import matplotlib.pyplot as plt
import seaborn as sns
from plotly import tools
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
import warnings
warnings.filterwarnings("ignore")
init_notebook_mode(connected=True)
pd.set_option('display.max_columns', None)
from ipywidgets import interact, interactive, interact_manual
import ipywidgets as widgets
import colorlover as cl


# In[ ]:


#loading data
DIR = '../input/kaggle-survey-2018/'
df_free = pd.read_csv(DIR + 'freeFormResponses.csv', low_memory=False, header=[0,1])
df_choice = pd.read_csv(DIR + 'multipleChoiceResponses.csv', low_memory=False, header=[0,1])
schema = pd.read_csv(DIR + 'SurveySchema.csv', low_memory=False, header=[0,1])
# Format Dataframes
df_free.columns = ['_'.join(col) for col in df_free.columns]
df_choice.columns = ['_'.join(col) for col in df_choice.columns]
schema.columns = ['_'.join(col) for col in schema.columns]

# For getting all columns
pd.set_option('display.max_columns', None)

# ### Responders in different countries

# In[ ]:


country_count = df_choice['Q3_In which country do you currently reside?'].value_counts().reset_index()
country_count.columns = ['country', 'people']

# Let's see how many people responded to the survey in different countries.

# In[ ]:


# I use dataset from plotly to get country codes, which are required to plot the data.
country_code = pd.read_csv('../input/plotly-country-code-mapping/2014_world_gdp_with_codes.csv')
country_code.columns = [i.lower() for i in country_code.columns]
country_count.loc[country_count['country'] == 'United States of America', 'country'] = 'United States'
country_count.loc[country_count['country'] == 'United Kingdom of Great Britain and Northern Ireland', 'country'] = 'United Kingdom'
country_count.loc[country_count['country'] == 'South Korea', 'country'] = '"Korea, South"'
country_count.loc[country_count['country'] == 'Viet Nam', 'country'] = 'Vietnam'
country_count.loc[country_count['country'] == 'Iran, Islamic Republic of...', 'country'] = 'Iran'
country_count.loc[country_count['country'] == 'Hong Kong (S.A.R.)', 'country'] = 'Hong Kong'
country_count.loc[country_count['country'] == 'Republic of Korea', 'country'] = '"Korea, North"'
country_count = pd.merge(country_count, country_code, on='country')

# In[ ]:


data = [ dict(
        type = 'choropleth',
        locations = country_count['code'],
        z = country_count['people'],
        text = country_count['country'],
        colorscale = 'Viridis',
        autocolorscale = False,
        reversescale = True,
        marker = dict(
            line = dict (
                color = 'rgb(180,180,180)',
                width = 0.5
            ) ),
        colorbar = dict(
            autotick = False,
            title = 'Responders'),
      ) ]

layout = dict(
    title = 'Responders by country',
    geo = dict(
        showframe = False,
        showcoastlines = False,
        projection = dict(
            type = 'Mercator'
        )
    )
)

fig = dict( data=data, layout=layout )
iplot( fig, validate=False, filename='d3-world-map' )

# As expected most of the responders are from USA, India and China.
# 
# USA is the country where Data Science is the most developed, India and China have a lot of aspiring DS.
# 
# My country (Russia) has the fourth place, Kaggle is becoming more and more popular there.

# #### Dividing responders in groups by countries
# 
# I have decided to compare countries based on the information of responders living there. While all countries are unique, it would be difficult to analyze each and every country. As a result I have decided to take USA and India separately, because they have the largest number of responders, Russia, because I live there and other countries will be grouped in one category.

# In[ ]:


df_choice['Q3_orig'] = df_choice['Q3_In which country do you currently reside?']
df_choice.loc[df_choice['Q3_In which country do you currently reside?'].isin(['United States of America', 'Russia', 'India']) == False,
              'Q3_In which country do you currently reside?'] = 'Other countries'

# ### How long did it take to answer the survey

# In[ ]:


df_choice['Time from Start to Finish (seconds)_Duration (in minutes)'] = df_choice['Time from Start to Finish (seconds)_Duration (in seconds)'] / 60

# In[ ]:


plt.hist(df_choice['Time from Start to Finish (seconds)_Duration (in minutes)'] / 60, bins=40);
plt.yscale('log');
plt.title('Distribution of time spent on the survey');
plt.xlabel('Time (hours)');

# Well... it seems that a lot of people closed the survey almost immediately after opening it, thus giving little info. And some people spent days on the survey! I suppose they opened the tab and forgot about it. Let's have a look at the surveys which took less than 3 hours.

# In[ ]:


data = []
for i in df_choice['Q3_In which country do you currently reside?'].unique():
    trace = {
            "type": 'violin',
            "x": df_choice.loc[(df_choice['Q3_In which country do you currently reside?'] == i) & (df_choice['Time from Start to Finish (seconds)_Duration (in minutes)'] < 180),
                               'Q3_In which country do you currently reside?'],
            "y": df_choice.loc[(df_choice['Q3_In which country do you currently reside?'] == i) & (df_choice['Time from Start to Finish (seconds)_Duration (in minutes)'] < 180),
                               'Time from Start to Finish (seconds)_Duration (in minutes)'],
            "name": i,
            "meanline": {
                "visible": True
            }
        }
    data.append(trace)

        
fig = {
    "data": data,
    "layout" : {
        "title": "",
        "yaxis": {
            "zeroline": False,
        }
    }
}

fig['layout'].update(title='Distribution of time spent on test by country');
iplot(fig)

# In[ ]:


data = []
for j, c in enumerate(df_choice['Q3_In which country do you currently reside?'].unique()):
    df_small = df_choice.loc[(df_choice['Q3_In which country do you currently reside?'] == c) & (df_choice['Time from Start to Finish (seconds)_Duration (in minutes)'] < 60),
                            'Time from Start to Finish (seconds)_Duration (in minutes)']
    trace = go.Histogram(
        x=df_small,
        name=c,
        marker=dict(color=j, opacity=0.5),
        showlegend=False
    )  
    data.append(trace)
fig = go.Figure(data=data)
fig['layout'].update(height=400, width=800, barmode='overlay', title='Distribution of time spent on test by country');
iplot(fig);

# For all countries situation is similar: those who seriously took the survey, spent ~15-30 minutes on it.

# ### Gender and age

# In[ ]:


data = []
for i in df_choice['Q1_What is your gender? - Selected Choice'].unique():
    trace = go.Bar(
        x=df_choice.loc[df_choice['Q1_What is your gender? - Selected Choice'] == i, 'Q2_What is your age (# years)?'].value_counts().sort_index().index,
        y=df_choice.loc[df_choice['Q1_What is your gender? - Selected Choice'] == i, 'Q2_What is your age (# years)?'].value_counts().sort_index().values,
        name=i
    )
    data.append(trace)
layout = go.Layout(
    barmode='stack'
)
fig = go.Figure(data=data, layout=layout)
iplot(fig, filename='stacked-bar')

# In[ ]:


s = pd.crosstab(df_choice['Q1_What is your gender? - Selected Choice'],
                df_choice['Q2_What is your age (# years)?'], normalize='index').style.background_gradient(cmap='viridis', low=.5, high=0).highlight_null('red')
s

# In[ ]:


def plot_country_two_vars_dropdown(var1='', var2='', title_name=''):
    df = df_choice.copy()
    df[var2] = df[var2].astype('category')
    df[var1] = df[var1].astype('category')

    data = []
    buttons = []
    n_mult = df[var1].nunique()
    n = df['Q3_orig'].nunique() * n_mult
    for j, c in enumerate(df['Q3_orig'].unique()):
        visibility = [False] * n
        for ind, i in enumerate(df[var1].unique()):
            grouped = df.loc[(df[var1] == i) & (df['Q3_orig'] == c),
                                var2].value_counts().sort_index()
            trace = go.Bar(
                x=grouped.index,
                y=grouped.values,
                name=i,

                showlegend=True if j == 0 else False,
                legendgroup=i,
                visible=True if j == 0 else False
            )

            data.append(trace)
        visibility[j*n_mult:j*n_mult + n_mult] = [True] * n_mult
        buttons.append(dict(label = c,
                            method = 'update',
                            args = [{'visible': visibility},
                                    {'title': f'Responders in {c} by {title_name}'}]))
            
    updatemenus = list([dict(active=-1, buttons=buttons, x=1, y=2)])
    layout = dict(height=400, width=800, title=f"Responders in {df['Q3_orig'].unique()[0]} by {title_name}", updatemenus=updatemenus)
    fig = dict(data=data, layout=layout)
    return fig

# In[ ]:


def plot_country_two_vars(var1='', var2='', title_name=''):
    colors = cl.scales[str(df_choice[var1].fillna('').nunique())]['qual']['Paired']
    fig = tools.make_subplots(rows=4, cols=1, subplot_titles=('United States of America', 'Other countries', 'India', 'Russia'), print_grid=False)
    for j, c in enumerate(df_choice['Q3_In which country do you currently reside?'].unique()):
        data = []
        for ind, i in enumerate(df_choice[var1].unique()):
            grouped = df_choice.loc[(df_choice[var1] == i) & (df_choice['Q3_In which country do you currently reside?'] == c),
                                var2].value_counts().sort_index()
            trace = go.Bar(
                x=grouped.index,
                y=grouped.values,
                name=i,
                marker=dict(color=colors[ind]),
                showlegend=True if j == 0 else False,
                legendgroup=i
            )
            fig.append_trace(trace, j + 1, 1)    

    fig['layout'].update(height=1000, width=800, title=f'Responders in countries by {title_name}');
    return fig
fig = plot_country_two_vars(var1='Q1_What is your gender? - Selected Choice', var2='Q2_What is your age (# years)?', title_name='age and gender')
iplot(fig);

# ### New interactive graphs
# I have decided that showing only 4 countries may be not interesting enough. So I have added interactive graphs to this notebook - you can choose what you see by yourself!
# You can select country for which data will be shown, column for colors and column for x labels. If you change values in widgets, it will take several seconds to update the plot.

# In[ ]:


fig = plot_country_two_vars_dropdown(var1='Q1_What is your gender? - Selected Choice', var2='Q2_What is your age (# years)?', title_name='age and gender')
iplot(fig);

# Well, there are much more males then other genders. This could be due to bias or due to higher interest in this sphere. I won't say whether there is a discrimination or barriers - this isn't the place for this. Let's have a look at other things:
# * In general there are a lot of students or young professionals. I suppose that a lot of young people try to take part in competitions to get experience or medals/prizes, which should boost their career;
# * It is worth noticing that India has a different trend - while in Russia, USA and other countries kagglers are 25-29 years old, in India most of responders are 18-21. I wonder what is the reason...
# * Also it is interesting that the ratio of women to other genders is higher in USA than in other countries. Good news!

# ### Degree

# In[ ]:


s = pd.crosstab(df_choice['Q2_What is your age (# years)?'],
                df_choice['Q4_What is the highest level of formal education that you have attained or plan to attain within the next 2 years?']).style.background_gradient(cmap='viridis', low=.5, high=0).highlight_null('red')
s

# It seems that there are two main clusters of kagglers based on education and age: bachelors of 18-29 years and masters of 22-34 years.

# In[ ]:


fig = plot_country_two_vars(var1='Q1_What is your gender? - Selected Choice',
                            var2='Q4_What is the highest level of formal education that you have attained or plan to attain within the next 2 years?',
                            title_name='degree')
iplot(fig);

# In[ ]:


fig = plot_country_two_vars_dropdown(var1='Q1_What is your gender? - Selected Choice',
                                     var2='Q4_What is the highest level of formal education that you have attained or plan to attain within the next 2 years?',
                                     title_name='degree')
iplot(fig);

# In[ ]:


s = pd.crosstab(df_choice['Q1_What is your gender? - Selected Choice'],
                df_choice['Q4_What is the highest level of formal education that you have attained or plan to attain within the next 2 years?'], normalize='index').style.background_gradient(cmap='viridis', low=.05, high=0).highlight_null('red')
s

# It isn't surprising that kagglers usually have (or plan to get) higher education degree. Master degree is the most common one (though in India Bachelor degree is more wide-spread).
# 
# It is quite interesting that the rate of having a higher degree (master and doctoral) is higher for women than for men.

# In[ ]:


def plot_country_one_var_dropdown(var='', title_name=''):
    df_choice[var] = df_choice[var].astype('category')
    data = []
    buttons = []
    n = df_choice['Q3_orig'].nunique()
    for j, c in enumerate(df_choice['Q3_orig'].unique()):
        visibility = [False] * n
        grouped = df_choice.loc[df_choice['Q3_orig'] == c,
                                var].value_counts().sort_index()
        grouped = grouped / grouped.sum()
        if var == 'Q9_What is your current yearly compensation (approximate $USD)?':
            map_dict = {'0-10,000': 0,
                        '10-20,000': 1,
                        '100-125,000': 10,
                        '125-150,000': 11,
                        '150-200,000': 12,
                        '20-30,000': 2,
                        '200-250,000': 13,
                        '250-300,000': 14,
                        '30-40,000': 3,
                        '300-400,000': 15,
                        '40-50,000': 4,
                        '400-500,000': 16,
                        '50-60,000': 5,
                        '60-70,000': 6,
                        '70-80,000': 7,
                        '80-90,000': 8,
                        '90-100,000': 9,
                        '500,000+': 17,
                        'I do not wish to disclose my approximate yearly compensation': 18}
            grouped = grouped.reset_index()
            grouped.columns = ['salary', 'counts']
            grouped['sorting'] = grouped['salary'].apply(lambda x: map_dict[x])
            grouped = grouped.sort_values('sorting', ascending=True)
            trace = go.Bar(
                x=grouped['salary'],
                y=grouped['counts'],
                name=c,
                marker=dict(color=j),
                showlegend=True,
                visible=True if j == 0 else False
            )
            data.append(trace)

        elif var == 'Q24_How long have you been writing code to analyze data?':
            map_dict = {'I have never written code but I want to learn': 8,
                        '5-10 years': 3,
                        '3-5 years': 2,
                        '< 1 year': 0,
                        '1-2 years': 1,
                        '10-20 years': 4,
                        '20-30 years': 5,
                        '30-40 years': 6,
                        'I have never written code and I do not want to learn': 9,
                        '40+ years': 7}
            grouped = grouped.reset_index()
            grouped.columns = ['salary', 'counts']
            grouped['sorting'] = grouped['salary'].apply(lambda x: map_dict[x])
            grouped = grouped.sort_values('sorting', ascending=True)
            trace = go.Bar(
                x=grouped['salary'],
                y=grouped['counts'],
                name=c,
                marker=dict(color=j),
                showlegend=True,
                visible=True if j == 0 else False
            )
            data.append(trace)
        else:
            trace = go.Bar(
                x=grouped.index,
                y=grouped.values,
                name=c,
                marker=dict(color=j),
                showlegend=True,
                visible=True if j == 0 else False
            )
            data.append(trace)

        visibility[j*1:j*1 + 1] = [True]
        buttons.append(dict(label = c,
                            method = 'update',
                            args = [{'visible': visibility},
                                    {'title': f'Responders in {c} by {title_name}'}]))
            
            
    updatemenus = list([dict(active=-1, buttons=buttons, x=1, y=2)])
    layout = dict(height=400, width=800, title=f"Responders in {df_choice['Q3_orig'].unique()[0]} by {title_name}", updatemenus=updatemenus)
    fig = dict(data=data, layout=layout)
    return fig

# ### Major

# In[ ]:


def plot_country_one_var(var='', title_name=''):
    data = []
    for j, c in enumerate(df_choice['Q3_In which country do you currently reside?'].unique()):
        grouped = df_choice.loc[df_choice['Q3_In which country do you currently reside?'] == c,
                                var].value_counts().sort_index()
        grouped = grouped / grouped.sum()
        if var == 'Q9_What is your current yearly compensation (approximate $USD)?':
            map_dict = {'0-10,000': 0,
                        '10-20,000': 1,
                        '100-125,000': 10,
                        '125-150,000': 11,
                        '150-200,000': 12,
                        '20-30,000': 2,
                        '200-250,000': 13,
                        '250-300,000': 14,
                        '30-40,000': 3,
                        '300-400,000': 15,
                        '40-50,000': 4,
                        '400-500,000': 16,
                        '50-60,000': 5,
                        '60-70,000': 6,
                        '70-80,000': 7,
                        '80-90,000': 8,
                        '90-100,000': 9,
                        '500,000+': 17,
                        'I do not wish to disclose my approximate yearly compensation': 18}
            grouped = grouped.reset_index()
            grouped.columns = ['salary', 'counts']
            grouped['sorting'] = grouped['salary'].apply(lambda x: map_dict[x])
            grouped = grouped.sort_values('sorting', ascending=True)
            trace = go.Bar(
                x=grouped['salary'],
                y=grouped['counts'],
                name=c,
                marker=dict(color=j),
                showlegend=True
            )
            data.append(trace)
        elif var == 'Q24_How long have you been writing code to analyze data?':
            map_dict = {'I have never written code but I want to learn': 8,
                        '5-10 years': 3,
                        '3-5 years': 2,
                        '< 1 year': 0,
                        '1-2 years': 1,
                        '10-20 years': 4,
                        '20-30 years': 5,
                        '30-40 years': 6,
                        'I have never written code and I do not want to learn': 9,
                        '40+ years': 7}
            grouped = grouped.reset_index()
            grouped.columns = ['salary', 'counts']
            grouped['sorting'] = grouped['salary'].apply(lambda x: map_dict[x])
            grouped = grouped.sort_values('sorting', ascending=True)
            trace = go.Bar(
                x=grouped['salary'],
                y=grouped['counts'],
                name=c,
                marker=dict(color=j),
                showlegend=True
            )
            data.append(trace)
        else:
            trace = go.Bar(
                x=grouped.index,
                y=grouped.values,
                name=c,
                marker=dict(color=j),
                showlegend=True
            )
            data.append(trace)    
    fig = go.Figure(data=data)
    fig['layout'].update(height=400, width=800, title=f'Responders in countries by {title_name}', barmode='group');
    return fig

fig = plot_country_one_var(var='Q5_Which best describes your undergraduate major? - Selected Choice', title_name='major')
iplot(fig);

# In[ ]:


fig = plot_country_one_var_dropdown(var='Q5_Which best describes your undergraduate major? - Selected Choice', title_name='major')
iplot(fig);

# In[ ]:


s = pd.crosstab(df_choice['Q1_What is your gender? - Selected Choice'],
                df_choice['Q5_Which best describes your undergraduate major? - Selected Choice'], normalize='index').style.background_gradient(cmap='viridis', low=.01, high=0).highlight_null('red')
s

# People come from various backgrounds, though most come from math and CS, which isn't surprising. On the other hand, we can say, that those who come from non-relevant background could have higher interest and drive in growing as a DS.
# 
# In is quite interesting that in India most kagglers have CS or engineering degree and business disciplines as well as maths aren't as popular as in other countries.

# ### Industry and profession

# In[ ]:


countsDf = df_choice['Q6_Select the title most similar to your current role (or most recent title if retired): - Selected Choice'].value_counts()
trace1 = go.Bar(
                x = countsDf.index,
                y = countsDf.values,
                name = "Kaggle",
                marker = dict(color = 'gold'),
                text = countsDf.index)
data = [trace1]
layout = go.Layout(barmode = "group",title='Title', yaxis= dict(title='Counts'),showlegend=False)
fig = go.Figure(data = data, layout = layout)
iplot(fig)

# I know that many people has shown this graph, but still let's look at it again. I think that most of these titles can be joined into several groups:
# - Students can be a separate group,
# - Let's leave DS also by themselves;
# - People in research;
# - Next we have analysts - DA, BA and others, who need a different set of skills, but could be considered a level before DS;
# - DE and SE who build production systems;
# - Managers to lead the products;
# - And others;
# 
# This grouping is arbitrate and could be wrong, but let's see what will be the result.

# In[ ]:


countsDf = countsDf.reset_index()
countsDf.columns = ['title', 'number']
countsDf.loc[countsDf['title'].isin(['Consultant', 'Business Analyst', 'Marketing Analyst']), 'title'] = 'Data Analyst'
countsDf.loc[countsDf['title'].isin(['Software Engineer', 'DBA/Database Engineer', 'Developer Advocate']), 'title'] = 'Data Engineer'
countsDf.loc[countsDf['title'].isin(['Research Assistant', 'Research Scientist', 'Statistician']), 'title'] = 'Research'
countsDf.loc[countsDf['title'].isin(['Product/Project Manager', 'Chief Officer']), 'title'] = 'Manager'
countsDf.loc[countsDf['title'].isin(['Salesperson', 'Principal Investigator', 'Data Journalist', 'Not employed']), 'title'] = 'Other'
countsDf = countsDf.groupby('title')['number'].sum()
trace1 = go.Bar(
                x = countsDf.index,
                y = countsDf.values,
                name = "Kaggle",
                marker = dict(color = 'brown'),
                text = countsDf.values,
                textposition = 'outside')
data = [trace1]
layout = go.Layout(barmode = "group",title='Grouped titles', yaxis= dict(title='Counts'),showlegend=False)
fig = go.Figure(data = data, layout = layout)
iplot(fig)

# Now we see that the number of DE and DS is almost equal and the number of DA isn't far behind.
# It is worth noticing that different companies could have very different titles. For example in Facebook DS could work as DA; in some companies situation could be opposite.

# In[ ]:


s = pd.crosstab(df_choice['Q7_In what industry is your current employer/contract (or your most recent employer if retired)? - Selected Choice'],
                df_choice['Q10_Does your current employer incorporate machine learning methods into their business?']).style.background_gradient(cmap='viridis', low=.1, high=0).highlight_null('red')
s

# I think it is interesting to see which industries have the most developed ML practices. It isn't surprising that IT companies are far ahead.

# In[ ]:


df_choice.loc[df_choice['Q6_Select the title most similar to your current role (or most recent title if retired): - Selected Choice'].isin(['Consultant', 'Business Analyst', 'Marketing Analyst']),
              'Q6_Select the title most similar to your current role (or most recent title if retired): - Selected Choice'] = 'Data Analyst'
df_choice.loc[df_choice['Q6_Select the title most similar to your current role (or most recent title if retired): - Selected Choice'].isin(['Software Engineer', 'DBA/Database Engineer', 'Developer Advocate']),
              'Q6_Select the title most similar to your current role (or most recent title if retired): - Selected Choice'] = 'Data Engineer'
df_choice.loc[df_choice['Q6_Select the title most similar to your current role (or most recent title if retired): - Selected Choice'].isin(['Research Assistant', 'Research Scientist', 'Statistician']),
              'Q6_Select the title most similar to your current role (or most recent title if retired): - Selected Choice'] = 'Research'
df_choice.loc[df_choice['Q6_Select the title most similar to your current role (or most recent title if retired): - Selected Choice'].isin(['Product/Project Manager', 'Chief Officer']),
              'Q6_Select the title most similar to your current role (or most recent title if retired): - Selected Choice'] = 'Manager'
df_choice.loc[df_choice['Q6_Select the title most similar to your current role (or most recent title if retired): - Selected Choice'].isin(['Salesperson', 'Principal Investigator', 'Data Journalist', 'Not employed']),
              'Q6_Select the title most similar to your current role (or most recent title if retired): - Selected Choice'] = 'Other'

# In[ ]:


fig = plot_country_one_var(var='Q6_Select the title most similar to your current role (or most recent title if retired): - Selected Choice', title_name='title')
iplot(fig);

# In[ ]:


fig = plot_country_one_var_dropdown(var='Q6_Select the title most similar to your current role (or most recent title if retired): - Selected Choice',
                                    title_name='title')
iplot(fig);

# We can see real differences between countries:
# * In Russia most of Kagglers are DE or DS, who are most relevant for Kaggle;
# * In USA there are more DA that DE among the Kagglers! I suppose this could be due to the fact that some DS are called DA. Also lot's of students;
# * In India most of Kagglers are students, who aspire to become DS (or DE);
# * Other countries are similar to Russia, but have a higher rate of students;

# In[ ]:


fig = plot_country_two_vars(var2='Q6_Select the title most similar to your current role (or most recent title if retired): - Selected Choice',
                            var1='Q8_How many years of experience do you have in your current role?', title_name='title and experience')
iplot(fig);

# In[ ]:


fig = plot_country_two_vars_dropdown(var1='Q8_How many years of experience do you have in your current role?',
                                     var2='Q6_Select the title most similar to your current role (or most recent title if retired): - Selected Choice',
                                     title_name='title and experience')
iplot(fig);

# As expected most of the students have none to little education. And there are a lot of starting DA, DE and DS.
# 
# One interesting thing which I see is that there are a lot of experienced DE in Russia. I think they are programmers who switched career to SE.

# ### Salary

# In[ ]:


fig = plot_country_one_var(var='Q9_What is your current yearly compensation (approximate $USD)?', title_name='salary')
iplot(fig);

# In[ ]:


fig = plot_country_one_var_dropdown(var='Q9_What is your current yearly compensation (approximate $USD)?', title_name='salary')
iplot(fig);

# It is difficult to compare salaries in different countries. 100-200k $ could be a norm for an American DS and a good salary in general, but in Russia 10-20k is a really big salary.

# ### Years of experience

# In[ ]:


fig = plot_country_one_var(var='Q24_How long have you been writing code to analyze data?', title_name='years of coding for data analysis')
iplot(fig);

# Most countries have similar patterns, but America has more experiences DS. This is understandable as DS popularity began there.

# ### Years of learning ML vs self-confidence

# In[ ]:


var1='Q26_Do you consider yourself to be a data scientist?'
var2='Q25_For how many years have you used machine learning methods (at work or in school)?'
title_name='years of learning ML and self-confidence'
colors = cl.scales[str(df_choice[var1].fillna('').nunique())]['qual']['Paired']
fig = tools.make_subplots(rows=4, cols=1, subplot_titles=('United States of America', 'Other countries', 'India', 'Russia'), print_grid=False)
for j, c in enumerate(df_choice['Q3_In which country do you currently reside?'].unique()):
    #data = []
    for ind, i in enumerate(df_choice[var1].unique()):
        grouped = df_choice.loc[(df_choice[var1] == i) & (df_choice['Q3_In which country do you currently reside?'] == c),
                            var2].value_counts().sort_index()
        map_dict = {'I have never studied machine learning but plan to learn in the future': 8,
                    '< 1 year': 0,
                    '4-5 years': 4,
                    '2-3 years': 2,
                    '1-2 years': 1,
                    '5-10 years': 5,
                    '3-4 years': 3,
                    'I have never studied machine learning and I do not plan to': 9,
                    '20+ years': 7,
                    '10-15 years': 6}
        grouped = grouped.reset_index()
        grouped.columns = ['years', 'counts']
        #print(grouped.shape)
        #break
        grouped['sorting'] = grouped['years'].apply(lambda x: map_dict[x])
        grouped = grouped.sort_values('sorting', ascending=True)
        trace = go.Bar(
            x=grouped.years,
            y=grouped.counts,
            name=i,
            marker=dict(color=colors[ind]),
            showlegend=True if j == 0 else False,
            legendgroup=i
        )
        fig.append_trace(trace, j + 1, 1)    

fig['layout'].update(height=1000, width=800, title=f'Responders in countries by {title_name}');

iplot(fig);

# Quite interesting. Usually people with 2 or more years of ML-experience are confident that they are DS, but in Russia people with 1-2 or even less that 1 year of experience consider themselves to be DS.

# ## Multiple choice queations

# ### Where do DS get useful information?

# In[ ]:


def plot_choice_var(var='', title_name=''):
    col_names = [col for col in df_choice.columns if f'{var}_Part' in col]
    data = []
    small_df = df_choice[col_names]
    text_values = [col.split('- ')[2] for col in col_names]
    counts = []
    for m, n in zip(col_names, text_values):
        if small_df[m].nunique() == 0:
            counts.append(0)
        else:
            counts.append(sum(small_df[m] == n))
    trace = go.Bar(
        x=text_values,
        y=counts,
        name=c,
        marker=dict(color='silver'),
        showlegend=False
    )
    data.append(trace)    
    fig = go.Figure(data=data)
    fig['layout'].update(height=400, width=800, title=f'Popular {title_name}');
    return fig

def plot_country_multiple_choice_var(var='', title_name=''):
    col_names = [col for col in df_choice.columns if f'{var}_Part' in col]
    #print(col_names)
    data = []
    for j, c in enumerate(df_choice['Q3_In which country do you currently reside?'].unique()):
        small_df = df_choice.loc[df_choice['Q3_In which country do you currently reside?'] == c, col_names]
        text_values = [col.split('- ')[2] for col in col_names]
        counts = []
        for m, n in zip(col_names, text_values):
            if small_df[m].nunique() == 0:
                counts.append(0)
            else:
                counts.append(sum(small_df[m] == n))
        counts = [i / len(small_df) for i in counts]
        trace = go.Bar(
            x=text_values,
            y=counts,
            name=c,
            marker=dict(color=j),
            showlegend=True
        )
        data.append(trace)    
    fig = go.Figure(data=data)
    fig['layout'].update(height=400, width=800, title=f'Popular {title_name} in different countries', barmode='group');
    return fig

def plot_one_text_var(q='', title=''):
    col_name = [col for col in df_free.columns if q in col][0]
    df_ = df_free[col_name].value_counts().head(7)
    trace = go.Pie(labels=df_.index, 
                   values=df_.values
                  )

    data = [trace]
    layout = go.Layout(
        title=title
    )

    fig = go.Figure(data=data, layout=layout)
    return fig

var_name = dict([('Q38', 'resources'), ('Q13', 'IDE'), ('Q14', 'hosted notebooks'), ('Q15', 'cloud computing services'), ('Q16', 'programming languages'),
         ('Q19', 'ML frameworks'), ('Q21', 'data visualization libraries'), ('Q29', 'data bases'), ('Q31', 'types of data used'), ('Q36', 'online platforms'),
         ('Q47', 'tools for interpretation'), ('Q49', 'tools for reproducibility')])
cols2 = ['Q38', 'Q13', 'Q14', 'Q15','Q16', 'Q19', 'Q21', 'Q29', 'Q31', 'Q36', 'Q47', 'Q49']

def plot_country_multiple_choice_var_dropdown(var=''):
    t = var_name[var]
    #print(t)
    col_names = [col for col in df_choice.columns if f'{var}_Part' in col]
    data = []
    buttons = []
    n = df_choice['Q3_orig'].nunique()
    for j, c in enumerate(sorted(df_choice['Q3_orig'].unique(), reverse=True)[::-1]):
        visibility = [False] * 58
        small_df = df_choice.loc[df_choice['Q3_orig'] == c, col_names]
        text_values = [col.split('- ')[2] for col in col_names]
        counts = []
        for m, n in zip(col_names, text_values):
            if small_df[m].nunique() == 0:
                counts.append(0)
            else:
                counts.append(sum(small_df[m] == n))
        orig_counts = counts.copy()
        counts = [i / len(small_df) for i in counts]
        trace = go.Bar(
            x=text_values,
            y=counts,
            name=c,
            marker=dict(color=j),
            showlegend=True,
            visible=True if j == 0 else False,
            text = orig_counts,
            textposition = 'outside'
        )
        data.append(trace)
        visibility[j:j + 1] = [True]
        buttons.append(dict(label = c,
                                method = 'update',
                                args = [{'visible': visibility},
                                        {'title': f"Responders in {c} by {t}"}]))
    updatemenus = list([dict(active=-1, buttons=buttons, x=1, y=2)])
    layout = dict(height=500, width=800, title=f"Responders in {sorted(df_choice['Q3_orig'].unique())[0]} by {t}", updatemenus=updatemenus)
    fig = dict(data=data, layout=layout)

    iplot(fig);

# In[ ]:


fig = plot_choice_var(var='Q38', title_name='resources')
iplot(fig);

# In[ ]:


fig = plot_country_multiple_choice_var(var='Q38', title_name='resources')
iplot(fig);

# In[ ]:


plot_country_multiple_choice_var_dropdown('Q38')

# Quite interesting!
# * Kaggle forums and Medium seem to be universally popular;
# * ArXiv is quite popular, but Russia likes it the most!;
# * Twitter, Knuggets and Reddit are also widely used;
# * In America https://fivethirtyeight.com is very popular, even though it is almost unknown in other countries;
# * And of course Siraj is popular in India with his famous teaching style;

# And now let's see what other resourses people use. I'll show data for resources which were named by at least 10 responders.

# In[ ]:


fig = plot_one_text_var('Q38', title='Other popular resources')
iplot(fig)

# First place is ods.ai - a slack community of Russian DS which is open for everyone wishing to diccuss DS.

# ### Most popular IDE

# In[ ]:


fig = plot_choice_var(var='Q13', title_name='IDE')
iplot(fig);

# In[ ]:


fig = plot_country_multiple_choice_var(var='Q13', title_name='IDE')
iplot(fig);

# In[ ]:


plot_country_multiple_choice_var_dropdown('Q13')

# In[ ]:


fig = plot_one_text_var('Q13', title='Other popular IDE')
iplot(fig)

# * Most popular IDE is Jupyter Notebooks. Well, obviously it is one of the best tools for fast EDA and modelling in Python;
# * RStudio is quite popular in USA - so lots of R-DS there? And it seems that in Russia the interest in R is quite low;
# * In Russia most people prefer to use Pycharm and other countries tend to use Notepad++. Different styles of coding?
# * Interesting that Spyder is quite popular in India;
# * Among other IDE Eclipse and Emacs are most widely used. A lot of people from Java-development?

# ### Hosted notebooks

# In[ ]:


fig = plot_choice_var(var='Q14', title_name='hosted notebooks')
iplot(fig);

# In[ ]:


fig = plot_country_multiple_choice_var(var='Q14', title_name='hosted notebooks')
iplot(fig);

# In[ ]:


plot_country_multiple_choice_var_dropdown('Q14')

# In[ ]:


fig = plot_one_text_var(q='Q14')
iplot(fig)

# Most countries have a similar pattern - high rate of using free resources. And many don't use them at all.
# Among other notebooks Databricks for Spark is used.

# ### cloud computing services

# In[ ]:


fig = plot_choice_var(var='Q15', title_name='cloud computing services')
iplot(fig);

# In[ ]:


fig = plot_country_multiple_choice_var(var='Q15', title_name='cloud computing services')
iplot(fig);

# In[ ]:


plot_country_multiple_choice_var_dropdown('Q15')

# In[ ]:


fig = plot_one_text_var('Q15', title='Other popular cloud computing services')
iplot(fig)


# Quite interesting that while in Russia and India most don't use cloud services, in USA situation is the opposite.
# Additionaly Digital Ocean is used by some people.

# ### programming languages

# In[ ]:


fig = plot_choice_var(var='Q16', title_name='programming languages')
iplot(fig);

# In[ ]:


fig = plot_country_multiple_choice_var(var='Q16', title_name='programming languages')
iplot(fig);

# In[ ]:


plot_country_multiple_choice_var_dropdown('Q16')

# In[ ]:


fig = plot_one_text_var('Q16', title='Other popular programming languages')
iplot(fig)

# Python and SQL are widely used. In Russia R is quite rare. Russia and India use C/C++ a lot and Java is popular in India.

# ### ML Frameworks

# In[ ]:


fig = plot_choice_var(var='Q19', title_name='ML frameworks')
iplot(fig);

# In[ ]:


fig = plot_country_multiple_choice_var(var='Q19', title_name='ML frameworks')
iplot(fig);

# In[ ]:


plot_country_multiple_choice_var_dropdown('Q19')

# In[ ]:


fig = plot_one_text_var('Q19', title='Other popular ML frameworks')
iplot(fig)

# In Russia catboost library from Yandex is quite popular. It is even more used than lightgbm, which for some reason is almost not used in other countries.
# 
# It is interesting that Theano is still used. It was one of the first frameworks, I suppose a lot of people used it (myself included :)).

# ### data visualization libraries

# In[ ]:


fig = plot_choice_var(var='Q21', title_name='data visualization libraries')
iplot(fig);

# In[ ]:


fig = plot_country_multiple_choice_var(var='Q21', title_name='data visualization libraries')
iplot(fig);

# In[ ]:


plot_country_multiple_choice_var_dropdown('Q21')

# In[ ]:


fig = plot_one_text_var('Q21', title='Other popular data visualization libraries')
iplot(fig)

# Nothing unexpected here. R users use ggplot2, Python - matplotlib/seaborn and plotly.
# 
# Tableau is quite a popular tool to visualize data in companies.

# ### data bases

# In[ ]:


fig = plot_choice_var(var='Q29', title_name='data bases')
iplot(fig);

# In[ ]:


fig = plot_country_multiple_choice_var(var='Q29', title_name='data bases')
iplot(fig);

# In[ ]:


plot_country_multiple_choice_var_dropdown('Q29')

# In[ ]:


fig = plot_one_text_var('Q29', title='Other popular data bases')
iplot(fig)

# Postgress is really popular in Russia.
# 
# By the way, I'm surpised that Teradata was not  in the main options.

# ### types of data used

# In[ ]:


fig = plot_choice_var(var='Q31', title_name='types of data used')
iplot(fig);

# In[ ]:


fig = plot_country_multiple_choice_var(var='Q31', title_name='types of data used')
iplot(fig);

# In[ ]:


plot_country_multiple_choice_var_dropdown('Q31')

# In[ ]:


fig = plot_one_text_var('Q31', title='Other popular types of data used')
iplot(fig)

# It is interestig that images are relatively less used in America, than in other countries.

# ### online platforms

# In[ ]:


fig = plot_choice_var(var='Q36', title_name='online platforms')
iplot(fig);

# In[ ]:


fig = plot_country_multiple_choice_var(var='Q36', title_name='online platforms')
iplot(fig);

# In[ ]:


plot_country_multiple_choice_var_dropdown('Q36')

# In[ ]:


fig = plot_one_text_var('Q36', title='Other popular types of online platforms')
iplot(fig)

# It is worth noticing while kagglers in most countries use several platforms, in Russia Coursera is dominating.
# 
# By the way, 46 people mentioned that they use mlcourse.ai. This is a recently created course by Russian community ods.ai and it is great :)

# ### tools for interpretation

# In[ ]:


fig = plot_choice_var(var='Q47', title_name='tools for interpretation')
iplot(fig);

# In[ ]:


fig = plot_country_multiple_choice_var(var='Q47', title_name='tools for interpretation')
iplot(fig);

# In[ ]:


plot_country_multiple_choice_var_dropdown('Q47')

# In Russia people tend examine feature importance and use partial dependency plots more than in other countries. In India plotting decision boundaries in popular. And in USA sensitivity analysis is relatively popular.

# ### tools for reproducability

# In[ ]:


fig = plot_choice_var(var='Q49', title_name='tools for reproducibility')
iplot(fig);

# In[ ]:


fig = plot_country_multiple_choice_var(var='Q49', title_name='tools for reproducibility')
iplot(fig);

# In[ ]:


plot_country_multiple_choice_var_dropdown('Q49')
