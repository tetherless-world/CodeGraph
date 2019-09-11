#!/usr/bin/env python
# coding: utf-8

# <img src='https://cdn.sstatic.net/insights/Img/Survey/2018/FacebookCard.png?v=c9eebbfb73c7'>

# <a href='#'>Preface</a><br>
# <a href='#intro'>Introduction</a><br>
# <a href='#about'>What exactly is this notebook about</a><br>
# <a href='#load'>Loading Libraries</a><br>
# <a href='#import'>Import data</a><br>
# <a href='#start'>Lets start analysing the data</a><br>
# 
# <a href='#0'>0 From which part of the world developers took the survey</a><br>
# 
# <a href='#1'>1 Hobby</a><br>
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <a href='#1.1'>1.1 How many respondants code as hobby</a><br>
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href='#1.2'>1.2 Hobbies based on gender</a><br>
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href='#1.3'>1.3 Countries where coding is a hobby</a><br>
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href='#1.4'>1.4 Does hobby remains as age grows?</a><br>
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href='#1.5'>1.5 People who code as hobby also open source their code?</a><br>
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href='#1.6'>1.6 Summary</a><br>
# 
# <a href='#2'>Open Source</a><br>
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href='#2.1'>2.1 How many respondants contribute to the socitey</a><br>
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href='#2.2'>2.2 Open sorce based on gender</a><br>
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href='#2.3'>2.3 Which country open source alot</a><br>
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href='#2.4'>2.4 Does people open source as age grows?</a><br>
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href='#2.5'>2.5 Which programing language does open source people use?</a><br>
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href='#2.6'>2.6 People who dont code as hobby as well as are not open sourcing</a><br>
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href='#2.7'>2.7 Summary</a><br>
# 
# <a href='#3'>Analysis on Languages</a><br>
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href='#3.1'>3.1 Top 10 languages worked in the year 2018</a><br>
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href='#3.2'>3.2 Top 10 desire language to work in the year 2019</a><br>
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href='#3.3'>3.3 2018 v/s 2019 ratio language</a><br>
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href='#3.4'>3.4 Language used as age grows</a><br>
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href='#3.5'>3.5 Language used for the type of companies</a><br>
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href='#3.6'>3.6 Summary</a><br>
# 
# 
# <a href='#4'>Analysis on Databases</a><br>
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href='#4.1'>4.1 Top 10 Databases worked in the year 2018</a><br>
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href='#4.2'>4.2 Top 10 desire Databases to work in the year 2019</a><br>
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href='#4.3'>4.3 2018 v/s 2019 ratio Databases</a><br>
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href='#4.4'>4.4 Databases used as age grows</a><br>
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href='#4.5'>4.5 Databases used for the type of companies</a><br>
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href='#4.6'>4.6 Summary</a><br>
# 
# <a href='#5'>5 Analysis on Platforms</a><br>
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href='#5.1'>5.1 Top 10 Platforms worked in the year 2018</a><br>
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href='#5.2'>5.2 Top 10 desire Platforms to work in the year 2019</a><br>
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href='#5.3'>5.3 2018 v/s 2019 ratio Platforms</a><br>
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href='#5.4'>5.4 Platforms used as age grows</a><br>
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href='#5.5'>5.5 Platforms used for the type of companies</a><br>
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href='#5.6'>5.6 Summary</a><br>
# 
# <a href='#6'>6 About year 2018</a><br>
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href='#6.1'>6.1 Top frameworks worked in the year 2018</a><br>
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href='#6.2'>6.2 Top operating system used by developers</a><br>
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href='#6.3'>6.3 Top methodology for project management</a><br>
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href='#6.4'>6.4 Top version control used across overseas</a><br>
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href='#6.5'>6.5 Summary</a><br>
# 
# 
# 
# <a href='#7'>7 About Artificial Inteligence</a><br>
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href='#7.1'>7.1 Is AI dangerous?</a><br>
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href='#7.2'>7.2 why AI is intresting?</a><br>
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href='#7.3'>7.3 who is responsible for AI?</a><br>
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href='#7.4'>7.4 Future of AI</a><br>
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href='#7.5'>7.5 Summary</a><br>
# 
# <a href='#8'>8 About Developers</a><br>
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href='#8.1'>8.1 How many people use more monitors?</a><br>
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href='#8.2'>8.2 Where do you see yourself in 5 years?</a><br>
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href='#8.3'>8.3 At what time developers wake up?</a><br>
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href='#8.4'>8.4 Career satisfaction</a><br>
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href='#8.5'>8.5 How often developers exercise?</a><br>
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href='#8.6'>8.6 How productive are developers?</a><br>
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href='#8.7'>8.7 Time taken after bootcamp to become developer</a><br>
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href='#8.8'>8.8 Do you feel a sense of kinship or connection to other developers?</a><br>
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href='#8.9'>8.9 Do you think of yourself as competing with my peers?</a><br>
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href='#8.10'>8.10 Do you think you're not as good at programming as most of my peers?</a><br>
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href='#8.11'>8.11 Favourite IDE's by Developers</a><br>
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href='#8.12'>8.12 How many times does developers checkin code?</a><br>
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href='#8.13'>8.13 Different types of role developers have</a><br>
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href='#8.14'>8.14 Highest Degree's- done by developers</a><br>
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href='#8.15'>8.15 Are coders happy to code ?</a><br>
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href='#8.16'>8.16 Main field of study of developers</a><br>
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href='#8.17'>8.17 Are developer pursuing any degree?</a><br>
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href='#8.18'>8.18 Summary</a><br>
# 
# <a href='#9'>9 About Companies</a><br>
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href='#9.1'>9.1 Company size</a><br>
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href='#9.2'>9.2 Top communication tool</a><br>
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href='#9.3'>9.3 Which type of compaines have quick productive?</a><br>
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href='#9.4'>9.4 As company size grows developers are satisfied?</a><br>
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href='#9.5'>9.5 As company size grows developers code as hobby or loose interest?</a><br>
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href='#9.6'>9.6 Summary</a><br>
# 
# <a href='#10'>10 Gender based Analysis</a><br>
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href='#10.1'>10.1 Top countries with female employees</a><br>
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href='#10.2'>10.2 Top country with less female employees</a><br>
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href='#10.3'>10.3 Summary</a><br>
# 
# 

# <hr >

# # Update

# Hello kagglers, Thanks a ton for the feedback and your valuable inputs.<br>
# Many of you had liked the pervious version of kernel and gave valuable feedback.<br><br>
# <b>Improvements made on your feeback</b><br>
# &nbsp;&nbsp;&nbsp;&nbsp;1.) Optimise Code (I have done to an extent please let me know if I can drill down further).<br>
# &nbsp;&nbsp;&nbsp;&nbsp;2.) Try different kind of chart (Took me a little time to study understand different kind of charts and have implemented them please share what you think about them).<br>
# &nbsp;&nbsp;&nbsp;&nbsp;3.) Try multiple columns to get better understanding of the data set(I have tried them and many more are on the way).<br>
# &nbsp;&nbsp;&nbsp;&nbsp;4.) Typo mistake :-) Its not like I dont know English its like I prefer typing in python more than in English(Managed to stick an entire day on this, do let me know if I can improve more)<br>
# &nbsp;&nbsp;&nbsp;&nbsp;5.) Create an index page for navigate to an appropriate graph(Thanks for the advice).<br>
# &nbsp;&nbsp;&nbsp;&nbsp;6.) Write a short summary of about graphs of what you observed while writing the code.<br>
# &nbsp;&nbsp;&nbsp;&nbsp;7.) Dont use pie charts. I am sorry to say but I love pie charts and I have reduced usage of pie charts and used only at the place where its necessary.

# <hr>

# #  <a id='intro'>Introduction</a>

# <text>Stack Overflow is a privately held website, the flagship site of the Stack Exchange Network, created in 2008 by Jeff Atwood and Joel Spolsky. It was created to be a more open alternative to earlier question and answer sites such as Experts-Exchange. The name for the website was chosen by voting in April 2008 by readers of Coding Horror, Atwood's popular programming blog.
# 
# It features questions and answers on a wide range of topics in computer programming.<text>   -   <i>source wikipedia</i>
#     
#     
# Each year, we ask the developer community about everything from their favorite technologies to their job preferences. This year marks the eighth year we’ve published our Annual Developer Survey results—with the largest number of respondents yet. Over 100,000 developers took the 30-minute survey this past January.  -   <i>source stackoverflow</i>

# <hr>

# # <a id='about'>What exactly is this notebook about</a>

# There are 98,855 responses in this public data release. These responses are what we consider “qualified” for analytical purposes based on completion and time spent on the survey and included at least one non-PII question. Approximately 20,000 responses were started but not included here because respondents did not answer enough questions, or only answered questions with personally identifying information. Of the qualified responses, 67,441 completed the entire survey.
# 
# 

# <hr>

# # <a id='load'>Let's start the analysis by loading required libraries</a><br>
# 

# In[ ]:


import numpy as np
import pandas as pd
import plotly.offline as py
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
from plotly import tools
init_notebook_mode(connected=True)  
from plotly.tools import FigureFactory as ff
import pycountry
import random
import squarify
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# <hr>

# # <a id='import'>Import the given datasets</a>

# In[ ]:


df = pd.read_csv('../input/survey_results_public.csv')
schema = pd.read_csv('../input/survey_results_schema.csv')

# <hr>

# ## Before diving into the analysis lets see what are the questions asked in the survey

# <hr>

# In[ ]:


pd.options.display.max_colwidth = 300
schema

# <hr>

# > # Let's see the first five columns

# In[ ]:


df.head()

# <hr>

# <h3> as you can see there are 10,000 rows and 129 columns (approx) </h3>

# <hr>

# In[ ]:


df.columns.values

# <text> these are columns present in the survey </text>

# <hr>

#  # first lets check the data set

# In[ ]:


# Auxilary functions
def remove_coma(val):
    value = val.replace(",","")
    return value

def random_colors(number_of_colors):
    color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
                 for i in range(number_of_colors)]
    return color

def remove_coma(val):
    value = val.replace(",","")
    return value

def get_list(col_name):
    full_list = ";".join(col_name)
    each_word = full_list.split(";")
    each_word = Counter(each_word).most_common()
    return pd.DataFrame(each_word)
def calculate_percent(val):
    return val/ len(df) *100


def simple_graph(dataframe,type_of_graph, top = 0):
    data_frame = df[dataframe].value_counts()
    layout = go.Layout()
    
    if type_of_graph == 'barh':
        top_category = get_list(df[dataframe].dropna())
        if top !=None:
            data = [go.Bar(
                x=top_category[1].head(top),
                y=top_category[0].head(top),
                orientation = 'h',
                marker=dict(color=random_colors(10), line=dict(color='rgb(8,48,107)',width=1.5,)),
                opacity = 0.6
            )]
        else:
            data = [go.Bar(
            x=top_category[1],
            y=top_category[0],
            orientation = 'h',
            marker=dict(color=random_colors(10), line=dict(color='rgb(8,48,107)',width=1.5,)),
            opacity = 0.6
        )]

    elif type_of_graph == 'barv':
        top_category = get_list(df[dataframe].dropna())
        if top !=None:
            data = [go.Bar(
                x=top_category[0].head(top),
                y=top_category[1].head(top),
                marker=dict(color=random_colors(10), line=dict(color='rgb(8,48,107)',width=1.5,)),
                opacity = 0.6
        )]
        else:
            data = [go.Bar(
                x=top_category[0],
                y=top_category[1],
                marker=dict(color=random_colors(10), line=dict(color='rgb(8,48,107)',width=1.5,)),
                opacity = 0.6
            )]      

    elif type_of_graph == 'pie':
        data = [go.Pie(
            labels = data_frame.index,
            values = data_frame.values,
            marker = dict(colors = random_colors(20)),
            textfont = dict(size = 20)
        )]
    
    elif type_of_graph == 'pie_':
        data = [go.Pie(
            labels = data_frame.index,
            values = data_frame.values,
            marker = dict(colors = random_colors(20)),
            textfont = dict(size = 20)
        )]
        layout = go.Layout(legend=dict(orientation="h"), autosize=False,width=700,height=700)
        pass
    
    fig = go.Figure(data = data, layout = layout)
    py.iplot(fig)
    
    
    
def funnel_chart(index, values):
    values = values
    phases = index
    colors = random_colors(10)
    n_phase = len(phases)
    plot_width = 400
    section_h = 100
    section_d = 10
    unit_width = plot_width / max(values)
    phase_w = [int(value * unit_width) for value in values]
    height = section_h * n_phase + section_d * (n_phase - 1)
    shapes = []
    label_y = []
    for i in range(n_phase):
            if (i == n_phase-1):
                    points = [phase_w[i] / 2, height, phase_w[i] / 2, height - section_h]
            else:
                    points = [phase_w[i] / 2, height, phase_w[i+1] / 2, height - section_h]

            path = 'M {0} {1} L {2} {3} L -{2} {3} L -{0} {1} Z'.format(*points)

            shape = {'type': 'path','path': path,'fillcolor': colors[i],
                    'line': {
                        'width': 1,
                        'color': colors[i]
                    }
            }
            shapes.append(shape)
            label_y.append(height - (section_h / 2))
            height = height - (section_h + section_d)
    label_trace = go.Scatter(
        x=[-350]*n_phase,
        y=label_y,
        mode='text',
        text=phases,
        textfont=dict(
            color='rgb(200,200,200)',
            size=15
        )
    )

    value_trace = go.Scatter(
        x=[350]*n_phase,
        y=label_y,
        mode='text',
        text=values,
        textfont=dict(
            color='rgb(200,200,200)',
            size=15
        )
    )

    data = [label_trace, value_trace]
    layout = go.Layout(title="<b>Funnel Chart</b>",titlefont=dict(size=20,color='rgb(203,203,203)'),
        shapes=shapes,
        height=560,
        width=800,
        showlegend=False,
        paper_bgcolor='rgba(44,58,71,1)',
        plot_bgcolor='rgba(44,58,71,1)',
        xaxis=dict(
            showticklabels=False,
            zeroline=False,
        ),
        yaxis=dict(
            showticklabels=False,
            zeroline=False
        )
    )
    fig = go.Figure(data=data, layout=layout)
    py.iplot(fig)

def bubble_chart(col):
    data = get_list(df[col].dropna())
    data = data[:10]
    data = data.reindex(index=data.index[::-1])

    size = np.array(data[1]*0.001)
    size
    trace0 = go.Scatter(
        x=data[0],
        y=data[1],
        mode='markers',
        marker=dict(color = random_colors(10),size= size)
    )

    data = [trace0]
    py.iplot(data)

# In[ ]:


percent = df.isnull().sum()/ len(df) *100
percent.sort_values(ascending = False)

# ## as you can see many coloums have null values (almost all the columns)

# <hr>

# # <a id='start'>Lets start analysing the data</a><br>

# <a id='0'>0 From which part of the world developers taken the survey</a><br>

# In[ ]:


countries = df['Country'].value_counts()

countries = countries.to_frame().reset_index()
countries.loc[2]['code'] = 'test'
for i,country in enumerate(countries['index']):
    user_input = country
    mapping = {country.name: country.alpha_3 for country in pycountry.countries}
    countries.set_value(i, 'code', mapping.get(user_input))
data = [ dict(
        type = 'choropleth',
        locations = countries['code'],
        z = countries['Country'],
        text = countries['index'],
        colorscale = [[0,"rgb(5, 10, 172)"],[0.35,"rgb(40, 60, 190)"],[0.5,"rgb(70, 100, 245)"],\
            [0.6,"rgb(90, 120, 245)"],[0.7,"rgb(106, 137, 247)"],[1,"rgb(220, 220, 220)"]],
        autocolorscale = False,
        reversescale = True,
        marker = dict(
            line = dict (
                color = 'rgb(180,180,180)',
                width = 0.5
            ) ),
        colorbar = dict(
            autotick = False,
            tickprefix = '',
            title = 'Total Count'),
      ) ]

layout = dict(
    title = 'countries which responded to the survey',
    geo = dict(
        showframe = False,
        showcoastlines = False,
        projection = dict(
            type = 'Mercator'
        )
    )
)

fig = dict( data=data, layout=layout )
py.iplot( fig, validate=False)

# <hr>

# # <a id='1'>1 Hobby</a><br>

# ### <a id='1.1'>1.1 How many respondants code as hobby</a>

# In[ ]:


simple_graph('Hobby','pie')

# <hr>

# <hr>

# ### <a id='1.2'>1.2 Hobbies based on gender</a><br>

# In[ ]:


data = df[['Hobby','Gender']].dropna()

trace1 = go.Bar(
    x=['Female', 'Male'],
    y=[data[(data['Gender'] == 'Female') & (data['Hobby'] == 'Yes')].count()[0], data[(data['Gender'] == 'Male') & (data['Hobby'] == 'Yes')].count()[0]],
    name='Yes',
    opacity=0.6
)
trace2 = go.Bar(
    x=['Female', 'Male'],
    y=[data[(data['Gender'] == 'Female') & (data['Hobby'] == 'No')].count()[0], data[(data['Gender'] == 'Male') & (data['Hobby'] == 'No')].count()[0]],
    name='No',
    opacity=0.6
)

data = [trace1, trace2]
layout = go.Layout(
    barmode='group'
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)

# <hr>

# ### <a id='1.3'>1.3 Countries where coding is a hobby</a><br>

# In[ ]:


data = df[ (df['Hobby'] == 'Yes')]
country = data["Country"].dropna()

for i in country.unique():
    if country[country == i].count() < 600:
        country[country == i] = 'Others'
x = 0
y = 0
width = 50
height = 50
type_list = country.value_counts().index
values = country.value_counts().values

normed = squarify.normalize_sizes(values, width, height)
rects = squarify.squarify(normed, x, y, width, height)

color_brewer = random_colors(20)
shapes = []
annotations = []
counter = 0

for r in rects:
    shapes.append( 
        dict(
            type = 'rect', 
            x0 = r['x'], 
            y0 = r['y'], 
            x1 = r['x']+r['dx'], 
            y1 = r['y']+r['dy'],
            line = dict( width = 1 ),
            fillcolor = color_brewer[counter]
        ) 
    )
    annotations.append(
        dict(
            x = r['x']+(r['dx']/2),
            y = r['y']+(r['dy']/2),
            text = "{}".format(type_list[counter]),
            showarrow = False
        )
    )
    counter = counter + 1
    if counter >= len(color_brewer):
        counter = 0

trace0 = go.Scatter(
    x = [ r['x']+(r['dx']/2) for r in rects ], 
    y = [ r['y']+(r['dy']/2) for r in rects ],
    text = [ str(v) for v in values ], 
    mode = 'text',
)

layout = dict(
    height=600, 
    width=850,
    xaxis=dict(showgrid=False,zeroline=False),
    yaxis=dict(showgrid=False,zeroline=False),
    shapes=shapes,
    annotations=annotations,
    hovermode='closest',
    font=dict(color="#FFFFFF"),
    margin = go.Margin(
            l=0,
            r=0,
            pad=0
        )
)

figure = dict(data=[trace0], layout=layout)
iplot(figure)

# <hr>

# ### <a id='1.4'>1.4 Does hobby remains as age grows?</a><br>

# In[ ]:


data = df[['Hobby','Age']]
age = data['Age'].dropna().unique()

label = np.concatenate((np.array(data['Age'].dropna().unique()), np.array(['Yes','No'])), axis = 0)
data = dict(
    type='sankey',
    node = dict(
      pad = 15,
      thickness = 20,
      line = dict(
        color = random_colors(1),
        width = 0.5
      ),
      label = label,
      color = random_colors(20)
    ),
    link = dict(
      source = [0,1,2,3,4,5,6,0,1,2,3,4,5,6],
      target = [7,7,7,7,7,7,7,8,8,8,8,8,8,8],
      value = [len(data[(data['Age'] =='25 - 34 years old') & (data['Hobby'] == 'Yes')])
               ,len(data[(data['Age'] =='35 - 44 years old') & (data['Hobby'] == 'Yes')])
               ,len(data[(data['Age'] =='18 - 24 years old') & (data['Hobby'] == 'Yes')])
               ,len(data[(data['Age'] =='45 - 54 years old') & (data['Hobby'] == 'Yes')])
               ,len(data[(data['Age'] =='55 - 64 years old') & (data['Hobby'] == 'Yes')])
               ,len(data[(data['Age'] =='Under 18 years old') & (data['Hobby'] == 'Yes')])
               ,len(data[(data['Age'] =='65 years or older') & (data['Hobby'] == 'Yes')])
               ,len(data[(data['Age'] =='25 - 34 years old') & (data['Hobby'] == 'No')])
               ,len(data[(data['Age'] =='35 - 44 years old') & (data['Hobby'] == 'No')])
               ,len(data[(data['Age'] =='18 - 24 years old') & (data['Hobby'] == 'No')])
               ,len(data[(data['Age'] =='45 - 54 years old') & (data['Hobby'] == 'No')])
               ,len(data[(data['Age'] =='55 - 64 years old') & (data['Hobby'] == 'No')])
               ,len(data[(data['Age'] =='Under 18 years old') & (data['Hobby'] == 'No')])
               ,len(data[(data['Age'] =='65 years or older') & (data['Hobby'] == 'No')])
              ]
  ))


fig = dict(data=[data])
py.iplot(fig, validate=False)

# <hr>

# ### <a id='1.5'>1.5 People who code as hobby also opensource their code ?</a>

# In[ ]:


data = df[['Hobby','OpenSource']].dropna()

trace1 = go.Bar(
    x=['Yes', 'No'],
    y=[data[(data['OpenSource'] == 'Yes') & (data['Hobby'] == 'Yes')].count()[0], data[(data['OpenSource'] == 'Yes') & (data['Hobby'] == 'No')].count()[0]],
    name='Yes',
    opacity=0.6
)

data = [trace1]
layout = go.Layout(
    barmode='group'
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)

# <hr>

# ### <a id='1.6'>1.6 Summary</a>

# - 80.8% people code as hobby, 19.2% people dont code as hobby. Its good to know that we are happy coders family.
# - Very few males and females dont enjoy coding( we will look into it deeply as we go on)
# - US and India tops where people enjoy coding.
# - 25-34 is the age where people enjoy coding,I am surprised to see people dont code as hobby at age of 18-24 (2.21K)
#     this is a serious problem and we have to solve, At initial stage of career if they dont enjoy coding then after few years they may get bored. The problem maybe because of peer pressure or for money.
# - over 38K people who enjoy coding also open source

# <hr>

# # <a id='2'>Open Source</a><br>

# ### <a id='2.1'>2.1 How many respondants contribute to the socitey</a>

# In[ ]:


simple_graph('OpenSource','pie')

# <hr>

# ### <a id='2.2'>2.2 open sorce based on gender</a>

# In[ ]:


data = df[['OpenSource','Gender']].dropna()

trace1 = go.Bar(
    x=['Female', 'Male'],
    y=[data[(data['Gender'] == 'Female') & (data['OpenSource'] == 'Yes')].count()[0], data[(data['Gender'] == 'Male') & (data['OpenSource'] == 'Yes')].count()[0]],
    name='Yes',
    opacity=0.6
)
trace2 = go.Bar(
    x=['Female', 'Male'],
    y=[data[(data['Gender'] == 'Female') & (data['OpenSource'] == 'No')].count()[0], data[(data['Gender'] == 'Male') & (data['OpenSource'] == 'No')].count()[0]],
    name='No',
    opacity=0.6
)

data = [trace1, trace2]
layout = go.Layout(
    barmode='group',
    orientation = 'v'
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)

# <hr>

# ### <a id='2.3'>2.3 Which country open source a lot</a>

# In[ ]:


data = df[df['OpenSource'] == 'Yes']
countries = data['Country'].value_counts()

countries = countries.to_frame().reset_index()
countries.loc[2]['code'] = 'test'
for i,country in enumerate(countries['index']):
    user_input = country
    mapping = {country.name: country.alpha_3 for country in pycountry.countries}
    countries.set_value(i, 'code', mapping.get(user_input))
data = [ dict(
        type = 'choropleth',
        locations = countries['code'],
        z = countries['Country'],
        text = countries['index'],
        colorscale = [[0,"rgb(5, 10, 172)"],[0.35,"rgb(40, 60, 190)"],[0.5,"rgb(70, 100, 245)"],\
            [0.6,"rgb(90, 120, 245)"],[0.7,"rgb(106, 137, 247)"],[1,"rgb(220, 220, 220)"]],
        autocolorscale = False,
        reversescale = True,
        marker = dict(
            line = dict (
                color = 'rgb(180,180,180)',
                width = 0.5
            ) ),
        colorbar = dict(
            autotick = False,
            tickprefix = '$',
            title = 'Total Count'),
      ) ]

layout = dict(
    title = 'countries which responded to the survey',
    geo = dict(
        showframe = False,
        showcoastlines = False,
        projection = dict(
            type = 'Mercator'
        )
    )
)

fig = dict( data=data, layout=layout )
py.iplot( fig, validate=False)

# <hr>

# ### <a id='2.4'>2.4 Does people open source as age grows ?</a>

# In[ ]:


data = df[['OpenSource','Age']]
age = data['Age'].dropna().unique()

label = np.concatenate((np.array(data['Age'].dropna().unique()), np.array(['Yes','No'])), axis = 0)
data = dict(
    type='sankey',
    node = dict(
      pad = 15,
      thickness = 20,
      line = dict(
        color = random_colors(1),
        width = 0.5
      ),
      label = label,
      color = random_colors(20)
    ),
    link = dict(
      source = [0,1,2,3,4,5,6,0,1,2,3,4,5,6],
      target = [8,8,8,8,8,8,8,7,7,7,7,7,7,7],
      value = [len(data[(data['Age'] =='25 - 34 years old') & (data['OpenSource'] == 'Yes')])
               ,len(data[(data['Age'] =='35 - 44 years old') & (data['OpenSource'] == 'Yes')])
               ,len(data[(data['Age'] =='18 - 24 years old') & (data['OpenSource'] == 'Yes')])
               ,len(data[(data['Age'] =='45 - 54 years old') & (data['OpenSource'] == 'Yes')])
               ,len(data[(data['Age'] =='55 - 64 years old') & (data['OpenSource'] == 'Yes')])
               ,len(data[(data['Age'] =='Under 18 years old') & (data['OpenSource'] == 'Yes')])
               ,len(data[(data['Age'] =='65 years or older') & (data['OpenSource'] == 'Yes')])
               ,len(data[(data['Age'] =='25 - 34 years old') & (data['OpenSource'] == 'No')])
               ,len(data[(data['Age'] =='35 - 44 years old') & (data['OpenSource'] == 'No')])
               ,len(data[(data['Age'] =='18 - 24 years old') & (data['OpenSource'] == 'No')])
               ,len(data[(data['Age'] =='45 - 54 years old') & (data['OpenSource'] == 'No')])
               ,len(data[(data['Age'] =='55 - 64 years old') & (data['OpenSource'] == 'No')])
               ,len(data[(data['Age'] =='Under 18 years old') & (data['OpenSource'] == 'No')])
               ,len(data[(data['Age'] =='65 years or older') & (data['OpenSource'] == 'No')])
              ]
  ))


fig = dict(data=[data])
py.iplot(fig, validate=False)

# <hr>

# ### <a id='2.5'>2.5 Which programing language does open source people use</a>

# In[ ]:


data = df[df['OpenSource'] == 'Yes']

data = get_list(df['LanguageWorkedWith'].dropna())

x = 0
y = 0
width = 50
height = 50
type_list = data[0]
values = data[1]

normed = squarify.normalize_sizes(values, width, height)
rects = squarify.squarify(normed, x, y, width, height)

color_brewer = random_colors(20)
shapes = []
annotations = []
counter = 0

for r in rects:
    shapes.append( 
        dict(
            type = 'rect', 
            x0 = r['x'], 
            y0 = r['y'], 
            x1 = r['x']+r['dx'], 
            y1 = r['y']+r['dy'],
            line = dict( width = 1 ),
            fillcolor = color_brewer[counter]
        ) 
    )
    annotations.append(
        dict(
            x = r['x']+(r['dx']/2),
            y = r['y']+(r['dy']/2),
            text = "{}".format(type_list[counter]),
            showarrow = False
        )
    )
    counter = counter + 1
    if counter >= len(color_brewer):
        counter = 0

trace0 = go.Scatter(
    x = [ r['x']+(r['dx']/2) for r in rects ], 
    y = [ r['y']+(r['dy']/2) for r in rects ],
    text = [ str(v) for v in values ], 
    mode = 'text',
)

layout = dict(
    height=600, 
    width=850,
    xaxis=dict(showgrid=False,zeroline=False),
    yaxis=dict(showgrid=False,zeroline=False),
    shapes=shapes,
    annotations=annotations,
    hovermode='closest',
    font=dict(color="#FFFFFF"),
    margin = go.Margin(
            l=0,
            r=0,
            pad=0
        )
)

figure = dict(data=[trace0], layout=layout)
iplot(figure)

# <hr>

# ### <a id='2.6'>2.6 People who dont code as hobby as well as are not open sourcing</a>

# In[ ]:


data = df[['Hobby','OpenSource']].dropna()

trace1 = go.Bar(
    x=['Yes', 'No'],
    y=[data[(data['OpenSource'] == 'No') & (data['Hobby'] == 'No')].count()[0]],
    name='Yes',
    opacity=0.6
)

data = [trace1]
layout = go.Layout(
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)

# <hr>

# ### <a id='2.7'>2.7 Summary</a>

# - 56.4% people contribute to opensource projects and 43.6% dont contribute to opensource world.
# - Majority of male contribute to opensource (32K) but most of the females dont contribute to opensource (2K).
# - On the evaluation of country US and India contribute more to open source (Happy to see India in Top 2).
# - From the age of 25-34 most of the people contribute to open source.
# - Most language which get Opensource contribution are HTML, CSS, Javascript (I myself was a Javascript developer before jumping to python ). To all JS lovers cheers.
# - Almost 14K people dont opensource as well as dont code as hobby.

# <hr>

# # <a id='3'>Analysis on Languages</a>

# ### <a id='3.1'>3.1 Top 10 languages worked in the year 2018</a>

# In[ ]:


bubble_chart('LanguageWorkedWith')

# <hr>

# ### <a id='3.2'>3.2 Top 10 desire language to work in the year 2019</a>

# In[ ]:


bubble_chart('LanguageDesireNextYear')

# <hr>

# ### <a id='3.3'>3.3 2018 v/s 2019 ratio language</a>

# In[ ]:


top_languages = get_list(df['LanguageWorkedWith'].dropna())
top_desire_languages = get_list(df['LanguageDesireNextYear'].dropna())
top_languages = top_languages.sort_values(by=[0])
top_desire_languages = top_desire_languages.sort_values(by=[0])

raise_fall_ratio = pd.DataFrame()
raise_fall_ratio['2018'] = top_languages[0]
raise_fall_ratio['2018_percent'] = top_languages[1].apply(calculate_percent)
raise_fall_ratio['2019_percent'] = top_desire_languages[1].apply(calculate_percent)

trace1 = go.Bar(
    x=raise_fall_ratio['2018'],
    y=raise_fall_ratio['2018_percent'],
    name='2018'
)
trace2 = go.Bar(
    x=raise_fall_ratio['2018'],
    y=raise_fall_ratio['2019_percent'],
    name='2019'
)

data = [trace1, trace2]
layout = go.Layout(
    barmode='stack'
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)


# <hr>

# ### <a id='3.4'>3.4 Language used as age grows</a>

# In[ ]:


data = df[['LanguageWorkedWith','Age']].dropna().reset_index()
d = []
for i, val in enumerate(data['LanguageWorkedWith']):
    lang = val.split(';')
    for j in lang:
        temp = np.array([j, data['Age'][i]])
        d.append(temp)
new_df = pd.DataFrame(d, columns=('Lang','Age'))

lang_list = ['C++','C#','PHP','Python','Bash/Shell','Java','SQL','CSS','HTML','JavaScript']
scatter = new_df.groupby(['Lang','Age']).size().reset_index()
scatter = scatter[scatter['Lang'].isin(lang_list)]


data = [go.Scatter3d(
    x=scatter['Lang'],
    y=scatter['Age'],
    z=scatter[0],
    mode='markers',
    marker=dict(line=dict(color=random_colors(1),width=0.5),opacity=0.8)
)]
layout = go.Layout(
    autosize=False,
    width=700,
    height=700,
    title = 'Please click and drag for the graph to begin'
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)

# <hr>

# ### <a id='3.5'>3.5 Language used for the type of companies</a>

# In[ ]:


data = df[['LanguageWorkedWith','CompanySize']].dropna().reset_index()
d = []
for i, val in enumerate(data['LanguageWorkedWith']):
    lang = val.split(';')
    for j in lang:
        temp = np.array([j, data['CompanySize'][i]])
        d.append(temp)
new_df = pd.DataFrame(d, columns=('Lang','CompanySize'))

company_list = ['20 to 99 employees', '10,000 or more employees','100 to 499 employees', '10 to 19 employees',
       '500 to 999 employees', '1,000 to 4,999 employees','5,000 to 9,999 employees', 'Fewer than 10 employees']


scatter = new_df.groupby(['Lang','CompanySize']).size().reset_index()
scatter = scatter[scatter['Lang'].isin(lang_list)]


data = [go.Scatter3d(
    x=scatter['Lang'],
    y=scatter['CompanySize'],
    z=scatter[0],
    mode='markers',
    marker=dict(line=dict(color=random_colors(1),width=0.9),opacity=0.8)
)]
layout = go.Layout(
    autosize=False,
    width=700,
    height=700,
    title = 'Please click and drag for the graph to begin'
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)

# <hr>

# ### <a id='3.6'>3.6 Summary</a>

# - Top languages worked in the year of 2018 are JavaScript,HTML,CSS,SQL,Java,Bash/Shell,Python,C#,PHP,C++.(Damn Javascript always stand on top, reason because of its cross platform and easy to learn and implement).
# - Top languages developer wants to learn in the next year are JavaScript, Python, HTML, CSS, SQL, Java, Bash/Shell, C#, TypeScript, Go.
# - 2018 v/s 2019 graphs shows how the popularity of language is going to be increased example Julia its geting doubled by next year (i.e if 1 developer learnt julia this year then 2 developers are going to learn julia next year).For type script the eager of learning is going down (it will increase because anglar 4 is catching a lot of attention).
# - young blood are leaning towards javascript more as age grows the ratio of each language get decreased (please drag on the empty space of 3.4 graph for the visualization to begin).
# - you can see most of the companies have javascript language(please drag on the empty space of 3.4 graph for the visualization to begin)

# <hr>

# ### <a id='4'>Analysis on Databases</a>

# ### <a id='4.1'>4.1 Top 10 Databases worked in the year 2018</a>

# In[ ]:


bubble_chart('DatabaseWorkedWith')

# <hr>

# ### <a id='4.2'>4.2 Top 10 desire Databases to work in the year 2019</a>

# In[ ]:


bubble_chart('DatabaseDesireNextYear')

# <hr>

# ### <a id='4.3'>4.3 2018 v/s 2019 ratio Databases</a>

# In[ ]:


top_database = get_list(df['DatabaseWorkedWith'].dropna())
top_desire_database = get_list(df['DatabaseDesireNextYear'].dropna())

top_database_ = top_database.sort_values(by=[0])
top_desire_database_ = top_desire_database.sort_values(by=[0])

raise_fall_ratio = pd.DataFrame()
raise_fall_ratio['2018'] = top_database_[0]
raise_fall_ratio['2018_percent'] = top_database_[1].apply(calculate_percent)
raise_fall_ratio['2019_percent'] = top_desire_database_[1].apply(calculate_percent)

trace1 = go.Bar(
    x=raise_fall_ratio['2018'],
    y=raise_fall_ratio['2018_percent'],
    name='2018'
)
trace2 = go.Bar(
    x=raise_fall_ratio['2018'],
    y=raise_fall_ratio['2019_percent'],
    name='2019'
)

data = [trace1, trace2]
layout = go.Layout(
    barmode='stack'
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)

# <hr>

# ### <a id='4.4'>4.4 Databases used as age grows</a>

# In[ ]:


data = df[['DatabaseWorkedWith','Age']].dropna().reset_index()
d = []
for i, val in enumerate(data['DatabaseWorkedWith']):
    lang = val.split(';')
    for j in lang:
        temp = np.array([j, data['Age'][i]])
        d.append(temp)
new_df = pd.DataFrame(d, columns=('Database','Age'))

database_list = ['Microsoft Azure (Tables, CosmosDB, SQL, etc)', 'Oracle','MariaDB', 'Elasticsearch',
       'Redis', 'SQLite','MongoDB', 'MySQL']


scatter = new_df.groupby(['Database','Age']).size().reset_index()
scatter = scatter[scatter['Database'].isin(database_list)]


data = [go.Scatter3d(
    x=scatter['Database'],
    y=scatter['Age'],
    z=scatter[0],
    mode='markers',
    marker=dict(line=dict(color=random_colors(1),width=0.9),opacity=0.8)
)]
layout = go.Layout(
    autosize=False,
    width=700,
    height=700,
    title = 'Please click and drag for the graph to begin'
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)

# <hr>

# ### <a id='4.5'>4.5 Databases used for the type of companies</a>

# In[ ]:


data = df[['DatabaseWorkedWith','CompanySize']].dropna().reset_index()
d = []
for i, val in enumerate(data['DatabaseWorkedWith']):
    lang = val.split(';')
    for j in lang:
        temp = np.array([j, data['CompanySize'][i]])
        d.append(temp)
new_df = pd.DataFrame(d, columns=('Database','CompanySize'))

database_list = ['Microsoft Azure (Tables, CosmosDB, SQL, etc)', 'Oracle','MariaDB', 'Elasticsearch',
       'Redis', 'SQLite','MongoDB', 'MySQL']


scatter = new_df.groupby(['Database','CompanySize']).size().reset_index()
scatter = scatter[scatter['Database'].isin(database_list)]


data = [go.Scatter3d(
    x=scatter['Database'],
    y=scatter['CompanySize'],
    z=scatter[0],
    mode='markers',
    marker=dict(line=dict(color=random_colors(1),width=0.9),opacity=0.8)
)]
layout = go.Layout(
    autosize=False,
    width=700,
    height=700,
    title = 'Please click and drag for the graph to begin'
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)

# <hr>

# ### <a id='4.6'>4.6 Summary</a>

# - Top database used in the year of 2018 are MySQL,SQL Server, PostgreSQL, MongoDB, SQLite, Redis, Elasticsearch, MariaDB, Oracle, Microsoft Azure
# - Top desire database to learn in the year of 2019 are MySQL, MongoDB, PostgreSQL, SQL Server, Redis, Elasticsearch, SQLite, Microsoft Azure, Google Cloud Storage, MariaDB. Love for MySQL never dies.
# - 2018 v/s 2019 graphs shows how the popularity in database is going to be increased example AWS is going to be on fire next year.
# - Majority of the age group love MySQL(please drag on the empty space of 4.4 graph for the visualization to begin).
# - Majority of the company group love MySQL(please drag on the empty space of 4.5 graph for the visualization to begin).

# <hr>

# ## <a id='5'>5 Analysis on Platforms</a>

# ### <a id='5.1'>5.1 Top 10 Platforms worked in the year 2018</a>

# In[ ]:


bubble_chart('PlatformWorkedWith')

# <hr>

# ### <a id='5.2'>5.2 Top 10 desire Platforms to work in the year 2019</a>

# In[ ]:


bubble_chart('PlatformDesireNextYear')

# <hr>

# ### <a id='5.3'>5.3 2018 v/s 2019 ratio Platforms</a>

# In[ ]:


top_platform = get_list(df['PlatformWorkedWith'].dropna())
top_desire_platform = get_list(df['PlatformDesireNextYear'].dropna())

top_platform_ = top_platform.sort_values(by=[0])
top_desire_platform_ = top_desire_platform.sort_values(by=[0])

raise_fall_ratio = pd.DataFrame()
raise_fall_ratio['2018'] = top_platform_[0]
raise_fall_ratio['2018_percent'] = top_platform_[1].apply(calculate_percent)
raise_fall_ratio['2019_percent'] = top_desire_platform_[1].apply(calculate_percent)

trace1 = go.Bar(
    x=raise_fall_ratio['2018'],
    y=raise_fall_ratio['2018_percent'],
    name='2018'
)
trace2 = go.Bar(
    x=raise_fall_ratio['2018'],
    y=raise_fall_ratio['2019_percent'],
    name='2019'
)

data = [trace1, trace2]
layout = go.Layout(
    barmode='stack'
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)


# <hr>

# ### <a id='5.4'>5.4 Platforms used as age grows</a>

# In[ ]:


data = df[['PlatformWorkedWith','Age']].dropna().reset_index()
d = []
for i, val in enumerate(data['PlatformWorkedWith']):
    lang = val.split(';')
    for j in lang:
        temp = np.array([j, data['Age'][i]])
        d.append(temp)
new_df = pd.DataFrame(d, columns=('Platform','Age'))

platform_list = ['Azure', 'Firebase','iOS', 'WordPress',
       'Android', 'Raspberry Pi','Mac OS', 'AWS','Linux']


scatter = new_df.groupby(['Platform','Age']).size().reset_index()
scatter = scatter[scatter['Platform'].isin(platform_list)]


data = [go.Scatter3d(
    x=scatter['Platform'],
    y=scatter['Age'],
    z=scatter[0],
    mode='markers',
    marker=dict(line=dict(color=random_colors(1),width=0.9),opacity=0.8)
)]
layout = go.Layout(
    autosize=False,
    width=700,
    height=700,
    title = 'Please click and drag for the graph to begin'
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)

# <hr>

# ### <a id='5.5'>5.5 Platforms used for the type of companies</a>

# In[ ]:


data = df[['PlatformWorkedWith','CompanySize']].dropna().reset_index()
d = []
for i, val in enumerate(data['PlatformWorkedWith']):
    lang = val.split(';')
    for j in lang:
        temp = np.array([j, data['CompanySize'][i]])
        d.append(temp)
new_df = pd.DataFrame(d, columns=('Platform','CompanySize'))

platform_list = ['Azure', 'Firebase','iOS', 'WordPress',
       'Android', 'Raspberry Pi','Mac OS', 'AWS','Linux']


scatter = new_df.groupby(['Platform','CompanySize']).size().reset_index()
scatter = scatter[scatter['Platform'].isin(platform_list)]


data = [go.Scatter3d(
    x=scatter['Platform'],
    y=scatter['CompanySize'],
    z=scatter[0],
    mode='markers',
    marker=dict(line=dict(color=random_colors(1),width=0.9),opacity=0.8)
)]
layout = go.Layout(
    autosize=False,
    width=700,
    height=700,
    title = 'Please click and drag for the graph to begin'
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)

# <hr>

# ### <a id='5.6'>5.6 Summary</a>

# - Top platform used in the year of 2018 are Linux, Windows Desktop or Server, Android, AWS, Mac OS, Raspberry Pi, WordPress, iOS, Firebase, Azure
# - Top desire platform to learn in the year of 2019 are Linux, Android, AWS, Raspberry Pi, Windows Desktop or Server, iOS, Mac OS, Firebase, Arduino, Google Cloud Platform/App Engine.
# - 2018 v/s 2019 graphs shows how the popularity in platform going to be increased example AWS is going to be on fire next year.
# - Majority of the age group love Linux(please drag on the empty space of 5.4 graph for the visualization to begin).
# - Majority of the company group love Linux(please drag on the empty space of 5.5 graph for the visualization to begin).

# <hr>

# ### <a id='6'>6 About year 2018</a><br>

# ### <a id='6.1'>6.1 Top frameworks worked in the year 2018</a>

# In[ ]:


data = get_list(df['FrameworkWorkedWith'].dropna())
funnel_chart(data[0][:5], data[1][:5])

# <hr>

# ### <a id='6.2'>6.2 top operating system used by developers</a>

# In[ ]:


data = get_list(df['OperatingSystem'].dropna())
funnel_chart(data[0][:5], data[1][:5])

# <hr>

# ### <a id='6.3'>6.3 Top Methodology for project management</a>

# In[ ]:


data = get_list(df['Methodology'].dropna())
funnel_chart(data[0][:5], data[1][:5])

# <hr>

# ### <a id='6.4'>6.4 Top version control used across overseas</a>

# In[ ]:


simple_graph('VersionControl','barh',10)

# <hr>

# ### <a id='6.5'>6.5 Summary</a>

# - Top frameworks are Node, Angular, React three of them belong to javascript.
# - Most of the developers love windows because of its simplicity and not many commands for its operation.
# - Agile, Scrum are the top 2 project management.
# - Git is the most used for version control

# <hr>

# ## <a id='7'>7 About Artificial Inteligence</a><br>

# ### <a id='7.1'>7.1 Is AI dangerous</a>

# In[ ]:


simple_graph('AIDangerous','barv',10)

# <hr>

# ### <a id='7.2'>7.2 why AI is intresting</a>

# In[ ]:


simple_graph('AIInteresting','barv',10)

# <hr>

# ### <a id='7.3'>7.3 who is responsible for AI</a>

# In[ ]:


simple_graph('AIResponsible','barv',10)

# <hr>

# ### <a id='7.4'>7.4 Future of AI</a>

# In[ ]:


simple_graph('AIFuture','barv',10)

# <hr>

# ### <a id='7.5'>7.5 Summary</a>

# - People think AI is dangerous because it makes important decision and it is surpasing human intelligence.
# - And the cons are the pros.

# <hr>

# ### <a id='8'>8 About Developers</a><br>

# ### <a id='8.1'>8.1 How many people use more monitors</a>

# In[ ]:


simple_graph('NumberMonitors','barv',10)

# <hr>

# ### <a id='8.2'>8.2 Where do you see your self in 5 years</a>

# In[ ]:


simple_graph('HopeFiveYears','barv',10)

# <hr>

# ### <a id='8.3'>8.3 What time developers wake up</a>

# In[ ]:


simple_graph('WakeTime','barv',10)

# <hr>

# ### <a id='8.4'>8.4 career satisfaction</a>

# In[ ]:


simple_graph('JobSatisfaction', 'pie')

# <hr>

# ### <a id='8.5'>8.5 How often developers exercise</a>

# In[ ]:


simple_graph('Exercise','barv',10)

# <hr>

# ### <a id='8.6'>8.6 How productive are developers</a>

# In[ ]:


simple_graph('TimeFullyProductive','barv',10)

# <hr>

# ### <a id='8.7'>8.7 Time taken after bootcamp to become developer</a>

# In[ ]:


simple_graph('TimeAfterBootcamp','barv',10)

# <hr>

# ### <a id='8.8'>8.8 Do you feel a sense of kinship or connection to other developers</a>

# In[ ]:


simple_graph('AgreeDisagree1','barv',10)

# <hr>

# ### <a id='8.9'>8.9 Do you think of your self as competing with my peers</a>

# In[ ]:


simple_graph('AgreeDisagree2','barv',10)

# <hr>

# ### <a id='8.10'>8.10 Do you think you're' not as good at programming as most of my peers</a>

# In[ ]:


simple_graph('AgreeDisagree3','barv',10)

# <hr>

# ### <a id='8.11'>8.11 Favourite IDE's by Developers</a>

# In[ ]:


simple_graph('IDE','barh',10)

# <hr>

# ### <a id='8.12'>8.12 How many times does developers checkin code</a>

# In[ ]:


simple_graph('CheckInCode','barv',10)

# <hr>

# ### <a id='8.13'>8.13 Different types of role developers have</a>

# In[ ]:


simple_graph('DevType','barv',10)

# <hr>

# ### <a id='8.14'>8.14 Highest Degree's- done by developers</a>

# In[ ]:


simple_graph('FormalEducation', 'pie_')

# <hr>

# ### <a id='8.15'>8.15 Are coders happy to code ?</a>

# In[ ]:


simple_graph('JobSatisfaction','pie')

# <hr>

# ### <a id='8.16'>8.16 Main field of study of developers</a>

# In[ ]:


simple_graph('UndergradMajor', 'pie')

# <hr>

# ### <a id='8.17'>8.17 Are developer pursuing any degree ?</a>

# In[ ]:


simple_graph('Student', 'pie')

# <hr>

# ### <a id='8.18'>8.18 Summary</a>

# - Many Developers use 2 monitors
# - Most of the devloper got promoted and many of them are co-founder of their own company
# - Developer wake up between 7 - 8
# - They are moderately satisfied
# - 46% people are bachelors degree
# - 12K people are extremely dissatisfied
# - Majority of developers are computer science students 
# - Most of the developers are not pursuing any kind of degree

# <hr>

# ## <a id='9'>9 About Companies</a><br>

# ### <a id='9.1'>9.1 Company size</a><br>

# In[ ]:


simple_graph('CompanySize','barv',10)

# <hr>

# ### <a id='9.2'>9.2 Top communication tool</a><br>
# 

# In[ ]:


simple_graph('CommunicationTools','barv',10)

# <hr>

# ### <a id='9.3'>9.3 Which type of compaines have quick productive</a><br>

# In[ ]:


data = pd.DataFrame(columns = df['CompanySize'].dropna().unique(),index = df['TimeFullyProductive'].dropna().unique())
for col in data.columns:
    for index in data.index:
        data[col][index] = len(df[(df['CompanySize'] == col) & (df['TimeFullyProductive'] == index)])
        
graph = [
    go.Scatter(
        x=data.index,
        y=data['20 to 99 employees'],
        name = '20 to 99 employees',
    ),
    go.Scatter(
        x=data.index,
        y=data['10,000 or more employees'],
        name = '10,000 or more employees',
    ),
    go.Scatter(
        x=data.index,
        y=data['100 to 499 employees'],
        name = '100 to 499 employees',
    ),
    go.Scatter(
        x=data.index,
        y=data['10 to 19 employees'],
        name = '10 to 19 employees',
    ),
    go.Scatter(
        x=data.index,
        y=data['500 to 999 employees'],
        name = '500 to 999 employees',
    ),
    go.Scatter(
        x=data.index,
        y=data['1,000 to 4,999 employees'],
        name = '1,000 to 4,999 employees',
    ),
    go.Scatter(
        x=data.index,
        y=data['5,000 to 9,999 employees'],
        name = '5,000 to 9,999 employees',
    ),
    go.Scatter(
        x=data.index,
        y=data['Fewer than 10 employees'],
        name = 'Fewer than 10 employees',
    )
]

layout = go.Layout(barmode='stack')

fig = go.Figure(data=graph, layout=layout)

py.iplot(fig)

# <hr>

# ### <a id='9.4'>9.4 As company size grows developers are satisfied</a><br>
# 

# In[ ]:


data = pd.DataFrame(columns = df['CompanySize'].dropna().unique(),index = df['JobSatisfaction'].dropna().unique())
for col in data.columns:
    for index in data.index:
        data[col][index] = len(df[(df['CompanySize'] == col) & (df['JobSatisfaction'] == index)])
        
graph = [
    go.Scatter(
        x=data.index,
        y=data['20 to 99 employees'],
        name = '20 to 99 employees',
    ),
    go.Scatter(
        x=data.index,
        y=data['10,000 or more employees'],
        name = '10,000 or more employees',
    ),
    go.Scatter(
        x=data.index,
        y=data['100 to 499 employees'],
        name = '100 to 499 employees',
    ),
    go.Scatter(
        x=data.index,
        y=data['10 to 19 employees'],
        name = '10 to 19 employees',
    ),
    go.Scatter(
        x=data.index,
        y=data['500 to 999 employees'],
        name = '500 to 999 employees',
    ),
    go.Scatter(
        x=data.index,
        y=data['1,000 to 4,999 employees'],
        name = '1,000 to 4,999 employees',
    ),
    go.Scatter(
        x=data.index,
        y=data['5,000 to 9,999 employees'],
        name = '5,000 to 9,999 employees',
    ),
    go.Scatter(
        x=data.index,
        y=data['Fewer than 10 employees'],
        name = 'Fewer than 10 employees',
    )
]

layout = go.Layout(barmode='stack')

fig = go.Figure(data=graph, layout=layout)

py.iplot(fig)

# <hr>

# ### <a id='9.5'>9.5 As company size grows developers code as hobby or loose interest?</a><br>

# In[ ]:


data = pd.DataFrame(columns = df['CompanySize'].dropna().unique(),index = df['Hobby'].dropna().unique())
for col in data.columns:
    for index in data.index:
        data[col][index] = len(df[(df['CompanySize'] == col) & (df['Hobby'] == index)])
        
graph = [
    go.Bar(
        x=data.index,
        y=data['20 to 99 employees'],
        name = '20 to 99 employees',
    ),
    go.Bar(
        x=data.index,
        y=data['10,000 or more employees'],
        name = '10,000 or more employees',
    ),
    go.Bar(
        x=data.index,
        y=data['100 to 499 employees'],
        name = '100 to 499 employees',
    ),
    go.Bar(
        x=data.index,
        y=data['10 to 19 employees'],
        name = '10 to 19 employees',
    ),
    go.Bar(
        x=data.index,
        y=data['500 to 999 employees'],
        name = '500 to 999 employees',
    ),
    go.Bar(
        x=data.index,
        y=data['1,000 to 4,999 employees'],
        name = '1,000 to 4,999 employees',
    ),
    go.Bar(
        x=data.index,
        y=data['5,000 to 9,999 employees'],
        name = '5,000 to 9,999 employees',
    ),
    go.Bar(
        x=data.index,
        y=data['Fewer than 10 employees'],
        name = 'Fewer than 10 employees',
    )
]

layout = go.Layout(barmode='stack')

fig = go.Figure(data=graph, layout=layout)

py.iplot(fig)

# <hr>

# ### <a id='9.6'>9.6 Summary</a><br>

# - 16K people work in the company size of 20 - 100, reason is because of rise in start-up industry.
# - Slack is the most popular communication tool used world wide.
# - productivity of startup companies are really great.
# - Employees of startup companies are happy :-).
# - There are people who are unhappy in startup companies maybe coz of short deadline.

# <hr>

# ### <a id='10'>10 Gender based Analysis</a><br>

# ### <a id='10.1'>10.1 Top countries with female employes</a>

# In[ ]:


countries = df[['Country','Gender']]
countries = countries[countries['Gender'] == 'Female']
countries = countries['Country'].value_counts()

countries = countries.to_frame().reset_index()
countries.loc[2]['code'] = ''
for i,country in enumerate(countries['index']):
    user_input = country
    mapping = {country.name: country.alpha_3 for country in pycountry.countries}
    countries.set_value(i, 'code', mapping.get(user_input))
data = [ dict(
        type = 'choropleth',
        locations = countries['code'],
        z = countries['Country'],
        text = countries['index'],
        colorscale = [[0,"rgb(5, 10, 172)"],[0.35,"rgb(40, 60, 190)"],[0.5,"rgb(70, 100, 245)"],\
            [0.6,"rgb(90, 120, 245)"],[0.7,"rgb(106, 137, 247)"],[1,"rgb(220, 220, 220)"]],
        autocolorscale = False,
        reversescale = True,
        marker = dict(
            line = dict (
                color = 'rgb(180,180,180)',
                width = 0.5
            ) ),
        colorbar = dict(
            autotick = False,
            title = 'Total Female'),
      ) ]

layout = dict(
    title = 'countries with female employees',
    geo = dict(
        showframe = False,
        showcoastlines = False,
        projection = dict(
            type = 'Mercator'
        )
    )
)

fig = dict( data=data, layout=layout )
py.iplot( fig, validate=False)

# <hr>

# ### <a id='10.3'>10.3 male vs female ratio</a>

# In[ ]:


simple_graph('Gender','barv',10)

# <hr>

# ### <a id='10.3'>10.3 Summary</a>

# - Female employees across the world
# - Male developers are more than female

# <hr>
