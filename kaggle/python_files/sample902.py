#!/usr/bin/env python
# coding: utf-8

# ____
# * **Day 1**: Determining what information should be monitored with a dashboard. [Notebook](https://www.kaggle.com/rtatman/dashboarding-with-notebooks-day-1), [Livestream Recording](https://www.youtube.com/watch?v=QO2ihJS2QLM)
# * **Day 2**: How to create effective dashboards in notebooks, [Python Notebook](https://www.kaggle.com/rtatman/dashboarding-with-notebooks-day-2-python), [R Notebook](https://www.kaggle.com/rtatman/dashboarding-with-notebooks-day-2-r), [Livestream](https://www.youtube.com/watch?v=rhi_nexCUMI)
# * **Day 3**: Running notebooks with the Kaggle API, [Notebook](https://www.kaggle.com/rtatman/dashboarding-with-notebooks-day-3), [Livestream](https://youtu.be/cdEUEe2scNo)
# * **Day 4**: Scheduling notebook runs using cloud services, [Notebook](https://www.kaggle.com/rtatman/dashboarding-with-notebooks-day-4), [Livestream](https://youtu.be/Oujj6nT7etY)
# * **Day 5**: Testing and validation, [Python Notebook](https://www.kaggle.com/rtatman/dashboarding-with-notebooks-day-5), [R Notebook](https://www.kaggle.com/rtatman/dashboarding-with-notebooks-day-5-r), [Livestream](https://www.youtube.com/watch?v=H6DcpIykT8E)
# 
# ____
# 
# Welcome to the second day of Dashboarding with scheduled notebooks. Today we're going to do two things:
# 
# * Make our visualizations interactive
# * Tidy up our notebook to remove unnecessary clutter
#  
# Today's timeline:
# 
# * **5 minutes**: Read notebook
# * **10 minutes**: Make your visualizations interactive
# * **5 minutes**: Tidy up notebook
# 
# # Interactive visualizations
# 
# Dashboards, unlike static charts, are designed to be viewed on a device, whether that’s a computer, phone or tablet. That means that you have the option to make your dashboard interactive. This lets the people you’re sharing the dashboard ask their own questions about the data. (And, perhaps more importantly, stops them from emailing you asking you to remake the chart with slightly different axes.)
# 
# 
# ## What should you make interactive?
# 
# Personally I think you get to the point of diminishing returns on interactive charts pretty quickly. Sure, an animation that changes the mouse shape to the flag of the country you’re hovering over on the map would be super cool… but coding up something like that probably isn’t the best use of your time.
# 
# Instead, the most important things to make interactive are:
# 
# * *Let viewers adjust axes*. Viewers should able to zoom in or out and adjust the range themselves. In particular, it's helpful to be able to adjust the time range for timelines and the latitude and longitude for maps.
# * *Make values or labels visible on hover*. This lets viewers ask questions like “What’s this outlier here?” or “How many units did we sell last Wednesday?".
# 
# ## What library should you use for interactive visualizations? 
# 
# You’ve got a lot of options for libraries for making interactive visualizations. Some of these include:
# 
# * [Altair](https://altair-viz.github.io/)
# * [Bokeh](https://bokeh.pydata.org/en/latest/)
# * [Folium](https://github.com/python-visualization/folium) for maps
# 
# You can use your favorite library (and feel free to give it a shoutout in the comments!), but I’m going to be using Plotly here. Plotly is an open source JavaScript graphing library that has wrappers available in a number of languages, including Python and R. 
# 
# > **Why Plotly?** I like Plotly because it's interactive by default, so I don't have to spend a bunch of time futzing with it. The syntax is also fairly similar between Python and R, which makes it easier to switch between languages. Finally, it renders well in notebooks, which can sometimes be a problem with fancy JavaScript based visualizations.
# 
# In both graphs below, you should be able to zoom and see more information for a specific data point by hovering your mouse over it. You can also download plots by clicking on the little camera icon, which is pretty nifty.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# read in data
data = pd.read_csv("../input/meets.csv")


## Hacky data munging
# parse dates
data['Date'] = pd.to_datetime(data['Date'], format = "%Y-%m-%d")

# count of meets per month
meets_by_month = data['Date'].groupby([data.Date.dt.year, data.Date.dt.month]).agg('count') 

# convert to dataframe
meets_by_month = meets_by_month.to_frame()

# move date month from index to column
meets_by_month['date'] = meets_by_month.index

# rename column
meets_by_month = meets_by_month.rename(columns={meets_by_month.columns[0]:"meets"})

# re-parse dates
meets_by_month['date'] = pd.to_datetime(meets_by_month['date'], format="(%Y, %m)")

# remove index
meets_by_month = meets_by_month.reset_index(drop=True)

# get month of meet
meets_by_month['month'] = meets_by_month.date.dt.month

# repeat to get number of meets per state
meet_by_state = data['MeetState'].value_counts().to_frame()
meet_by_state['state'] = meet_by_state.index
meet_by_state = meet_by_state.rename(columns={meet_by_state.columns[0]:"meets"})

# In[ ]:


# import plotly
import plotly.plotly as py
import plotly.graph_objs as go

# these two lines are what allow your code to show up in a notebook!
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode()

# sepcify that we want a scatter plot with, with date on the x axis and meet on the y axis
data = [go.Scatter(x=meets_by_month.date, y=meets_by_month.meets)]

# specify the layout of our figure
layout = dict(title = "Number of Powerlifting Meets per Month",
              xaxis= dict(title= 'Date',ticklen= 5,zeroline= False))

# create and show our figure
fig = dict(data = data, layout = layout)
iplot(fig)

# In[ ]:


# specify what we want our map to look like
data = [ dict(
        type='choropleth',
        autocolorscale = False,
        locations = meet_by_state['state'],
        z = meet_by_state['meets'],
        locationmode = 'USA-states'
       ) ]

# chart information
layout = dict(
        title = 'Number of Powerlifting Meets per State',
        geo = dict(
            scope='usa',
            projection=dict( type='albers usa' ),
            showlakes = True,
            lakecolor = 'rgb(255, 255, 255)'),
             )
   
# actually show our figure
fig = dict( data=data, layout=layout )
iplot( fig, filename='d3-cloropleth-map' )

# # Tidying up your notebook
# 
# If you’re using a script, you can ignore this bit. If you’re using a notebook, however, there are some changes you’re probably going to want to make before you share your dashboard. 
# 
# 1. **Hide your code**. Since the focus in making a dashboard is to make information easy to quickly see, code is just a distraction. You can hide your code cells in a Kaggle notebook by clicking in the cell you want to hide and clicking "input" in the menu in the upper right hand corner. 
# 2. **Remove text**. I personally don’t like a lot of text, like markdown cells, in dashboards. Instead, try using informative labels for your charts and axes so they can stand on their own.
# 3. **Consider putting small charts or tables together in the same line.** This helps save screen space so viewers don’t have to scroll as much. You can do this using [layout()](https://www.rdocumentation.org/packages/graphics/versions/3.5.1/topics/layout) in R. In Python you’ll need package-specific functions to do this, like [plt.subplot](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.subplot.html) for Matplotlib. 
# 
# Ideally, you want to be able to see the entire dashboard on you rscreen at the same time so that you can quickly glance at it and see the information you need. 
# 
# # Your turn!
# 
# Now that I’ve given you some tips and tricks on how to get your notebook looking good, it’s time for you to apply them! Revisit your visualizations from yesterday and make them interactive. Then spend some time removing distracting elements (text and code) from your notebook.
# 
# If you like, you can make your kernel public and share a link to it in the comments on this dataset to share with other participants. (And you can take a peek at other people's work to see what they've chosen to look at!) I'll pick a couple that I especially like to highlight as examples. :)
