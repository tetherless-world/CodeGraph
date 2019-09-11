#!/usr/bin/env python
# coding: utf-8

# # Guide to Animated bubble charts using Plotly

# I used to be amazed by [Hans Rosling's Ted Talks](https://www.ted.com/playlists/474/the_best_hans_rosling_talks_yo) even before *data science* was in my vocabulary. The animated graphs in his presentations would bring data to life. One of his signature graphs is re-created below using the dataset of global indicators provided by [Gapminder](https://www.gapminder.org/).  

# In[ ]:


import pandas as pd
import numpy as np
from bubbly.bubbly import bubbleplot 
from __future__ import division
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode()

gapminder_indicators = pd.read_csv('../input/gapminder/gapminder.tsv', delimiter='\t')
honey_production = pd.read_csv('../input/honey-production/honeyproduction.csv')
world_happiness = pd.read_csv('../input/world-happiness/2017.csv')

# Using the function `bubbleplot` from the module [`bubbly`(bubble charts with plotly)](https://github.com/AashitaK/bubbly):

# In[ ]:


from bubbly.bubbly import bubbleplot 

figure = bubbleplot(dataset=gapminder_indicators, x_column='gdpPercap', y_column='lifeExp', 
    bubble_column='country', time_column='year', size_column='pop', color_column='continent', 
    x_title="GDP per Capita", y_title="Life Expectancy", title='Gapminder Global Indicators',
    x_logscale=True, scale_bubble=3, height=650)

iplot(figure, config={'scrollzoom': True})

# In the graph above, the size corresponds to the population of each country and the values of GDP per capital and life expectency along with the name of the country can be seen by hovering over the cursor on the bubbles.

# The animated bubble charts convey a great deal of information since they can accomodate upto *seven variables* in total, namely:
# * X-axis (GDP per capita)
# * Y-axis (Life Expectancy)
# * Z-axis 
# * Bubbles (Countries, can be seen by hovering the cursor over the dots)
# * Time (in years)
# * Size of bubbles (Population)
# * Color of bubbles (Continents, variable can be categorical or numerical)
# 
# This means that these captivating graphs can be used very creatively on different datasets to convey a lot of information in a compact manner. The code for plotting them ran over 300 lines and hence I decided to write a python module [`bubbly`](https://github.com/AashitaK/bubbly), short form for bubble charts with plotly, from which the function `bubbleplot` is imported. The function has the following self-explanatory arguments:
# * dataset (pandas DataFrame)
# * x_column (numerical column) 
# * y_column (numerical column) 
# * z_column (optional, numerical column)
# * bubble_column (numerical or categorical column) 
# * time_column (optional, numerical column)
# * size_column (optional, numerical column) 
# * color_column (optional, can be either `category`, `object`, `bool`, `int` or `float` dtype)
# 
# More arguments for formatting the plots:
# * x_title (optional) 
# * y_title (optional) 
# * z_title (optional) 
# * title (optional) 
# * colorbar_title (optional, ignored in case of categorical color_column)
# * x_logscale (optional: True or False, default: False)
# * y_logscale (optional: True or False, default: False) 
# * z_logscale (optional: True or False, default: False) 
# * x_range (optional)
# * y_range (optional)
# * z_range (optional)
# * scale_bubble (optional, default: 1)
# * colorscale (optional, ignored in case of categorical color_column)
# * width (optional)
# * height (optional)
# * show_slider (optional: True or False, default: True)
# * show_button (optional: True or False, default: True)
# * show_colorbar (optional: True or False, default: True, ignored in case of categorical color_column)
# 
# 
# The function `bubbleplot` is ready-to-use on any suitable dataset, so you are welcome to use it in your kernels, simply by adding `AashitaK/bubbly` as a custom package in the settings section. I would appreciate a reference with link to this kernel and/or the module  [`bubbly`](https://github.com/AashitaK/bubbly) in case you do so. 
# 
# Below are more charts on [Honey production dataset](https://www.kaggle.com/jessicali9530/honey-production/kernels) using the function `bubbleplot` from the module [`bubbly`](https://github.com/AashitaK/bubbly):

# In[ ]:


figure = bubbleplot(dataset=honey_production, x_column='totalprod', y_column='prodvalue',
    bubble_column='state', time_column='year', size_column='yieldpercol', color_column='numcol',
    x_title="Total production", y_title="Production value", 
    title='Timeline of honey production for each state in USA', 
    colorbar_title='# colonies', colorscale='Viridis', 
    x_logscale=True, y_logscale=True, scale_bubble=1.8, height=650) 

iplot(figure, config={'scrollzoom': True})

# In[ ]:


figure = bubbleplot(dataset=honey_production, x_column='prodvalue', y_column='priceperlb',
    bubble_column='state', time_column='year', size_column='yieldpercol', color_column='numcol',
    x_title="Production value", y_title="Price per pound", 
    title='Timeline of honey production and its prices for each state in USA', 
    colorbar_title='Yield per colony (lbs)', colorscale='Blackbody', y_range=[-0.5,0.9],
    x_logscale=True, y_logscale=True, scale_bubble=1.8, height=650) 

iplot(figure, config={'scrollzoom': True})

# In[ ]:


figure = bubbleplot(dataset=honey_production, x_column='totalprod', y_column='stocks', 
    bubble_column='state', time_column='year', size_column='numcol', color_column='yieldpercol', 
    x_title="Total production", y_title="Stocks", 
    title='Timeline of total honey production and its stock value for each state in USA', 
    colorbar_title='Yield per colony (lbs)', scale_bubble=1.8, colorscale='Earth',
    x_logscale=True, y_logscale=True, height=650) 

iplot(figure, config={'scrollzoom': True})

# 
# # Some static charts:
# It is not always the case that animated bubble charts are most suitable for use. The package [`bubbly`](https://github.com/AashitaK/bubbly/) very well supports non-animated charts as well.
# 
# The following static chart conveys it better that the regional inequality in GDP per capita is increasing over the last decade.  

# In[ ]:


gapminder_indicators_continents = gapminder_indicators.groupby(['continent', 'year']).mean().reset_index()

figure = bubbleplot(dataset=gapminder_indicators_continents, 
    x_column='year', y_column='gdpPercap', bubble_column='continent',  
    size_column='pop', color_column='continent', 
    x_title="Years", y_title="GDP per capita", 
    title='GDP per capita inequality among geographical regions is increasing over time',
    x_range=[1945, 2015], y_range=[0, 35000],
    scale_bubble=1.5, height=650)

iplot(figure, config={'scrollzoom': True})

# In the following chart, time is denoted by color to show that even though Vietnam lags far behind Hungary in terms of GDP per capita, it has catched up with the same level of life expentancy within a decade starting far below.  The size of bubbles corresponds to the population.

# In[ ]:


dataset = gapminder_indicators[gapminder_indicators['country'].isin(['Hungary', 'Vietnam'])]
figure = bubbleplot(dataset=dataset, x_column='gdpPercap', y_column='lifeExp', 
    bubble_column='country', color_column='year', size_column='pop',
    x_title="GDP per Capita", y_title="Life Expectancy", 
    title='GDP per capita and Life Expectancy of Hungary and Vietnam over time',
    colorbar_title = 'Time in years',
    x_logscale=True, scale_bubble=2, height=650)

iplot(figure, config={'scrollzoom': True})

# The above chart captures the common pattern among many emerging nations that the human development indicators first rise steeply followed by economic indicators, unlike the development patterns of the western nations in the past century. 

# # Some 3D charts

# Data for the following chart comes from [World Happiness Report](https://www.kaggle.com/unsdsn/world-happiness) and the size of the bubbles corresponds to the generosity of nations.

# In[ ]:


figure = bubbleplot(dataset=world_happiness, 
    x_column='Economy..GDP.per.Capita.', y_column='Health..Life.Expectancy.', bubble_column='Country',  
    color_column='Happiness.Score', z_column='Trust..Government.Corruption.', size_column='Generosity',
    x_title="GDP per capita", y_title="Life Expectancy", z_title="Corruption",
    title='Impact of Economy, Health and Govt. on Happiness Scores of Nations',
    colorbar_title='Happiness Score', marker_opacity=1, colorscale='Portland',
    scale_bubble=0.8, height=650)

iplot(figure, config={'scrollzoom': True})

# In[ ]:


figure = bubbleplot(dataset=gapminder_indicators, x_column='pop', y_column='lifeExp', 
    bubble_column='country', time_column='year', z_column='gdpPercap', color_column='continent', 
    x_title="Population", y_title="Life Expectancy", z_title="GDP per Capita",
    title='Gapminder Global Indicators', x_logscale=True, z_logscale=True, 
    scale_bubble=0.8, marker_opacity=0.8, height=700)

iplot(figure, config={'scrollzoom': True})

# # Building it from the scratch
# 
# If you are like me, with the habit of taking things apart and putting them back together to learn and understand them fully, then below is a step-by-step guide to plotting interactive bubble charts using `Plotly` that only expands on the functions coded in the module  [`bubbly`](https://github.com/AashitaK/bubbly). 
# 
# ## Steps:
# * [Setting up the grid](#grid)
# * [Plotting basic scatter plot](#scatter)
# * [Adding animated time frames](#frames)
# * [Adding slider bar for time scale](#slider)
# * [Adding pause-play button](#button)
# * [Using bubble size as a variable](#size)
# * [Using colors to classify into categories](#color)
# 
# Note: The following section is not required for going ahead to use  [`bubbly`](https://github.com/AashitaK/bubbly) to make fun animated bubble charts in your kernels using the function `bubbleplot`. 
# 
# We are using `Plotly` in offline mode for it is the [only mode that works in Kaggle](https://www.kaggle.com/residentmario/introduction-to-plotly-optional).

# ## Setting up the grid <a name="grid"></a>

# In the [`Plotly`'s official tutorial](https://plot.ly/python/gapminder-example/) for the animated bubble charts, the grids are made using the `Grid` and  `Column` objects from `plotly.grid_objs`. However, these objects are not supported for the offline mode and hence, they cannot be used as it is in Kaggle kernels. This can be fixed by simply defining the grid as a `pandas DataFrame` object and modifying the plotting of the time frames on the grid accordingly as shown below.

# In[ ]:


# Define the dataset and the columns
dataset = gapminder_indicators
x_column = 'gdpPercap'
y_column = 'lifeExp'
bubble_column = 'country'
time_column = 'year'

# In[ ]:


# Get the years in the dataset
years = dataset[time_column].unique()

# Make the grid
grid = pd.DataFrame()
col_name_template = '{year}+{header}_grid'
for year in years:
    dataset_by_year = dataset[(dataset['year'] == int(year))]
    for col_name in [x_column, y_column, bubble_column]:
        # Each column name is unique
        temp = col_name_template.format(
            year=year, header=col_name
        )
        if dataset_by_year[col_name].size != 0:
            grid = grid.append({'value': list(dataset_by_year[col_name]), 'key': temp}, 
                               ignore_index=True)

grid.sample(10)

# ## Plotting basic scatter plot <a name="scatter"></a>

# Once the grid is defined with the data points, we next use it to scatter the dataset for the earliest year viz. $1952$

# In[ ]:


# Define figure
figure = {
    'data': [],
    'layout': {},
    'frames': []
}

# Get the earliest year
year = min(years)

# Make the trace
trace = {
    'x': grid.loc[grid['key']==col_name_template.format(
        year=year, header=x_column
    ), 'value'].values[0], 
    'y': grid.loc[grid['key']==col_name_template.format(
        year=year, header=y_column
    ), 'value'].values[0],
    'mode': 'markers',
    'text': grid.loc[grid['key']==col_name_template.format(
        year=year, header=bubble_column
    ), 'value'].values[0]
}
# Append the trace to the figure
figure['data'].append(trace)

# Plot the figure
iplot(figure, config={'scrollzoom': True})

# We convert the `x-axis` into the logarithmic scale, which is more suitable for the visualization of our dataset. We set up the basic layout, noting that, in this case of changing frames, using `autorange` for the axes does not render the graph well, so we set up the ranges manually.

# In[ ]:


# Get the max and min range of both axes
xmin = min(np.log10(dataset[x_column]))*0.98
xmax = max(np.log10(dataset[x_column]))*1.02
ymin = min(dataset[y_column])*0.75
ymax = max(dataset[y_column])*1.25

# Modify the layout
figure['layout']['xaxis'] = {'title': 'GDP per Capita', 'type': 'log', 
                             'range': [xmin, xmax]}   
figure['layout']['yaxis'] = {'title': 'Life Expectancy', 
                             'range': [ymin, ymax]} 
figure['layout']['title'] = 'Gapminder Global Indicators'
figure['layout']['showlegend'] = False
figure['layout']['hovermode'] = 'closest'
iplot(figure, config={'scrollzoom': True})

# ## Adding animated time frames <a name="frames"></a>

# Next we add frames for each year resulting in an animated graph, though not interactive yet.

# In[ ]:


for year in years:
    # Make a frame for each year
    frame = {'data': [], 'name': str(year)}
    
    # Make a trace for each frame
    trace = {
        'x': grid.loc[grid['key']==col_name_template.format(
            year=year, header=x_column
        ), 'value'].values[0],
        'y': grid.loc[grid['key']==col_name_template.format(
            year=year, header=y_column
        ), 'value'].values[0],
        'mode': 'markers',
        'text': grid.loc[grid['key']==col_name_template.format(
            year=year, header=bubble_column
        ), 'value'].values[0],
        'type': 'scatter'
    }
    # Add trace to the frame
    frame['data'].append(trace)
    # Add frame to the figure
    figure['frames'].append(frame) 

iplot(figure, config={'scrollzoom': True})

# The animation happened only once, right after executing the code. To be able to make it interactive, we add a slider bar for the time. 

# ## Adding slider bar for time scale <a name="slider"></a>

# The function `add_slider_steps` used here is imported from the module  [`bubbly`](https://github.com/AashitaK/bubbly).

# In[ ]:


figure['layout']['sliders'] = {
    'args': [
        'slider.value', {
            'duration': 400,
            'ease': 'cubic-in-out'
        }
    ],
    'initialValue': min(years),
    'plotlycommand': 'animate',
    'values': years,
    'visible': True
}
sliders_dict = {
    'active': 0,
    'yanchor': 'top',
    'xanchor': 'left',
    'currentvalue': {
        'font': {'size': 20},
        'prefix': 'Year:',
        'visible': True,
        'xanchor': 'right'
    },
    'transition': {'duration': 300, 'easing': 'cubic-in-out'},
    'pad': {'b': 10, 't': 50},
    'len': 0.9,
    'x': 0.1,
    'y': 0,
    'steps': []
}

from bubbly.bubbly import add_slider_steps

for year in years:
    add_slider_steps(sliders_dict, year)
    
figure['layout']['sliders'] = [sliders_dict]
iplot(figure, config={'scrollzoom': True})

# We can also add pause and play buttons for the interactive chart.

# ## Adding pause-play button <a name="button"></a>

# In[ ]:


figure['layout']['updatemenus'] = [
    {
        'buttons': [
            {
                'args': [None, {'frame': {'duration': 500, 'redraw': False},
                         'fromcurrent': True, 'transition': {'duration': 300, 
                                                             'easing': 'quadratic-in-out'}}],
                'label': 'Play',
                'method': 'animate'
            },
            {
                'args': [[None], {'frame': {'duration':0, 'redraw': False}, 'mode': 'immediate',
                'transition': {'duration': 0}}],
                'label': 'Pause',
                'method': 'animate'
            }
        ],
        'direction': 'left',
        'pad': {'r': 10, 't': 87},
        'showactive': False,
        'type': 'buttons',
        'x': 0.1,
        'xanchor': 'right',
        'y': 0,
        'yanchor': 'top'
    }
]
iplot(figure, config={'scrollzoom': True})

# ## Using bubble size as a variable <a name="size"></a>

# Now we build on the above interactive graph by setting the size of the bubble as another variable viz. population. Since, there will be repetition of the code illustrated above, this time we have simply used the functions  `make_grid`, `set_layout` and `get_trace` imported from the module  [`bubbly`](https://github.com/AashitaK/bubbly). 

# In[ ]:


from bubbly.bubbly import make_grid, set_layout, get_trace

# Define the new variable
size_column = 'pop' 

# Make the grid
years = dataset[time_column].unique()
col_name_template = '{}+{}_grid'
column_names = [x_column, y_column, bubble_column, size_column]
grid = make_grid(dataset, column_names, time_column, years)
    
# Set the layout
figure, sliders_dict = set_layout(x_title='GDP per Capita', y_title='Life Expectancy', 
            title='Gapminder Global Indicators', x_logscale=True, y_logscale=False, 
            show_slider=True, slider_scale=years, show_button=True, show_legend=False, 
            height=650)

# Add the base frame
year = min(years)
col_name_template_year = col_name_template.format(year, {})
trace = get_trace(grid=grid, col_name_template=col_name_template_year, 
                    x_column=x_column, y_column=y_column, 
                    bubble_column=bubble_column, size_column=size_column)
figure['data'].append(trace)


# Add time frames
for year in years:
    frame = {'data': [], 'name': str(year)}
    col_name_template_year = col_name_template.format(year, {})
    trace = get_trace(grid=grid, col_name_template=col_name_template_year, 
                    x_column=x_column, y_column=y_column, 
                    bubble_column=bubble_column, size_column=size_column)
    frame['data'].append(trace)
    figure['frames'].append(frame) 
    add_slider_steps(sliders_dict, year)

# Set the layout once more
figure['layout']['xaxis']['autorange'] = True
figure['layout']['yaxis']['range'] = [20, 100]
figure['layout']['sliders'] = [sliders_dict]

# Plot the animation
iplot(figure, config={'scrollzoom': True})

# ## Using colors to classify into categories <a name="color"></a>

# Now, a category variable, namely continent in our case, is added. For making the grid, we will have an additional `for` loop for continents. Since the function is still not that different from the code above, we have used the function `make_grid_with_categories` imported from the module  [`bubbly`](https://github.com/AashitaK/bubbly) instead of lengthening the code block. Similarly, there is an additional `for` loop for continents while adding data to time frames as well. The rest of the process is similar to that of above, except that the function `make_data_dictionary` is passed an additional argument, namely `continent`.

# In[ ]:


from bubbly.bubbly import make_grid_with_categories

# Define the new variable
category_column = 'continent'

# Make the grid
years = dataset[time_column].unique()
continents = dataset[category_column].unique()
col_name_template = '{}+{}+{}_grid'
column_names = [x_column, y_column, bubble_column, size_column]
grid = make_grid_with_categories(dataset, column_names, time_column, 
                                 category_column, years, continents)
    
# Set the layout
figure, sliders_dict = set_layout(x_title='GDP per Capita', y_title='Life Expectancy', 
            title='Gapminder Global Indicators', x_logscale=True, y_logscale=False, 
            show_slider=True, slider_scale=years, show_button=True, show_legend=False, height=650)

# Add the base frame
year = min(years)
col_name_template_year = col_name_template.format(year, {}, {})
for continent in continents:
    trace = get_trace(grid=grid, col_name_template=col_name_template_year, 
                    x_column=x_column, y_column=y_column, 
                    bubble_column=bubble_column, size_column=size_column, category=continent)
    figure['data'].append(trace)

# Add time frames
for year in years:
    frame = {'data': [], 'name': str(year)}
    col_name_template_year = col_name_template.format(year, {}, {})
    for continent in continents:
        trace = get_trace(grid=grid, col_name_template=col_name_template_year, 
                    x_column=x_column, y_column=y_column, 
                    bubble_column=bubble_column, size_column=size_column, category=continent)
        frame['data'].append(trace)

    figure['frames'].append(frame) 
    add_slider_steps(sliders_dict, year)

# Set the layout once more
figure['layout']['xaxis']['autorange'] = True
figure['layout']['yaxis']['range'] = [20, 100]
figure['layout']['showlegend'] = True
figure['layout']['sliders'] = [sliders_dict]

# Plot the animation
iplot(figure, config={'scrollzoom': True})

# So, we have finally generated the same interactive graph as the one generated above using the function `bubbleplot`.

# ### References:
# * https://plot.ly/python/animations/
# * https://stackoverflow.com/questions/45780920/plotly-icreate-animations-offline-on-jupyter-notebook
# * https://plot.ly/python/gapminder-example/
# 
# ### More kernels illustrating the use of `bubbly`:
# * https://www.kaggle.com/gpreda/meta-kaggle-what-happened-to-the-team-size
# * https://www.kaggle.com/gpreda/earthquakes-animation-using-bubbly

# All questions, comments and/or suggestions for improvement of the module  [`bubbly`](https://github.com/AashitaK/bubbly) are most welcome! 
# 
# ***Update: I am looking for a Maintainer for this [python package](https://pypi.org/project/bubbly/). There are some open requests and issues to fix that I am not able to address in a timely manner because of other work commitments. If you are interested, please reach out. My contact information is given in my Kaggle profile.***

# In[ ]:



