#!/usr/bin/env python
# coding: utf-8

# # <font color="#703bdb">Part 0. Getting Familier with the Dataset - CPE </font> <hr>
# 
# <a href="http://policingequity.org/">Center of Policing Equity</a> is a research and action think tank that works collaboratively with law enforcement, communities, and political stakeholders to identify ways to strengthen relationships with the communities they serve. CPE is also the home of the nation’s first and largest <a href="http://policingequity.org/national-justice-database/">database</a> tracking national statistics on police behavior. 
# 
# The main aim of CPE is to bridge the divide created by communication problems, suffering and generational mistrust, and forge a path towards public safety, community trust, and racial equity. This kernel series is my contribution to the <a href="https://www.kaggle.com/center-for-policing-equity/data-science-for-good">Data Science for Good: Center for Policing Equity</a>. The contribution is focused on providing a generic, robust, and automated approach to integrate, standardize the data and further diagnose disparities in policing, shed light on police behavior, and provide actionable recommendations. 
# 
# ### <font color="#703bdb">Main Submission: </font>
# 
# <ul>
#     <li><a href="https://www.kaggle.com/shivamb/1-solution-workflow-science-of-policing-equity/">Part 1: Solution Workflow - The Science of Policing Equity </a>  </li>
#     <li><a href="https://www.kaggle.com/shivamb/2-automation-pipeline-integration-processing">Part 2: Data Integration and Processing : Automation Pipeline</a>  </li>
#     <li><a href="https://www.kaggle.com/shivamb/3-example-runs-of-automation-pipeline">Part 3: Example Runs of Automation Pipeline </a>  </li> 
#     <li><a href="https://www.kaggle.com/shivamb/4-1-analysis-report-minneapolis-24-00013">Part 4.1: Analysis Report - Measuring Equity - Minneapolis Police Department </a>   </li>
#     <li><a href="https://www.kaggle.com/shivamb/4-2-analysis-report-lapd-49-00033">Part 4.2: Analysis Report - Los Angles Police Department (49-00033) </a>   </li>
#     <li><a href="https://www.kaggle.com/shivamb/4-3-analysis-report-officer-level-analysis">Part 4.3: Analysis Report - Indianapolis Officer Level Analysis (23-00089) </a>   </li></ul>
# 
# The complete overview of the solution is shared in the *first kernel*. It explains the process and flow of automation, standardization, processing, and analysis of data. In the *second kernel*, the first component of the solution pipeline : data integration and processing is implemented. It processes both core level data as well as department level data. In the *third kernel*, this pipeline is executed and run for several departments. After all the standardized and clean data is produced, it is analysed with different formats of the Analysis Framework in 4.1, 4.2 and 4.3 kernels. In *kernel 4.1*, core analysis is done along with link with crime rate and poverty data. In *kernel 4.2*, core analysis is done along with statistical analysis. In *kernel 4.3*, officer level analysis is done. 
# 
# <hr>
# 
# ## About this Kernel : 
# 
# This kernel is just a starter kernel that aims to provide understanding of the data and unearth the hidden insights from the data shared. First part is a quick exploration of the shared data and the next part is the complete GIS analysis. Lets Load the required libraries first. 

# In[ ]:


import numpy as np 
import pandas as pd 
import folium
from folium import plugins
from io import StringIO
import geopandas as gpd
from pprint import pprint 
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
import plotly.plotly as py
from plotly import tools
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
import seaborn as sns
import os 
init_notebook_mode(connected=True)

# ### About Dataset : Department Files
# 
# The dataset consists of different data files for different police deparments. Lets quickly look at those department names. 

# In[ ]:


depts = [f for f in os.listdir("../input/cpe-data/") if f.startswith("Dept")]
pprint(depts)

# ### About Dataset : Different Data Files for Police Departments
# 
# Among different departments, different files are shared corresponding to different data files, such as Education, Race, Poverty etc. Lets have a look

# In[ ]:


files = os.listdir("../input/cpe-data/Dept_23-00089/23-00089_ACS_data/")
files

# Now, lets start exploring these data files. 
# 
# ### Department : Dept_23-00089, Metric : Race, Sex, Age
# 
# Lets load the dataset

# In[ ]:


basepath = "../input/cpe-data/Dept_23-00089/23-00089_ACS_data/23-00089_ACS_race-sex-age/"
rca_df = pd.read_csv(basepath + "ACS_15_5YR_DP05_with_ann.csv")
rca_df.head()

# The meanings of columns is given in an another file. Here is the description of all the columns used in the avove dataset. 

# In[ ]:


a_df = pd.read_csv(basepath + "ACS_15_5YR_DP05_metadata.csv")

# for j, y in a_df.iterrows():
#     if y['Id'].startswith("Estimate"):
#         print (y['GEO.id'], y['Id'])

a_df.head()

# So there are coluns about Estimate, Margin of Error, Percent related to Sex, Age, Race, and Total Population. Lets start exploring these variables. 
# 
# ### Distribution of Total Population across Census Tracts
# 
# <br>
# 
# **Census Tracts:** 
# Census tracts (CTs) are small, relatively stable geographic areas that usually have a population between 2,500 and 8,000 persons. They are located in census metropolitan areas and in census agglomerations that had a core population of 50,000 or more in the previous census.
# 
# 

# In[ ]:


total_population = rca_df["HC01_VC03"][1:]

trace = go.Histogram(x=total_population, marker=dict(color='orange', opacity=0.6))
layout = dict(title="Total Population Distribution - Across the counties", margin=dict(l=200), width=800, height=400)
data = [trace]
fig = go.Figure(data=data, layout=layout)
iplot(fig)

male_pop = rca_df["HC01_VC04"][1:]
female_pop = rca_df["HC01_VC05"][1:]

trace1 = go.Histogram(x=male_pop, name="male population", marker=dict(color='blue', opacity=0.6))
trace2 = go.Histogram(x=female_pop, name="female population", marker=dict(color='pink', opacity=0.6))
layout = dict(title="Population Distribution Breakdown - Across the Census Tracts", margin=dict(l=200), width=800, height=400)
data = [trace1, trace2]
fig = go.Figure(data=data, layout=layout)
iplot(fig)

# So about 50 census tracts have population around 3000 - 4000. One Census tract has very high population. Female gender percentage is higher in only two of the census tracts. 
# 
# ### Distribution of Age Groups
# 
# Lets plot the census tract wise different agegroup's population count 

# In[ ]:


age_cols = []
names = []
for i in range(13):
    if i < 2:
        i = "0"+str(i+8)
        relcol = "HC01_VC" + str(i)
    else:
        relcol = "HC01_VC" + str(i+8)
    age_cols.append(relcol)
    name = a_df[a_df["GEO.id"] == relcol]["Id"].iloc(0)[0].replace("Estimate; SEX AND AGE - ","")
    names.append(name)

rca_df['GEO.display-label_cln'] = rca_df["GEO.display-label"].apply(lambda x : x.replace(", Marion County, Indiana", "").replace("Census Tract ", "CT: "))

traces = []
for i,agecol in enumerate(age_cols):
    x = rca_df["GEO.display-label_cln"][1:]
    y = rca_df[agecol][1:]
    trace = go.Bar(y=y, x=x, name=names[i])
    traces.append(trace)

tmp = pd.DataFrame()
vals = []
Geo = []
Col = []
for i,age_col in enumerate(age_cols):
    Geo += list(rca_df["GEO.display-label_cln"][1:].values)
    Col += list([names[i]]*len(rca_df[1:]))
    vals += list(rca_df[age_col][1:].values)

tmp['Geo'] = Geo
tmp['Col'] = Col
tmp['Val'] = vals
tmp['Val'] = tmp['Val'].astype(int)  * 0.01

data = [go.Scatter(x = tmp["Geo"], y = tmp["Col"], mode="markers", marker=dict(size=list(tmp["Val"].values)))]
layout = dict(title="Age Distribution by Census Tract - Marion County, Indiana", legend=dict(x=-0.1, y=1, orientation="h"), 
              margin=dict(l=150, b=100), height=600, barmode="stack")
fig = go.Figure(data=data, layout=layout)
iplot(fig)

# The above plot gives a view about which age groups are located in which areas. Lets look at an other view of age group distributions. 

# In[ ]:


trace1 = go.Histogram(x = rca_df["HC01_VC26"][1:], name="18+", marker=dict(opacity=0.4)) 
trace2 = go.Histogram(x = rca_df["HC01_VC27"][1:], name="21+", marker=dict(opacity=0.3)) 
trace3 = go.Histogram(x = rca_df["HC01_VC28"][1:], name="62+", marker=dict(opacity=0.4)) 
trace4 = go.Histogram(x = rca_df["HC01_VC29"][1:], name="65+", marker=dict(opacity=0.3)) 

titles = ["Age : 18+","Age : 21+","Age : 62+","Age : 65+",]
fig = tools.make_subplots(rows=2, cols=2, print_grid=False, subplot_titles=titles)
fig.append_trace(trace1, 1, 1);
fig.append_trace(trace2, 1, 2);
fig.append_trace(trace3, 2, 1);
fig.append_trace(trace4, 2, 2);
fig['layout'].update(height=600, title="Distribution of Age across the Census Tracts", showlegend=False);
iplot(fig, filename='simple-subplot');

# Let's plot the population distribution by different Race. First, lets consider only the single Race variables

# In[ ]:


single_race_df = rca_df[["HC01_VC49", "HC01_VC50", "HC01_VC51", "HC01_VC56", "HC01_VC64", "HC01_VC69"]][1:]
ops = [1, 0.85, 0.75, 0.65, 0.55, 0.45]
traces = []
for i, col in enumerate(single_race_df.columns):
    nm = a_df[a_df["GEO.id"] == col]["Id"].iloc(0)[0].replace("Estimate; RACE - One race - ", "")
    trace = go.Bar(x=rca_df["GEO.display-label_cln"][1:], y=single_race_df[col], name=nm, marker=dict(opacity=0.6))
    traces.append(trace)
layout = dict(barmode="stack", title="Population Breakdown by Race (Single)", margin=dict(b=100), height=600, legend=dict(x=-0.1, y=1, orientation="h"))
fig = go.Figure(data=traces, layout=layout)
iplot(fig)

# We can see that majority wise White or Black American population exists. It will be interesting to look at which ones are the dominating other races. Lets remove white and black population and plot again

# In[ ]:


traces = []
for i, col in enumerate(single_race_df.columns):
    nm = a_df[a_df["GEO.id"] == col]["Id"].iloc(0)[0].replace("Estimate; RACE - One race - ", "")
    if nm in ["White", "Black or African American"]:
        continue
    trace = go.Bar(x=rca_df["GEO.display-label_cln"][1:], y=single_race_df[col], name=nm, marker=dict(opacity=0.6))
    traces.append(trace)
layout = dict(barmode="stack", title="Population Breakdown by Race (Single)", margin=dict(b=100), height=400, legend=dict(x=-0.1, y=1, orientation="h"))
fig = go.Figure(data=traces, layout=layout)
iplot(fig)

# Lets explore other metrics of the same district
# 
# ### Dept_23-00089, Metric : Poverty
# 

# In[ ]:


basepath2 = "../input/cpe-data/Dept_23-00089/23-00089_ACS_data/23-00089_ACS_poverty/"
a_df = pd.read_csv(basepath2 + "ACS_15_5YR_S1701_metadata.csv")
# for j, y in a_df.iterrows():
#     if "Below poverty level; Estimate" in y['Id']:
#         print (y['GEO.id'], y['Id'])        
        
a_df.T.head()

# In[ ]:


pov_df = pd.read_csv(basepath2 + "ACS_15_5YR_S1701_with_ann.csv")[1:]
pov_df.head()

# pov_df[["HC02_EST_VC66", ""]]
# pov_df["HC02_EST_VC01"] = pov_df["HC02_EST_VC01"].astype(float)
# pov_df.sort_values("HC02_EST_VC01", ascending = False)["HC02_EST_VC01"]

# In[ ]:


age_bp = ["HC02_EST_VC04", "HC02_EST_VC05", "HC02_EST_VC08", "HC02_EST_VC09", "HC02_EST_VC11"]
pov_df[age_bp]

pov_df['GEO.display-label_cln'] = pov_df["GEO.display-label"].apply(lambda x : x.replace(", Marion County, Indiana", "").replace("Census Tract ", "CT: "))

names = ["Below 5", "5-17", "18-34", "34-64", "65+"]

vals = []
Geo = []
Col = []
tmp = pd.DataFrame()
for i,age_col in enumerate(age_bp):
    Geo += list(pov_df["GEO.display-label_cln"][1:].values)
    Col += list([names[i]]*len(pov_df[1:]))
    vals += list(pov_df[age_col][1:].values)

tmp['Geo'] = Geo
tmp['Col'] = Col
tmp['Val'] = vals
tmp['Val'] = tmp['Val'].astype(int)  * 0.025

geos = tmp.groupby("Geo").agg({"Val" : "sum"}).sort_values("Val", ascending = False)[:75].reset_index()['Geo']
tmp1 = tmp[tmp["Geo"].isin(geos)]
data = [go.Scatter(x = tmp1["Geo"], y = tmp1["Col"], mode="markers", marker=dict(color="red", size=list(tmp1["Val"].values)))]
layout = dict(title="Age Distribution by Census Tract - Marion County, Indiana", legend=dict(x=-0.1, y=1, orientation="h"), 
              margin=dict(l=150, b=100), height=600, barmode="stack")
fig = go.Figure(data=data, layout=layout)
iplot(fig)

# ### Dept_23-00089, Metric : Owner Occupied Housing
# 

# In[ ]:


basepath = "../input/cpe-data/Dept_23-00089/23-00089_ACS_data/23-00089_ACS_owner-occupied-housing/"
a_df = pd.read_csv(basepath + "ACS_15_5YR_S2502_metadata.csv")
# for i,val in a_df.iterrows():
#     if "Estimate" in val['Id']:
#         if "Owner-occupied" in val["Id"]:
#             print (val['GEO.id'], val["Id"])
a_df.T.head()    

# ### Department : Dept_23-00089 | Metric : Education
# 

# In[ ]:


basepath = "../input/cpe-data/Dept_23-00089/23-00089_ACS_data/23-00089_ACS_education-attainment/"
a_df = pd.read_csv(basepath + "ACS_15_5YR_S1501_metadata.csv")
a_df.T.head()

# In[ ]:


a_df = pd.read_csv(basepath + "ACS_15_5YR_S1501_with_ann.csv")
a_df.head()

# Similar files are shared for other departments as well. Lets look at an other department. 
# 
# ### Department : Dept_35-00103
# 
# Lets explore the prepped file which contains information about the incidents that occured in that area

# In[ ]:


path = "../input/cpe-data/Dept_35-00103/35-00103_UOF-OIS-P_prepped.csv"
incidents = pd.read_csv(path)
incidents.head()

# In[ ]:


incidents["SUBJECT_INJURY_TYPE"].value_counts()

# Total Incidents reported : 

# In[ ]:


incidents.shape[0]

# ### Location of Incidents 

# In[ ]:


kmap = folium.Map([35.22, -80.89], height=400, zoom_start=10, tiles='CartoDB dark_matter')
for j, rown in incidents[1:].iterrows():
    if str(rown["LOCATION_LONGITUDE"]) != "nan":
        lon = float(rown["LOCATION_LATITUDE"])
        lat = float(rown["LOCATION_LONGITUDE"])
        folium.CircleMarker([lon, lat], radius=5, color='red', fill=True).add_to(kmap)
kmap

# Incidents by Race : Legend - 
# 
# Black : Black Person   
# Green : White Person  
# Yellow : Hispanic  
# Red : All Others  
# 

# In[ ]:


imap = folium.Map([35.22, -80.89], height=400, zoom_start=10, tiles='CartoDB positron')
for j, rown in incidents[1:].iterrows():
    if str(rown["LOCATION_LONGITUDE"]) != "nan":
        lon = float(rown["LOCATION_LATITUDE"])
        lat = float(rown["LOCATION_LONGITUDE"])
        
        if rown["SUBJECT_RACE"] == "Black":
            col = "black"
        elif rown["SUBJECT_RACE"]== "White":
            col = "green"
        elif rown["SUBJECT_RACE"]== "Hispanic":
            col = "yellow"
        else:
            col = "red"
                
        folium.CircleMarker([lon, lat], radius=5, color=col, fill=True).add_to(imap)    
imap

# Lets plot these incidents by gender

# In[ ]:


imap = folium.Map([35.22, -80.89], height=400, zoom_start=10, tiles='CartoDB positron')
for j, rown in incidents[1:].iterrows():
    if str(rown["LOCATION_LONGITUDE"]) != "nan":
        lon = float(rown["LOCATION_LATITUDE"])
        lat = float(rown["LOCATION_LONGITUDE"])
        
        if rown["SUBJECT_GENDER"] == "Male":
            col = "blue"
        else:
            col = "red"
                
        folium.CircleMarker([lon, lat], radius=5, color=col, fill=True).add_to(imap)        
imap

# Only one incident involves female, rest all others were males. 
# 
# Incidents by Subject Injury Types

# In[ ]:


imap = folium.Map([35.22, -80.89], height=400, zoom_start=10, tiles='CartoDB positron')
for j, rown in incidents[1:].iterrows():
    if str(rown["LOCATION_LONGITUDE"]) != "nan":
        lon = float(rown["LOCATION_LATITUDE"])
        lat = float(rown["LOCATION_LONGITUDE"])
        
        if rown["SUBJECT_INJURY_TYPE"] == "Non-Fatal Injury":
            col = "red"
        elif rown["SUBJECT_INJURY_TYPE"] == "Fatal Injury":
            col = "green"
        else:
            col = "blue"                
        folium.CircleMarker([lon, lat], radius=5, color=col, fill=True).add_to(imap)        
imap

# Lets locate the location of Police Offices as well

# In[ ]:


p2 = """../input/cpe-data/Dept_35-00103/35-00103_Shapefiles/CMPD_Police_Division_Offices.shp"""
One = gpd.read_file(p2) 
for j, rown in One.iterrows():
    lon = float(str(rown["geometry"]).split()[1].replace("(",""))
    lat = float(str(rown["geometry"]).split()[2].replace(")",""))
    folium.CircleMarker([lat, lon], radius=5, color='blue', fill=True).add_to(kmap)
kmap

# 
# ### Indianapolis Police Zones
# 
# Lets plot the shape file and related data 

# In[ ]:


p1 = """../input/cpe-data/Dept_23-00089/23-00089_Shapefiles/Indianapolis_Police_Zones.shp"""
One = gpd.read_file(p1)  
One.head()

# In[ ]:


mapa = folium.Map([39.81, -86.26060805912148], height=400, zoom_start=10, tiles='CartoDB dark_matter',API_key='wrobstory.map-12345678')
folium.GeoJson(One).add_to(mapa)
mapa 

# Lets plot the districts and juridiction realted with this shapefile data

# In[ ]:


f, ax = plt.subplots(1, figsize=(10, 8))
One.plot(column="DISTRICT", ax=ax, cmap='Accent',legend=True);
plt.title("Districts : Indianapolis Police Zones")
plt.show()

# In[ ]:


f, ax = plt.subplots(1, figsize=(10, 8))
One.plot(column="JURISDCTN", ax=ax, cmap='Accent', legend=True);
plt.title("JuriDiction : Indianapolis Police Zones")
plt.show()

# ### Bostan Police Districts

# In[ ]:


p3 = """../input/cpe-data/Dept_11-00091/11-00091_Shapefiles/boston_police_districts_f55.shp"""
One = gpd.read_file(p3)  
mapa = folium.Map([42.3, -71.0], height=400, zoom_start=10,  tiles='CartoDB dark_matter',API_key='wrobstory.map-12345678')
folium.GeoJson(One).add_to(mapa)
mapa 

# ### Dallas Districts

# In[ ]:


p4 = """../input/cpe-data/Dept_37-00049/37-00049_Shapefiles/EPIC.shp"""
One = gpd.read_file(p4)  
mapa = folium.Map([32.7, -96.7],zoom_start=10, height=400, tiles='CartoDB dark_matter',API_key='wrobstory.map-12345678')
folium.GeoJson(One).add_to(mapa)
mapa 

# ### Austin City 
# 
# Lets plot the incidents of Austin, Tx

# In[ ]:


p5 = "../input/cpe-data/Dept_37-00027/37-00027_UOF-P_2014-2016_prepped.csv"
dept_37_27_df = pd.read_csv(p5)[1:]
dept_37_27_df["INCIDENT_DATE"] = pd.to_datetime(dept_37_27_df["INCIDENT_DATE"]).astype(str)
dept_37_27_df["MonthDate"] = dept_37_27_df["INCIDENT_DATE"].apply(lambda x : x.split("-")[0] +'-'+ x.split("-")[1] + "-01")

tmp = dept_37_27_df.groupby("MonthDate").agg({"INCIDENT_REASON" : "count"}).reset_index()
tmp

import plotly.plotly as py
import plotly.graph_objs as go

trace1 = go.Scatter(x=tmp["MonthDate"], y=tmp.INCIDENT_REASON, name="Month wise Incidents")
# trace2 = go.Scatter(x=tmp["MonthDate"], y=tmp.INCIDENT_REASON)

data = [trace1]
layout = go.Layout(height=400, title="Incidents in Austin Texas")
fig = go.Figure(data, layout)
iplot(fig)

# In[ ]:


a = dept_37_27_df["SUBJECT_GENDER"].value_counts()
tr1 = go.Bar(x = a.index, y = a.values, name="Gender")

a = dept_37_27_df["INCIDENT_REASON"].value_counts()
tr2 = go.Bar(x = a.index, y = a.values, name="INCIDENT_REASON")

a = dept_37_27_df["SUBJECT_RACE"].value_counts()
tr3 = go.Bar(x = a.index, y = a.values, name="SUBJECT_RACE")


fig = tools.make_subplots(rows=1, cols=3, print_grid=False, subplot_titles=["Gender", "Incident Reason", "Subject Race"])
fig.append_trace(tr1, 1, 1);
fig.append_trace(tr2, 1, 2);
fig.append_trace(tr3, 1, 3);
fig['layout'].update(height=400, title="Austin Incidents Distribution", showlegend=False);
iplot(fig, filename='simple-subplot');

# In[ ]:


a = dept_37_27_df["REASON_FOR_FORCE"].value_counts()[:6]
tr1 = go.Bar(x = a.index, y = a.values, name="Gender")

a = dept_37_27_df["TYPE_OF_FORCE_USED1"].value_counts()[:8]
tr2 = go.Bar(x = a.index, y = a.values, name="INCIDENT_REASON")

fig = tools.make_subplots(rows=1, cols=2, print_grid=False, subplot_titles=["REASON_FOR_FORCE", "TYPE_OF_FORCE_USED1"])
fig.append_trace(tr1, 1, 1);
fig.append_trace(tr2, 1, 2);
fig['layout'].update(height=400, margin=dict(b=140), title="Austin Incidents Distribution", showlegend=False);
iplot(fig, filename='simple-subplot');

# The shape file is also given:

# In[ ]:


p5 = "../input/cpe-data/Dept_37-00027/37-00027_Shapefiles/APD_DIST.shp"
dept_37_27_shp = gpd.read_file(p5)
dept_37_27_shp.head()

# In[ ]:


f, ax = plt.subplots(1, figsize=(10, 12))
dept_37_27_shp.plot(column="SECTOR", ax=ax, cmap='Accent',legend=True);
plt.title("Sectors ")
plt.show()

# In[ ]:


f, ax = plt.subplots(1, figsize=(10, 12))
dept_37_27_shp.plot(column="PATROL_ARE", ax=ax, cmap='coolwarm',legend=True);
plt.title("Patrol Areas ")
plt.show()

# Lets try to map the multiple shape files / data together.  Taking the notes from @Chris [kernel](https://www.kaggle.com/crawford/another-world-famous-starter-kernel-by-chris) and @dsholes [kernel](https://www.kaggle.com/dsholes/confused-start-here), First we can create the GeoPandas dataframe from the normal dataframe, by converting the latlongs to POINTS. 

# In[ ]:


from shapely.geometry import Point

## remove na
notna = dept_37_27_df[['LOCATION_LATITUDE','LOCATION_LONGITUDE']].dropna().index
dept_37_27_df = dept_37_27_df.iloc[notna].reset_index(drop=True)
dept_37_27_df['coordinates'] = (dept_37_27_df.apply(lambda x: Point(float(x['LOCATION_LONGITUDE']), float(x['LOCATION_LATITUDE'])), axis=1))
dept_37_27_gdf = gpd.GeoDataFrame(dept_37_27_df, geometry='coordinates')

# ## make the corrdinate system same
dept_37_27_gdf.crs = {'init' :'epsg:4326'}
dept_37_27_shp.crs = {'init' :'esri:102739'}
dept_37_27_shp = dept_37_27_shp.to_crs(epsg='4326')

# Plot incidents by patrol areas, sectors etc

# In[ ]:


## plot
f, ax = plt.subplots(1, figsize=(10, 12))
dept_37_27_shp.plot(ax=ax, column='PATROL_ARE', cmap = "gray", legend=True)
dept_37_27_gdf.plot(ax=ax, marker='*', color='red', markersize=10)
plt.title("Incident Locations and Patrol Areas ")
plt.show()

# In[ ]:


## plot
f, ax = plt.subplots(1, figsize=(10, 12))
dept_37_27_shp.plot(ax=ax, column='SECTOR', cmap = "Oranges", legend=True)
dept_37_27_gdf.plot(ax=ax, marker='*', color='Black', markersize=10)
plt.title("Incident Locations and Sectors ")
plt.show()

# Great, now we have understanding about what kinds of dataset we have.  I have now shared my complete solution : 
# 
# ### <font color="#703bdb">Main Submission: </font>
# 
# <ul>
#     <li><a href="https://www.kaggle.com/shivamb/1-solution-workflow-science-of-policing-equity/">Part 1: Solution Workflow - The Science of Policing Equity </a>  </li>
#     <li><a href="https://www.kaggle.com/shivamb/2-automation-pipeline-integration-processing">Part 2: Data Integration and Processing : Automation Pipeline</a>  </li>
#     <li><a href="https://www.kaggle.com/shivamb/3-example-runs-of-automation-pipeline">Part 3: Example Runs of Automation Pipeline </a>  </li> 
#     <li><a href="https://www.kaggle.com/shivamb/4-1-analysis-report-minneapolis-24-00013">Part 4.1: Analysis Report - Measuring Equity - Minneapolis Police Department </a>   </li>
#     <li><a href="https://www.kaggle.com/shivamb/4-2-analysis-report-lapd-49-00033">Part 4.2: Analysis Report - Los Angles Police Department (49-00033) </a>   </li>
#     <li><a href="https://www.kaggle.com/shivamb/4-3-analysis-report-officer-level-analysis">Part 4.3: Analysis Report - Indianapolis Officer Level Analysis (23-00089) </a>   </li></ul>
# 
