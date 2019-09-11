#!/usr/bin/env python
# coding: utf-8

# **About Kiva:**
# [Kiva](https://www.kiva.org/about) is an international nonprofit, founded in 2005 and based in San Francisco, with a mission to connect people through lending to alleviate poverty.
# Kiva is in 83 countries, with about 2.7 Million borrowers. Kiva has funded around 1.11 Billion USD worth of loans. It also has around 450 volunteers worldwide. 
# 
# **Objective of the dataset:**
# Pair Kiva's data with additional data sources to estimate the welfare level of borrowers in specific regions, based on shared economic and demographic characteristics.
# 
# **Objective of the notebook:**
# To get a better understanding of the data provided by Kiva and also to discuss ideas for additional data sources. Most of the plots are done using plotly and so will be interactive. Please feel free to hover over the plots to get more insights.
# 
# Let us first import the necessary modules.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 999

# Let us list down the files present in this data and also take a look at the top few rows of each files.

# In[ ]:


from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# **kiva_loans.csv**
# 
# This is a subset of Kiva's data snapshots. This file has some of the loans given by Kiva. 

# In[ ]:


kiva_loans_df = pd.read_csv("../input/kiva_loans.csv")
kiva_loans_df.head()

# **kiva_mpi_region_locations.csv**
# 
# This file contains Kiva’s estimates as to the geolocation of subnational MPI regions

# In[ ]:


kiva_mpi_locations_df = pd.read_csv("../input/kiva_mpi_region_locations.csv")
kiva_mpi_locations_df.head()

# **loan_theme_ids.csv**
# 
# This file contains records from the Kiva Data Snapshot and can be matched to the loan theme regions to get a loan’s location.

# In[ ]:


loan_theme_ids_df = pd.read_csv("../input/loan_theme_ids.csv")
loan_theme_ids_df.head()

# **loan_themes_by_region.csv**
# 
# This file contains regional data related to loan themes and partner ids. They are Kiva’s estimates as to the various geolocations in which a loan theme has been offered, as well as the resulting estimate of which MPI Region(s) the loan theme is in.

# In[ ]:


loan_themes_by_region_df = pd.read_csv("../input/loan_themes_by_region.csv")
loan_themes_by_region_df.head()

# We can now look at the snapshot of the loans data to understand more about the problem.

# In[ ]:


kiva_loans_df.shape

# Now let us see the countrywise distribution of loans in the given snapshot data.

# In[ ]:


cnt_srs = kiva_loans_df['country'].value_counts().head(50)
trace = go.Bar(
    y=cnt_srs.index[::-1],
    x=cnt_srs.values[::-1],
    orientation = 'h',
    marker=dict(
        color=cnt_srs.values[::-1],
        colorscale = 'Viridis',
        reversescale = True
    ),
)

layout = go.Layout(
    title='Country wise distribution of loans',
    width=700,
    height=1000,
    )
data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename="CountryLoan")

# Philippines has more number of loans given by Kiva followed by Kenya and  El salvador. Now let us plot the same in world map.

# In[ ]:


con_df = pd.DataFrame(kiva_loans_df['country'].value_counts()).reset_index()
con_df.columns = ['country', 'num_loans']
con_df = con_df.reset_index().drop('index', axis=1)

#Find out more at https://plot.ly/python/choropleth-maps/
data = [ dict(
        type = 'choropleth',
        locations = con_df['country'],
        locationmode = 'country names',
        z = con_df['num_loans'],
        text = con_df['country'],
        colorscale = [[0,'rgb(255, 255, 255)'],[1,'rgb(56, 142, 60)']],
        autocolorscale = False,
        reversescale = False,
        marker = dict(
            line = dict (
                color = 'rgb(180,180,180)',
                width = 0.5
            ) ),
        colorbar = dict(
            autotick = False,
            tickprefix = '',
            title = 'Number of Loans'),
      ) ]

layout = dict(
    title = 'Number of loans by Country',
    geo = dict(
        showframe = False,
        showcoastlines = False,
        projection = dict(
            type = 'Mercator'
        )
    )
)

fig = dict( data=data, layout=layout )
py.iplot( fig, validate=False, filename='loans-world-map')

# **Sectorwise distribution of loans:**

# In[ ]:


cnt_srs = kiva_loans_df['sector'].value_counts().head(25)
trace = go.Bar(
    y=cnt_srs.index[::-1],
    x=cnt_srs.values[::-1],
    orientation = 'h',
    marker=dict(
        color=cnt_srs.values[::-1],
        colorscale = 'Rainbow',
        reversescale = True
    ),
)

layout = dict(
    title='Sector wise distribution of loans',
    )
data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename="SectorLoan")

# Sector wise, Agriculture has the highest number of loans followed by food and retail.  Now let us look at the loan details at activity level.

# In[ ]:


cnt_srs = kiva_loans_df['activity'].value_counts().head(25)
trace = go.Bar(
    y=cnt_srs.index[::-1],
    x=cnt_srs.values[::-1],
    orientation = 'h',
    marker=dict(
        color=cnt_srs.values[::-1],
        colorscale = 'Picnic',
        reversescale = True
    ),
)

layout = dict(
    title='Activity wise distribution of loans',
    )
data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename="ActivityLoan")

# **Loan Amount & Funded Amunt:**
# 
# Now let us look at the loan amount column to know about the distribution. First let us see if there are any outliers in the column by doing a scatter pliot.

# In[ ]:


plt.figure(figsize=(8,6))
plt.scatter(range(kiva_loans_df.shape[0]), np.sort(kiva_loans_df.loan_amount.values))
plt.xlabel('index', fontsize=12)
plt.ylabel('loan_amount', fontsize=12)
plt.title("Loan Amount Distribution")
plt.show()

# In[ ]:


plt.figure(figsize=(8,6))
plt.scatter(range(kiva_loans_df.shape[0]), np.sort(kiva_loans_df.funded_amount.values))
plt.xlabel('index', fontsize=12)
plt.ylabel('loan_amount', fontsize=12)
plt.title("Funded Amount Distribution")
plt.show()

# Looks like there is one loan worth 100K USD and it is funded too. Now let us truncate the extreme values and then do a histogram plot.

# In[ ]:


ulimit = np.percentile(kiva_loans_df.loan_amount.values, 99)
llimit = np.percentile(kiva_loans_df.loan_amount.values, 1)
kiva_loans_df['loan_amount_trunc'] = kiva_loans_df['loan_amount'].copy()
kiva_loans_df['loan_amount_trunc'].loc[kiva_loans_df['loan_amount']>ulimit] = ulimit
kiva_loans_df['loan_amount_trunc'].loc[kiva_loans_df['loan_amount']<llimit] = llimit

plt.figure(figsize=(12,8))
sns.distplot(kiva_loans_df.loan_amount_trunc.values, bins=50, kde=False)
plt.xlabel('loan_amount_trunc', fontsize=12)
plt.title("Loan Amount Histogram after outlier truncation")
plt.show()

# Loan amount is rightly skewed with majority of the loans falling under sub 1000 USD category.
# 
# **Repayment Term:**
# 

# In[ ]:


cnt_srs = kiva_loans_df.term_in_months.value_counts()

trace = go.Bar(
    x=cnt_srs.index,
    y=cnt_srs.values,
    marker=dict(
        color=cnt_srs.values,
        colorscale = 'Picnic',
        reversescale = True
    ),
)

layout = go.Layout(
    title='Repayment Term in Months'
)

data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename="RepaymentIntervals")

# Looks like "14 month term" loans are most availed followed by "8 month term". I expected that half yearly and yearly loans will be more but was wrong. 
# 
# Now let us look at the repayment_interval (which is Frequency at which lenders are scheduled to receive installments)

# In[ ]:


cnt_srs = kiva_loans_df.repayment_interval.value_counts()

trace = go.Bar(
    x=cnt_srs.index,
    y=cnt_srs.values,
    marker=dict(
        color=cnt_srs.values,
        colorscale = 'Rainbow',
        reversescale = True
    ),
)

layout = go.Layout(
    title='Repayment Interval of loans'
)

data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename="RepaymentIntervals")

# Monthly loans are higher followed by irregular loans. 
# 
# **Lender Count:**
# 
# We also have a veriable lender_count - Number of lenders contributing to loan. Looks like more than one person lends the loan. So we can take a look at this variable.

# In[ ]:


cnt_srs = kiva_loans_df.lender_count.value_counts()

trace = go.Bar(
    x=cnt_srs.index,
    y=cnt_srs.values,
    marker=dict(
        color=cnt_srs.values,
        colorscale = 'Portland',
        reversescale = True
    ),
)

layout = go.Layout(
    title='Lender Count'
)

data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename="LenderCount")

# Looks like there are few loans with more than 500 lenders and is highly skewed. So we can look only at the initial left side of the graph.

# In[ ]:


cnt_srs = kiva_loans_df.lender_count.value_counts().head(100)

trace = go.Bar(
    x=cnt_srs.index,
    y=cnt_srs.values,
    marker=dict(
        color=cnt_srs.values,
        colorscale = 'Portland',
        reversescale = True
    ),
)

layout = go.Layout(
    title='Lender Count Top 100'
)

data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename="LenderCount")

# Interestingly there are few loans with 0 lenders. We might have to look into them.
# 
# **Borrower Gender:**
# 
# Now let us look at the gender distribution of the borrowers.

# In[ ]:


olist = []
for ll in kiva_loans_df["borrower_genders"].values:
    if str(ll) != "nan":
        olist.extend( [l.strip() for l in ll.split(",")] )
temp_series = pd.Series(olist).value_counts()

labels = (np.array(temp_series.index))
sizes = (np.array((temp_series / temp_series.sum())*100))

trace = go.Pie(labels=labels, values=sizes)
layout = go.Layout(
    title='Borrower Gender'
)
data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename="BorrowerGender")

# Nearly 80% of the borrowers are female.

# More to come. Stay tuned.!
