#!/usr/bin/env python
# coding: utf-8

# In[ ]:


QUICK = False
import os
import datetime as dt
import pandas as pd
import numpy as np
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 999

# Visualization
from wordcloud import WordCloud
from wordcloud import STOPWORDS
from PIL import Image
import IPython.display as display
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
if not QUICK:
    from mpl_toolkits.basemap import Basemap, cm
plt.rcParams['figure.figsize'] = [16, 10]
plt.rcParams['font.size'] = 14
import seaborn as sns
sns.set_palette(sns.color_palette('tab20', 20))
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

# NLP
import string
import re
from nltk.stem.porter import *
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction import stop_words

# # Summary
# 
# [Kiva.org](https://www.kiva.org/) is an online crowdfunding platform to extend financial services to poor and financially excluded people around the world. Kiva lenders have provided over $1 billion dollars in loans to over 2 million people. In order to set investment priorities, help inform lenders, and understand their target communities, knowing the level of poverty of each borrower is critical. However, this requires inference based on a limited set of information for each borrower.
# 
# ### Objective
# The objective of the  [Data Science for Good challenge](https://www.kaggle.com/kiva/data-science-for-good-kiva-crowdfunding), is to  help Kiva to build more localized models to estimate the poverty/welfare levels of residents in the regions where Kiva has active loans.  Our main goal is to pair the provided data with additional data sources to estimate the welfare level of borrowers in specific regions, based on shared economic and demographic characteristics.
# A good solution would connect the features of each loan to one of several poverty mapping datasets, which indicate the average level of welfare in a region on as granular a level as possible.
# 
# 
# We already have quite a lot excellent EDA kernels for the original competition data set.
# 
# Here I will try to focus more on main objective to enhance the original dataset with addititional poverty/welfare.
# I have already uploaded a [Dataset](https://www.kaggle.com/gaborfodor/additional-kiva-snapshot)  with varying granularity data:
# 
# * **Loan level**: Additional information from kiva.org (more loans, detailed description, lenders and loan- lender connection)
# * **Country level**: Global Gridded Geographically Based Economic Data with (lat, lon) coords
# * **Country level**: Public statistics merged manually to fix country name differences. (Population, HDI, Population below Poverty)
# 
# Feel free to fork the kernel or use the dataset!
# 
# *To be cont'd...*

# In[ ]:


start = dt.datetime.now()
display.Image(filename='../input/additional-kiva-snapshot/cover.png', width=800) 

# You can add new datasets to your kernel and read them from subdirectories.
# 
# Please note that after adding a new dataset the original competition data directory changes from *../input/* to *../input/data-science-for-good-kiva-crowdfunding/*.

# In[ ]:


competition_data_dir = '../input/data-science-for-good-kiva-crowdfunding/'
additional_data_dir = '../input/additional-kiva-snapshot/'
os.listdir(competition_data_dir)
os.listdir(additional_data_dir)

# ## Loans
# 
# It has more rows (1.4 M) and more columns. It is easy to join to the original *kiva_loans.csv*.
# 
# Some of the new columns has different name but the same content (e.g. activity and activity_name)

# In[ ]:


if QUICK:
    loans = pd.read_csv(additional_data_dir + 'loans.csv', nrows=10**5)
else:
    loans = pd.read_csv(additional_data_dir + 'loans.csv')
kiva_loans = pd.read_csv(competition_data_dir + 'kiva_loans.csv')
merged_loans = pd.merge(kiva_loans, loans, how='left', left_on='id', right_on='loan_id')

print('Loans provided for the challenge: {}'.format(kiva_loans.shape))
print('Loans from the additional snapshot: {}'.format(loans.shape))
print('Match ratio {:.3f}%'.format(100 * merged_loans.loan_id.count() / merged_loans.id.count()))

loans.head(2)

# ## Lenders
# More than 2M lenders. You need to work with lots of missing values.

# In[ ]:


if QUICK:
    lenders = pd.read_csv(additional_data_dir + 'lenders.csv', nrows=10**5)
else:
    lenders = pd.read_csv(additional_data_dir + 'lenders.csv')
lenders.head(5)
lenders.shape

# ## Loans - Lenders
# Connections between loans and lenders. It probably does not help directly the goal of the competition. Though it allows to use Network Analysis or try out Recommendation Systems.

# In[ ]:


loans_lenders = pd.read_csv(additional_data_dir + 'loans_lenders.csv')
loans_lenders.head(2)
loans_lenders.count()
loans_lenders.shape

# ## Free text fields available for NLP
# 
# We have descriptions for each loan. Most of them are in English some of the different languages have translated versions as well. The original competion set already has gender parsing these descriptions you could add other demographic features (e.g. age, marital status, number of children, household size, etc. ) for most of the loans.
# 
# Some of the lenders have provided reason why do they provide loans. While it is not essential given the goal of the competition it might be interesting.

# In[ ]:


stop = set(stopwords.words('english'))
def tokenize(text):
    try: 
        regex = re.compile('[' + re.escape(string.punctuation) + '0-9\\r\\t\\n]')
        text = regex.sub(" ", text)
        tokens_ = [word_tokenize(s) for s in sent_tokenize(text)]
        tokens = []
        for token_by_sent in tokens_:
            tokens += token_by_sent
        tokens = list(filter(lambda t: t.lower() not in stop, tokens))
        filtered_tokens = [w for w in tokens if re.search('[a-zA-Z]', w)]
        filtered_tokens = [w.lower() for w in filtered_tokens if len(w)>=3]
        return " ".join(filtered_tokens)
    except TypeError as e: print(text,e)

# In[ ]:


lenders_reason = lenders[~pd.isnull(lenders['loan_because'])][['loan_because']]
if QUICK:
    lenders_reason = lenders_reason.sample(frac=0.1)
lenders_reason['tokens'] = lenders_reason['loan_because'].map(tokenize)
lenders_reason_string = " ".join(lenders_reason.tokens.values)
lenders_reason_wc = WordCloud(background_color='white', max_words=2000, width=3200, height=2000)
_ = lenders_reason_wc.generate(lenders_reason_string)

lenders_reason.head()
lenders_reason.shape

# In[ ]:


fig, ax = plt.subplots(figsize=(16, 10))
plt.imshow(lenders_reason_wc)
plt.axis("off")
plt.title('Reason to give loan', fontsize=24)
plt.show();

# In[ ]:


loan_descriptions = loans[loans['original_language'] == 'English'][['description']]
loan_descriptions = loan_descriptions[~pd.isnull(loan_descriptions['description'])]
loan_description_sample = loan_descriptions.sample(frac=0.1)

loan_description_sample['tokens'] = loan_description_sample['description'].map(tokenize)
loan_description_string = " ".join(loan_description_sample.tokens.values)
loan_description_wc = WordCloud(background_color='white', max_words=2000, width=3200, height=2000)
_ = loan_description_wc.generate(loan_description_string)

print(loan_descriptions.shape, loan_description_sample.shape)
loan_description_sample.head()

# In[ ]:


fig, ax = plt.subplots(figsize=(16, 10))
plt.imshow(loan_description_wc)
plt.axis("off")
plt.title('Loan description', fontsize=24)
plt.show();

# # Country level statistics
# 
# We can merge  poverty, HDI statistics on country level. With the additional snapshot we can show the top countries based on the number of loans and lenders as well.
# 

# In[ ]:


country_stats = pd.read_csv(additional_data_dir + 'country_stats.csv')
loan_country_cnt = loans.groupby(['country_code']).count()[['loan_id']].reset_index()
loan_country_cnt.columns = ['country_code', 'loan_cnt']
loan_country_cnt = loan_country_cnt.sort_values(by='loan_cnt', ascending=False)
loan_country_cnt.head()

lender_country_cnt = lenders.groupby(['country_code']).count()[['permanent_name']].reset_index()
lender_country_cnt.columns = ['country_code', 'lender_cnt']
lender_country_cnt = lender_country_cnt.sort_values(by='lender_cnt', ascending=False)
lender_country_cnt.head()

country_count = pd.merge(loan_country_cnt, lender_country_cnt, how='outer', on='country_code')
country_count = country_count.merge(country_stats[['country_code']])
threshold = 10
country_count.loc[country_count.loan_cnt < threshold, 'loan_cnt'] = 0
country_count.loc[country_count.lender_cnt < threshold, 'lender_cnt'] = 0
country_count = country_count.fillna(0)


# In[ ]:


country_stats = pd.read_csv(additional_data_dir + 'country_stats.csv')
country_stats = pd.merge(country_count, country_stats, how='inner', on='country_code')
country_stats['population_in_poverty'] = country_stats['population'] * country_stats['population_below_poverty_line'] / 100.

country_stats.shape
country_stats.head()

# In[ ]:


data = [dict(
        type='choropleth',
        locations=country_stats['country_name'],
        locationmode='country names',
        z=np.log10(country_stats['loan_cnt'] + 1),
        text=country_stats['country_name'],
        colorscale='Reds',
        reversescale=False,
        marker=dict(line=dict(color='rgb(180,180,180)', width=0.5)),
        colorbar=dict(autotick=False, tickprefix='', title='Loans'),
)]
layout = dict(
    title = 'Number of loans by Country',
    geo = dict(showframe=False, showcoastlines=True, projection=dict(type='Mercator'))
)
fig = dict(data=data, layout=layout)
py.iplot(fig, validate=False, filename='loans-world-map')

# In[ ]:


data = [dict(
        type='choropleth',
        locations=country_stats['country_name'],
        locationmode='country names',
        z=np.log10(country_stats['lender_cnt'] + 1),
        text=country_stats['country_name'],
        colorscale='Greens',
        reversescale=True,
        marker=dict(line=dict(color='rgb(180,180,180)', width=0.5)),
        colorbar=dict(autotick=False, tickprefix='', title='Lenders'),
)]
layout = dict(
    title = 'Number of lenders by Country',
    geo = dict(showframe=False, showcoastlines=True, projection=dict(type='Mercator'))
)
fig = dict(data=data, layout=layout)
py.iplot(fig, validate=False, filename='lenders-world-map')

# In[ ]:


data = [dict(
        type='choropleth',
        locations=country_stats['country_name'],
        locationmode='country names',
        z=country_stats['hdi'],
        text=country_stats['country_name'],
        colorscale='Portland',
        reversescale=True,
        marker=dict(line=dict(color='rgb(180,180,180)', width=0.5)),
        colorbar=dict(autotick=False, tickprefix='', title='HDI'),
)]
layout = dict(
    title = 'Human Development Index',
    geo = dict(showframe=False, showcoastlines=True, projection=dict(type='Mercator'))
)
fig = dict(data=data, layout=layout)
py.iplot(fig, validate=False, filename='hdi-world-map')

# In[ ]:


largest_countries = country_stats.sort_values(by='population', ascending=False).copy()[:30]

data = [go.Scatter(
    y = largest_countries['hdi'],
    x = largest_countries['population_below_poverty_line'],
    mode='markers+text',
    marker=dict(
        sizemode = 'diameter',
        sizeref = 1,
        size= 2 * (np.log(largest_countries.population) - 10),
        color=largest_countries['hdi'],
        colorscale='Portland',
        reversescale=True,
        showscale=True)
    ,text=largest_countries['country_name']
    ,textposition=["top center"]
)]
layout = go.Layout(
    autosize=True,
    title='Poverty vs. HDI',
    hovermode='closest',
    xaxis= dict(title='Poverty%', ticklen= 5, showgrid=False, zeroline=False, showline=False),
    yaxis=dict(title='HDI', showgrid=False, zeroline=False, ticklen=5, gridwidth=2)
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='scatter_hdi_poverty')

#  The following plot shows the poverty population and the loan counts in each country.  The color indicatess the HDI.

# In[ ]:


kiva_loan_country_stats = country_stats[country_stats['loan_cnt'] > 0]
data = [go.Scatter(
    y = np.log10(kiva_loan_country_stats['loan_cnt'] + 1),
    x = np.log10(kiva_loan_country_stats['population_in_poverty'] + 1),
    mode='markers',
    marker=dict(
        sizemode = 'diameter',
        sizeref = 1,
        size= 5 * (np.log(kiva_loan_country_stats.population) - 10),
        color = kiva_loan_country_stats['hdi'],
        colorscale='Portland',
        reversescale=True,
        showscale=True)
    ,text=kiva_loan_country_stats['country_name']
)]
layout = go.Layout(
    autosize=True,
    title='Population in poverty vs. Kiva.org loan count',
    hovermode='closest',
    xaxis= dict(title='Population in poverty (log10 scale)', ticklen= 5, showgrid=False, zeroline=False, showline=False),
    yaxis=dict(title='Loan count (log10 scale)',showgrid=False, zeroline=False, ticklen=5, gridwidth=2),
    showlegend=False
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='scatter_countries')

# ### Countries without kiva loans
# 
# 

# In[ ]:


country_stats_wo_kiva_loans = country_stats[np.logical_and(country_stats['loan_cnt'] == 0, country_stats['hdi'] < 0.8)]
country_stats_wo_kiva_loans = country_stats_wo_kiva_loans.sort_values(by='population_in_poverty', ascending=False)
country_stats_wo_kiva_loans = country_stats_wo_kiva_loans[['country_name', 'population', 'population_below_poverty_line', 'population_in_poverty', 'hdi', 'life_expectancy', 'gni']]
country_stats_wo_kiva_loans.head(20)

# # Regional Economic Data

# In[ ]:


gecon = pd.read_csv(additional_data_dir + 'GEconV4.csv', sep=';')

gecon['MER1990_40'] = pd.to_numeric(gecon['MER1990_40'], errors='coerce')
gecon['MER1995_40'] = pd.to_numeric(gecon['MER1995_40'], errors='coerce')
gecon['MER2000_40'] = pd.to_numeric(gecon['MER2000_40'], errors='coerce')
gecon['MER2005_40'] = pd.to_numeric(gecon['MER2005_40'], errors='coerce')

gecon['PPP1990_40'] = pd.to_numeric(gecon['PPP1990_40'], errors='coerce')
gecon['PPP1995_40'] = pd.to_numeric(gecon['PPP1995_40'], errors='coerce')
gecon['PPP2000_40'] = pd.to_numeric(gecon['PPP2000_40'], errors='coerce')
gecon['PPP2005_40'] = pd.to_numeric(gecon['PPP2005_40'], errors='coerce')
gecon = gecon.dropna()
gecon['GCP'] = gecon['PPP2005_40'] / (gecon['POPGPW_2005_40'] + 1) * 10**6

gecon.head()
gecon.describe()

# In[ ]:


fig, ax = plt.subplots(figsize=(16, 8))
if not QUICK:
    corners = {'llcrnrlat': -60, 'urcrnrlat': 60, 'llcrnrlon': -140, 'urcrnrlon': 160}
    m = Basemap(projection='merc', resolution='h', **corners)
    m.fillcontinents(color='white', lake_color=plt.cm.Blues(0.5), alpha=0.3)
    m.drawcoastlines(linewidth=0.5, color='k')
    m.drawcountries(linewidth=0.5, color='k')
    sc = m.scatter(gecon['LONGITUDE'].values, gecon['LAT'].values, c=np.clip(gecon['GCP'].values, 0, 30),
                   s=15, marker='o', cmap=plt.cm.magma, lw=0, alpha=1., latlon=True)
    m.drawparallels(np.arange(corners['llcrnrlat'], corners['urcrnrlat'], 10), labels=[1, 1, 0, 0],
                    linewidth=0.5, dashes=[2, 2], alpha=0.3)
    m.drawmeridians(np.arange(corners['llcrnrlon'], corners['urcrnrlon'], 20), labels=[0, 0, 1, 1],
                    linewidth=0.5, dashes=[2, 2], alpha=0.3)
    plt.tight_layout(pad=3.0)
    plt.colorbar(sc)
    plt.suptitle('Regional GDP per capita (2005 Thousand US$ purchase power parity)')
plt.show();

# In[ ]:


end =  dt.datetime.now()
print('Total time {} sec'.format((end - start).seconds))
