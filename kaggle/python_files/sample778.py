#!/usr/bin/env python
# coding: utf-8

# ![](https://i.imgur.com/dsg47jV.jpg)
# 
# # Introduction
# > About this notebook: In this notebook, I will be analyzing Kiva Crowdfunding dataset. I will try to find helpful information on this dataset. If you find something wrong or have any suggestion feel free to reach me at the comment. And don't forget to upvote. I will continue adding new information, so please visit frequentrly.
# 
# ## Table of contents
# 1. [Popular loan sector](#popular_loan_sector)
# 2. [Loan due](#loan_due)
# 3. [Gender combination](#gender_cobination)
# 4. [Usecase of loan](#usecase_of_loan)
# 5. [Who needs help ](#who_needs_help)
# 
# 
# ### 6. [Kiva challenge solution](#kiva_challenge_solution)

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt 
plt.style.use('fivethirtyeight')
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS # this module is for making wordcloud in python

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# import plotly
import plotly
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.tools as tls
import plotly.graph_objs as go
import plotly.tools as tls
import plotly.figure_factory as fig_fact
plotly.tools.set_config_file(world_readable=True, sharing='public')


import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

# In[2]:


df_kiva_loans = pd.read_csv('../input/data-science-for-good-kiva-crowdfunding/kiva_loans.csv')
df_kiva_region_location = pd.read_csv('../input/data-science-for-good-kiva-crowdfunding/kiva_mpi_region_locations.csv')
df_kiva_loans.head()

# # TL;DR

# In[3]:


df_kiva_loans['due_loan'] = df_kiva_loans['loan_amount'] - df_kiva_loans['funded_amount']
df_kiva_loans['country_iso_3'] = df_kiva_loans['country'].map(pd.DataFrame(df_kiva_region_location.groupby(['country','ISO']).size()).reset_index().drop(0,axis=1).set_index('country')['ISO'])

# In[4]:


plot_df_country_popular_loan = pd.DataFrame(df_kiva_loans.groupby(['country','country_iso_3'])['loan_amount', 'funded_amount'].mean()).reset_index()


# In[5]:


plot_df_country_popular_due = pd.DataFrame(df_kiva_loans.groupby(['country','country_iso_3'])['due_loan'].max()).reset_index()
plot_df_country_popular_due = plot_df_country_popular_due[plot_df_country_popular_due['due_loan'] > 0]  
plot_df_country_popular_loan['due_loan'] = plot_df_country_popular_loan['country_iso_3'].map(plot_df_country_popular_due.set_index('country_iso_3')['due_loan'])
plot_df_country_popular_loan = plot_df_country_popular_loan.fillna(0)

# In[6]:


scl = [[0.0, 'rgb(242,240,247)'],[0.2, 'rgb(218,218,235)'],[0.4, 'rgb(188,189,220)'],[0.6, 'rgb(158,154,200)'],[0.8, 'rgb(117,107,177)'],[1.0, 'rgb(84,39,143)']]


data = [ dict(
        type='choropleth',
        colorscale = scl,
        autocolorscale = False,
        locations = plot_df_country_popular_loan['country_iso_3'],
        z = plot_df_country_popular_loan['loan_amount'].astype(float),
        text =  'Country name: ' + plot_df_country_popular_loan['country'] +'</br>' + 'Average loan amount: ' + plot_df_country_popular_loan['loan_amount'].astype(str) \
    + '</br>' + 'Average funded_amount: ' + plot_df_country_popular_loan['funded_amount'].astype(str) + '</br>' \
     + 'Loan due: ' + plot_df_country_popular_loan['due_loan'].astype(str),
        marker = dict(
            line = dict (
                color = 'rgb(255,255,255)',
                width = 2
            ) ),
        colorbar = dict(
            title = "Average loan amount in dollar")
        ) ]

layout = dict(
        title = 'Average loan amount taken by different country<br>(Hover for breakdown)',
        geo = dict(
            projection=dict( type='orthographic' ),
            showlakes = True,
            lakecolor = 'rgb(255, 255, 255)'),
             )




    
fig = dict( data=data, layout=layout )

py.iplot( fig, filename='d3-cloropleth-map' )

# <a id="popular_loan_sector"></a>
# ## 1. Popular loan sector
# > In this section, I will be analyzing which loan sector and activity take more loan. 

# In[7]:


plot_df_sector_popular_loan = pd.DataFrame(df_kiva_loans.groupby(['sector'])['loan_amount'].mean()).reset_index()
plt.subplots(figsize=(15,7))
sns.barplot(x='sector',y='loan_amount',data=plot_df_sector_popular_loan,palette='RdYlGn_r',edgecolor=sns.color_palette('dark',7))
plt.ylabel('Average loan amount in Dollar', fontsize=20)
plt.xticks(rotation=90,fontsize=20)
plt.xlabel('Loan sector', fontsize=20)
plt.title('Popular loan sector in terms of loan amount', fontsize=24)
plt.savefig('popular_loan_amount_sector.png')
plt.show()

# It looks like Entertainment sector is popular for taking large amount of loan!

# In[8]:


plot_df_sector_popular_loan = pd.DataFrame(df_kiva_loans.groupby(['activity'])['loan_amount'].mean().sort_values(ascending=False)[:20]).reset_index()
plt.subplots(figsize=(15,7))
sns.barplot(x='activity',y='loan_amount',data=plot_df_sector_popular_loan,palette='RdYlGn_r',edgecolor=sns.color_palette('dark',7))
plt.ylabel('Average loan amount in Dollar', fontsize=20)
plt.xticks(rotation=90,fontsize=20)
plt.xlabel('Loan sector', fontsize=20)
plt.title('Popular loan activity in terms of loan amount', fontsize=24)
# plt.savefig('ave_ozone.png')
plt.show()

# The plot shows us that Technology, Gardening, communication is most popular activity that takes large amount of loan. 

# <a id="loan_due"></a>
# 
# # 2. Loan due
# > In this section, I will to find which country, gender, sector have most due loan. 

# In[9]:


plot_df_due_loan = pd.DataFrame(df_kiva_loans.groupby(['country'])['due_loan'].max().sort_values(ascending=False)[:20]).reset_index()
plot_df_due_loan = plot_df_due_loan[plot_df_due_loan['due_loan'] > 0]  
plot_df_due_loan['country'] = plot_df_due_loan['country'].replace('The Democratic Republic of the Congo', 'Congo')
plt.subplots(figsize=(15,7))
sns.barplot(x='country',y='due_loan',data=plot_df_due_loan,palette='RdYlGn_r',edgecolor=sns.color_palette('dark',7))
plt.ylabel('Due loan($)', fontsize=20)
plt.xticks(rotation=90,fontsize=20)
plt.xlabel('Country', fontsize=20)
plt.title('Country with highest due loan', fontsize=24)
# plt.savefig('ave_ozone.png')
plt.show()

# Haiti, Mexico and Peru has most due loans. 

# In[10]:


plot_df_due_loan = pd.DataFrame(df_kiva_loans.groupby(['sector'])['due_loan'].max().sort_values(ascending=False)[:20]).reset_index()
plot_df_due_loan = plot_df_due_loan[plot_df_due_loan['due_loan'] > 0]  
plt.subplots(figsize=(15,7))
sns.barplot(x='sector',y='due_loan',data=plot_df_due_loan,palette='RdYlGn_r',edgecolor=sns.color_palette('dark',7))
plt.ylabel('Due loan($)', fontsize=20)
plt.xticks(rotation=90,fontsize=20)
plt.xlabel('Loan sector', fontsize=20)
plt.title('Popular loan sector with highest due loan', fontsize=24)
# plt.savefig('ave_ozone.png')
plt.show()

# It looks like Wholesale, Transportation and Food sector has highest due loan. 

# <a id="gender_cobination"></a>
# 
# 
# # 3. Gender combination
# > Some time loan only taken by one woman/man or sometime woman and man together take lone. Let's see what's going on in data set

# In[11]:


from collections import Counter
def count_word(x):
    y = Counter(x.split(', '))
    return y
df_kiva_loans_without_null_gender = df_kiva_loans.dropna(subset = ['borrower_genders'])
plot_df_gender = pd.DataFrame.from_dict(df_kiva_loans_without_null_gender['borrower_genders'].apply(lambda x: count_word(x)))
plot_df_gender['borrower_genders'] = plot_df_gender['borrower_genders'].astype(str).replace({'Counter':''}, regex=True)

# In[12]:


plot_df_gender = pd.DataFrame(plot_df_gender['borrower_genders'].value_counts()[:10]).reset_index()
plt.subplots(figsize=(15,7))
sns.barplot(x='index',y='borrower_genders',data=plot_df_gender,palette='RdYlGn_r',edgecolor=sns.color_palette('dark',7))
plt.ylabel('Gender count', fontsize=20)
plt.xticks(rotation=90,fontsize=20)
plt.xlabel('Gender combinations', fontsize=20)
plt.title('Popular gender combinations', fontsize=24)
# plt.savefig('ave_ozone.png')
plt.show()

# It's look like most of the time female alone take loan!

# 

# <a id="usecase_of_loan"></a>
# # 4. Usecase of loan
# > In this section, I will create a wordcloud of usecase described by the borrower when taking loan. 

# In[13]:


wc = WordCloud(width=1600, height=800, random_state=1,max_words=200000000)
# generate word cloud using df_yelp_tip_top['text_clear']
wc.generate(str(df_kiva_loans['use']))
# declare our figure 
plt.figure(figsize=(20,10), facecolor='k')
# add title to the graph
plt.title("Usecase of loan", fontsize=40,color='white')
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.tight_layout(pad=10)
# after lot of congiguration finally plot the graph
# plt.savefig('word.png', dpi=900)
plt.show()

# It's look like most of the people take loan for purchase something!

# <a id="who_needs_help"></a>
# 
# # 5. Let's see who needs help more! 
# > In this section, I will analyze Kiva's dataset with other dataset and try to find some informative information. Like where Kiva needs to provide more loans etc. 

# In[14]:


df_youth = pd.read_csv('../input/youth-unemployment-gdp-and-literacy-percentage/youth.csv', sep='\s*,\s*',engine='python')
df_youth['country'] = df_youth['country'].replace('Congo (Democratic Republic)', 'The Democratic Republic of the Congo')
df_youth['country'] = df_youth['country'].replace('United States of America', 'United States')
df_youth['country'] = df_youth['country'].replace('Palestinian Territories', 'Palestine')
df_youth['country'] = df_youth['country'].replace('East Timor', 'Timor-Leste')
df_youth['country'] = df_youth['country'].replace('East Timor', 'Timor-Leste')
df_youth['country'] = df_youth['country'].replace('Laos', "Lao People's Democratic Republic")
df_youth['country'] = df_youth['country'].replace('Congo (Republic)', "Congo")
df_youth['country'] = df_youth['country'].replace('Virgin Islands of the U.S.', "Virgin Islands")


# In[15]:


df_youth = df_youth.set_index('country')
df_youth.index.names = [None]
df_kiva_loans['youth'] = df_kiva_loans['country'].map(df_youth['youth_percentage'])

# In[16]:


plot_df_top_youth_country = pd.DataFrame(df_kiva_loans.groupby('country')['youth'].mean()).reset_index().nlargest(20, 'youth')
plot_df_pop_country = pd.DataFrame(df_kiva_loans.country.value_counts().reset_index()).nlargest(20, 'country')
fig, axs = plt.subplots(figsize=(15,10), ncols=2)
# plt.subplots_adjust(right=0.9)
ax1 = sns.barplot(y='country',x='youth',data=plot_df_top_youth_country,palette='RdYlGn_r',edgecolor=sns.color_palette('dark',7), ax=axs[0])
ax2 = sns.barplot(y='index', x = 'country', data=plot_df_pop_country, ax=axs[1])
ax1.invert_xaxis()
ax1.set_xlabel('Youth percentage', fontsize=20)
for tick in ax1.get_xticklabels():
    tick.set_rotation(90)
    tick.set_fontsize(20)
# ax1.set_r(rotation=90,fontsize=20)
ax1.set_ylabel('Country', fontsize=20)
ax1.set_title('Country with highest youth percentage', fontsize=24)


ax2.set_xlabel('Number of loans', fontsize=20)
for tick in ax2.get_xticklabels():
    tick.set_rotation(90)
    tick.set_fontsize(20)
# ax1.set_r(rotation=90,fontsize=20)
ax2.set_ylabel('')
ax2.set_title('Country with highest number of loan', fontsize=24)
# plt.savefig('ave_ozone.png')
plt.show()

# **Takeaways from the plot:**
# > The left plot shows us the countries with the highest percentage. Youth are the main backbone of a country. If you can empower the youth the country can prosper. Unemployment is the disaster for a country. Because Kiva wants to improve people lives, it needs to invest in those countries with the highest percentage of youth. On the right side plot, it shows us the countries with the highest number of loans. It clearly shows the gap between the popular country with youth percentage and popular country with the number of loans. So Kiva should increase the number of loans in those countries with the highest percentage of youth. 

# In[17]:


df_unemployment = pd.read_csv('../input/youth-unemployment-gdp-and-literacy-percentage/unemployment.csv', sep='\s*,\s*',engine='python')
df_unemployment['country'] = df_unemployment['country'].replace('Congo (Democratic Republic)', 'The Democratic Republic of the Congo')
df_unemployment['country'] = df_unemployment['country'].replace('United States of America', 'United States')
df_unemployment['country'] = df_unemployment['country'].replace('Palestinian Territories', 'Palestine')
df_unemployment['country'] = df_unemployment['country'].replace('East Timor', 'Timor-Leste')
df_unemployment['country'] = df_unemployment['country'].replace('East Timor', 'Timor-Leste')
df_unemployment['country'] = df_unemployment['country'].replace('Laos', "Lao People's Democratic Republic")
df_unemployment['country'] = df_unemployment['country'].replace('Congo (Republic)', "Congo")
df_unemployment['country'] = df_unemployment['country'].replace('Virgin Islands of the U.S.', "Virgin Islands")

# In[18]:


df_unemployment = df_unemployment.set_index('country')
df_unemployment.index.names = [None]
df_kiva_loans['unemployment'] = df_kiva_loans['country'].map(df_unemployment['unemployment_percentage'])

# In[19]:


plot_df_top_unemployment_country = pd.DataFrame(df_kiva_loans.groupby('country')['unemployment'].mean()).reset_index().nlargest(20, 'unemployment')
plot_df_pop_country = pd.DataFrame(df_kiva_loans.country.value_counts().reset_index()).nlargest(20, 'country')
fig, axs = plt.subplots(figsize=(15,10), ncols=2)
# plt.subplots_adjust(right=0.9)
ax1 = sns.barplot(y='country',x='unemployment',data=plot_df_top_unemployment_country,palette='RdYlGn_r',edgecolor=sns.color_palette('dark',7), ax=axs[0])
ax2 = sns.barplot(y='index', x = 'country', data=plot_df_pop_country, ax=axs[1])
ax1.invert_xaxis()
ax1.set_xlabel('Unemployment percentage', fontsize=20)
for tick in ax1.get_xticklabels():
    tick.set_rotation(90)
    tick.set_fontsize(20)
# ax1.set_r(rotation=90,fontsize=20)
ax1.set_ylabel('Country', fontsize=20)
ax1.set_title('Country with highest unemployment rate', fontsize=24)


ax2.set_xlabel('Number of loans', fontsize=20)
for tick in ax2.get_xticklabels():
    tick.set_rotation(90)
    tick.set_fontsize(20)
# ax1.set_r(rotation=90,fontsize=20)
ax2.set_ylabel('')
ax2.set_title('Country with highest number of loan', fontsize=24)
# plt.savefig('unemployment.png')
plt.show()

# **Takeaways from the plot: **
# > If Kiva wants to help countries for fighting unemployment Kiva should focus on those countries with highest unemployment countries. The graph shows us that those countries with highest unemployment rate have really less number of loans compare to other countries. So Kiva should focus on this matter. 

# In[20]:


df_literacy = pd.read_csv('../input/youth-unemployment-gdp-and-literacy-percentage/literacy_rate.csv', sep='\s*,\s*',engine='python')
df_literacy['country'] = df_literacy['country'].replace('Congo (Democratic Republic)', 'The Democratic Republic of the Congo')
df_literacy['country'] = df_literacy['country'].replace('United States of America', 'United States')
df_literacy['country'] = df_literacy['country'].replace('Palestinian Territories', 'Palestine')
df_literacy['country'] = df_literacy['country'].replace('East Timor', 'Timor-Leste')
df_literacy['country'] = df_literacy['country'].replace('East Timor', 'Timor-Leste')
df_literacy['country'] = df_literacy['country'].replace('Laos', "Lao People's Democratic Republic")
df_literacy['country'] = df_literacy['country'].replace('Congo (Republic)', "Congo")
df_literacy['country'] = df_literacy['country'].replace('Virgin Islands of the U.S.', "Virgin Islands")

df_literacy = df_literacy.set_index('country')
df_literacy.index.names = [None]
df_kiva_loans['literacy'] = df_kiva_loans['country'].map(df_literacy['literacy_rate_percent_all'])

# In[21]:


df_kiva_loans.literacy = df_kiva_loans.literacy.astype(float)
plot_df_top_literacy_country = pd.DataFrame(df_kiva_loans.groupby('country')['literacy'].mean()).reset_index().nsmallest(20, 'literacy')
plot_df_top_literacy_country['literacy'] = plot_df_top_literacy_country.literacy.astype(float)
plot_df_pop_country = pd.DataFrame(df_kiva_loans.country.value_counts().reset_index()).nlargest(20, 'country')
fig, axs = plt.subplots(figsize=(15,10), ncols=2)
# plt.subplots_adjust(right=0.9)
ax1 = sns.barplot(y='country',x='literacy',data=plot_df_top_literacy_country,palette='RdYlGn_r',edgecolor=sns.color_palette('dark',7), ax=axs[0])
ax2 = sns.barplot(y='index', x = 'country', data=plot_df_pop_country, ax=axs[1])
ax1.invert_xaxis()
ax1.set_xlabel('Literacy percentage', fontsize=20)
for tick in ax1.get_xticklabels():
    tick.set_rotation(90)
    tick.set_fontsize(20)
# ax1.set_r(rotation=90,fontsize=20)
ax1.set_ylabel('Country', fontsize=20)
ax1.set_title('Country with lowest literacy rate', fontsize=24)


ax2.set_xlabel('Number of loans', fontsize=20)
for tick in ax2.get_xticklabels():
    tick.set_rotation(90)
    tick.set_fontsize(20)
# ax1.set_r(rotation=90,fontsize=20)
ax2.set_ylabel('')
ax2.set_title('Country with highest number of loan', fontsize=24)
plt.show()

# **Takeaways from the plot: **
# > On the left side it shows the countries with lowest literacy rate and right side with the highest number of loans. Those countries with the lowest number of literacy need more help compare to other countries. So Kiva should consider this.  

# <a id="loan_themes"></a>
# # 6. Loan themes
# > In this section, I will analyse what is popular loan theme like education, agriculture etc. 

# In[22]:


df_loan_theme = pd.read_csv('../input/data-science-for-good-kiva-crowdfunding/loan_themes_by_region.csv')

# In[23]:


plot_df_loan_theme = df_loan_theme['Loan Theme Type'].value_counts()[:10].reset_index()
plt.subplots(figsize=(15,7))
sns.barplot(x='index',y='Loan Theme Type',data=plot_df_loan_theme,palette='RdYlGn_r',edgecolor=sns.color_palette('dark',7))
plt.ylabel('Loan theme count', fontsize=20)
plt.xticks(rotation=90,fontsize=20)
plt.xlabel('Loan theme names', fontsize=20)
plt.title('Popular loan theme', fontsize=24)
# plt.savefig('ave_ozone.png')
plt.show()

# In[24]:


plot_df_sector_popular_loan_by_amount = pd.DataFrame(df_loan_theme.groupby(['Loan Theme Type'])['amount'].mean().sort_values(ascending=False)[:10]).reset_index()
plt.subplots(figsize=(15,7))
sns.barplot(y='Loan Theme Type',x='amount',data=plot_df_sector_popular_loan_by_amount,palette='RdYlGn_r',edgecolor=sns.color_palette('dark',7))
plt.ylabel('Average loan amount in Dollar', fontsize=20)
plt.xticks(rotation=90,fontsize=20)
plt.xlabel('Loan sector', fontsize=20)
plt.title('Popular loan sector in terms of loan amount', fontsize=24)
# plt.savefig('popular_loan_amount_sector.png')
plt.show()

# Hm! It looks like most popular loan theme in terms of frequency is General, Agriculture and Higher education. But most popular loan theme in terms of amount is totally different. 

# <a id="kiva_challenge_solution"></a>
# 
# # Kiva challenge solution
# > In this section, I will try to solve Kiva challenge. Kiva wants to know the borrower poverty label. So we have to know poverty score as much local as we can. But no dataset provide that information. So we have to use some tactics to predict that data. Also, Kiva currently uses MPI as their poverty metrics. But we have to find smarter poverty metrics.  

# # 1. Cleaned and Processed Data
# 

# In[25]:


# read kivadhsv1 dataset
# the next two or three cell of code is taken by Mhamed Jabri kernel. 
clusters = pd.read_csv('../input/kivadhsv1/KIVA.DHSv4.csv')

# drop duplicates 
clusters= clusters.drop_duplicates('DHSCLUST')[['DHSCLUST','DHS.lat', 'DHS.lon','country','region','MPI.median', 'Nb.HH', 'AssetInd.median','URBAN_RURA',
                                                'Nb.Electricity', 'Nb.fuel', 'Nb.floor', 'Nb.imp.sanitation', 'Nb.imp.water', 'Median.educ', 'Nb.television', 'Nb.phone','names']]
# convert Nb+'' colulumns to percentage
for indic in ['Nb.Electricity','Nb.fuel','Nb.floor','Nb.imp.sanitation','Nb.imp.water','Nb.television','Nb.phone'] : 
    clusters[indic]=round(100*clusters[indic]/clusters['Nb.HH'],2)
    
clusters['URBAN_RURA']=clusters['URBAN_RURA'].apply (lambda x : 1 if x=='U' else 0 )

# scale the AssetInd.median because it contains some negetive value
max_asset = max(clusters['AssetInd.median'])
min_asset = min(clusters['AssetInd.median'])
clusters['AssetInd.median']=clusters['AssetInd.median'].apply(lambda x : round((100/(max_asset-min_asset)) * (x-min_asset),2))

# rename column names
clusters.rename(columns={"MPI.median":"MPI_cluster", "AssetInd.median" : "wealth_index", "URBAN_RURA": "urbanity" , 'Nb.Electricity':'ratio_electricity',
                        'Nb.imp.sanitation':'ratio_sanitation','Nb.imp.water':'ratio_water', 'Nb.phone':'ratio_phone',
                        'Nb.floor':'ratio_nakedSoil','Nb.fuel':'ratio_cookingFuel', 'names':'location_details'}, inplace=True)

# In[26]:


# we input loan_coords and loans_extended and join them on loan_id
loan_coords = pd.read_csv('../input/additional-kiva-snapshot/loan_coords.csv')
loans_extended = pd.read_csv('../input/additional-kiva-snapshot/loans.csv')
un_data = pd.read_csv('../input/undata-country-profiles/kiva_country_profile_variables.csv')
un_data = un_data.rename(columns={'country': 'country_name'})
loans_with_coords = loans_extended.merge(loan_coords, how='left', on='loan_id')
loans_with_coords = loans_with_coords.merge(un_data, how='left', on='country_name')

loans_with_coords=loans_with_coords[['loan_id','country_code','country_name','town_name','latitude','longitude',
                                    'original_language','description','description_translated','tags', 'activity_name','sector_name','loan_use',
                                    'loan_amount','funded_amount',
                                    'posted_time','planned_expiration_time','disburse_time', 'raised_time', 'lender_term', 'num_lenders_total','repayment_interval', 'Population in thousands (2017)', 'GDP: Gross domestic product (million current US$)', 'Unemployment (% of labour force)', 'International trade: Exports (million US$)', 'Infant mortality rate (per 1000 live births', 'Education: Government expenditure (% of GDP)', 'Individuals using the Internet (per 100 inhabitants)']]
# handle null values
loans_with_coords = loans_with_coords[np.isfinite(loans_with_coords['latitude'])]
# we will work with most occurances country
loans = loans_with_coords[loans_with_coords['country_name'].isin(['Philippines','Colombia','Armenia','Kenya','Haiti'])]

loans.head()

# In[58]:


'''Performing Knn with k=1 to find the cluster for each loan'''
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
neigh = KNeighborsClassifier(n_neighbors=1 , metric='haversine')
neigh.fit(np.radians(clusters[['DHS.lat', 'DHS.lon']]), clusters['DHSCLUST']) 
loans['DHSCLUST'] = neigh.predict(np.radians(loans[['latitude','longitude']]))

'''Build a table to show the coordinates of the loan and coordinates of the cluster it is assigned to, then calculate the Haversine distance in kilometers between cluster and loan'''
precision_knn = loans[['DHSCLUST','loan_id','country_code','country_name','town_name','latitude','longitude','original_language','sector_name','loan_use','loan_amount','funded_amount','num_lenders_total','Population in thousands (2017)', 'GDP: Gross domestic product (million current US$)', 'Unemployment (% of labour force)', 'International trade: Exports (million US$)', 'Infant mortality rate (per 1000 live births', 'Education: Government expenditure (% of GDP)', 'Individuals using the Internet (per 100 inhabitants)']].merge(clusters[['DHSCLUST','DHS.lat','DHS.lon',"MPI_cluster", "wealth_index","urbanity" ,'ratio_electricity','ratio_sanitation','ratio_water','ratio_phone','ratio_nakedSoil','ratio_cookingFuel','location_details']], how='left', on='DHSCLUST')
lat1 = np.radians(precision_knn['latitude'])
lat2 = np.radians(precision_knn['DHS.lat'])
lon1 = np.radians(precision_knn['longitude'])
lon2 = np.radians(precision_knn['DHS.lon'])
temp = np.power((np.sin((lat2-lat1)/2)),2) + np.cos(lat1) * np.cos(lat2) * np.power((np.sin((lon2-lon1)/2)),2)
precision_knn['distance_km'] = 6371 * (2 * np.arcsin(np.sqrt(temp))) #6371 is the radius of the earth

precision_knn.sample(10)

# In[59]:


print("The median distance in kilometers between a loan and the cluster it's assigned to is : " , round(precision_knn['distance_km'].median(),2))

# In[60]:


precision_knn.rename(columns={"GDP: Gross domestic product (million current US$)":"GDP", "Population in thousands (2017)":"population_in_thousands", "Unemployment (% of labour force)": "unemployment","International trade: Exports (million US$)":"internation_trade_exports", "Infant mortality rate (per 1000 live births":"infant_mortality_rate", "Education: Government expenditure (% of GDP)":"education_gov_expenditure","Individuals using the Internet (per 100 inhabitants)":"individuals_using_internet"}, inplace=True)

# # 2. A Predictive Model
# > Now we have our desired data.  Now I will build a model that predict poverty targeting score. 

# In[77]:


import statsmodels.api as sm
from statsmodels.formula.api  import ols
model = ols(formula = 'MPI_cluster ~ urbanity+ratio_electricity',
          data = precision_knn).fit()
print(model.summary())

# P-value is less than 0.05. It means the relation is statistically significant. It means we can predict MPI_cluster given those variable. Let's see the relation by a plot. 

# In[78]:


fig = plt.figure(figsize=(12,8))
fig = sm.graphics.plot_partregress_grid(model, fig=fig)

# wealth_index can be a poverty targeting score. Let' see! 

# In[79]:


import statsmodels.api as sm
from statsmodels.formula.api  import ols
model = ols(formula = 'wealth_index ~ urbanity+ratio_electricity',
          data = precision_knn).fit()
print(model.summary())

# P-value is leass than 0.05. So we can also predict wealth_index by using those variables. Let's see some relationship between those variables.

# In[80]:


fig = plt.figure(figsize=(12,8))
fig = sm.graphics.plot_partregress_grid(model, fig=fig)

# In[84]:


precision_knn.plot(kind='scatter', x='MPI_cluster', y='wealth_index', title="MPI_cluster versus wealth_index");

# # 3. Cleaned and Processed Kiva data and Poverty Scores: Output and Analysis
# 
# 

# In[92]:


df_kiva_fresh = pd.read_csv('../input/data-science-for-good-kiva-crowdfunding/kiva_mpi_region_locations.csv')
df_kiva_fresh = df_kiva_fresh[df_kiva_fresh['country'].isin(['Philippines','Colombia','Armenia','Kenya','Haiti'])]

# In[97]:


df_kiva_fresh.groupby('country')['MPI'].mean().plot(kind='hist', figsize=(10, 6), title='Histogram of kiva\'s current country label mpi')

# In[98]:


precision_knn.groupby('country_name')['MPI_cluster'].mean().plot(kind='hist', figsize=(10, 6), title='Histogram of new city label mpi ')

# We can clearly see the difference. By using new localized MPI, Kiva can know the poverty condition of borrower more efficiently. 

# Thanks you for reading my kernel. 
