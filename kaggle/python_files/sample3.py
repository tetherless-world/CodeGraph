#!/usr/bin/env python
# coding: utf-8

# # ** Kiva - Data analysis and Poverty estimation **
# ***
# 
# **Mhamed Jabri — 02/27/2018**
# 
# Machine Learning is **disruptive**. That's no news, everyone knows that by now. Every industry is being affected by AI, AI/ML startups are booming ... No doubt, now sounds like the perfect time to be a data scientist !  
# That being said, two industries stand out for me when it comes to applying machine learning : **Healthcare and Economic Development**. Not that other applications aren't useful or interesting but those two are, in my opinion, showing how we can really use technology to make the world a better place. Some impressive projects on those fields are being conducted right now by several teams; *Stanford* has an ongoing project about predicting poverty rates with satellite imagery, how impressive is that ?!
# 
# Here in Kaggle, we already have experience with the first one (healthcare), for example, every year there's the *Data Science Bowl* challenge where competitors do their very best to achieve something unprecedented, in 2017, the competition's goal was to **improve cancer screening care and prevention**.  
# I was very excited and pleased when I got the email informing me about the Kiva Crowdfunding challenge and it's nice to know that this is only the beggining, with many more other competitions to come in the Data Science for Good program.
# 
# Being myself very interested in those issues and taking courses such as *Microeconomics* and *Data Analysis for Social Scientists* (If interested, you can find both courses [here](https://micromasters.mit.edu/dedp/), excellent content proposed by MIT and Abdul Latif Jameel Poverty Action Lab), I decided to publish a notebook in this challenge and take the opportunity to use everything I've learned so far.     
# **Through this notebook**, I hope that not only will you learn some Data Analysis / Machine Learning stuff, but also (and maybe mostly) learn a lot about economics (I'll do my best), learn about poverty challenges in the countries where Kiva is heavily involved, learn about how you can collect data that's useful in those problems and hopefully inspire you to apply your data science skills to build a better living place in the future !
# 
# **P.S : This will be a work in progress for at least a month. I will constantly try to improve the content, add new stuff and make use of any interesting new dataset that gets published for this competition.**
# 
# Enjoy !

# # Table of contents
# ***
# 
# * [About Kiva and the challenge](#introduction)
# 
# * [1. Exploratory Data Analysis](#EDA)
#    * [1.1. Data description](#description)
#    * [1.2. Use of Kiva around the world](#users)
#    * [1.3. Loans, how much and what for ?](#projects)
#    * [1.4. How much time until you get funded ?](#dates)
#    * [1.5. Lenders : who are they and what drives them ?](#lenders)
# 
# * [2. Poverty estimation by region](#predict)
#    * [2.1. What's poverty ?](#definition)
#    * [2.2. External data souces](#data)
# 
# * [Conclusion](#conclusion)

# #  About Kiva and the challenge
# ***
# 
# Kiva is a non-profit organization that allows anyone to lend money to people in need in over 80 countries. When you go to kiva.org, you can choose a theme (Refugees, Shelter, Health ...) or a country and you'll get a list of all the loans you can fund with a description of the borrower, his needs and the time he'll need for repayment. So far, Kiva has funded more than 1 billion dollars to 2 million borrowers and is considered a major actor in the fight against poverty, especially in many African countries.
# 
# In this challenge, the ultimate goal is to obtain as precise informations as possible about the poverty level of each borrower / region because that would help setting investment priorities. Kagglers are invited to use Kiva's data as well as any external public datasets to build their poverty estimation model.  
# As for Kiva's data, here's what we've got : 
# * **kiva_loans** : That's the dataset that contains most of the informations about the loans (id of borrower, amount of loan, time of repayment, reason for borrowing ...)
# * **kiva_mpi_region_locations** : This dataset contains the MPI of many regions (subnational) in the world.
# * **loan_theme_ids** : This dataset has the same unique_id as the kiva_loans (id of loan) and contains information about the theme of the loan.
# * **loan_themes_by_region** : This dataset contains specific informations about geolocation of the loans.
# 
# This notebook will be divided into two parts : 
# 1. First I will conduct an EDA using mainly the 4 datasets provided by Kiva. 
# 2. After that, I'll try to use the informations I got from the EDA and external public datasets to build a model for poverty level estimation.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import missingno as msno
from datetime import datetime, timedelta


import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn-darkgrid')
palette = plt.get_cmap('Set1')

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import plotly.figure_factory as ff


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

# Any results you write to the current directory are saved as output.

# # 1. Exploratory Data Analysis
# <a id="EDA"></a>
# *** 
# In this part, the goal is to understand the data that was given to us through plots and statistics, draw multiple conclusions and see how we can use those results to build the features that will be needed for our machine learning model. 
# 
# Let's first see what this data is about.

# ## 1.1 Data description
# <a id="description"></a>
# *** 
# Let's load the 4 csv files we have and start by analyzing the biggest one : kiva loans.

# In[ ]:


df_kiva_loans = pd.read_csv("../input/data-science-for-good-kiva-crowdfunding/kiva_loans.csv")
df_loc = pd.read_csv("../input/data-science-for-good-kiva-crowdfunding/loan_themes_by_region.csv")
df_themes = pd.read_csv("../input/data-science-for-good-kiva-crowdfunding/loan_theme_ids.csv")
df_mpi = pd.read_csv("../input/data-science-for-good-kiva-crowdfunding/kiva_mpi_region_locations.csv")

df_kiva_loans.head(5)

# Before going any further, let's take a look at the missing values so that we don't encounter any bad surprises along the way.

# In[ ]:


msno.matrix(df_kiva_loans)

# Seems that this dataset is pretty clean ! the *tags* column got a lot of missing values but that's not a big deal. The *funded_time* has little less than 10% of missing values, that's quite a few but since we have more than 600 000 rows, we can drop the missing rows if we need to and we'll still get some telling results !  
# Let's get some global information about each of our columns.

# In[ ]:


df_kiva_loans.describe(include = 'all')

# Plenty of useful informations in this summary :
# * There are exactly 87 countries where people borrowed money according to this snapshot.
# * There are 11298 genders in this dataset ! That's obviously impossible so we'll see later on why we have this value. 
# * The funding mean over the world is 786 dollars while the funding median is 450 dollars.
# * More importantly : there's only 1298 different dates on which loans were posted. If we calculate the ratio, **it means that there's more than 500 loans posted per day on Kiva** and that's just a snapshot (a sample of their entire data). This gives you a clear idea about how important this crowdsourcing platform is and what impact it has.

# ## 1.2. Kiva users 
# <a id="users"></a>
# *** 
# In this part we will focus on the basic demographic properties of people who use Kiva to ask for loans : Where do they live ? what's their gender ? Their age would be a nice property but we don't have direct access to that for now, we'll get to that later.
# 
# Let's first start with their countries : as seen above, the data contains 671205 rows. In order to have the most (statistically) significant results going further, I'll only keep the countries that represent at least 0.5% of Kiva's community. 

# In[ ]:


countries = df_kiva_loans['country'].value_counts()[df_kiva_loans['country'].value_counts()>3400]
list_countries = list(countries.index) #this is the list of countries that will be most used.

# In[ ]:


plt.figure(figsize=(12,8))
sns.barplot(y=countries.index, x=countries.values, alpha=0.6)
plt.title("Country distribution of Kiva's users", fontsize=16)
plt.xlabel("Number of borrowers", fontsize=16)
plt.ylabel("Country", fontsize=16)
plt.show();

# Philippines is the country with most borrowers with approximately 25% of all users being philippinians. Elliott Collins, from the Kiva team, explained that this is due to the fact that a couple of Philippine field partners tend to make smaller short-term loans (popular low-risk loans + fast turnover rate). 
# 
# 
# We also notice that several african countries are in the list such as *Kenya, Mali, Nigeria, Ghana ...* and no european union country at all !     
# For me, the most surprising was actually the presence of the US in this list, as it doesn't have the same poverty rate as the other countries but it turns out it's indeed a specific case, **I'll explain that in 1.4**.
# 
# Let's now move on to the genders.

# In[ ]:


df_kiva_loans['borrower_genders']=[elem if elem in ['female','male'] else 'group' for elem in df_kiva_loans['borrower_genders'] ]
borrowers = df_kiva_loans['borrower_genders'].value_counts()
labels = (np.array(borrowers.index))

values = (np.array((borrowers / borrowers.sum())*100))

trace = go.Pie(labels=labels, values=values,
              hoverinfo='label+percent',
               textfont=dict(size=20),
                showlegend=True)

layout = go.Layout(
    title="Borrowers' genders"
)

data_trace = [trace]
fig = go.Figure(data=data_trace, layout=layout)
py.iplot(fig, filename="Borrowers_genders")


# In many loans (16.4% as you can see), the borrower is not actually a single person but a group of people that have a project, here's an [example](https://www.kiva.org/lend/1440912). In the dataset, they're listed as 'female, female, female' or 'male, female' ... I decided to use the label *mixed group* to those borrowers on the pie chart above.
# 
# You can see that most borrowers are female, I didn't expect that and it was actually a great surprise. This means that **women are using Kiva to get funded and work on their projects in countries (most of them are third world countries) where breaking in as a woman is still extremely difficult.**

# ## 1.3 Activities, sectors and funding amounts
# ***
# 
# Now let's take a peek at what people are needing loans for and what's the amounts they're asking for. Let's start with the sectors. There were 15 unique sectors in the summary we've seen above, let's see how each of them fare.

# In[ ]:


plt.figure(figsize=(12,8))
sectors = df_kiva_loans['sector'].value_counts()
sns.barplot(y=sectors.index, x=sectors.values, alpha=0.6)
plt.title("Country Distribution of the suervey participants", fontsize=16)
plt.xlabel("Number of participants", fontsize=16)
plt.ylabel("Sector", fontsize=16)
plt.show();

# **The most dominant sector is Agriculture**, that's not surprising given the list of countries that heavily use Kavi. A fast research for Kenya for example shows that all the top page is about agriculture loans, here's a sample of what you would find:  *buy quality seeds and fertilizers to use in farm*, *buy seeds to start a horticulture farming business so as a single mom*, *Purchase hybrid maize seed and fertilizer* ... Food sector occupies an important part too because many people are looking to buy fish, vegetables and stocks for their businesses to keep running.  
# It's important to note that *Personal Use* occupy a significant part too, this means there are people who don't use Kavi to get a hand with their work but because they are highly in need.
# 
# Let's see the more detailed version and do a countplot for **activities**

# In[ ]:


plt.figure(figsize=(15,10))
activities = df_kiva_loans['activity'].value_counts().head(50)
sns.barplot(y=activities.index, x=activities.values, alpha=0.6)
plt.title("Country Distribution of the suervey participants", fontsize=16)
plt.xlabel("Number of participants", fontsize=16)
plt.ylabel("Activity", fontsize=16)
plt.show();

# This plot is only a confirmation of the previous one, activities related to agriculture come in the top : *Farming, Food production, pigs ...*. All in all, we notice that none of the activities belong to the world of 'sophisticated'. Everything is about basic daily needs or small businesses like buying and reselling clothes ...
# 
# How about the money those people need to pursue their goals ?

# In[ ]:


plt.figure(figsize=(10,6))
sns.distplot(df_kiva_loans['loan_amount']);

# Some outliers are clearly skewing the distribution and the plot doesn't give much information in this form : We need to **truncate the data**, how do we do that ? 
# 
# We'll use a basic yet really powerful rule : the **68–95–99.7 rule**. This rule states that for a normal distribution :
# * 68.27% of the values $ \in [\mu - \sigma , \mu + \sigma]$
# * 95.45% of the values $ \in [\mu - 2\sigma , \mu + 2\sigma]$
# * 99.7% of the values $ \in [\mu - 3\sigma , \mu + 3\sigma]$     
# where $\mu$ and $\sigma$ are the mean and standard deviation of the normal distribution.
# 
# Here it's true that the distribution isn't necessarily normal but for a shape like the one we've got, we'll see that applying the third filter will **improve our results radically**.
# 

# In[ ]:


temp = df_kiva_loans['loan_amount']

plt.figure(figsize=(10,6))
sns.distplot(temp[~((temp-temp.mean()).abs()>3*temp.std())] );

# Well, that's clearly a lot better !    
# * Most of the loans are between 100\$ and 600\$ with a first peak at 300\$.
# * The amount is naturally decreasing but we notice that we have a clear second peak at 1000\$. This suggets that there may be a specific class of projects that are more 'sophisticated' and get funded from time to time, interesting.

# ## 1.4. Waiting time for funds
# <a id="dates"></a>
# *** 
# 
# So far we got to see where Kiva is most popular, the nature of activities borrowers need the money for and how much money they usually ask for, great !    
# 
# An interesting question now would also be : **how long do they actually have to wait for funding ?** As we've seen before, some people on the plateform are asking for loans for critical needs and can't afford to wait for months to buy groceries or have a shelter. Fortunately, we've got two columns that will help us in our investigation : 
# * funded_time : corresponds to the date + exact hour **when then funding was completed.**
# * posed_time : corresponds to the date + exact hour **when the post appeared on the website.**
# 
# We've also seen before that we have some missing values for 'funded_time' so we'll drop those rows, get the columns in the correct date format and then calculate the difference between them.

# In[ ]:


loans_dates = df_kiva_loans.dropna(subset=['disbursed_time', 'funded_time'], how='any', inplace=False)

dates = ['posted_time','disbursed_time','funded_time']
loans_dates[dates] = loans_dates[dates].applymap(lambda x : x.split('+')[0])

loans_dates[dates]=loans_dates[dates].apply(pd.to_datetime)
loans_dates['time_funding']=loans_dates['funded_time']-loans_dates['posted_time']
loans_dates['time_funding'] = loans_dates['time_funding'] / timedelta(days=1) 
#this last line gives us the value for waiting time in days and float format,
# for example: 3 days 12 hours = 3.5

# Now first thing first, we'll plot the this difference that we called *time_funding*. To avoid any outliers, we'll apply the same rule for normal distribution as before.

# In[ ]:


temp = loans_dates['time_funding']

plt.figure(figsize=(10,6))
sns.distplot(temp[~((temp-temp.mean()).abs()>3*temp.std())] );

# I was really surprised when I got this plot (and happy too), you'll rarely find a histogram where the distribution fits in this smoothly !   
# On top of that, getting two peaks was the icing on the cake, it makes perfect sense ! **We've seen above that there are two peaks for loans amounts, at 300\$ and 1000\$, we're basically saying that for the first kind of loan you would be waiting 7 days and for the second kind a little more than 30 days !   **
# This gives us a great intuition about how those loans work going forward.
# 
# Let's be more specific and check for both loan amounts and waiting time country-wise :   
# We'll build two new DataFrames using the groupby function and we'll aggregate using the median : what we'll get is the median loan amount (respectively waiting time) for each country.

# In[ ]:


df_ctime = round(loans_dates.groupby(['country'])['time_funding'].median(),2)
df_camount = round(df_kiva_loans.groupby(['country'])['loan_amount'].median(),2)

# In[ ]:


df_camount = df_camount[df_camount.index.isin(list_countries)].sort_values()
df_ctime = df_ctime[df_ctime.index.isin(list_countries)].sort_values()

f,ax=plt.subplots(1,2,figsize=(20,10))

sns.barplot(y=df_camount.index, x=df_camount.values, alpha=0.6, ax=ax[0])
ax[0].set_title("Medians of funding amounts per loan country wise ")
ax[0].set_xlabel('Amount in dollars')
ax[0].set_ylabel("Country")

sns.barplot(y=df_ctime.index, x=df_ctime.values, alpha=0.6,ax=ax[1])
ax[1].set_title("Medians of waiting days per loan to be funded country wise  ")
ax[1].set_xlabel('Number of days')
ax[1].set_ylabel("")

plt.tight_layout()
plt.show();

# **Left plot**    
# We notice that in most countries, funded loans don't usually exceed 1000\$. For Philippines, Kenya and El Salvador (the three most present countries as seen above), the medians of fund per loan are respectively : 275.00\$, 325.00\$ and 550.00\$ .
# 
# The funded amount for US-based loans seem to be a lot higher than for other countries. I dug deeper and looked in Kiva's website. **It appears that there's a special section called 'Kiva U.S.' which goal is to actually fund small businesses for *financially excluded and socially impactful borrowers*.  ** 
# Example of such businesses : Expanding donut shop in Detroit (10k\$),  Purchasing equipment and paying for services used to properly professionally train basketball kids ... You can see more of that in [here](https://www.kiva.org/lend/kiva-u-s).    
# This explains what we've been seeing earliers : the fact that the US is among the countries, the big amount of loan, the two-peaks plots ...
# 
# **Right plot**   
# The results in this one aren't that intuitive. 
# * Paraguay is the second fastest country when it comes to how much to wait for a loan to be funded while it was also the country with the second highest amount per loan in the plot above !  
# * The US loans take the most time to get funded and that's only natural since their amount of loans are much higher than the other countries.
# * Most of African countries are in the first half of the plot.

# ## 1.5. Lenders community
# <a id="lenders"></a>
# *** 
# We said that we would talk about Kiva users, that include lenders too ! It's true that our main focus here remains the borrowers and their critical need but it's still nice to know more about who uses Kiva in the most broad way and also get an idea about **what drives people to actually fund projects ?   **
# Thanks to additional datasets, we got freefrom text data about the lenders and their reasons for funding, let's find about that.

# In[ ]:


lenders = pd.read_csv('../input/additional-kiva-snapshot/lenders.csv')
lenders.head()

# Seems like this dataset is filled with missing values :). We'll still be able to retrieve some informations, let's start by checking which country has most lenders.

# In[ ]:


lender_countries = lenders.groupby(['country_code']).count()[['permanent_name']].reset_index()
lender_countries.columns = ['country_code', 'Number of Lenders']
lender_countries.sort_values(by='Number of Lenders', ascending=False,inplace=True)
lender_countries.head()

# Two things here :    
# * The US is, by far, the country with most lenders. It has approximately 9 times more lenders than any other country. If we want to plot a map or a barplot with this information, we have two choices : either we leave out the US or we use a logarithmic scale, which means we'll apply $ ln(1+x) $ for each $x$ in the column *Number of Lenders*. The logarithmic scale allows us to respond to skewness towards large values when one or more points are much larger than the bulk of the data (here, the US).
# * We don't have a column with country names so we'll need to use another dataset to get those and plot a map.
# 
# Here's another additional dataset that contains poverty informations about each country. For the time being, we'll only use the column *country_name* to merge it with our previous dataset.

# In[ ]:


countries_data = pd.read_csv( '../input/additional-kiva-snapshot/country_stats.csv')
countries_data.head()

# In[ ]:


countries_data = pd.read_csv( '../input/additional-kiva-snapshot/country_stats.csv')
lender_countries = pd.merge(lender_countries, countries_data[['country_name','country_code']],
                            how='inner', on='country_code')

data = [dict(
        type='choropleth',
        locations=lender_countries['country_name'],
        locationmode='country names',
        z=np.log10(lender_countries['Number of Lenders']+1),
        colorscale='Viridis',
        reversescale=False,
        marker=dict(line=dict(color='rgb(180,180,180)', width=0.5)),
        colorbar=dict(autotick=False, tickprefix='', title='Lenders'),
    )]
layout = dict(
    title = 'Lenders per country in a logarithmic scale ',
    geo = dict(showframe=False, showcoastlines=True, projection=dict(type='Mercator'))
)
fig = dict(data=data, layout=layout)
py.iplot(fig, validate=False, filename='lenders-map')

# The US have the largest community of lenders and it is followed by Canada and Australia. On the other hand, the African continent seems to have the lowest number of funders which is to be expected, since it's also the region with highest poverty rates and funding needs.
# 
# So now that we know more about lenders location, let's analyze the textual freeform column *loan_because* and construct a wordcloud to get an insight about their motives for funding proejcts on Kiva.

# In[ ]:


import matplotlib as mpl 
from wordcloud import WordCloud, STOPWORDS
import imageio

heart_mask = imageio.imread('../input/heartmask2/heart_msk.jpg') #because displaying this wordcloud as a heart seems just about right :)

mpl.rcParams['figure.figsize']=(12.0,8.0)    #(6.0,4.0)
mpl.rcParams['font.size']=10                #10 

more_stopwords = {'org', 'default', 'aspx', 'stratfordrec','nhttp','Hi','also','now','much'}
STOPWORDS = STOPWORDS.union(more_stopwords)

lenders_reason = lenders[~pd.isnull(lenders['loan_because'])][['loan_because']]
lenders_reason_string = " ".join(lenders_reason.loan_because.values)

wordcloud = WordCloud(
                      stopwords=STOPWORDS,
                      background_color='white',
                      width=3200, 
                      height=2000,
                      mask=heart_mask
            ).generate(lenders_reason_string)

plt.imshow(wordcloud)
plt.axis("off")
plt.savefig('./reason_wordcloud.png', dpi=900)
plt.show()

# Lenders' answers are heartwarming :) Most reasons contain *help people / others* or *want to help*. We also find that it's the *right thing* (to do), it helps *less fortunate* and makes the world a *better place*.  
# Kiva provides a platform for people who need help to fund their projects but it also provides a platform for people who want to make a difference by helping others and maybe changing their lives !

# # 2. Welfare estimation
# <a id="prediction"></a>
# *** 
# In this part we'll delvo into what's this competition is really about : **welfare and poverty estimation.  ** 
# As a lender, you basically have two criterias when you're looking for a loan to fund : the loan description and how much the borrower does really need that loan. For the second, Kiva's trying to have as granular poverty estimates as possible through this competition.
# 
# In this part, I'll be talking about what poverty really means and how it is measures by economists. I'll also start with a country-level model as an example to what will be said.
# 
# Let's start.

#     No society can surely be flourishing and happy, of which by far the greater part of the numbers are poor and miserable. - Adam Smith, 1776       
# 

# ## 2.1 What's poverty ?
# <a id="definition"></a>
# *** 
# The World Bank defines poverty in terms of **income**. The bank defines extreme poverty as living on less than US\$1.90 per day (PPP), and moderate poverty as less than \$3.10 a day.  
# P.S: In this part, we'll say (PPP) a lot. It refers to Purchasing Power Parity. I have a notebook that is entirely dedicated to PPP and if interested and want to know more about how it works, you can check it [here](https://www.kaggle.com/mhajabri/salary-and-purchasing-power-parity).  
# Over the past half century, significant improvements have been made and still, extreme poverty remains widespread in the developing countries. Indeed, an estimated **1.374 billion people live on less than  1.25 \$ per day** (at 2005 U.S. PPP) and around **2.6 billion (which is basically 40% of the worlds's population !!) live on less than \$ 2 per day**. Those impoverished people suffer from : undernutrition / poor health, live in environmentally degraded areas, have little literacy ...
# 
# As you can see, poverty seems to be defined exactly by the way it's actually measured, but what's wrong with that definition ? **In developing countries, many of the poor work in the informal sector and lack verifiable income records => income data isn't reliable**. Suppose you're the government and you have a specific program that benefits the poorest or you're Kiva and you want to know who's in the most critical condition, then relying on income based poverty measures in developing countries will be misleading and using unreliable information to identify eligible households can result in funds being diverted to richer households and leave fewer resources for the program’s intended beneficiaries. We need another way of measuring poverty. 

# ## 2.1 Multidimensional Poverty Index
# <a id="definition"></a>
# *** 
# 
# Well one day the UNDP (United Nations Development Programme) came and said well *salary* is only one **dimension** that can describe poverty levels but it's far from the only indicator. Indeed, if you visit someone's house and take a look at how it is and what it has, it gives an intuition. Based on that, the UNDP came up with the **Multidimensional Poverty Index **, an index that has **3 dimensions and a total of 10 factors **assessing poverty : 
# * **Health **: Child Mortality - Nutrition
# * **Education** : Years of schooling - School attendance
# * **Living Standards** : Cooking fuel - Toilet - Water - Electricity - Floor - Assets
# 
# How is the MPI calculated ? Health's and Education's indicators (there are 4 in total) are weighted equally at 1/6. Living standards' indicators are weighted equally at 1/18. The sum of the weights $2*1/6 + 2*1/6 + 6*1/18 = 1$. Going from here, **a person is considered poor if they are deprived in at least a third of the weighted indicators.** 
# Example : Given a household with no electricity, bad sanitation, no member with more than 6 years of schooling and no access to safe drinking water, the MPI score would be : 
# $$ 1/18 + 1/18 + 1/6 + 1/18 = 1/3$$
# So this household is deprived in at least a third of the weighted indicators (MPI > 0.33) and is considered MPI-poor.
# 
# Kiva actually included MPI data so let's get a look at it :

# In[ ]:


df_mpi.head(10)

# This dataset gives the MPI of different regions for each country, to have a broad view, let's use a groupby *country* and take the average MPI and plot that in a map.

# In[ ]:


mpi_country = df_mpi.groupby('country')['MPI'].mean().reset_index()


data = [dict(
        type='choropleth',
        locations=mpi_country['country'],
        locationmode='country names',
        z=mpi_country['MPI'],
        colorscale='Greens',
        reversescale=True,
        marker=dict(line=dict(color='rgb(180,180,180)', width=0.5)),
        colorbar=dict(autotick=False, tickprefix='', title='MPI'),
    )]
layout = dict(
    title = 'Average MPI per country',
    geo = dict(showframe=False, showcoastlines=True, projection=dict(type='Mercator'))
)
fig = dict(data=data, layout=layout)
py.iplot(fig, validate=False, filename='mpi-map')

# As you can notice, the data provides the MPI for the African continent essentially. That shouldn't surpise you, as we said before, for developed countries, income data is actually reliable and good enough to measure poverty so we don't need to send researchs on the field to run surveys and get the necessary data for the MPI. That's why you'll find more data about MPI measurements in developing / poor countries.   
# 

# # **Work in progress, stay tuned**
