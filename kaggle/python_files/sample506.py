#!/usr/bin/env python
# coding: utf-8

# ![](http://www.ewdn.com/wp-content/uploads/sites/6/2017/02/logo-avito.png)

# # More To Come. Stay Tuned. !!
# If there are any suggestions/changes you would like to see in the Kernel please let me know :). Appreciate every ounce of help!
# 
# **This notebook will always be a work in progress**. Please leave any comments about further improvements to the notebook! Any feedback or constructive criticism is greatly appreciated!. **If you like it or it helps you , you can upvote and/or leave a comment :).**
# 
# I am using [Yandex Translate](https://translate.yandex.com/?lang=ru-en&text=Челябинск) for converting russian language to english language.
# 

# - <a href='#intro'>1. Introduction</a>  
# - <a href='#rtd'>2. Retrieving the Data</a>
#      - <a href='#ll'>2.1 Load libraries</a>
#      - <a href='#rrtd'>2.2 Read the Data</a>
# - <a href='#god'>3. Glimpse of Data</a>
#      - <a href='#oot'>3.1 Overview of tables</a>
#      - <a href='#sootd'>3.2 Statistical overview of the Data</a>
# - <a href='#dp'>4. Data preparation</a>
#      - <a href='#cfmd'> 4.1 Check for missing data</a>
# - <a href='#de'>5. Data Exploration</a>
#     - <a href='#hadodp'>5.1 Histogram and distribution of deal probability</a>
#     - <a href='#hadoap'>5.2 Histogram and distribution of Ad price</a>
#     - <a href='#dodar'>5.3 Distribution of differnet Ad regions</a>
#     - <a href='#t'>5.4 Top 10</a> 
#         - <a href='#ttat'>5.4.1 Top 10 Ad titles</a>
#         - <a href='#ttac'>5.4.2 Top 10 Ad city</a>
#         - <a href='#ttara'>5.4.3 Top 10 Ad regions</a>
#         - <a href='#tttlcalm'>5.4.4 Top 10 Fine grain ad category as classified by Avito's ad mode</a>
#         - <a href='#ttlcmsd'>5.4.5 Top 10 Top level ad category as classified by Avito's ad model</a>
#     - <a href='#pvdpp'>5.5 Price V.S. Deal probability</a>
#     - <a href='#5-6'>5.6 Deal probability V.S.  Price for regions</a>
#     - <a href='#dout'>5.7 Distribution of user type</a>
#     - <a href='#mdodadr'>5.8 Monthly distribution of Ad prices in different regions </a>
#     - <a href='#dorpdp'>5.9 Distribution of regions, per deal probability</a>
#     - <a href='#tkfad'>5.10 Top Keywords from Ad description</a>
#     - <a href='#toapaa'>5.10 Time Series Analysis</a>
#     - <a href='#5-11'>5.11 Ad sequential number for user V.S. deal probability</a>
#     - <a href='#5-12'>5.12 Deal probability V.S. Ad sequential number for user for regions</a>
#     - <a href='#5-13'>5.13 Number of words in description column</a>
#     - <a href='#toapaa'>5.14 Time series Analysis</a>
#         - <a href='#toadp'>5.14.1 Trend of Ad price</a>
#         - <a href='#paetsd'>5.14.2 Price average every two days</a>
#         - <a href='#paeed'>5.14.3 Price average every day</a>
#         - <a href='#dapevtr'>5.14.4 deal probability average every two days</a>
#         - <a href='#dapevtttr'>5.14.5 Deal probability average every  days</a>
#         - <a href='#tnodawdjshhs'>5.14.6 Total number of days a Ad was dispalyed when it was posted on particular day</a>
#         - <a href='#fodsbdir'>5.14.7  frequency and pattern of ad activation date in train and test data</a>
# - <a href='#bsc'>6. Brief summary and conclusion :</a>

# # <a id='intro'>1. Introduction</a>  

# Avito, Russia’s largest classified advertisements website, is deeply familiar with this problem. Sellers on their platform sometimes feel frustrated with both too little demand (indicating something is wrong with the product or the product listing) or too much demand (indicating a hot item with a good description was underpriced).
# 
# In their fourth Kaggle competition, Avito is challenging you to predict demand for an online advertisement based on its full description (title, description, images, etc.), its context (geographically where it was posted, similar ads already posted) and historical demand for similar ads in similar contexts. With this information, Avito can inform sellers on how to best optimize their listing and provide some indication of how much interest they should realistically expect to receive.
# 

# # <a id='rtd'>2. Retrieving the Data</a>
# ## <a id='ll'>2.1 Load libraries</a>

# In[1]:


import pandas as pd # package for high-performance, easy-to-use data structures and data analysis
import numpy as np # fundamental package for scientific computing with Python
import matplotlib
import matplotlib.pyplot as plt # for plotting
import seaborn as sns # for making plots with seaborn
color = sns.color_palette()
import plotly.offline as py
py.init_notebook_mode(connected=True)
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.offline as offline
offline.init_notebook_mode()
import plotly.tools as tls
import squarify
from mpl_toolkits.basemap import Basemap
from numpy import array
from matplotlib import cm

# Supress unnecessary warnings so that presentation looks clean
import warnings
warnings.filterwarnings("ignore")

# Print all rows and columns
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

from nltk.corpus import stopwords
from textblob import TextBlob
import datetime as dt
import warnings
import string
import time
# stop_words = []
stop_words = list(set(stopwords.words('english')))
warnings.filterwarnings('ignore')
punctuation = string.punctuation


# ## <a id='rrtd'>2.2 Read the Data</a>

# In[2]:


periods_test = pd.read_csv('../input/periods_test.csv', parse_dates=["activation_date", "date_from", "date_to"])
periods_train = pd.read_csv('../input/../input/periods_train.csv', parse_dates=["activation_date", "date_from", "date_to"])
test = pd.read_csv('../input/test.csv')
train = pd.read_csv('../input/train.csv')
# train_active = pd.read_csv('../input/train_active.csv')
# test_active = pd.read_csv('../input/test_active.csv')

# In[3]:


print("size of train data", train.shape)
print("size of test data", test.shape)
print("size of periods_train data", periods_train.shape)
print("size of periods_test data", periods_test.shape)

# # <a id='god'>3. Glimpse of Data</a>
# ## <a id='oot'>3.1 Overview of tables</a>

# **train data**

# In[4]:


train.head()

# **test data**

# In[5]:


test.head()

# **periods train data**

# In[6]:


periods_train.head()

# **periods test data**

# In[7]:


periods_test.head()

# ## <a id='sootd'>3.2 Statistical overview of the Data</a>

# **Training Data some little info**

# In[8]:


train.info()

# **Little description of training data for numerical features**

# In[9]:


train.describe()

# **Little description of training data for categorical features**

# In[10]:


train.describe(include=["O"])

# # <a id='dp'>4. Data preparation</a>
#  ## <a id='cfmd'> 4.1 Check for missing data</a>

# **checking missing data in training data **

# In[11]:


# checking missing data in training data 
total = train.isnull().sum().sort_values(ascending = False)
percent = (train.isnull().sum()/train.isnull().count()*100).sort_values(ascending = False)
missing_train_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_train_data.head(10)

# **checking missing data in periods training data **

# In[12]:


# checking missing data in periods data 
total = periods_train.isnull().sum().sort_values(ascending = False)
percent = (periods_train.isnull().sum()/periods_train.isnull().count()*100).sort_values(ascending = False)
missing_train_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_train_data.head()

# # <a id='de'>5. Data Exploration</a>

# ## <a id='hadodp'>5.1 Histogram and distribution of deal probability</a>

# In[13]:


plt.figure(figsize = (12, 8))

sns.distplot(train['deal_probability'])
plt.xlabel('likelihood that an ad actually sold something', fontsize=12)
plt.title("Histogram of likelihood that an ad actually sold something")
plt.show() 
plt.figure(figsize = (12, 8))
plt.scatter(range(train.shape[0]), np.sort(train.deal_probability.values))
plt.xlabel('likelihood that an ad actually sold something', fontsize=12)
plt.title("Distribution of likelihood that an ad actually sold something")
plt.show()

# ## <a id='hadoap'>5.2 Histogram and distribution of Ad price</a>

# In[14]:


plt.figure(figsize = (12, 8))

sns.distplot(train['price'].dropna())
plt.xlabel('Ad price', fontsize=12)
plt.title("Histogram of Ad price")
plt.show() 
plt.figure(figsize = (12, 8))
plt.scatter(range(train.shape[0]), np.sort(train.price.values))
plt.xlabel('Ad price', fontsize=12)
plt.title("Distribution of Ad price")
plt.show()

# In[15]:


#https://www.kaggle.com/sudalairajkumar/simple-exploration-notebook-avito
    
from io import StringIO

conversion = StringIO("""
region,region_english
Свердловская область, Sverdlovsk oblast
Самарская область, Samara oblast
Ростовская область, Rostov oblast
Татарстан, Tatarstan
Волгоградская область, Volgograd oblast
Нижегородская область, Nizhny Novgorod oblast
Пермский край, Perm Krai
Оренбургская область, Orenburg oblast
Ханты-Мансийский АО, Khanty-Mansi Autonomous Okrug
Тюменская область, Tyumen oblast
Башкортостан, Bashkortostan
Краснодарский край, Krasnodar Krai
Новосибирская область, Novosibirsk oblast
Омская область, Omsk oblast
Белгородская область, Belgorod oblast
Челябинская область, Chelyabinsk oblast
Воронежская область, Voronezh oblast
Кемеровская область, Kemerovo oblast
Саратовская область, Saratov oblast
Владимирская область, Vladimir oblast
Калининградская область, Kaliningrad oblast
Красноярский край, Krasnoyarsk Krai
Ярославская область, Yaroslavl oblast
Удмуртия, Udmurtia
Алтайский край, Altai Krai
Иркутская область, Irkutsk oblast
Ставропольский край, Stavropol Krai
Тульская область, Tula oblast
""")

conversion = pd.read_csv(conversion)
train = pd.merge(train, conversion, how="left", on="region")


# ## <a id='dodar'>5.3 Distribution of differnet Ad regions</a>

# In[16]:


temp = train['region_english'].value_counts()
labels = temp.index
sizes = (temp / temp.sum())*100
trace = go.Pie(labels=labels, values=sizes, hoverinfo='label+percent')
layout = go.Layout(title='Distribution of differnet Ad regions')
data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)

# ## <a id='t'>5.4 Top 10</a> 

# ## <a id='ttat'>5.4.1 Top 10 Ad titles</a>

# In[17]:


temp = train["title"].value_counts().head(10)
print("Top 10 Ad titles :", temp)
print("Total Ad titles : ",len(train["title"]))
trace = go.Bar(
    x = temp.index,
    y = (temp / temp.sum())*100,
)
data = [trace]
layout = go.Layout(
    title = "Top 10 Ad titles",
    xaxis=dict(
        title='Ad title',
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
    ),
    yaxis=dict(
        title='Count of Ad titles in %',
        titlefont=dict(
            size=16,
            color='rgb(107, 107, 107)'
        ),
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
)
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)

# *** Top 5 Ad titles are :**
# *  Туфли (Shoes) - 28 %
# *  Куртка(Jacket) - 12 %
# * Пальто (Coat)  - 12 %
# * Джинсы(Jeans) - 9 %
# * Комбинезон (Overall) - 9 %
#    

# ## <a id='ttac'>5.4.2 Top 10 Ad city</a>

# In[18]:


temp = train["city"].value_counts().head(10)
print('Top 10 Ad cities :', temp)
print("Total Ad cities : ",len(train["title"]))
trace = go.Bar(
    x = temp.index,
    y = (temp / temp.sum())*100,
)
data = [trace]
layout = go.Layout(
    title = "Top 10 Ad city",
    xaxis=dict(
        title='Ad title name',
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
    ),
    yaxis=dict(
        title='Count of Ad cities in %',
        titlefont=dict(
            size=16,
            color='rgb(107, 107, 107)'
        ),
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
)
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)

# * **Top 5 Ad cities :**
#   * Екатеринбург (Yekaterinburg) - 12 %
#   * Новосибирск (Novosibirsk) - 12 %
#   * Ростов-на-Дону  (Rostov-on-don) - 11 %
#   * Нижний Новгород  (Nizhny Novgorod) 10 %
#   * Челябинск (Chelyabinsk) - 10 %

# ## <a id='ttara'>5.4.3 Top 10 Ad regions</a>

# In[19]:


temp = train["region_english"].value_counts().head(10)
print('Top 10 Ad regions',temp)
print("Total Ad regions : ",len(train["title"]))
trace = go.Bar(
    x = temp.index,
    y = (temp / temp.sum())*100,
)
data = [trace]
layout = go.Layout(
    title = "Top 10 Ad regions",
    xaxis=dict(
        title='Ad region name',
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
    ),
    yaxis=dict(
        title='Count of Ad regions in %',
        titlefont=dict(
            size=16,
            color='rgb(107, 107, 107)'
        ),
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
)
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)

# *** Top 5 Ad regions :**
# * verdlovsk oblast  - 17 %
# * Rostov oblast - 12 %
# * Tatarstan - 11 %
# * Chelyabinsk oblast - 10 %         
# * Nizhny Novgorod oblast  - 10 %

# ## <a id='tttlcalm'>5.4.4 Top 10 Fine grain ad category as classified by Avito's ad mode</a>

# In[20]:


#https://www.kaggle.com/sudalairajkumar/simple-exploration-notebook-avito
conversion = StringIO("""
category_name,category_name_english
"Одежда, обувь, аксессуары","Clothing, shoes, accessories"
Детская одежда и обувь,Children's clothing and shoes
Товары для детей и игрушки,Children's products and toys
Квартиры,Apartments
Телефоны,Phones
Мебель и интерьер,Furniture and interior
Предложение услуг,Offer services
Автомобили,Cars
Ремонт и строительство,Repair and construction
Бытовая техника,Appliances
Товары для компьютера,Products for computer
"Дома, дачи, коттеджи","Houses, villas, cottages"
Красота и здоровье,Health and beauty
Аудио и видео,Audio and video
Спорт и отдых,Sports and recreation
Коллекционирование,Collecting
Оборудование для бизнеса,Equipment for business
Земельные участки,Land
Часы и украшения,Watches and jewelry
Книги и журналы,Books and magazines
Собаки,Dogs
"Игры, приставки и программы","Games, consoles and software"
Другие животные,Other animals
Велосипеды,Bikes
Ноутбуки,Laptops
Кошки,Cats
Грузовики и спецтехника,Trucks and buses
Посуда и товары для кухни,Tableware and goods for kitchen
Растения,Plants
Планшеты и электронные книги,Tablets and e-books
Товары для животных,Pet products
Комнаты,Room
Фототехника,Photo
Коммерческая недвижимость,Commercial property
Гаражи и машиноместа,Garages and Parking spaces
Музыкальные инструменты,Musical instruments
Оргтехника и расходники,Office equipment and consumables
Птицы,Birds
Продукты питания,Food
Мотоциклы и мототехника,Motorcycles and bikes
Настольные компьютеры,Desktop computers
Аквариум,Aquarium
Охота и рыбалка,Hunting and fishing
Билеты и путешествия,Tickets and travel
Водный транспорт,Water transport
Готовый бизнес,Ready business
Недвижимость за рубежом,Property abroad
""")

conversion = pd.read_csv(conversion)
train = pd.merge(train, conversion, on="category_name", how="left")

# In[21]:


temp = train["category_name_english"].value_counts().head(10)
#print("Top 10 Fine grain ad category as classified by Avito's ad mode", temp)
print("Total Fine grain ad category as classified by Avito's ad mode : ",len(train["title"]))
trace = go.Bar(
    x = temp.index,
    y = (temp / temp.sum())*100,
)
data = [trace]
layout = go.Layout(
    title = "Top 10 Fine grain ad category as classified by Avito's ad mode",
    xaxis=dict(
        title='Fine grain ad category as classified by Avitos ad mode',
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
    ),
    yaxis=dict(
        title='Count of Fine grain ad category in %',
        titlefont=dict(
            size=16,
            color='rgb(107, 107, 107)'
        ),
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
)
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)

# * **Top 5 Fine grain ad category as classified by Avito's ad mode :**
#  * Clothing, shoes and accessories - 26 %
#  * Children clothing and shoes - 25 %
#  * Childrens product and toys  - 9 %
#  * Apartments - 8 %
#  * Phones - 6 %

# ## <a id='ttlcmsd'>5.4.5 Top 10 Top level ad category as classified by Avito's ad model</a>

# In[22]:


conversion = StringIO("""
parent_category_name,parent_category_name_english
Личные вещи,Personal belongings
Для дома и дачи,For the home and garden
Бытовая электроника,Consumer electronics
Недвижимость,Real estate
Хобби и отдых,Hobbies & leisure
Транспорт,Transport
Услуги,Services
Животные,Animals
Для бизнеса,For business
""")

conversion = pd.read_csv(conversion)
train = pd.merge(train, conversion, on="parent_category_name", how="left")

# In[23]:


temp = train["parent_category_name_english"].value_counts().head(10)
print("Total Top level ad category as classified by Avito's ad model : ",len(train["title"]))
trace = go.Bar(
    x = temp.index,
    y = (temp / temp.sum())*100,
)
data = [trace]
layout = go.Layout(
    title = "Top 10 Top level ad category as classified by Avito's ad model",
    xaxis=dict(
        title='Top level ad category as classified by Avitos ad model',
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
    ),
    yaxis=dict(
        title='Count of Top level ad category in %',
        titlefont=dict(
            size=16,
            color='rgb(107, 107, 107)'
        ),
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
)
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)

# *** Top 5 Top level ad category as classified by Avito's ad model :**
# * Personal belongings - 46 %
# * For the home and garden - 12 %
# * Consumer electronics - 12 %
# * Real estate - 10 %
# * Hobbies & leisure - 6 %

# ## <a id='pvdpp'>5.5 Price V.S. Deal probability</a>

# In[24]:


plt.figure(figsize=(15,6))
plt.scatter(np.log(train.price), train.deal_probability)
plt.xlabel('Ad price')
plt.ylabel('deal probability')
plt.show()

# ## <a id='5-6'>5.6 Deal probability V.S.  Price for regions</a>

# In[25]:


populated_states = train[:100]

data = [go.Scatter(
    y = populated_states['deal_probability'],
    x = populated_states['price'],
    mode='markers+text',
    marker=dict(
        size= np.log(populated_states.price) - 2,
        color=populated_states['deal_probability'],
        colorscale='Portland',
        showscale=True
    ),
    text=populated_states['region_english'],
    textposition=["top center"]
)]
layout = go.Layout(
    title='Deal probability V.S.  Price for regions',
    xaxis= dict(title='Ad price'),
    yaxis=dict(title='deal probability')
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)

# ## <a id='dout'>5.7 Distribution of user type</a>

# In[26]:


temp = train['user_type'].value_counts()
labels = temp.index
sizes = (temp / temp.sum())*100
trace = go.Pie(labels=labels, values=sizes, hoverinfo='label+percent')
layout = go.Layout(title='Distribution of user type')
data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)

# * **Distribution of user types :**
#   * Private users constitutes 71.6 % data
#   * Comapny users constitutes 23.1 % data
#   * Shop users constitutes 5.35 % data

# ## <a id='mdodadr'>5.8 Monthly distribution of Ad prices in different regions </a>

# In[27]:


train['activation_date'] = pd.to_datetime(train['activation_date'])
train['month'] = train.activation_date.dt.month
pr = train.groupby(['region_english', 'month'])['price'].mean().unstack()
#pr = pr.sort_values([12], ascending=False)
f, ax = plt.subplots(figsize=(15, 20)) 
pr = pr.fillna(0)
temp = sns.heatmap(pr, cmap='Reds')
plt.show()

# * Highest Ad prices is in **Irkutsk oblast** region followed by **Krasnodar Krai** region.

# ## <a id='dorpdp'>5.9 Distribution of regions, per deal probability</a>

# In[28]:


plt.figure(figsize=(17,8))
boxplot = sns.boxplot(x="region_english", y="deal_probability", data=train)
boxplot.set(xlabel='', ylabel='')
plt.title('Distribution of regions, per deal probability', fontsize=17)
plt.xticks(rotation=80, fontsize=17)
plt.yticks(fontsize=17)
plt.xlabel('region name')
plt.ylabel('deal probability')
plt.show()

_, ax = plt.subplots(figsize=(17, 8))
sns.violinplot(ax=ax, x="region_english", y="deal_probability", data=train)
plt.title('Distribution of regions, per deal probability', fontsize=17)
plt.xticks(rotation=80, fontsize=17)
plt.yticks(fontsize=17)
plt.xlabel('region name')
plt.ylabel('deal probability')
plt.show()

# ## <a id='tkfad'>5.10 Top Keywords from Ad description</a>

# In[29]:


from wordcloud import WordCloud, STOPWORDS
names = test["description"][~pd.isnull(test["description"])]
#print(names)
wordcloud = WordCloud(max_font_size=50, width=600, height=300).generate(' '.join(names))
plt.figure(figsize=(15,8))
plt.imshow(wordcloud)
plt.title("Wordcloud for Ad description", fontsize=35)
plt.axis("off")
plt.show() 

# ## <a id='5-11'>5.11 Ad sequential number for user V.S. deal probability</a>

# In[44]:


plt.figure(figsize=(15,6))
plt.scatter(train.item_seq_number, train.deal_probability)
plt.xlabel('Ad sequential number for user')
plt.ylabel('deal probability')
plt.show()

# ## <a id='5-12'>5.12 Deal probability V.S. Ad sequential number for user for regions</a>

# In[49]:


populated_regions = train[:50]

data = [go.Scatter(
    y = populated_regions['deal_probability'],
    x = populated_regions['item_seq_number'],
    mode='markers+text',
    marker=dict(
        size= np.log(populated_regions.price) - 2,
        color=populated_regions['deal_probability'],
        colorscale='Portland',
        showscale=True
    ),
    text=populated_regions['region_english'],
    textposition=["top center"]
)]
layout = go.Layout(
    title='Deal probability V.S. Ad sequential number for user for regions',
    xaxis= dict(title='Ad sequential number for user'),
    yaxis=dict(title='deal probability')
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)

# ## <a id='5-13'>5.13 Number of words in description column</a>

# In[43]:


train["description"].fillna("NA", inplace=True)
train["desc_numOfWords"] = train["description"].apply(lambda x: len(x.split()))
temp = train["desc_numOfWords"].value_counts().head(80)
trace = go.Bar(
    x = temp.index,
    y = temp.values,
)
data = [trace]
layout = go.Layout(
    title = "Number of words in description column",
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
del train["desc_numOfWords"]

#  ## <a id='toapaa'>5.14 Time series Analysis</a>

# ## <a id='toadp'>5.14.1 Trend of Ad price</a>

# In[30]:


train.posted_time = pd.to_datetime(train['activation_date'])
train.index = pd.to_datetime(train['activation_date'])
plt.figure(figsize = (12, 8))
ax = train['price'].resample('w').sum().plot()
#ax = kiva_loans_data['funded_amount'].resample('w').sum().plot()
ax.set_ylabel('Price')
ax.set_xlabel('day-month')
ax.set_xlim((pd.to_datetime(train['activation_date'].min()), 
             pd.to_datetime(train['activation_date'].max())))
ax.legend(["Ad Price"])
plt.title('Trend of Ad price')

# In[31]:


df = pd.read_csv('../input/train.csv').set_index('activation_date')
df.index = pd.to_datetime(df.index)
#df.head()

# In[32]:


df.head()

# ## <a id='paetsd'>5.14.2 Price average every two days</a>

# In[33]:


df["price"].resample("2D").apply([np.mean]).plot()
plt.title("Price average every two days")
plt.ylabel("Price")

# ## <a id='paeed'>5.14.3 Price average every day</a>

# In[34]:


df["price"].resample("D").apply([np.mean]).plot()
plt.title("Price average every day")
plt.ylabel("Price")

# ## <a id='dapevtr'>5.14.4 Deal probability average every two days</a>

# In[35]:


df["deal_probability"].resample("2D").apply([np.mean]).plot()
plt.title("deal probability average every two days")
plt.ylabel("deal probability")

# ## <a id='dapevtttr'>5.14.5 Deal probability average every  days</a>

# In[36]:


df["deal_probability"].resample("D").apply([np.mean]).plot()
plt.title("deal probability average every day")
plt.ylabel("deal probability")

# ## <a id='tnodawdjshhs'>5.14.6 Total number of days a Ad was dispalyed when it was posted on particular day</a>

# In[37]:


periods_train['total_days'] = periods_train['date_to'] - periods_train['date_from']
periods_test['total_days'] = periods_test['date_to'] - periods_test['date_from']

# In[38]:


periods_train.head()

# In[39]:


periods_train['total_days_value'] = periods_train['total_days'].dt.days
#periods_train['total_days'], _ = zip(*periods_train['total_days'].map(lambda x: x.split(' ')))

# In[40]:


periods_train.head()

# In[41]:


#df = periods_train.set_index('activation_date')
periods_train.index = pd.to_datetime(periods_train['activation_date'])
plt.figure(figsize = (12, 8))
ax = periods_train['total_days_value'].resample('w').sum().plot()
#ax = kiva_loans_data['funded_amount'].resample('w').sum().plot()
ax.set_ylabel('Total days')
ax.set_xlabel('day-month')
ax.set_xlim((pd.to_datetime(periods_train['activation_date'].min()), 
             pd.to_datetime(periods_train['activation_date'].max())))
ax.legend(["Total number of days"])
plt.title('Total number of days a Ad was dispalyed when it was posted on particular day')

# ## <a id='fodsbdir'>5.14.7  frequency and pattern of ad activation date in train and test data</a>
# 

# In[42]:


temp = train["activation_date"].value_counts()
temp1 = test["activation_date"].value_counts()
trace0 = go.Bar(
    x = temp.index,
    y = (temp / temp.sum())*100,
    name = 'Ad activation dates in training data'
)
trace1 = go.Bar(
    x = temp1.index,
    y = (temp1 / temp1.sum())*100,
    name = 'Ad activation dates in test data'
)
data = [trace0, trace1]
layout = go.Layout(
    title = "frequency and pattern of ad activation date in train and test data",
    xaxis=dict(
        title='Ad Activation Date',
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
    ),
    yaxis=dict(
        title='frequency in %',
        titlefont=dict(
            size=16,
            color='rgb(107, 107, 107)'
        ),
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
)
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)

# * Most Ad activation date range is from 15 march 2017 to 28 March 2017 in training data and in test data the range is 12 April 2017 to 19 April 2017

# # <a id='bsc'>6. Brief summary and conclusion :</a>

# ## This is only a brief summary if want more details please go through my Notebook.

# *  ** Top 5 Ad titles are :**
#     *  Туфли (Shoes) - 28 %
#     *  Куртка(Jacket) - 12 %
#     * Пальто (Coat)  - 12 %
#     * Джинсы(Jeans) - 9 %
#     * Комбинезон (Overall) - 9 %
# * **Top 5 Ad cities :**
#   * Екатеринбург (Yekaterinburg) - 12 %
#   * Новосибирск (Novosibirsk) - 12 %
#   * Ростов-на-Дону  (Rostov-on-don) - 11 %
#   * Нижний Новгород  (Nizhny Novgorod) 10 %
#   * Челябинск (Chelyabinsk) - 10 %
# * ** Top 5 Ad regions :**
#   * verdlovsk oblast  - 17 %
#   * Rostov oblast - 12 %
#   * Tatarstan - 11 %
#   * Chelyabinsk oblast - 10 %         
#   * Nizhny Novgorod oblast  - 10 %
# * **Top 5 Fine grain ad category as classified by Avito's ad mode :**
#   * Clothing, shoes and accessories - 26 %
#   * Children clothing and shoes - 25 %
#   * Childrens product and toys  - 9 %
#   * Apartments - 8 %
#   * Phones - 6 %
# * ** Top 5 Top level ad category as classified by Avito's ad model :**
#    * Personal belongings - 46 %
#    * For the home and garden - 12 %
#    * Consumer electronics - 12 %
#    * Real estate - 10 %
#    * Hobbies & leisure - 6 %
# * **Distribution of user types :**
#   * Private users constitutes 71.6 % data
#   * Comapny users constitutes 23.1 % data
#   * Shop users constitutes 5.35 % data
# * Highest Ad prices is in **Irkutsk oblast** region followed by **Krasnodar Krai** region.  
# * Most Ad activation date range is from 15 march 2017 to 28 March 2017 in training data and in test data the range is 12 April 2017 to 19 April 2017

# # More to come. Stayed Tuned !!
