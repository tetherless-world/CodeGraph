#!/usr/bin/env python
# coding: utf-8

# <a id="0"></a> <br>
# ## Kernel Headlines
# 1. [Introduction and Crisp Methodology](#1)
# 2. [Data Analysis](#2)
#     1.  [imports](#3)
# 	1.  [Reading Data](#4)
# 	1.  [Features Descriptions](#5)
# 	1.  [ChannelGrouping_barchart](#6)
# 	1.  [date and visitStartTime_describe](#7)
# 	1.  [device_barchart](#8)
# 	1.  [geoNetwork_barchart](#9)
# 	1.  [socialEngagement_describe](#10)
# 	1.  [totals_line_violin](#11)
# 	1.  [visitNumber_line_violin_hist](#12)
# 	1.  [trafficSource_barchart](#13)
# 	1.  [fullVisitorId_qpercentile](#14)
# 	
# 3. [Compound Features](#15)
#      1.  [Churn Rate and Conversion Rate](#16)
# 	 1.  [revenue_datetime](#17)
# 	 1.  [device_revenue](#18)
# 4. [Basic Regression](#19)
# 5. [Preparing for More Evaluations and Tests](#20)
# 	 1.  [Investigation of Feature Importance](#21)

# <a id="1"></a> <br>
# #  1-INTRODUCTION AND CRISP METHODOLOGY
# 
# ![](https://www.kdnuggets.com/wp-content/uploads/crisp-dm-4-problems-fig1.png)
#     Crisp methodology is on the acceptable manners for data mining tasks. As it is belowed in the following figure, it contains three main parts should be passed to deliver a product to business
# *     Data cleaning
#         1. Understanding the business and data.
#         2. Try to comprehent the business and extract the data which is needed
#         3. Understand the dependencies between attributes. Analyzing the target variables. Handling missing values. Transforming data formats to standard data format.
# *     Data Modeling
#         1. Understanding the business and data.
#         2. Selecting more accurate classfier or regression engine based on the charactristic any of them have.
#         3. Train a model 
# *     Evaluation and Deployment.
#         1. Evalute created model using evaluation methods (test-data, cross-validation, etc)
#         2. Catrefully Evaluate model with real data (i.e AB testing) (As it is shown in crisp diagram, there is a link between business undestanding and evaluation part). 
#         3. Migrate to new model and replace the old one with new version.
# 

# <a id="1"></a> <br>
# #  2-DATA ANALYSIS
# 
# 
# <a id="3"></a> <br>
# * **A. IMPORTS**
# 
# Importing packages and libraries.

# In[ ]:


import os
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import datetime as datetime
from datetime import timedelta, date
import seaborn as sns
import matplotlib.cm as CM
import lightgbm as lgb
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, train_test_split

# <a id="4"></a> <br>
# * **B. READING DATA**
# 
# Reading data and caughting a glimpse of what data it is.

# In[ ]:


train_data = pd.read_csv("../input/train_v2.csv",nrows=700000)
train_data.head()

# In[ ]:


train_data.describe()

# In[ ]:


list(train_data.columns.values)

# <a id="5"></a> <br>
# * **C. FEATURES DESCRIPTION**
# 
# Returning back to Data description for understanding features.
# 
# *     channelGrouping - The channel via which the user came to the Store.
# *     date - The date on which the user visited the Store.
# *     device - The specifications for the device used to access the Store.
# *     fullVisitorId- A unique identifier for each user of the Google Merchandise Store.
# *     geoNetwork - This section contains information about the geography of the user.
# *     sessionId - A unique identifier for this visit to the store.
# *     socialEngagementType - Engagement type, either "Socially Engaged" or "Not Socially Engaged".
# *     totals - This section contains aggregate values across the session.
# *     trafficSource - This section contains information about the Traffic Source from which the session originated.
# *     visitId - An identifier for this session. This is part of the value usually stored as the _utmb cookie. This is only unique to the user. For a completely unique ID, you should use a combination of fullVisitorId and visitId.
# *     visitNumber - The session number for this user. If this is the first session, then this is set to 1.
# *     visitStartTime - The timestamp (expressed as POSIX time).
# 

# <a id="6"></a> <br>
# * **D. CHANNEL_GROUPING**

# In[ ]:


train_data.channelGrouping.value_counts().plot(kind="bar",title="channelGrouping distro",figsize=(8,8),rot=25,colormap='Paired')

# <a id="7"></a> <br>
# * **E. DATE&VISIT_START_TIME**
# 
# There are two varialbe related to time and can be used in time dependent analyzes specially TimeSeries.

# In[ ]:


"date :{}, visitStartTime:{}".format(train_data.head(1).date[0],train_data.head(1).visitStartTime[0])

# date is stored in String and should be converted to pandas datetime format.
# visitStartTime is stored in epoch unix format and should be converted to pandas datetime format.
# doing the correspondence transforms and storing on the same attribute.

# In[ ]:


train_data["date"] = pd.to_datetime(train_data["date"],format="%Y%m%d")
train_data["visitStartTime"] = pd.to_datetime(train_data["visitStartTime"],unit='s')

# Checking the transformed features.

# In[ ]:


train_data.head(1)[["date","visitStartTime"]]

# <a id="8"></a> <br>
# * **F. DEVICE**
# 
# device is stored in json format. There is a need to extract its fields and analyze them. Using json library to deserializing json values.

# In[ ]:


list_of_devices = train_data.device.apply(json.loads).tolist()
keys = []
for devices_iter in list_of_devices:
    for list_element in list(devices_iter.keys()):
        if list_element not in keys:
            keys.append(list_element)

# keys existed in device attribute are listed below.
# Now we should ignore the features which are not usefull in rest of the process. If feature is misrelated, or it contains lot of "NaN" values it should be discarded.
# We select the ["browser","operatingSystem","deviceCategory","isMobile"] for doing the analyzing. The rest of the device features are ignored and will be removed.

# In[ ]:


"keys existed in device attribute are:{}".format(keys)

# In[ ]:


tmp_device_df = pd.DataFrame(train_data.device.apply(json.loads).tolist())[["browser","operatingSystem","deviceCategory","isMobile"]]

# In[ ]:


tmp_device_df.head()

# In[ ]:


tmp_device_df.describe()

# In[ ]:


fig, axes = plt.subplots(2,2,figsize=(15,15))
tmp_device_df["isMobile"].value_counts().plot(kind="bar",ax=axes[0][0],rot=25,legend="isMobile",color='tan')
tmp_device_df["browser"].value_counts().head(10).plot(kind="bar",ax=axes[0][1],rot=40,legend="browser",color='teal')
tmp_device_df["deviceCategory"].value_counts().head(10).plot(kind="bar",ax=axes[1][0],rot=25,legend="deviceCategory",color='lime')
tmp_device_df["operatingSystem"].value_counts().head(10).plot(kind="bar",ax=axes[1][1],rot=80,legend="operatingSystem",color='c')

# <a id="9"></a> <br>
# * **G. GEO_NETWORK**
# 
# It is json and the similar manner to previous feature (device) should be done.
# 

# In[ ]:


tmp_geo_df = pd.DataFrame(train_data.geoNetwork.apply(json.loads).tolist())[["continent","subContinent","country","city"]]

# In[ ]:


tmp_geo_df.head()

# In[ ]:


tmp_geo_df.describe()

# analysing the distribution of users in 5 continents.

# In[ ]:


fig, axes = plt.subplots(3,2, figsize=(15,15))
tmp_geo_df["continent"].value_counts().plot(kind="bar",ax=axes[0][0],title="Global Distributions",rot=0,color="c")
tmp_geo_df[tmp_geo_df["continent"] == "Americas"]["subContinent"].value_counts().plot(kind="bar",ax=axes[1][0], title="America Distro",rot=0,color="tan")
tmp_geo_df[tmp_geo_df["continent"] == "Asia"]["subContinent"].value_counts().plot(kind="bar",ax=axes[0][1], title="Asia Distro",rot=0,color="r")
tmp_geo_df[tmp_geo_df["continent"] == "Europe"]["subContinent"].value_counts().plot(kind="bar",ax=axes[1][1],  title="Europe Distro",rot=0,color="lime")
tmp_geo_df[tmp_geo_df["continent"] == "Oceania"]["subContinent"].value_counts().plot(kind="bar",ax = axes[2][0], title="Oceania Distro",rot=0,color="teal")
tmp_geo_df[tmp_geo_df["continent"] == "Africa"]["subContinent"].value_counts().plot(kind="bar" , ax=axes[2][1], title="Africa Distro",rot=0,color="silver")

# <a id="10"></a> <br>
# * **H.SOCIAL_ENGANEMENT_TYPE **
# 
# Describing this feature confirms its uniqueness. It should be dropped. Because its entropy is 0. 

# In[ ]:


train_data["socialEngagementType"].describe()

# <a id="11"></a> <br>
# * **I. TOTALS**
# 

# In[ ]:


train_data.head()
train_data["revenue"] = pd.DataFrame(train_data.totals.apply(json.loads).tolist())[["transactionRevenue"]]


# Extracting all the revenues can bring us an overview about the total revenue.

# In[ ]:


revenue_datetime_df = train_data[["revenue" , "date"]].dropna()
revenue_datetime_df["revenue"] = revenue_datetime_df.revenue.astype(np.int64)
revenue_datetime_df.head()

# Aggregation on days and plotting daily revenue.

# In[ ]:


daily_revenue_df = revenue_datetime_df.groupby(by=["date"],axis = 0 ).sum()
import matplotlib.pyplot as plt
fig, axes = plt.subplots(figsize=(20,10))
axes.set_title("Daily Revenue")
axes.set_ylabel("Revenue")
axes.set_xlabel("date")
axes.plot(daily_revenue_df["revenue"])


# In[ ]:


fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(9, 9))
axes.set_title("Daily revenue Violin")
axes.set_ylabel("revenue")
axes.violinplot(list(daily_revenue_df["revenue"].values),showmeans=False,showmedians=True)

# <a id="12"></a> <br>
# * **J. VISIT_NUMBER**
# 
# Number of visits have profound potential to be an important factor in regression progress. 

# In[ ]:


visit_datetime_df = train_data[["date","visitNumber"]]
visit_datetime_df["visitNumber"] = visit_datetime_df.visitNumber.astype(np.int64)

# In[ ]:


daily_visit_df = visit_datetime_df.groupby(by=["date"], axis = 0).sum()

fig, axes = plt.subplots(1,1,figsize=(20,10))
axes.set_ylabel("# of visits")
axes.set_xlabel("date")
axes.set_title("Daily Visits")
axes.plot(daily_visit_df["visitNumber"])

# In[ ]:


fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(9, 9))
axes.set_title("Daily visits Violin")
axes.set_ylabel("# of visitors")
axes.violinplot(list(daily_visit_df["visitNumber"].values),showmeans=False,showmedians=True)

# Now, lets check another side of 'visitNumber' feature. As it is mentioned in data description, visitNumber is the number of sessions for each user. It can also be the factor of users interest. lets 'describe' and  'visualize' them.
# using 'collections' package, we can count repetition of each element.

# In[ ]:


train_data.visitNumber.describe()

# The 75% of sessions have visitNumber lower than one time. You can get more information about percentiles by calling np.percentile method.

# In[ ]:


"90 percent of sessions have visitNumber lower than {} times.".format(np.percentile(list(train_data.visitNumber),90))

# Lets find most_common and least_common visitNumbers for being familiar with collections module and its powrefull tools ;-) 

# In[ ]:


import collections

tmp_least_10_visitNumbers_list = collections.Counter(list(train_data.visitNumber)).most_common()[:-10-1:-1]
tmp_most_10_visitNumbers_list = collections.Counter(list(train_data.visitNumber)).most_common(10)
least_visitNumbers = []
most_visitNumbers = []
for i in tmp_least_10_visitNumbers_list:
    least_visitNumbers.append(i[0])
for i in tmp_most_10_visitNumbers_list:
    most_visitNumbers.append(i[0])
"10 most_common visitNumbers are {} times and 10 least_common visitNumbers are {} times".format(most_visitNumbers,least_visitNumbers)

#  It is clear that the dispersion of the 'visitNumber' per session is huge. for this sort of features, we can use Log and map the feature space to
# new lower space. As a result of this mapping, visualization the data will be easier.

# In[ ]:


fig,ax = plt.subplots(1,1,figsize=(9,5))
ax.set_title("Histogram of log(visitNumbers) \n don't forget it is per session")
ax.set_ylabel("Repetition")
ax.set_xlabel("Log(visitNumber)")
ax.grid(color='b', linestyle='-', linewidth=0.1)
ax.hist(np.log(train_data.visitNumber))

# <a id="13"></a> <br>
# * **K. TRAFFIC_SOURCE**
# 
# What is the most conventional manner for visitor who visit to the website and do their shopping ? trafficSource attribute can resolve this qurestion.
# Like a previous Json elements existed in the dataset, this attribute is also Json file. so, we use the similar way to deserialize it. We have select keyword, source and the medium as a features which can bring more useful infromation.
# 

# In[ ]:


traffic_source_df = pd.DataFrame(train_data.trafficSource.apply(json.loads).tolist())[["keyword","medium" , "source"]]

# In[ ]:


fig,axes = plt.subplots(1,2,figsize=(15,10))
traffic_source_df["medium"].value_counts().plot(kind="bar",ax = axes[0],title="Medium",rot=0,color="tan")
traffic_source_df["source"].value_counts().head(10).plot(kind="bar",ax=axes[1],title="source",rot=75,color="teal")

# As it is completely obvious in source diagram, google is the most repetitive source. It would be interesting if we replace all google subdomains with exact 'google' and do the same analyze again. let's do it.

# In[ ]:


traffic_source_df.loc[traffic_source_df["source"].str.contains("google") ,"source"] = "google"
fig,axes = plt.subplots(1,1,figsize=(8,8))
traffic_source_df["source"].value_counts().head(15).plot(kind="bar",ax=axes,title="source",rot=75,color="teal")

# Google dependent redirects are more than twice the youtube sources. Combination of this feature with revenue and visits may have important result. We will do it in next step (when we are analyzing feature correlations).
# Now let's move on keywords feature.
# A glance to keyword featre represnets lot of missing values '(not provided)'. Drawing a bar chart for both of them...
# 

# In[ ]:


fig,axes = plt.subplots(1,2,figsize=(15,10))
traffic_source_df["keyword"].value_counts().head(10).plot(kind="bar",ax=axes[0], title="keywords (total)",color="orange")
traffic_source_df[traffic_source_df["keyword"] != "(not provided)"]["keyword"].value_counts().head(15).plot(kind="bar",ax=axes[1],title="keywords (dropping NA)",color="c")

# <a id="14"></a> <br>
# * **L. FULL_VISITOR_ID**
# 
# Now, lets see how many of users are repetitive ?! This feature can represent important information answering this question ? (Is more repeation proportional to more buy ?! ) 
# The response will be discussed in next section (Where we are analyzing compound features) but now, lets move on calculation of repetitive visits percentiles.

# In[ ]:


repetitive_users = list(np.sort(list(collections.Counter(list(train_data["fullVisitorId"])).values())))
"25% percentile: {}, 50% percentile: {}, 75% percentile: {}, 88% percentile: {}, 88% percentile: {}".format(
np.percentile(repetitive_users,q=25),np.percentile(repetitive_users,q=50),
np.percentile(repetitive_users,q=75),np.percentile(repetitive_users,q=88), np.percentile(repetitive_users,q=89))

# As it is shown, only 12 percent of users are repetitive and visited the website more than once. 
# (Search about churn rate and conversion rate if you want to know why we have analyzed this feature ;-) )
# 

# #  3-COMPOUND FEATURES
# 
# <a id="16"></a> <br>
# * **A. CHURN&CONVERSION_VISUALIZATION**
# 
# The main definition of ChurnRate is The percentage rate at which customers stop subscribing to a service or employees leave a job. Churn rate period can be various from day to a year correspoding to business type. In this section, we will compute and visualize the monthly churn rate. 
# Lets do more investigation on features date and fullVisitorId for more detail mining.
# 

# In[ ]:


date_list = np.sort(list(set(list(train_data["date"]))))
"first_day:'{}' and last_day:'{}' and toal number of data we have is: '{}' days.".format(date_list[0], date_list[-1],len(set(list(train_data["date"]))))

# So, we have 366 days (12 month = 1 year) from August 2016  to August 2017  data for churn rate calculations. The bes period for churn maybe is the monthly churn rate.
# Now, lets list all the months existed in library for checking having no missing months.

# In[ ]:


month = 8
start_date = datetime.date(2016, month, 1)
end_date = datetime.date(2017, month, 1)
def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        yield start_date + timedelta(n)
dates_month = []
for single_date in daterange(start_date, end_date):
    dates_month.append(single_date.strftime("%Y-%m"))
dates_month = list(set(dates_month))
dates_month

# Whole the period exist in data. So, Lets define new empty dataframe and do this calculations on it and copy the our requirements ot it.
# for churn rate calculations, we need to check which users visited have visited the website monthly. this information is located in fullVisitorId. we will copy it to new df.

# In[ ]:


tmp_churn_df = pd.DataFrame()
tmp_churn_df["date"] = train_data["date"]
tmp_churn_df["yaer"] = pd.DatetimeIndex(tmp_churn_df["date"]).year
tmp_churn_df["month"] =pd.DatetimeIndex(tmp_churn_df["date"]).month
tmp_churn_df["fullVisitoId"] = train_data["fullVisitorId"]
tmp_churn_df.head()

# For calculation of churn rate we need to count the users appeared in two, three, four, etc. continus months. 
# We will using the following format for collocation of users. 
# For example assume we want to extract the number of distinct users who visited the website on 2016-08.

# In[ ]:


"distinct users who visited the website on 2016-08 are:'{}'persons".format(len(set(tmp_churn_df[(tmp_churn_df.yaer == 2016) & (tmp_churn_df.month == 8) ]["fullVisitoId"])))

# By generalizing the above solution we have:

# In[ ]:


target_intervals_list = [(2016,8),(2016,9),(2016,10),(2016,11),(2016,12),(2017,1),(2017,2),(2017,3),(2017,4),(2017,5),(2017,6),(2017,7)]
intervals_visitors = []
for tmp_tuple in target_intervals_list:
    intervals_visitors.append(tmp_churn_df[(tmp_churn_df.yaer == tmp_tuple[0]) & (tmp_churn_df.month == tmp_tuple[1]) ]["fullVisitoId"])
"Size of intervals_visitors:{} ".format(len(intervals_visitors))

# So, we have 12 list and each elemets contains the users who visited the website on the correspondence period.
# Now its time to do some matrix calculation for filling the churn-rate matrix.
# It is very probable that you calculate the matrix with more efficient ways. I used this manner for more simplicity.

# In[ ]:


tmp_matrix = np.zeros((11,11))

for i in range(0,11):
    k = False
    tmp_set = []
    for j in range(i,11): 
        if k:
            tmp_set = tmp_set & set(intervals_visitors[j])
        else:
            tmp_set = set(intervals_visitors[i]) & set(intervals_visitors[j])
        tmp_matrix[i][j] = len(list(tmp_set))
        k = True

# Now we have 2D matrix containig the continus visited users.

# In[ ]:


xticklabels = ["interval 1","interval 2","interval 3","interval 4","interval 5","interval 6","interval 7","interval 8",
              "interval 9","interval 10","interval 11"]
yticklabels = [(2016,8),(2016,9),(2016,10),(2016,11),(2016,12),(2017,1),(2017,2),(2017,3),(2017,4),(2017,5),(2017,6),(2017,7)]
fig, ax = plt.subplots(figsize=(11,11))
ax = sns.heatmap(np.array(tmp_matrix,dtype=int), annot=True, cmap="RdBu_r",xticklabels=xticklabels,fmt="d",yticklabels=yticklabels)
ax.set_title("Churn-rate heatmap")
ax.set_xlabel("intervals")
ax.set_ylabel("months")


# A churn-rate heat map is the one the important keys of business. The more repetitive users in continues time periods, the more success in user loyalty.
# 
# Generaly it is better to drop the zeors below the main diagonal for better visualization and more clearer representaion. 
# (I couldn't find the sns pleasant visualization for half churn rate matrix. If you find it, it will be appreciated to ping me. Ill replace the below diagram with your recommendation as soon as possible).

# In[ ]:


A = tmp_matrix
mask =  np.tri(A.shape[0], k=-1)
A = np.ma.array(A, mask=mask) # mask out the lower triangle
fig = plt.figure(figsize=(9,9))
ax = fig.add_subplot(111)
ax.set_xlabel("interval")
ax.set_ylabel("period")
cmap = CM.get_cmap('RdBu_r', 50000) 
cmap.set_bad('w') # default value is 'k'
ax.imshow(A, interpolation="nearest", cmap=cmap)

# 
# <a id="17"></a> <br>
# * **B. REVENUE&DATETIME**
# 
# 
# Now, it is time to move on to analysing compound features. The main target of this section is undestanding the features correlation. 
# At the first point, lets analyze this probable assumption :
# 
# "Is more visitNumber proportional to more Revenue ?!"
# 

# In[ ]:


revenue_datetime_df = train_data[["revenue" , "date"]].dropna()
revenue_datetime_df["revenue"] = revenue_datetime_df.revenue.astype(np.int64)
revenue_datetime_df.head()

# Doing groupby on date and getting the total revenue per day:

# In[ ]:


total_revenue_daily_df = revenue_datetime_df.groupby(by=["date"],axis=0).sum()
total_revenue_daily_df.head()

# Doing similar process on visitNumber and getting total visitNumber per day.

# In[ ]:


total_visitNumber_daily_df = train_data[["date","visitNumber"]].groupby(by=["date"],axis=0).sum()
total_visitNumber_daily_df.head()

# Concatenate these two dataframe and compound visualization.

# In[ ]:


datetime_revenue_visits_df = pd.concat([total_revenue_daily_df,total_visitNumber_daily_df],axis=1)

fig, ax1 = plt.subplots(figsize=(20,10))
t = datetime_revenue_visits_df.index
s1 = datetime_revenue_visits_df["visitNumber"]
ax1.plot(t, s1, 'b-')
ax1.set_xlabel('day')
# Make the y-axis label, ticks and tick labels match the line color.
ax1.set_ylabel('visitNumber', color='b')
ax1.tick_params('y', colors='b')

ax2 = ax1.twinx()
s2 = datetime_revenue_visits_df["revenue"]
ax2.plot(t, s2, 'r--')
ax2.set_ylabel('revenue', color='r')
ax2.tick_params('y', colors='r')
fig.tight_layout()

# By comparing the revenue and visitNumbers we can confirm our consumption. Where there is more visit the revenue is also more than neighbour days.
# The behaviour of line charts is completely similar.
# 
# Another point to touch on is the rate of visitNumber and revenue  which have a peak on December. Before christmas people visit and buy more than other days. The same behaviour is represented in the days after christmas where people have bought their requirements and the level of visit and buy goes down (They are in vacation and have less time to check the website ;-) )
# 
# The above diagram can be represent more detail if you visualize period of it. Do it yourself with by quering on daterange and check our confirmed assumtion :-D.

# Lets focus on user who have addressed our challenge (users who have revenue) and checking some assumptions

# In[ ]:


revenue_df = train_data.dropna(subset=["revenue"])
revenue_os_df = pd.DataFrame(revenue_df.device.apply(json.loads).tolist())[["browser","operatingSystem","deviceCategory","isMobile"]]

buys_is_mobile_dict = dict(collections.Counter(list(revenue_os_df.isMobile)))
percent_buys_is_mobile_dict = {k: v / total for total in (sum(buys_is_mobile_dict.values()),) for k, v in buys_is_mobile_dict.items()}
sizes = list(percent_buys_is_mobile_dict.values())
explode=(0,0.1)
labels = 'isNotMobile', 'isMobile'
fig, ax = plt.subplots(1,1, figsize=(8,8))
ax.set_title("buys mobile distro")
ax.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',shadow=True, startangle=90)

# Going deeper to subclasses ...

# In[ ]:


mobiles_browsers = dict(collections.Counter(revenue_os_df[revenue_os_df["isMobile"] == True]["browser"]))
not_mobiles_browsers = dict(collections.Counter(revenue_os_df[revenue_os_df["isMobile"] == False]["browser"]))
print("for mobile users:")
for i,v in mobiles_browsers.items():
    print("{}:{}".format(i,v))
print("\nfor not mobile users:")
for i,v in not_mobiles_browsers.items():
    print("{}:{}".format(i,v))
vals = np.array([[552.,6.,2.,12.,16.,431.], [9801.,189.,58.,93.,5.,349.]])

fig, ax = plt.subplots(subplot_kw=dict(polar=True),figsize=(9,9))
size = 0.3
valsnorm = vals / np.sum(vals) * 2 * np.pi

# obtain the ordinates of the bar edges
valsleft = np.cumsum(np.append(0, valsnorm.flatten()[:-1])).reshape(vals.shape)

cmap = plt.get_cmap("tab20c")
outer_colors = cmap(np.arange(3) * 4)
inner_colors = cmap(np.array([1, 2, 5, 6, 9, 10]))

ax.bar(x=valsleft[:, 0],
       width=valsnorm.sum(axis=1), bottom=1 - size, height=size,
       color=outer_colors, edgecolor='w', linewidth=1, align="edge")

ax.bar(x=valsleft.flatten(),
       width=valsnorm.flatten(), bottom=1 - 2 * size, height=size,
       color=inner_colors, edgecolor='w', linewidth=1, align="edge")
# ax.set_axis_off()

ax.set(title="Nested pi-plot for buyers devices.")

# For buyers who use mobile for purchasing, Safari and chrome approximately have equal number of users. But for users who do not use mobile for purchasing, Safari have a few percent of total users. 

# <a id="18"></a> <br>
# #  4-BASIC REGRESSION
# 
# Now, based on the analysis we have done, it's time to start the regression progress.
# Lets do a simple regression for testing our regression method.
# Calling our main_df and dropping the features that may have not any positive effect on the regression results ( HINT: consider it is starting point. We want to learn how we can use the regression for combination of categorical features and continus features. We will change our features in next steps for getting better results).
# We use the tricks we used during this tutorial for dealing with features.

# In[ ]:


df_train = train_data.drop(["date", "socialEngagementType", "visitStartTime", "visitId", "fullVisitorId" , "revenue","customDimensions"], axis=1)

devices_df = pd.DataFrame(df_train.device.apply(json.loads).tolist())[["browser", "operatingSystem", "deviceCategory", "isMobile"]]
geo_df = pd.DataFrame(df_train.geoNetwork.apply(json.loads).tolist())[["continent", "subContinent", "country", "city"]]
traffic_source_df = pd.DataFrame(df_train.trafficSource.apply(json.loads).tolist())[["keyword", "medium", "source"]]
totals_df = pd.DataFrame(df_train.totals.apply(json.loads).tolist())[["transactionRevenue", "newVisits", "bounces", "pageviews", "hits"]]


df_train = pd.concat([df_train.drop(["hits"],axis=1), devices_df, geo_df, traffic_source_df, totals_df], axis=1)
df_train = df_train.drop(["device", "geoNetwork", "trafficSource", "totals"], axis=1)


# In[ ]:


df_train.head(1)

# Replacing NaN variables with 0 (It may have positive/negative effect. We will check it later).
# Another point we must touch on is we need to convert

# In[ ]:


df_train["transactionRevenue"] = df_train["transactionRevenue"].fillna(0)
df_train["bounces"] = df_train["bounces"].fillna(0)
df_train["pageviews"] = df_train["pageviews"].fillna(0)
df_train["hits"] = df_train["hits"].fillna(0)
df_train["newVisits"] = df_train["newVisits"].fillna(0)

# Using train_test_split for splitting data to train and evaluate sets. Converting revenue (our target variable) to float for performing regression.

# In[ ]:


df_train, df_test = train_test_split(df_train, test_size=0.2, random_state=42)

df_train["transactionRevenue"] = df_train["transactionRevenue"].astype(np.float)
df_test["transactionRevenue"] = df_test["transactionRevenue"].astype(np.float)
"Finaly, we have these columns for our regression problems: {}".format(df_train.columns)

# In[ ]:


df_train.head(1)

# Another point to touch on is we need to convert our categorical data to (int) values. As a result regression algorithm and classifiers can deal with these sort of features ( String features are not supported).
# (Search about LabelEncoder and its use case. This tool helps us to convert the categorical features to Integer ones).

# In[ ]:


categorical_features = ['channelGrouping', 'browser', 'operatingSystem', 'deviceCategory', 'isMobile',
                        'continent', 'subContinent', 'country', 'city', 'keyword', 'medium', 'source']

numerical_features = ['visitNumber', 'newVisits', 'bounces', 'pageviews', 'hits']

for column_iter in categorical_features:
    lbl = preprocessing.LabelEncoder()
    lbl.fit(list(df_train[column_iter].values.astype('str')) + list(df_test[column_iter].values.astype('str')))
    df_train[column_iter] = lbl.transform(list(df_train[column_iter].values.astype('str')))
    df_test[column_iter] = lbl.transform(list(df_test[column_iter].values.astype('str')))

for column_iter in numerical_features:
    df_train[column_iter] = df_train[column_iter].astype(np.float)
    df_test[column_iter] = df_test[column_iter].astype(np.float)

# OK. Now we have all requirements which are needed for first regression test ;-).
# 
# Check [this link](http://github.com/Microsoft/LightGBM/tree/master/examples/python-guide) out for get more information about Regression algorithm we are using.
# 
# It is mentioned in competition description that we need to use ln(x+1) for evaluation of results. So, check [this links](https://docs.scipy.org/doc/numpy-1.15.1/reference/generated/numpy.log1p.html) and [this link](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.expm1.html) about these scipy built-in methods.
# We can control prints with 'verbose_eval' parameter.

# In[ ]:


params = {
    "objective": "regression",
    "metric": "rmse",
    "num_leaves": 30,
    "min_child_samples": 100,
    "learning_rate": 0.1,
    "bagging_fraction": 0.7,
    "feature_fraction": 0.5,
    "bagging_frequency": 5,
    "bagging_seed": 2018,
    "verbosity": -1
}
lgb_train = lgb.Dataset(df_train.loc[:,df_train.columns != "transactionRevenue"], np.log1p(df_train.loc[:,"transactionRevenue"]))
lgb_eval = lgb.Dataset(df_test.loc[:,df_test.columns != "transactionRevenue"], np.log1p(df_test.loc[:,"transactionRevenue"]), reference=lgb_train)
gbm = lgb.train(params, lgb_train, num_boost_round=2000, valid_sets=[lgb_eval], early_stopping_rounds=100,verbose_eval=100)

# Congrate ! 
# You get reasonable rmse in first step. Now, lets predict the revenues for our evaluation dataset to be familar with transformation needed.
# 
# (Point that the revenue of user cant be negative ;-) so, remove them).

# In[ ]:


predicted_revenue = gbm.predict(df_test.loc[:,df_test.columns != "transactionRevenue"], num_iteration=gbm.best_iteration)
predicted_revenue[predicted_revenue < 0] = 0 
df_test["predicted"] = np.expm1(predicted_revenue)
df_test[["transactionRevenue","predicted"]].head(10)

# Now, you can generalize this procedure and use do your submission.
# 
# We will try to do a more supervised regression in next steps. We will utilize our preprocess facts for getting regression results with less error.

# <a id="19"></a> <br>
# #  5-PREPARING FOR MORE EVALUATIONS AND TESTS
# 
# Now, we have all requirements for doing more tests. Lets summarize all the prcess we have done earlier in one script. You can change parameters regarding your understanding about feature engineering process. If you cant understand any section of it, please read the details which are described below.

# In[ ]:


import gc; gc.collect()
import time; time.sleep(5)

df_train = pd.read_csv(filepath_or_buffer="../input/train_v2.csv",nrows=50000)
df_actual_test = pd.read_csv(filepath_or_buffer="../input/test_v2.csv",nrows=25000)

# drop useless features => date, fullVisitorId, sessionId, socialEngagement, visitStartTime
df_train = df_train.drop(["date", "socialEngagementType", "visitStartTime", "visitId", "fullVisitorId"], axis=1)
df_actual_test = df_actual_test.drop(["date", "socialEngagementType", "visitStartTime", "visitId"], axis=1)


#preprocessing for trains
devices_df = pd.DataFrame(df_train.device.apply(json.loads).tolist())[["browser", "operatingSystem", "deviceCategory", "isMobile"]]
geo_df = pd.DataFrame(df_train.geoNetwork.apply(json.loads).tolist())[["continent", "subContinent", "country", "city"]]
traffic_source_df = pd.DataFrame(df_train.trafficSource.apply(json.loads).tolist())[["keyword", "medium", "source"]]
totals_df = pd.DataFrame(df_train.totals.apply(json.loads).tolist())[["transactionRevenue", "newVisits", "bounces", "pageviews", "hits"]]
df_train = pd.concat([df_train.drop(["hits"],axis=1), devices_df, geo_df, traffic_source_df, totals_df], axis=1)
df_train = df_train.drop(["device", "geoNetwork", "trafficSource", "totals"], axis=1)
df_train["transactionRevenue"] = df_train["transactionRevenue"].fillna(0)
df_train["bounces"] = df_train["bounces"].fillna(0)
df_train["pageviews"] = df_train["pageviews"].fillna(0)
df_train["hits"] = df_train["hits"].fillna(0)
df_train["newVisits"] = df_train["newVisits"].fillna(0)

#preprocessing for tests
devices_df = pd.DataFrame(df_actual_test.device.apply(json.loads).tolist())[["browser", "operatingSystem", "deviceCategory", "isMobile"]]
geo_df = pd.DataFrame(df_actual_test.geoNetwork.apply(json.loads).tolist())[["continent", "subContinent", "country", "city"]]
traffic_source_df = pd.DataFrame(df_actual_test.trafficSource.apply(json.loads).tolist())[["keyword", "medium", "source"]]
totals_df = pd.DataFrame(df_actual_test.totals.apply(json.loads).tolist())[["newVisits", "bounces", "pageviews", "hits"]]
df_actual_test = pd.concat([df_actual_test.drop(["hits"],axis=1), devices_df, geo_df, traffic_source_df, totals_df], axis=1)
df_actual_test = df_actual_test.drop(["device", "geoNetwork", "trafficSource", "totals"], axis=1)
# df_actual_test["transactionRevenue"] = df_train["transactionRevenue"].fillna(0)
df_actual_test["bounces"] = df_train["bounces"].fillna(0)
df_actual_test["pageviews"] = df_train["pageviews"].fillna(0)
df_actual_test["hits"] = df_train["hits"].fillna(0)
df_actual_test["newVisits"] = df_train["newVisits"].fillna(0)

#garbage collector ';-)'
del devices_df,geo_df,traffic_source_df,totals_df


#evaluation 
df_train, df_eval = train_test_split(df_train, test_size=0.2, random_state=42)

# lgb_train = lgb.Dataset(df_train.loc[:, df_train.columns != "revenue"], df_train["revenue"])
# lgb_eval = lgb.Dataset(df_test.loc[:, df_test.columns != "revenue"], df_test["revenue"], reference=lgb_train)

df_train["transactionRevenue"] = df_train["transactionRevenue"].astype(np.float)
df_eval["transactionRevenue"] = df_eval["transactionRevenue"].astype(np.float)

print(df_train.columns)

# In[ ]:


params = {
    "objective": "regression",
    "metric": "rmse",
    "num_leaves": 30,
    "min_child_samples": 100,
    "learning_rate": 0.1,
    "bagging_fraction": 0.7,
    "feature_fraction": 0.5,
    "bagging_frequency": 5,
    "bagging_seed": 2018,
    "verbosity": -1
}

print('Start training...')

# In[ ]:


df_actual_test = df_actual_test.drop(["customDimensions"],axis=1)
df_train = df_train.drop(["customDimensions"],axis=1)
df_eval = df_eval.drop(["customDimensions"],axis=1)

# In[ ]:


categorical_features = ['channelGrouping', 'browser', 'operatingSystem', 'deviceCategory', 'isMobile',
                        'continent', 'subContinent', 'country', 'city', 'keyword', 'medium', 'source']

numerical_features = ['visitNumber', 'newVisits', 'bounces', 'pageviews', 'hits']

for column_iter in categorical_features:
    lbl = preprocessing.LabelEncoder()
    lbl.fit(list(df_train[column_iter].values.astype('str')) + list(df_eval[column_iter].values.astype('str')) + list(df_actual_test[column_iter].values.astype('str')))
    
    df_train[column_iter] = lbl.transform(list(df_train[column_iter].values.astype('str')))
    df_eval[column_iter] = lbl.transform(list(df_eval[column_iter].values.astype('str')))
    df_actual_test[column_iter] = lbl.transform(list(df_actual_test[column_iter].values.astype('str')))

for column_iter in numerical_features:
    df_train[column_iter] = df_train[column_iter].astype(np.float)
    df_eval[column_iter] = df_eval[column_iter].astype(np.float)
    df_actual_test[column_iter] = df_actual_test[column_iter].astype(np.float)

# In[ ]:


lgb_train = lgb.Dataset(df_train.loc[:,df_train.columns != "transactionRevenue"], np.log1p(df_train.loc[:,"transactionRevenue"]))
lgb_eval = lgb.Dataset(df_eval.loc[:,df_eval.columns != "transactionRevenue"], np.log1p(df_eval.loc[:,"transactionRevenue"]), reference=lgb_train)

# In[ ]:


gbm = lgb.train(params, lgb_train, num_boost_round=2000, valid_sets=[lgb_eval], early_stopping_rounds=100,verbose_eval=100)

# In[ ]:


eval_predicted_revenue = gbm.predict(df_eval.loc[:,df_eval.columns != "transactionRevenue"], num_iteration=gbm.best_iteration)
eval_predicted_revenue[eval_predicted_revenue < 0] = 0 
df_eval["predicted"] = np.expm1(eval_predicted_revenue)
df_eval[["transactionRevenue","predicted"]].head()

# In[ ]:


actual_predicted_revenue = gbm.predict(df_actual_test.loc[:,df_actual_test.columns != "fullVisitorId"], num_iteration=gbm.best_iteration)
actual_predicted_revenue[actual_predicted_revenue < 0] = 0 
# df_actual_test["predicted"] = np.expm1(actual_predicted_revenue)
df_actual_test["predicted"] = actual_predicted_revenue
df_actual_test.head()

df_actual_test = df_actual_test[["fullVisitorId" , "predicted"]]
df_actual_test["fullVisitorId"] = df_actual_test.fullVisitorId.astype('str')
df_actual_test["predicted"] = df_actual_test.predicted.astype(np.float)
df_actual_test.index = df_actual_test.fullVisitorId
df_actual_test = df_actual_test.drop("fullVisitorId",axis=1)

# In[ ]:


df_actual_test.head()

# In[ ]:


df_submission_test = pd.read_csv(filepath_or_buffer="../input/sample_submission_v2.csv",index_col="fullVisitorId")
df_submission_test.shape

# In[ ]:


"test shape is :{} and submission shape is : {}".format(df_actual_test.shape , df_submission_test.shape)
final_df = df_actual_test.loc[df_submission_test.index,:]

# In[ ]:


final_df = final_df[~final_df.index.duplicated(keep='first')]
final_df = final_df.rename(index=str, columns={"predicted": "PredictedLogRevenue"})
final_df.PredictedLogRevenue.fillna(0).head()
# final_df.head()

# We will try to do more supervisored regressions in next steps...

# <a id="20"></a> <br>
# * **A. INVESTIGATION OF FEATURE IMPORTANCE**
# 
# LightGBM have a method for representation of feature importance.
# 

# In[ ]:


fig, ax = plt.subplots(figsize=(10,16))
lgb.plot_importance(gbm, max_num_features=30, height=0.8, ax=ax)
plt.title("Feature Importance", fontsize=15)
plt.show()

# Now you have a information about which features are more important that others.
