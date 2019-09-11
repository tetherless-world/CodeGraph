#!/usr/bin/env python
# coding: utf-8

# # Support a classroom. Build a future - DonorsChoose.org
# 
# ![](http://i.huffpost.com/gen/1331818/images/o-ELEMENTARY-SCHOOL-TEACHER-facebook.jpg)
# DonorsChoose.org is a United States based nonprofit organization that allows individuals to donate directly to public school classroom projects. It was founded in 2000 by former public school teacher Charles Best, DonorsChoose.org was among the first civic crowdfunding platforms of its kind. The organization has been given Charity Navigator’s highest rating every year since 2005. In January 2018, they announced that 1 million projects had been funded.
# 
# Similar to Kiva.org, where people from over the world helped the poor for work, DonorsChoose is donation for student projects. Now what we have in the dataset, are the previous donations done on DonorsChoose, the donors , amount donated, etc. In this notebook, we will try to explore this dataset, and try to find any interesting patterns hidden in this trove of data.
# 
# 

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import os
print(os.listdir("../input"))
import folium
import folium.plugins
from folium import IFrame
# Any results you write to the current directory are saved as output.

# In[2]:


donations = pd.read_csv('../input/io/Donations.csv')
donors = pd.read_csv('../input/io/Donors.csv', low_memory=False)
schools = pd.read_csv('../input/io/Schools.csv', error_bad_lines=False)
teachers = pd.read_csv('../input/io/Teachers.csv', error_bad_lines=False)
projects = pd.read_csv('../input/io/Projects.csv', error_bad_lines=False, warn_bad_lines=False, parse_dates=["Project Posted Date","Project Fully Funded Date"])
resources = pd.read_csv('../input/io/Resources.csv', error_bad_lines=False, warn_bad_lines=False)

# # Donations Received

# In[3]:


donations.head()

# In[4]:


print('Minimum Donated Amount is: $',donations['Donation Amount'].min())
print('Maximum Donated Amount is: $',donations['Donation Amount'].max())
print('Average Donated Amount is: $',donations['Donation Amount'].mean())
print('Median Donated Amount is: $',donations['Donation Amount'].median())

# The reason for showing the median is because of the skewness of the data. As you can see that the lowest donation is **$0.01 **. While the maximum donation is **60000**. The median is handy measure to describe data with a significant skew or long tail. For example, if we look at incomes, a small number of people take home multi-million dollar salaries. These outliers carry more weight in the calculation of the mean than they do in the median calculation. Mean income is higher than median income. The median income would be closer to something we associate with middle-class. Means are great when the distribution is something like normally distributed. 
# 
# Thus in this case, the **median donation** amount will give a better idea about how much do people generally donate. Lets check the distribution. 

# In[5]:


f,ax=plt.subplots(1,2,figsize=(18,6))
donations[donations['Donation Amount']>0]['Donation Amount'].hist(bins=20,ax=ax[0])
donations[(donations['Donation Amount']>0.0)&(donations['Donation Amount']<100)]['Donation Amount'].hist(bins=20,ax=ax[1])
ax[0].set_title('Donations Distribution')
ax[1].set_title('Donations Distribution')

# As said earlier,  the donations data is highly skewed, and thus the left histogram doesn't show anything great. Thus we have the histogram on the right, where we have donations < $100. This histogram is a bit significant. As seen earlier, frequency of 25 dollars donations is the highest, followed by  50 and 100 dollar donations.

# ## Optional Amount Included?
# 
# Now what is this Optional Amount? Since DonorsChoose is a non-profit organisation, there is no primary source of income for them. Thus they ask the donors to dedicate **15%** of each donation to support the work they do. This enables them to pay their bills, their rent, and their employees. Lets see how many donors even help the organisation.

# In[6]:


sns.countplot(donations['Donation Included Optional Donation'])
plt.title('Optional Amount Included??')

# This looks great. About **75-77%** people not only help the students, but also help the organisation in covering their expenditures. This is really a great sign of appreciation. 

# # Grateful Donors

# In[7]:


print('DonorsChoose.org has received donations from:',donors.shape[0],'donors')
repeat=donations['Donor ID'].value_counts().to_frame()
print('There are',repeat[repeat['Donor ID']>1].shape[0],'repeating donors')

# So we have a good number of donors,  but the number of repeating donors are very less, which is just about **27%** of the total donors.

# ## Donations by States and Cities

# In[8]:


f,ax=plt.subplots(1,2,figsize=(18,12))
donors['Donor State'].value_counts()[:10].plot.barh(ax=ax[0],width=0.9,color=sns.color_palette('viridis_r',10))
donors['Donor City'].value_counts()[:10].plot.barh(ax=ax[1],width=0.9,color=sns.color_palette('viridis_r',10))
for i, v in enumerate(donors['Donor State'].value_counts()[:10].values): 
    ax[0].text(.5, i, v,fontsize=18,color='white',weight='bold')
ax[0].invert_yaxis()
ax[0].set_title('Top Donating States')
for i, v in enumerate(donors['Donor City'].value_counts()[:10].values): 
    ax[1].text(.5, i, v,fontsize=18,color='white',weight='bold')
ax[1].invert_yaxis()
ax[1].set_title('Top Donating Cities')

# State wise **California** has the highest number of donors, and it is well ahead from **New York**, with the number of donors being more than **double** from that of New York.
# 
# However in terms of city, **Chicago, Illinois** ranks 1, followed by **New York City**. The first **Californian** city comes at number 5 i.e **San Francisco**.
# 
# As California is the 3rd largest state in USA, it obviously has more cities, thus the number of donors might be well distributed among the cities.

# ## Are the Donors Teachers?

# In[9]:


don_amt=donors.merge(donations,left_on='Donor ID',right_on='Donor ID',how='left')
donor=donors['Donor Is Teacher'].value_counts().to_frame()
donation=don_amt['Donor Is Teacher'].value_counts().to_frame()
donor=donor.merge(donation,left_index=True,right_index=True,how='left')
donor.rename({'Donor Is Teacher_x':'Donors','Donor Is Teacher_y':'Donations'},axis=1,inplace=True)
donor.plot.bar()
plt.title('Are Donors Teachers??')
plt.gcf().set_size_inches(8,4)

# Couple of things to note in this plot. The number of Donors who are teachers is very less i.e around **12%**. But the total donations made by the teachers is about **39%** of the total number of donations. Lets see the mean donations donated by Teachers and other professionals.
# 
# ## Trend in Donations By Teachers 

# In[10]:


trend=donations[['Donor ID','Donation Amount','Donation Received Date']].merge(donors,left_on='Donor ID',right_on='Donor ID',how='left')
trend['year']=trend['Donation Received Date'].apply(lambda x:x[:4])
f,ax=plt.subplots(1,2,figsize=(28,8))
sns.heatmap(trend.groupby(['year','Donor Is Teacher'])['Donor ID'].count().reset_index().pivot('Donor Is Teacher','year','Donor ID'),fmt='2.0f',annot=True,cmap='RdYlGn',ax=ax[1])
trend.groupby(['year','Donor Is Teacher'])['Donor ID'].count().reset_index().pivot('year','Donor Is Teacher','Donor ID').plot.bar(width=0.9,ax=ax[0])

# ## No of Donations Received By States Each Year

# In[11]:


locations=pd.read_csv('../input/usa-latlong-for-state-abbreviations/statelatlong.csv')

# In[12]:


trend_don=trend.groupby(['year','Donor State'])['Donor City'].count().reset_index()
trend_don=trend_don.merge(locations,left_on='Donor State',right_on='City',how='left')
trend_don.dropna(inplace=True)
trend_don.rename({'Donor City':'Donations'},inplace=True,axis=1)
from mpl_toolkits.basemap import Basemap
from matplotlib import animation,rc
import io
import base64
from IPython.display import HTML, display
fig=plt.figure(figsize=(10,8))
fig.text(.3, .1, 'I,Coder', ha='right')
def animate(Year):
    ax = plt.axes()
    ax.clear()
    ax.set_title('Donations By State '+'\n'+'Year:' +str(Year))
    m6 = Basemap(llcrnrlon=-119,llcrnrlat=22,urcrnrlon=-64,urcrnrlat=49,projection='lcc',lat_1=33,lat_2=45,lon_0=-95)
    lat_gif1=list(trend_don[trend_don['year']==Year].Latitude)
    long_gif1=list(trend_don[trend_don['year']==Year].Longitude)
    x_gif1,y_gif1=m6(long_gif1,lat_gif1)
    m6.scatter(x_gif1, y_gif1,s=[donation for donation in trend_don[trend_don['year']==Year].Donations*0.05],color ='r') 
    m6.drawcoastlines()
    m6.drawcountries()
    m6.fillcontinents(color='burlywood',lake_color='lightblue', zorder = 1,alpha=0.4)
    m6.drawmapboundary(fill_color='lightblue')
ani = animation.FuncAnimation(fig,animate,list(trend_don.year.unique()), interval = 1500)    
ani.save('animation.gif', writer='imagemagick', fps=1)
plt.close(1)
filename = 'animation.gif'
video = io.open(filename, 'r+b').read()
encoded = base64.b64encode(video)
HTML(data='''<img src="data:image/gif;base64,{0}" type="gif" />'''.format(encoded.decode('ascii')))

# The above animation shows the number of donations contributed by each state per year from 2012-2018. The year 2012 saw negligible amount of donations. However the numbers have increased tremendously over the years. California,Texas and states in the eastern coast have majority of the donations.
# 
# ## Median/Mean Donations

# In[13]:


print('Average Donation By Teachers',don_amt[don_amt['Donor Is Teacher']=='Yes']['Donation Amount'].mean())
print('Median Donation By Teachers',don_amt[don_amt['Donor Is Teacher']=='Yes']['Donation Amount'].median())
print('Average Donation By Others',don_amt[don_amt['Donor Is Teacher']=='No']['Donation Amount'].mean())
print('Median Donation By Others',don_amt[don_amt['Donor Is Teacher']=='No']['Donation Amount'].median())

# It is evident that Teachers have donated lesser funds as compared to others.
# 
# ## Tracking The Highest Donor
# 
# Lets find out the most returning donor.

# In[14]:


print('Person with Donor ID',donors[donors['Donor ID']==donations['Donor ID'].value_counts().index[0]]['Donor ID'].values[0],'has the highest number of donations i.e: ',donations['Donor ID'].value_counts().values[0])

# Thats so generous. The Donor with the above mentioned Donor ID has **18035** donations in total. Lets see his full credentials.

# In[15]:


donors[donors['Donor ID']==donations['Donor ID'].value_counts().index[0]]

# This generous donor is from **Manhattan Beach, California**. Lets check his total amount donated.

# In[16]:


print(don_amt[don_amt['Donor ID']==donations['Donor ID'].value_counts().index[0]]['Donation Amount'].sum())

# The total amount donated by the donor is **37121.72$**. This comes around to about **2.1 dollars** per donation. This donor surely doesn't have the highest donation amount. Lets check the user with highest donation amount.

# In[17]:


donors[donors['Donor ID']==don_amt.groupby('Donor ID')['Donation Amount'].sum().sort_values(ascending=False).index[0]]

# In[18]:


don_amt.groupby('Donor ID')['Donation Amount'].sum().sort_values(ascending=False)[0]

# Wow!! Thats a humungous amount. So definetly there are 2 types of donors:
# 
# 1) Who donate a small amount but maybe very frequently.
# 
# 2) Who donate a very large amount at once. Lets check the trend of donations of both these donors.

# In[19]:


don_max=donations['Donor ID'].value_counts().index[0]
amt_max=don_amt.groupby('Donor ID')['Donation Amount'].sum().sort_values(ascending=False).index[0]
don=don_amt[don_amt['Donor ID'].isin([don_max,amt_max])]
don['Donation Year']=don['Donation Received Date'].apply(lambda x:x[:4]).astype('int')
f,ax=plt.subplots(1,2,figsize=(20,10))
year_c=don.groupby(['Donor ID','Donation Year'])['Donor State'].count().reset_index()
year_c.pivot('Donation Year','Donor ID','Donor State').plot(marker='o',ax=ax[0])
ax[0].set_title('Donations By Year')
amt_c=don.groupby(['Donor ID','Donation Year'])['Donation Amount'].sum().reset_index()
amt_c.pivot('Donation Year','Donor ID','Donation Amount').plot(marker='o',ax=ax[1])
ax[1].set_title('Donation Amount By Year')
plt.show()

# I didn't expect the above graph. The blue line is for the donor with the highest number of donations, where as the red line is for the donor with the highest total amount donated. Since the blue donor has such a high number of donations, I was expecting a longer blue line right from 2013 till 2018. But as we can see, this donor has started donating from 2016, with the peak almost around 14000 donations in the year 2017. On the other hand, the donations from the red donor has been decreasing over the years, with the total donated amount falling even sharply. 

# ## Creating A User Network
# 
# It is possible that the Donors might be a bunch of friends/colleagues who donate on the same projects together. Here I will try to give a very naive approach for finding such group of donors. What we will do is find a group of 2-3 users, who have a high number of common unique project donations. That means lets say I have a donated for 10 projects, and you have donated in 5 of these 10 projects, so we will be a group of donors. This idea was proposed by **Mhamed Jabri.** Thanks Mhamed.
# 
# I am going to consider the user with the highest total donation amount till now. We will try to find some donors who have many common project donations as this user.

# In[20]:


import itertools
ID='a0e1d358aa17745ff3d3f4e4909356f3'
n=1000
Ux=don_amt.loc[don_amt['Donor ID']==ID,'Project ID'].unique().tolist()
s=don_amt.loc[don_amt['Project ID'].isin(Ux)&~don_amt['Donor ID'].isin([ID]),].groupby('Donor ID')['Project ID'].count()
s1=s[s>=n].index.tolist()

d=don_amt.loc[don_amt['Donor ID'].isin(s1)&don_amt['Project ID'].isin(Ux)]
d=d.drop_duplicates(subset=['Donor ID','Project ID'])
d=d.groupby('Donor ID')['Project ID'].apply(list).to_dict()
key_to_value_lengths = {k:len(v) for k, v in d.items()}
keys=list(d.keys())
new_list=[d[k] for k in keys]
new_list=list(itertools.chain.from_iterable(new_list))

key_to_value_lengths

# So here is a list of donors who share common projects with our target user. Lets map them. 

# In[21]:


zipcodes=pd.read_csv('../input/usa-zip-codes-to-locations/US Zip Codes from 2013 Government Data.csv')
mapping=don_amt[don_amt['Project ID'].isin(new_list)&don_amt['Donor ID'].isin(keys)]
mapping=mapping.drop_duplicates(subset=['Project ID','Donor ID'])
mapping=mapping[['Donor ID','Project ID']].merge(projects[['Project ID','School ID','Project Subject Category Tree']],left_on='Project ID',right_on='Project ID',how='left')
mapping=mapping.merge(schools[['School ID','School Name','School Metro Type','School Zip','School City']],left_on='School ID',right_on='School ID',how='left')
mapping=mapping.merge(zipcodes,left_on='School Zip',right_on='ZIP',how='left')

# In[22]:


locate=mapping[['LAT','LNG']]
city=mapping['School City']
name=mapping['School Name']
metro=mapping['School Metro Type']
cat=mapping['Project Subject Category Tree']
map1 = folium.Map(location=[39.50, -98.35],tiles='Mapbox Control Room',zoom_start=3.5)
folium.Marker(location=[34.0522,-118.2437],popup='<b>User</b>: 0e345dcdef0d2a36c9bd17bf1ac3e10a').add_to(map1)
folium.Marker(location=[39.9509,-86.2619],popup='<b>User</b>: 237db43817f34988f9d543ca518be4ee').add_to(map1)
folium.Marker(location=[35.1495,-90.0490],popup='<b>User</b>: a1929a1172ad0b3d14bc84f54018c563').add_to(map1)
folium.Marker(location=[40.7128,-74.0060],popup='<b>User</b>: a0e1d358aa17745ff3d3f4e4909356f3',icon=folium.Icon(color='red',icon='info-sign')).add_to(map1)
for point in mapping.dropna().index:
    info='<b>School Name: </b>'+str(name.loc[point])+'<br><b>City: </b>'+str(city.loc[point])+'<br><b>Project Category: </b>'+str(cat.loc[point])+'<br><b>School Metro: </b>'+str(metro.loc[point])
    iframe = folium.IFrame(html=info, width=250, height=250)
    folium.CircleMarker(list(locate.loc[point]),popup=folium.Popup(iframe),radius=0.5,color='red').add_to(map1)
map1

# The markers are for the donors, with the **red marker** for the user with the highest total donation amount. The donors are not from the same state. Majority of their donations are for the projects with categories like **Literacy and Language, followed by Music and Arts**. Majority of their donations are for **urban** schools. This approach is very basic, so I wish someone can take it ahead and make something more informative.
# 

# ## Mapping Donors By State

# In[23]:


state_don=don_amt.groupby('Donor State')['Donation Amount'].sum().to_frame()
state_num=donors['Donor State'].value_counts().to_frame()
map_states=state_don.merge(state_num,left_index=True,right_index=True,how='left')
map_states=locations.merge(map_states,left_on='City',right_index=True,how='left')
map1 = folium.Map(location=[39.50, -98.35],tiles='Mapbox Control Room',zoom_start=3.5)
locate=map_states[['Latitude','Longitude']]
count=map_states['Donor State']
state=map_states['City']
amt=map_states['Donation Amount'].astype('int')
for point in map_states.index:
    folium.CircleMarker(list(locate.loc[point]),popup='<b>State: </b>'+str(state.loc[point])+'<br><b>No of Donors: </b>'+str(count.loc[point])+'<br><b>Total Funds Donated: </b>'+str(amt.loc[point])+' <b>Million $<br>',radius=count.loc[point]*0.0002,color='red',fill=True).add_to(map1)
map1

# #### Click on the markers for more information
# 
# The above map shows **No of donors, Total Donated Amount** statewise. The size of the markers is based on the **Total Donations By States**. Lets drill down by cities.
# 
# ## Mapping Donors By City

# In[24]:


cities=pd.read_csv('../input/cities/cities2.csv')
city_don=don_amt.groupby('Donor City')['Donation Amount'].sum().to_frame()
city_num=donors['Donor City'].value_counts().to_frame()
city_don=city_don.merge(city_num,left_index=True,right_index=True,how='left')
city_don.columns=[['Amount','Donors']]
map_cities=cities[['city','lat','lng']].merge(city_don,left_on='city',right_index=True)
map_cities.columns=[['City','lat','lon','Amount','Donors']]
map2 = folium.Map(location=[39.50, -98.35],tiles='Mapbox Control Room',zoom_start=3.5)
locate=map_cities[['lat','lon']]
count=map_cities['Donors']
city=map_cities['City']
amt=map_cities['Amount']
def color_producer(donors):
    if donors < 100:
        return 'red'
    else:
        return 'yellow'
for point in map_cities.index:
    info='<b>City: </b>'+str(city.loc[point].values[0])+'<br><b>No of Donors: </b>'+str(count.loc[point].values[0])+'<br><b>Total Funds Donated: </b>'+str(amt.loc[point].values[0])+' <b>$<br>'
    iframe = folium.IFrame(html=info, width=250, height=250)
    folium.CircleMarker(list(locate.loc[point]),popup=folium.Popup(iframe),radius=amt.loc[point].values[0]*0.000005,color=color_producer(count.loc[point].values[0]),fill_color=color_producer(count.loc[point].values[0]),fill=True).add_to(map2)
map2

# The above map shows the number of donors, donated amount by their respective cities. The color of the markers represent the number of donors in that city, i.e **red for number of donors <100 and green for >100.** While the size is proportional to the total amount donated till date. The main motive of plotting the map was to see if there are any areas with low number of donors, but higher donation amount, i.e **larger red markers**. However, I don't see any such city. Thus the total amount donated to the orgainsation is proportional to the number of donors. 

# # Teachers
# 
# The projects submitted on DonorsChoose are from the teachers from various schools. Lets see the trend in the number of projects posted by first timer's i.e  Teachers posting their first project on the website.

# In[25]:


f,ax=plt.subplots(figsize=(15,8))
year_pro=teachers['Teacher First Project Posted Date'].apply(lambda x: x[:4]).astype('int').value_counts().reset_index()
sns.barplot(y='Teacher First Project Posted Date',x='index',data=year_pro,palette=sns.color_palette('RdYlGn',20))
plt.title('Projects Posted By Year')

# As we know DonorsChoose was established in 2000, the initial years had very little number of projects. After the year **2006**, the number of projects posted have gone up significantly, with the maximum numbers in the Year **2016**. I hope the year 2018 would have even higher numbers than 2016.
# 
# Don't misinterpret the graph with number of projects posted per year. This graph shows **Number of projects posted by a new teacher** by year. Thus DonorsChoose is attracting new teachers to post their projects and get some funding.

# ## Teacher with the highest projects posted in the last 5 years.

# In[26]:


print('Teacher with Teacher ID',projects['Teacher ID'].value_counts().index[0],'has posted the maximum number of projects i.e:',projects['Teacher ID'].value_counts()[0])
print('First Project Posted by the Teacher was on:',teachers[teachers['Teacher ID']==projects['Teacher ID'].value_counts().index[0]]['Teacher First Project Posted Date'].values[0])

# In[27]:


max_pro=projects[projects['Teacher ID']==projects['Teacher ID'].value_counts().index[0]]
max_pro['year']=max_pro['Project Posted Date'].dt.year
f,ax=plt.subplots(1,2,figsize=(25,10))
pro_year=max_pro.groupby('year')['Project Title'].count().reset_index()
sns.barplot(y='Project Title',x='year',data=pro_year,palette=sns.color_palette('inferno_r',5),ax=ax[0])
ax[0].set_title('Projects posted by Year')
max_pro['Project Current Status'].value_counts().plot.pie(autopct='%1.1f%%',colors=sns.color_palette('Paired',10),ax=ax[1],startangle=90)
my_circle=plt.Circle( (0,0), 0.7, color='white')
p=plt.gcf()
p.gca().add_artist(my_circle)
plt.ylabel('')
ax[1].set_title('Project Status')

# Okay so we get few points about this Teacher. She had posted her first project on **13/04/2008**. In the last 5 years, she has posted the highest number of projects i.e **320**. But since the data is limited to just 5 years, we can't tell the exact number of projects she has actually posted. Her number of project postings has been increasing since 2013, but with a plunge in 2014. A very positive thing to note is that **99%** of her projects get fully funded, and only  **0.5%** have expired.  There can be many reasons for this, the school may have very promising donors

# In[28]:


max_pro['Project Cost'].median()

# The median cost of projects for this teacher is $274.11. We will come back to this point later.

# # Projects
# 
# This dataset has data the Projects posted in last 5 years. Lets analyse and see if we can find some interesting insights.

# In[29]:


#projects['Project Cost']=projects['Project Cost'].str.replace(',','').str.strip('$').astype('float')
print('Total Number of Projects Proposed in the last 5 Years: ',projects.shape[0])

# ## Projects Posted By Years

# In[30]:


f,ax=plt.subplots(1,2,figsize=(25,10))
sns.barplot(ax=ax[0],x=projects['Project Posted Date'].dt.year.value_counts().index,y=projects['Project Posted Date'].dt.year.value_counts().values,palette=sns.color_palette('inferno',20))
ax[0].set_title('Number of Projects Posted in the last 5 Years')
sns.barplot(ax=ax[1],x=projects['Project Posted Date'].dt.month.value_counts().index,y=projects['Project Posted Date'].dt.month.value_counts().values,palette=sns.color_palette('inferno',20))
ax[1].set_title('Number of Projects Posted by Months')

# As expected, the number of projects posting is increasing in the last 5 years. We see a high jump in the Year 2016. We had seen earlier that the number of new teachers posting had gone up in 2016, thus it was expected that the year 2016 would have a great jump compared to the last year. Also month wise, August and September have high numbers. 

# In[31]:


plt.figure(figsize=(25,10))
projects['Project Posted Date'].value_counts().plot()
plt.axvline('2018-03-30',color='red', linestyle='-',linewidth=15,alpha = 0.3)
plt.axvline('2015-09-13',color='red', linestyle='-',linewidth=10,alpha = 0.3)

# Here too I see regular spikes in the months of August and September. I would like to thank **Wei Chun Chang** for suggesting to highlight the dates with highest posting dates.
# 
# ## Time Taken to Fund the Project Completely

# In[32]:


funding_time=projects[projects['Project Current Status']=='Fully Funded']
funding_time['Project Fully Funded Date']=pd.to_datetime(funding_time['Project Fully Funded Date'])
funding_time['Project Posted Date']=pd.to_datetime(funding_time['Project Posted Date'])
funding_time['Days Elapsed']=funding_time['Project Fully Funded Date']-funding_time['Project Posted Date']
funding_time['Days Elapsed']=funding_time['Days Elapsed'].dt.days
plt.figure(figsize=(18,6))
funding_time['Days Elapsed'].hist(bins=20,edgecolor='black')
plt.xlabel('Days')
plt.title('Distribution of time for Full Project Funding')

# I have considered only fully funded projects for the distribution. As we can see, majority of the projects get funded within 20-30 days i.e within a month. The maximum number of days that a project took to get funded is 238.

# ## Donations received by Years/Months

# In[33]:


f,ax=plt.subplots(1,2,figsize=(25,12))
don_date=donations['Donation Received Date'].apply(lambda x:x[:4]).value_counts().to_frame().sort_index()
don_date.plot.barh(width=0.9,ax=ax[0])
for i, v in enumerate(don_date['Donation Received Date']): 
    ax[0].text(.5, i, v,fontsize=18,color='red',weight='bold')
ax[0].set_title('Donations Received By Years')
don_date=donations['Donation Received Date'].apply(lambda x:x[5:7]).value_counts().to_frame().sort_index()
don_date.plot.barh(width=0.9,ax=ax[1])
for i, v in enumerate(don_date['Donation Received Date']): 
    ax[1].text(.5, i, v,fontsize=18,color='red',weight='bold')
ax[1].set_title('Donations Received By Months')

# The donations have increased steadily over time. Similar to previous graph on projects posting, the months of August and September have higher amount of donations. As we saw earlier that most projects get funded in less than a month, it was expected. More projects are posted in August and September, and they funded within those months itself. One thing to note is for the year 2016. Even though the number of projects did shoot up in 2016, the number of donations didn't go exceedingly high.
# 
# ## Project Status

# In[34]:


plt.figure(figsize=(6,6))
projects['Project Current Status'].value_counts().plot.pie(autopct='%1.1f%%',colors=sns.color_palette('Set2',10))
my_circle=plt.Circle( (0,0), 0.7, color='white')
p=plt.gcf()
p.gca().add_artist(my_circle)
plt.ylabel('')
plt.title('Project Status')
plt.show()

# Great to see that 74.5% of the projects get fully funded. The expired projects are those who don't get the funding. Projects are archived due to following reasons:
# 
#  - A project draft has not been modified within 30 days.
#  - Action is needed for a project and the teacher has not responded to our communication attempts.
#  - A teacher removes the project from the “live” site or there are circumstances that prevent the project from moving forward.
# 

# ## Funded/Non Funded Projects by Year

# In[35]:


funded=funding_time['Project Posted Date'].dt.year.value_counts().to_frame()
not_funded=projects[(projects['Project Current Status']!='Live')&(projects['Project Current Status']!='Fully Funded')]['Project Posted Date'].dt.year.value_counts().to_frame()
funded=funded.merge(not_funded,left_index=True,right_index=True,how='left')
funded['total']=funded['Project Posted Date_x']+funded['Project Posted Date_y']
funded['% Funded']=funded['Project Posted Date_x']/funded['total']
funded['% Not Funded']=funded['Project Posted Date_y']/funded['total']
funded.drop(['Project Posted Date_x','Project Posted Date_y','total'],axis=1,inplace=True)
funded.sort_index(inplace=True)
funded.plot.barh(stacked=True,width=0.9)
fig=plt.gcf()
fig.set_size_inches(10,5)
plt.title('Projects Funded/ Not Funded By Years')

# We see an increasing trend in the %funded projects till 2015, but it falls in the year 2016. The lowest percent of funded projects as compared to other years is in 2016. One reason might be the increase in first time teachers. SInce the number of new teacher project postings increased in 2016, it might be possible that they were not able to address the main motto of their project or something else. 

# ## Cost Difference between Funded/Non Funded Projects

# In[36]:


not_funded=projects[(projects['Project Current Status']!='Live')&(projects['Project Current Status']!='Fully Funded')]
print('Median Cost of Non Funded Projects:',not_funded['Project Cost'].median(),'$')
print('Median Cost of Funded Projects:',funding_time['Project Cost'].median(),'$')
print('Mean Cost of Non Funded Projects:',not_funded['Project Cost'].mean(),'$')
print('Mean Cost of Funded Projects:',funding_time['Project Cost'].mean(),'$')

# It is evident that the cost of Non-Funded projects are pretty high as compared to funded projects. As I said earlier, the project cost might be a very important factor for project funding. This hypothesis looks to be strong with the above observation.

# ## Project Categories

# In[37]:


from sklearn.feature_extraction.text import CountVectorizer
f,ax=plt.subplots(1,2,figsize=(25,15))
vec = CountVectorizer(tokenizer=lambda x: [i.strip() for i in x.split(',')], lowercase=False)
counts = vec.fit_transform(projects['Project Subject Category Tree'].dropna())
count=dict(zip(vec.get_feature_names(), counts.sum(axis=0).tolist()[0]))
count=pd.DataFrame(list(count.items()),columns=['Category','Count'])
count.set_index('Category',inplace=True)
count.plot.barh(width=0.9,ax=ax[0])
ax[0].set_title('Project Categories')

counts = vec.fit_transform(projects['Project Subject Subcategory Tree'].dropna())
count=dict(zip(vec.get_feature_names(), counts.sum(axis=0).tolist()[0]))
count=pd.DataFrame(list(count.items()),columns=['Category','Count'])
count.set_index('Category',inplace=True)
count.plot.barh(width=0.9,ax=ax[1])
ax[1].set_title('Project Sub Categories')
ax[1].set_ylabel('')

# As the donations are for helping school students, it was expected that projects will be based on teaching literature,maths and other subjects, technology,etc. Similar observations can be made from the above graphs.

# ## No of Projects Posted By States By Year

# In[38]:


school_pro=projects[['School ID','Project Posted Date','Project Current Status']].merge(schools[['School ID','School Name','School Metro Type','School State','School County']],left_on='School ID',right_on='School ID',how='left')
school_pro['year']=school_pro['Project Posted Date'].dt.year

state_pro=school_pro.groupby(['School State','year'])['Project Posted Date'].count().reset_index()
state_pro=state_pro.merge(locations,left_on='School State',right_on='City',how='left')
state_pro.rename({'Project Posted Date':'Count'},axis=1,inplace=True)

# In[39]:


sta_trend=state_pro[state_pro['School State'].isin(school_pro['School State'].value_counts()[:5].index)]
sta_trend.pivot('year','School State','Count').plot(marker='o')
plt.gcf().set_size_inches(16,5)
plt.title('Trend in No of Projects By State')

# There is a continous increase in the number of projects over the years. No sudden increase in any of the Top 5 states. Lets create an animation to see if we can spot a state with a big jump in the past years.

# In[40]:


fig=plt.figure(figsize=(10,8))
fig.text(.3, .1, 'I,Coder', ha='right')
def animate(Year):
    ax = plt.axes()
    ax.clear()
    ax.set_title('No of Projects Posted By States '+'\n'+'Year:' +str(Year))
    m6 = Basemap(llcrnrlon=-119,llcrnrlat=22,urcrnrlon=-64,urcrnrlat=49,projection='lcc',lat_1=33,lat_2=45,lon_0=-95)
    lat_gif1=list(state_pro[state_pro['year']==Year].Latitude)
    long_gif1=list(state_pro[state_pro['year']==Year].Longitude)
    x_gif1,y_gif1=m6(long_gif1,lat_gif1)
    m6.scatter(x_gif1, y_gif1,s=[donation for donation in state_pro[state_pro['year']==Year].Count*0.1],color ='r') 
    m6.drawcoastlines()
    m6.drawcountries()
    m6.fillcontinents(color='burlywood',lake_color='lightblue', zorder = 1,alpha=0.4)
    m6.drawmapboundary(fill_color='lightblue')
ani = animation.FuncAnimation(fig,animate,list(state_pro.year.unique()), interval = 1500)    
ani.save('animation.gif', writer='imagemagick', fps=1)
plt.close(1)
filename = 'animation.gif'
video = io.open(filename, 'r+b').read()
encoded = base64.b64encode(video)
HTML(data='''<img src="data:image/gif;base64,{0}" type="gif" />'''.format(encoded.decode('ascii')))


# The above animation shows the number of projects posted as per states by year. It looks similar to the previous animation we saw for number of donations by state. California, Texas and the states on the east coast have many project postings over the year. By looking at both the animations, it seems that many of the projects posted get funded by their home state itself, i.e project posted from a California based school gets funding from a donor from California itself. We need to check this further, and I will leave this for the future sections.
# 
# 

# 
# # Schools
# 
# Lets explore the schools who post their projects on DonorsChoose.org
# 

# In[41]:


print('Total Number of Schools are:',schools.shape[0])

# ## State with maximum number of schools

# In[42]:


plt.figure(figsize=(10,8))
ax=schools['School State'].value_counts()[:10].plot.barh(width=0.9,color=sns.color_palette('winter_r',20))
plt.gca().invert_yaxis()
for i, v in enumerate(schools['School State'].value_counts()[:10].values): 
    ax.text(.5, i, v,fontsize=10,color='black',weight='bold')
plt.title('States with Number of Schools')


# **California,Texas and New York** are the top 3 states by number of schools. It was kind of expected as the number of projects and number of donations were highest from these states.

# ## Schools with highest  number of Project Postings

# In[43]:


proj_schools=projects[['School ID','Project Title','Project Current Status']].merge(schools,left_on='School ID',right_on='School ID',how='left')
print(proj_schools['School Name'].value_counts().index[0],'has the highest number of proposed projects with count =',proj_schools['School Name'].value_counts()[0])

# Okay so **Lincoln Elementary school has 2429 projects in the past 5 years**. This is surprising as it comes to around **1 project per day**. But I dont think it is possible. The reason we get the above value is because **many schools have the same name**. Here in India also, we have many Kendriya Vidyalayas. So lets see which school actually has the highest number of projects.
# 

# In[44]:


school_max=school_pro.groupby(['School Name','School ID','School County'],as_index=False)['School Metro Type'].count().sort_values(ascending=False,by='School Metro Type')
school_max.set_index(['School Name','School County'],inplace=True)
ax=school_max['School Metro Type'][:10].plot.barh(width=0.9,color=sns.color_palette('viridis_r',20))
plt.gca().invert_yaxis()
for i, v in enumerate(school_max['School Metro Type'][:10]): 
    ax.text(.5, i, v,fontsize=10,color='black',weight='bold')
plt.gcf().set_size_inches(8,8)
plt.title('Schools with highest Project Postings')

# So **Dawes Elementary School at Cook** has the highest project postings till date.
# 

# ## Funded Projects By States

# In[45]:


schools_sta=projects[['School ID','Project Current Status']].merge(schools[['School ID','School State']],left_on='School ID',right_on='School ID',how='left')
sta_total=schools_sta.groupby('School State')['Project Current Status'].count().reset_index()
sta_fund=schools_sta[schools_sta['Project Current Status']=='Fully Funded'].groupby('School State')['Project Current Status'].count().reset_index()
sta_total=sta_total.merge(sta_fund,left_on='School State',right_on='School State',how='left')
sta_total.rename({'Project Current Status_x':'Total','Project Current Status_y':'Funded'},axis=1,inplace=True)
sta_total['%Funded']=(sta_total['Funded'])/sta_total['Total']
plt.figure(figsize=(20,20))
plt.scatter('Funded','%Funded',data=sta_total,s=sta_total['%Funded']*750)
for i in range(sta_total.shape[0]):
    plt.text(sta_total['Funded'].values[i],sta_total['%Funded'].values[i],s=sta_total['School State'].values[i],color='r',size=15)
plt.xlabel('No of Funded Projects')
plt.ylabel('% Funded Projects')
plt.title('Funded Projects By States')

# **District of Columbia** has the higest percentage of Funded Projects, but it is due to the fact that it has lesser projects are compared to some other states. We can see that as the number of projects for a state increases, the % Funded Projects decreases.
# 

# ## School Metro Type

# In[46]:


plt.figure(figsize=(12,4))
sns.countplot(schools['School Metro Type'])
plt.title('Count of School Metro Type')

# **Urban and Suburban** schools make a majority from the total schools. Lets locate all these schools.
# 

# ## Locating Schools By Metro Type

# In[47]:


zipcodes=pd.read_csv('../input/usa-zip-codes-to-locations/US Zip Codes from 2013 Government Data.csv')
schools_zip=schools[['School Metro Type','School Zip']].merge(zipcodes,left_on='School Zip',right_on='ZIP',how='left')
fig, axes = plt.subplots(1, 2,figsize=(25,15))

axes[0].set_title("Rural Schools")
map = Basemap(llcrnrlon=-119,llcrnrlat=22,urcrnrlon=-64,urcrnrlat=49,projection='lcc',lat_1=33,lat_2=45,lon_0=-95, ax=axes[0])
rural=schools_zip[schools_zip['School Metro Type']=='rural']
lat_rural=list(rural['LAT'])
lon_rural=list(rural['LNG'])
x_1,y_1=map(lon_rural,lat_rural)
map.plot(x_1, y_1,'go',markersize=2,color = 'b')
map.drawmapboundary(fill_color='aqua')
map.fillcontinents(color='coral',lake_color='aqua')
map.drawcoastlines()

axes[1].set_title("Urban Schools")
map = Basemap(llcrnrlon=-119,llcrnrlat=22,urcrnrlon=-64,urcrnrlat=49,projection='lcc',lat_1=33,lat_2=45,lon_0=-95, ax=axes[1])
urban=schools_zip[schools_zip['School Metro Type']=='urban']
lat_urban=list(urban['LAT'])
lon_urban=list(urban['LNG'])
x_1,y_1=map(lon_urban,lat_urban)
map.plot(x_1, y_1,'go',markersize=2,color = 'r')
map.drawmapboundary(fill_color='aqua')
map.fillcontinents(color='coral',lake_color='aqua')
map.drawcoastlines()

fig1, axes1 = plt.subplots(1, 2,figsize=(25,15))

axes1[0].set_title("Suburban Schools")
map = Basemap(llcrnrlon=-119,llcrnrlat=22,urcrnrlon=-64,urcrnrlat=49,projection='lcc',lat_1=33,lat_2=45,lon_0=-95, ax=axes1[0])
suburb=schools_zip[schools_zip['School Metro Type']=='suburban']
lat_suburb=list(suburb['LAT'])
lon_suburb=list(suburb['LNG'])
x_1,y_1=map(lon_suburb,lat_suburb)
map.plot(x_1, y_1,'go',markersize=2,color = 'g')
map.drawmapboundary(fill_color='aqua')
map.fillcontinents(color='coral',lake_color='aqua')
map.drawcoastlines()

axes1[1].set_title("Town Schools")
map = Basemap(llcrnrlon=-119,llcrnrlat=22,urcrnrlon=-64,urcrnrlat=49,projection='lcc',lat_1=33,lat_2=45,lon_0=-95, ax=axes1[1])
town=schools_zip[schools_zip['School Metro Type']=='town']
lat_town=list(town['LAT'])
lon_town=list(town['LNG'])
x_1,y_1=map(lon_town,lat_town)
map.plot(x_1, y_1,'go',markersize=2,color = 'black')
map.drawmapboundary(fill_color='aqua')
map.fillcontinents(color='coral',lake_color='aqua')
map.drawcoastlines()
plt.show()

# **Observations:**
# 
# 1) Even though the number of urban and suburban schools are high, they are largely concentrated at particular areas, like California, New York, New Jersey,etc.
# 
# 2) Rural Schools are very well scattered,but the prominance is in the right half of the country.
# 
# ## Project Costs By School Type

# In[48]:


cost=schools[['School ID','School Metro Type']].merge(projects[['School ID','Project Cost']],left_on='School ID',right_on='School ID',how='left')
cost.groupby('School Metro Type',as_index=False).agg(['median','mean','max','min'])

# ## Project Cost By Metro Type

# In[49]:


metro=projects[['School ID','Project Cost']].merge(schools[['School ID','School Metro Type']],left_on='School ID',right_on='School ID',how='left')
rural=metro[metro['School Metro Type']=='rural']
urban=metro[metro['School Metro Type']=='urban']
suburb=metro[metro['School Metro Type']=='suburban']
town=metro[metro['School Metro Type']=='town']
f,ax=plt.subplots(2,2,figsize=(25,12))
np.log(rural['Project Cost'].dropna()).hist(ax=ax[0,0],bins=30,edgecolor='black')
ax[0,0].set_title('Project Cost of Rural Schools')
np.log(urban['Project Cost'].dropna()).hist(ax=ax[0,1],bins=30,edgecolor='black')
ax[0,1].set_title('Project Cost of Urban Schools')
np.log(suburb['Project Cost'].dropna()).hist(ax=ax[1,0],bins=30,edgecolor='black')
ax[1,0].set_title('Project Cost of Suburban Schools')
np.log(town['Project Cost'].dropna()).hist(ax=ax[1,1],bins=30,edgecolor='black')
ax[1,1].set_title('Project Cost of Town Schools')


# ## Do Schools get donations from Home State??

# In[50]:


home_don=donors[['Donor ID','Donor State']].merge(donations[['Donor ID','Project ID']],left_on='Donor ID',right_on='Donor ID',how='right')
home_don=home_don.merge(projects[['Project ID','School ID']],left_on='Project ID',right_on='Project ID',how='right')
home_don=home_don.merge(schools[['School ID','School State','School Zip']],left_on='School ID',right_on='School ID',how='left')
home_don['Home State Donation']=np.where(home_don['Donor State']==home_don['School State'],'YES','NO')
home_don['Home State Donation'].value_counts().plot.pie(autopct='%1.1f%%',startangle=90,shadow=True,explode=[0,0.1])
plt.gcf().set_size_inches(5,5)

# 61% of projects get funding from their home state. Lets better visualize the donations from the top 3 states, i.e California, Texas and New York. We will see in-state and out of state donations for the Top 3 states.

# ## Donations From California

# In[51]:


home_don1=home_don[['Donor State','School State','School Zip']]
home_don1=home_don1.merge(zipcodes,left_on='School Zip',right_on='ZIP',how='left')
import branca.colormap as cm
cali=home_don1[home_don1['Donor State']=='California']
cali.dropna(inplace=True)
cali=cali.groupby(['LAT','LNG'])['School Zip'].count().sort_values(ascending=False).reset_index()
data=[]
for i,j,k in zip(cali['LAT'],cali['LNG'],cali['School Zip']):
    data.append([i,j,k])
map1=folium.Map([39.3714557,-94.3541242], zoom_start=4,tiles='OpenStreetMap')
map1.add_child(folium.plugins.HeatMap(data, radius=12))
cp = cm.LinearColormap(['green', 'yellow', 'red'],vmin=cali['School Zip'].min(), vmax=cali['School Zip'].max())
map1.add_child(cp)
map1

# ## Donations from New-York

# In[52]:


york=home_don1[home_don1['Donor State']=='New York']
york.dropna(inplace=True)
york=york.groupby(['LAT','LNG'])['School Zip'].count().sort_values(ascending=False).reset_index()
data=[]
for i,j,k in zip(york['LAT'],york['LNG'],york['School Zip']):
    data.append([i,j,k])
map1=folium.Map([39.3714557,-94.3541242], zoom_start=4,tiles='OpenStreetMap')
map1.add_child(folium.plugins.HeatMap(data, radius=12))
cp = cm.LinearColormap(['green', 'yellow', 'red'],vmin=york['School Zip'].min(), vmax=york['School Zip'].max())
map1.add_child(cp)
map1

# ## Donations from Texas

# In[53]:


texas=home_don1[home_don1['Donor State']=='Texas']
texas.dropna(inplace=True)
texas=texas.groupby(['LAT','LNG'])['School Zip'].count().sort_values(ascending=False).reset_index()
data=[]
for i,j,k in zip(texas['LAT'],texas['LNG'],texas['School Zip']):
    data.append([i,j,k])
map1=folium.Map([39.3714557,-94.3541242], zoom_start=4,tiles='OpenStreetMap')
map1.add_child(folium.plugins.HeatMap(data, radius=12))
cp = cm.LinearColormap(['green', 'yellow', 'red'],vmin=texas['School Zip'].min(), vmax=texas['School Zip'].max())
map1.add_child(cp)
map1

# We can see that majority of donations from California is for projects from California itself. Same is the case for New-York and Texas.So we can see that majority of the donations for the projects come from their home state itself. 

# ## Number Of Schools and Projects By Metro Type

# In[54]:


trying1=schools.groupby(['School State','School Metro Type'])['School Zip'].count().to_frame()
trying1=schools.groupby(['School State','School Metro Type'])['School Zip'].count().to_frame()
trying1.rename({'School Zip':'School Count'},axis=1,inplace=True)
trying2=projects[['Project ID','School ID']].merge(schools[['School ID','School Metro Type','School State']],left_on='School ID',right_on='School ID',how='left')
trying2=trying2.groupby(['School State','School Metro Type'])['School ID'].count().to_frame()
trying2.rename({'School ID':'Total Projects'},axis=1,inplace=True)
trying2=trying2.merge(trying1,left_index=True,right_index=True,how='left').reset_index()
trying2=trying2.merge(locations,left_on='School State',right_on='City',how='left')
trying2=trying2[['School State','School Metro Type','Total Projects','School Count','Latitude','Longitude']]
map2 = folium.Map(location=[39.50, -98.35],tiles='Mapbox Control Room',zoom_start=3.5)
for i in trying2['School State'].unique():
    df=trying2[trying2['School State']==i]
    df=df[df.columns.difference(['Latitude','Longitude'])]
    df.set_index('School State',inplace=True)
    df=df[['School Metro Type','School Count','Total Projects']]
    html=df.to_html()
    popup = folium.Popup(html)
    location=trying2[trying2['School State']==i][['Latitude','Longitude']].values[0]
    size=trying2[trying2['School State']==i].groupby('School State')['School Count'].sum().values[0]
    def color_producer(schools):
        if schools < 1000:
            return 'red'
        else:
            return 'yellow'
    folium.CircleMarker(location, popup=popup,radius=size*0.005,color=color_producer(size),fill=True).add_to(map2)
map2

# Click on the marker to get details. Each marker on the map segregates the School Metro Type and the number of schools for each type by State.

# # Resource Vendors

# In[55]:


resources['Total Cost']=resources['Resource Quantity']*resources['Resource Unit Price']
vendor1=resources.groupby('Resource Vendor Name')['Total Cost'].sum().to_frame()
vendor2=resources.drop_duplicates(subset=['Resource Vendor Name','Project ID'])['Resource Vendor Name'].value_counts().to_frame()
final=vendor1.merge(vendor2,left_index=True,right_index=True,how='left')
f,ax=plt.subplots(ncols=2,figsize=(20,12),sharey=True)
final=final.sort_values(by='Total Cost',ascending=False)[:15]
final['Total Cost'].plot.barh(ax=ax[0],width=0.9,color=sns.color_palette('viridis'))
final['Resource Vendor Name'].plot.barh(ax=ax[1],width=0.9,color=sns.color_palette('viridis'))
ax[0].invert_xaxis()
ax[0].yaxis.tick_right()
plt.subplots_adjust(wspace=0.6)
ax[0].set_title('Total Cost')
ax[1].set_title('Total Projects')
ax[0].invert_yaxis()

# The right side graph shows the number of **unique projects** the Vendor has supplied resurces to, while the left graph shows the **total cost** of supplies bought from the vendor till date. Many projects have bought supplies from the same vendor multiple times, thus it is necessary to count the number of unique projects.
# 
# 
# **Amazon Business** has been the Vendor for most of the projects, followed by **Best Buy Education and Lakeshore Learning Materials.** The total cost is however not completely the same as the number of projects supplied to, i.e we can see some Vendors with a higher total cost of purchases even if they have a relatively smaller number of purchases. This is due to the varying cost of products, and many vendors specialize in a certain category, thereby increasing the cost of purchase.

# ## Major Vendors With Their Supply Resources
# 

# In[56]:


major=resources[resources['Resource Vendor Name'].isin(final.sort_values('Resource Vendor Name',ascending=False).index[:10])][['Project ID','Resource Vendor Name']]
major=projects[['Project ID','Project Resource Category']].merge(major,left_on='Project ID',right_on='Project ID',how='left')
major=major[major['Project Resource Category'].isin(projects['Project Resource Category'].value_counts()[:10].index)]
major=major.drop_duplicates(subset=['Resource Vendor Name','Project ID','Project Resource Category'])
major=major.groupby(['Resource Vendor Name','Project Resource Category'],as_index=False)['Project ID'].count()
sns.heatmap(major.pivot('Resource Vendor Name','Project Resource Category','Project ID'),cmap='viridis',annot=True,fmt='2.0f',linewidths=.5)
plt.gcf().set_size_inches(14,11)

# We are trying to see the major resources that the Top Vendors Supply to the Projects. In resources regarding to **Technology**, we see that **Best Buy Education** has the highest supplies followed by **Amazon Business**. This may be the factor I had mentioned. Since **Best Buy Education** has the highest Technological Supplies, it has a comparatively higher total cost of purchases.
# 
# Similarly in books department, **Amazon and AKJ Education** have the highest supplies, while in supplies **Amazon and Lakeshore Learning Materials** are the top sellers. So **Amazon** has supplies for every need. **#ApniDukaan**...:p

# # Text Analysis
# 
# ## Topic Modeling
# 
# Topics can be defined as “a repeating pattern in a cluster of documents or a corpus”. Topic Models are very useful for the purpose for document clustering, organizing large blocks of textual data, information retrieval from unstructured text and feature selection. Lets use topic modeling on the Project Need Statement for getting the top needs by the schools. The Project  Essay would have provided even better topics, but it takes a lot of time to process even 1000 project essay. Even for the Project Need, I have considered the first 100000 rows for avoiding the infinite loop...:p. There are many approaches for obtaining topics from a text such as – Term Frequency and Inverse Document Frequency (TF-IDF), NonNegative Matrix Factorization techniques(NMF), Latent Dirichlet Allocation(LDA),etc. I will use LDA and NMF in this notebook. 
# 
# 
# ### Latent Dirichlet Allocation(LDA)
# 
# LDA assumes documents are produced from a mixture of topics. Those topics then generate words based on their probability distribution. Given a dataset of documents, LDA backtracks and tries to figure out what topics would create those documents in the first place. Since LDA is a probablistic model, we will use CountVectorizer which will give a raw count of the words.

# In[57]:


from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
need=projects['Project Need Statement'].dropna().values[:100000]
tf_vectorizer = CountVectorizer(max_df = 0.95, min_df = 2, max_features = 1000, stop_words='english')
tf = tf_vectorizer.fit_transform(need)
tf_feature_names = tf_vectorizer.get_feature_names()

# In[58]:


from sklearn.decomposition import NMF, LatentDirichletAllocation
lda = LatentDirichletAllocation(n_components = 10, max_iter = 5, learning_method = 'online', learning_offset = 50.,random_state = 123).fit(tf)

# In[59]:


for topic_idx, topic in enumerate(lda.components_):
    print("Topic %d:" % (topic_idx))
    print(" ".join([tf_feature_names[i] for i in topic.argsort()[:-10 - 1:-1]]))

# 
# ### NonNegative Matrix Factorization techniques(NMF)
# 
# NMF is applied to a matrix of TF-IDF that can be used to extract topics from large collections of text. TF-IDF gives us a measure of how important a word is to a document in the corpus.
# 

# In[60]:


tfidf_vectorizer = TfidfVectorizer(max_df = 0.95, min_df = 2, max_features = 1000, stop_words='english')
tfidf = tfidf_vectorizer.fit_transform(need)
tfidf_feature_names = tfidf_vectorizer.get_feature_names()

# In[61]:


nmf = NMF(n_components = 10, random_state = 123, alpha = .1, l1_ratio = .5, init = 'nndsvd').fit(tfidf)
for topic_idx, topic in enumerate(nmf.components_):
    print("Topic %d:" % (topic_idx))
    print(" ".join([tfidf_feature_names[i] for i in topic.argsort()[:-10 - 1:-1]]))

# ## WordCloud for Project Titles
# 
# Lets see some common words/phrases in the Project Titles. We will first remove the stopwords and some redundant words like **student,us,etc.**

# In[62]:


img=b'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'

# In[63]:


import codecs
from scipy.misc import imread
from wordcloud import WordCloud, STOPWORDS
import nltk
from nltk.corpus import stopwords
stop_words=stopwords.words('english')
stop_words.extend([',',';','!','?','.','(',')','$','#','+',':','...','``',"''",'student','students','us'])

helping=projects['Project Title'][:100000].dropna().apply(nltk.word_tokenize)
helpers=[]
for i in helping:
    helpers.extend(i)
helpers=pd.Series(helpers)
helpers=([i for i in helpers.str.lower() if i not in stop_words])
f1=open("img.png", "wb")
f1.write(codecs.decode(img,'base64'))
f1.close()
img1 = imread("img.png")
hcmask1 = img1
wc = WordCloud(background_color="black", max_words=10000, mask=hcmask1, 
               stopwords=STOPWORDS, max_font_size= 60,colormap='RdYlGn')
wc.generate(" ".join(helpers))
plt.imshow(wc)
plt.axis('off')
plt.gcf().set_size_inches(10,10)


# ## Projects for Hurricane Disaster Recovery
# 
# So I was randomly reading about natural disasters yesterday and I read an article about how ferquently hurricanes hit the United States. On reading further, I found that hurricanes are more frequent in the months of **September and October**. As we had seen earlier that projects posting are at the peak during the months of **August and September**.So I decided to check whether this peak is due to the hurricanes or something else.Lets check it..
# 

# In[64]:


hurri=projects.dropna(subset=['Project Title'])
sns.countplot(hurri[hurri['Project Title'].str.contains('hurricanes|hurricane|Hurricane|Hurricanes')]['Project Posted Date'].dt.month)
plt.title('Projects For Hurricane DisRecovery')

# We do see a peak in the month of September, but the hight peak is during January. The number of projects are high after the months of September till January. So hurricanes are not the reason for high projects during August and September.

# # Conclusions
# 
# ## Donations
# 
# 
#  - The minimum donation is **0.01$ **', while the maximum donation amount is **60000**. The median amount is **25**. Most common donation amount are **25,50 and 10** dollars.
# 
#  - **75%** of the donations include the **Optional Amount**, which helps Donors Choose cover their operational costs.
# 
#  - Donations have been increasing substantially over the years, with the maximum donations being in the months of **August and September.**
# 
#  - **61.8%** of the projects get the donations from their home state itself.
# 
# ## Donors
# 
#  - Statewise **California** has the highest number of donors, while Citywise, **Chicago** leads. Also California has the highest total amount donated, being almost **double** the amount by New-York at the 2nd position.
# 
#  - Out of the total donors, only **12%** of them are teachers. However, these 12% donor teachers account to about **39%** of the total donations. 
# 
#  - Median donations from Teachers and Others are **20 and 30**$'s respectively.
# 
# 
# ## Projects
# 
#  - Number of projects have also increased substantially over the years, with the maximum projects being posted in the months of **August and September**.
# 
#  - Majority of the projects get fully funded in **20-30** days period.
# 
#  - The percentage of Fully Funded Projects have increased through the years, but it dips in the year 2016.
# 
#  - Fully Funded Projects have a lower project costs as compared to Non Funded Projects.
# 
#  - Majority of the projects are for Educational purpose like Literacy, Language and other subjects.
# 
#  - **District of Columbia** has the highest percentage of Fully Funded Projects, but the number of projects are also on the lower side. **Nevada** has the lowest percentage of Fully Funded Projects.
# 
# ## Schools
# 
#  - **California** has the highest number of schools followed by **Texas**.
# 
#  - **Urban and Suburban** are the 2 major School Metro Types.
# 
#  - **Wyoming** has the highest number of rural schools, while **California** has the majority in the other 3 types.
# 

# 

# In[ ]:



