#!/usr/bin/env python
# coding: utf-8

# <h1>General Analysis of Suicide Rates</h1>
# 
# <p>Our aim is to make general analysis of suicide rates and we will explain these rates in detail.</p>
# 
# <p>
# 
# Dataset Feature List
# 
# <ul>
# <li>country</li>
# <li>year</li>
# <li>sex</li>
# <li>age</li>
# <li>suicides_no</li>
# <li>population</li>
# <li>suicides/100k pop</li>
# <li>country-year</li>
# <li>HDI for year</li>
# <li>gdp_for_year</li>
# <li>gdp_per_capita</li>
# <li>generation</li>
# 
# </ul>
# 
# </p>
# 
# <p><a href='#20'>References</a></p>
# <p><a href='#21'>Conclusion</a></p>
# 
# <p>Last Updated : <b>12.05.2019</b></p>
# <p><h2>if you like it, please upvoted</h2></p>

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

# In[ ]:


#dataset read operation
#read_csv function is required to read the data.
data=pd.read_csv('../input/master.csv')

# In[ ]:


#show data first 5 rows
data.head()

# In[ ]:


#show data last 5 rows
data.tail()

# In[ ]:


#random rows in dataset
data.sample(5)

# In[ ]:


data.sample(frac=0.1)

# In[ ]:


#Describe function includes analysis of all our numerical data. For this, count, mean, std, min,% 25,% 50,% 75, max values are given.
data.describe()

# In[ ]:


data.iloc[:,1:5].describe()

# In[ ]:


#The info function shows the data types and numerical values of the features in our data set.
data.info()

# In[ ]:


#We will now set the headings of the feature values in the data set.
data.columns

# In[ ]:


#so,change the names of the column. Because there may be problems for future analysis.
data=data.rename(columns={'country':'Country','year':'Year','sex':'Gender','age':'Age','suicides_no':'SuicidesNo','population':'Population','suicides/100k pop':'Suicides100kPop','country-year':'CountryYear','HDI for year':'HDIForYear',' gdp_for_year ($) ':'GdpForYearMoney','gdp_per_capita ($)':'GdpPerCapitalMoney','generation':'Generation'})

# In[ ]:


data.columns

# In[ ]:


#And, how many rows and columns are there for all data?
print('Data Shape :')
data.shape

# In[ ]:


data.isnull().any()

# In[ ]:


data.isnull().values.any()

# In[ ]:


#Now,I will check null on all data and If data has null, I will sum of null data's. In this way, how many missing data is in the data.
data.isnull().sum()

# In[ ]:


#As you can see, most of the HDIForYear value is empty. That's why I want this value deleted.
data=data.drop(['HDIForYear','CountryYear'],axis=1)

# In[ ]:


#Now start analysis, min year and max year will find them
min_year=min(data.Year)
max_year=max(data.Year)
print('Min Year :',min_year)
print('Max Year :',max_year)

#1985 min year,2016 max year.

data_country=data[(data['Year']==min_year)]

country_1985=data[(data['Year']==min_year)].Country.unique()
country_1985_male=[]
country_1985_female=[]

for country in country_1985:
    country_1985_male.append(len(data_country[(data_country['Country']==country)&(data_country['Gender']=='male')]))
    country_1985_female.append(len(data_country[(data_country['Country']==country)&(data_country['Gender']=='female')])) 
    
#We found the ratio of men and women who committed suicide in some countries in 1985 and we are now charting.

plt.figure(figsize=(10,10))
sns.barplot(y=country_1985,x=country_1985_male,color='red')
sns.barplot(y=country_1985,x=country_1985_female,color='yellow')
plt.ylabel('Countries')
plt.xlabel('Count Male vs Female')
plt.title('1985 Year Suicide Rate Gender')
plt.show()

#Very odd all the rates came on an equal level. So let's do max year.

data_country=data[(data['Year']==max_year)]

country_2016=data[(data['Year']==max_year)].Country.unique()
country_2016_male=[]
country_2016_female=[]

for country in country_2016:
    country_2016_male.append(len(data_country[(data_country['Country']==country)&(data_country['Gender']=='male')]))
    country_2016_female.append(len(data_country[(data_country['Country']==country)&(data_country['Gender']=='female')])) 
    
#We found the ratio of men and women who committed suicide in some countries in 1985 and we are now charting.

plt.figure(figsize=(10,10))
sns.barplot(y=country_2016,x=country_2016_male,color='red')
sns.barplot(y=country_2016,x=country_2016_female,color='yellow')
plt.ylabel('Countries')
plt.xlabel('Count Male vs Female')
plt.title('2016 Year Suicide Rate Gender')
plt.show()

# <p>While the suicide rate was widespread in more countries in 1985, this ratio has fallen considerably in 2016. Now let us examine the other features of these countries.</p>

# In[ ]:


data_country=data[(data['Year']==min_year)]

country_1985_population=[]

for country in country_1985:
    country_1985_population.append(sum(data_country[(data_country['Country']==country)].Population))    

#Now year 1985 find sum population every country

plt.figure(figsize=(10,10))
sns.barplot(y=country_1985,x=country_1985_population)
plt.xlabel('Population Count')
plt.ylabel('Countries')
plt.title('1985 Year Sum Population for Suicide Rate')
plt.show()

#######################################################

data_country=data[(data['Year']==max_year)]

country_2016_population=[]

for country in country_2016:
    country_2016_population.append(sum(data_country[(data_country['Country']==country)].Population))    

#Now year 1985 find sum population every country

plt.figure(figsize=(10,10))
sns.barplot(y=country_2016,x=country_2016_population)
plt.xlabel('Population Count')
plt.ylabel('Countries')
plt.title('2016 Year Sum Population for Suicide Rate')
plt.show()


# <p>The values we found on the upper side give the total population we found in 1985 in 2016.</p>

# In[ ]:


suicideGender1985=data_country.groupby(['Country','Gender']).SuicidesNo.sum()

# In[ ]:


suicideGender1985

# In[ ]:


data_country=data[(data['Year']==min_year)]

data_age_5_14=[]
data_age_15_24=[]
data_age_25_34=[]
data_age_35_54=[]
data_age_55_74=[]
data_age_75=[]

for country in country_1985:
        data_age_5_14.append(len(data_country[(data_country['Country']==country)&(data_country['Age']=='5-14 years')]))
        data_age_15_24.append(len(data_country[(data_country['Country']==country)&(data_country['Age']=='15-24 years')]))
        data_age_25_34.append(len(data_country[(data_country['Country']==country)&(data_country['Age']=='25-34 years')]))
        data_age_35_54.append(len(data_country[(data_country['Country']==country)&(data_country['Age']=='35-54 years')]))
        data_age_55_74.append(len(data_country[(data_country['Country']==country)&(data_country['Age']=='55-74 years')]))
        data_age_75.append(len(data_country[(data_country['Country']==country)&(data_country['Age']=='75+ years')]))
        

#######################################################

data_country=data[(data['Year']==max_year)]

data_age_5_14=[]
data_age_15_24=[]
data_age_25_34=[]
data_age_35_54=[]
data_age_55_74=[]
data_age_75=[]

for country in country_2016:
        data_age_5_14.append(len(data_country[(data_country['Country']==country)&(data_country['Age']=='5-14 years')]))
        data_age_15_24.append(len(data_country[(data_country['Country']==country)&(data_country['Age']=='15-24 years')]))
        data_age_25_34.append(len(data_country[(data_country['Country']==country)&(data_country['Age']=='25-34 years')]))
        data_age_35_54.append(len(data_country[(data_country['Country']==country)&(data_country['Age']=='35-54 years')]))
        data_age_55_74.append(len(data_country[(data_country['Country']==country)&(data_country['Age']=='55-74 years')]))
        data_age_75.append(len(data_country[(data_country['Country']==country)&(data_country['Age']=='75+ years')]))
        
#there is an equal rate. We need to make the query process a little more complicated.

# In[ ]:


sns.countplot(data.Gender)
plt.show()
#there has been an even gender distribution.

# In[ ]:


plt.figure(figsize=(10,5))
sns.countplot(data.Gender,hue=data.Age)
plt.title('Gender & Age')
plt.show()
#there has been an even gender & hue age distribution.

# In[ ]:


data.groupby('Age')['Gender'].count()

# In[ ]:


sns.barplot(x=data.groupby('Age')['Gender'].count().index,y=data.groupby('Age')['Gender'].count().values)
plt.xticks(rotation=90)
plt.show()

# In[ ]:


suicidesNo=[]
for country in data.Country.unique():
    suicidesNo.append(sum(data[data['Country']==country].SuicidesNo))   

# In[ ]:


suicidesNo=pd.DataFrame(suicidesNo,columns=['suicidesNo'])
country=pd.DataFrame(data.Country.unique(),columns=['country'])
data_suicide_countr=pd.concat([suicidesNo,country],axis=1)
#sns.barplot(x=data.Country.unique(),y=suicidesNo) 
#plt.show()

# In[ ]:


data_suicide_countr=data_suicide_countr.sort_values(by='suicidesNo',ascending=False)

# In[ ]:


sns.barplot(y=data_suicide_countr.country[:15],x=data_suicide_countr.suicidesNo[:15])
plt.show()

# <p>It appears that after examining our data, we examined the total suicide rates and prepared a chart that handled the most from the least.</p>

# In[ ]:


grouop_data=data.groupby(['Age','Gender'])['SuicidesNo'].sum().unstack()
grouop_data=grouop_data.reset_index().melt(id_vars='Age')

# In[ ]:


grouop_data_female=grouop_data.iloc[:6,:]
grouop_data_male=grouop_data.iloc[6:,:]

# In[ ]:


grouop_data_female

# In[ ]:


grouop_data_male

# In[ ]:


female_=[175437,208823,506233,16997,430036,221984]
male_=[633105,915089,1945908,35267,1228407,431134]
plot_id = 0
for i,age in enumerate(['15-24 years','25-34 years','35-54 years','5-14 years','55-74 years','75+ years']):
    plot_id += 1
    plt.subplot(3,2,plot_id)
    plt.title(age)
    fig, ax = plt.gcf(), plt.gca()
    sns.barplot(x=['female','male'],y=[female_[i],male_[i]],color='blue')
    plt.tight_layout()
    fig.set_size_inches(10, 15)
plt.show()  

# <p>All data were analyzed. Graphical analysis was performed for all age rates for suicide rates.</p>

# In[ ]:


sns.countplot(data.Generation)
plt.title('Generation Counter')
plt.xticks(rotation=45)
plt.show()

# In[ ]:


sns.countplot(data.Generation,hue=data.Gender)
plt.title('Generation hue Gender Counter')
plt.show()

# In[ ]:


data['Age'].unique()

# In[ ]:


index_population=[]
for age in data['Age'].unique():
    index_population.append(sum(data[data['Age']==age].Population)/len(data[data['Age']==age].Population))
    
plt.bar(['15-24 years','35-54 years','75+ years','25-34 years','55-74 years','5-14 years'],index_population,align='center',alpha=0.5)
plt.xticks(rotation=90)
plt.show()

# In[ ]:


index_population

# In[ ]:


plt.figure(figsize=(10,5))
sns.set(style='whitegrid')
sns.boxplot(data['Population'])
plt.show()

# In[ ]:


data.head()

# In[ ]:


sns.set(style='whitegrid')
sns.boxplot(data['GdpPerCapitalMoney'])
plt.show()

# In[ ]:


sns.set(style='whitegrid')
sns.boxplot(data.Year)
plt.show()

# In[ ]:


# Plot the crashes where alcohol was involved
sns.set_color_codes("muted")
sns.barplot(x="Year", y="SuicidesNo", data=data,
            label="Year Suicides", color="b")
plt.xticks(rotation=90)
plt.show()

# In[ ]:



plt.figure(figsize=(7,7))
g = sns.FacetGrid(data, col="Generation", hue="Year",
                  subplot_kws=dict(projection='polar'), height=5.5,
                  sharex=False, sharey=False, despine=False)

plt.show()

# In[ ]:


# Plot sepal with as a function of sepal_length across days
g = sns.lmplot(x="Year", y="SuicidesNo", hue="Generation",
               truncate=True, height=5, data=data)

# Use more informative axis labels than are provided by default
g.set_axis_labels("Year", "Suicides No")
plt.show()

# In[ ]:


data.head()

# In[ ]:


data.describe().plot(kind = "Area",fontsize=15, figsize = (20,10), table = True,colormap="rainbow")
plt.xlabel('Statistics',)
plt.ylabel('Value')
plt.title("General Statistics")
plt.show()

# In[ ]:


f,ax=plt.subplots(1,2,figsize=(18,8))
data['Generation'].value_counts().plot.pie(explode=[0.1,0.1,0.1,0.1,0.1,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True)
ax[0].set_title('Generations Count')
ax[0].set_ylabel('Count')
sns.countplot('Generation',data=data,ax=ax[1])
ax[1].set_title('Generations Count')
plt.show()

# In[ ]:


fig=sns.jointplot(y='Suicides100kPop',x='Year',data=data)
plt.show()

# In[ ]:


fig=sns.jointplot(y='SuicidesNo',x='Population',data=data)
plt.show()

# In[ ]:


sns.jointplot("Year", "GdpPerCapitalMoney", data=data, kind="reg")
plt.show()

# In[ ]:


# Show the joint distribution using kernel density estimation
g = sns.jointplot(data.Year,data.GdpPerCapitalMoney, kind="kde", height=7, space=0)
plt.show()

# In[ ]:


fig=sns.jointplot(y='GdpPerCapitalMoney',x='SuicidesNo',kind='hex',data=data[data['Country']=='United States'])
plt.show()

# In[ ]:


sns.jointplot("Suicides100kPop", "SuicidesNo", data=data[data['Country']=='Russian Federation'], kind="kde",space=0,color='g')
plt.show()

# In[ ]:


import matplotlib.pyplot as plt
sns.FacetGrid(data,hue='Year',size=5).map(plt.scatter,'GdpPerCapitalMoney','Population').add_legend()
plt.show()

# In[ ]:


sns.FacetGrid(data, hue="Generation", size=6).map(sns.kdeplot, "Population").add_legend()
plt.ioff() 
plt.show()

# In[ ]:


sns.countplot(x="Generation", hue="Gender",
                 data=data)
plt.xticks(rotation=45)
plt.show()

# In[ ]:


sns.stripplot(x=data['SuicidesNo'])
plt.show()

# In[ ]:


sns.set(style="white")
# Plot miles per gallon against horsepower with other semantics
sns.relplot(x="Generation",y="SuicidesNo",hue="Gender",
            sizes=(40, 400), alpha=.5, palette="muted",
            height=6, data=data)
plt.show()

# In[ ]:


plt.figure(figsize=(10,7))
sns.stripplot(x="Year",y='Suicides100kPop',data=data)
plt.xticks(rotation=45)
plt.show()

# In[ ]:


plt.figure(figsize=(10,10))
sns.factorplot(x="SuicidesNo", y="Gender", kind='violin',data=data)
plt.show()

# In[ ]:


plt.scatter(x=np.arange(1,27821),y=data['Suicides100kPop'].values.tolist())
plt.show()

# In[ ]:


plt.scatter(x=np.arange(1,27821),y=data['SuicidesNo'].values.tolist())
plt.show()

# In[ ]:


plt.scatter(x=np.arange(1,27821),y=data['Population'].values.tolist())
plt.show()

# In[ ]:


plt.scatter(x=np.arange(1,27821),y=data['GdpPerCapitalMoney'].values.tolist())
plt.show()

# In[ ]:


sns.distplot(data['Suicides100kPop'])
plt.show()

# In[ ]:


sns.set_color_codes()
sns.distplot(data['Country'].value_counts().values,color='r')
plt.show()

# In[ ]:


data[(data['Gender']=='male')].Age.value_counts().values

# In[ ]:


# Plot the responses for different events and regions
sns.lineplot(x="SuicidesNo", y="Suicides100kPop",
             hue="Gender",data=data.sort_values(by='Suicides100kPop',ascending=False))
plt.show()

# In[ ]:


sns.pairplot(data,hue='Generation')
plt.show()

# In[ ]:


sns.pairplot(data, hue="Gender")
plt.show()

# In[ ]:


# Plot the residuals after fitting a linear model
sns.residplot('SuicidesNo','Suicides100kPop', data=data,lowess=True, color="g")
plt.show()

# In[ ]:


sns.distplot(data[(data['Gender']=='female')].Age.value_counts().values)
plt.show()

# In[ ]:


sns.violinplot(x=data['Generation'],y=data['Population'])
plt.show()

# In[ ]:


sns.heatmap(data.corr(),cmap='YlGnBu',annot=True)
plt.show()

# In[ ]:


# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(data.corr(), vmax=.3, center=1,
            square=True, linewidths=.5,annot=True)
plt.show()

# In[ ]:


cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(data.corr(), cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.show()

# In[ ]:


sns.boxplot(x=data['Generation'],y=data['Population'])
plt.xticks(rotation=90)

# In[ ]:


sns.boxenplot(x="Generation", y="SuicidesNo",
              color="b",
              scale="linear", data=data)
plt.tight_layout()
plt.show()

# In[ ]:


data.head()

# In[ ]:


max(data.SuicidesNo)

# In[ ]:


min(data.SuicidesNo)

# In[ ]:


data[data.SuicidesNo==max(data.SuicidesNo)]

# In[ ]:


data[data['Country']=='Russian Federation'].hist()
plt.tight_layout()
plt.show()

# In[ ]:


data[data.SuicidesNo==min(data.SuicidesNo)]

# In[ ]:


data[data['Country']=='Albania'].hist()
plt.tight_layout()
plt.show()

# In[ ]:


# Show the results of a linear regression within each dataset
sns.lmplot(x="Population", y="GdpPerCapitalMoney", data=data)
plt.show()

# In[ ]:


sns.set(style="whitegrid")

# Make an example dataset with y ~ x
rs = np.random.RandomState(7)
x = rs.normal(2, 1, 75)
y = 2 + 1.5 * x + rs.normal(0, 2, 75)

# Plot the residuals after fitting a linear model
sns.residplot(x, y, lowess=True, color="g")
plt.show()

# In[ ]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data.iloc[:,-1]=le.fit_transform(data.iloc[:,-1])

# In[ ]:


data.head(2)

# In[ ]:


sns.pairplot(data, hue="Generation")
plt.show()

# <p id='20'><h3><b>References</b></h3></p>
# <p>https://www.kaggle.com/spscientist/students-performance-in-exams</p>
# <p>https://seaborn.pydata.org/</p>
# <p>https://www.kaggle.com/kanncaa1/seaborn-tutorial-for-beginners</p>
# <p>https://www.kaggle.com/biphili/seaborn-plot-to-visualize-iris-data</p>
# <p>https://www.kaggle.com/kralmachine/seaborn-tutorial-for-beginners</p>

# <p id='21'><h3><b>Conclusion</b></h3></p>
# <p>As a result, we have explained the seaborn library in a very detailed way and created a wide variety of graphs. If you like it, I expect your support. If you like <b>UPVOTED</b> I would be very happy if you do. If you have any questions, I am ready to answer your questions. At the bottom there are the kernel values that I have already done.</p>
# <p>https://www.kaggle.com/kralmachine/data-visualization-of-suicide-rates</p>
# <p>https://www.kaggle.com/kralmachine/gradient-admission-eda-ml-0-92</p>
# <p>https://www.kaggle.com/kralmachine/football-results-from-1872-to-2018-datavisulation</p>
# <p>https://www.kaggle.com/kralmachine/pandas-tutorial-for-beginner</p>
# <p>https://www.kaggle.com/kralmachine/visual-analysis-of-world-happiness-in-2015</p>
