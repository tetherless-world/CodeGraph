#!/usr/bin/env python
# coding: utf-8

# # Novice to Grandmaster- What Data Scientists say?
# 
# -Ashwin

# ![](https://www.kaggle.com/static/images/host-home/host-home-recruiting.png?v=daa83a72816a)

# Kaggle is the world's largest Data Science platform with more than 1 million users, and it is an excellent platform for students like me to learn and grow in the field of Data Science and Machine Learning. It has users from various domains,like statisticians,Data Scientists and Machine Learning Practitioners.This dataset published by Kaggle is a gem for people like me, who like to analyse and investigate data. In this notebook, we will try to find some trending or some common questions, each budding data scientist would like to know, like the most used tools, the resources to learn data science ,etc. 
# 
# The biggest problem that we might face is fake and bogus responses. As it is a survey, not everyone will answer with proper credentials, and thus I assume that there will be a lot many outlier. Let's dive in straight into the pool of data and gain some insights..

# # Introduction 

# ### Who are Data Scientists?
# 
# A data scientist is a statistician or a programmer, who cleans, manages and organizes data, perform descriptive statistics and analysis to develop insights,build predictive models and solve business related problems. Let's see what do Data Scientists on kaggle say..

# In[ ]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import squarify
plt.style.use('fivethirtyeight')
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# In[ ]:


response=pd.read_csv('../input/multipleChoiceResponses.csv',encoding='ISO-8859-1')

# In[ ]:


response.head()

# ## Some Basic Analysis

# In[ ]:


print('The total number of respondents:',response.shape[0])
print('Total number of Countries with respondents:',response['Country'].nunique())
print('Country with highest respondents:',response['Country'].value_counts().index[0],'with',response['Country'].value_counts().values[0],'respondents')
print('Youngest respondent:',response['Age'].min(),' and Oldest respondent:',response['Age'].max())

# Seriously?? Youngest Rspondent is not even a year old. LOL!! And how come grandpa is still coding at the age of 100. It may be a fake response.

# ## Gender Split

# In[ ]:


f,ax=plt.subplots(1,2,figsize=(22,8))
response['GenderSelect'].value_counts().plot.pie(ax=ax[0],explode=[0,0.1,0,0],shadow=True,autopct='%1.1f%%')
sns.countplot(y=response['GenderSelect'],ax=ax[1])
plt.subplots_adjust(wspace=0.8)
ax[0].set_ylabel('')
ax[1].set_ylabel('')
plt.show()

# The graph clearly shows that there are a lot more male respondents as compared to female. It seems that Ladies were either busy with their coding, **or ladies don't code**...:p. Just Kidding.

# ## Respondents By Country

# In[ ]:


resp_coun=response['Country'].value_counts()[:15].to_frame()
sns.barplot(resp_coun['Country'],resp_coun.index,palette='inferno')
plt.title('Top 15 Countries by number of respondents')
plt.xlabel('')
fig=plt.gcf()
fig.set_size_inches(10,10)
plt.show()
tree=response['Country'].value_counts().to_frame()
squarify.plot(sizes=tree['Country'].values,label=tree.index,color=sns.color_palette('RdYlGn_r',52))
plt.rcParams.update({'font.size':20})
fig=plt.gcf()
fig.set_size_inches(40,15)
plt.show()

# **USA and India**, constitute maximum respondents, about 1/3 of the total. Similarly Chile has the lowest number of respondents. Is this graph sufficient enough to say that majority of Kaggle Users are from India and USA. I don't think so, as the total users on Kaggle are more than 1 million while the number of respondents are only 16k.

# ## Compensation
# 
# Data Scientists are one of the most highest payed indviduals. Lets check what the surveyors say..

# In[ ]:


response['CompensationAmount']=response['CompensationAmount'].str.replace(',','')
response['CompensationAmount']=response['CompensationAmount'].str.replace('-','')
rates=pd.read_csv('../input/conversionRates.csv')
rates.drop('Unnamed: 0',axis=1,inplace=True)
salary=response[['CompensationAmount','CompensationCurrency','GenderSelect','Country','CurrentJobTitleSelect']].dropna()
salary=salary.merge(rates,left_on='CompensationCurrency',right_on='originCountry',how='left')
salary['Salary']=pd.to_numeric(salary['CompensationAmount'])*salary['exchangeRate']
print('Maximum Salary is USD $',salary['Salary'].dropna().astype(int).max())
print('Minimum Salary is USD $',salary['Salary'].dropna().astype(int).min())
print('Median Salary is USD $',salary['Salary'].dropna().astype(int).median())

# Look at that humungous Salary!! Thats **even larger than GDP of many countries**. Another example of bogus response. The minimum salary maybe a case of a student. The median salary shows that Data Scientist enjoy good salary benefits.

# In[ ]:


plt.subplots(figsize=(15,8))
salary=salary[salary['Salary']<1000000]
sns.distplot(salary['Salary'])
plt.title('Salary Distribution',size=15)
plt.show()

# ### Compensation by Country

# In[ ]:


f,ax=plt.subplots(1,2,figsize=(25,10))
sal_coun=salary.groupby('Country')['Salary'].median().sort_values(ascending=False)[:15].to_frame()
sns.barplot('Salary',sal_coun.index,data=sal_coun,palette='RdYlGn',ax=ax[0])
ax[0].axvline(salary['Salary'].median(),linestyle='dashed')
ax[0].set_title('Highest Salary Paying Countries')
ax[0].set_xlabel('')
max_coun=salary.groupby('Country')['Salary'].median().to_frame()
max_coun=max_coun[max_coun.index.isin(resp_coun.index)]
sns.barplot('Salary',max_coun.index,data=max_coun,palette='RdYlGn',ax=ax[1])
ax[1].axvline(salary['Salary'].median(),linestyle='dashed')
ax[1].set_title('Compensation of Top 15 Respondent Countries')
ax[1].set_xlabel('')
plt.show()

# The left graph shows the Top 15 high median salary paying countries. It is good to see that these countries provide salary more than the median salary of the complete dataset. Similarly,the right graph shows median salary of the Top 15 Countries by respondents. The most shocking graph is for **India**. India has the 2nd highest respondents, but still it has the lowest median salary in the graph. Individuals in USA have a salary almost 10% more than their counterparts in India. What may be the reason?? Are IT professionals in India really underpaid?? We will check that later.

# ### Salary By Gender

# In[ ]:


plt.subplots(figsize=(10,8))
sns.boxplot(y='GenderSelect',x='Salary',data=salary)
plt.ylabel('')
plt.show()

# The salary for males look to be high as compared to others.

# ## Age

# In[ ]:


plt.subplots(figsize=(15,8))
response['Age'].hist(bins=50,edgecolor='black')
plt.xticks(list(range(0,80,5)))
plt.title('Age Distribution')
plt.show() 

# The respondents are young people with majority of them being in the age bracket if 25-35.

# ## Profession & Major

# In[ ]:


f,ax=plt.subplots(1,2,figsize=(25,15))
sns.countplot(y=response['MajorSelect'],ax=ax[0])
ax[0].set_title('Major')
ax[0].set_ylabel('')
sns.countplot(y=response['CurrentJobTitleSelect'],ax=ax[1])
ax[1].set_title('Current Job')
ax[1].set_ylabel('')
plt.subplots_adjust(wspace=0.8)
plt.show()

# Data Science and Machine Learning is used in almost every industry. This is evident from the left graph,as people from different areas of interest like Physics, Biology, etc are taking it up for better understanding of the data. The right side graph shows the Current Job of the respondents. A major portion of the respondents are Dats Scientists. But as it is survey data, we know that there may be many ambigious responses. Later on we will check are these respondents real datas-scientists or self proclaimed data-scientists.

# ## Compensation By Job Title

# In[ ]:


sal_job=salary.groupby('CurrentJobTitleSelect')['Salary'].median().to_frame()
ax=sal_job.plot.barh(width=0.9,color='orange')
plt.title('Compensation By Job Title',size=15)
for i, v in enumerate(sal_job.Salary): 
    ax.text(0.1, i, v,fontsize=10,color='white',weight='bold')
fig=plt.gcf()
fig.set_size_inches(10,10)
plt.show()

# Operations Research Practitioner has the highest median salary followed by Predictive Modeler and Data Scientist. Computer Scientist and Programmers have the lowest compensation.

# ## Machine Learning

# In[ ]:


f,ax=plt.subplots(1,2,figsize=(25,12))
skills=response['MLSkillsSelect'].str.split(',')
skills_set=[]
for i in skills.dropna():
    skills_set.extend(i)
pd.Series(skills_set).value_counts().sort_values(ascending=True).plot.barh(width=0.9,color=sns.color_palette('inferno',15),ax=ax[0])
ax[0].set_title('ML Skills')
tech=response['MLTechniquesSelect'].str.split(',')
techniques=[]
for i in tech.dropna():
    techniques.extend(i)
pd.Series(techniques).value_counts().sort_values(ascending=True).plot.barh(width=0.9,color=sns.color_palette('inferno',15),ax=ax[1])
ax[1].set_title('ML Techniques used')
plt.subplots_adjust(wspace=0.8)
plt.show()

# It is evident that most of the respondents are working with Supervised Learning, and Logistic Regression being the favorite among them.

# In[ ]:


f,ax=plt.subplots(1,2,figsize=(25,12))
ml_nxt=response['MLMethodNextYearSelect'].str.split(',')
nxt_year=[]
for i in ml_nxt.dropna():
    nxt_year.extend(i)
pd.Series(nxt_year).value_counts()[:15].sort_values(ascending=True).plot.barh(width=0.9,color=sns.color_palette('winter_r',15),ax=ax[0])
tool=response['MLToolNextYearSelect'].str.split(',')
tool_nxt=[]
for i in tool.dropna():
    tool_nxt.extend(i)
pd.Series(tool_nxt).value_counts()[:15].sort_values(ascending=True).plot.barh(width=0.9,color=sns.color_palette('winter_r',15),ax=ax[1])
plt.subplots_adjust(wspace=0.8)
ax[0].set_title('ML Method Next Year')
ax[1].set_title('ML Tool Next Year')
plt.show()

# It is evident that the next year is going to see a jump in number of **Deep Learning** practitioners. Deep Learning and neural nets or in short AI is a favorite hot-topic for the next Year. Also in terms of Tools, Python is preferred more over R. Big Data Tools like Spark and Hadoop also have a good share in the coming years.

# ## Best Platforms to Learn

# In[ ]:


plt.subplots(figsize=(6,8))
learn=response['LearningPlatformSelect'].str.split(',')
platform=[]
for i in learn.dropna():
    platform.extend(i)
pd.Series(platform).value_counts()[:15].sort_values(ascending=True).plot.barh(width=0.9,color=sns.color_palette('winter',15))
plt.title('Best Platforms to Learn',size=15)
plt.show()

# My personal Kaggle, is the most sought after source for learning Data Science.

# ## Hardware Used

# In[ ]:


plt.subplots(figsize=(10,10))
hard=response['HardwarePersonalProjectsSelect'].str.split(',')
hardware=[]
for i in hard.dropna():
    hardware.extend(i)
pd.Series(hardware).value_counts().sort_values(ascending=True).plot.barh(width=0.9,color=sns.color_palette('inferno',10))
plt.title('Machines Used')
plt.show()

# Since majority of the respondents fall in the age category below 25, which is where a majority of students fall under, thus a basic Laptop is the most commonly used machine for work.

# ## Where Do I get Datasets From??

# In[ ]:


plt.subplots(figsize=(15,15))
data=response['PublicDatasetsSelect'].str.split(',')
dataset=[]
for i in data.dropna():
    dataset.extend(i)
pd.Series(dataset).value_counts().plot.pie(autopct='%1.1f%%',colors=sns.color_palette('Paired',10),startangle=90,wedgeprops = { 'linewidth' : 2, 'edgecolor' : 'white' })
plt.title('Dataset Source')
my_circle=plt.Circle( (0,0), 0.7, color='white')
p=plt.gcf()
p.gca().add_artist(my_circle)
plt.ylabel('')
plt.show()

# With hundreds of Dataset available, Kaggle is the most sought after source for datasets.

# ## Code Sharing

# In[ ]:


plt.subplots(figsize=(15,15))
code=response['WorkCodeSharing'].str.split(',')
code_share=[]
for i in code.dropna():
    code_share.extend(i)
pd.Series(code_share).value_counts().plot.pie(autopct='%1.1f%%',shadow=True,colors=sns.color_palette('Set3',10),startangle=90,wedgeprops = { 'linewidth' : 2, 'edgecolor' : 'white' })
plt.title('Code Sharing Medium')
my_circle=plt.Circle( (0,0), 0.65, color='white')
p=plt.gcf()
p.gca().add_artist(my_circle)
plt.ylabel('')
plt.show()

# ## Challenges in Data Science

# In[ ]:


plt.subplots(figsize=(15,15))
challenge=response['WorkChallengesSelect'].str.split(',')
challenges=[]
for i in challenge.dropna():
    challenges.extend(i)
pd.Series(challenges).value_counts().sort_values(ascending=True).plot.barh(width=0.9,color=sns.color_palette('inferno',25))
plt.title('Challenges in Data Science')
plt.show()

# The main challenge in Data Science is **getting the proper Data**. The graph clearly shows that dirty data is the bigget challenge. Now what is dirty data?? Dirty data is a database record that contains errors. Dirty data can be caused by a number of factors including duplicate records, incomplete or outdated data, and the improper parsing of record fields from disparate systems. Luckily Kaggle datasets are pretty clean and standardised.
# 
# Some other major challenges are the **Lack of Data Science and machine learning talent, difficulty in getting data and lack of tools**. Thats why Data Science is the sexiest job in 21st century.With the increasing amount of data, this demand will substantially grow.

# ## Job Satisfaction

# In[ ]:


satisfy=response.copy()
satisfy['JobSatisfaction'].replace({'10 - Highly Satisfied':'10','1 - Highly Dissatisfied':'1','I prefer not to share':np.NaN},inplace=True)
satisfy.dropna(subset=['JobSatisfaction'],inplace=True)
satisfy['JobSatisfaction']=satisfy['JobSatisfaction'].astype(int)
satisfy_job=satisfy.groupby(['CurrentJobTitleSelect'])['JobSatisfaction'].mean().sort_values(ascending=True).to_frame()
ax=satisfy_job.plot.barh(width=0.9,color='orange')
fig=plt.gcf()
fig.set_size_inches(8,12)
for i, v in enumerate(satisfy_job.JobSatisfaction): 
    ax.text(.1, i, v,fontsize=10,color='white',weight='bold')
plt.title('Job Satisfaction out of 10')
plt.show()

# Data Scientists and Machine Learning engineers are the most satisfied people(who won't be happy with so much money), while Programmers have the lowest job satisfaction.
# 
# ## Job Satisfication By Country

# In[ ]:


satisfy=response.copy()
satisfy['JobSatisfaction'].replace({'10 - Highly Satisfied':'10','1 - Highly Dissatisfied':'1','I prefer not to share':np.NaN},inplace=True)
satisfy.dropna(subset=['JobSatisfaction'],inplace=True)
satisfy['JobSatisfaction']=satisfy['JobSatisfaction'].astype(int)
satisfy_job=satisfy.groupby(['Country'])['JobSatisfaction'].mean().sort_values(ascending=True).to_frame()
data = [ dict(
        type = 'choropleth',
        autocolorscale = False,
        colorscale = 'Viridis',
        reversescale = True,
        showscale = True,
        locations = satisfy_job.index,
        z = satisfy_job['JobSatisfaction'],
        locationmode = 'country names',
        text = satisfy_job['JobSatisfaction'],
        marker = dict(
            line = dict(color = 'rgb(200,200,200)', width = 0.5)),
            colorbar = dict(autotick = True, tickprefix = '', 
            title = 'Satisfaction')
            )
       ]

layout = dict(
    title = 'Job Satisfaction By Country',
    geo = dict(
        showframe = True,
        showocean = True,
        oceancolor = 'rgb(0,0,255)',
        projection = dict(
        type = 'orthographic',
            rotation = dict(
                    lon = 60,
                    lat = 10),
        ),
        lonaxis =  dict(
                showgrid = False,
                gridcolor = 'rgb(102, 102, 102)'
            ),
        lataxis = dict(
                showgrid = False,
                gridcolor = 'rgb(102, 102, 102)'
                )
            ),
        )
fig = dict(data=data, layout=layout)
py.iplot(fig, validate=False, filename='worldmap2010')

# # Python vs R or (Batman vs Superman)
# 
# ![](https://blog.webhose.io/wp-content/uploads/2017/08/python-versus-R.png)'
# 
# Python and R are the most widely used Open-Source languages for Data Science and Machine-Learning stuff. For a budding data scientist or analyst, the biggest and trickiest doubt is: **Which Language Should I Start With??** While both the languages have their own advantages and shortcomings, it depends on the individual's purpose while selecting a language of his/her choice. Both the languages cater the needs of different kinds of work. Python is a general purpose langauge, thus web and application integration is easier, while R is meant for pure statistical and analytics purpose. The area where R will completely beat Python is visualisations with the help of packages like **ggplot2 and shiny**. But Python has an upperhand in Machine Learning stuff. So lets see what the surveyers say..

# In[ ]:


resp=response.dropna(subset=['WorkToolsSelect'])
resp=resp.merge(rates,left_on='CompensationCurrency',right_on='originCountry',how='left')
python=resp[(resp['WorkToolsSelect'].str.contains('Python'))&(~resp['WorkToolsSelect'].str.contains('R'))]
R=resp[(~resp['WorkToolsSelect'].str.contains('Python'))&(resp['WorkToolsSelect'].str.contains('R'))]
both=resp[(resp['WorkToolsSelect'].str.contains('Python'))&(resp['WorkToolsSelect'].str.contains('R'))]

# ### Recommended Language For Begineers

# In[ ]:


response['LanguageRecommendationSelect'].value_counts()[:2].plot.bar()
plt.show()

# Clearly Python is the recommended language for begineers. The reason for this maybe due to its simple english-like syntax and general purpose functionality.

# ## Recommendation By Python and R users

# In[ ]:


labels1=python['LanguageRecommendationSelect'].value_counts()[:5].index
sizes1=python['LanguageRecommendationSelect'].value_counts()[:5].values

labels2=R['LanguageRecommendationSelect'].value_counts()[:5].index
sizes2=R['LanguageRecommendationSelect'].value_counts()[:5].values


fig = {
  "data": [
    {
      "values": sizes1,
      "labels": labels1,
      "domain": {"x": [0, .48]},
      "name": "Language",
      "hoverinfo":"label+percent+name",
      "hole": .4,
      "type": "pie"
    },     
    {
      "values": sizes2 ,
      "labels": labels2,
      "text":"CO2",
      "textposition":"inside",
      "domain": {"x": [.54, 1]},
      "name": "Language",
      "hoverinfo":"label+percent+name",
      "hole": .4,
      "type": "pie"
    }],
  "layout": {
        "title":"Language Recommended By Python and R users",
        "annotations": [
            {
                "font": {
                    "size": 30
                },
                "showarrow": False,
                "text": "Python",
                "x": 0.17,
                "y": 0.5
            },
            {
                "font": {
                    "size": 30
                },
                "showarrow": False,
                "text": "R",
                "x": 0.79,
                "y": 0.5}]}}
py.iplot(fig, filename='donut')

# This is a interesting find. About **91.6%** Python users recommend Python as the first language for begineers, whereas only **67.2%** R users recommend R as the first language. Also **20.6%** R users recommend Python but only **1.68%** Python users recommend R as the first language. One thing to note is that users of both recommend the same Languages i.e SQL, Matlab and C/C++. I have only considered the Top 5 recommended languages, so the percentage will change if we consider all of them. But the difference would be just 2-3%.

# ### Necessary or Not??

# In[ ]:


f,ax=plt.subplots(1,2,figsize=(18,8))
response['JobSkillImportancePython'].value_counts().plot.pie(ax=ax[0],autopct='%1.1f%%',explode=[0.1,0,0],shadow=True,colors=['g','lightblue','r'])
ax[0].set_title('Python Necessity')
ax[0].set_ylabel('')
response['JobSkillImportanceR'].value_counts().plot.pie(ax=ax[1],autopct='%1.1f%%',explode=[0,0.1,0],shadow=True,colors=['lightblue','g','r'])
ax[1].set_title('R Necessity')
ax[1].set_ylabel('')
plt.show()

# Clearly Python is a much more necessary skill compared to R.
# 
# Special Thanks to [Steve Broll](https://www.kaggle.com/stevebroll) for helping in the color scheme.

# ### Number Of Users By Language

# In[ ]:


plt.subplots(figsize=(10,6))
ax=pd.Series([python.shape[0],R.shape[0],both.shape[0]],index=['Python','R','Both']).plot.bar()
ax.set_title('Number of Users',size=15)
for p in ax.patches:
    ax.annotate(format(p.get_height()), (p.get_x()+0.15, p.get_height()+0.25),size=12)
plt.show()

# The number of Python users are definetely more than R users. This may be due to the easy learning curve of Python. However there are more users who know both the languages. These responses might be from established Data Scientists,as they tend to have a knowledge in multiple languages and tools.

# ## Compensation

# In[ ]:


py_sal=(pd.to_numeric(python['CompensationAmount'].dropna())*python['exchangeRate']).dropna()
py_sal=py_sal[py_sal<1000000]
R_sal=(pd.to_numeric(R['CompensationAmount'].dropna())*R['exchangeRate']).dropna()
R_sal=R_sal[R_sal<1000000]
both_sal=(pd.to_numeric(both['CompensationAmount'].dropna())*both['exchangeRate']).dropna()
both_sal=both_sal[both_sal<1000000]
trying=pd.DataFrame([py_sal,R_sal,both_sal])
trying=trying.transpose()
trying.columns=['Python','R','Both']
print('Median Salary For Individual using Python:',trying['Python'].median())
print('Median Salary For Individual using R:',trying['R'].median())
print('Median Salary For Individual knowing both languages:',trying['Both'].median())


# In[ ]:


trying.plot.box()
plt.title('Compensation By Language')
fig=plt.gcf()
fig.set_size_inches(10,6)
plt.show()

# Python coders have a slightly higher median salary as that compared to their R counterparts. However, the people who know both these languages, have a pretty high median salary as compared to both of them.
# 
# ## Language Used By Professionals

# In[ ]:


py1=python.copy()
r=R.copy()
py1['WorkToolsSelect']='Python'
r['WorkToolsSelect']='R'
r_vs_py=pd.concat([py1,r])
r_vs_py=r_vs_py.groupby(['CurrentJobTitleSelect','WorkToolsSelect'])['Age'].count().to_frame().reset_index()
r_vs_py.pivot('CurrentJobTitleSelect','WorkToolsSelect','Age').plot.barh(width=0.8)
fig=plt.gcf()
fig.set_size_inches(10,15)
plt.title('Job Title vs Language Used',size=15)
plt.show()

# As I had mentioned earlier, R beats Python in visuals. Thus people with Job-Titles like Data Analyst, Business Analyst where graphs and visuals play a very prominent role, prefer R over Python. Similarly almost 90% of statisticians use R. Also as stated earlier, Python is better in Machine Learning stuff, thus Machine Learning engineers, Data Scientists and others like DBA or Programmers prefer Python over R. 
# 
# Thus for data visuals--->R else---->Python.
# 
# **Note: This graph is not for Language Recommended by professionals, but the tools used by the professionals.**

# ## Job Function vs Language

# In[ ]:


r_vs_py=pd.concat([py1,r])
r_vs_py=r_vs_py.groupby(['JobFunctionSelect','WorkToolsSelect'])['Age'].count().to_frame().reset_index()
r_vs_py.pivot('JobFunctionSelect','WorkToolsSelect','Age').plot.barh(width=0.8)
fig=plt.gcf()
fig.set_size_inches(10,15)
plt.title('Job Description vs Language Used')
plt.show()

# As I had already mentioned ** R excels in analytics, but Python beats in Machine Learning.** The graph shows that R has influence when it comes to pure analytics, but other ways python wins.

# ## Tenure vs Language Used

# In[ ]:


r_vs_py=pd.concat([py1,r])
r_vs_py=r_vs_py.groupby(['Tenure','WorkToolsSelect'])['Age'].count().to_frame().reset_index()
r_vs_py.pivot('Tenure','WorkToolsSelect','Age').plot.barh(width=0.8)
fig=plt.gcf()
fig.set_size_inches(10,10)
plt.title('Job Tenure vs Language Used')
plt.show()

# As we had seen earlier, Python is highly recommended for beginners. Thus the proportion of Python users is more in the initial years of coding. The gap between the languages however reduces over the years, as the coding experience increases.

# ## Common Tools with Python and R

# In[ ]:


f,ax=plt.subplots(1,2,figsize=(20,15))
py_comp=python['WorkToolsSelect'].str.split(',')
py_comp1=[]
for i in py_comp:
    py_comp1.extend(i)
pd.Series(py_comp1).value_counts()[1:15].sort_values(ascending=True).plot.barh(width=0.9,ax=ax[0],color=sns.color_palette('inferno',15))
R_comp=R['WorkToolsSelect'].str.split(',')
R_comp1=[]
for i in R_comp:
    R_comp1.extend(i)
pd.Series(R_comp1).value_counts()[1:15].sort_values(ascending=True).plot.barh(width=0.9,ax=ax[1],color=sns.color_palette('inferno',15))
ax[0].set_title('Commonly Used Tools with Python')
ax[1].set_title('Commonly Used Tools with R')
plt.subplots_adjust(wspace=0.8)
plt.show()

# **SQL** seems to be the most common complementory tool used with both the languages.

# # Asking the Data Scientists
# 
# 
# 
# ![](https://ewebdesign.com/wp-content/uploads/2017/01/zarget_banner2.gif)
# 
# If I successfully write a Hello World program, then does that make me programmer or a developer?? If I beat my friends in a race, then does that make me the fastest person on the earth?? The answer is pretty obviously **NO.** This is the problem with the emerging Computer Science and IT folks. Based on their limited skills and experience, they start considering themselves much more than they really are. Many of them start calling them Machine Learning Practitioners even if they haven't really worked on real life projects and have just worked on some simple datasets. Similarly many responses here must be a bluff response. Lets check how many of them are the real Data Science practitioners.

# In[ ]:


response['DataScienceIdentitySelect'].value_counts()

# So about 26% of the total respondents consider themselves as Data Scientist. What does Sort of mean?? Are they still learning or are they unemployed. For now lets consider them as a No.
# 
# ## Current Job Titles

# In[ ]:


plt.subplots(figsize=(10,8))
scientist=response[response['DataScienceIdentitySelect']=='Yes']
scientist['CurrentJobTitleSelect'].value_counts().sort_values(ascending=True).plot.barh(width=0.9,color=sns.color_palette('inferno',15))
plt.title('Job Titles',size=15)
plt.show()

# Surprisingly there is **no entry for the Job Title Data Scientist**. There reasons for this could be that the people with CurrentJobTitleSelect as Data Scientist(who might be working as Data Scientist) might have not answered the question: **"Do you currently consider yourself a Data Scientist?"**
# 
# There are many overlapping and common skills between the jobs like Data Analyst,Data Scientist and Machine Learning experts, Statisticians,etc. Thus they too have similar skills and consider themselves as Data Scientists even though they are not labeled the same. Now lets check if the previous assumption was True.

# In[ ]:


true=response[response['CurrentJobTitleSelect']=='Data Scientist']

# It was indeed **True**. People with their CurrentJobTitle as Data Scientist did not answer the question **"Do you currently consider yourself a Data Scientist?"**. So I am considering them also to be real Data Scientists.

# In[ ]:


scientist=pd.concat([scientist,true])
scientist['CurrentJobTitleSelect'].shape[0]

# So out of the total respondents, about **40%** of them are Data Scientists or have skills for the same.
# 
# ## Country-Wise Split

# In[ ]:


plt.subplots(figsize=(10,8))
scientist['Country'].value_counts()[:15].sort_values(ascending=True).plot.barh(width=0.9,color=sns.color_palette('inferno',25))
plt.title('Countries By Number Of Data Scientists',size=15)
plt.show()

# The graph is similar to the demographic graph where we had shown number of users by country. The difference is that the numbers have reduced as we have only considered Data Scientists.
# 
# ## Employment Status

# In[ ]:


plt.subplots(figsize=(8,6))
sns.countplot(y=scientist['EmploymentStatus'])
plt.show()

# About **67%** of the data scientists are employed full-time, while about **11-12%** of them are unemployed but looking for job.
# 
# ## Previous Job and Salary Change

# In[ ]:


f,ax=plt.subplots(1,2,figsize=(30,15))
past=scientist['PastJobTitlesSelect'].str.split(',')
past_job=[]
for i in past.dropna():
    past_job.extend(i)
pd.Series(past_job).value_counts().sort_values(ascending=True).plot.barh(width=0.9,color=sns.color_palette('summer',25),ax=ax[0])
ax[0].set_title('Previous Job')
sal=scientist['SalaryChange'].str.split(',')
sal_change=[]
for i in sal.dropna():
    sal_change.extend(i)
pd.Series(sal_change).value_counts().sort_values(ascending=True).plot.barh(width=0.9,color=sns.color_palette('summer',10),ax=ax[1])
ax[1].set_title('Salary Change')
plt.subplots_adjust(wspace=0.8)
plt.show()

# Clearly majority of people switching to Data Science get a salary hike about **6-20% or more**.
# 
# ## Tools used at Work

# In[ ]:


plt.subplots(figsize=(8,8))
tools=scientist['WorkToolsSelect'].str.split(',')
tools_work=[]
for i in tools.dropna():
    tools_work.extend(i)
pd.Series(tools_work).value_counts()[:15].sort_values(ascending=True).plot.barh(width=0.9,color=sns.color_palette('RdYlGn',15))
plt.show()

# Similar observations, Python, R and SQL are the most used tools or languages in Data Science
# 
# 

# ## Where Did they Learn From??

# In[ ]:


course=scientist['CoursePlatformSelect'].str.split(',')
course_plat=[]
for i in course.dropna():
    course_plat.extend(i)
course_plat=pd.Series(course_plat).value_counts()
blogs=scientist['BlogsPodcastsNewslettersSelect'].str.split(',')
blogs_fam=[]
for i in blogs.dropna():
    blogs_fam.extend(i)
blogs_fam=pd.Series(blogs_fam).value_counts()
labels1=course_plat.index
sizes1=course_plat.values

labels2=blogs_fam[:5].index
sizes2=blogs_fam[:5].values


fig = {
  "data": [
    {
      "values": sizes1,
      "labels": labels1,
      "domain": {"x": [0, .48]},
      "name": "MOOC",
      "hoverinfo":"label+percent+name",
      "hole": .4,
      "type": "pie"
    },     
    {
      "values": sizes2 ,
      "labels": labels2,
      "text":"CO2",
      "textposition":"inside",
      "domain": {"x": [.54, 1]},
      "name": "Blog",
      "hoverinfo":"label+percent+name",
      "hole": .4,
      "type": "pie"
    }],
  "layout": {
        "title":"Blogs and Online Platforms",
        "showlegend":False,
        "annotations": [
            {
                "font": {
                    "size": 25
                },
                "showarrow": False,
                "text": "MOOC's",
                "x": 0.17,
                "y": 0.5
            },
            {
                "font": {
                    "size": 25
                },
                "showarrow": False,
                "text": "BLOGS",
                "x": 0.85,
                "y": 0.5}]}}
py.iplot(fig, filename='donut')

# The average Job Satisfaction level is between **6-7.5** for most of the countries. It is lower in Japan(where people work for about 14 hours) and China. It is higher in come countries like Sweden and Mexico.
# 
# ## Time Spent on Tasks
# 
# A Data Scientist is not always building predictive models, he is also responsible for the data quality, gathering the right data, analytics,etc. Lets see how much time a data scientist spends on these differnt tasks.

# In[ ]:


import itertools
plt.subplots(figsize=(22,10))
time_spent=['TimeFindingInsights','TimeVisualizing','TimeGatheringData','TimeModelBuilding']
length=len(time_spent)
for i,j in itertools.zip_longest(time_spent,range(length)):
    plt.subplot((length/2),2,j+1)
    plt.subplots_adjust(wspace=0.2,hspace=0.5)
    scientist[i].hist(bins=10,edgecolor='black')
    plt.axvline(scientist[i].mean(),linestyle='dashed',color='r')
    plt.title(i,size=20)
plt.show()

# Lets do it stepwise:
# 
#   - **TimeGatheringData:** It is undoubtedly the most time consuming part. Getting the data is the most painstaking task in the entire process, which is followed by Data Cleaning(not shown as data not available) which is yet other time consuming process. Thus gathering right data and scrubing the data are the most time consuming process.
#   
#   - **TimeVisualizing:** It is probably the least time consuming process(and probably the most enjoyable one..:p), and it reduces even further if we use Enterprise Tools like Tableau,Qlik,Tibco,etc, which helps in building graphs and dashboards with simple drag and drop features.
#   
#   - **TimeFindingInsights:** It is followed after visualising the data, which involves finding facts and patterns in the data, slicing and dicing it to find insights for business processes.It looks to a bit more time consuming as compared to TimeVisualizing.
#   
#   - **TimeModelBuilding:** It is where the data scientists build predictive models, tune these models,etc. It is the 2nd most time consuming process after TimeDataGathering.
# 
# ## Importance Of Visualisations

# In[ ]:


f,ax=plt.subplots(1,2,figsize=(25,12))
sns.countplot(scientist['JobSkillImportanceVisualizations'],ax=ax[0])
ax[0].set_title('Job Importance For Visuals')
ax[0].set_xlabel('')
scientist['WorkDataVisualizations'].value_counts().plot.pie(autopct='%2.0f%%',colors=sns.color_palette('Paired',10),ax=ax[1])
ax[1].set_title('Use Of Visualisations in Projects')
my_circle=plt.Circle( (0,0), 0.7, color='white')
p=plt.gcf()
p.gca().add_artist(my_circle)
plt.ylabel('')
plt.show()

# Visualisations are a very integral part of Data Science Projects, and the above graph also shows the same. Almost all data science projects i.e **99%** of the projects have visualisations in them, doesn't matter how big or small. About **95%** of Data Scientists say that Visualisations skills are nice to have or necessary.Visuals help to understand and comprehend the data faster not only to the professionals but also to target customers, who may not be technically skilled.
# 
# ## Knowledge Of Algorithms (Maths and Stats)

# In[ ]:


f,ax=plt.subplots(1,2,figsize=(25,12))
sns.countplot(y=scientist['AlgorithmUnderstandingLevel'],ax=ax[0])
sns.countplot(response['JobSkillImportanceStats'],ax=ax[1])
ax[0].set_title('Algorithm Understanding')
ax[0].set_ylabel('')
ax[1].set_title('Knowledge of Stats')
ax[1].set_xlabel('')
plt.show()

# Data Scientists have a good knowledge of mathematical concepts like Statistics and Linear Algebra, which are the most important part of Machine Learning algorithms. But is this maths really required, as many standard libraries like scikit,tensorflow,keras etc have all these things already implemented. But the experienced data scientists say that we should have a good understanding of the maths behind the algorithms. About **95%** of the data scientists say the stats is an important asset in Data Science.
# 
# ## Learning Platform Usefullness

# In[ ]:


plt.subplots(figsize=(25,35))
useful=['LearningPlatformUsefulnessBlogs','LearningPlatformUsefulnessCollege','LearningPlatformUsefulnessCompany','LearningPlatformUsefulnessKaggle','LearningPlatformUsefulnessCourses','LearningPlatformUsefulnessProjects','LearningPlatformUsefulnessTextbook','LearningPlatformUsefulnessYouTube']
length=len(useful)
for i,j in itertools.zip_longest(useful,range(length)):
    plt.subplot((length/2),3,j+1)
    plt.subplots_adjust(wspace=0.2,hspace=0.2)
    scientist[i].value_counts().plot.pie(autopct='%2.0f%%',colors=['g','lightblue','r'],wedgeprops = { 'linewidth' : 2, 'edgecolor' : 'white' })
    plt.title(i,size=25)
    my_circle=plt.Circle( (0,0), 0.7, color='white')
    p=plt.gcf()
    p.gca().add_artist(my_circle)
    plt.xlabel('')
    plt.ylabel('')
plt.show()

# The above donut charts shows the opinion of Data Scientists about the various platforms to learn Data Science. The plot looks best for **Projects**,where the percentage for not useful is **0%**.According to my personal opinion too, projects are the best platform or way for learning anything in the IT industry. The other excellent platforms are **Online Courses and Kaggle**. The graphs for other platforms are quite similar to each other.
# 
# ## What should the Resume have??

# In[ ]:


f,ax=plt.subplots(1,2,figsize=(22,8))
sns.countplot(y=scientist['ProveKnowledgeSelect'],ax=ax[0])
ax[0].set_title('How to prove my knowledge')
sns.countplot(scientist['JobSkillImportanceKaggleRanking'],ax=ax[1])
ax[1].set_title('Kaggle Rank')
plt.show()

# It is evident that Work experience in ML projects and Kaggle competitions reflects the knowledge of Data Science. Also a kaggle rank can be a good thing in one's resume.
# 
# # Conclusions
# 
# Some brief insights that we gathered from the notebook:
# 
# 1) Majority of the respondents are from USA followed by India. USA also had the maximum number of data scientists followed by India. Also the median Salary is highest in USA.
# 
# 2) Majority of the respondents are in the age bracket 20-35, which shows that data science is quite famous in the youngsters.
# 
# 3) The respondents are not just limited to Computer Science major, but also from majors like Statistics, health sciences,etc showing that Data Science is an interdisciplinary domain.
# 
# 4) Majority of the respondents are fully employed.
# 
# 5) Kaggle, Online Courses(Coursera,eDx,etc), Projects and Blogs(KDNuggets,AnalyticsVidya,etc) are the top resources/platforms for learning Data Science.
# 
# 6) Kaggle has the highest share for data acquisition whereas Github has the highest share for code sharing.
# 
# 7) Data Scientists have the highest Job Satisfaction level and the second highest median salary (after Operations Research Analyst). On the contrary, Programmers have the least Job Satisfaction level and one of the least median salary also.
# 
# 8) Data Scientists also get a hike of about 6-20% from their previous jobs.
# 
# #### Tips For Budding Data Scientists
# 
# 1) Learn **Python,R and SQL** as they are the most used languages by the Data Scientists. Python and R will help in analytics and predictive modeling while SQL is best for querying the databases.
# 
# 2) Learn Machine Learning Techniques like **Logistic Regression, Decision Trees, Support Vector Machines**, etc as they are most commonly used Machine Learning techniques/algorithms.
# 
# 3) **Deep Learning and Neural Nets** will be the most sought after techniques in the future, thus a good knowledge in them will be very helpful.
# 
# 4) Develop skills for **Gathering Data** and **Cleaning The Data** as they are the most time consuming processes in the workflow of a data scientist. 
# 
# 5) **Visualisations** are very important in Data Science projects and almost all projects require Visualisations for understanding the data better. So one should learn Data Visualisation as Data Scientists consider it to be a **necessary or nice to have skill.**
# 
# 6) **Maths and Stats** are very important in Data Science, so we should have good understanding of it for actually understanding how the algorithm works.
# 
# 7) **Projects** are the best way to learn Data Science according to Data Scientists.So working on projects will help you learn data science better.
# 
# 8) **Experience with ML Projects in company and Kaggle Competitions** are the best ways to show your working knowledge in Data Science. Working on ML projects in a company gives the experience of working with real world datasets, thereby enhancing the knowledge. Kaggle competitions are also a great medium, as you will be competing with Data Scientists over the world. Also a **Kaggle Rank** can be a good USP in the resume.
# 

# So I would like my conclude my analysis here. Thanks a lot for having a look at this notebook.
# 
# I Hope all you liked the notebook. Any suggestions and feedback are always welcome.
# 
# ### Please Upvote this notebook as it encourages me in doing better.
# 
# 
# ![](http://68.media.tumblr.com/e1aed171ded2bd78cc8dc0e73b594eaf/tumblr_o17frv0cdu1u9u459o1_500.gif)

# In[ ]:



