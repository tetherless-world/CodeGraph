#!/usr/bin/env python
# coding: utf-8

# ![google](http://img.technews.tw/wp-content/uploads/2015/09/Google-logo_1.jpg)
# **Google** is one of my dream company. I believe lots of data scientists want to join Google, just like me. However, people might wonder what I need to acquire to be qualified to work in Google. I create this kernel to answer this question. And hope you enjoy!
# 
# # Outline
# 
# [Data Cleaning](#DC)   
# [Exploratory](#0)   
# [Functions](#12)
# 
# ## Positions
# [1. Analyst](#1)   
# &nbsp;&nbsp;&nbsp;&nbsp;[1.1 Languages and Degrees](#1.1)   
# [2. Developer](#2)   
# &nbsp;&nbsp;&nbsp;&nbsp;[2.1 Languages and Degrees](#2.1)   
# [3. MBA intern](#3)   
# &nbsp;&nbsp;&nbsp;&nbsp;[3.1 Languages and Degrees](#3.1)   
# [4. Sales](#4)   
# &nbsp;&nbsp;&nbsp;&nbsp;[4.1 Languages and Degrees](#4.1)   
# ## Tools
# [5. Microsoft Office](#5)   
# [6. Data Visualization Tools](#6)   
# [7. Statistical Analsis Tools](#7)
# 
# ## Positions Distribution
# 
# [8. Jobs in the US](#8)   
# [9. The so-called PMs](#9)   
# [10. Pivot table](#10)
# 
# ## [Conclusion](#11)

# ## Dataset Overlook

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')

# In[ ]:


df = pd.read_csv('../input/google-job-skills/job_skills.csv')

# In[ ]:


df.head()

# ## [Data Cleaning](#dc)

# In[ ]:


# I modify the column name so that I can use df dot column name more easily
df = df.rename(columns={'Minimum Qualifications': 'Minimum_Qualifications', 'Preferred Qualifications': 'Preferred_Qualifications'})

# I'll check if there is any NaN

# In[ ]:


pd.isnull(df).sum()

# I'll straightly drop these rows with NaN

# In[ ]:


df = df.dropna(how='any',axis='rows')

# Here, the first thing I want to check is the values_count of each column

# In[ ]:


df.Company.value_counts()

# Ah! Right. I forgot that Youtube is also part of Google. However, for me, working in Youtube is not as appealing as working in Google. But Youtube is still a great company. No offence haha!

# In[ ]:


# So I drop Youtube
df = df[df.Company != 'YouTube']

# In[ ]:


df.Title.value_counts()[:10]

# In[ ]:


df.Location.value_counts()[:10]

# In[ ]:


df['Country'] = df['Location'].apply(lambda x : x.split(',')[-1])

# In[ ]:


df.Country.value_counts()[:15]

# **Here, I want to extract the year of work experience in each position.**
# 
# The challenge is : 
# 
# * There might be some positions requiring work experience in different field
# * There might be some positions that don't mention work experience at all

# In[ ]:


from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 

stop_words = set(stopwords.words('english')) 

df['Responsibilities'] = df.Responsibilities.apply(lambda x: word_tokenize(x))
df['Responsibilities'] = df.Responsibilities.apply(lambda x: [w for w in x if w not in stop_words])
df['Responsibilities'] = df.Responsibilities.apply(lambda x: ' '.join(x))

df['Minimum_Qualifications'] = df.Minimum_Qualifications.apply(lambda x: word_tokenize(x))
df['Minimum_Qualifications'] = df.Minimum_Qualifications.apply(lambda x: [w for w in x if w not in stop_words])
df['Minimum_Qualifications'] = df.Minimum_Qualifications.apply(lambda x: ' '.join(x))

df['Preferred_Qualifications'] = df.Preferred_Qualifications.apply(lambda x: word_tokenize(x))
df['Preferred_Qualifications'] = df.Preferred_Qualifications.apply(lambda x: [w for w in x if w not in stop_words])
df['Preferred_Qualifications'] = df.Preferred_Qualifications.apply(lambda x: ' '.join(x))

# In[ ]:


# The way to extract year refer to https://www.kaggle.com/niyamatalmass/what-you-need-to-get-a-job-at-google.
# Thanks Niyamat Ullah for such brilliant way. Go check his kernel. It's great!
import re
df['Minimum_years_experience'] = df['Minimum_Qualifications'].apply(lambda x : re.findall(r'([0-9]+) year',x))
# Fill empty list with [0]
df['Minimum_years_experience'] = df['Minimum_years_experience'].apply(lambda y : [0] if len(y)==0 else y)
#Then extract maximum in the list to have the work experience requirement
df['Minimum_years_experience'] = df['Minimum_years_experience'].apply(lambda z : max(z))
df['Minimum_years_experience'] = df.Minimum_years_experience.astype(int)

# In[ ]:


df.head(3)

# In[ ]:


df.Minimum_years_experience.describe()

# In[ ]:


df.Category.value_counts()[:10]

# ## <a id="0">Exploratory</a>

# In[ ]:


pd.set_option('display.max_colwidth', -1)
df.head(1)

# Here, I want to extract degree requirement of each rows. Also, the language required for each role.

# In[ ]:


Degree = ['BA','BS','Bachelor','MBA','Master','PhD']

Degrees = dict((x,0) for x in Degree)
for i in Degree:
    x = df['Minimum_Qualifications'].str.contains(i).sum()
    if i in Degrees:
        Degrees[i] = x
        
print(Degrees)

# In[ ]:


degree_requirement = sorted(Degrees.items(), key=lambda x: x[1], reverse=True)
degree = pd.DataFrame(degree_requirement,columns=['Degree','Count'])
degree['Count'] = degree.Count.astype('int')
degree

# In[ ]:


degree.plot.barh(x='Degree',y='Count',legend=False)
plt.title('Degrees Distribution',fontsize=14)
plt.xlabel('Count')

# Obviously, most of the positions require basic degree, while some require further education degree, like Master and PhD.
# 
# Now, I want to see the distribution of the requiring work experience.

# In[ ]:


df.Minimum_years_experience.plot(kind='box')
plt.title('Minimum work experience')
plt.ylabel('Years')

# Well, obviously, there are few outliers. It must be some real senior positions.

# In[ ]:


import seaborn as sns
sns.countplot('Minimum_years_experience',data=df)
plt.suptitle('Minimum work experience')

# Basically, most of the position didn't mention experience. However, I'll dig deeper later.

# In[ ]:


Programming_Languages = ['Python', 'Java ','C#', 'PHP', 'Javascript', 'Ruby', 'Perl', 'SQL','Go ']

Languages = dict((x,0) for x in Programming_Languages)
for i in Languages:
    x = df['Minimum_Qualifications'].str.contains(i).sum()
    if i in Languages:
        Languages[i] = x
        
print(Languages)

# In[ ]:


languages_requirement = sorted(Languages.items(), key=lambda x: x[1], reverse=True)
language = pd.DataFrame(languages_requirement,columns=['Language','Count'])
language['Count'] = language.Count.astype('int')
language

# In[ ]:


language.plot.barh(x='Language',y='Count',legend=False)
plt.suptitle('Languages Distribution',fontsize=14)
plt.xlabel('Count')

# Python, SQL are also important, which indicates that the growing demand of data analysis

# ## <a id=12>Functions</a>
# As a guy on twitter said, " If you write a code more than three times, write a function instead." I'm here to define the functions that I'm going to use in the next few sections

# In[ ]:


def MadeWordCloud(title,text):
    df_subset = df.loc[df.Title.str.contains(title).fillna(False)]
    long_text = ' '.join(df_subset[text].tolist())
    sns.set(rc={'figure.figsize':(11.7,8.27)})
    wordcloud = WordCloud(mask=G,background_color="white").generate(long_text)
    plt.figure()
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.margins(x=0, y=0)
    plt.title(text,size=24)
    plt.show()

# ### Here, I want to create word clouds to know more about how to be qualified to be a competitive candidates for data-related positions in Google
# ## <a id="1">Analyst :</a>
# 
# **I'll demonstrate the original way I made word cloud here, after that, I'll replace it with the function**

# In[ ]:


# Refer to https://python-graph-gallery.com/262-worcloud-with-specific-shape/
# https://amueller.github.io/word_cloud/auto_examples/masked.html

df_Analyst = df.loc[df.Title.str.contains('Analyst').fillna(False)]

# In[ ]:


df_Analyst.head(1)

# In[ ]:


df_Analyst.Country.value_counts()

# In[ ]:


Res_AN = ' '.join(df_Analyst['Responsibilities'].tolist())

# In[ ]:


from wordcloud import WordCloud, ImageColorGenerator
from PIL import Image

G = np.array(Image.open('../input/googlelogo/img_2241.png'))
# I spent a while to realize that the image must be black-shaped to be a mask

# In[ ]:


sns.set(rc={'figure.figsize':(11.7,8.27)})

wordcloud = WordCloud(mask=G,background_color="white").generate(Res_AN)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.margins(x=0, y=0)
plt.title('Responsibilites',size=24)
plt.show()

# Here we can see some keywords to know more about what a Data Analyst do in Google
# 
# ### Keywords
# 
# * Criteria : Data/Team/Product/Business/Work
# * Insight : strategic/quality/key/projects/plan/identify/analysis/action/business/infrastructure
# * Audience : sales/operation/stakeholders
# * Verb : maintain/improve/support/model/draw/customize/identify/provide
# * Characteristic : leadership/quantitative/efficiency
# 
# To sum up in a sentence :
# 
# **Looking for analysts with business and data knowledge, familiar with product to work as a team**

# In[ ]:


MadeWordCloud('Analyst','Minimum_Qualifications')

# Here we can see some keywords to know how to meet minimum requirements to be a Data Analyst in Google
# 
# ### Keywords
# 
# * Fileds : Business/Computer Science/Mathematics/Statistics/Economics/Engineering
# * Degree : BS/BA
# * Languages : Python/SAS/JAVA/SQL
# * Tools : Tableau

# In[ ]:


MadeWordCloud('Analyst','Preferred_Qualifications')

# Here we can see some keywords to know how to be more competitive candidates to be a Data Analyst in Google
# 
# ### Keywords
# 
# * Fileds : Business/Computer Science/Mathematics/Statistics
# * Skills : Oral/Written/Comunication/Management
# * Experience : Consulting/Analytics/Developing/Cross-functioned

# ### <a id="1.1">I want to know what Google think about Python vs R</a>

# In[ ]:


DataSkill = [' R','Python','SQL','SAS']

DataSkills = dict((x,0) for x in DataSkill)
for i in DataSkill:
    x = df_Analyst['Minimum_Qualifications'].str.contains(i).sum()
    if i in DataSkill:
        DataSkills[i] = x
        
print(DataSkills)

# ### And then, the degrees

# In[ ]:


Degrees = dict((x,0) for x in Degree)
for i in Degree:
    x = df_Analyst['Minimum_Qualifications'].str.contains(i).sum()
    if i in Degrees:
        Degrees[i] = x
        
print(Degrees)

# In[ ]:


Degrees = dict((x,0) for x in Degree)
for i in Degree:
    x = df_Analyst['Preferred_Qualifications'].str.contains(i).sum()
    if i in Degrees:
        Degrees[i] = x
        
print(Degrees)

# **It seems that Google do prefer a further education degree like master or PhD**

# In[ ]:


sns.countplot('Minimum_years_experience',data=df_Analyst)
plt.suptitle('Minimum work experience')

# Most of the positions don't require work experience or didn't mention it. However, we can see some of them require 2-5 years experience.
# 
# ## <a id="2">Developer</a>

# In[ ]:


df_Developer = df.loc[df.Title.str.contains('Developer').fillna(False)]

# In[ ]:


df_Developer.Country.value_counts()

# In[ ]:


MadeWordCloud('Developer','Responsibilities')

# ### Keywords
# 
# * Fileds : Business
# * Skills : Manage/Comunication/Management
# * Experience : Engineers/Sales/Developer/Android/iOS
# 
# I found that the developer positions in Google actually requires some leader's characteristics! It seems that Google don't want an engineer that only know how to code but play a team leader, or even more

# In[ ]:


MadeWordCloud('Developer','Minimum_Qualifications')

# ### Keywords
# 
# * Degrees : BA/BS
# * Languages : Go/Kotlin/Javascipt/Python/Java
# * Criteria : practical experience/degree
# 
# The main point for this plot is mostly about experience. It seems that the most importanat thing to be a developer at Google

# In[ ]:


MadeWordCloud('Developer','Preferred_Qualifications')

# ### Keywords
# 
# * Criteria : Effective/Ability/Knowledge/Experience
# 
# I'll sum up this plot by one sentence :
# **Looking for developer that can work effectively and organizedly, having related experience and knowledge, and understanding industry and stakeholders.**

# ### <a id="2.1">Also, I want to check the languages requirements of developer positions</a>

# In[ ]:


DataSkill = ['Java ','Javascript','Go ','Python','Kotlin','SQL']

DataSkills = dict((x,0) for x in DataSkill)
for i in DataSkill:
    x = df_Developer['Minimum_Qualifications'].str.contains(i).sum()
    if i in DataSkill:
        DataSkills[i] = x
        
print(DataSkills)

# ### Of course, degrees as well

# In[ ]:


Degrees = dict((x,0) for x in Degree)
for i in Degree:
    x = df_Developer['Minimum_Qualifications'].str.contains(i).sum()
    if i in Degrees:
        Degrees[i] = x
        
print(Degrees)

# In[ ]:


Degrees = dict((x,0) for x in Degree)
for i in Degree:
    x = df_Developer['Preferred_Qualifications'].str.contains(i).sum()
    if i in Degrees:
        Degrees[i] = x
        
print(Degrees)

# **Compare to those of Analyst, Google don't actually prefer candidates with a further education degree.**
# **I guess it's because there are many theories, like mathematics, statistics, calculus required for analysts** 

# In[ ]:


sns.countplot('Minimum_years_experience',data=df_Developer)
plt.suptitle('Minimum work experience')

# Though most of the positions didn't mention the required work experience, we can still see 3 years experience might be a good qualification.
# 
# ## <a id="3">MBA Intern</a>
# 
# I'm interested in these MBA Intern positions since it's also happening in my country, Taiwan, too. I wonder what they need to get the position and if they need to know how to code, what languages they need, etc. So, another time when Word Cloud come to be useful.

# In[ ]:


df_MBA = df.loc[df.Title.str.contains('MBA').fillna(False)]

# In[ ]:


df_MBA.head(1)

# In[ ]:


df_MBA.Category.value_counts()

# In[ ]:


df_MBA.Country.value_counts()

# In[ ]:


MadeWordCloud('MBA','Responsibilities')

# In[ ]:


MadeWordCloud('MBA','Minimum_Qualifications')

# In[ ]:


MadeWordCloud('MBA','Preferred_Qualifications')

# OK, so I finally see some criteria here:
# 
# * Soft skills : Management/Organizational
# * Characteristics : Strategic/Independent/Changing environment
# * Good to have : Project/interest of technology

# ### <a id="3.1">How about the languages</a>

# In[ ]:


Languages = dict((x,0) for x in Programming_Languages)
for i in Languages:
    x = df_MBA['Minimum_Qualifications'].str.contains(i).sum()
    if i in Languages:
        Languages[i] = x
        
print(Languages)

# In[ ]:


Languages = dict((x,0) for x in Programming_Languages)
for i in Languages:
    x = df_MBA['Preferred_Qualifications'].str.contains(i).sum()
    if i in Languages:
        Languages[i] = x
        
print(Languages)

# In[ ]:


sns.countplot('Minimum_years_experience',data=df_MBA)
plt.suptitle('Minimum work experience')

# We can see that since it's MBA intern positions, there is no need for work experience
# 
# ## <a id="4">Sales</a>

# In[ ]:


df_Sales = df.loc[df.Title.str.contains('Sales').fillna(False)]

# In[ ]:


df_Sales.Category.value_counts()

# In[ ]:


df_Sales.Country.value_counts()[:5]

# In[ ]:


MadeWordCloud('Sales','Responsibilities')

# There are a lot of words in this plot. However, I can still sum it up:
# 
# * Soft skills : Management/Plan
# * Characteristics : Cross functional/Strategic
# * Good to be familiar with : Product/Google Cloud Platform/Client/Partner/Develop

# In[ ]:


MadeWordCloud('Sales','Minimum_Qualifications')

# There are a lot of words in this plot. However, I can still sum it up:
# 
# * Degree : BA/BS/Bachelor
# * Speaking : Fluentual/Idiomatically/English
# * Characteristics : Experienced/Practical
# * Good to be familiar with : Cloud computing

# In[ ]:


MadeWordCloud('Sales','Preferred_Qualifications')

# There are a lot of words in this plot. However, I can still sum it up:
# 
# * Skills : Project Management
# * Characteristics : Fast Paced/Demonstrated/Cross functional/Effectively/Experienced
# * Good to be familiar with : PaaS/IaaS/Big Data/Google Cloud/Computer Science

# ### <a id="4.1">Now, let's talk about the languages</a>

# In[ ]:


Languages = dict((x,0) for x in Programming_Languages)
for i in Languages:
    x = df_Sales['Minimum_Qualifications'].str.contains(i).sum()
    if i in Languages:
        Languages[i] = x
        
print(Languages)

# In[ ]:


Languages = dict((x,0) for x in Programming_Languages)
for i in Languages:
    x = df_Sales['Preferred_Qualifications'].str.contains(i).sum()
    if i in Languages:
        Languages[i] = x
        
print(Languages)

# As I expected, some of the sales need to know SQL. And since some sales are in techical department, they need to know some other languages.

# In[ ]:


sns.countplot('Minimum_years_experience',data=df_Sales)
plt.suptitle('Minimum work experience')

# Still, most of the positions didn't mention work experience required. However, we can see that there are more senior sales positions in Google that requrie more than 5 years experience.
# 
# ## <a id="5">Microsoft Office</a>
# 
# I just came up with this question. Do Google put Microsoft Office in their requirements? Some people told me that Microsoft Office is so basic that I should not put them in my LinkedIn Skills. However, a lot of company still mention it in their position requirement. Let's see if Google do it or not.

# In[ ]:


Microsoft_Office = ['Excel','Powerpoint','Word','Microsoft']

MO = dict((x,0) for x in Microsoft_Office)
for i in MO:
    x = df['Minimum_Qualifications'].str.contains(i).sum()
    if i in Microsoft_Office:
        MO[i] = x
        
print(MO)

# Cool, so Google generally agree with the idea that Microsoft Office is basic. They only mention Word. I'l say that there is no need to mention it in the requirements.

# ## <a id="6">Data Visualization Tools</a>
# 
# I wonder if Google has a preference about the data visualization. There are some leaders in the field, like Tableau, Power BI, Qlik, and Google Visual Studio. Let me check if these are in the minimum requirements and preference requirements.

# In[ ]:


DV_Tools = ['Tableau','Power BI','Qlik','Data Studio','Google Analytics','GA']

DV = dict((x,0) for x in DV_Tools)
for i in DV:
    x = df['Minimum_Qualifications'].str.contains(i).sum()
    if i in DV_Tools:
        DV[i] = x
        
print(DV)

# It seems that even though Google has Visual Studio made by themselves, Tableau is still taking lead in the field.

# ## <a id="7">Statistical Analysis Tools</a>
# 
# I wonder if Google do prefer any of the statistical analysis tools. In my acknowledge, there are SPSS, R, Matlab, Excel, Google Spreadsheet, and SAS in this field.

# In[ ]:


SA_Tools = ['SPSS','R ','Matlab','Excel','Spreadsheet','SAS']

SA = dict((x,0) for x in SA_Tools)
for i in SA:
    x = df['Minimum_Qualifications'].str.contains(i).sum()
    if i in SA_Tools:
        SA[i] = x
        
print(SA)

# So, there are a variety of preference of statistical analysis tools in Google. However, I think most of them prefer R and SAS instead. Still, I'm surprised to see that they do mention SPSS in the requriement.

# ## <a id="8">Let's see the positions in the US</a>

# In[ ]:


df_US = df.loc[df.Country == ' United States']

# In[ ]:


df_US_Type = df_US.Category.value_counts()
df_US_Type = df_US_Type.rename_axis('Type').reset_index(name='counts')

# In[ ]:


import squarify
import matplotlib
matplotlib.rcParams.update({'font.size': 8})
cmap = matplotlib.cm.Blues
norm = matplotlib.colors.Normalize(vmin=min(df_US_Type.counts), vmax=max(df_US_Type.counts))
colors = [cmap(norm(value)) for value in df_US_Type.counts]
fig = plt.gcf()
ax = fig.add_subplot()
fig.set_size_inches(24, 6)
squarify.plot(sizes=df_US_Type['counts'], label=df_US_Type['Type'], alpha=.8, color=colors)
plt.title('Type of positions',fontsize=20,fontweight="bold")
plt.axis('off')
plt.show()

# ## <a id="9">PMs (Product Manager, Project Manager, and Program Manager</a>
# 
# I wonder how is the distribution of PMs in Google. In my opinion, Project Manager might be the most of them since project management is important in each position category.

# In[ ]:


PM_positions = ['Product Manager','Project Manager','Program Manager']

PM = dict((x,0) for x in PM_positions)
for i in PM:
    x = df['Title'].str.contains(i).sum()
    if i in PM_positions:
        PM[i] = x
        
print(PM)

# Well, different from my expectation, it turns out that most of them are program managers. It does make sense because Google is more like a Software as a Service company in many aspects.   
# However, I still want to see if Jira, scrum, and agile, those project management phrases are mentioned or not.

# In[ ]:


Project_Management_words = ['Jira','scrum','agile']

Project_Management = dict((x,0) for x in Project_Management_words)
for i in Project_Management:
    x = df['Minimum_Qualifications'].str.contains(i).sum()
    if i in Project_Management_words:
        Project_Management[i] = x
        
print(Project_Management)

# In[ ]:


Project_Management = dict((x,0) for x in Project_Management_words)
for i in Project_Management:
    x = df['Preferred_Qualifications'].str.contains(i).sum()
    if i in Project_Management_words:
        Project_Management[i] = x
        
print(Project_Management)

# Well, obvious, I think Agile is important to many roles, and Google doesn't have specific Project Management tools preference.

# ## <a id="10">Pivot tables</a>

# In[ ]:


df_groupby_country_category = df.groupby(['Country','Category'])['Category'].count()

# In[ ]:


df_groupby_country_category.loc[' United States']

# In this way, I can more thoroughly see the distribution of positions in each country.

# In[ ]:


category_country = df.pivot_table(index=['Country','Category'],values='Minimum_years_experience',aggfunc='median')

# In[ ]:


category_country.loc[' United States']

# 

# In[ ]:


category_country.loc['Singapore']

# In[ ]:


category_country.loc[' Taiwan']

# In[ ]:


category_country.loc[' India']

# We can see that the people in Legal dept required more experience than other. And so does the Supply Chain dept in the US.

# **<a id=11>To sum up, there are two parts I want to talk about:</a>**
# 
# ### 1. Application of this EDA
# 
# * With str.contains and re.findall, I can extract some keywords and count appearance through the dataset after I browse it roughly by myself
# * I can easily find some keywords with wordcloud, then dig deeper afterward
# * This can be used on large data of social network posts or articles
# * Maybe sentimental analysis
# 
# ### 2. What I know about Google after the research
# 
# * Basically, you need a bachelor or equivalent degree to get in Google, master and MBA can earn you a better position among the candidates
# * Java and Python are three most important languages in Google, while SQL is also important to analysts and sales
# * Knowing business and having good communication, management skills are great characteristic to get a job in Google
# * Project management and agile methodologies might be preferred
# * Tableau takes the lead of data visualization tools
# * The data center in the states requires more experience than other department
# 
# # For more text preprocessing and NLP model, please check out [my this kernel](https://www.kaggle.com/justjun0321/are-voice-assistants-really-improving-our-lives)

# In[ ]:



