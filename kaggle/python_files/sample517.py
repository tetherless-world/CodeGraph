#!/usr/bin/env python
# coding: utf-8

# # Home Credit Default Risk - Exploration + Baseline Model
# 
# Many people struggle to get loans due to insufficient or non-existent credit histories. And, unfortunately, this population is often taken advantage of by untrustworthy lenders. Home Credit strives to broaden financial inclusion for the unbanked population by providing a positive and safe borrowing experience. In order to make sure this underserved population has a positive loan experience, Home Credit makes use of a variety of alternative data--including telco and transactional information--to predict their clients' repayment abilities.
# 
# While Home Credit is currently using various statistical and machine learning methods to make these predictions, they're challenging Kagglers to help them unlock the full potential of their data. Doing so will ensure that clients capable of repayment are not rejected and that loans are given with a principal, maturity, and repayment calendar that will empower their clients to be successful.
# 
# This is a simple notebook on exploration and baseline model of home credit default risk data 
# 
# **Contents**   
# [1. Dataset Preparation](#1)    
# [2. Exploration - Applications Train](#2)  
# &nbsp;&nbsp;&nbsp;&nbsp; [2.1 Snapshot - Application Train](#2.1)    
# &nbsp;&nbsp;&nbsp;&nbsp; [2.2 Distribution of Target Variable](#2.2)    
# &nbsp;&nbsp;&nbsp;&nbsp; [2.3 Applicant's Gender Type](#2.3)    
# &nbsp;&nbsp;&nbsp;&nbsp; [2.4 Family Status of Applicants who takes the loan](#2.4)  
# &nbsp;&nbsp;&nbsp;&nbsp; [2.5 Does applicants own Real Estate or Car](#2.5)    
# &nbsp;&nbsp;&nbsp;&nbsp; [2.6 Suite Type and Income Type of Applicants](#2.6)   
# &nbsp;&nbsp;&nbsp;&nbsp; [2.7 Applicants Contract Type](#2.7)   
# &nbsp;&nbsp;&nbsp;&nbsp; [2.8 Education Type and Occupation Type](#2.8)   
# &nbsp;&nbsp;&nbsp;&nbsp; [2.9 Organization Type and Occupation Type](#2.9)   
# &nbsp;&nbsp;&nbsp;&nbsp; [2.10 Walls Material, Foundation and House Type](#2.10)   
# &nbsp;&nbsp;&nbsp;&nbsp; [2.11 Amount Credit Distribution](#2.11)    
# &nbsp;&nbsp;&nbsp;&nbsp; [2.12 Amount Annuity Distribution - Distribution](#2.12)  
# &nbsp;&nbsp;&nbsp;&nbsp; [2.13 Amount Goods Price - Distribution](#2.13)   
# &nbsp;&nbsp;&nbsp;&nbsp; [2.14 Amount Region Population Relative](#2.14)    
# &nbsp;&nbsp;&nbsp;&nbsp; [2.15 Days Birth - Distribution](#2.15)   
# &nbsp;&nbsp;&nbsp;&nbsp; [2.16 Days Employed - Distribution](#2.16)    
# &nbsp;&nbsp;&nbsp;&nbsp; [2.17 Distribution of Num Days Registration](#2.17)  
# &nbsp;&nbsp;&nbsp;&nbsp; [2.18 Applicants Number of Family Members](#2.18)  
# &nbsp;&nbsp;&nbsp;&nbsp; [2.19 Applicants Number of Children](#2.19)  
# [3. Exploration - Bureau Data](#3)  
# &nbsp;&nbsp;&nbsp;&nbsp; [3.1 Snapshot - Bureau Data](#3)    
# [4. Exploration - Bureau Balance Data](#4)  
# &nbsp;&nbsp;&nbsp;&nbsp; [4.1 Snapshot - Bureau Balance Data](#3)     
# [5. Exploration - Credit Card Balance Data](#5)   
# &nbsp;&nbsp;&nbsp;&nbsp; [5.1 Snapshot - Credit Card Balance Data](#3)   
# [6. Exploration - POS Cash Balance Data](#6)   
# &nbsp;&nbsp;&nbsp;&nbsp; [6.1 Snapshot - POS Cash Balance Data](#3)   
# [7. Exploration - Previous Application Data](#7)   
# &nbsp;&nbsp;&nbsp;&nbsp; [7.1 Snapshot - Previous Application Data](#7.1)  
# &nbsp;&nbsp;&nbsp;&nbsp; [7.2 Contract Status Distribution - Previous Applications](#7.2)  
# &nbsp;&nbsp;&nbsp;&nbsp; [7.3 Suite Type Distribution - Previous Application](#7.3)    
# &nbsp;&nbsp;&nbsp;&nbsp; [7.4 Client Type Distribution  - Previous Application](#7.4)    
# &nbsp;&nbsp;&nbsp;&nbsp; [7.5 Channel Type Distribution - Previous Applications](#7.5)  
# [8. Exploration - Installation Payments](#8)  
# &nbsp;&nbsp;&nbsp;&nbsp; [8.1 Snapshot of Installation Payments](#3)  
# [9. Baseline Model](#9)  
# &nbsp;&nbsp;&nbsp;&nbsp; [9.1 Dataset Preparation](#9.1)  
# &nbsp;&nbsp;&nbsp;&nbsp; [9.2 Handelling Categorical Features](#9.2)     
# &nbsp;&nbsp;&nbsp;&nbsp; [9.3 Create Flat Dataset](#9.3)     
# &nbsp;&nbsp;&nbsp;&nbsp; [9.4 Validation Sets Preparation](#9.4)    
# &nbsp;&nbsp;&nbsp;&nbsp; [9.5 Model Fitting](#9.5)    
# &nbsp;&nbsp;&nbsp;&nbsp; [9.6 Feature Importance](#9.6)    
# &nbsp;&nbsp;&nbsp;&nbsp; [9.7 Prediction](#9.7)   
# 
# 
# 
# ## <a id="1">1. Dataset Preparation </a>

# In[ ]:


from plotly.offline import init_notebook_mode, iplot
from wordcloud import WordCloud
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import plotly.plotly as py
from plotly import tools
from datetime import date
import pandas as pd
import numpy as np 
import seaborn as sns
import random 
import warnings
warnings.filterwarnings("ignore")
init_notebook_mode(connected=True)

path = "../input/"

def bar_hor(df, col, title, color, w=None, h=None, lm=0, limit=100, return_trace=False, rev=False, xlb = False):
    cnt_srs = df[col].value_counts()
    yy = cnt_srs.head(limit).index[::-1] 
    xx = cnt_srs.head(limit).values[::-1] 
    if rev:
        yy = cnt_srs.tail(limit).index[::-1] 
        xx = cnt_srs.tail(limit).values[::-1] 
    if xlb:
        trace = go.Bar(y=xlb, x=xx, orientation = 'h', marker=dict(color=color))
    else:
        trace = go.Bar(y=yy, x=xx, orientation = 'h', marker=dict(color=color))
    if return_trace:
        return trace 
    layout = dict(title=title, margin=dict(l=lm), width=w, height=h)
    data = [trace]
    fig = go.Figure(data=data, layout=layout)
    iplot(fig)

def bar_hor_noagg(x, y, title, color, w=None, h=None, lm=0, limit=100, rt=False):
    trace = go.Bar(y=x, x=y, orientation = 'h', marker=dict(color=color))
    if rt:
        return trace
    layout = dict(title=title, margin=dict(l=lm), width=w, height=h)
    data = [trace]
    fig = go.Figure(data=data, layout=layout)
    iplot(fig)


def bar_ver_noagg(x, y, title, color, w=None, h=None, lm=0, rt = False):
    trace = go.Bar(y=y, x=x, marker=dict(color=color))
    if rt:
        return trace
    layout = dict(title=title, margin=dict(l=lm), width=w, height=h)
    data = [trace]
    fig = go.Figure(data=data, layout=layout)
    iplot(fig)
    
def gp(col, title):
    df1 = app_train[app_train["TARGET"] == 1]
    df0 = app_train[app_train["TARGET"] == 0]
    a1 = df1[col].value_counts()
    b1 = df0[col].value_counts()
    
    total = dict(app_train[col].value_counts())
    x0 = a1.index
    x1 = b1.index
    
    y0 = [float(x)*100 / total[x0[i]] for i,x in enumerate(a1.values)]
    y1 = [float(x)*100 / total[x1[i]] for i,x in enumerate(b1.values)]

    trace1 = go.Bar(x=a1.index, y=y0, name='Target : 1', marker=dict(color="#96D38C"))
    trace2 = go.Bar(x=b1.index, y=y1, name='Target : 0', marker=dict(color="#FEBFB3"))
    return trace1, trace2 

# ## <a href="2">2. Exploration of Applications Data </a>
# 
# ### <a href="2.1">2.1 Snapshot of Application Train</a>
# 
# Application data consists of static data for all applications and every row represents one loan.

# In[ ]:


app_train = pd.read_csv(path + "application_train.csv")
app_train.head()

# > There are total 307,511 rows which contains the information of loans and there are 122 variables. 
# 
# > The target variable defines if the client had payment difficulties meaning he/she had late payment more than X days on at least one of the first Y installments of the loan. Such case is marked as 1 while other all other cases as 0.
# 
# ### <a href="2.2">2.2 Distribution of Target Variable </a>
# 
# The target variable defines weather the loan was repayed or not. Let us look at what is the distribution of loan repayment in the training dataset. 

# In[ ]:


# Target Variable Distribution 
bar_hor(app_train, "TARGET", "Distribution of Target Variable" , ["#44ff54", '#ff4444'], h=350, w=600, lm=200, xlb = ['Target : 1','Target : 0'])

# > The target variable is slightly imbalance with the majority of loans has the target equals to 0 which indicates that individuals did not had any problems in paying installments in given time. There are about 91% loans which is equal to about 282K with target = 0, While only 9% of the total loans (about 24K applicants) in this dataset involved the applicants having problems in repaying the loan / making installments.  
# 
# ### <a href="2.3">2.3 Gender Type of Applicants </a>

# In[ ]:


tr0 = bar_hor(app_train, "CODE_GENDER", "Distribution of CODE_GENDER Variable" ,"#f975ae", w=700, lm=100, return_trace= True)
tr1, tr2 = gp('CODE_GENDER', 'Distribution of Target with Applicant Gender')

fig = tools.make_subplots(rows=1, cols=3, print_grid=False, subplot_titles = ["Gender Distribution" , "Gender, Target=1" ,"Gender, Target=0"])
fig.append_trace(tr0, 1, 1);
fig.append_trace(tr1, 1, 2);
fig.append_trace(tr2, 1, 3);
fig['layout'].update(height=350, showlegend=False, margin=dict(l=50));
iplot(fig);

# > In the applicant's data women have applied for a larger majority of loans which is almost the double as the men. In total, there are about 202,448 loan applications filed by females in contrast to about 105,059 applications filed by males. However, a larger percentage (about 10% of the total) of men had the problems in paying the loan or making installments within time as compared to women applicants (about 7%). 
# 
# ### <a href="2.4">2.4 Family Status of Applicants </a>

# In[ ]:


tr0 = bar_hor(app_train, "NAME_FAMILY_STATUS", "Distribution of CODE_GENDER Variable" ,"#f975ae", w=700, lm=100, return_trace= True)
tr1, tr2 = gp('NAME_FAMILY_STATUS', 'Distribution of Target with Applicant Gender')

fig = tools.make_subplots(rows=1, cols=3, print_grid=False, subplot_titles = ["Family Status Distribution" , "Family Status, Target = 1" ,"Family Status, Target = 0"])
fig.append_trace(tr0, 1, 1);
fig.append_trace(tr1, 1, 2);
fig.append_trace(tr2, 1, 3);
fig['layout'].update(height=350, showlegend=False, margin=dict(l=120));
iplot(fig);

# > Married people have applied for a larger number of loan applications about 196K, However, people having Civil Marriage has the highest percentage (about 10%) of loan problems and challenges. 
# 
# ### <a href="2.5">2.5. Does applicants own Real Estate or Car ?</a>

# In[ ]:


## real estate 
t = app_train['FLAG_OWN_REALTY'].value_counts()
labels = t.index
values = t.values
colors = ['#96D38C','#FEBFB3']
trace = go.Pie(labels=labels, values=values,
               hoverinfo='', textinfo='',
               textfont=dict(size=12),
               marker=dict(colors=colors,
                           line=dict(color='#fff', width=2)))
layout = go.Layout(title='Applicants Owning Real Estate', height=400)
fig = go.Figure(data=[trace], layout=layout)
iplot(fig)


t = app_train['FLAG_OWN_CAR'].value_counts()
labels = t.index
values = t.values
colors = ['#FEBFB3','#96D38C']
trace = go.Pie(labels=labels, values=values,
               hoverinfo='', textinfo='',
               textfont=dict(size=12),
               marker=dict(colors=colors,
                           line=dict(color='#fff', width=2)))
layout = go.Layout(title='Applicants Owning Car', height=400)
fig = go.Figure(data=[trace], layout=layout)
iplot(fig)


tr1, tr2 = gp('FLAG_OWN_REALTY', 'Applicants Owning Real Estate wrt Target Variable')
tr3, tr4 = gp('FLAG_OWN_CAR', 'Applicants Owning Car wrt Target Variable')
fig = tools.make_subplots(rows=1, cols=2, print_grid=False, 
                          subplot_titles = ["% Applicants with RealEstate and Target = 1", "% Applicants with Car and Target = 1"])
fig.append_trace(tr1, 1, 1);
fig.append_trace(tr3, 1, 2);
fig['layout'].update(height=350, showlegend=False, margin=dict(l=120));
iplot(fig);

# > About 70% of the applicants own Real Estate, while only 34% of applicants own Car who had applied for the loan in the past years. However, a higher percentage of people having payment difficulties was observed with applicants which did not owned Car or which did not owned Real Estate. 
# 
# ### <a href="2.6">2.6 Suite Type and Income Type of Applicants </a>

# In[ ]:


tr0 = bar_hor(app_train, "NAME_TYPE_SUITE", "Distribution of CODE_GENDER Variable" ,"#f975ae", w=700, lm=100, return_trace= True)
tr1 = bar_hor(app_train, "NAME_INCOME_TYPE", "Distribution of CODE_GENDER Variable" ,"#f975ae", w=700, lm=100, return_trace= True)

fig = tools.make_subplots(rows=1, cols=2, print_grid=False, subplot_titles = ['Applicants Suite Type' , 'Applicants Income Type'])
fig.append_trace(tr0, 1, 1);
fig.append_trace(tr1, 1, 2);
fig['layout'].update(height=400, showlegend=False, margin=dict(l=100));
iplot(fig);

# > Top 3 Type Suites which applies for loan are the houses which are:  
#     - Unaccompanined (about 248K applicants) 
#     - Family (about 40K applicants)  
#     - Spouse, partner (about 11K applicants)    
# > The income type of people who applies for loan include about 8 categroes, top ones are : 
#     - Working Class (158K)
#     - Commercial Associate (71K)
#     - Pensiner (55K)
# 
# ### <a id="2.6.1">2.6.1 How does Target Varies with Suite and Income Type of Applicants </a>

# In[ ]:


tr1, tr2 = gp('NAME_TYPE_SUITE', 'Applicants Type Suites which repayed the loan')
fig = tools.make_subplots(rows=1, cols=2, print_grid=False, 
                          subplot_titles = ["Applicants Type Suites distribution when Target = 1", "Applicants Type Suites distribution when Target = 0"])
fig.append_trace(tr1, 1, 1);
fig.append_trace(tr2, 1, 2);
fig['layout'].update(height=350, showlegend=False, margin=dict(l=120));
iplot(fig);


tr1, tr2 = gp('NAME_INCOME_TYPE', 'Applicants Income Types which repayed the loan')
fig = tools.make_subplots(rows=1, cols=2, print_grid=False, 
                          subplot_titles = ["Applicants Income Types when Target = 1", "Applicants Income Type When Target = 0"])
fig.append_trace(tr1, 1, 1);
fig.append_trace(tr2, 1, 2);
fig['layout'].update(height=350, showlegend=False, margin=dict(l=120));
iplot(fig);

# > We see that Applicants having Income Types : Maternity Leaves and UnEmployed has the highest percentage (about 40% and 36% approx) of Target = 1 ie. having more payment problems, while Pensioners have the least (about 5.3%). 
# 
# ### <a id="2.7">2.7. Applicant's Contract Type</a>

# In[ ]:


t = app_train['NAME_CONTRACT_TYPE'].value_counts()
labels = t.index
values = t.values
colors = ['#FEBFB3','#96D38C']
trace = go.Pie(labels=labels, values=values,
               hoverinfo='', textinfo='',
               textfont=dict(size=12),
               marker=dict(colors=colors,
                           line=dict(color='#fff', width=2)))
layout = go.Layout(title='Applicants Contract Type', height=400)
fig = go.Figure(data=[trace], layout=layout)
iplot(fig)

# > Cash loans with about 278K loans contributes to a majorty of total lonas in this dataset. Revolving loans has significantly lesser number equal to about 29K as compared to Cash loans. 
# 
# ### <a id="2.8">2.8 Education Type and Housing Type </a>

# In[ ]:


tr1 = bar_hor(app_train, "NAME_EDUCATION_TYPE", "Distribution of " ,"#f975ae", w=700, lm=100, return_trace= True)
tr2 = bar_hor(app_train, "NAME_HOUSING_TYPE", "Distribution of " ,"#f975ae", w=700, lm=100, return_trace = True)

fig = tools.make_subplots(rows=1, cols=2, print_grid=False, subplot_titles = ['Applicants Education Type', 'Applicants Housing Type' ])
fig.append_trace(tr1, 1, 1);
fig.append_trace(tr2, 1, 2);
fig['layout'].update(height=400, showlegend=False, margin=dict(l=100));
iplot(fig);


tr1, tr2 = gp('NAME_EDUCATION_TYPE', 'Applicants Income Types which repayed the loan')
tr3, tr4 = gp('NAME_HOUSING_TYPE', 'Applicants Income Types which repayed the loan')
fig = tools.make_subplots(rows=1, cols=2, print_grid=False, 
                          subplot_titles = ["Applicants Education Types, Target=1", "Applicants Housing Type, Target=1"])
fig.append_trace(tr1, 1, 1);
fig.append_trace(tr3, 1, 2);
fig['layout'].update(height=350, showlegend=False, margin=dict(l=30));
iplot(fig);

# > A large number of applications (218K) are filed by people having secondary education followed by people with Higher Education with 75K applications. Applicants living in House / apartments has the highest number of loan apllications equal to 272K. While we see that the applicants with Lower Secondary education status has the highest percentage of payment related problems. Also, Applicants living in apartments or living with parents also shows the same trend. 

# ### <a id="2.9">2.9. Which Organization and Occupation Type applies for loan and which repays </a>

# In[ ]:


tr1 = bar_hor(app_train, "ORGANIZATION_TYPE", "Distribution of " ,"#f975ae", w=700, lm=100, return_trace= True)
tr2 = bar_hor(app_train, "OCCUPATION_TYPE", "Distribution of " ,"#f975ae", w=700, lm=100, return_trace = True)
fig = tools.make_subplots(rows=1, cols=2, print_grid=False, subplot_titles = ['Applicants Organization Type', 'Applicants Occupation Type' ])
fig.append_trace(tr1, 1, 1);
fig.append_trace(tr2, 1, 2);
fig['layout'].update(height=600, showlegend=False, margin=dict(l=150));
iplot(fig);

# > Top Applicant's who applied for loan : Laborers - Approx 55 K, Sales Staff - Approx 32 K, Core staff - Approx 28 K. Entity Type 3 type organizations have filed maximum number of loans equal to approx 67K
# 
# ### <a id="2.9.1">2.9.1 Target Variable with respect to Organization and Occupation Type </a>

# In[ ]:


tr1, tr2 = gp('ORGANIZATION_TYPE', 'Applicants Income Types which repayed the loan')
tr3, tr4 = gp('OCCUPATION_TYPE', 'Applicants Income Types which repayed the loan')
fig = tools.make_subplots(rows=1, cols=2, print_grid=False, 
                          subplot_titles = ["Applicants Organization Types - Repayed", "Applicants Occupation Type - Repayed"])
fig.append_trace(tr1, 1, 1);
fig.append_trace(tr3, 1, 2);
fig['layout'].update(height=350, showlegend=False, margin=dict(l=120));
iplot(fig);

# ### <a id="2.10">2.10 Walls Material, Foundation, and House Type </a>

# In[ ]:


tr1 = bar_hor(app_train, "FONDKAPREMONT_MODE", "Distribution of FLAG_OWN_REALTY" ,"#639af2", w=700, lm=100, return_trace= True)
tr2 = bar_hor(app_train, "WALLSMATERIAL_MODE", "Distribution of FLAG_OWN_CAR" ,"#a4c5f9", w=700, lm=100, return_trace = True)
tr1 = bar_hor(app_train, "HOUSETYPE_MODE", "Distribution of FLAG_OWN_CAR" ,"#a4c5f9", w=700, lm=100, return_trace = True)

fig = tools.make_subplots(rows=1, cols=2, print_grid=False, subplot_titles = [ 'House Type', 'Walls Material'])
fig.append_trace(tr1, 1, 1);
fig.append_trace(tr2, 1, 2);
# fig.append_trace(tr3, 1, 3);

fig['layout'].update(height=400, showlegend=False, margin=dict(l=100));
iplot(fig);

# > - "Blocks and Flats" related house types have filed the largest number of loan applications equal to about 150K, rest of the other categories : Specific Housing and Terraced house have less than 1500 applications. Similarly houses having Panel and Stone Brick type walls material have filed the largest applciations close to 120K combined. 
# 
# ### <a id="2.10.1">2.10.1 Target Variable with respect to Walls Material, Fondkappremont, House Type </a>

# In[ ]:


tr1, tr2 = gp('HOUSETYPE_MODE', 'Applicants Income Types which repayed the loan')
tr3, tr4 = gp('WALLSMATERIAL_MODE', 'Applicants Income Types which repayed the loan')
fig = tools.make_subplots(rows=1, cols=2, print_grid=False, subplot_titles = ["HouseTypes - Repayed", "WallsMaterial - Repayed"])
fig.append_trace(tr1, 1, 1);
fig.append_trace(tr3, 1, 2);
fig['layout'].update(height=350, showlegend=False, margin=dict(l=120));
iplot(fig);

# ### <a id="2.11">2.11. Distribution of Amount Credit </a>

# In[ ]:


plt.figure(figsize=(12,5))
plt.title("Distribution of AMT_CREDIT")
ax = sns.distplot(app_train["AMT_CREDIT"])

# ### <a id="2.12">2.12 Distribution of Amount AMT_ANNUITY </a>

# In[ ]:


plt.figure(figsize=(12,5))
plt.title("Distribution of AMT_ANNUITY")
ax = sns.distplot(app_train["AMT_ANNUITY"].dropna())

# ### <a id="2.13">2.13 Distribution of Amount AMT_GOODS_PRICE </a>

# In[ ]:


plt.figure(figsize=(12,5))
plt.title("Distribution of AMT_GOODS_PRICE")
ax = sns.distplot(app_train["AMT_GOODS_PRICE"].dropna())

# ### <a id="2.14">2.14 Distribution of Amount REGION_POPULATION_RELATIVE </a>

# In[ ]:


plt.figure(figsize=(12,5))
plt.title("Distribution of REGION_POPULATION_RELATIVE")
ax = sns.distplot(app_train["REGION_POPULATION_RELATIVE"])

# ### <a id="2.15">2.15 Distribution of Amount DAYS_BIRTH </a>

# In[ ]:


plt.figure(figsize=(12,5))
plt.title("Distribution of DAYS_BIRTH")
ax = sns.distplot(app_train["DAYS_BIRTH"])

# ### <a id="2.16">2.16 Distribution of Amount DAYS_EMPLOYED </a>

# In[ ]:


plt.figure(figsize=(12,5))
plt.title("Distribution of DAYS_EMPLOYED")
ax = sns.distplot(app_train["DAYS_EMPLOYED"])

# ### <a id="2.17">2.17 Distribution of Number of Days for Registration</a>

# In[ ]:


plt.figure(figsize=(12,5))
plt.title("Distribution of DAYS_REGISTRATION")
ax = sns.distplot(app_train["DAYS_REGISTRATION"])

# ### <a id="2.18">2.18 How many Family Members does the applicants has </a>

# In[ ]:


t = app_train["CNT_FAM_MEMBERS"].value_counts()
t1 = pd.DataFrame()
t1['x'] = t.index 
t1['y'] = t.values 

plt.figure(figsize=(12,5));
plt.title("Distribution of Applicant's Family Members Count");
ax = sns.barplot(data=t1, x="x", y="y", color="#f975ae");
ax.spines['right'].set_visible(False);
ax.spines['top'].set_visible(False);

ax.set_ylabel('');    
ax.set_xlabel('');

# > Most of the applicants who applied for loan had 2 family members in total
# 
# ### <a id="2.19"> 2.19 How many Children does the applicants have </a>

# In[ ]:


t = app_train["CNT_CHILDREN"].value_counts()
t1 = pd.DataFrame()
t1['x'] = t.index 
t1['y'] = t.values 

plt.figure(figsize=(12,5));
plt.title("Distribution of Applicant's Number of Children");
ax = sns.barplot(data=t1, x="x", y="y", color="#f975ae");
ax.spines['right'].set_visible(False);
ax.spines['top'].set_visible(False);

ax.set_ylabel('');    
ax.set_xlabel('');

# > A large majority of applicants did not had children when they applied for loan
# 
# ## <a id="3">3. Exploration of Bureau Data</a>
# 
# All client's previous credits provided by other financial institutions that were reported to Credit Bureau (for clients who have a loan in our sample). For every loan in our sample, there are as many rows as number of credits the client had in Credit Bureau before the application date.
# 
# ### <a id="3.1">3.1 Snapshot of Bureau Data</a>

# In[ ]:


bureau = pd.read_csv(path + "bureau.csv")
bureau.head()

# ## <a id="4">4. Exploration of Bureau Balance Data</a>
# 
# Monthly balances of previous credits in Credit Bureau. This table has one row for each month of history of every previous credit reported to Credit Bureau – i.e the table has (#loans in sample * # of relative previous credits * # of months where we have some history observable for the previous credits) rows.
# 
# ### <a id="4.1">4.1 Snapshot of Bureau Balance Data</a>

# In[ ]:


bureau_balance = pd.read_csv(path + "bureau_balance.csv")
bureau_balance.head()

# ## <a id="5">5. Exploration of Credit Card Balance</a>
# 
# Monthly balance snapshots of previous credit cards that the applicant has with Home Credit.This table has one row for each month of history of every previous credit in Home Credit (consumer credit and cash loans) related to loans in our sample – i.e. the table has (#loans in sample * # of relative previous credit cards * # of months where we have some history observable for the previous credit card) rows.
# 
# ### <a id="5.1">5.1 Snapshot of Credit Card Balance</a>

# In[ ]:


credit_card_balance = pd.read_csv(path + "credit_card_balance.csv")
credit_card_balance.head()

# ## <a id="6">6. Exploration of POS CASH Balance Data</a>
# 
# Monthly balance snapshots of previous POS (point of sales) and cash loans that the applicant had with Home Credit. This table has one row for each month of history of every previous credit in Home Credit (consumer credit and cash loans) related to loans in our sample – i.e. the table has (#loans in sample * # of relative previous credits * # of months in which we have some history observable for the previous credits) rows.
# 
# ### <a id="6.1">6.1 Snapshot of POS CASH Balance Data</a>

# In[ ]:


pcb = pd.read_csv(path + "POS_CASH_balance.csv")
pcb.head()

# ## <a id="7">7. Exploration of Prev Application</a>
# 
# ### <a id="7.1">7.1 Snapshot of Prev Application</a>

# In[ ]:


previous_application = pd.read_csv(path + "previous_application.csv")
previous_application.head()

# ### <a id="7.2">7.2 Contract Status Distribution in Previously Filed Applications</a>

# In[ ]:


t = previous_application['NAME_CONTRACT_STATUS'].value_counts()
labels = t.index
values = t.values

colors = ['#FEBFB3', '#E1396C', '#96D38C', '#D0F9B1']

trace = go.Pie(labels=labels, values=values,
               hoverinfo='', textinfo='',
               textfont=dict(size=12),
               marker=dict(colors=colors,
                           line=dict(color='#fff', width=2)))

layout = go.Layout(title='Name Contract Status in Previous Applications', height=400)
fig = go.Figure(data=[trace], layout=layout)
iplot(fig)

# > - A large number of people (about 62%) had their previous applications approved, while about 19% of them had cancelled and other 17% were resued. 
# 
# ### <a id="7.3">7.3 Suite Type Distribution of Previous Applications</a>

# In[ ]:


t = previous_application['NAME_TYPE_SUITE'].value_counts()
labels = t.index
values = t.values

colors = ['#FEBFB3', '#E1396C', '#96D38C', '#D0F9B1']

trace = go.Pie(labels=labels, values=values,
               hoverinfo='', textinfo='',
               textfont=dict(size=12),
               marker=dict(colors=colors,
                           line=dict(color='#fff', width=2)))

layout = go.Layout(title='Suite Type in Previous Application Distribution', height=400)
fig = go.Figure(data=[trace], layout=layout)
iplot(fig)

# > - A majority of applicants had previous applications having Unaccompanied Suite Type (about 60%) followed by Family related suite type (about 25%)
# 
# ### <a id="7.4">7.4 Client Type of Previous Applications</a>

# In[ ]:


t = previous_application['NAME_CLIENT_TYPE'].value_counts()
labels = t.index
values = t.values

colors = ['#FEBFB3', '#E1396C', '#96D38C', '#D0F9B1']

trace = go.Pie(labels=labels, values=values,
               hoverinfo='', textinfo='',
               textfont=dict(size=12),
               marker=dict(colors=colors,
                           line=dict(color='#fff', width=2)))

layout = go.Layout(title='Client Type in Previous Applications', height=400)
fig = go.Figure(data=[trace], layout=layout)
iplot(fig)

# > - About 74% of the previous applications were Repeater Clients, while only 18% are new. About 8% are refreshed. 
# 
# ### <a id="7.5">7.5 Channel Type - Previous Applications </a>

# In[ ]:


t = previous_application['CHANNEL_TYPE'].value_counts()
labels = t.index
values = t.values

colors = ['#FEBFB3', '#E1396C', '#96D38C', '#D0F9B1']

trace = go.Pie(labels=labels, values=values,
               hoverinfo='', textinfo='',
               textfont=dict(size=12),
               marker=dict(colors=colors,
                           line=dict(color='#fff', width=2)))

layout = go.Layout(title='Channel Type in Previous Applications', height=400)
fig = go.Figure(data=[trace], layout=layout)
iplot(fig)

# ## <a id="8">8. Exploration of Installation Payments </a>
# ### <a id="8.1">8.1 Snapshot of Installation Payments </a>

# In[ ]:


installments_payments = pd.read_csv(path + "installments_payments.csv")
installments_payments.head()

# ## <a id="9">9. Baseline Model </a>
# 
# ### <a id="9.1">9.1 Dataset Preparation</a>

# In[ ]:


from sklearn.model_selection import train_test_split 
import lightgbm as lgb

# read the test files 
app_test = pd.read_csv('../input/application_test.csv')

app_test['is_test'] = 1 
app_test['is_train'] = 0
app_train['is_test'] = 0
app_train['is_train'] = 1

# target variable
Y = app_train['TARGET']
train_X = app_train.drop(['TARGET'], axis = 1)

# test ID
test_id = app_test['SK_ID_CURR']
test_X = app_test

# merge train and test datasets for preprocessing
data = pd.concat([train_X, test_X], axis=0)

# ### <a id="9.2">9.2 Handelling Categorical Features</a>

# In[ ]:


# function to obtain Categorical Features
def _get_categorical_features(df):
    feats = [col for col in list(df.columns) if df[col].dtype == 'object']
    return feats

# function to factorize categorical features
def _factorize_categoricals(df, cats):
    for col in cats:
        df[col], _ = pd.factorize(df[col])
    return df 

# function to create dummy variables of categorical features
def _get_dummies(df, cats):
    for col in cats:
        df = pd.concat([df, pd.get_dummies(df[col], prefix=col)], axis=1)
    return df 

# get categorical features
data_cats = _get_categorical_features(data)
prev_app_cats = _get_categorical_features(previous_application)
bureau_cats = _get_categorical_features(bureau)
pcb_cats = _get_categorical_features(pcb)
ccbal_cats = _get_categorical_features(credit_card_balance)

# create additional dummy features - 
previous_application = _get_dummies(previous_application, prev_app_cats)
bureau = _get_dummies(bureau, bureau_cats)
pcb = _get_dummies(pcb, pcb_cats)
credit_card_balance = _get_dummies(credit_card_balance, ccbal_cats)

# factorize the categorical features from train and test data
data = _factorize_categoricals(data, data_cats)

# ### <a id="9.3">9.3 Feature Engineering</a>

# ### <a id="9.3.1">9.3.1 Feature Engineering - Previous Applications</a>
# 
# Credits to excellent kernel shared by Olivier for more ideas: https://www.kaggle.com/ogrellier/good-fun-with-ligthgbm 
# 

# In[ ]:


## More Feature Ideas Reference : https://www.kaggle.com/ogrellier/good-fun-with-ligthgbm 

## count the number of previous applications for a given ID
prev_apps_count = previous_application[['SK_ID_CURR', 'SK_ID_PREV']].groupby('SK_ID_CURR').count()
previous_application['SK_ID_PREV'] = previous_application['SK_ID_CURR'].map(prev_apps_count['SK_ID_PREV'])

## Average values for all other features in previous applications
prev_apps_avg = previous_application.groupby('SK_ID_CURR').mean()
prev_apps_avg.columns = ['p_' + col for col in prev_apps_avg.columns]
data = data.merge(right=prev_apps_avg.reset_index(), how='left', on='SK_ID_CURR')

# ### <a id="9.3.2">9.3.2 Feature Engineering - Bureau Data</a>

# In[ ]:


# Average Values for all bureau features 
bureau_avg = bureau.groupby('SK_ID_CURR').mean()
bureau_avg['buro_count'] = bureau[['SK_ID_BUREAU','SK_ID_CURR']].groupby('SK_ID_CURR').count()['SK_ID_BUREAU']
bureau_avg.columns = ['b_' + f_ for f_ in bureau_avg.columns]
data = data.merge(right=bureau_avg.reset_index(), how='left', on='SK_ID_CURR')


# ### <a id="9.3.3">9.3.3 Feature Engineering - Previous Installments</a>

# In[ ]:


## count the number of previous installments
cnt_inst = installments_payments[['SK_ID_CURR', 'SK_ID_PREV']].groupby('SK_ID_CURR').count()
installments_payments['SK_ID_PREV'] = installments_payments['SK_ID_CURR'].map(cnt_inst['SK_ID_PREV'])

## Average values for all other variables in installments payments
avg_inst = installments_payments.groupby('SK_ID_CURR').mean()
avg_inst.columns = ['i_' + f_ for f_ in avg_inst.columns]
data = data.merge(right=avg_inst.reset_index(), how='left', on='SK_ID_CURR')

# ### <a id="9.3.4">9.3.4 Feature Engineering - Pos Cash Balance</a>

# In[ ]:


### count the number of pos cash for a given ID
pcb_count = pcb[['SK_ID_CURR', 'SK_ID_PREV']].groupby('SK_ID_CURR').count()
pcb['SK_ID_PREV'] = pcb['SK_ID_CURR'].map(pcb_count['SK_ID_PREV'])

## Average Values for all other variables in pos cash
pcb_avg = pcb.groupby('SK_ID_CURR').mean()
data = data.merge(right=pcb_avg.reset_index(), how='left', on='SK_ID_CURR')

# ### <a id="9.3.5">9.3.5 Feature Engineering - Credit Card Balance </a>

# In[ ]:


### count the number of previous applications for a given ID
nb_prevs = credit_card_balance[['SK_ID_CURR', 'SK_ID_PREV']].groupby('SK_ID_CURR').count()
credit_card_balance['SK_ID_PREV'] = credit_card_balance['SK_ID_CURR'].map(nb_prevs['SK_ID_PREV'])

### average of all other columns 
avg_cc_bal = credit_card_balance.groupby('SK_ID_CURR').mean()
avg_cc_bal.columns = ['cc_bal_' + f_ for f_ in avg_cc_bal.columns]
data = data.merge(right=avg_cc_bal.reset_index(), how='left', on='SK_ID_CURR')

# ### <a id="9.3.6">9.3.6 Prepare Final Train and Test data</a>

# In[ ]:


#### prepare final Train X and Test X dataframes 
ignore_features = ['SK_ID_CURR', 'is_train', 'is_test']
relevant_features = [col for col in data.columns if col not in ignore_features]
trainX = data[data['is_train'] == 1][relevant_features]
testX = data[data['is_test'] == 1][relevant_features]

# ### <a id="9.4">9.4 Create Validation Sets</a>

# In[ ]:


x_train, x_val, y_train, y_val = train_test_split(trainX, Y, test_size=0.2, random_state=18)
lgb_train = lgb.Dataset(data=x_train, label=y_train)
lgb_eval = lgb.Dataset(data=x_val, label=y_val)

# ### <a id="9.5">9.5 Fit the Model</a>

# In[ ]:


params = {'task': 'train', 'boosting_type': 'gbdt', 'objective': 'binary', 'metric': 'auc', 
          'learning_rate': 0.01, 'num_leaves': 48, 'num_iteration': 5000, 'verbose': 0 ,
          'colsample_bytree':.8, 'subsample':.9, 'max_depth':7, 'reg_alpha':.1, 'reg_lambda':.1, 
          'min_split_gain':.01, 'min_child_weight':1}
model = lgb.train(params, lgb_train, valid_sets=lgb_eval, early_stopping_rounds=150, verbose_eval=200)

# ### <a id="9.6">9.6 Feature Importance </a>

# In[ ]:


lgb.plot_importance(model, figsize=(12, 25), max_num_features=100);

# ### <a id="9.7">9.7 Predict</a>

# In[ ]:


preds = model.predict(testX)
sub_lgb = pd.DataFrame()
sub_lgb['SK_ID_CURR'] = test_id
sub_lgb['TARGET'] = preds
sub_lgb.to_csv("lgb_baseline.csv", index=False)
sub_lgb.head()

# Thanks for viewing. 

# In[ ]:



