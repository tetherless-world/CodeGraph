#!/usr/bin/env python
# coding: utf-8

# <h1><center><font size="6">Home Credit Default Risk Extensive EDA</font></center></h1>
# 
# 
# <img src="https://storage.googleapis.com/kaggle-media/competitions/home-credit/about-us-home-credit.jpg"></img>
# 
# # <a id='0'>Content</a>
# 
# - <a href='#1'>Introduction</a>  
# - <a href='#2'>Load the data</a>   
# - <a href='#3'>Check the data</a>
#     - <a href='#31'>Data model</a>  
#     - <a href='#32'>Glimpse the data</a>  
#     - <a href='#33'>Check missing data</a>  
#     - <a href='#34'>Check data unbalance</a>
# - <a href='#4'>Explore the data</a>
#     - <a href='#41'>Application data</a>
#     - <a href='#42'>Bureau data</a>
#     - <a href='#43'>Previous application data</a> 
#     - <a href='#44'>Contract type</a>     
# - <a href='#5'>References</a>

# # <a id="1">Introduction</a>

# <a href="http://www.homecredit.net/">Home Credit</a> is a non-banking financial institution, founded in 1997 in the Czech Republic.
# 
# The company operates in 14 countries (including United States, Russia, Kazahstan, Belarus, China, India) and focuses on lending primarily to people with little or no credit history which will either not obtain loans or became victims of untrustworthly lenders.
# 
# Home Credit group has over 29 million customers, total assests of 21 billions Euro, over 160 millions loans, with the majority in Asia and and almost half of them in China (as of 19-05-2018). 
# 
# The company uses of a variety of alternative data - including telco and transactional information - to predict their clients' repayment abilities.
# 
# They made available their data to the Kaggle community and are challenging Kagglers to help them unlock the full potential of their data.
# 
# 

# <a href="#0"><font size="1">Go to top</font></a>
# 
# # <a id="2">Load the data</a>

# In[ ]:


import pandas as pd 
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# In[ ]:


IS_LOCAL = False

import os

if(IS_LOCAL):
    PATH="../input/home-credit-default-risk"
else:
    PATH="../input"
print(os.listdir(PATH))


# In[ ]:


application_train = pd.read_csv(PATH+"/application_train.csv")
application_test = pd.read_csv(PATH+"/application_test.csv")
bureau = pd.read_csv(PATH+"/bureau.csv")
bureau_balance = pd.read_csv(PATH+"/bureau_balance.csv")
credit_card_balance = pd.read_csv(PATH+"/credit_card_balance.csv")
installments_payments = pd.read_csv(PATH+"/installments_payments.csv")
previous_application = pd.read_csv(PATH+"/previous_application.csv")
POS_CASH_balance = pd.read_csv(PATH+"/POS_CASH_balance.csv")

# # <a id="3">Check the data</a> 

# ## <a id="31">Data model</a>
# 
# The structure of the data is explained in the following image (from the data description on the competition page)
# 
# <img src="https://storage.googleapis.com/kaggle-media/competitions/home-credit/home_credit.png" width="800"></img>
# 
# The dataframe *application_train* and *application_test* contains the loan and loan applicants. The dataframe *bureau* contains the application data from other loans that the client took from other credit institutions and were reported to the credit bureau. The dataframe *previous_applications* contains information about previous loans at **Home Credit** by the same client, previous loans information and client information at the time of the loan (there is a line in the dataframe per previous loan application).
# 
# **SK_ID_CURR** is connecting the dataframes *application_train*|*test* with *bureau*, *previous_application* and also with dataframes *POS_CASH_balance*, *installments_payment* and *credit_card_balance*. **SK_ID_PREV** connects dataframe *previous_application* with *POS_CASH_balance*, *installments_payment* and *credit_card_balance*. **SK_ID_BUREAU** connects dataframe *bureau* with dataframe *bureau_balance*.

# <a href="#0"><font size="1">Go to top</font></a>
# 
# 
# ## <a id="32">Glimpse the data</a>

# In[ ]:


print("application_train -  rows:",application_train.shape[0]," columns:", application_train.shape[1])
print("application_test -  rows:",application_test.shape[0]," columns:", application_test.shape[1])
print("bureau -  rows:",bureau.shape[0]," columns:", bureau.shape[1])
print("bureau_balance -  rows:",bureau_balance.shape[0]," columns:", bureau_balance.shape[1])
print("credit_card_balance -  rows:",credit_card_balance.shape[0]," columns:", credit_card_balance.shape[1])
print("installments_payments -  rows:",installments_payments.shape[0]," columns:", installments_payments.shape[1])
print("previous_application -  rows:",previous_application.shape[0]," columns:", previous_application.shape[1])
print("POS_CASH_balance -  rows:",POS_CASH_balance.shape[0]," columns:", POS_CASH_balance.shape[1])

# ### application_train

# In[ ]:


application_train.head()

# In[ ]:


application_train.columns.values

# ### application_test

# In[ ]:


application_test.head()

# In[ ]:


application_test.columns.values

# ### bureau

# In[ ]:


bureau.head()

# In[ ]:


bureau.columns.values

# ### bureau_balance

# In[ ]:


bureau_balance.head()

# ### credit_card_balance

# In[ ]:


credit_card_balance.head()

# In[ ]:


credit_card_balance.columns.values

# ### installments_payments

# In[ ]:


installments_payments.head()

# In[ ]:


installments_payments.columns.values

# ### previous_applications

# In[ ]:


previous_application.head()

# In[ ]:


previous_application.columns.values

# ### POS_CASH_balance

# In[ ]:


POS_CASH_balance.head()

# In[ ]:


POS_CASH_balance.columns.values

# <a href="#0"><font size="1">Go to top</font></a>
# 
# ## <a id="33">Check missing data</a>

# In[ ]:


def missing_data(data):
    total = data.isnull().sum().sort_values(ascending = False)
    percent = (data.isnull().sum()/data.isnull().count()*100).sort_values(ascending = False)
    return pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

# ## application_train

# In[ ]:


missing_data(application_train).head(10)


# In[ ]:


### application_test

# In[ ]:


missing_data(application_test).head(10)

# In[ ]:


### bureau

# In[ ]:


missing_data(bureau)

# ### bureau_balance

# In[ ]:


missing_data(bureau_balance)

# ### credit_card_balance

# In[ ]:


missing_data(credit_card_balance)

# ### installments_payments

# In[ ]:


missing_data(installments_payments)

# ### previous_applications

# In[ ]:


missing_data(previous_application).head(20)

# ### POS_CASH_balance

# In[ ]:


missing_data(POS_CASH_balance)

# <a href="#0"><font size="1">Go to top</font></a>
# 
# 
# ## <a id="34">Check data unbalance</a>
# 
# **TARGET** value 0 means loan is repayed, value 1 means loan is not repayed.

# In[ ]:


temp = application_train["TARGET"].value_counts()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
plt.figure(figsize = (6,6))
plt.title('Application loans repayed - train dataset')
sns.set_color_codes("pastel")
sns.barplot(x = 'labels', y="values", data=df)
locs, labels = plt.xticks()
plt.show()


# <a href="#0"><font size="1">Go to top</font></a>
# 
# # <a id="4">Explore the data</a>

# ## <a id="41">Application data</a>
# 
# ### Loan types
# 
# 
#     
# Let's see the type of the loans taken and also, on a separate plot, the percent of the loans (by type of the loan) with **TARGET** value 1 (not returned loan).

# In[ ]:


def plot_stats(feature,label_rotation=False,horizontal_layout=True):
    temp = application_train[feature].value_counts()
    df1 = pd.DataFrame({feature: temp.index,'Number of contracts': temp.values})

    # Calculate the percentage of target=1 per category value
    cat_perc = application_train[[feature, 'TARGET']].groupby([feature],as_index=False).mean()
    cat_perc.sort_values(by='TARGET', ascending=False, inplace=True)
    
    if(horizontal_layout):
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12,6))
    else:
        fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(12,14))
    sns.set_color_codes("pastel")
    s = sns.barplot(ax=ax1, x = feature, y="Number of contracts",data=df1)
    if(label_rotation):
        s.set_xticklabels(s.get_xticklabels(),rotation=90)
    
    s = sns.barplot(ax=ax2, x = feature, y='TARGET', order=cat_perc[feature], data=cat_perc)
    if(label_rotation):
        s.set_xticklabels(s.get_xticklabels(),rotation=90)
    plt.ylabel('Percent of target with value 1 [%]', fontsize=10)
    plt.tick_params(axis='both', which='major', labelsize=10)

    plt.show();

# In[ ]:


def plot_distribution(var):
    
    i = 0
    t1 = application_train.loc[application_train['TARGET'] != 0]
    t0 = application_train.loc[application_train['TARGET'] == 0]

    sns.set_style('whitegrid')
    plt.figure()
    fig, ax = plt.subplots(2,2,figsize=(12,12))

    for feature in var:
        i += 1
        plt.subplot(2,2,i)
        sns.kdeplot(t1[feature], bw=0.5,label="TARGET = 1")
        sns.kdeplot(t0[feature], bw=0.5,label="TARGET = 0")
        plt.ylabel('Density plot', fontsize=12)
        plt.xlabel(feature, fontsize=12)
        locs, labels = plt.xticks()
        plt.tick_params(axis='both', which='major', labelsize=12)
    plt.show();

# In[ ]:


plot_stats('NAME_CONTRACT_TYPE')

# Contract type *Revolving loans* are just a small fraction (10%) from the total number of loans; in the same time, a larger amount of *Revolving loans*, comparing with their frequency, are not repaid.

# ### Client gender
# 
# Let's see the gender of the clients and also, on a separate plot, the percent of the loans (by client gender) with TARGET value 1 (not returned loan).

# In[ ]:


plot_stats('CODE_GENDER')

# The number of female clients is almost double  the number of male clients. Looking to the percent of defaulted credits, males have a higher chance of not returning their loans (~10%), comparing with women (~7%).

# ### Flag own car and flag own real estate
# 
# Let's inspect the  flags that tell us if a client owns a car or real estate and, on separate plots, the percent of the loans value of these flags) with TARGET value 1 (not returned loan).

# In[ ]:


plot_stats('FLAG_OWN_CAR')
plot_stats('FLAG_OWN_REALTY')

# The clients that owns a car are almost a half of the ones that doesn't own one. The clients that owns a car are less likely to not repay a car that the ones that own. Both categories have not-repayment rates around 8%.
# 
# The clients that owns real estate are more than double of the ones that doesn't own. Both categories (owning real estate or not owning) have not-repayment rates less than 8%.
# 

# ### Family status of client

# In[ ]:


plot_stats('NAME_FAMILY_STATUS',True, True)

# Most of clients are married, followed by Single/not married and civil marriage.
# 
# In terms of percentage of not repayment of loan, Civil marriage has the highest percent of not repayment (10%), with Widow the lowest (exception being *Unknown*).

# ### Number of children
# 
# Let's see what is the distribution of the number of children of the clients.

# In[ ]:


plot_stats('CNT_CHILDREN')

# Most of the clients taking a loan have no children. The number of loans associated with the clients with one children are 4 times smaller, the number of loans associated with the clients with two children are 8 times smaller; clients with 3, 4 or more children are much more rare. 
# 
# As for repayment, clients with no children, 1, 2, 3, and 5 children have percents of no repayment around the average (10%). The clients with 4 and 6 children are above average in terms of percent of not paid back loans (over 25% for families with 6 children).
# 
# As for clients with 9 or 11 children, the percent of loans not repaid is 100%.
# 
# 

# ### Number of family members of client

# In[ ]:


plot_stats('CNT_FAM_MEMBERS',True)

# Clients with family members of 2 are most numerous, followed by 1 (single persons), 3 (families with one child) and 4.
# 
# Clients with family size of 11 and 13 have 100% not repayment rate. Other families with 10 or 8 members have percents of not repayment of loans over 30%. Families with 6 or less members have repayment rates close to the 10% average.
# 

# ### Income type of client
# 
# Let's investigate the numbers of clients with different income type. As well, let's see the percent of not returned loans per income type of applicants.

# In[ ]:


plot_stats('NAME_INCOME_TYPE',False,False)

# Most of applicants for loans are income from *Working*, followed by *Commercial associate*, *Pensioner* and *State servant*.
# 
# The applicants with the type of income *Maternity leave* have almost 40% ratio of not returning loans, followed by *Unemployed* (37%). The rest of types of incomes are under the average of 10% for not returning loans.
# 

# ### Ocupation of client
# 
# 

# In[ ]:


plot_stats('OCCUPATION_TYPE',True, False)

# Most of the loans are taken by *Laborers*, followed by *Sales staff*. *IT staff* take the lowest amount of loans.
# 
# The category with highest percent of not repaid loans are *Low-skill Laborers* (above 17%), followed by *Drivers* and *Waiters/barmen staff*, *Security staff*, *Laborers* and *Cooking staff*.

# ### Organization type

# In[ ]:


plot_stats('ORGANIZATION_TYPE',True, False)

# Oraganizations with highest percent of loans not repaid are *Transport: type 3* (16%), *Industry: type 13* (13.5%), *Industry: type 8* (12.5%) and *Restaurant* (less than 12%).

# ### Education type of the client

# In[ ]:


plot_stats('NAME_EDUCATION_TYPE',True)

# Majority of the clients have Secondary / secondary special education, followed by clients with Higher education. Only a very small number having an academic degree.
# 
# The Lower secondary category, although rare, have the largest rate of not returning the loan (11%). The people with Academic degree have less than 2% not-repayment rate.
# 

# ### Type of the housing of client

# In[ ]:


plot_stats('NAME_HOUSING_TYPE',True)

# Over 250,000 applicants for credits registered their housing as House/apartment. Following categories have a very small number of clients (With parents, Municipal appartment).
# 
# From these categories, *Rented apartment* and *With parents* have higher than 10% not-repayment rate.

# ### Total income distribution
# 
# Let's plot the distribution of total income for the clients.

# In[ ]:


# Plot distribution of one feature
def plot_distribution(feature,color):
    plt.figure(figsize=(10,6))
    plt.title("Distribution of %s" % feature)
    sns.distplot(application_train[feature].dropna(),color=color, kde=True,bins=100)
    plt.show()   

# In[ ]:


# Plot distribution of multiple features, with TARGET = 1/0 on the same graph
def plot_distribution_comp(var,nrow=2):
    
    i = 0
    t1 = application_train.loc[application_train['TARGET'] != 0]
    t0 = application_train.loc[application_train['TARGET'] == 0]

    sns.set_style('whitegrid')
    plt.figure()
    fig, ax = plt.subplots(nrow,2,figsize=(12,6*nrow))

    for feature in var:
        i += 1
        plt.subplot(nrow,2,i)
        sns.kdeplot(t1[feature], bw=0.5,label="TARGET = 1")
        sns.kdeplot(t0[feature], bw=0.5,label="TARGET = 0")
        plt.ylabel('Density plot', fontsize=12)
        plt.xlabel(feature, fontsize=12)
        locs, labels = plt.xticks()
        plt.tick_params(axis='both', which='major', labelsize=12)
    plt.show();

# In[ ]:


plot_distribution('AMT_INCOME_TOTAL','green')


# ### Credit distribution
# 
# Let's plot the credit distribution.

# In[ ]:


plot_distribution('AMT_CREDIT','blue')

# ### Annuity distribution
# 
# Let's plot the annuity distribution.

# In[ ]:


plot_distribution('AMT_ANNUITY','tomato')

# ### Goods price
# 
# Let's plot the good price distribution.

# In[ ]:


plot_distribution('AMT_GOODS_PRICE','brown')

# ### Days from birth distribution
# 
# Let's plot the distribution number of days from birth.

# In[ ]:


plot_distribution('DAYS_BIRTH','blue')

# The negative value means that the date of birth is in the past. The age range is between approximative 20 and 68 years.

# ### Days employed distribution
# 
# Let's represent the distribution of number of days employed.

# In[ ]:


plot_distribution('DAYS_EMPLOYED','red')

# The negative values means *Days since employed* and most probably these negative values means *Unemployed*. It is not clear what will be the meaning of the very large numbers at the far end (it is not realistic such a large set of people employed more than 100 years).
# 

# ### Days of registration distribution
# 
# Let's plot the distribution of `DAYS_REGISTRATION`.

# In[ ]:


plot_distribution('DAYS_REGISTRATION','green')

# ### Days ID publish distribution
# 
# Let's plot the distribution of DAYS_ID_PUBLISH.

# In[ ]:


plot_distribution('DAYS_ID_PUBLISH','blue')

# ### Comparison of interval values with TARGET = 1 and TARGET = 0
# 
# Let's compare the distribution of interval values ploted above for values of **TARGET = 1** and **TARGET = 0**

# In[ ]:


var = ['AMT_ANNUITY','AMT_GOODS_PRICE','DAYS_EMPLOYED', 'DAYS_REGISTRATION','DAYS_BIRTH','DAYS_ID_PUBLISH']
plot_distribution_comp(var,nrow=3)

# ### Region registered not live region and not work region
# 
# Let's represent the values of region registered and not live region and region registered and not work region.

# In[ ]:


plot_stats('REG_REGION_NOT_LIVE_REGION')
plot_stats('REG_REGION_NOT_WORK_REGION')

# Very few people are registered in not live or not work region. Generally, the rate of not return is slightly larger for these cases than in the rest (slightly above 8% compared with approx. 8%)

# ### City registered not live city and not work city
# 
# Let's represent the values of City registered not live city and not work city.

# In[ ]:


plot_stats('REG_CITY_NOT_LIVE_CITY')
plot_stats('REG_CITY_NOT_WORK_CITY')

# Generally, much more people register in the city they live or work (a larger number register differently in the working city than living city).
# 
# The ones that register in different city than the working or living city are more frequently not-repaying the loans than the ones that register same city (work 11% or live 12%).

# <a href="#0"><font size="1">Go to top</font></a>
# 
# 
# ## <a id="42">Bureau data</a>
# 
# Bureau data contains all client's previous credits provided by other financial institutions that were reported to Credit Bureau (for clients who have a loan in the sample). For every loan in the sample, there are as many rows as number of credits the client had in Credit Bureau before the application date. **SK_ID_CURR** is the key connecting *application_train*|*test* data with *bureau* data.
# 
# Let's merge *application_train* with *bureau*.

# In[ ]:


application_bureau_train = application_train.merge(bureau, left_on='SK_ID_CURR', right_on='SK_ID_CURR', how='inner')

# In[ ]:


print("The resulting dataframe `application_bureau_train` has ",application_bureau_train.shape[0]," rows and ", 
      application_bureau_train.shape[1]," columns.")

# Let's now analize the *application_bureau_train* data.

# In[ ]:


def plot_b_stats(feature,label_rotation=False,horizontal_layout=True):
    temp = application_bureau_train[feature].value_counts()
    df1 = pd.DataFrame({feature: temp.index,'Number of contracts': temp.values})

    # Calculate the percentage of target=1 per category value
    cat_perc = application_bureau_train[[feature, 'TARGET']].groupby([feature],as_index=False).mean()
    cat_perc.sort_values(by='TARGET', ascending=False, inplace=True)
    
    if(horizontal_layout):
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12,6))
    else:
        fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(12,14))
    sns.set_color_codes("pastel")
    s = sns.barplot(ax=ax1, x = feature, y="Number of contracts",data=df1)
    if(label_rotation):
        s.set_xticklabels(s.get_xticklabels(),rotation=90)
    
    s = sns.barplot(ax=ax2, x = feature, y='TARGET', order=cat_perc[feature], data=cat_perc)
    if(label_rotation):
        s.set_xticklabels(s.get_xticklabels(),rotation=90)
    plt.ylabel('Percent of target with value 1 [%]', fontsize=10)
    plt.tick_params(axis='both', which='major', labelsize=10)

    plt.show();

# ### Credit status
# 
# Let's see the credit status distribution. We show first the number of credits per category (could be *Closed*, *Active*, *Sold* and *Bad debt*).

# In[ ]:


plot_b_stats('CREDIT_ACTIVE')

# Most of the credits registered at the Credit Bureau are in the status *Closed* (~900K). On the second place are the *Active* credits ( a bit under 600K). *Sold* and *Bad debt* are just a few.
# 
# In the same time, as percent having **TARGET = 1** from total number per category, clients with credits registered to the Credit Bureau with *Bad debt* have 20%  default on the currrent applications. 
# 
# 
# Clients with credits *Sold*, *Active* and *Closed* have percent of **TARGET == 1** (default credit) equal or less than 10% (10% being the rate overall). The smallest rate of default credit have the clients with credits registered at the Credit Bureau with *Closed* credits.
# 
# That means the former registered credit history (as registered at Credit Bureau) is a strong predictor for the dafault credit, since the percent of applications defaulting with a history of *Bad debt* is twice as large as for *Sold* or *Active* and almost three times larger as for *Closed*.

# ### Credit currency
# 
# Let's check now the number of credits registered at the Credit Bureau with different currencies. Also, let's check procent of defaulting credits (for current applications) per different currencies of credits credits registered at the Credit Bureau in the past for the same client.

# In[ ]:


plot_b_stats('CREDIT_CURRENCY')

# Credits are mostly in *currency_1*.
# 
# Depending on the currency, the percent of clients defaulting is quite different. Starting with *currency_3*, then *currency_1* and *currency_2*, the percent of clients defaulting is 11%, 8% and 5%. Percent of defaulting applications for clients that have credits registered with *currency_4* is close to 0.

# ### Credit type
# 
# Let's check now the credit types for credits registered at the Credit Bureau.

# In[ ]:


plot_b_stats('CREDIT_TYPE', True, True)

# Majority of historical credits registered at the Credit Bureau are *Consumer credit* and *Credit card*. Smaller number of credits are *Car loan*, *Mortgage* and *Microloan*.
# 
# Looking now to the types of historical credits registered at the Credit Bureau, there are few types with a high percent of current credit defaults, as following:  
# * *Loan for the purchase of equipment* - with over 20% current credits defaults;  
# * *Microloan* - with over 20% current credits defaults;  
# * *Loan for working capital replenishement* - with over 12% current credits defaults.  
# 

# ### Duration of credit (DAYS_CREDIT)
# 
# Let's check the distribution of number of days for the credit (registered at the Credit bureau).
# 

# In[ ]:


def plot_b_distribution(feature,color):
    plt.figure(figsize=(10,6))
    plt.title("Distribution of %s" % feature)
    sns.distplot(application_bureau_train[feature].dropna(),color=color, kde=True,bins=100)
    plt.show() 
    
plot_b_distribution('DAYS_CREDIT','green')

# The credit duration (in days) is ranging between less than 3000 days (with a local sadle around 2000 days) and with a increasing frequence for shorter number of days - and with a peak around 300 days (or less than one year).

# ### Credit overdue (CREDIT_DAY_OVERDUE)
# 
# Let's check the distribution of number of days for the credit overdue (registered at the Credit bureau).

# In[ ]:


plot_b_distribution('CREDIT_DAY_OVERDUE','red')

# Most of the credits have 0 or close to 0 days overdue. The maximum number of credit days overdue is ~3000 days.

# ### Credit sum  (AMT_CREDIT_SUM)

# In[ ]:


plot_b_distribution('AMT_CREDIT_SUM','blue')

# The distribution of the AMT_CREDIT_SUM shows a concentration of the credits for the lower credit sum range.
# 
# Let's remove the outliers so that we can see better the distribution around 0.
# 
# Let's introduce a function to identify and filter the outliers (with a predefined threshold). 
# 
# Then, let's also modify the function to display a distribution, this time by using the function to filter the outliers.

# In[ ]:


# Source: https://stackoverflow.com/questions/11882393/matplotlib-disregard-outliers-when-plotting (see references)

def is_outlier(points, thresh=3.5):
    """
    Returns a boolean array with True if points are outliers and False 
    otherwise.

    Parameters:
    -----------
        points : An numobservations by numdimensions array of observations
        thresh : The modified z-score to use as a threshold. Observations with
            a modified z-score (based on the median absolute deviation) greater
            than this value will be classified as outliers.

    Returns:
    --------
        mask : A numobservations-length boolean array.

    References:
    ----------
        Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
        Handle Outliers", The ASQC Basic References in Quality Control:
        Statistical Techniques, Edward F. Mykytka, Ph.D., Editor. 
    """
    if len(points.shape) == 1:
        points = points[:,None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > thresh

def plot_b_o_distribution(feature,color):
    plt.figure(figsize=(10,6))
    plt.title("Distribution of %s" % feature)
    x = application_bureau_train[feature].dropna()
    filtered = x[~is_outlier(x)]
    sns.distplot(filtered,color=color, kde=True,bins=100)
    plt.show() 

plot_b_o_distribution('AMT_CREDIT_SUM','blue')

# We can observe that the distribution function shows several peaks and the maximum concentration of the values is around 20,000 but we also see several other peaks at higher values.

# ### Credit sum limit (AMT_CREDIT_SUM_LIMIT)

# In[ ]:


plot_b_distribution('AMT_CREDIT_SUM_LIMIT','blue')

# ### Comparison of interval values with TARGET = 1 and TARGET = 0
# 
# Let's compare the distribution of interval values ploted above for values of **TARGET = 1** and **TARGET = 0**

# In[ ]:


# Plot distribution of multiple features, with TARGET = 1/0 on the same graph
def plot_b_distribution_comp(var,nrow=2):
    
    i = 0
    t1 = application_bureau_train.loc[application_bureau_train['TARGET'] != 0]
    t0 = application_bureau_train.loc[application_bureau_train['TARGET'] == 0]

    sns.set_style('whitegrid')
    plt.figure()
    fig, ax = plt.subplots(nrow,2,figsize=(12,6*nrow))

    for feature in var:
        i += 1
        plt.subplot(nrow,2,i)
        sns.kdeplot(t1[feature], bw=0.5,label="TARGET = 1")
        sns.kdeplot(t0[feature], bw=0.5,label="TARGET = 0")
        plt.ylabel('Density plot', fontsize=12)
        plt.xlabel(feature, fontsize=12)
        locs, labels = plt.xticks()
        plt.tick_params(axis='both', which='major', labelsize=12)
    plt.show();

# In[ ]:


var = ['DAYS_CREDIT','CREDIT_DAY_OVERDUE','AMT_CREDIT_SUM','AMT_CREDIT_SUM_LIMIT']

plot_b_distribution_comp(var, nrow=2)

# <a href="#0"><font size="1">Go to top</font></a>
# 
# ## <a id="43">Previous application data</a>
# 
# 
# The dataframe *previous_application* contains information about all previous applications for Home Credit loans of clients who have loans in the sample. There is one row for each previous application related to loans in our data sample. **SK_ID_CURR** is the key connecting *application_train*|*test* data with *previous_application* data.
# 
# Let's merge *application_train* with *previous_application*.

# In[ ]:


application_prev_train = application_train.merge(previous_application, left_on='SK_ID_CURR', right_on='SK_ID_CURR', how='inner')

# In[ ]:


print("The resulting dataframe `application_prev_train` has ",application_prev_train.shape[0]," rows and ", 
      application_prev_train.shape[1]," columns.")

# In[ ]:


def plot_p_stats(feature,label_rotation=False,horizontal_layout=True):
    temp = application_prev_train[feature].value_counts()
    df1 = pd.DataFrame({feature: temp.index,'Number of contracts': temp.values})

    # Calculate the percentage of target=1 per category value
    cat_perc = application_prev_train[[feature, 'TARGET']].groupby([feature],as_index=False).mean()
    cat_perc.sort_values(by='TARGET', ascending=False, inplace=True)
    
    if(horizontal_layout):
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12,6))
    else:
        fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(12,14))
    sns.set_color_codes("pastel")
    s = sns.barplot(ax=ax1, x = feature, y="Number of contracts",data=df1)
    if(label_rotation):
        s.set_xticklabels(s.get_xticklabels(),rotation=90)
    
    s = sns.barplot(ax=ax2, x = feature, y='TARGET', order=cat_perc[feature], data=cat_perc)
    if(label_rotation):
        s.set_xticklabels(s.get_xticklabels(),rotation=90)
    plt.ylabel('Percent of target with value 1 [%]', fontsize=10)
    plt.tick_params(axis='both', which='major', labelsize=10)

    plt.show();

# <a href="#0"><font size="1">Go to top</font></a>
# 
# 
# ## <a id="44">Contract type</a>
# 

# In[ ]:


plot_p_stats('NAME_CONTRACT_TYPE_y')

# There are three types of contract in the previous application data: *Cash loans*, *Consumer loans*, *Revolving loans*. *Cash loans* and *Consumer loans* are almost the same number (~600K) whilst *Revolving loans* are ~150K.
# 
# The percent of defauls loans for clients with previous applications is different for the type of previous applications contracts, decreasing from ~10% for *Revolving loans*, then ~ 9.5% for *Cash loans* and ~8% for *Consumer loans*.

# ### Cash loan purpose
# 
# Let's look to the cash loan purpose, in the case of cash loans.

# In[ ]:


plot_p_stats('NAME_CASH_LOAN_PURPOSE', True, True)

# Besides not identifed/not available categories, *Repairs*, *Other*, *Urgent needs*, *Buying a used car*, *Building a house or an annex* accounts for the largest number of contracts.
# 
# 
# In terms of percent of defaults for current applications in the sample, clients with history of previous applications have largest percents of defaults when in their history are previous applications for cash loans for *Refusal to name the goal* - ~23% (which makes a lot of sense), *Hobby* (20%), *Car repairs* (~18%).

# ### Contract status
# 
# Let's look to the contract status.

# In[ ]:


plot_p_stats('NAME_CONTRACT_STATUS', True, True)

# Most previous applications contract statuses are *Approved* (~850K), *Canceled* and *Refused* (~240K). There are only ~20K in status *Unused offer*.
# 
# In terms of percent of defaults for current applications in the sample, clients with history of previous applications have largest percents of defaults when in their history contract statuses are *Refused* (12%), followed by *Canceled* (9%), *Unused offer* (~8%) and *Approved* (lowest percent of defaults in current applictions, with less than 8%).

# ### Payment type
# 
# Let's check the payment type.

# In[ ]:


plot_p_stats('NAME_PAYMENT_TYPE', True, True)

# Most of the previous applications were paid with *Cash through the bank* (~850K). Payments using *Non-cash from your account* or *Cashless from the account of the employer* are much rare. These three types of payments in previous applications results in allmost the same percent of defaults for current clients (~8% each).

# ### Client type
# 
# Let's check the client type for previous applications.

# In[ ]:


plot_p_stats('NAME_CLIENT_TYPE')

# Most of the previous applications have client type *Repeater* (~1M), just over 200K are *New* and ~100K are *Refreshed*.
# 
# In terms of default percent for current applications of clients with history of previous applications, current clients with previous applications have values of percent of defaults ranging from from 8.5%, 8.25% and 7% corresponding to client types in the past *New*, *Repeater* and *Refreshed*, respectivelly.

# <a href="#0"><font size="1">Go to top</font></a>
# 
# # <a id="5">References</a>
# 
# [1] Home Credit Group, http://www.homecredit.net/  
# [2] Home Credit, Wikipedia page, https://en.wikipedia.org/wiki/Home_Credit   
# [3] Home Credit, Financial report, http://www.homecredit.net/~/media/Files/H/Home-Credit-Group/documents/reports/2016/hcbv-ar-2015.pdf   
# [4] Function to disregard outliers when plotting, using matplotlib, https://stackoverflow.com/questions/11882393/matplotlib-disregard-outliers-when-plotting   
# 
# 
# 
# 
