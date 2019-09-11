#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 
import sys
pd.set_option('display.max_colwidth', -1)
import warnings
warnings.filterwarnings("ignore")

# In[2]:


train = pd.read_csv('../input/application_train.csv')

# In[3]:


bureau = pd.read_csv('../input/bureau.csv')

# # HOW TO INTERPRET BUREAU DATA
# 
# This table talks about the Loan data of each unique customer with all financial institutions other than Home Credit
# For each unique SK_ID_CURR we have multiple SK_ID_BUREAU Id's, each being a unique loan transaction from other financial institutions availed by the same customer and reported to the bureau. 

# # EXAMPLE OF BUREAU TRANSACTIONS 
# 
# - In the example below customer with SK_ID_CURR = 100001 had  7 credit transactions before the current application. 

# In[4]:


bureau[bureau['SK_ID_CURR'] == 100001]

# # UNDERSTANDING OF VARIABLES 

# CREDIT_ACTIVE - Current status of a Loan - Closed/ Active (2 values)
# 
# CREDIT_CURRENCY - Currency in which the transaction was executed -  Currency1, Currency2, Currency3, Currency4 
#                                         ( 4 values)
#                                         
# CREDIT_DAY_OVERDUE -  Number of overdue days 
# 
# CREDIT_TYPE -  Consumer Credit, Credit card, Mortgage, Car loan, Microloan, Loan for working capital replemishment, 
#                              Loan for Business development, Real estate loan, Unkown type of laon, Another type of loan. 
#                              Cash loan, Loan for the purchase of equipment, Mobile operator loan, Interbank credit, 
#                              Loan for purchase of shares ( 15 values )
# 
# DAYS_CREDIT -   Number of days ELAPSED since customer applied for CB credit with respect to current application 
# Interpretation - Are these loans evenly spaced time intervals? Are they concentrated within a same time frame?
# 
# 
# DAYS_CREDIT_ENDDATE - Number of days the customer CREDIT is valid at the time of application 
# CREDIT_DAY_OVERDUE - Number of days the customer CREDIT is past the end date at the time of application
# 
# AMT_CREDIT_SUM -  Total available credit for a customer 
# AMT_CREDIT_SUM_DEBT -  Total amount yet to be repayed
# AMT_CREDIT_SUM_LIMIT -   Current Credit that has been utilized 
# AMT_CREDIT_SUM_OVERDUE - Current credit payment that is overdue 
# CNT_CREDIT_PROLONG - How many times was the Credit date prolonged 
# 
# # NOTE: 
# For a given loan transaction 
#  'AMT_CREDIT_SUM' =  'AMT_CREDIT_SUM_DEBT' +'AMT_CREDIT_SUM_LIMIT'
# 
# 
# 
# AMT_ANNUITY -  Annuity of the Credit Bureau data
# DAYS_CREDIT_UPDATE -  Number of days before current application when last CREDIT UPDATE was received 
# DAYS_ENDDATE_FACT -    Days since CB credit ended at the time of application 
# AMT_CREDIT_MAX_OVERDUE - Maximum Credit amount overdue at the time of application 
# 

# # FEATURE ENGINEERING WITH BUREAU CREDIT 

# # FEATURE 1 - NUMBER OF PAST LOANS PER CUSTOMER 

# In[5]:


B = bureau[0:10000]
grp = B[['SK_ID_CURR', 'DAYS_CREDIT']].groupby(by = ['SK_ID_CURR'])['DAYS_CREDIT'].count().reset_index().rename(index=str, columns={'DAYS_CREDIT': 'BUREAU_LOAN_COUNT'})
B = B.merge(grp, on = ['SK_ID_CURR'], how = 'left')
print(B.shape)

# # FEATURE 2 - NUMBER OF TYPES OF PAST LOANS PER CUSTOMER 

# In[7]:


B = bureau[0:10000]
grp = B[['SK_ID_CURR', 'CREDIT_TYPE']].groupby(by = ['SK_ID_CURR'])['CREDIT_TYPE'].nunique().reset_index().rename(index=str, columns={'CREDIT_TYPE': 'BUREAU_LOAN_TYPES'})
B = B.merge(grp, on = ['SK_ID_CURR'], how = 'left')
print(B.shape)

# # FEATURE 3 - AVERAGE NUMBER OF PAST LOANS PER TYPE PER CUSTOMER
# 
# # Is the Customer diversified in taking multiple types of Loan or Focused on a single type of loan
# 

# In[10]:


B = bureau[0:10000]
# Number of Loans per Customer
grp = B[['SK_ID_CURR', 'DAYS_CREDIT']].groupby(by = ['SK_ID_CURR'])['DAYS_CREDIT'].count().reset_index().rename(index=str, columns={'DAYS_CREDIT': 'BUREAU_LOAN_COUNT'})
B = B.merge(grp, on = ['SK_ID_CURR'], how = 'left')

# Number of types of Credit loans for each Customer 
grp = B[['SK_ID_CURR', 'CREDIT_TYPE']].groupby(by = ['SK_ID_CURR'])['CREDIT_TYPE'].nunique().reset_index().rename(index=str, columns={'CREDIT_TYPE': 'BUREAU_LOAN_TYPES'})
B = B.merge(grp, on = ['SK_ID_CURR'], how = 'left')

# Average Number of Loans per Loan Type
B['AVERAGE_LOAN_TYPE'] = B['BUREAU_LOAN_COUNT']/B['BUREAU_LOAN_TYPES']
del B['BUREAU_LOAN_COUNT'], B['BUREAU_LOAN_TYPES']
import gc
gc.collect()
print(B.shape)

# # FEATURE 4 - % OF ACTIVE LOANS FROM BUREAU DATA 

# In[16]:


B = bureau[0:10000]
# Create a new dummy column for whether CREDIT is ACTIVE OR CLOED 
B['CREDIT_ACTIVE_BINARY'] = B['CREDIT_ACTIVE']

def f(x):
    if x == 'Closed':
        y = 0
    else:
        y = 1    
    return y

B['CREDIT_ACTIVE_BINARY'] = B.apply(lambda x: f(x.CREDIT_ACTIVE), axis = 1)

# Calculate mean number of loans that are ACTIVE per CUSTOMER 
grp = B.groupby(by = ['SK_ID_CURR'])['CREDIT_ACTIVE_BINARY'].mean().reset_index().rename(index=str, columns={'CREDIT_ACTIVE_BINARY': 'ACTIVE_LOANS_PERCENTAGE'})
B = B.merge(grp, on = ['SK_ID_CURR'], how = 'left')
del B['CREDIT_ACTIVE_BINARY']
import gc
gc.collect()
print(B.shape)

B[B['SK_ID_CURR'] == 100653]

# # FEATURE 5
# 
# # AVERAGE NUMBER OF DAYS BETWEEN SUCCESSIVE PAST APPLICATIONS FOR EACH CUSTOMER 
# 
# # How often did the customer take credit in the past? Was it spaced out at regular time intervals - a signal of good financial planning OR were the loans concentrated around a smaller time frame - indicating potential financial trouble?
# 

# In[12]:


B = bureau[0:10000]
# Groupby each Customer and Sort values of DAYS_CREDIT in ascending order
grp = B[['SK_ID_CURR', 'SK_ID_BUREAU', 'DAYS_CREDIT']].groupby(by = ['SK_ID_CURR'])
grp1 = grp.apply(lambda x: x.sort_values(['DAYS_CREDIT'], ascending = False)).reset_index(drop = True)#rename(index = str, columns = {'DAYS_CREDIT': 'DAYS_CREDIT_DIFF'})
print("Grouping and Sorting done")

# Calculate Difference between the number of Days 
grp1['DAYS_CREDIT1'] = grp1['DAYS_CREDIT']*-1
grp1['DAYS_DIFF'] = grp1.groupby(by = ['SK_ID_CURR'])['DAYS_CREDIT1'].diff()
grp1['DAYS_DIFF'] = grp1['DAYS_DIFF'].fillna(0).astype('uint32')
del grp1['DAYS_CREDIT1'], grp1['DAYS_CREDIT'], grp1['SK_ID_CURR']
gc.collect()
print("Difference days calculated")

B = B.merge(grp1, on = ['SK_ID_BUREAU'], how = 'left')
print("Difference in Dates between Previous CB applications is CALCULATED ")
print(B.shape)

# # FEATURE 6  
# 
# # % of LOANS PER CUSTOMER WHERE END DATE FOR CREDIT IS PAST
# 
#  # INTERPRETING CREDIT_DAYS_ENDDATE 
#  
#  #  NEGATIVE VALUE - Credit date was in the past at time of application( Potential Red Flag !!! )
#  
#  # POSITIVE VALUE - Credit date is in the future at time of application ( Potential Good Sign !!!!)
#  
#  # NOTE : This is not the same as % of Active loans since Active loans 
#  # can have Negative and Positive values for DAYS_CREDIT_ENDDATE

# In[17]:


B = bureau[0:10000]
B['CREDIT_ENDDATE_BINARY'] = B['DAYS_CREDIT_ENDDATE']

def f(x):
    if x<0:
        y = 0
    else:
        y = 1   
    return y

B['CREDIT_ENDDATE_BINARY'] = B.apply(lambda x: f(x.DAYS_CREDIT_ENDDATE), axis = 1)
print("New Binary Column calculated")

grp = B.groupby(by = ['SK_ID_CURR'])['CREDIT_ENDDATE_BINARY'].mean().reset_index().rename(index=str, columns={'CREDIT_ENDDATE_BINARY': 'CREDIT_ENDDATE_PERCENTAGE'})
B = B.merge(grp, on = ['SK_ID_CURR'], how = 'left')

del B['CREDIT_ENDDATE_BINARY']
gc.collect()
print(B.shape)

# # FEATURE 7 
# 
# # AVERAGE NUMBER OF DAYS IN WHICH CREDIT EXPIRES IN FUTURE -INDICATION OF CUSTOMER DELINQUENCY IN FUTURE??

# In[20]:


# Repeating Feature 6 to Calculate all transactions with ENDATE as POSITIVE VALUES 

B = bureau[0:10000]
# Dummy column to calculate 1 or 0 values. 1 for Positive CREDIT_ENDDATE and 0 for Negative
B['CREDIT_ENDDATE_BINARY'] = B['DAYS_CREDIT_ENDDATE']

def f(x):
    if x<0:
        y = 0
    else:
        y = 1   
    return y

B['CREDIT_ENDDATE_BINARY'] = B.apply(lambda x: f(x.DAYS_CREDIT_ENDDATE), axis = 1)
print("New Binary Column calculated")

# We take only positive values of  ENDDATE since we are looking at Bureau Credit VALID IN FUTURE 
# as of the date of the customer's loan application with Home Credit 
B1 = B[B['CREDIT_ENDDATE_BINARY'] == 1]
B1.shape

#Calculate Difference in successive future end dates of CREDIT 

# Create Dummy Column for CREDIT_ENDDATE 
B1['DAYS_CREDIT_ENDDATE1'] = B1['DAYS_CREDIT_ENDDATE']
# Groupby Each Customer ID 
grp = B1[['SK_ID_CURR', 'SK_ID_BUREAU', 'DAYS_CREDIT_ENDDATE1']].groupby(by = ['SK_ID_CURR'])
# Sort the values of CREDIT_ENDDATE for each customer ID 
grp1 = grp.apply(lambda x: x.sort_values(['DAYS_CREDIT_ENDDATE1'], ascending = True)).reset_index(drop = True)
del grp
gc.collect()
print("Grouping and Sorting done")

# Calculate the Difference in ENDDATES and fill missing values with zero 
grp1['DAYS_ENDDATE_DIFF'] = grp1.groupby(by = ['SK_ID_CURR'])['DAYS_CREDIT_ENDDATE1'].diff()
grp1['DAYS_ENDDATE_DIFF'] = grp1['DAYS_ENDDATE_DIFF'].fillna(0).astype('uint32')
del grp1['DAYS_CREDIT_ENDDATE1'], grp1['SK_ID_CURR']
gc.collect()
print("Difference days calculated")

# Merge new feature 'DAYS_ENDDATE_DIFF' with original Data frame for BUREAU DATA
B = B.merge(grp1, on = ['SK_ID_BUREAU'], how = 'left')
del grp1
gc.collect()

# Calculate Average of DAYS_ENDDATE_DIFF

grp = B[['SK_ID_CURR', 'DAYS_ENDDATE_DIFF']].groupby(by = ['SK_ID_CURR'])['DAYS_ENDDATE_DIFF'].mean().reset_index().rename( index = str, columns = {'DAYS_ENDDATE_DIFF': 'AVG_ENDDATE_FUTURE'})
B = B.merge(grp, on = ['SK_ID_CURR'], how = 'left')
del grp 
#del B['DAYS_ENDDATE_DIFF']
del B['CREDIT_ENDDATE_BINARY'], B['DAYS_CREDIT_ENDDATE']
gc.collect()
print(B.shape)

# In[21]:


# Verification of Feature 
B[B['SK_ID_CURR'] == 100653]
# In the Data frame below we have 3 values not NAN 
# Average of 3 values = (0 +0 + 3292)/3 = 1097.33 
#The NAN Values are Not Considered since these values DO NOT HAVE A FUTURE CREDIT END DATE 

# # FEATURE 8 - DEBT OVER CREDIT RATIO 
# # The Ratio of Total Debt to Total Credit for each Customer 
# # A High value may be a red flag indicative of potential default

# In[22]:


B[~B['AMT_CREDIT_SUM_LIMIT'].isnull()][0:2]

# WE can see in the Table Below 
# AMT_CREDIT_SUM = AMT_CREDIT_SUM_DEBT + AMT_CREDIT_SUM_LIMIT

# In[23]:


B = bureau[0:10000]

B['AMT_CREDIT_SUM_DEBT'] = B['AMT_CREDIT_SUM_DEBT'].fillna(0)
B['AMT_CREDIT_SUM'] = B['AMT_CREDIT_SUM'].fillna(0)

grp1 = B[['SK_ID_CURR', 'AMT_CREDIT_SUM_DEBT']].groupby(by = ['SK_ID_CURR'])['AMT_CREDIT_SUM_DEBT'].sum().reset_index().rename( index = str, columns = { 'AMT_CREDIT_SUM_DEBT': 'TOTAL_CUSTOMER_DEBT'})
grp2 = B[['SK_ID_CURR', 'AMT_CREDIT_SUM']].groupby(by = ['SK_ID_CURR'])['AMT_CREDIT_SUM'].sum().reset_index().rename( index = str, columns = { 'AMT_CREDIT_SUM': 'TOTAL_CUSTOMER_CREDIT'})

B = B.merge(grp1, on = ['SK_ID_CURR'], how = 'left')
B = B.merge(grp2, on = ['SK_ID_CURR'], how = 'left')
del grp1, grp2
gc.collect()

B['DEBT_CREDIT_RATIO'] = B['TOTAL_CUSTOMER_DEBT']/B['TOTAL_CUSTOMER_CREDIT']

del B['TOTAL_CUSTOMER_DEBT'], B['TOTAL_CUSTOMER_CREDIT']
gc.collect()
print(B.shape)

# # FEATURE 9 - OVERDUE OVER DEBT RATIO 
# #  What fraction of total Debt is overdue per customer?
# # A high value could indicate a potential DEFAULT 

# In[24]:


B = bureau[0:10000]

B['AMT_CREDIT_SUM_DEBT'] = B['AMT_CREDIT_SUM_DEBT'].fillna(0)
B['AMT_CREDIT_SUM_OVERDUE'] = B['AMT_CREDIT_SUM_OVERDUE'].fillna(0)

grp1 = B[['SK_ID_CURR', 'AMT_CREDIT_SUM_DEBT']].groupby(by = ['SK_ID_CURR'])['AMT_CREDIT_SUM_DEBT'].sum().reset_index().rename( index = str, columns = { 'AMT_CREDIT_SUM_DEBT': 'TOTAL_CUSTOMER_DEBT'})
grp2 = B[['SK_ID_CURR', 'AMT_CREDIT_SUM_OVERDUE']].groupby(by = ['SK_ID_CURR'])['AMT_CREDIT_SUM_OVERDUE'].sum().reset_index().rename( index = str, columns = { 'AMT_CREDIT_SUM_OVERDUE': 'TOTAL_CUSTOMER_OVERDUE'})

B = B.merge(grp1, on = ['SK_ID_CURR'], how = 'left')
B = B.merge(grp2, on = ['SK_ID_CURR'], how = 'left')
del grp1, grp2
gc.collect()

B['OVERDUE_DEBT_RATIO'] = B['TOTAL_CUSTOMER_OVERDUE']/B['TOTAL_CUSTOMER_DEBT']

del B['TOTAL_CUSTOMER_OVERDUE'], B['TOTAL_CUSTOMER_DEBT']
gc.collect()
print(B.shape)

# # FEATURE 10 - AVERAGE NUMBER OF LOANS PROLONGED 

# In[25]:


B = bureau[0:10000]

B['CNT_CREDIT_PROLONG'] = B['CNT_CREDIT_PROLONG'].fillna(0)
grp = B[['SK_ID_CURR', 'CNT_CREDIT_PROLONG']].groupby(by = ['SK_ID_CURR'])['CNT_CREDIT_PROLONG'].mean().reset_index().rename( index = str, columns = { 'CNT_CREDIT_PROLONG': 'AVG_CREDITDAYS_PROLONGED'})
B = B.merge(grp, on = ['SK_ID_CURR'], how = 'left')
print(B.shape)

# In[ ]:



