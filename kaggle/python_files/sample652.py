#!/usr/bin/env python
# coding: utf-8

# # Summary
# 
# I'm exploring the data a bit to get a feel of where loans are more commonly being disbursed, where larger loans are being disbursed, how various factors affect loan amount. After this initial analysis, I will go into borrower profiles and how Kiva can use these to make decisions about disbursing loans.
# 
# 1. Number of Loans By Country
# 2. Most popular sectors in which loans are taken
# 3.  Distribution of Loan duration
# 4. Distribution of number of lenders
# 5. Gender of borrowers
# 6. Distribution of Loan Amount
#     * 6.1 Analysis of loan amount below \$2000
#     * 6.2 Analysis of loan amount  \$2,000 - \$20,000
#     * 6.3 Analysis of loan amount \$20,000 to \$60,000
#     * 6.4 Loan amount above \$60,000
#     * 6.5 Loan amount by Sector
#     * 6.6 Loan Amount by Gender
#     * 6.7 Loan Amount by Country
# 7. Time taken to fund loans
#     * 7.1 Maximum time taken for a loan to be funded
#     * 7.2 Distribution of time taken for a loan to be funded
#     * 7.3 Distribution of time taken for a loan to be funded greater than 100 days
#     * 7.4 Is there any difference in the time taken to fund a loan based on the gender of the borrower?
#     
# 
# 
# 
# 

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pylab
import seaborn as sns
plt.style.use('fivethirtyeight')
pylab.rcParams['figure.figsize'] = (10.0, 8.0)

# In[ ]:


loans_df = pd.read_csv("../input/kiva_loans.csv", parse_dates=['disbursed_time', 'funded_time', 'posted_time'])

# In[ ]:


loans_df.shape

# In[ ]:


loans_df.head()

# In[ ]:


# From: https://deparkes.co.uk/2016/11/04/sort-pandas-boxplot/
def boxplot_sorted(df, by, column):
    # use dict comprehension to create new dataframe from the iterable groupby object
    # each group name becomes a column in the new dataframe
    df2 = pd.DataFrame({col:vals[column] for col, vals in df.groupby(by)})
    # find and sort the median values in this new dataframe
    meds = df2.median().sort_values()
    # use the columns in the dataframe, ordered sorted by median value
    # return axes so changes can be made outside the function
    return df2[meds.index].boxplot(return_type="axes")

# ## 1. Number of loans by country
# - Since Kiva extends services to financially excluded people around the globe, it makes sense that the countries where the most loans are given out are developing nations like the Phillipines and Kenya

# In[ ]:


pylab.rcParams['figure.figsize'] = (24.0, 8.0)
plt.style.use('fivethirtyeight')
loans_df.groupby(loans_df.country).id.count().sort_values().plot.bar(color='cornflowerblue');
plt.title("Loan Count by Country");

# ## 2. Most popular sectors in which loans are taken
# - Agriculture, food and retail are the most popular sectors in which loans are taken

# In[ ]:


pylab.rcParams['figure.figsize'] = (8.0, 8.0)
loans_df.groupby(loans_df.sector).id.count().sort_values().plot.bar(color='cornflowerblue');
plt.title("Loan Count by Sector");

# ## 3. Distribution of Loan duration
# Most loans are short term loans of less than 24 months(two years

# In[ ]:


loans_df.term_in_months.plot.hist(bins=100);
plt.title("Loan Count by Loan Duration");

# ## 4. Distribution of number of lenders
# - Most loans have between 1 to 150 lenders with some outliers having large numbers of lenders with the maximum number of lenders being 2986

# In[ ]:


loans_df.lender_count.plot.box();
plt.title("Distribution of Number of Lenders per loan");

# In[ ]:


axes = plt.gca()
axes.set_xlim([0,500])
loans_df.lender_count.plot.hist(bins=1000);
plt.title("Distribution of Number of Lenders where number < 500");

# In[ ]:


max(loans_df.lender_count)

# ## 5. Gender of borrowers
# - Female only borrowers(single or group) are significantly more than male only borrowers and mixed groups

# In[ ]:


def process_gender(x):
    
    if type(x) is float and np.isnan(x):
        return "nan"
    genders = x.split(",")
    male_count = sum(g.strip() == 'male' for g in genders)
    female_count = sum(g.strip() == 'female' for g in genders)
    
    if(male_count > 0 and female_count > 0):
        return "MF"
    elif(female_count > 0):
        return "F"
    elif (male_count > 0):
        return "M"

# In[ ]:


loans_df.borrower_genders = loans_df.borrower_genders.apply(process_gender)

# In[ ]:


loans_df.borrower_genders.value_counts().plot.bar(color='cornflowerblue');
plt.title("Loan Count by Gender of Borrower");

# ## 6. Distribution of Loan Amount
# - We will consider the funded_amount variable as this is the amount which is disbursed to the borrower by the field agent
# - As all amounts are in USD, no currency conversion is required
# - Most of the values are below $2000, with 8\% of all loans lying above this value

# In[ ]:


loans_df.funded_amount.plot.box();
plt.title("Distribution of Loan Funded Amount");

# In[ ]:


# Q3 + 1.5 * IQR
IQR = loans_df.funded_amount.quantile(0.75) - loans_df.funded_amount.quantile(0.25)
upper_whisker = loans_df.funded_amount.quantile(0.75) + 1.5 * IQR
loans_above_upper_whisker = loans_df[loans_df.funded_amount > upper_whisker]
loans_above_upper_whisker.shape

# In[ ]:


# percentage of loans above upper whisker
loans_above_upper_whisker.shape[0]/loans_df.shape[0]

# ### 6.1 Analysis of loan amount below \$2000
# - The distribution is skewed to the right with higher loan amounts being less common

# In[ ]:


loans_below_upper_whisker = loans_df[loans_df.funded_amount < upper_whisker]

# In[ ]:


loans_below_upper_whisker.funded_amount.plot.hist();
plt.title("Distribution of Loan Funded amount < $2000");

# ### 6.2 Analysis of loan amount  \$2,000 - \$20,000
# - Most of the outliers lie in this range

# In[ ]:


df = loans_above_upper_whisker[loans_above_upper_whisker.funded_amount < 20000]
df.funded_amount.plot.hist();
plt.title("Distribution of Loan Funded Amount between \$2,000 and \$20,000");
df.shape

# ### 6.3 Analysis of loan amount \$20,000 to \$60,000
# - A few values lie in this range
# - Most of the high value loans are disbursed for Agriculture and Retail

# In[ ]:


df = loans_above_upper_whisker[(loans_above_upper_whisker.funded_amount > 20000) & (loans_above_upper_whisker.funded_amount < 60000)]
df.funded_amount.plot.hist()
plt.title("Distribution of Loan Funded Amount between \$20,000 and \$60,000");
df.shape

# In[ ]:


df.sector.value_counts().sort_values().plot.bar(color='cornflowerblue');
plt.title("Loan Count by Sector for Loan Amount between \$20,000 and \$60,000");

# ### 6.4 Loan amount above \$60,000
# - There is only a single loan amount with a value of \$100,000 in this range distributed for Agriculture in Haiti
# 

# In[ ]:


loans_df[loans_df.funded_amount > 60000]

# ### 6.5 Loan amount by Sector
# - Not much observable difference in distributions of loan amount by sector except that loan amounts for the personal use sector tends to be on the lower side

# In[ ]:


pylab.rcParams['figure.figsize'] = (16.0, 8.0)
boxplot_sorted(loans_df[loans_df.funded_amount < 10000], by=["sector"], column="funded_amount");
plt.xticks(rotation=90);

# > ### 6.6 Loan Amount by Gender
# - The distribution of loan amount show slightly lower amounts for female only borrowers than for male only borrowers. I will dig into this a bit more after exploring the breakdown by country. 
# - The distribution for mixed gender groups of borrowers is much more widespread

# In[ ]:


pylab.rcParams['figure.figsize'] = (8.0, 8.0)
boxplot_sorted(loans_df[(loans_df.funded_amount < 10000) & (loans_df.borrower_genders != "nan")], by=["borrower_genders"], column="funded_amount");

# In[ ]:


loan_amount_values = loans_df[(loans_df.funded_amount < 10000) & (loans_df.borrower_genders != "nan")].groupby("borrower_genders").loan_amount
loan_amount_values.median()

# In[ ]:


loan_amount_values.quantile(0.75) - loan_amount_values.quantile(0.25)

# ### 6.7 Loan Amount by Country
# - There's a lot going on here, but some countries that are clearly on the higher end of the loan amount spectrum are Afhanistan, Congo, Chile. It will be interesting to see what kind of sectors the loans in these countries were made for
# - In Afghanistan, there were only 2 loans disbursed with amounts between \$6,000 and \$8,000 and both of them were for Textile activity.
# - On the other end of the spectrum are countries like Nigeria which has the lowest distribution. A possible explanation for Nigeria is that the value of the dollar against the Nigerian naira is so high that even small loan amounts in dollar value go a long way

# In[ ]:


pylab.rcParams['figure.figsize'] = (24.0, 8.0)
boxplot_sorted(loans_df[(loans_df.funded_amount < 10000) & (loans_df.borrower_genders != "nan")], by=["country"], column="funded_amount");
plt.xticks(rotation=90);

# In[ ]:


loans_df[loans_df.country == 'Afghanistan']

# In[ ]:


pylab.rcParams['figure.figsize'] = (8.0, 8.0)
loans_df[loans_df.country == 'Chile'].sector.value_counts().plot.bar(color='cornflowerblue');

# ## Time taken to fund loans
# The loan disbursal process works like this in Kiva: the field agent disburses the loan to the borrower at the **disbursed_time**. The agent then posts the loan to Kiva at the **posted_time**. Lenders are then able to see the loan on Kiva and make contributions towards it. The time at which the loan has been completely funded is the **funded_time**
# 
# We can then consider the time taken to completed fund a loan to be **funded_time** - **posted_time**. There is only one case where the posted time is 17 days greater than the funded time. This is probably an error which I need to look into further.
# 
# - The longest time it took for a loan to get funded was 1 year and 2 months
# - Most of the loans are funded within a 100 days. Only 0.1% of all loans in the entire sample take more than a 100 days to be funded

# In[ ]:


time_to_fund = (loans_df.funded_time - loans_df.posted_time)
time_to_fund_in_days = (time_to_fund.astype('timedelta64[s]')/(3600 * 24))
loans_df = loans_df.assign(time_to_fund=time_to_fund)
loans_df = loans_df.assign(time_to_fund_in_days=time_to_fund_in_days)



# ### 7.1 Maximum time taken for a loan to be funded

# In[ ]:


max(time_to_fund_in_days)

# ### 7.2 Distribution of time taken for a loan to be funded

# In[ ]:


lower = loans_df.time_to_fund_in_days.quantile(0.01)
upper = loans_df.time_to_fund_in_days.quantile(0.99)
loans_df[(loans_df.time_to_fund_in_days > lower)].time_to_fund_in_days.plot.hist();

# ### 7.3 Distribution of time taken for a loan to be funded greater than 100 days

# In[ ]:


loans_df[(loans_df.time_to_fund_in_days > 100)].shape

# In[ ]:


loans_df[(loans_df.time_to_fund_in_days > 100)].shape[0]/loans_df.shape[0]

# In[ ]:


loans_df[(loans_df.time_to_fund_in_days > 100)].time_to_fund_in_days.plot.hist();

# ### 7.4 Is there any difference in the time taken to fund a loan based on the gender of the borrower?
#  - It looks like female only borrower/borrower groups take *slightly* less time to get funded. We would need to investigate if this is significant.

# In[ ]:


pylab.rcParams['figure.figsize'] = (8.0, 8.0)
boxplot_sorted(loans_df[loans_df.borrower_genders != 'nan'], by=["borrower_genders"], column="time_to_fund_in_days");

# ### 7.5 Is there any difference in the time taken to fund a loan based on the country of the borrower?
# - Countries are sorted in the increasing order of their median time to fund
# - Countries like Afghanistan and Chile are on the lower end of the spectrum. 
# - It's surprising to see the United States on the higher end of the spectrum.
# 

# In[ ]:


pylab.rcParams['figure.figsize'] = (24.0, 8.0)
#loans_df[["time_to_fund_in_days", "country"]].boxplot(by="country");
axes = boxplot_sorted(loans_df, by=["country"], column="time_to_fund_in_days")
axes.set_title("Time to Fund by country in days")
plt.xticks(rotation=90);

# In[ ]:



