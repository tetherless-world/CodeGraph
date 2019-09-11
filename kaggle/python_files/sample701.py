#!/usr/bin/env python
# coding: utf-8

# **[SQL Micro-Course Home Page](https://www.kaggle.com/learn/SQL)**
# 
# ---
# 

# # Introduction
# 
# **SELECT** statements or queries are the most important part of SQL.  
# 
# We use the keywords **SELECT**, **FROM** and **WHERE** to get data from specific columns based on conditions you specify. 
# 
# ### SELECT ... FROM
# 
# The most basic SQL query selects a single column from a single table. To do this, you specify the column you want after the word **SELECT** and then specify what table to pull the column from after the word **FROM**. 
# 
# Let's see a query in a small imaginary database, `pet_records` which has just one table in it, called `pets`.
# 
# ![](https://i.imgur.com/Ef4Puo3.png)
# 
# 
# So, if we wanted to select the `Name` column from the `pets` table of the `pet_records` database (if that database were accessible as a BigQuery dataset on Kaggle, which it is not, because I made it up), we would do this:
# 
#     SELECT Name
#     FROM `bigquery-public-data.pet_records.pets`
# 
# It would return the highlighted data from this figure.
# 
# ![](https://i.imgur.com/8FdVyFP.png)
# 
# Note that the argument we pass to **FROM** is *not* in single or double quotation marks (' or "). It is in backticks (\`). We use it to identify the relevant BigQuery data source.
# 
# > **Do you need to capitalize SELECT and FROM?** No, SQL doesn't care about capitalization. However, it's customary to capitalize your SQL commands, and it makes your queries a bit easier to read.
# 
# 
# ### WHERE ...
# 
# BigQuery datasets are large. So you'll usually want to return only the rows meeting specific conditions. You can do this using the **WHERE** clause:
# 
# Here's an example:
# 
#     SELECT Name
#     FROM `bigquery-public-data.pet_records.pets`
#     WHERE Animal = 'Cat'
# 
# This query will only return the entries from the `Name` column that are in rows where the `Animal` column has the text `Cat` in it. Those are the cells highlighted in blue in this figure:
# 
# ![](https://i.imgur.com/Va52Qdl.png)
# 
# 

# ## Example: What are all the U.S. cities in the OpenAQ dataset?
# 
# Now that you've got the basics down, let's work through an example with a real dataset. We'll use the OpenAQ dataset about air quality.
# 
# First, we'll set up everything we need to run queries and take a quick peek at what tables are in our database.

# In[ ]:


# import package with helper functions 
import bq_helper

# create a helper object for this dataset
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")

# print all the tables in this dataset (there's only one!)
open_aq.list_tables()

# We can peek at the first few rows to see what sort of data is in this dataset.

# In[ ]:


# print the first couple rows of the "global_air_quality" dataset
open_aq.head("global_air_quality")

# Everything looks good! Let's put together a query. I want to select all the values from the "city" column for the rows where the "country" column is `US` (for "United States"). 
# 
# > **What's up with the triple quotation marks (""")?** These tell Python that everything inside them is a single string, even though we have line breaks in it. The line breaks aren't necessary, but they make it easier to read your query.

# In[ ]:


# query to select all the items from the "city" column where the
# "country" column is "us"
query = """SELECT city
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE country = 'US'
        """

# Now I can use this query to get information from our open_aq dataset. I'm using the `BigQueryHelper.query_to_pandas_safe()` method here because it won't run a query if it's too large. More about that soon.

# In[ ]:


# the query_to_pandas_safe will only return a result if it's less
# than one gigabyte (by default)
us_cities = open_aq.query_to_pandas_safe(query)

# Now I've got a Pandas dataframe called us_cities, which I can use like any other dataframe:

# In[ ]:


# What five cities have the most measurements taken there?
us_cities.city.value_counts().head()

# If you want multiple columns, you can select them with a column between the names:

# In[ ]:


query = """SELECT city, country
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE country = 'US'
        """

# You can select all columns of data with a `*` like this:

# In[ ]:


query = """SELECT *
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE country = 'US'
        """

# 
# # Working With Big Datasets
# 
# BigQuery datasets can be huge. We allow you to do a lot of computation for free, but everyone has some limit. 
# 
# **Each Kaggle user can scan 5TB every 30 days for free.  Once you hit that limit, you'll have to wait for it to reset.**
# 
# The [biggest dataset currently on Kaggle](https://www.kaggle.com/github/github-repos) is 3 terabytes, so if you aren't a little careful, you can go through your 30-day limit in a couple queries.
# 
# Don't worry though: if you use `query_to_pandas_safe` you won't pull too much data at once and run over your limit.
# 
# Another way to be careful is to estimate how big your query will be before you actually execute it. You can do this with the `BigQueryHelper.estimate_query_size()` method. 
# 
# This is better than relying on your intuition for query size, because your quota is on data *scanned*, not the amount of data returned. And it's tricky to know how much data a database will need to "scan" to return your results, even if you have a good sense of how large the results will be.
# 
# Here's an example workflow using a big dataset.

# In[ ]:


hacker_news = bq_helper.BigQueryHelper(active_project= "bigquery-public-data", 
                                       dataset_name = "hacker_news")

# this query looks in the full table in the hacker_news
# dataset, then gets the score column from every row where 
# the type column has "job" in it.
query = """SELECT score
            FROM `bigquery-public-data.hacker_news.full`
            WHERE type = "job" """

# check how big this query will be
hacker_news.estimate_query_size(query)

# The `query_to_pandas_safe` has an optional parameter to specify how much data you are willing to scan for any specific query.

# In[ ]:


# only run this query if it's less than 100 MB
hacker_news.query_to_pandas_safe(query, max_gb_scanned=0.1)

# And here's an example where the same query returns a dataframe. 

# In[ ]:


# check out the scores of job postings (if the 
# query is smaller than 1 gig)
job_post_scores = hacker_news.query_to_pandas_safe(query)

# We can work with the resulting DataFrame as we would any other dataframe. For example, we can get the mean of the column:

# In[ ]:


# average score for job posts
job_post_scores.score.mean()

# # Your Turn
# 
# Writing **SELECT** statements is the key to using SQL. So **[try your new skills](https://www.kaggle.com/kernels/fork/681989)**!
# 

# ---
# **[SQL Micro-Course Home Page](https://www.kaggle.com/learn/SQL)**
# 
# 
