#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# The last exercise in the [Intro to SQL](https://www.kaggle.com/learn/sql) built the queries for an **Expert-Finder** website. This website would identify Stack Overflow users who answered questions on any technical topic.
# 
# But Stack Overflow is a very large dataset. So, if your website serves 1000s of requests a day, you should optimize the queries to keep your website snappy and lower operating cost. In this exercise, you'll optimize sample queries and see how much more efficient you can make them. Then, you'll apply your insights to make the website more generally efficient.
# 
# ## Quick Data Overview
# 
# As a reminder, here are the tables in the publicly available Stack Overflow dataset.

# In[ ]:


from google.cloud import bigquery

# Create client object to access database
client = bigquery.Client()

# Specify dataset for high level overview of data
dataset_ref = client.dataset("stackoverflow", "bigquery-public-data")
dataset = client.get_dataset(dataset_ref)

# List all the tables
tables = client.list_tables(dataset)
for table in tables:  
    print(table.table_id)

# ## Review Structure of Answers Data
# Your primary focus is finding users who answered questions. So, you will need to use the **Answers** data. Here is a short overview of this data.

# In[ ]:


table_ref = dataset_ref.table("posts_answers")
table = client.get_table(table_ref)
# See the first five rows of data
client.list_rows(table, max_results=5).to_dataframe()

# You may notice that the `tags` field is empty. Here's a quick overview of the questions data, which can be joined to the answers and which has useful `tags` data for most questions.

# In[ ]:


table_ref = dataset_ref.table("posts_questions")
table = client.get_table(table_ref)
client.list_rows(table, max_results=5).to_dataframe()

# In[ ]:


query = \
"""
WITH bq_questions as (
SELECT title, accepted_answer_id 
FROM `bigquery-public-data.stackoverflow.posts_questions` 
WHERE tags like '%bigquery%' and accepted_answer_id is not NULL
)
SELECT ans.* 
FROM bq_questions inner join `bigquery-public-data.stackoverflow.posts_answers` ans
ON ans.Id = bq_questions.accepted_answer_id
"""

result = client.query(query).result().to_dataframe()

# In[ ]:


result.head()

# In[ ]:


query = \
"""
WITH bq_questions as (
SELECT title, accepted_answer_id 
FROM `bigquery-public-data.stackoverflow.posts_questions` 
WHERE tags like '%bigquery%' and accepted_answer_id is not NULL
)
SELECT ans.* 
FROM bq_questions inner join `bigquery-public-data.stackoverflow.posts_answers` ans
ON ans.Id = bq_questions.accepted_answer_id
"""

result = client.query(query).result().to_dataframe()
