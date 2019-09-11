#!/usr/bin/env python
# coding: utf-8

# # Access a Database with Python - Iris Dataset
# 
# The Iris dataset is a popular dataset especially in the Machine Learning community, it is a set of features of 50  Iris flowers and their classification into 3 species.
# It is often used to introduce classification Machine Learning algorithms.
# 
# First let's download the dataset in `SQLite` format from Kaggle:

# <p><img   src="https://upload.wikimedia.org/wikipedia/commons/4/49/Iris_germanica_%28Purple_bearded_Iris%29%2C_Wakehurst_Place%2C_UK_-_Diliff.jpg" alt="Iris germanica (Purple bearded Iris), Wakehurst Place, UK - Diliff.jpg" height="145" width="114"></p>
# 
# <p><br> From <a href="https://commons.wikimedia.org/wiki/File:Iris_germanica_(Purple_bearded_Iris),_Wakehurst_Place,_UK_-_Diliff.jpg#/media/File:Iris_germanica_(Purple_bearded_Iris),_Wakehurst_Place,_UK_-_Diliff.jpg">Wikimedia</a>, by <a href="//commons.wikimedia.org/wiki/User:Diliff" title="User:Diliff">Diliff</a> - <span class="int-own-work" lang="en">Own work</span>, <a href="http://creativecommons.org/licenses/by-sa/3.0" title="Creative Commons Attribution-Share Alike 3.0">CC BY-SA 3.0</a>, <a href="https://commons.wikimedia.org/w/index.php?curid=33037509">Link</a></p>

# First let's check that the sqlite database is available and display an error message if the file is not available (`assert` checks if the expression is `True`, otherwise throws `AssertionError` with the error message string provided):

# In[ ]:


import os
data_iris_folder_content = os.listdir("data/iris")

# In[ ]:


error_message = "Error: sqlite file not available, check instructions above to download it"
assert "database.sqlite" in data_iris_folder_content, error_message

# ## Access the Database with the sqlite3 Package

# We can use the `sqlite3` package from the Python standard library to connect to the `sqlite` database:

# In[ ]:


import sqlite3

# In[ ]:


conn = sqlite3.connect('data/iris/database.sqlite')

# In[ ]:


cursor = conn.cursor()

# In[ ]:


type(cursor)

# A `sqlite3.Cursor` object is our interface to the database, mostly throught the `execute` method that allows to run any `SQL` query on our database.
# 
# First of all we can get a list of all the tables saved into the database, this is done by reading the column `name` from the `sqlite_master` metadata table with:
# 
#     SELECT name FROM sqlite_master
#     
# The output of the `execute` method is an iterator that can be used in a `for` loop to print the value of each row.

# In[ ]:


for row in cursor.execute("SELECT name FROM sqlite_master"):
    print(row)

# a shortcut to directly execute the query and gather the results is the `fetchall` method:

# In[ ]:


cursor.execute("SELECT name FROM sqlite_master").fetchall()

# **Notice**: this way of finding the available tables in a database is specific to `sqlite`, other databases like `MySQL` or `PostgreSQL` have different syntax.

# Then we can execute standard `SQL` query on the database, `SQL` is a language designed to interact with data stored in a relational database. It has a standard specification, therefore the commands below work on any database.
# 
# If you need to connect to another database, you would use another package instead of `sqlite3`, for example:
# 
# * [MySQL Connector](https://dev.mysql.com/doc/connector-python/en/) for MySQL
# * [Psycopg](http://initd.org/psycopg/docs/install.html) for PostgreSQL
# * [pymssql](http://pymssql.org/en/stable/) for Microsoft MS SQL
# 
# then you would connect to the database using specific host, port and authentication credentials but then you could execute the same exact `SQL` statements.
# 
# Let's take a look for example at the first 3 rows in the Iris table:

# In[ ]:


sample_data = cursor.execute("SELECT * FROM Iris LIMIT 20").fetchall()

# In[ ]:


print(type(sample_data))
sample_data

# In[ ]:


[row[0] for row in cursor.description]

# It is evident that the interface provided by `sqlite3` is low-level, for data exploration purposes we would like to directly import data into a more user friendly library like `pandas`.

# ## Import data from a database to `pandas`

# In[ ]:


import pandas as pd

# In[ ]:


iris_data = pd.read_sql_query("SELECT * FROM Iris", conn)

# In[ ]:


iris_data.head()

# In[ ]:


iris_data.dtypes

# `pandas.read_sql_query` takes a `SQL` query and a connection object and imports the data into a `DataFrame`, also keeping the same data types of the database columns. `pandas` provides a lot of the same functionality of `SQL` with a more user-friendly interface.
# 
# However, `sqlite3` is extremely useful for downselecting data **before** importing them in `pandas`.
# 
# For example you might have 1 TB of data in a table stored in a database on a server machine. You are interested in working on a subset of the data based on some criterion, unfortunately it would be impossible to first load data into `pandas` and then filter them, therefore we should tell the database to perform the filtering and just load into `pandas` the downsized dataset.

# In[ ]:


iris_setosa_data = pd.read_sql_query("SELECT * FROM Iris WHERE Species == 'Iris-setosa'", conn)

# In[ ]:


iris_setosa_data
print(iris_setosa_data.shape)
print(iris_data.shape)


# In[ ]:



