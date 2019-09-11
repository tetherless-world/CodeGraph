#!/usr/bin/env python
# coding: utf-8

# # **Pandas Tutorial** <a id="0"></a>
# <hr>
# 1. [Overview](#1)
# 2. [Pandas Library About](#2)
# 3. [Import Library](#3)
# 4. [Pandas Data Structure](#4)
#     * [Series](#5)
#     * [DataFrame](#6)
# 5. [Import Data](#7)
#     * [CSV](#8)
#     * [Excel](#9)
#     * [Others(json, SQL, html)](#10)
# 6. [Exporting Data](#11)
# 7. [Create Test Objects](#12)
# 8. [Summariza Data](#13)
#     * [df.info()](#14)
#     * [df.shape()](#15)
#     * [df.index](#16)
#     * [df.columns](#17)
#     * [df.count()](#18)
#     * [df.sum()](#19)
#     * [df.cumsum()](#20)
#     * [df.min()](#21)
#     * [df.max()](#22)
#     * [idxmin()](#23)
#     * [idxmax()](#24)
#     * [df.describe()](#25)
#     * [df.mean()](#26)
#     * [df.median()](#27)
#     * [df.quantile([0.25,0.75])](#28)
#     * [df.var()](#29)
#     * [df.std()](#30)
#     * [df.cummax()](#31)
#     * [df.cummin()](#32)
#     * [df['columnName'].cumproad()](#33)
#     * [len(df)](#34)
#     * [df.isnull()](#35)
#     * [df.corr()](#81)
# 9. [Pandas with Selection & Filtering](#36)
#     * [series['index']](#37)
#     * [df[n:n]](#38)
#     * [df.iloc[[0],[5]]](#39)
#     * [df.loc[n:n]](#40)
#     * [df['columnName']](#41)
#     * [df['columnName][n]](#42)
#     * [df['columnName'].nunique()](#43)
#     * [df['columnName'].unique()](#44)
#     * [df.columnName](#45)
#     * [df['columnName'].value_counts(dropna =False)](#46)
#     * [df.head(n)](#47)
#     * [df.tail(n)](#48)
#     * [df.sample(n)](#49)
#     * [df.sample(frac=0.5)](#50)
#     * [df.nlargest(n,'columnName')](#51)
#     * [df.nsmallest(n,'columnName')](#52)
#     * [df[df.columnName < n]](#53)
#     * [df[['columnName','columnName']] ](#54)
#     * [df.loc[:,"columnName1":"columnName2"]](#55)
#     * [Create Filter](#56)
#     * [df.filter(regex = 'code')](#57)
#     * [np.logical_and](#58)
#     * [Filtering with &](#59)
# 10. [Sort Data](#60)
#     * [df.sort_values('columnName')](#61)
#     * [df.sort_values('columnName', ascending=False)](#62)
#     * [df.sort_index()](#63)
# 11. [Rename & Defining New & Change Columns](#64)
#     * [df.rename(columns= {'columnName' : 'newColumnName'})](#65)
#     * [Defining New Column](#66)
#     * [Change Index Name](#67)
#     * [Make all columns lowercase](#68)
#     * [Make all columns uppercase](#69)
# 12. [Drop Data](#70)
#     * [df.drop(columns=['columnName'])](#71)
#     * [Series.drop(['index'])](#72)
#     * [Drop an observation (row)](#82)
#     * [Drop a variable (column)](#83)
# 13. [Convert Data Types](#73)
#     * [df.dtypes](#74)
#     * [df['columnName'] = df['columnName'].astype('dataType')](#75)
#     * [pd.melt(frame=dataFrameName,id_vars = 'columnName', value_vars= ['columnName'])](#76)
# 14. [Apply Function](#77)
#     * [Method 1](#78)
#     * [Method 2](#79)
# 15. [Utilities Code](#80)

# # **Overview** <a id="1"></a>
# <mark>[Return Contents](#0)
# <hr>
# 
# Welcome to my Kernel! In this kernel, I show you Pandas functions and how to use pandas. Why do I this? Because everyone who's just starting out or who's a professional is using the pandas.
# 
# If you have a question or feedback, do not hesitate to write and if you **like** this kernel, please do not forget to **UPVOTE**.

# # **What is the pandas?** <a id="2"></a>
# <mark>[Return Contents](#0)
# <hr>
# 
# pandas is an open source, BSD-licensed library providing high-performance, easy-to-use data structures and data analysis tools for the Python programming language.
# 
# pandas is a NumFOCUS sponsored project. This will help ensure the success of development of pandas as a world-class open-source project, and makes it possible to donate to the project.

# # **Import Library** <a id="3"></a>
# <mark>[Return Contents](#0)
# <hr>

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # import in pandas

import os
print(os.listdir("../input"))

# # **Pandas Data Structure** <a id="4"></a>
# <mark>[Return Contents](#0)
# <hr>
# 
# Pandas has two types of data structures. These are series and dataframe.
# 
# ### **Series** <a id="5"></a>
# 
# The series is a one-dimensional labeled array. It can accommodate any type of data in it.

# In[ ]:


mySeries = pd.Series([3,-5,7,4], index=['a','b','c','d'])
type(mySeries)

# ### **DataFrame** <a id="6"></a>
# 
# The dataframe is a two-dimensional data structure. It contains columns.

# In[ ]:


data = {'Country' : ['Belgium', 'India', 'Brazil' ],
        'Capital': ['Brussels', 'New Delhi', 'Brassilia'],
        'Population': [1234,1234,1234]}
datas = pd.DataFrame(data, columns=['Country','Capital','Population'])
print(type(data))
print(type(datas))

# # **Import Library** <a id="7"></a>
# <mark>[Return Contents](#0)
# <hr>
# 
# With pandas, we can open CSV, Excel and SQL databases. I will show you how to use this method for CSV and Excel files only.
# 
# ### **CSV(comma - separated values)** <a id="8"></a>
# 
# It is very easy to open and read CSV files and to overwrite the CSV file.

# In[ ]:


df = pd.read_csv('../input/DJIA_table.csv')
type(df)
# If your Python file is not in the same folder as your CSV file, you should do this as follows.
# df = pd.read_csv('/home/desktop/Iris.csv')

# ### **Excel** <a id="9"></a>
# 
# When we want to work with Excel files, we need to type the following code.

# In[ ]:


# pd.read_excel('filename')
# pd.to_excel('dir/dataFrame.xlsx', sheet_name='Sheet1')

# ### **Others(json, SQL, table, html)** <a id="10"></a>

# In[ ]:


# pd.read_sql(query,connection_object) -> Reads from a SQL table/database
# pd.read_table(filename) -> From a delimited text file(like TSV)
# pd.read_json(json_string) -> Reads from a json formatted string, URL or file
# pd.read_html(url) -> Parses an html URL, string or file and extracts tables to a list of dataframes
# pd.read_clipboard() -> Takes the contentes of your clipboard and passes it to read_table()
# pd.DataFrame(dict) -> From a dict, keys for columns names, values for data as lists

# # **Exporting Data** <a id="11"></a>
# <mark>[Return Contents](#0)
# <hr>

# In[ ]:


# df.to_csv(filename) -> Writes to a CSV file
# df.to_excel(filename) -> Writes on an Excel file
# df.to_sql(table_name, connection_object) -> Writes to a SQL table
# df.to_json(filename) -> Writes to a file in JSON format
# df.to_html(filename) -> Saves as an HTML table
# df.to_clipboard() -> Writes to the clipboard

# # **Create Test Objects** <a id="12"></a>
# <mark>[Return Contents](#0)
# <hr>

# In[ ]:


pd.DataFrame(np.random.rand(20,5)) # 5 columns and 20 rows of random floats

# # **Summarize Data** <a id="13"></a>
# <mark>[Return Contents](#0)
# <hr>
# 
# It's easy to get information about data with pandas. It makes it easier for us. Let's examine the existing functions one by one

# ### **df.info()** <a id="14"></a>
# This Code provides detailed information about our data.
# 
# * **RangeIndex:** Specifies how many data there is.
# * **Data Columns:** Specifies how many columns are found.
# * **Columns:** Gives information about Columns.
# * **dtypes:** It says what kind of data you have and how many of these data you have.
# * **Memory Usage:** It says how much memory usage is.
# 
# <mark>[Return Contents](#0)
# <hr>

# In[ ]:


df.info()

# ### **df.shape()** <a id="15"></a>
# This code shows us the number of rows and columns.
# 
# <mark>[Return Contents](#0)
# <hr>

# In[ ]:


df.shape

# ### **df.index** <a id="16"></a>
# This code shows the total number of index found.
# 
# <mark>[Return Contents](#0)
# <hr>

# In[ ]:


df.index

# ### **df.columns** <a id="17"></a>
# This code shows all the columns contained in the data we have examined.
# 
# <mark>[Return Contents](#0)
# <hr>

# In[ ]:


df.columns

# ### **df.count()** <a id="18"></a>
# This code shows us how many pieces of data are in each column.
# 
# <mark>[Return Contents](#0)
# <hr>

# In[ ]:


df.count()

# ### **df.sum()** <a id="19"></a>
# This code shows us the sum of the data in each column.
# 
# <mark>[Return Contents](#0)
# <hr>

# In[ ]:


df.sum()

# ### **df.cumsum()** <a id="20"></a>
# This code gives us cumulative sum of the data.
# 
# <mark>[Return Contents](#0)
# <hr>

# In[ ]:


df.cumsum().head()

# ### **df.min()** <a id="21"></a>
# This code brings us the smallest of the data.
# 
# <mark>[Return Contents](#0)
# <hr>

# In[ ]:


df.min()

# ### **df.max()** <a id="22"></a>
# This code brings up the largest among the data.
# 
# <mark>[Return Contents](#0)
# <hr>

# In[ ]:


df.max()

# ### **idxmin()**  <a id="23"></a>
# This code fetches the smallest value in the data. The use on series and dataframe is different.
# 
# <mark>[Return Contents](#0)
# <hr>

# In[ ]:


print("df: ",df['Open'].idxmin())
print("series", mySeries.idxmin())

# ### **idxmax()**  <a id="24"></a>
# This code returns the largest value in the data.
# 
# <mark>[Return Contents](#0)
# <hr>

# In[ ]:


print("df: ",df['Open'].idxmax())
print("series: ",mySeries.idxmax())

# ### **df.describe()**  <a id="25"></a>
# This Code provides basic statistical information about the data. The numerical column is based.
# 
# * **count:** vnumber of entries
# * **mean: **average of entries
# * **std:** standart deviation
# * **min:** minimum entry
# * **25%:** first quantile
# * **50%:** median or second quantile
# * **75%:** third quantile
# * **max:** maximum entry
# 
# <mark>[Return Contents](#0)
# <hr>

# In[ ]:


df.describe()

# ### **df.mean()**  <a id="26"></a>
# This code returns the mean value for the numeric column.
# 
# <mark>[Return Contents](#0)
# <hr>

# In[ ]:


df.mean()

# ### **df.median()**  <a id="27"></a>
# This code returns median for columns with numeric values.
# 
# <mark>[Return Contents](#0)
# <hr>

# In[ ]:


df.median()

# ### **df.quantile([0.25,0.75])**  <a id="28"></a>
# This code calculates the values 0.25 and 0.75 of the columns for each column.
# 
# <mark>[Return Contents](#0)
# <hr>

# In[ ]:


df.quantile([0.25,0.75])

# ### **df.var()**  <a id="29"></a>
# This code calculates the variance value for each column with a numeric value.
# 
# <mark>[Return Contents](#0)
# <hr>

# In[ ]:


df.var()

# ### **df.std()** <a id="30"></a>
# This code calculates the standard deviation value for each column with numeric value.
# 
# <mark>[Return Contents](#0)
# <hr>

# In[ ]:


df.std()

# ### **df.cummax()** <a id="31"></a>
# This code calculates the cumulative max value between the data.
# 
# <mark>[Return Contents](#0)
# <hr>

# In[ ]:


df.cummax()

# ### **df.cummin()** <a id="32"></a>
# This code returns the cumulative min value of the data.
# 
# <mark>[Return Contents](#0)
# <hr>

# In[ ]:


df.cummin()

# ### **df['columnName'].cumproad()** <a id="33"></a>
# This code returns the cumulative production of the data.
# 
# <mark>[Return Contents](#0)
# <hr>

# In[ ]:


df['Open'].cumprod().head()

# ### **len(df)** <a id="34"></a>
# This code gives you how many data there is.
# 
# <mark>[Return Contents](#0)
# <hr>

# In[ ]:


len(df)

# ### **df.isnull()** <a id="35"></a>
# Checks for null values, returns boolean.
# 
# <mark>[Return Contents](#0)
# <hr>

# In[ ]:


df.isnull().head()

# ### **df.corr()** <a id="81"></a>
# it gives information about the correlation between the data.
# 
# <mark>[Return Contents](#0)
# <hr>

# In[ ]:


df.corr()

# # **Selection & Filtering** <a id="36"></a>
# <mark>[Return Contents](#0)
# <hr>
# 
# This is how we can choose the data we want with pandas, how we can bring unique data.

# ### **mySeries['b']** <a id="37"></a>
# This code returns data with a value of B in series.
# 
# <mark>[Return Contents](#0)
# <hr>

# In[ ]:


mySeries['b']

# ### **df[n:n]** <a id="38"></a>
# This code fetches data from N to N.
# 
# <mark>[Return Contents](#0)
# <hr>

# In[ ]:


df[1982:]
#Or
#df[5:7]

# ### **df.iloc[[n],[n]]** <a id="39"></a>
# This code brings the data in the N row and N column in the DataFrame.
# 
# <mark>[Return Contents](#0)
# <hr>

# In[ ]:


df.iloc[[0],[3]]

# ### **df.loc[n:n]** <a id="40"></a>
# This code allows us to fetch the data in the range we specify.
# 
# <mark>[Return Contents](#0)
# <hr>

# In[ ]:


#df.loc[n:]
# OR
df.loc[5:7]

# ### **df['columnName']** <a id="41"></a>
# With this code, we can select and bring any column we want.
# 
# <mark>[Return Contents](#0)
# <hr>

# In[ ]:


df['Open'].head()
# OR
# df.Open

# ### **df['columnName'][n]** <a id="42"></a>
# With this code, we can select and return any value of the column we want.
# 
# <mark>[Return Contents](#0)
# <hr>

# In[ ]:


df['Open'][0]
# OR
# df.Open[0]
# df["Open"][1]
# df.loc[1,["Open"]]

# ### **df['columnName'].nunique()** <a id="43"></a>
# This code shows how many of the data that is in the selected column and does not repeat.
# 
# <mark>[Return Contents](#0)
# <hr>

# In[ ]:


df['Open'].nunique()

# ### **df['columnName'].unique()** <a id="44"></a>
# This code shows which of the data in the selected column repeats.
# 
# <mark>[Return Contents](#0)
# <hr>

# In[ ]:


df['Open'].unique()
# We can write the above code as follows:: df.Open.unique()

# ### **df.columnName** <a id="45"></a>
# This code is another way to select the column we want.
# 
# <mark>[Return Contents](#0)
# <hr>

# In[ ]:


df.Open.head()

# ### **df['columnName'].value_counts(dropna =False)** <a id="46"></a>
# This code counts all of the data in the column we have specified, but does not count the null/none values.
# 
# <mark>[Return Contents](#0)
# <hr>

# In[ ]:


print(df.Open.value_counts(dropna =True).head())
# OR
# print(df['Item'].value_counts(dropna =False))

# ### **df.head(n)** <a id="47"></a>
# This code optionally brings in the first 5 data. returns the number of data that you type instead of N.
# 
# <mark>[Return Contents](#0)
# <hr>

# In[ ]:


df.head()
# OR
# df.head(15)

# ### **df.tail(n)** <a id="48"></a>
# This code optionally brings 5 data at the end. returns the number of data that you type instead of N.
# 
# <mark>[Return Contents](#0)
# <hr>

# In[ ]:


df.tail()
# OR
# df.tail(20)

# ### **df.sample(n)** <a id="49"></a>
# This code fetches random n data from the data.
# 
# <mark>[Return Contents](#0)
# <hr>

# In[ ]:


df.sample(5)

# ### **df.sample(frac=0.5)** <a id="50"></a>
# This code selects the fractions of random rows and fetches the data to that extent.
# 
# <mark>[Return Contents](#0)
# <hr>

# In[ ]:


df.sample(frac=0.5).head()

# ### **df.nlargest(n,'columnName')** <a id="51"></a>
# This code brings N from the column where we have specified the largest data.
# 
# <mark>[Return Contents](#0)
# <hr>

# In[ ]:


df.nlargest(5,'Open')

# ### **df.nsmallest(n,'columnName')** <a id="52"></a>
# This code brings N from the column where we have specified the smallest data.
# 
# <mark>[Return Contents](#0)
# <hr>

# In[ ]:


df.nsmallest(3,'Open')

# ### **df[df.columnName < 5]** <a id="53"></a>
# This code returns the column name we have specified, which is less than 5.
# 
# <mark>[Return Contents](#0)
# <hr>

# In[ ]:


df[df.Open > 18281.949219]

# ### **df[['columnName','columnName']]** <a id="54"></a>
# This code helps us pick and bring any columns we want.
# 
# <mark>[Return Contents](#0)
# <hr>

# In[ ]:


df[['High','Low']].head()
# df.loc[:,["High","Low"]]

# ### **df.loc[:,"columnName1":"columnName2"]** <a id="55"></a>
# This code returns columns from columnname1 to columnname2.
# 
# <mark>[Return Contents](#0)
# <hr>

# In[ ]:


df.loc[:,"Date":"Close"].head()
# OR
# data.loc[:3,"Date":"Close"]

# ### **Create Filter** <a id="56"></a>
# <mark>[Return Contents](#0)
# <hr>

# In[ ]:


filters = df.Date > '2016-06-27'
df[filters]

# ### **df.filter(regex = 'code')** <a id="57"></a>
# This code allows regex to filter any data we want.
# 
# <mark>[Return Contents](#0)
# <hr>

# In[ ]:


df.filter(regex='^L').head()

# ### **np.logical_and** <a id="58"></a>
# Filtering with logical_and. Lets look at the example.
# 
# <mark>[Return Contents](#0)
# <hr>

# In[ ]:


df[np.logical_and(df['Open']>18281.949219, df['Date']>'2015-05-20' )]

# ### **Filtering with &** <a id="59"></a>
# <mark>[Return Contents](#0)
# <hr>

# In[ ]:


df[(df['Open']>18281.949219) & (df['Date']>'2015-05-20')]

# # **Sort Data** <a id="60"></a>
# 
# <mark>[Return Contents](#0)
# <hr>
# 
# ### **df.sort_values('columnName')** <a id="61"></a>
# This code sorts the column we specify in the form of low to high.

# In[ ]:


df.sort_values('Open').head()

# ### **df.sort_values('columnName', ascending=False)** <a id="62"></a>
# This code is the column we specify in the form of high to low.
# 
# <mark>[Return Contents](#0)
# <hr>

# In[ ]:


df.sort_values('Date', ascending=False).head()

# ### **df.sort_index()** <a id="63"></a>
# This code sorts from small to large according to the DataFrame index.
# 
# <mark>[Return Contents](#0)
# <hr>

# In[ ]:


df.sort_index().head()

# # **Rename & Defining New & Change Columns** <a id="64"></a>
# <mark>[Return Contents](#0)
# <hr>

# ### **df.rename(columns= {'columnName' : 'newColumnName'})** <a id="65"></a>
# This code helps us change the column name. The code I wrote below changes the ID value, but as we did not assign the change to the variable DF, it seems to be unchanged as you see below.

# In[ ]:


df.rename(columns= {'Adj Close' : 'Adjclose'}).head()
# df = df.rename(columns= {'Id' : 'Identif'}, inplace=True) -> True way
# inplace= True or False; This meaning, overwrite the data set.
# Other Way
# df.columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'adjclose']

# ###  **Defining New Column** <a id="66"></a>
# Create a new column
# 
# <mark>[Return Contents](#0)
# <hr>

# In[ ]:


df["Difference"] = df.High - df.Low
df.head()

# ### **Change Index Name** <a id="67"></a>
# Change index name to new index name
# 
# <mark>[Return Contents](#0)
# <hr>

# In[ ]:


print(df.index.name)
df.index.name = "index_name"
df.head()

# ### **Make all columns lowercase** <a id="68"></a>
# <mark>[Return Contents](#0)
# <hr>

# In[ ]:


#df.columns = map(str.lower(), df.columns)

# ### **Make all columns uppercase** <a id="69"></a>
# <mark>[Return Contents](#0)
# <hr>

# In[ ]:


#df.columns = map(str.upper(), df.columns)

# # **Drop Data** <a id="70"></a>
# 
# <mark>[Return Contents](#0)
# <hr>

# ### **df.drop(columns=['columnName'])** <a id="71"></a>
# This code deletes the column we have specified. But as above, I have to reset the delete to the df variable again.

# In[ ]:


df.drop(columns=['Adj Close']).head()
# df = df.drop(columns=['Id']) -> True way
# OR
# df = df.drop('col', axis=1)
# axis = 1 is meaning delete columns
# axis = 0 is meaning delete rows

# ### **mySeries.drop(['a'])** <a id="72"></a>
# This code allows us to delete the value specified in the series.
# 
# <mark>[Return Contents](#0)
# <hr>

# In[ ]:


mySeries.drop(['a'])

# ### **Drop an observation (row)** <a id="82"></a>
# 
# <mark>[Return Contents](#0)
# <hr>

# In[ ]:


# df.drop(['2016-07-01', '2016-06-27'])

# ### **Drop a variable (column)** <a id="83"></a>
# 
# <mark>[Return Contents](#0)
# <hr>
# Note: axis=1 denotes that we are referring to a column, not a row

# In[ ]:


# df.drop('Volume', axis=1)

# # **Convert Data Types** <a id="73"></a>
# 
# <mark>[Return Contents](#0)
# <hr>
# 
# ### **df.dtypes** <a id="74"></a>
# This code shows what data type of columns are. Boolean, int, float, object(String), date and categorical.

# In[ ]:


df.dtypes

# ### **df['columnName'] = df['columnName'].astype('dataType')** <a id="75"></a>
# This code convert the column we specify into the data type we specify.
# 
# <mark>[Return Contents](#0)
# <hr>

# In[ ]:


df.Date.astype('category').dtypes
# OR Convert Datetime
# df.Date= pd.to_datetime(df.Date)

# ### **pd.melt(frame=dataFrameName,id_vars = 'columnName', value_vars= ['columnName'])** <a id="76"></a>
# This code is confusing, so lets look at the example.
# 
# <mark>[Return Contents](#0)
# <hr>

# In[ ]:


df_new = df.head()
melted = pd.melt(frame=df_new,id_vars = 'Date', value_vars= ['Low'])
melted

# # **Apply Function** <a id="77"></a>
# 
# <mark>[Return Contents](#0)
# <hr>
# 
# ### **Method 1** <a id="78"></a>

# In[ ]:


def examples(x):   #create a function
    return x*2

df.Open.apply(examples).head()  #use the function with apply() 

# ### **Method 2** <a id="79"></a>

# In[ ]:


df.Open.apply(lambda x: x*2).head()

# # **Utilities Code** <a id="80"></a>
# 
# <mark>[Return Contents](#0)
# <hr>

# In[ ]:


# pd.get_option OR pd.set_option
# pd.reset_option("^display")

# pd.reset_option("display.max_rows")
# pd.get_option("display.max_rows")
# pd.set_option("max_r",102)                 -> specifies the maximum number of rows to display.
# pd.options.display.max_rows = 999          -> specifies the maximum number of rows to display.

# pd.get_option("display.max_columns")
# pd.options.display.max_columns = 999       -> specifies the maximum number of columns to display.

# pd.set_option('display.width', 300)

# pd.set_option('display.max_columns', 300)  -> specifies the maximum number of rows to display.
# pd.set_option('display.max_colwidth', 500) -> specifies the maximum number of columns to display. 

# pd.get_option('max_colwidth')
# pd.set_option('max_colwidth',40)
# pd.reset_option('max_colwidth')

# pd.get_option('max_info_columns')
# pd.set_option('max_info_columns', 11)
# pd.reset_option('max_info_columns')

# pd.get_option('max_info_rows')
# pd.set_option('max_info_rows', 11)
# pd.reset_option('max_info_rows')

# pd.set_option('precision',7) -> sets the output display precision in terms of decimal places. This is only a suggestion.
# OR
# pd.set_option('display.precision',3)

# pd.set_option('chop_threshold', 0) -> sets at what level pandas rounds to zero when it displays a Series of DataFrame. This setting does not change the precision at which the number is stored.
# pd.reset_option('chop_threshold') 
