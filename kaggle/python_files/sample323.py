#!/usr/bin/env python
# coding: utf-8

#  # <div align="center">  The Data Scientist’s Toolbox Tutorial - 1 </div>
#  ### <div align="center"><b>CLEAR DATA. MADE MODEL.</b></div>
#  
# <div style="text-align:center">last update: <b>01/02/2019</b></div>

# >###### You may  be interested have a look at 10 Steps to Become a Data Scientist: 
# 
# 1. <font color="red">You are in the first step</font>
# 2. [Python Packages](https://www.kaggle.com/mjbahmani/the-data-scientist-s-toolbox-tutorial-2)
# 3. [Mathematics and Linear Algebra](https://www.kaggle.com/mjbahmani/linear-algebra-for-data-scientists)
# 4. [Programming &amp; Analysis Tools](https://www.kaggle.com/mjbahmani/20-ml-algorithms-15-plot-for-beginners)
# 5. [Big Data](https://www.kaggle.com/mjbahmani/a-data-science-framework-for-quora)
# 6. [Data visualization](https://www.kaggle.com/mjbahmani/top-5-data-visualization-libraries-tutorial)
# 7. [Data Cleaning](https://www.kaggle.com/mjbahmani/machine-learning-workflow-for-house-prices)
# 8. [How to solve a Problem?](https://www.kaggle.com/mjbahmani/the-data-scientist-s-toolbox-tutorial-2)
# 9. [Machine Learning](https://www.kaggle.com/mjbahmani/a-comprehensive-ml-workflow-with-python)
# 10. [Deep Learning](https://www.kaggle.com/mjbahmani/top-5-deep-learning-frameworks-tutorial)
# 
# ---------------------------------------------------------------------
# Fork, Run and Follow this kernel on GitHub:
# > ###### [ GitHub](https://github.com/mjbahmani/10-steps-to-become-a-data-scientist)
# 
# 
# -------------------------------------------------------------------------------------------------------------
#  **I hope you find this kernel helpful and some <font color="red">UPVOTES</font> would be very much appreciated**
#  
#  -----------
# 

#  <a id="top"></a> <br>
# **Notebook Content**
# 1. [Introduction](#1)
#     1. [Import](#11)
#     1. [Version](#12)
#     1. [setup](#13)
# 1. [Python](#2)
#     1. [Python Syntax compared to other programming languages](#21)
#     1. [Python: Basics](#22)
#         1. [Variables](#221)
#         1. [Operators](#222)
#     1. [Functions](#23)
#     1. [Types and Sequences](#24)
#     1. [More on Strings](#25)
#     1. [Reading and Writing CSV files](#26)
#     1. [Dates and Times](#27)
#     1. [Objects and map()](#28)
#     1. [Lambda and List Comprehensions](#29)
#     1. [OOP](#210)
#         1. [Inheritance](#2101)
#     1. [ Python JSON](#211)
#         1. [Convert from Python to JSON](#2111)
#     1. [Python PIP](#212)
#         1. [Install PIP](#2121)
#     1. [Python Try Except](#213)
#     1. [Python Iterators](#214)
#         1. [Looping Through an Iterator](#2141)
#     1. [Dictionary](#215)
#     1. [Tuples](#216)
#     1. [Set](#217)
#         1. [Add Items](#2171)
# 1. [Python Packages](#3)
#     1. [Numpy](#31)
#     1. [Pandas](#32)
# 1. [Courses](#4)
# 1. [Ebooks](#5)
# 1. [Conclusion](#6)
# 1. [References](#7)

# <a id="1"></a> <br>
# ## 1-Introduction
# In this kernel, we have a comprehensive tutorials for some important packages in python after that you can read my other kernels about machine learning.
# most of the code are based on some amazing course on internet for example  [data science python in Coursera](https://www.coursera.org/specializations/data-science-python). I have used plenty of references  to create this kernel for see all of the references please see [credits section](#7) and also Kaggle have great [learn section](https://www.kaggle.com/learn) that help you start your journey in data science

#  <img src="http://s8.picofile.com/file/8340141626/packages1.png"  height="420" width="420">
#  
#  ###### [Go to top](#top)

# <a id="11"></a> <br>
# ## 1-1 Import
# 
# > <font color="red"><b>Note</b></font>
# 
# >> 20 Python libraries you can’t live without : [pythontips](https://pythontips.com/2013/07/30/20-python-libraries-you-cant-live-without/)

# In[ ]:


from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import scipy.stats as stats
import plotly.plotly as py
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib
import warnings
import sklearn
import scipy
import json
import sys
import csv
import os

# <a id="12"></a> <br>
# ## 1-2 Version

# In[ ]:


print('matplotlib: {}'.format(matplotlib.__version__))
print('sklearn: {}'.format(sklearn.__version__))
print('scipy: {}'.format(scipy.__version__))
print('seaborn: {}'.format(sns.__version__))
print('pandas: {}'.format(pd.__version__))
print('numpy: {}'.format(np.__version__))
print('Python: {}'.format(sys.version))

# <a id="13"></a> <br>
# ##  1-3 Setup

# In[ ]:


warnings.filterwarnings('ignore')
sns.set(color_codes=True)
plt.style.available

# <a id="14"></a> <br>
# ## 1-4 Import DataSets

# In[ ]:


print(os.listdir("../input/"))

# In[ ]:


hp_train=pd.read_csv('../input/train.csv')

#  <a id="2"></a> <br>
# # 2-Python
# Python :
# 1. Python is an interpreted,
# 1. high-level
# 1. general-purpose programming language. 
# 1. Created by Guido van Rossum 
# 1. first released in 1991
# 1. Python has a design philosophy that emphasizes code readability, notably using significant whitespace.[6]
# 
# Applications for Python:
# 
# 1. Web and Internet Development
# 1. Scientific and Numeric
# 1. Education
# 1. Desktop GUIs
# 1. Software Development
# 1. Business Applications[7]
# 
# ###### [Go to top](#top)

# <a id="21"></a> <br>
# # 2-1  Basic
# 

# In[ ]:


import this

# <a id="211"></a> <br>
# ### 2-1-1 How to define a Variable?
# you can do not defne your variable in python

# In[ ]:


x = 2
y = 5
xy = 'Hey'

# <a id="212"></a> <br>
# ### 2-1-2 Operators

# > <font color="red"><b>Note</b></font>
# 
# >> 
# Python language supports the following types of operators. [tutorialspoint](https://www.tutorialspoint.com/python/python_basic_operators.htm)
# 
# * Arithmetic Operators
# * Comparison (Relational) Operators
# * Assignment Operators
# * Logical Operators
# * Bitwise Operators
# * Membership Operators
# * Identity Operators
# 
# | Symbol | Task Performed |
# |----|---|
# | +  | Addition |
# | -  | Subtraction |
# | /  | division |
# | %  | mod |
# | *  | multiplication |
# | //  | floor division |
# | **  | to the power of |
# 
# ### Relational Operators
# | Symbol | Task Performed |
# |----|---|
# | == | True, if it is equal |
# | !=  | True, if not equal to |
# | < | less than |
# | > | greater than |
# | <=  | less than or equal to |
# | >=  | greater than or equal to |
# ### Bitwise Operators
# | Symbol | Task Performed |
# |----|---|
# | &  | Logical And |
# | l  | Logical OR |
# | ^  | XOR |
# | ~  | Negate |
# | >>  | Right shift |
# | <<  | Left shift |
# 
# ###### [Go to top](#top)

# <a id="22"></a> <br>
# # 2-2 How to define a Functions?
# `add_numbers` is a function that takes two numbers and adds them together. and also you can[ check it on tutorialspoint](https://www.tutorialspoint.com/python/python_functions.htm)

# In[ ]:


def add_numbers(x, y):
    return x + y

add_numbers(1, 2)

# <a id="23"></a> <br>
# # 2-3 Types and Sequences

# <br>
# **type** fuction help you find type of object in python

# In[ ]:


type('This is a string')

# In[ ]:


type(None)

# In[ ]:


type(1)

# In[ ]:


type(1.0)

# In[ ]:


type(add_numbers)

# <a id="231"></a> <br>
# ### 2-3-1 Tuples
#  The differences between tuples and lists are, the tuples cannot be changed unlike lists and tuples use parentheses, whereas lists use square brackets.[ check it on tutorialspoint](https://www.tutorialspoint.com/python/python_tuples.htm)
# > <font color="red"><b>Note</b></font>
# >>  Tuples are an **immutable** data structure (cannot be altered).

# In[ ]:


tuple_sample = (1, 'a', 2, 'b')
type(tuple_sample)

# <a id="232"></a> <br>
# ### 2-3-2 Lists
# > <font color="red"><b>Note</b></font>
# 
# >> 
# Lists are a mutable data structure.

# In[ ]:


list_sample = [1, 'a', 2, 'b']
type(list_sample)

# <br>
# Use `append` to append an object to a list.

# In[ ]:


list_sample.append(3.3)
print(list_sample)

# <br>
# This is an example of how to** loop through** each item in the list.

# In[ ]:


for item in list_sample:
    print(item)

# <br>
# Or using the indexing operator:

# In[ ]:


i=0
while( i != len(list_sample) ):
    print(list_sample[i])
    i = i + 1

# <br>
# Now let's look at strings. Use bracket notation to slice a string.
# 
# ###### [Go to top](#top)

# In[ ]:


x = 'This is a string'
print(x[0]) #first character
print(x[0:1]) #first character, but we have explicitly set the end character
print(x[0:2]) #first two characters


# > <font color="red"><b>Note</b></font>
# 
# >> 
# **Dictionaries** associate keys with values.

# In[ ]:


x = {'MJ Bahmani': 'Mohamadjavad.bahmani@gmail.com', 'irmatlab': 'irmatlab.ir@gmail.com'}
x['MJ Bahmani'] # Retrieve a value by using the indexing operator


# <a id="25"></a> <br>
# # 2-5 Strings
# 
# ###### [Go to top](#top)

# In[ ]:


print('MJ' + str(2))

# <br>
# Python has a built in method for convenient string formatting.

# In[ ]:


sales_record = {
'price': 3.24,
'num_items': 4,
'person': 'MJ'}



# In[ ]:


type(sales_record)

# In[ ]:


sales_statement = '{} bought {} item(s) at a price of {} each for a total of {}'

print(sales_statement.format(sales_record['person'],
                             sales_record['num_items'],
                             sales_record['price'],
                             sales_record['num_items']*sales_record['price']))


# <a id="26"></a> <br>
# # 2-6 How read a CSV files?
# > <font color="red"><b>Note</b></font>
# 
# >> 
# Let's import our datafile train.csv [realpython](https://realpython.com/python-csv/)
# ###### [Go to top](#top)

# 

# In[ ]:


#there are many way to import a csv file

with open('../input/train.csv') as csvfile:
    house_train = list(csv.DictReader(csvfile))
    
house_train[:1] # The first three dictionaries in our list.

# In[ ]:


type(house_train)

# <br>
# `csv.Dictreader` has read in each row of our csv file as a dictionary. `len` shows that our list is comprised of 234 dictionaries.

# In[ ]:


len(house_train)

# <br>
# `keys` gives us the column names of our csv. 
# For house price data set, we print, all of the feature

# In[ ]:


house_train[0].keys()

# <br>
# How to do some math action on the data set

# In[ ]:


sum(float(d['SalePrice']) for d in house_train) / len(house_train)

# <br>
# Use `set` to return the unique values for the type of YearBuilt  in our dataset have.

# In[ ]:


YearBuilt = set(d['YearBuilt'] for d in house_train)
print(type(YearBuilt ))
len(YearBuilt)

# In[ ]:


type(house_train[0] )

# <a id="27"></a> <br>
# # 2-7 Dates and Times

# In[ ]:


# just memorize this library
import datetime as dt
import time as tm

# <br>
# `time` returns the current time in seconds since the Epoch. (April 2st, 2019)
# 
# ###### [Go to top](#top)

# In[ ]:


tm.time()

# <br>
# Convert the timestamp to datetime.

# In[ ]:


dtnow = dt.datetime.fromtimestamp(tm.time())
dtnow

# <br>
# Handy datetime attributes:

# In[ ]:


dtnow.year, dtnow.month, dtnow.day, dtnow.hour, dtnow.minute, dtnow.second # get year, month, day, etc.from a datetime

# <br>
# `timedelta` is a duration expressing the difference between two dates.

# In[ ]:


delta = dt.timedelta(days = 100) # create a timedelta of 100 days
delta

# <br>
# `date.today` returns the current local date.

# In[ ]:


today = dt.date.today()

# <a id="28"></a> <br>
# # 2-8 Objects and map()

# <br>
# An example of a class in python:

# In[ ]:


class Person:
    department = 'School of Information' #a class variable

    def set_name(self, new_name): #a method
        self.name = new_name
    def set_location(self, new_location):
        self.location = new_location

# In[ ]:


# define an object
person = Person()
# set value for the object
person.set_name('MJ Bahmani')
person.set_location('MI, Berlin, Germany')
print('{} live in {} and works in the department {}'.format(person.name, person.location, person.department))

# <a id="29"></a> <br>
# # 2-9 Lambda
# A lambda function is a small anonymous function.
# 
# A lambda function can take any number of arguments, but can only have one expression. [w3schools](https://www.w3schools.com/python/python_lambda.asp)

# In[ ]:


my_function = lambda a, b, c : a + b +c

# In[ ]:


my_function(1, 2, 3)

# <br>
# Let's **iterate** from 0 to 9 and return the even numbers. with range function.
# 
# ###### [Go to top](#top)

# In[ ]:


my_list = []
for number in range(0, 9):
    if number % 2 == 0:
        my_list.append(number)
my_list

# <a id="291"></a> <br>
# ### 2-9-1  List Comprehensions

# <br>
# Now the same thing but with list comprehension.

# In[ ]:


my_list = [number for number in range(0,10) if number % 2 == 0]
my_list

# <a id="210"></a> <br>
# # 2-10 OOP
# 1. **Class** − A user-defined prototype for an object that defines a set of attributes that characterize any object of the class. The attributes are data members (class variables and instance variables) and methods, accessed via dot notation.[tutorialspoint](https://www.tutorialspoint.com/python/python_classes_objects.htm)
# 
# 1. **Class variable** − A variable that is shared by all instances of a class. Class variables are defined within a class but outside any of the class's methods. Class variables are not used as frequently as instance variables are.
# 
# 1. **Data member** − A class variable or instance variable that holds data associated with a class and its objects.
# 
# 1. **Function overloading** − The assignment of more than one behavior to a particular function. The operation performed varies by the types of objects or arguments involved.
# 
# 1. **Instance variable** − A variable that is defined inside a method and belongs only to the current instance of a class.
# 
# 1. **Inheritance** − The transfer of the characteristics of a class to other classes that are derived from it.
# 
# 1. **Instance** − An individual object of a certain class. An object obj that belongs to a class Circle, for example, is an instance of the class Circle.
# 
# 1. **Instantiation** − The creation of an instance of a class.
# 
# 1. **Method** − A special kind of function that is defined in a class definition.
# 
# 1. **Object** − A unique instance of a data structure that's defined by its class. An object comprises both data members (class variables and instance variables) and methods.
# 
# 1. **Operator overloading** − The assignment of more than one function to a particular operator.[4]
# 
# ###### [Go to top](#top)

# In[ ]:


class FirstClass:
    test = 'test'
    def __init__(self,name,symbol):
        self.name = name
        self.symbol = symbol

# In[ ]:


eg3 = FirstClass('Three',3)

# In[ ]:


print (eg3.test, eg3.name)

# In[ ]:


class FirstClass:
    def __init__(self,name,symbol):
        self.name = name
        self.symbol = symbol
    def square(self):
        return self.symbol * self.symbol
    def cube(self):
        return self.symbol * self.symbol * self.symbol
    def multiply(self, x):
        return self.symbol * x

# In[ ]:


eg4 = FirstClass('Five',5)

# In[ ]:


print (eg4.square())
print (eg4.cube())

# In[ ]:


eg4.multiply(2)

# In[ ]:


FirstClass.multiply(eg4,2)

# <a id="2101"></a> <br>
# ### 2-10-1 Inheritance
# 
# There might be cases where a new class would have all the previous characteristics of an already defined class. So the new class can "inherit" the previous class and add it's own methods to it. This is called as inheritance.
# 
# Consider class SoftwareEngineer which has a method salary.[python-textbok.readthedocs](https://python-textbok.readthedocs.io/en/1.0/Classes.html)

# In[ ]:


class SoftwareEngineer:
    def __init__(self,name,age):
        self.name = name
        self.age = age
    def salary(self, value):
        self.money = value
        print (self.name,"earns",self.money)

# In[ ]:


a = SoftwareEngineer('Kartik',26)

# In[ ]:


a.salary(40000)

# In[ ]:


dir(SoftwareEngineer)

# In[ ]:


class Artist:
    def __init__(self,name,age):
        self.name = name
        self.age = age
    def money(self,value):
        self.money = value
        print (self.name,"earns",self.money)
    def artform(self, job):
        self.job = job
        print (self.name,"is a", self.job)

# In[ ]:


b = Artist('Nitin',20)

# In[ ]:


b.money(50000)
b.artform('Musician')

# In[ ]:


dir(Artist)

# <a id="211"></a> <br>
# ## 2-11 Python JSON
# 

# In[ ]:




# some JSON:
x =  '{ "name":"John", "age":30, "city":"New York"}'

# parse x:
y = json.loads(x)

# the result is a Python dictionary:
print(y["age"])

# <a id="2111"></a> <br>
# ## 2-11-1 Convert from Python to JSON
# sometimes you need to convert your object to o Json file.[w3schools](https://www.w3schools.com/python/python_json.asp)
# 1. JSON is a syntax for storing and exchanging data.
# 1. JSON is text, written with JavaScript object notation.

# In[ ]:


# a Python object (dict):
x = {
  "name": "John",
  "age": 30,
  "city": "New York"
}

# convert into JSON:
y = json.dumps(x)

# the result is a JSON string:
print(y)

# You can convert Python objects of the following types, into JSON strings:[w3schools](https://www.w3schools.com/python/python_json.asp)
# 
# * dict
# * list
# * tuple
# * string
# * int
# * float
# * True
# * False
# * None
# 
# ###### [Go to top](#top)

# In[ ]:


print(json.dumps({"name": "John", "age": 30}))
print(json.dumps(["apple", "bananas"]))
print(json.dumps(("apple", "bananas")))
print(json.dumps("hello"))
print(json.dumps(42))
print(json.dumps(31.76))
print(json.dumps(True))
print(json.dumps(False))
print(json.dumps(None))

# Convert a Python object containing all the legal data types:

# In[ ]:


x = {
  "name": "MJ",
  "age": 32,
  "married": True,
  "divorced": False,
  "children": ("Ann","Billy"),
  "pets": None,
  "cars": [
    {"model": "BMW 230", "mpg": 27.5},
    {"model": "Ford Edge", "mpg": 24.1}
  ]
}

print(json.dumps(x))

# <a id="212"></a> <br>
# ## 2-12 Python PIP
# 

# <a id="21"></a> <br>
# ### 2-12-1 What is a Package?
# A package contains all the files you need for a **module**.
# 
# **Modules** are Python code libraries you can include in your project.
# 
# ###### [Go to top](#top)

# <a id="2122"></a> <br>
# ### 2-12-2 Install PIP
# If you do not have PIP installed, you can download and install it from this page: https://pypi.org/project/pip/

# <a id="213"></a> <br>
# ## 2-13 Python Try Except
# The **try** block lets you test a block of code for errors.
# 
# The **except** block lets you handle the error.
# 
# The **finally** block lets you execute code, regardless of the result of the try- and except blocks.

# In[ ]:


try:
  print(x)
except NameError:
  print("Variable x is not defined")
except:
  print("Something else went wrong")

# In[ ]:


try:
  print(x)
except:
  print("Something went wrong")
finally:
  print("The 'try except' is finished")

# <a id="214"></a> <br>
# ## 2-14 Python Iterators
# An iterator is an object that contains a countable number of values.
# 
# An iterator is an object that can be iterated upon, meaning that you can traverse through all the values.
# 
# Technically, in Python, an iterator is an object which implements the iterator protocol, which consist of the methods __iter__() and __next__().
# ###### [Go to top](#top)

# Return a iterator from a tuple, and print each value:

# In[ ]:


mytuple = ("apple", "banana", "cherry")
myit = iter(mytuple)

print(next(myit))
print(next(myit))
print(next(myit))

# <a id="2141"></a> <br>
# ### 2- 14-1 Looping Through an Iterator
# 

# In[ ]:


mytuple = ("apple", "banana", "cherry")

for x in mytuple:
  print(x)

# <a id="215"></a> <br>
# ## 2- 15 Dictionary
# A **dictionary** is a collection which is **unordered, changeable and indexed**. In Python dictionaries are written with curly brackets, and they have **keys and values**.

# In[ ]:


thisdict =	{
  "brand": "Ford",
  "model": "Mustang",
  "year": 1964
}
print(thisdict)

# <a id="216"></a> <br>
# ## 2-16 Tuples
# A **tuple** is a collection which is **ordered and unchangeable**. In Python tuples are written with round brackets.
# 
# 

# In[ ]:


thistuple = ("apple", "banana", "cherry")
print(thistuple)

# <a id="217"></a> <br>
# ## 2-17 Set
# A set is a collection which is unordered and unindexed. In Python sets are written with curly brackets.
# ###### [Go to top](#top)

# In[ ]:


house_price = {"store", "apartment", "house"}
print(house_price)

# In[ ]:


house_price = {"store", "apartment", "house"}

for x in house_price:
  print(x)

# <a id="2171"></a> <br>
# ### 2-17-1 Add Items
# To add one item to a set use the add() method.
# 
# To add more than one item to a set use the update() method.
# ###### [Go to top](#top)

# In[ ]:


thisset = {"apple", "banana", "cherry"}

thisset.add("orange")

print(thisset)

# ## 2- 18 Python Clean Code Principles
# how to develop a clean code with python:[https://www.pythonforengineers.com/writing-great-code/](https://www.pythonforengineers.com/writing-great-code/)
# 1. Code is read a million times, written once.
# 1. Follow a coding standard.
# 1. Your code should be modular / independent.
# 1. Easily testable.
# 1. One statement per line.
# 1. Avoid Abusing Comments
# 1. Avoid Extremely Large Functions

# <a id="3"></a> <br>
# # 3- Python Packages
# * Numpy
# * Pandas
# * Matplotlib
# * Seaborn
# * Sklearn
# * plotly

# <a id="31"></a> <br>
# # 3-1 NumPy
# 
# For Reading this section please fork and run the following kernel:
# 
# [The Data Scientist’s Toolbox Tutorial - 2](https://www.kaggle.com/mjbahmani/the-data-scientist-s-toolbox-tutorial-2/)

# <a id="41"></a> <br>
# # 3-2 Pandas
# 
# For Reading this section please fork and run the following kernel:
# 
# [The Data Scientist’s Toolbox Tutorial - 2](https://www.kaggle.com/mjbahmani/the-data-scientist-s-toolbox-tutorial-2/)

# <a id="4"></a> <br>
# ## 4- Courses
# There are a lot of online courses that can help you develop your knowledge, here I have just  listed some of them:
# 
# 1. [Machine Learning Certification by Stanford University (Coursera)](https://www.coursera.org/learn/machine-learning/)
# 
# 2. [Machine Learning A-Z™: Hands-On Python & R In Data Science (Udemy)](https://www.udemy.com/machinelearning/)
# 
# 3. [Deep Learning Certification by Andrew Ng from deeplearning.ai (Coursera)](https://www.coursera.org/specializations/deep-learning)
# 
# 4. [Python for Data Science and Machine Learning Bootcamp (Udemy)](Python for Data Science and Machine Learning Bootcamp (Udemy))
# 
# 5. [Mathematics for Machine Learning by Imperial College London](https://www.coursera.org/specializations/mathematics-machine-learning)
# 
# 6. [Deep Learning A-Z™: Hands-On Artificial Neural Networks](https://www.udemy.com/deeplearning/)
# 
# 7. [Complete Guide to TensorFlow for Deep Learning Tutorial with Python](https://www.udemy.com/complete-guide-to-tensorflow-for-deep-learning-with-python/)
# 
# 8. [Data Science and Machine Learning Tutorial with Python – Hands On](https://www.udemy.com/data-science-and-machine-learning-with-python-hands-on/)
# 
# 9. [Machine Learning Certification by University of Washington](https://www.coursera.org/specializations/machine-learning)
# 
# 10. [Data Science and Machine Learning Bootcamp with R](https://www.udemy.com/data-science-and-machine-learning-bootcamp-with-r/)
# 11. [Creative Applications of Deep Learning with TensorFlow](https://www.class-central.com/course/kadenze-creative-applications-of-deep-learning-with-tensorflow-6679)
# 12. [Neural Networks for Machine Learning](https://www.class-central.com/mooc/398/coursera-neural-networks-for-machine-learning)
# 13. [Practical Deep Learning For Coders, Part 1](https://www.class-central.com/mooc/7887/practical-deep-learning-for-coders-part-1)
# 14. [Machine Learning](https://www.cs.ox.ac.uk/teaching/courses/2014-2015/ml/index.html)

#  <a id="5"></a> <br>
# ## 5 Ebooks
# If you love reading , here is **10 free machine learning books**
# 1. [Probability and Statistics for Programmers](http://www.greenteapress.com/thinkstats/)
# 2. [Bayesian Reasoning and Machine Learning](http://web4.cs.ucl.ac.uk/staff/D.Barber/textbook/091117.pdf)
# 2. [An Introduction to Statistical Learning](http://www-bcf.usc.edu/~gareth/ISL/)
# 2. [Understanding Machine Learning](http://www.cs.huji.ac.il/~shais/UnderstandingMachineLearning/index.html)
# 2. [A Programmer’s Guide to Data Mining](http://guidetodatamining.com/)
# 2. [Mining of Massive Datasets](http://infolab.stanford.edu/~ullman/mmds/book.pdf)
# 2. [A Brief Introduction to Neural Networks](http://www.dkriesel.com/_media/science/neuronalenetze-en-zeta2-2col-dkrieselcom.pdf)
# 2. [Deep Learning](http://www.deeplearningbook.org/)
# 2. [Natural Language Processing with Python](https://www.researchgate.net/publication/220691633_Natural_Language_Processing_with_Python)
# 2. [Machine Learning Yearning](http://www.mlyearning.org/)

# 
# <a id="6"></a> <br>
# ## 6 -Conclusion
# You have got an introduction to the python and ideas in the **The Data Scientist’s Toolbox Tutorial - 1**. I hope,  you have had fun with it also, I want to hear your voice to update this kernel. in addition, there is the new and second version of the kernel that introduces the next packages. to continue please click on [**The Data Scientist’s Toolbox Tutorial - 2**](https://www.kaggle.com/mjbahmani/the-data-scientist-s-toolbox-tutorial-2)

# 
# 
# ---------------------------------------------------------------------
# Fork, Run and Follow this kernel on GitHub:
# > ###### [ GitHub](https://github.com/mjbahmani/10-steps-to-become-a-data-scientist)
# 
# 
# -------------------------------------------------------------------------------------------------------------
#  **I hope you find this kernel helpful and some <font color="red">UPVOTES</font> would be very much appreciated.**
#  
#  -----------
# 

# <a id="7"></a> <br>
# ## 7- References & Credits
# 1. [data science python in Coursera](https://www.coursera.org/specializations/data-science-python)
# 1. [plot.ly](https://plot.ly/python/offline/)
# 1. [tutorialspoint](https://www.tutorialspoint.com/python/python_classes_objects.htm)
# 1. [Top 28 Cheat Sheets for Machine Learning](https://www.analyticsvidhya.com/blog/2017/02/top-28-cheat-sheets-for-machine-learning-data-science-probability-sql-big-data/)
# 1. [Python](https://en.wikipedia.org/wiki/Python_(programming_language))
# 1. [python-app](https://www.python.org/about/apps/)
# 1. [tutorialspoint-operators](https://www.tutorialspoint.com/python/python_basic_operators.htm)
# 1. [w3schools](https://www.w3schools.com/python/python_lambda.asp)
# 1. [Machine Learning Certification by Stanford University (Coursera)](https://www.coursera.org/learn/machine-learning/)
# 1. [Machine Learning A-Z™: Hands-On Python & R In Data Science (Udemy)](https://www.udemy.com/machinelearning/)
# 1. [Deep Learning Certification by Andrew Ng from deeplearning.ai (Coursera)](https://www.coursera.org/specializations/deep-learning)
# 1. [Python for Data Science and Machine Learning Bootcamp (Udemy)](Python for Data Science and Machine Learning Bootcamp (Udemy))
# 1. [Mathematics for Machine Learning by Imperial College London](https://www.coursera.org/specializations/mathematics-machine-learning)
# 1. [Deep Learning A-Z™: Hands-On Artificial Neural Networks](https://www.udemy.com/deeplearning/)
# 1. [Complete Guide to TensorFlow for Deep Learning Tutorial with Python](https://www.udemy.com/complete-guide-to-tensorflow-for-deep-learning-with-python/)
# 1. [Data Science and Machine Learning Tutorial with Python – Hands On](https://www.udemy.com/data-science-and-machine-learning-with-python-hands-on/)
# 1. [Machine Learning Certification by University of Washington](https://www.coursera.org/specializations/machine-learning)
# 1. [Data Science and Machine Learning Bootcamp with R](https://www.udemy.com/data-science-and-machine-learning-bootcamp-with-r/)
# 1. [Creative Applications of Deep Learning with TensorFlow](https://www.class-central.com/course/kadenze-creative-applications-of-deep-learning-with-tensorflow-6679)
# 1. [Neural Networks for Machine Learning](https://www.class-central.com/mooc/398/coursera-neural-networks-for-machine-learning)
# 1. [Practical Deep Learning For Coders, Part 1](https://www.class-central.com/mooc/7887/practical-deep-learning-for-coders-part-1)
# 1. [Machine Learning](https://www.cs.ox.ac.uk/teaching/courses/2014-2015/ml/index.html)
# 1. [writing-great-code](https://www.pythonforengineers.com/writing-great-code/)
# 
# ###### [Go to top](#top)

# Go to first step: [Course Home Page](https://www.kaggle.com/mjbahmani/10-steps-to-become-a-data-scientist)
# 
# Go to next step : [Titanic](https://www.kaggle.com/mjbahmani/a-comprehensive-ml-workflow-with-python)
