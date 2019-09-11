#!/usr/bin/env python
# coding: utf-8

# # <div style="text-align: center">A Data Science Framework for Quora </div>
# ### <div align="center"><b>Quite Practical and Far from any Theoretical Concepts</b></div>
# <img src='http://s9.picofile.com/file/8342477368/kq.png'>
# <div style="text-align:center">last update: <b>19/01/2019</b></div>
# 
# You can Fork and Run this kernel on **Github**:
# > ###### [ GitHub](https://github.com/mjbahmani/10-steps-to-become-a-data-scientist)
# 

#  <a id="1"></a> <br>
# ## 1- Introduction
# <font color="red">Quora</font> has defined a competition in **Kaggle**. A realistic and attractive data set for data scientists.
# on this notebook, I will provide a **comprehensive** approach to solve Quora classification problem for **beginners**.
# 
# I am open to getting your feedback for improving this **kernel**.

# <a id="top"></a> <br>
# ## Notebook  Content
# 1. [Introduction](#1)
# 1. [Data Science Workflow for Quora](#2)
# 1. [Problem Definition](#3)
#     1. [Business View](#31)
#         1. [Real world Application Vs Competitions](#311)
#     1. [What is a insincere question?](#32)
#     1. [How can we find insincere question?](#33)
# 1. [Problem feature](#4)
#     1. [Aim](#41)
#     1. [Variables](#42)
#     1. [ Inputs & Outputs](#43)
# 1. [Select Framework](#5)
#     1. [Import](#51)
#     1. [Version](#52)
#     1. [Setup](#53)
# 1. [Exploratory data analysis](#6)
#     1. [Data Collection](#61)
#         1. [Features](#611)
#         1. [Explorer Dataset](#612)
#     1. [Data Cleaning](#62)
#     1. [Data Preprocessing](#63)
#         1. [Is data set imbalance?](#631)
#         1. [Some Feature Engineering](#632)
#     1. [Data Visualization](#64)
#         1. [countplot](#641)
#         1. [pie plot](#642)
#         1. [Histogram](#643)
#         1. [violin plot](#645)
#         1. [kdeplot](#646)
# 1. [Apply Learning](#7)
# 1. [Conclusion](#8)
# 1. [References](#9)

# -------------------------------------------------------------------------------------------------------------
# 
#  **I hope you find this kernel helpful and some <font color="red"><b>UPVOTES</b></font> would be very much appreciated**
#  
#  -----------

# <a id="2"></a> <br>
# ## 2- A Data Science Workflow for Quora
# Of course, the same solution can not be provided for all problems, so the best way is to create a **general framework** and adapt it to new problem.
# 
# **You can see my workflow in the below image** :
# 
#  <img src="http://s8.picofile.com/file/8342707700/workflow2.png"  />
# 
# **You should feel free	to	adjust 	this	checklist 	to	your needs**
# ###### [Go to top](#top)

# <a id="3"></a> <br>
# ## 3- Problem Definition
# I think one of the important things when you start a new machine learning project is Defining your problem. that means you should understand business problem.( **Problem Formalization**)
# > **we will be predicting whether a question asked on Quora is sincere or not.**
# <a id="31"></a> <br>
# ## 3-1 About Quora
# Quora is a platform that empowers people to learn from each other. On Quora, people can ask questions and connect with others who contribute unique insights and quality answers. A key challenge is to weed out insincere questions -- those founded upon false premises, or that intend to make a statement rather than look for helpful answers.
# <a id="32"></a> <br>
# ## 3-2 Business View 
# An existential problem for any major website today is how to handle toxic and divisive content. **Quora** wants to tackle this problem head-on to keep their platform a place where users can feel safe sharing their knowledge with the world.
# 
# **Quora** is a platform that empowers people to learn from each other. On Quora, people can ask questions and connect with others who contribute unique insights and quality answers. A key challenge is to weed out insincere questions -- those founded upon false premises, or that intend to make a statement rather than look for helpful answers.
# 
# In this kernel, I will develop models that identify and flag insincere questions.we Help Quora uphold their policy of “Be Nice, Be Respectful” and continue to be a place for sharing and growing the world’s knowledge.
# <a id="321"></a> <br>
# ### 3-2-1 Real world Application Vs Competitions
# Just a simple comparison between real-world apps with competitions:
# <img src="http://s9.picofile.com/file/8339956300/reallife.png" height="600" width="500" />
# <a id="33"></a> <br>
# ## 3-3 What is a insincere question?
# Is defined as a question intended to make a **statement** rather than look for **helpful answers**.
# <img src='http://s8.picofile.com/file/8342711526/Quora_moderation.png'>
# <a id="34"></a> <br>
# ## 3-4 How can we find insincere question?
# Some characteristics that can signify that a question is insincere:
# 
# 1. **Has a non-neutral tone**
#     1. Has an exaggerated tone to underscore a point about a group of people
#     1. Is rhetorical and meant to imply a statement about a group of people
# 1. **Is disparaging or inflammatory**
#     1. Suggests a discriminatory idea against a protected class of people, or seeks confirmation of a stereotype
#     1. Makes disparaging attacks/insults against a specific person or group of people
#     1. Based on an outlandish premise about a group of people
#     1. Disparages against a characteristic that is not fixable and not measurable
# 1. **Isn't grounded in reality**
#     1. Based on false information, or contains absurd assumptions
#     1. Uses sexual content (incest, bestiality, pedophilia) for shock value, and not to seek genuine answers
#     ###### [Go to top](#top)

# <a id="4"></a> <br>
# ## 4- Problem Feature
# Problem Definition has three steps that have illustrated in the picture below:
# 
# 1. Aim
# 1. Variable
# 1. Inputs & Outputs
# 
# 
# 
# 
# 
# <a id="41"></a> <br>
# ### 4-1 Aim
# We will be predicting whether a question asked on Quora is **sincere** or not.
# 
# 
# <a id="42"></a> <br>
# ### 4-2 Variables
# 
# 1. qid - unique question identifier
# 1. question_text - Quora question text
# 1. target - a question labeled "insincere" has a value of 1, otherwise 0
# 
# <a id="43"></a> <br>
# ### 4-3 Inputs & Outputs
# we use train.csv and test.csv as Input and we should upload a  submission.csv as Output
# 
# 
# **<< Note >>**
# > You must answer the following question:
# How does your company expect to use and benefit from **your model**.
# ###### [Go to top](#top)

# <a id="5"></a> <br>
# ## 5- Select Framework
# After problem definition and problem feature, we should select our framework to solve the problem.
# What we mean by the framework is that  the programming languages you use and by what modules the problem will be solved.
# 
# <a id="51"></a> <br>
# ## 5-1 Python Deep Learning Packages
# <img src='https://cdn-images-1.medium.com/max/800/1*dYjDEI0mLpsCOySKUuX1VA.png' width=500 height=500>
# *State of open source deep learning frameworks in 2017*
# 1. **keras**[11]
# >Well known for being minimalistic, the Keras neural network library (with a supporting interface of Python) supports both convolutional and recurrent networks that are capable of running on either TensorFlow or Theano. The library is written in Python and was developed keeping quick experimentation as its USP.
# 1. **TensorFlow**
# > TensorFlow is arguably one of the best deep learning frameworks and has been adopted by several giants such as Airbus, Twitter, IBM, and others mainly due to its highly flexible system architecture.
# 1. **Caffe**
# > Caffe is a deep learning framework that is supported with interfaces like C, C++, Python, and MATLAB as well as the command line interface. It is well known for its speed and transposability and its applicability in modeling convolution neural networks (CNN).
# 1. **Microsoft Cognitive Toolkit/CNTK**
# > Popularly known for easy training and the combination of popular model types across servers, the Microsoft Cognitive Toolkit (previously known as CNTK) is an open-source deep learning framework to train deep learning models. It performs efficient convolution neural networks and training for image, speech, and text-based data. Similar to Caffe, it is supported by interfaces such as Python, C++, and the command line interface.
# 1. **Torch/PyTorch**
# > Torch is a scientific computing framework that offers wide support for machine learning algorithms. It is a Lua-based deep learning framework and is used widely amongst industry giants such as Facebook, Twitter, and Google. It employs CUDA along with C/C++ libraries for processing and was basically made to scale the production of building models and provide overall flexibility.
# 1. **MXNet**
# > Designed specifically for the purpose of high efficiency, productivity, and flexibility, MXNet(pronounced as mix-net) is a deep learning framework supported by Python, R, C++, and Julia.
# 1. **Chainer**
# >Highly powerful, dynamic and intuitive, Chainer is a Python-based deep learning framework for neural networks that is designed by the run strategy. Compared to other frameworks that use the same strategy, you can modify the networks during runtime, allowing you to execute arbitrary control flow statements.
# 1. **Deeplearning4j**
# >Parallel training through iterative reduce, microservice architecture adaptation, and distributed CPUs and GPUs are some of the salient features of the Deeplearning4j deep learning framework. It is developed in Java as well as Scala and supports other JVM languages, too.
# 1. **Theano**
# >Theano is beautiful. Without Theano, we wouldn’t have anywhere near the amount of deep learning libraries (specifically in Python) that we do today. In the same way that without NumPy, we couldn’t have SciPy, scikit-learn, and scikit-image, the same can be said about Theano and higher-level abstractions of deep learning.
# 1. **Lasagne**
# >Lasagne is a lightweight library used to construct and train networks in Theano. The key term here is lightweight — it is not meant to be a heavy wrapper around Theano like Keras is. While this leads to your code being more verbose, it does free you from any restraints, while still giving you modular building blocks based on Theano.
# 1. **PaddlePaddle**
# >PaddlePaddle (PArallel Distributed Deep LEarning) is an easy-to-use, efficient, flexible and scalable deep learning platform, which is originally developed by Baidu scientists and engineers for the purpose of applying deep learning to many products at Baidu.
# 
# ###### [Go to top](#top)

# <a id="52"></a> <br>
# ## 5-2 Import

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from wordcloud import WordCloud as wc
from nltk.corpus import stopwords
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
from pandas import get_dummies
import matplotlib as mpl
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib
import warnings
import sklearn
import string
import scipy
import numpy
import nltk
import json
import sys
import csv
import os

# <a id="53"></a> <br>
# ## 5-3 version

# In[ ]:


print('matplotlib: {}'.format(matplotlib.__version__))
print('sklearn: {}'.format(sklearn.__version__))
print('scipy: {}'.format(scipy.__version__))
print('seaborn: {}'.format(sns.__version__))
print('pandas: {}'.format(pd.__version__))
print('numpy: {}'.format(np.__version__))
print('Python: {}'.format(sys.version))


# <a id="54"></a> <br>
# ## 5-4 Setup
# 
# A few tiny adjustments for better **code readability**

# In[ ]:


sns.set(style='white', context='notebook', palette='deep')
pylab.rcParams['figure.figsize'] = 12,8
warnings.filterwarnings('ignore')
mpl.style.use('ggplot')
sns.set_style('white')

# <a id="55"></a> <br>
# ## 5-5 NLTK
# In this kernel, we use the NLTK library So, before we begin the next step, we will first introduce this library.
# The Natural Language Toolkit (NLTK) is one of the leading platforms for working with human language data and Python, the module NLTK is used for natural language processing. NLTK is literally an acronym for Natural Language Toolkit. with it you can tokenizing words and sentences.
# NLTK is a library of Python that can mine (scrap and upload data) and analyse very large amounts of textual data using computational methods.
# <img src='https://arts.unimelb.edu.au/__data/assets/image/0005/2735348/nltk.jpg' width=300 height=300>

# In[ ]:


from nltk.tokenize import sent_tokenize, word_tokenize
 
data = "All work and no play makes jack a dull boy, all work and no play"
print(word_tokenize(data))

# <a id="551"></a> <br>
# All of them are words except the comma. Special characters are treated as separate tokens.
# 
# ## 5-5-1 Tokenizing sentences
# The same principle can be applied to sentences. Simply change the to sent_tokenize()
# We have added two sentences to the variable data:

# In[ ]:


from nltk.tokenize import sent_tokenize, word_tokenize
 
data = "All work and no play makes jack dull boy. All work and no play makes jack a dull boy."
print(sent_tokenize(data))

# <a id="552"></a> <br>
# ## 5-5-2 NLTK and arrays
# If you wish to you can store the words and sentences in arrays

# In[ ]:


from nltk.tokenize import sent_tokenize, word_tokenize
 
data = "All work and no play makes jack dull boy. All work and no play makes jack a dull boy."
 
phrases = sent_tokenize(data)
words = word_tokenize(data)
 
print(phrases)
print(words)

# <a id="553"></a> <br>
# ## 5-5-3 NLTK stop words
# Stop words are basically a set of commonly used words in any language, not just English. The reason why stop words are critical to many applications is that, if we remove the words that are very commonly used in a given language, we can focus on the important words instead.[12]

# In[ ]:


from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
 
data = "All work and no play makes jack dull boy. All work and no play makes jack a dull boy."
stopWords = set(stopwords.words('english'))
words = word_tokenize(data)
wordsFiltered = []
 
for w in words:
    if w not in stopWords:
        wordsFiltered.append(w)
 
print(wordsFiltered)

# A module has been imported:
# 
# 

# In[ ]:


from nltk.corpus import stopwords


# We get a set of English stop words using the line:
# 
# 

# In[ ]:


stopWords = set(stopwords.words('english'))


# The returned list stopWords contains 153 stop words on my computer.
# You can view the length or contents of this array with the lines:

# In[ ]:


print(len(stopWords))
print(stopWords)

# We create a new list called wordsFiltered which contains all words which are not stop words.
# To create it we iterate over the list of words and only add it if its not in the stopWords list.

# In[ ]:


for w in words:
    if w not in stopWords:
        wordsFiltered.append(w)

# <a id="554"></a> <br>
# ## 5-5-4 NLTK – stemming
# Start by defining some words:

# In[ ]:


words = ["game","gaming","gamed","games"]


# In[ ]:


from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize

# And stem the words in the list using:

# In[ ]:


from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize

words = ["game","gaming","gamed","games"]
ps = PorterStemmer()
 
for word in words:
    print(ps.stem(word))

# <a id="555"></a> <br>
# ## 5-5-5  NLTK speech tagging
# The module NLTK can automatically tag speech.
# Given a sentence or paragraph, it can label words such as verbs, nouns and so on.
# 
# NLTK – speech tagging example
# The example below automatically tags words with a corresponding class.

# In[ ]:


import nltk
from nltk.tokenize import PunktSentenceTokenizer
 
document = 'Whether you\'re new to programming or an experienced developer, it\'s easy to learn and use Python.'
sentences = nltk.sent_tokenize(document)   
for sent in sentences:
    print(nltk.pos_tag(nltk.word_tokenize(sent)))

# We can filter this data based on the type of word:

# In[ ]:


import nltk
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer
 
document = 'Today the Netherlands celebrates King\'s Day. To honor this tradition, the Dutch embassy in San Francisco invited me to'
sentences = nltk.sent_tokenize(document)   
 
data = []
for sent in sentences:
    data = data + nltk.pos_tag(nltk.word_tokenize(sent))
 
for word in data: 
    if 'NNP' in word[1]: 
        print(word)

# In[ ]:


sns.set(style='white', context='notebook', palette='deep')
pylab.rcParams['figure.figsize'] = 12,8
warnings.filterwarnings('ignore')
mpl.style.use('ggplot')
sns.set_style('white')

# <a id="556"></a> <br>
# ## 5-5-6 Natural Language Processing – prediction
# We can use natural language processing to make predictions. Example: Given a product review, a computer can predict if its positive or negative based on the text. In this article you will learn how to make a prediction program based on natural language processing.

# <a id="55561"></a> <br>
# ### 5-5-5-6-1  nlp prediction example
# Given a name, the classifier will predict if it’s a male or female.
# 
# To create our analysis program, we have several steps:
# 
# 1. Data preparation
# 1. Feature extraction
# 1. Training
# 1. Prediction
# 1. Data preparation
# The first step is to prepare data. We use the names set included with nltk.

# In[ ]:


from nltk.corpus import names
 
# Load data and training 
names = ([(name, 'male') for name in names.words('male.txt')] + 
	 [(name, 'female') for name in names.words('female.txt')])

# This dataset is simply a collection of tuples. To give you an idea of what the dataset looks like:

# In[ ]:


[(u'Aaron', 'male'), (u'Abbey', 'male'), (u'Abbie', 'male')]
[(u'Zorana', 'female'), (u'Zorina', 'female'), (u'Zorine', 'female')]

# You can define your own set of tuples if you wish, its simply a list containing many tuples.
# 
# Feature extraction
# Based on the dataset, we prepare our feature. The feature we will use is the last letter of a name:
# We define a featureset using:

# featuresets = [(gender_features(n), g) for (n,g) in names]
# and the features (last letters) are extracted using:

# In[ ]:


def gender_features(word): 
    return {'last_letter': word[-1]}

# Training and prediction
# We train and predict using:

# In[ ]:


import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import names
 
def gender_features(word): 
    return {'last_letter': word[-1]} 
 
# Load data and training 
names = ([(name, 'male') for name in names.words('male.txt')] + 
	 [(name, 'female') for name in names.words('female.txt')])
 
featuresets = [(gender_features(n), g) for (n,g) in names] 
train_set = featuresets
classifier = nltk.NaiveBayesClassifier.train(train_set) 
 
# Predict
print(classifier.classify(gender_features('Frank')))

# If you want to give the name during runtime, change the last line to:
# 

# In[ ]:


# Predict, you can change name
name = 'Sarah'
print(classifier.classify(gender_features(name)))

# <a id="6"></a> <br>
# ## 6- EDA
#  In this section, you'll learn how to use graphical and numerical techniques to begin uncovering the structure of your data. 
#  
# * Which variables suggest interesting relationships?
# * Which observations are unusual?
# * Analysis of the features!
# 
# By the end of the section, you'll be able to answer these questions and more, while generating graphics that are both insightful and beautiful.  then We will review analytical and statistical operations:
# 
# 1. Data Collection
# 1. Visualization
# 1. Data Cleaning
# 1. Data Preprocessing
# <img src="http://s9.picofile.com/file/8338476134/EDA.png" width=400 height=400>
# 
#  ###### [Go to top](#top)

# <a id="61"></a> <br>
# ## 6-1 Data Collection
# **Data collection** is the process of gathering and measuring data, information or any variables of interest in a standardized and established manner that enables the collector to answer or test hypothesis and evaluate outcomes of the particular collection.[techopedia]
# 
# I start Collection Data by the training and testing datasets into **Pandas DataFrames**.
# ###### [Go to top](#top)

# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

# **<< Note 1 >>**
# 
# * Each **row** is an observation (also known as : sample, example, instance, record).
# * Each **column** is a feature (also known as: Predictor, attribute, Independent Variable, input, regressor, Covariate).
# ###### [Go to top](#top)

# In[ ]:


train.sample(1) 

# In[ ]:


test.sample(1) 

# Or you can use others command to explorer dataset, such as 

# In[ ]:


train.tail(1)

# <a id="611"></a> <br>
# ## 6-1-1 Features
# Features can be from following types:
# * numeric
# * categorical
# * ordinal
# * datetime
# * coordinates
# 
# Find the type of features in **Qoura dataset**?!
# 
# For getting some information about the dataset you can use **info()** command.

# In[ ]:


print(train.info())

# In[ ]:


print(test.info())

# <a id="612"></a> <br>
# ## 6-1-2 Explorer Dataset
# 1- Dimensions of the dataset.
# 
# 2- Peek at the data itself.
# 
# 3- Statistical summary of all attributes.
# 
# 4- Breakdown of the data by the class variable.
# 
# Don’t worry, each look at the data is **one command**. These are useful commands that you can use again and again on future projects.
# ###### [Go to top](#top)

# In[ ]:


# shape for train and test
print('Shape of train:',train.shape)
print('Shape of test:',test.shape)

# In[ ]:


#columns*rows
train.size

# After loading the data via **pandas**, we should checkout what the content is, description and via the following:

# In[ ]:


type(train)

# In[ ]:


type(test)

# In[ ]:


train.describe()

# To pop up 5 random rows from the data set, we can use **sample(5)**  function and find the type of features.

# In[ ]:


train.sample(5) 

# <a id="62"></a> <br>
# ## 6-2 Data Cleaning
# When dealing with real-world data, dirty data is the norm rather than the exception. We continuously need to predict correct values, impute missing ones, and find links between various data artefacts such as schemas and records. We need to stop treating data cleaning as a piecemeal exercise (resolving different types of errors in isolation), and instead leverage all signals and resources (such as constraints, available statistics, and dictionaries) to accurately predict corrective actions.
# 
# The primary goal of data cleaning is to detect and remove errors and **anomalies** to increase the value of data in analytics and decision making. While it has been the focus of many researchers for several years, individual problems have been addressed separately. These include missing value imputation, outliers detection, transformations, integrity constraints violations detection and repair, consistent query answering, deduplication, and many other related problems such as profiling and constraints mining.[4]
# ###### [Go to top](#top)

# How many NA elements in every column!!
# 
# Good news, it is Zero!
# 
# To check out how many null info are on the dataset, we can use **isnull().sum()**.

# In[ ]:


train.isnull().sum()

# But if we had , we can just use **dropna()**(be careful sometimes you should not do this!)

# In[ ]:


# remove rows that have NA's
print('Before Droping',train.shape)
train = train.dropna()
print('After Droping',train.shape)

# 
# We can get a quick idea of how many instances (rows) and how many attributes (columns) the data contains with the shape property.

# To print dataset **columns**, we can use columns atribute.

# In[ ]:


train.columns

# You see number of unique item for Target  with command below:

# In[ ]:


train_target = train['target'].values

np.unique(train_target)

# YES, quora problem is a **binary classification**! :)

# To check the first 5 rows of the data set, we can use head(5).

# In[ ]:


train.head(5) 

# Or to check out last 5 row of the data set, we use tail() function.

# In[ ]:


train.tail() 

# To give a **statistical summary** about the dataset, we can use **describe()**
# 

# In[ ]:


train.describe() 

# As you can see, the statistical information that this command gives us is not suitable for this type of data
# **describe() is more useful for numerical data sets**

# <a id="63"></a> <br>
# ## 6-3 Data Preprocessing
# **Data preprocessing** refers to the transformations applied to our data before feeding it to the algorithm.
#  
# Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
# there are plenty of steps for data preprocessing and we just listed some of them in general(Not just for Quora) :
# 1. removing Target column (id)
# 1. Sampling (without replacement)
# 1. Making part of iris unbalanced and balancing (with undersampling and SMOTE)
# 1. Introducing missing values and treating them (replacing by average values)
# 1. Noise filtering
# 1. Data discretization
# 1. Normalization and standardization
# 1. PCA analysis
# 1. Feature selection (filter, embedded, wrapper)
# 1. Etc.
# 
# What methods of preprocessing can we run on  Quora?! 
# ###### [Go to top](#top)

# **<< Note 2 >>**
# in pandas's data frame you can perform some query such as "where"

# In[ ]:


train.where(train ['target']==1).count()

# As you can see in the below in python, it is so easy perform some query on the dataframe:

# In[ ]:


train[train['target']>1]

# Some examples of questions that they are insincere

# In[ ]:


train[train['target']==1].head(5)

# <a id="631"></a> <br>
# ## 6-3-1 Is data set imbalance?
# 

# In[ ]:


train_target.mean()

# A large part of the data is unbalanced, but **how can we  solve it?**

# In[ ]:


train["target"].value_counts()
# data is imbalance

# **Imbalanced dataset** is relevant primarily in the context of supervised machine learning involving two or more classes. 
# 
# **Imbalance** means that the number of data points available for different the classes is different:
# If there are two classes, then balanced data would mean 50% points for each of the class. For most machine learning techniques, little imbalance is not a problem. So, if there are 60% points for one class and 40% for the other class, it should not cause any significant performance degradation. Only when the class imbalance is high, e.g. 90% points for one class and 10% for the other, standard optimization criteria or performance measures may not be as effective and would need modification.
# 
# 
# <img src='https://www.datascience.com/hs-fs/hubfs/imbdata.png?t=1542328336307&width=487&name=imbdata.png'>
# [Image source](http://api.ning.com/files/vvHEZw33BGqEUW8aBYm4epYJWOfSeUBPVQAsgz7aWaNe0pmDBsjgggBxsyq*8VU1FdBshuTDdL2-bp2ALs0E-0kpCV5kVdwu/imbdata.png)
# 
# A typical example of imbalanced data is encountered in e-mail classification problem where emails are classified into ham or spam. The number of spam emails is usually lower than the number of relevant (ham) emails. So, using the original distribution of two classes leads to imbalanced dataset.
# 
# Using accuracy as a performace measure for highly imbalanced datasets is not a good idea. For example, if 90% points belong to the true class in a binary  classification problem, a default prediction of true for all data poimts leads to a classifier which is 90% accurate, even though the classifier has not learnt anything about the classification problem at hand![9]

# <a id="632"></a> <br>
# ## 6-3-2 Exploreing Question

# In[ ]:


question = train['question_text']
i=0
for q in question[:5]:
    i=i+1
    print('sample '+str(i)+':' ,q)

# In[ ]:


text_withnumber = train['question_text']
result = ''.join([i for i in text_withnumber if not i.isdigit()])

# <a id="632"></a> <br>
# ## 6-3-2 Some Feature Engineering

# [NLTK](https://www.nltk.org/) is one of the leading platforms for working with human language data and Python, the module NLTK is used for natural language processing. NLTK is literally an acronym for Natural Language Toolkit.
# 
# We get a set of **English stop** words using the line

# In[ ]:


#from nltk.corpus import stopwords
eng_stopwords = set(stopwords.words("english"))

# The returned list stopWords contains **179 stop words**  on my computer.
# You can view the length or contents of this array with the lines:

# In[ ]:


print(len(eng_stopwords))
print(eng_stopwords)

# The metafeatures that we'll create based on  SRK's  EDAs, [sudalairajkumar](http://http://www.kaggle.com/sudalairajkumar/simple-feature-engg-notebook-spooky-author) and [tunguz](https://www.kaggle.com/tunguz/just-some-simple-eda) are:
# 1. Number of words in the text
# 1. Number of unique words in the text
# 1. Number of characters in the text
# 1. Number of stopwords
# 1. Number of punctuations
# 1. Number of upper case words
# 1. Number of title case words
# 1. Average length of the words
# 
# ###### [Go to top](#top)

# Number of words in the text 

# In[ ]:


train["num_words"] = train["question_text"].apply(lambda x: len(str(x).split()))
test["num_words"] = test["question_text"].apply(lambda x: len(str(x).split()))
print('maximum of num_words in train',train["num_words"].max())
print('min of num_words in train',train["num_words"].min())
print("maximum of  num_words in test",test["num_words"].max())
print('min of num_words in train',test["num_words"].min())


# Number of unique words in the text

# In[ ]:


train["num_unique_words"] = train["question_text"].apply(lambda x: len(set(str(x).split())))
test["num_unique_words"] = test["question_text"].apply(lambda x: len(set(str(x).split())))
print('maximum of num_unique_words in train',train["num_unique_words"].max())
print('mean of num_unique_words in train',train["num_unique_words"].mean())
print("maximum of num_unique_words in test",test["num_unique_words"].max())
print('mean of num_unique_words in train',test["num_unique_words"].mean())

# Number of characters in the text 

# In[ ]:



train["num_chars"] = train["question_text"].apply(lambda x: len(str(x)))
test["num_chars"] = test["question_text"].apply(lambda x: len(str(x)))
print('maximum of num_chars in train',train["num_chars"].max())
print("maximum of num_chars in test",test["num_chars"].max())

# Number of stopwords in the text

# In[ ]:


train["num_stopwords"] = train["question_text"].apply(lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords]))
test["num_stopwords"] = test["question_text"].apply(lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords]))
print('maximum of num_stopwords in train',train["num_stopwords"].max())
print("maximum of num_stopwords in test",test["num_stopwords"].max())

# Number of punctuations in the text

# In[ ]:



train["num_punctuations"] =train['question_text'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]) )
test["num_punctuations"] =test['question_text'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]) )
print('maximum of num_punctuations in train',train["num_punctuations"].max())
print("maximum of num_punctuations in test",test["num_punctuations"].max())

# Number of title case words in the text

# In[ ]:



train["num_words_upper"] = train["question_text"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))
test["num_words_upper"] = test["question_text"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))
print('maximum of num_words_upper in train',train["num_words_upper"].max())
print("maximum of num_words_upper in test",test["num_words_upper"].max())

# Number of title case words in the text

# In[ ]:



train["num_words_title"] = train["question_text"].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))
test["num_words_title"] = test["question_text"].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))
print('maximum of num_words_title in train',train["num_words_title"].max())
print("maximum of num_words_title in test",test["num_words_title"].max())

#  Average length of the words in the text 

# In[ ]:



train["mean_word_len"] = train["question_text"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))
test["mean_word_len"] = test["question_text"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))
print('mean_word_len in train',train["mean_word_len"].max())
print("mean_word_len in test",test["mean_word_len"].max())

# We add some new feature to train and test data set now, print columns agains

# In[ ]:


print(train.columns)
train.head(1)

# **<< Note >>**
# >**Preprocessing and generation pipelines depend on a model type**

# ## What is Tokenizer?
# Tokenizing raw text data is an important pre-processing step for many NLP methods. As explained on **wikipedia**, tokenization is “the process of breaking a stream of text up into words, phrases, symbols, or other meaningful elements called tokens.” In the context of actually working through an NLP analysis, this usually translates to converting a string like "My favorite color is blue" to a list or array like ["My", "favorite", "color", "is", "blue"].[11]

# In[ ]:


import nltk
mystring = "I love Kaggle"
mystring2 = "I'd love to participate in kaggle competitions."
nltk.word_tokenize(mystring)

# In[ ]:


nltk.word_tokenize(mystring2)

# <a id="64"></a> <br>
# ## 6-4 Data Visualization
# **Data visualization**  is the presentation of data in a pictorial or graphical format. It enables decision makers to see analytics presented visually, so they can grasp difficult concepts or identify new patterns.
# 
# > * Two** important rules** for Data visualization:
# >     1. Do not put too little information
# >     1. Do not put too much information
# 
# ###### [Go to top](#top)

# <a id="641"></a> <br>
# ## 6-4-1 CountPlot

# In[ ]:


ax=sns.countplot(x='target',hue="target", data=train  ,linewidth=5,edgecolor=sns.color_palette("dark", 3))
plt.title('Is data set imbalance?');

# In[ ]:


ax = sns.countplot(y="target", hue="target", data=train)
plt.title('Is data set imbalance?');

# <a id="642"></a> <br>
# ## 6-4-2  Pie Plot

# In[ ]:



ax=train['target'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%' ,shadow=True)
ax.set_title('target')
ax.set_ylabel('')
plt.show()

# In[ ]:


#plt.pie(train['target'],autopct='%1.1f%%')
 
#plt.axis('equal')
#plt.show()

# <a id="643"></a> <br>
# ## 6-4-3  Histogram

# In[ ]:


f,ax=plt.subplots(1,2,figsize=(20,10))
train[train['target']==0].num_words.plot.hist(ax=ax[0],bins=20,edgecolor='black',color='red')
ax[0].set_title('target= 0')
x1=list(range(0,85,5))
ax[0].set_xticks(x1)
train[train['target']==1].num_words.plot.hist(ax=ax[1],color='green',bins=20,edgecolor='black')
ax[1].set_title('target= 1')
x2=list(range(0,85,5))
ax[1].set_xticks(x2)
plt.show()

# In[ ]:


f,ax=plt.subplots(1,2,figsize=(18,8))
train[['target','num_words']].groupby(['target']).mean().plot.bar(ax=ax[0])
ax[0].set_title('num_words vs target')
sns.countplot('num_words',hue='target',data=train,ax=ax[1])
ax[1].set_title('num_words:target=0 vs target=1')
plt.show()

# In[ ]:


# histograms
train.hist(figsize=(15,20))
plt.figure()

# In[ ]:


train["num_words"].hist();

# <a id="644"></a> <br>
# ## 6-4-4 Violin Plot

# In[ ]:


sns.violinplot(data=train,x="target", y="num_words")

# In[ ]:


sns.violinplot(data=train,x="target", y="num_words_upper")

# <a id="645"></a> <br>
# ## 6-4-5 KdePlot

# In[ ]:


sns.FacetGrid(train, hue="target", size=5).map(sns.kdeplot, "num_words").add_legend()
plt.show()

# <a id="646"></a> <br>
# ## 6-4-6 BoxPlot

# In[ ]:


train['num_words'].loc[train['num_words']>60] = 60 #truncation for better visuals
axes= sns.boxplot(x='target', y='num_words', data=train)
axes.set_xlabel('Target', fontsize=12)
axes.set_title("Number of words in each class", fontsize=15)
plt.show()

# In[ ]:


train['num_chars'].loc[train['num_chars']>350] = 350 #truncation for better visuals

axes= sns.boxplot(x='target', y='num_chars', data=train)
axes.set_xlabel('Target', fontsize=12)
axes.set_title("Number of num_chars in each class", fontsize=15)
plt.show()

# <a id="646"></a> <br>
# ## 6-4-6 WordCloud

# In[ ]:


def generate_wordcloud(text): 
    wordcloud = wc(relative_scaling = 1.0,stopwords = eng_stopwords).generate(text)
    fig,ax = plt.subplots(1,1,figsize=(10,10))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    ax.margins(x=0, y=0)
    plt.show()

# In[ ]:


text =" ".join(train.question_text)
generate_wordcloud(text)

# <a id="7"></a> <br>
# ## 7- Apply Learning
# How to understand what is the best way to solve our problem?!
# 
# The answer is always "**It depends**." It depends on the **size**, **quality**, and **nature** of the **data**. It depends on what you want to do with the answer. It depends on how the **math** of the algorithm was translated into instructions for the computer you are using. And it depends on how much **time** you have. Even the most **experienced data scientists** can't tell which algorithm will perform best before trying them.(see a nice [cheatsheet](https://github.com/mjbahmani/10-steps-to-become-a-data-scientist/blob/master/cheatsheets/microsoft-machine-learning-algorithm-cheat-sheet-v7.pdf) for this section)
# Categorize the problem
# The next step is to categorize the problem. This is a two-step process.
# 
# 1. **Categorize by input**:
#     1. If you have labelled data, it’s a supervised learning problem.
#     1. If you have unlabelled data and want to find structure, it’s an unsupervised learning problem.
#     1. If you want to optimize an objective function by interacting with an environment, it’s a reinforcement learning problem.
# 1. **Categorize by output**.
#     1. If the output of your model is a number, it’s a regression problem.
#     1. If the output of your model is a class, it’s a classification problem.
#     1. If the output of your model is a set of input groups, it’s a clustering problem.
#     1. Do you want to detect an anomaly ? That’s anomaly detection
# 1. **Understand your constraints**
#     1. What is your data storage capacity? Depending on the storage capacity of your system, you might not be able to store gigabytes of classification/regression models or gigabytes of data to clusterize. This is the case, for instance, for embedded systems.
#     1. Does the prediction have to be fast? In real time applications, it is obviously very important to have a prediction as fast as possible. For instance, in autonomous driving, it’s important that the classification of road signs be as fast as possible to avoid accidents.
#     1. Does the learning have to be fast? In some circumstances, training models quickly is necessary: sometimes, you need to rapidly update, on the fly, your model with a different dataset.
# 1. **Find the available algorithms**
#     1. Now that you a clear understanding of where you stand, you can identify the algorithms that are applicable and practical to implement using the tools at your disposal. Some of the factors affecting the choice of a model are:
# 
#     1. Whether the model meets the business goals
#     1. How much pre processing the model needs
#     1. How accurate the model is
#     1. How explainable the model is
#     1. How fast the model is: How long does it take to build a model, and how long does the model take to make predictions.
#     1. How scalable the model is
# 

# -----------------
# <a id="8"></a> <br>
# # 8- Conclusion

# This kernel is not completed yet , I have tried to cover all the parts related to the process of **Quora problem** with a variety of Python packages and I know that there are still some problems then I hope to get your feedback to improve it.
# 

# you can Fork and Run this kernel on **Github**:
# > ###### [ GitHub](https://github.com/mjbahmani/10-steps-to-become-a-data-scientist)
# 
# --------------------------------------
# 
#  **I hope you find this kernel helpful and some <font color="red"><b>UPVOTES</b></font> would be very much appreciated** 

# <a id="9"></a> <br>
# 
# -----------
# 
# # 9- References
# ## 9-1 Kaggle's Kernels
# **In the end , I want to thank all the kernels I've used in this notebook**:
# 1. [SRK](https://www.kaggle.com/sudalairajkumar/simple-exploration-notebook-qiqc)
# 1. [mihaskalic](https://www.kaggle.com/mihaskalic/lstm-is-all-you-need-well-maybe-embeddings-also)
# 1. [artgor](https://www.kaggle.com/artgor/eda-and-lstm-cnn)
# 1. [tunguz](https://www.kaggle.com/tunguz/just-some-simple-eda)
# 
# ## 9-2 Other References
# 
# 1. [Machine Learning Certification by Stanford University (Coursera)](https://www.coursera.org/learn/machine-learning/)
# 
# 1. [Machine Learning A-Z™: Hands-On Python & R In Data Science (Udemy)](https://www.udemy.com/machinelearning/)
# 
# 1. [Deep Learning Certification by Andrew Ng from deeplearning.ai (Coursera)](https://www.coursera.org/specializations/deep-learning)
# 
# 1. [Python for Data Science and Machine Learning Bootcamp (Udemy)](Python for Data Science and Machine Learning Bootcamp (Udemy))
# 
# 1. [Mathematics for Machine Learning by Imperial College London](https://www.coursera.org/specializations/mathematics-machine-learning)
# 
# 1. [Deep Learning A-Z™: Hands-On Artificial Neural Networks](https://www.udemy.com/deeplearning/)
# 
# 1. [Complete Guide to TensorFlow for Deep Learning Tutorial with Python](https://www.udemy.com/complete-guide-to-tensorflow-for-deep-learning-with-python/)
# 
# 1. [Data Science and Machine Learning Tutorial with Python – Hands On](https://www.udemy.com/data-science-and-machine-learning-with-python-hands-on/)
# 1. [imbalanced-dataset](https://www.quora.com/What-is-an-imbalanced-dataset)
# 1. [algorithm-choice](https://docs.microsoft.com/en-us/azure/machine-learning/studio/algorithm-choice)
# 1. [tokenizing-raw-text-in-python](http://jeffreyfossett.com/2014/04/25/tokenizing-raw-text-in-python.html)
# 1. [text-analytics101](http://text-analytics101.rxnlp.com/2014/10/all-about-stop-words-for-text-mining.html)
# -------------
# 
# ###### [Go to top](#top)

# Go to first step: [Course Home Page](https://www.kaggle.com/mjbahmani/10-steps-to-become-a-data-scientist)
# 
# Go to next step : [Titanic](https://www.kaggle.com/mjbahmani/a-comprehensive-ml-workflow-with-python)
# 

# #### The kernel is not complete and will be updated soon  !!!
