#!/usr/bin/env python
# coding: utf-8

# # Introduction

# Building neural networks is a complex endeavor with many parameters to tweak prior to achieving the final version of a model. On top of this, the two most widely used numerical platforms for deep learning and neural network machine learning models, TensorFlow and Theano, are too complex to allow for rapid prototyping. The Keras Deep Learning library for Python helps bridge the gap between prototyping speed and the utilization of the advanced numerical platforms for deep learning.

# ## Keras
# ![Keras Image](https://s3.amazonaws.com/keras.io/img/keras-logo-2018-large-1200.png)
# 
# Keras is a high-level neural networks API, written in Python and capable of running on top of TensorFlow, CNTK, or Theano. It was developed with a focus on enabling fast experimentation. Being able to go from idea to result with the least possible delay is key to doing good research.
# 
# Use Keras if you need a deep learning library that:
# 
# * Allows for easy and fast prototyping (through user friendliness, modularity, and extensibility).
# * Supports both convolutional networks and recurrent networks, as well as combinations of the two.
# * Runs seamlessly on CPU and GPU.

# # Getting Started

# ## Problem Definition

# In this problem, we will be using the famous IRIS Flower dataset.
# 
# This dataset is well studied and is a good problem for practicing on neural networks because all of the 4 input variables are numeric and have the same scale in centimeters. Each instance describes the properties of an observed flower measurements and the output variable is specific iris species.
# 
# This is a multi-class classification problem, meaning that there are more than two classes to be predicted, in fact there are three flower species. This is an important type of problem on which to practice with neural networks because the three class values require specialized handling.
# 
# The iris flower dataset is a well studied problem and a such we can expect to achieve an model accuracy in the range of 95% to 97%. This provides a good target to aim for when developing our models.
# 
# You can [download](http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data) the iris flowers dataset from the UCI Machine Learning repository and place it in your current working directory with the filename iris.csv
# 
# However, in our case - we have imported the dataset from kaggle in sqlite3 format.

# ## Import libraries and Functions

# In[34]:


import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

import sqlite3

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler

import tensorflow as tf

# > Tensorflow version

# In[35]:


tf.__version__

# > Intialization Seed
# 
# This is important to ensure that the results we achieve from this model can be achieved again precisely. It ensures that the stochastic process of training a neural network model can be reproduced.

# In[36]:


# fix random seed for reproducibility
seed = 42
np.random.seed(seed)

# ## Load the dataset

# Here, we will loading data from a a sqlite database using pandas. Alternatively, we can load the data from csv as well. Also, from sklearn.datasets as well

# In[48]:


connection = sqlite3.connect('../input/database.sqlite')
data = pd.read_sql_query(''' SELECT * FROM IRIS ''', connection)
print("Shape of data: {}".format(data.shape))

# Printing the first 3 rows from the database will give a higher level understanding of "what's in the data actually"

# In[38]:


data.head(3)

# Checking if the dataset contains null/na values or not.

# In[39]:


data.info()

# **Observation** - 
# 1. There are no null values in the dataset.
# 2. Total number of observations are 150.
# 3. All the features except the output feature i.e. Species are of float dtype.

# In[40]:


Y = data['Species']
X = data.drop(['Id', 'Species'], axis=1)
print("Shape of Input  features: {}".format(X.shape))
print("Shape of Output features: {}".format(Y.shape))

# ## Encoding the Output/Response Variable

# In[41]:


Y.value_counts()

# In[42]:


lbl_clf = LabelEncoder()
Y_encoded = lbl_clf.fit_transform(Y)

#Keras requires your output feature to be one-hot encoded values.
Y_final = tf.keras.utils.to_categorical(Y_encoded)

print("Therefore, our final shape of output feature will be {}".format(Y_final.shape))

# ## Splitting the dataset in 75-25 ratio

# In[43]:


x_train, x_test, y_train, y_test = train_test_split(X, Y_final, test_size=0.25, random_state=seed, stratify=Y_encoded, shuffle=True)

print("Training Input shape\t: {}".format(x_train.shape))
print("Testing Input shape\t: {}".format(x_test.shape))
print("Training Output shape\t: {}".format(y_train.shape))
print("Testing Output shape\t: {}".format(y_test.shape))

# ## Standardizing the dataset

# In[44]:


std_clf = StandardScaler()
x_train_new = std_clf.fit_transform(x_train)
x_test_new = std_clf.transform(x_test)

# In[45]:


model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(10, input_dim=4, activation=tf.nn.relu, kernel_initializer='he_normal', 
                                kernel_regularizer=tf.keras.regularizers.l2(0.01)))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Dense(7, activation=tf.nn.relu, kernel_initializer='he_normal', 
                                kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.001, l2=0.001)))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Dense(5, activation=tf.nn.relu, kernel_initializer='he_normal', 
                                kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.001, l2=0.001)))
model.add(tf.keras.layers.Dense(3, activation=tf.nn.softmax))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

iris_model = model.fit(x_train_new, y_train, epochs=700, batch_size=7)

# In[47]:


model.evaluate(x_test_new, y_test)
