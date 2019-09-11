#!/usr/bin/env python
# coding: utf-8

# # <div style="text-align: center">Top 5 Deep Learning Frameworks Tutorial </div>
#  ### <div align="center"><b>CLEAR DATA. MADE MODEL.</b></div>
#  <div align="center">Each framework is built in a different manner for different purposes. In this Notebook, we look at the 5 deep learning frameworks to give you a better idea of which framework will be the perfect fit or come handy in solving your **business challenges**.</div>
# <div style="text-align:center">last update: <b>12/04/2018</b></div>
# >###### you may  be interested have a look at it: [**10-steps-to-become-a-data-scientist**](https://github.com/mjbahmani/10-steps-to-become-a-data-scientist)
# 
# ---------------------------------------------------------------------
# Fork and run my kernels on **GiHub**  and follow me:
# 
# > ###### [ GitHub](https://github.com/mjbahmani/10-steps-to-become-a-data-scientist)
# 
# -------------------------------------------------------------------------------------------------------------
#  **I hope you find this kernel helpful and some <font color="red"><b>UPVOTES</b></font> would be very much appreciated**
#  
#  -----------

#  <a id="top"></a> <br>
# ## Notebook  Content
# 1. [Introduction](#1)
#     1. [Courses](#2)
#     1. [Kaggle kernels](#3)
#     1. [Ebooks](#4)
#     1. [Cheat Sheets](#5)
#     1. [Deep Learning vs Machine Learning](#6)
# 1. [Loading Packages & Data](#7)
#     1. [Version](#8)
#     1. [Setup](#9)
#     1. [Loading Data](#10)
#         1. [Data fields](#11)
#     1. [EDA](#12)
# 1. [Python Deep Learning Packages](#31)
#     1. [Keras](#33)
#         1. [Analysis](#34)
#         1. [Text Classification with Keras](#331)
#     1. [TensorFlow](#35)
#         1. [Import the Fashion MNIST dataset](#36)
#         1. [Explore the data](#37)
#         1. [Preprocess the data](#38)
#         1. [Build the model](#39)
#             1. [Setup the layers](#40)
#         1. [Compile the model](#41)
#         1. [Train the model](#42)
#         1. [Evaluate accuracy](#43)
#         1. [Make predictions](#44)
#     1. [Theano](#45)
#         1. [Theano( example)](#46)
#         1. [Calculating multiple results at once](#47)
#     1. [Pytroch](#48)
#         1. [Tensors](#49)
#         1. [Operations](#50)
#     1. [CNTK](#51)
# 1. [Conclusion](#51)
# 1. [References](#52)

#  <a id="1"></a> <br>
# ## 1- Introduction
# This is a **comprehensive Deep Learning techniques with python**, it is clear that everyone in this community is familiar with **MNIST dataset**  and other data sets that I want to use in this kernel sach as **Quora** but if you need to review your information about the dataset please visit [MNIST](https://en.wikipedia.org/wiki/MNIST_database), [Quora](https://www.kaggle.com/c/quora-insincere-questions-classification).
# 
# I have tried to help  Kaggle users  how to face deep learning problems. and I think it is a great opportunity for who want to learn deep learning workflow with python completely.
# <a id="2"></a> <br>
# ## 1-1 Courses
# There are a lot of online courses that can help you develop your knowledge, here I have just  listed some of them that I have used in my Kernel:
# 
# 1. [Deep Learning Certification by Andrew Ng from deeplearning.ai (Coursera)](https://www.coursera.org/specializations/deep-learning)
# 1. [Deep Learning A-Z™: Hands-On Artificial Neural Networks](https://www.udemy.com/deeplearning/)
# 
# 1. [Creative Applications of Deep Learning with TensorFlow](https://www.class-central.com/course/kadenze-creative-applications-of-deep-learning-with-tensorflow-6679)
# 1. [Neural Networks for Machine Learning](https://www.class-central.com/mooc/398/coursera-neural-networks-for-machine-learning)
# 1. [Practical Deep Learning For Coders, Part 1](https://www.class-central.com/mooc/7887/practical-deep-learning-for-coders-part-1)
# 1. [cs231n](http://cs231n.stanford.edu/)
# <a id="3"></a> <br>
# 
# ## 1-2 Kaggle kernels
# I want to thanks **Kaggle team**  and  all of the **kernel's authors**  who develop this huge resources for Data scientists. I have learned from The work of others and I have just listed some more important kernels that inspired my work and I've used them in this kernel:
# 
# 1. [Deep Learning Tutorial for Beginners](https://www.kaggle.com/kanncaa1/deep-learning-tutorial-for-beginners)
# 1. [introduction-to-cnn-keras-0-997-top-6](https://www.kaggle.com/yassineghouzam/introduction-to-cnn-keras-0-997-top-6)
# 
# <a id="4"></a> <br>
# ## 1-3 Ebooks
# So you love reading , here is **10 free machine learning books**
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
# <a id="5"></a> <br>
# 
# ## 1-4 Cheat Sheets
# Data Science is an ever-growing field, there are numerous tools & techniques to remember. It is not possible for anyone to remember all the functions, operations and formulas of each concept. That’s why we have cheat sheets. But there are a plethora of cheat sheets available out there, choosing the right cheat sheet is a tough task. So, I decided to write this article.
# 
# Here I have selected the cheat sheets on the following criteria: comprehensiveness, clarity, and content [26]:
# 1. [Quick Guide to learn Python for Data Science ](https://github.com/mjbahmani/10-steps-to-become-a-data-scientist/blob/master/cheatsheets/Data-Science-in-Python.pdf)
# 1. [Python for Data Science Cheat sheet ](https://github.com/mjbahmani/10-steps-to-become-a-data-scientist/blob/master/cheatsheets/beginners_python_cheat_sheet.pdf)
# 1. [Python For Data Science Cheat Sheet NumPy](https://github.com/mjbahmani/10-steps-to-become-a-data-scientist/blob/master/cheatsheets/Numpy_Python_Cheat_Sheet.pdf)
# 1. [Exploratory Data Analysis in Python]()
# 1. [Data Exploration using Pandas in Python](https://github.com/mjbahmani/10-steps-to-become-a-data-scientist/blob/master/cheatsheets/Data-Exploration-in-Python.pdf)
# 1. [Data Visualisation in Python](https://github.com/mjbahmani/10-steps-to-become-a-data-scientist/blob/master/cheatsheets/data-visualisation-infographics1.jpg)
# 1. [Python For Data Science Cheat Sheet Bokeh](https://github.com/mjbahmani/10-steps-to-become-a-data-scientist/blob/master/cheatsheets/Python_Bokeh_Cheat_Sheet.pdf)
# 1. [Cheat Sheet: Scikit Learn ](https://github.com/mjbahmani/10-steps-to-become-a-data-scientist/blob/master/cheatsheets/Scikit-Learn-Infographic.pdf)
# 1. [MLalgorithms CheatSheet](https://github.com/mjbahmani/10-steps-to-become-a-data-scientist/blob/master/cheatsheets/MLalgorithms-.pdf)
# 1. [Probability Basics  Cheat Sheet ](https://github.com/mjbahmani/10-steps-to-become-a-data-scientist/blob/master/cheatsheets/probability_cheatsheet.pdf)
# <a id="6"></a> <br>
# 
# ## 1-5 Deep Learning vs Machine Learning
# We use a **machine algorithm** to parse data, learn from that data, and make informed decisions based on what it has learned. Basically, **Deep Learning** is used in layers to create an **Artificial Neural Network** that can learn and make intelligent decisions on its own. We can say **Deep Learning is a sub-field of Machine Learning**.
# 
# <img src ="http://blog.thinkwik.com/wp-content/uploads/2018/07/Insights-of-The-Machine-Learning-and-The-Deep-Learning.png">
# [image Credits](http://blog.thinkwik.com/insights-of-the-machine-learning-and-the-deep-learning/)
# 
# I am open to getting your feedback for improving this **kernel**
# 
# ###### [Go to top](#top)

# <a id="7"></a> <br>
# # 2 Loading Packages & Data
# In this kernel we are using the following packages:

# In[ ]:


from pandas import get_dummies
import matplotlib as mpl
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib
import warnings
import sklearn
import keras
import scipy
import numpy
import sys
import csv
import os

# <a id="8"></a> <br>
# ## 2-1 Version
# Print version of each package.
# ###### [Go to top](#top)

# In[ ]:


print('matplotlib: {}'.format(matplotlib.__version__))
print('sklearn: {}'.format(sklearn.__version__))
print('scipy: {}'.format(scipy.__version__))
print('seaborn: {}'.format(sns.__version__))
print('pandas: {}'.format(pd.__version__))
print('numpy: {}'.format(np.__version__))
print('Python: {}'.format(sys.version))
print('Keras: {}'.format(keras.__version__))
print('tensorflow: {}'.format(tf.__version__))

# <a id="9"></a> <br>
# ## 2-2 Setup
# 
# A few tiny adjustments for better **code readability**

# In[ ]:


sns.set(style='white', context='notebook', palette='deep')
warnings.filterwarnings('ignore')
mpl.style.use('ggplot')
sns.set_style('white')
np.random.seed(1337)

# <a id="10"></a> <br>
# ## 2-3 Loading Data
# Data collection is the process of gathering and measuring data, information or any variables of interest in a standardized and established manner that enables the collector to answer or test hypothesis and evaluate outcomes of the particular collection.[techopedia](https://www.techopedia.com/definition/30318/data-collection) I start Collection Data by the training and testing datasets into Pandas DataFrames. Each row is an observation (also known as : sample, example, instance, record) Each column is a feature (also known as: Predictor, attribute, Independent Variable, input, regressor, Covariate).
# ### 2-3-1 What is a insincere question?
# is defined as a question intended to make a **statement** rather than look for helpful **answers**. 
# ### 2-3-2 how can we find  insincere question?
# Some **characteristics** that can signify that a question is **insincere**:
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
# 1. **Uses sexual content** (incest, bestiality, pedophilia) for shock value, and not to seek genuine answers
# 
# After loading the data via pandas, we should checkout what the content is, description and via the following:
# ###### [Go to top](#top)

# In[ ]:


print(os.listdir("../input"))

# <a id="11"></a> <br>
# ### 2-3-1 Data fields
# 1. qid - unique question identifier
# 1. question_text - Quora question text
# 1. target - a question labeled "insincere" has a value of 1, otherwise 0

# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

# In[ ]:


type(train)

# To check the first 5 rows of the data set, we can use head(5).

# In[ ]:


train.head(5) 

# <a id="11"></a> <br>
# ### 2-3-2 Target
# You will be predicting whether a question asked on **Quora** is sincere or not.

# <a id="12"></a> <br>
# ## 2-4 EDA
# In this section, you'll learn how to use graphical and numerical techniques to begin uncovering the structure of your data.
# 1. Which variables suggest interesting relationships?
# 1. Which observations are unusual?
# 1. Analysis of the features! By the end of the section, you'll be able to answer these questions and more, while generating. 

# In[ ]:


train.sample(5)

# To pop up 5 random rows from the data set, we can use **sample(5**) function.

# In[ ]:


test.sample(5)

# To check out last 5 row of the data set, we use tail() function.

# In[ ]:


train.tail()

# In[ ]:


print(train.shape)
print(test.shape)

# To check out how many null info are on the dataset, we can use **isnull().sum()**.

# In[ ]:


train.isnull().sum()

# In[ ]:


test.isnull().sum()

# ### 2-4-1 About Quora
# Quora is a platform that empowers people to learn from each other. On Quora, people can ask questions and connect with others who contribute unique insights and quality answers. A key challenge is to weed out insincere questions -- those founded upon false premises, or that intend to make a statement rather than look for helpful answers.

# In[ ]:


print(train.info())

# to give a statistical summary about the dataset, we can use **describe()**.

# In[ ]:


train.describe() 


# ###### [Go to top](#top)

# <a id="31"></a> <br>
# # 3- Python Deep Learning Packages
# <img src='https://cdn-images-1.medium.com/max/800/1*dYjDEI0mLpsCOySKUuX1VA.png'>
# *State of open source deep learning frameworks in 2017* [**towardsdatascience**](https://towardsdatascience.com/battle-of-the-deep-learning-frameworks-part-i-cff0e3841750)
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

# <a id="32"></a> <br>
# # 4- Frameworks
# Let's Start Learning, in this section we intrduce 5 deep learning frameworks.

# <a id="33"></a> <br>
# ## 4-1 Keras
# Our workflow will be as follow[10] [deep-learning-with-python-notebooks](https://github.com/fchollet/deep-learning-with-python-notebooks):
# 1. first we will present our neural network with the training data, `train_images` and `train_labels`. 
# 1. The network will then learn to associate images and labels. 
# 1. Finally, we will ask the network to produce predictions for `test_images`, 
# 1. and we  will verify if these predictions match the labels from `test_labels`.
# 
# **Let's build our network **
# 
# ###### [Go to top](#top)

# In[ ]:


# import Dataset to play with it
from keras.datasets import mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# In[ ]:


from keras import models
from keras import layers

network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation='softmax'))

# <a id="34"></a> <br>
# ## 4-1-1 Analysis
# The core building block of neural networks is the "**layer**", a data-processing module which you can conceive as a "**filter**" for data. Some  data comes in, and comes out in a more useful form. Precisely, layers extract _representations_ out of the data fed into them -- hopefully  representations that are more meaningful for the problem at hand. Most of deep learning really consists of chaining together simple layers which will implement a form of progressive "**data distillation**". [colab.research.google](https://colab.research.google.com/github/alzayats/Google_Colab/blob/master/2_1_a_first_look_at_a_neural_network.ipynb)
# A deep learning model is like a sieve for data processing, made of a succession of increasingly refined data filters -- the "layers".
# Here our network consists of a sequence of two `Dense` layers, which are densely-connected (also called "fully-connected") neural layers. 
# The second (and last) layer is a 10-way "**softmax**" layer, which means it will return an array of 10 probability scores (summing to 1). Each score will be the probability that the current digit image belongs to one of our 10 digit classes.
# To make our network ready for training, we need to pick three more things, as part of "compilation" step:
# 
# 1. A loss function: the is how the network will be able to measure how good a job it is doing on its training data, and thus how it will be able to steer itself in the right direction.
# 1. An optimizer: this is the mechanism through which the network will update itself based on the data it sees and its loss function.
# 1. Metrics to monitor during training and testing. Here we will only care about accuracy (the fraction of the images that were correctly classified).
# 
# ###### [Go to top](#top)

# In[ ]:


network.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

# 
# Before training, we will **preprocess** our data by reshaping it into the shape that the network expects, and **scaling** it so that all values are in 
# the `[0, 1]` interval. Previously, our training images for instance were stored in an array of shape `(60000, 28, 28)` of type `uint8` with 
# values in the `[0, 255]` interval. We transform it into a `float32` array of shape `(60000, 28 * 28)` with values between 0 and 1.
# 
# ###### [Go to top](#top)

# In[ ]:


train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255

# We also need to **categorically encode** the labels

# In[ ]:


from keras.utils import to_categorical

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# We are now ready to train our network, which in **Keras** is done via a call to the `fit` method of the network: 
# we "fit" the model to its training data.
# 
# ###### [Go to top](#top)

# In[ ]:


#please change epochs to 5
network.fit(train_images, train_labels, epochs=1, batch_size=128)

# **Two quantities** are being displayed during training: the "**loss**" of the network over the training data, and the accuracy of the network over 
# the training data.
# 
# We quickly reach an accuracy of **0.989 (i.e. 98.9%)** on the training data. Now let's check that our model performs well on the test set too:
# 
# ###### [Go to top](#top)

# In[ ]:


test_loss, test_acc = network.evaluate(test_images, test_labels)

# In[ ]:


print('test_acc:', test_acc)

# 
# **Our test set accuracy turns out to be 97.8%**

# <a id="331"></a> <br>
# ## 4-1-2  Text Classification with Keras
# A simple text classification from [16].

# In[ ]:


from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding, GlobalAveragePooling1D
from keras.datasets import imdb

# In[ ]:


max_features = 5000
maxlen = 400
batch_size = 32
embedding_dims = 20

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

# In[ ]:


model = Sequential()
model.add(Embedding(input_dim=5000, output_dim=embedding_dims))
model.add(GlobalAveragePooling1D())
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          validation_data=(x_test, y_test))

# <a id="35"></a> <br>
# ## 4-2 TensorFlow
# **TensorFlow** is an open-source machine learning library for research and production. TensorFlow offers **APIs** for beginners and experts to develop for desktop, mobile, web, and cloud. See the sections below to get started.[12] [tensorflow](https://www.tensorflow.org/tutorials)
# 
# ###### [Go to top](#top)

# In[ ]:


# Simple hello world using TensorFlow
hello = tf.constant('Hello, TensorFlow!')
# Start tf session
sess = tf.Session()
# Run graph
print(sess.run(hello))
mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(512, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
#please change epochs to 5
model.fit(x_train, y_train, epochs=1)
model.evaluate(x_test, y_test)

# <a id="36"></a> <br>
# ## 4-2-1 Import the Fashion MNIST dataset
# 

# In[ ]:


fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Loading the dataset returns **four NumPy arrays**:
# 
# 1. The train_images and train_labels arrays are the training set—the data the model uses to learn.
# 1. The model is tested against the test set, the test_images, and test_labels arrays.
# 1. The images are 28x28 NumPy arrays, with pixel values ranging between 0 and 255.
# 1. The labels are an array of integers, ranging from 0 to 9. These correspond to the class of clothing the image represents:
# 
# ###### [Go to top](#top)

# <img src='https://tensorflow.org/images/fashion-mnist-sprite.png'>
# [image credit](https://tensorflow.org)

# Each image is **mapped** to a single label. Since the class names are not included with the dataset, store them here to use later when plotting the images:

# In[ ]:


class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# <a id="37"></a> <br>
# ## 4-2-2 Explore the data
# Let's explore the format of the dataset before training the model. The following shows there are **60,000** images in the training set, with each image represented as 28 x 28 pixels:
# 
# ###### [Go to top](#top)

# In[ ]:


train_images.shape

# Likewise, there are 60,000 labels in the training set:
# 
# 

# In[ ]:


len(train_labels)


# Each label is an integer between 0 and 9:
# 
# 

# In[ ]:


train_labels


# There are 10,000 images in the test set. Again, each image is represented as 28 x 28 pixels:
# 
# 

# In[ ]:


test_images.shape


# And the test set contains 10,000 images labels:
# 
# 

# In[ ]:


len(test_labels)


# <a id="38"></a> <br>
# ## 4-2-3 Preprocess the data
# 

# The data must be preprocessed before training the network. If you inspect the first image in the training set, you will see that the pixel values fall in the range of 0 to 255:
# 
# ###### [Go to top](#top)
# 

# In[ ]:


plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)

# We scale these values to a range of 0 to 1 before feeding to the neural network model. For this, cast the datatype of the image components from an** integer to a float,** and divide by 255. Here's the function to preprocess the images:
# 
# It's important that the training set and the testing set are preprocessed in the same way:

# In[ ]:


train_images = train_images / 255.0

test_images = test_images / 255.0

# Display the first 25 images from the training set and display the class name below each image. **Verify** that the data is in the correct format and we're ready to build and train the network.

# In[ ]:


plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])

# <a id="39"></a> <br>
# ## 4-2-4 Build the model
# 

# **Building the neural network requires configuring the layers of the model, then compiling the model.**
# <a id="40"></a> <br>
# ### 4-2-4-1 Setup the layers
# The basic building block of a neural network is the layer. **Layers** extract representations from the data fed into them. And, hopefully, these representations are more meaningful for the problem at hand.
# 
# Most of deep learning consists of chaining together simple layers. Most layers, like tf.keras.layers.Dense, have parameters that are learned during training.
# 
# ###### [Go to top](#top)

# In[ ]:


model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

# The **first layer** in this network, tf.keras.layers.Flatten, transforms the format of the images from a 2d-array (of 28 by 28 pixels), to a 1d-array of 28 * 28 = 784 pixels. Think of this layer as unstacking rows of pixels in the image and lining them up. This layer has no parameters to learn; it only reformats the data.
# 
# After the pixels are flattened, the network consists of a sequence of two tf.keras.layers.Dense layers. These are densely-connected, or fully-connected, neural layers. The first Dense layer has 128 nodes (or neurons). **The second (and last) layer** is a 10-node softmax layer—this returns an array of 10 probability scores that sum to 1. Each node contains a score that indicates the probability that the current image belongs to one of the 10 classes.
# 
# ###### [Go to top](#top)

# <a id="41"></a> <br>
# ## 4-2-5 Compile the model
# Before the model is ready for training, it needs a few more settings. These are added during the model's compile step:
# 
# 1. **Loss function** —This measures how accurate the model is during training. We want to minimize this function to "steer" the model in the right direction.
# 1. **Optimizer** —This is how the model is updated based on the data it sees and its loss function.
# 1. **Metrics** —Used to monitor the training and testing steps. The following example uses accuracy, the fraction of the images that are correctly classified.
# 
# ###### [Go to top](#top)

# In[ ]:


model.compile(optimizer=tf.train.AdamOptimizer(), 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# <a id="42"></a> <br>
# ## 4-2-6 Train the model
# Training the neural network model requires the following steps:
# 
# Feed the training data to the model—in this example, the train_images and train_labels arrays.
# The model learns to associate images and labels.
# We ask the model to make predictions about a test set—in this example, the test_images array. We verify that the predictions match the labels from the test_labels array.
# To start training, call the model.fit method—the model is "fit" to the training data:
# 
# ###### [Go to top](#top)

# In[ ]:


#please change epochs to 5
model.fit(train_images, train_labels, epochs=1)

# As the model trains, the loss and accuracy metrics are displayed. This model reaches an accuracy of about 0.88 (or 88%) on the training data.

# <a id="43"></a> <br>
# ## 4-2-7 Evaluate accuracy
# Next, compare how the model performs on the test dataset:
# 
# ###### [Go to top](#top)

# In[ ]:


test_loss, test_acc = model.evaluate(test_images, test_labels)

print('Test accuracy:', test_acc)

# It turns out, the accuracy on the test dataset is a little less than the accuracy on the training dataset. This gap between training accuracy and test accuracy is an example of overfitting. Overfitting is when a machine learning model performs worse on new data than on their training data.

# <a id="44"></a> <br>
# ## 4-2-8 Make predictions
# With the model trained, we can use it to make predictions about some images.
# ###### [Go to top](#top)

# In[ ]:


predictions = model.predict(test_images)


# Here, the model has predicted the label for each image in the testing set. Let's take a look at the first prediction:
# 
# 

# In[ ]:


predictions[0]


# A prediction is an array of 10 numbers. These describe the "confidence" of the model that the image corresponds to each of the 10 different articles of clothing. We can see which label has the highest confidence value:
# 
# 

# In[ ]:


np.argmax(predictions[0])


# We can graph this to look at the full set of 10 channels
# 
# 

# In[ ]:


def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  
  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'
  
  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array[i], true_label[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1]) 
  predicted_label = np.argmax(predictions_array)
 
  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')

# Let's look at the 0th image, predictions, and prediction array.
# 
# ###### [Go to top](#top)

# In[ ]:


i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions,  test_labels)

# In[ ]:


i = 12
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions,  test_labels)

# Let's plot several images with their predictions. Correct prediction labels are blue and incorrect prediction labels are red. The number gives the percent (out of 100) for the predicted label. Note that it can be wrong even when very confident.
# 
# ###### [Go to top](#top)

# In[ ]:


# Plot the first X test images, their predicted label, and the true label
# Color correct predictions in blue, incorrect predictions in red
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions, test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions, test_labels)

# Finally, use the trained model to make a **prediction** about a single image.
# 
# ###### [Go to top](#top)

# In[ ]:


# Grab an image from the test dataset
img = test_images[0]

print(img.shape)

# **tf.keras** models are optimized to make predictions on a batch, or collection, of examples at once. So even though we're using a single image, we need to add it to a list:
# 
# 

# In[ ]:


# Add the image to a batch where it's the only member.
img = (np.expand_dims(img,0))

print(img.shape)

# Now predict the image:
# 
# ###### [Go to top](#top)

# In[ ]:


predictions_single = model.predict(img)

print(predictions_single)

# In[ ]:


plot_value_array(0, predictions_single, test_labels)
_ = plt.xticks(range(10), class_names, rotation=45)

# <a id="45"></a> <br>
# # 4-3 Theano 
# **Theano** is a numerical computation library for Python. It is a common choice for implementing neural network models as it allows you to efficiently define, optimize and evaluate mathematical expressions, including multi-dimensional arrays (numpy.ndaray).[13] [ Credits to journaldev](https://www.journaldev.com/17840/theano-python-tutorial)
# 
# ###### [Go to top](#top)

# Theano has got an amazing compiler which can do various optimizations of varying complexity. A few of such optimizations are:
# 
# 1. Arithmetic simplification (e.g: --x -> x; x + y - x -> y)
# 1. Using memory aliasing to avoid calculation
# 1. Constant folding
# 1. Merging similar subgraphs, to avoid redundant calculation
# 1. Loop fusion for elementwise sub-expressions
# 1. GPU computations

# In[ ]:


import theano
from theano import tensor

x = tensor.dscalar()
y = tensor.dscalar()

z = x + y
f = theano.function([x,y], z)
print(f(1.5, 2.5))

# <a id="46"></a> <br>
# ## 4-3-1 Theano( example)

# Let’s have a look at rather more elaborate example than just adding two numbers. Let’s try to compute the **logistic** curve, which is given by:

# <img src='https://cdn.journaldev.com/wp-content/uploads/2018/01/logistic-curve.png'>

# If we plot a graph for this equation, it will look like:
# 

# <img src='https://cdn.journaldev.com/wp-content/uploads/2018/01/logistic-curve-1.png'>

# Logistic function is applied to each element of matrix. Let’s write a code snippet to demonstrate this:
# 
# ###### [Go to top](#top)

# In[ ]:


# declare a variable
x = tensor.dmatrix('x')

# create the expression
s = 1 / (1 + tensor.exp(-x))

# convert the expression into a callable object which takes
# a matrix as parameter and returns s(x)
logistic = theano.function([x], s)

# call the function with a test matrix and print the result
print(logistic([[0, 1], [-1, -2]]))

# <a id="47"></a> <br>
# ## 4-3-2 Calculating multiple results at once
# Let’s say we have to compute elementwise difference, absolute difference and difference squared between two matrices ‘x’ and ‘y’. Doing this at same time optimizes program with significant duration as we don’t have to go to each element again and again for each operation.
# 
# ###### [Go to top](#top)

# In[ ]:


# declare variables
x, y = tensor.dmatrices('x', 'y')

# create simple expression for each operation
diff = x - y

abs_diff = abs(diff)
diff_squared = diff**2

# convert the expression into callable object
f = theano.function([x, y], [diff, abs_diff, diff_squared])

# call the function and store the result in a variable
result= f([[1, 1], [1, 1]], [[0, 1], [2, 3]])

# format print for readability
print('Difference: ')
print(result[0])

# <a id="48"></a> <br>
# ## 4-4 Pytroch

# It’s a **Python-based** scientific computing package targeted at two sets of audiences:[Credits to pytorch-dynamic-computational](https://medium.com/intuitionmachine/pytorch-dynamic-computational-graphs-and-modular-deep-learning-7e7f89f18d1)
# 
# 1. A replacement for NumPy to use the power of GPUs.
# 1. A deep learning research platform that provides maximum flexibility and speed.
# <img src='https://cdn-images-1.medium.com/max/800/1*5PLIVNA5fIqEC8-kZ260KQ.gif'>
# *PyTorch dynamic computational graph — source: http://pytorch.org/about/*
# 
# ###### [Go to top](#top)

# <a id="49"></a> <br>
# ## 4-4-1 Tensors
# **Tensors** are similar to NumPy’s ndarrays, with the addition being that Tensors can also be used on a GPU to accelerate computing.

# In[ ]:


from __future__ import print_function
import torch

# Construct a 5x3 matrix, uninitialized:

# In[ ]:


x = torch.empty(5, 3)
print(x)

# Construct a randomly initialized matrix:
# 
# 

# In[ ]:


x = torch.rand(5, 3)
print(x)

# Construct a matrix filled zeros and of dtype long:
# 
# ###### [Go to top](#top)

# In[ ]:


x = torch.zeros(5, 3, dtype=torch.long)
print(x)

# Construct a tensor directly from data:
# 
# 

# In[ ]:


x = torch.tensor([5.5, 3])
print(x)

# Or create a tensor based on an existing tensor. These methods will reuse properties of the input tensor, e.g. dtype, unless new values are provided by user.
# 
# ###### [Go to top](#top)

# In[ ]:


x = x.new_ones(5, 3, dtype=torch.double)      # new_* methods take in sizes
print(x)

x = torch.randn_like(x, dtype=torch.float)    # override dtype!
print(x)                                      # result has the same size

# Get its size:

# In[ ]:


print(x.size())


# <a id="50"></a> <br>
# ## 4-4-2 Operations
# There are multiple syntaxes for operations. In the following example, we will take a look at the addition operation.
# 
# Addition: syntax 1.
# 
# ###### [Go to top](#top)

# In[ ]:


y = torch.rand(5, 3)
print(x + y)

# Addition: syntax 2
# 
# 

# In[ ]:


print(torch.add(x, y))


# Addition: providing an output **tensor** as argument.
# 
# 

# In[ ]:


result = torch.empty(5, 3)
torch.add(x, y, out=result)
print(result)

# <a id="51"></a> <br>
# ## 4-4 CNTK
# let's start learning how to use CNTK:
# To train a deep model, you will need to define your model structure, prepare your data so that it can be fed to CNTK, train the model and evaluate its accuracy, and deploy it. [Credit to cntk.ai](https://cntk.ai/pythondocs/CNTK_200_GuidedTour.html)
# 1. Defining your model structure
#     1. The CNTK programming model: Networks as Function Objects
#     1. CNTK's Data Model: Tensors and Sequences of Tensors
#     1. Your First CNTK Network: Logistic Regression
#     1. Your second CNTK Network: MNIST Digit Recognition
#     1. The Graph API: MNIST Digit Recognition Once More
# 1. Feeding your data
#     1. Small data sets that fit into memory: numpy/scipy arrays/
#     1. Large data sets: MinibatchSource class
#     1. Spoon-feeding data: your own minibatch loop
# 1. Training
#     1. Distributed Training
#     1. Logging
#     1. Checkpointing
#     1. Cross-validation based training control
#     1. Final evaluation
# 1. Deploying the model
#     1. From Python
#     1. From C++ and C#
#     1. From your own web service
#     1. Via an Azure web service
# 
# >**Note**:
# To run this tutorial, you will need CNTK v2 and ideally a CUDA-capable GPU (deep learning is no fun without GPUs).
# 
# Coming Soon!!!!
# 
# ###### [Go to top](#top)

# You can follow and fork my work  in **GitHub**:
# > ###### [ GitHub](https://github.com/mjbahmani/10-steps-to-become-a-data-scientist)
# 
# 
# --------------------------------------
# 
#  **I hope you find this kernel helpful and some <font color="green"><b>UPVOTES</b></font> would be very much appreciated**

# <a id="52"></a> <br>
# # 5- Conclusion
# In this kernel we have just tried to create a **comprehensive deep learning workflow** for helping you to  start your jounery in DL.
# surly it is not **completed yet**!! also I want to hear your voice to improve kernel together.

# <a id="53"></a> <br>
# 
# -----------
# 
# # 6- References & Credits
# 1. [https://skymind.ai/wiki/machine-learning-workflow](https://skymind.ai/wiki/machine-learning-workflow)
# 1. [keras](https://www.kaggle.com/yassineghouzam/introduction-to-cnn-keras-0-997-top-6)
# 1. [Problem-define](https://machinelearningmastery.com/machine-learning-in-python-step-by-step/)
# 1. [Sklearn](http://scikit-learn.org/)
# 1. [machine-learning-in-python-step-by-step](https://machinelearningmastery.com/machine-learning-in-python-step-by-step/)
# 1. [Data Cleaning](http://wp.sigmod.org/?p=2288)
# 1. [Kaggle kernel that I use it](https://www.kaggle.com/yassineghouzam/introduction-to-cnn-keras-0-997-top-6)
# 1. [DL vs ML](https://medium.com/swlh/ill-tell-you-why-deep-learning-is-so-popular-and-in-demand-5aca72628780)
# 1. [neural-networks-deep-learning](https://www.coursera.org/learn/neural-networks-deep-learning)
# 1. [deep-learning-with-python-notebooks](https://github.com/fchollet/deep-learning-with-python-notebooks)
# 1. [8-best-deep-learning-frameworks-for-data-science-enthusiasts](https://medium.com/the-mission/8-best-deep-learning-frameworks-for-data-science-enthusiasts-d72714157761)
# 1. [tensorflow](https://www.tensorflow.org/tutorials/keras/basic_classification)
# 1. [Theano](https://www.journaldev.com/17840/theano-python-tutorial)
# 1. [pytorch](https://pytorch.org/tutorials/beginner/blitz/tensor_tutorial.html#sphx-glr-beginner-blitz-tensor-tutorial-py)
# 1. [CNTK](https://github.com/Microsoft/CNTK/)
# 1. [arxiv](https://arxiv.org/abs/1607.01759)
# 
# -------------
# 
# ###### [Go to top](#top)

# You can follow and fork my work  in **GitHub**:
# > ###### [ GitHub](https://github.com/mjbahmani/10-steps-to-become-a-data-scientist)
# #### New Chapter Coming Soon, it is not completed yet. Following up
