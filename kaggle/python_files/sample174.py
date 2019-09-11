#!/usr/bin/env python
# coding: utf-8

# ### <div style="text-align: center">A Comprehensive Machine Learning Workflow with Python </div>
# 
# <div style="text-align: center">There are plenty of <b>courses and tutorials</b> that can help you learn machine learning from scratch but here in <b>Kaggle</b>, I want to solve <font color="red"><b>Titanic competition</b></font>  a popular machine learning Dataset as a comprehensive workflow with python packages. 
# After reading, you can use this workflow to solve other real problems and use it as a template to deal with <b>machine learning</b> problems.</div>
# <div style="text-align:center">last update: <b>17/02/2019</b></div>
# 
# 
# 
# > You are reading **10 Steps to Become a Data Scientist** and are now in the 9th step : 
# 
# 1. [Leren Python](https://www.kaggle.com/mjbahmani/the-data-scientist-s-toolbox-tutorial-1)
# 2. [Python Packages](https://www.kaggle.com/mjbahmani/the-data-scientist-s-toolbox-tutorial-2)
# 3. [Mathematics and Linear Algebra](https://www.kaggle.com/mjbahmani/linear-algebra-for-data-scientists)
# 4. [Programming &amp; Analysis Tools](https://www.kaggle.com/mjbahmani/20-ml-algorithms-15-plot-for-beginners)
# 5. [Big Data](https://www.kaggle.com/mjbahmani/a-data-science-framework-for-quora)
# 6. [Data visualization](https://www.kaggle.com/mjbahmani/top-5-data-visualization-libraries-tutorial)
# 7. [Data Cleaning](https://www.kaggle.com/mjbahmani/machine-learning-workflow-for-house-prices)
# 8. [How to solve a Problem?](https://www.kaggle.com/mjbahmani/the-data-scientist-s-toolbox-tutorial-2)
# 9. <font color="red">You are in the ninth step</font>
# 10. [Deep Learning](https://www.kaggle.com/mjbahmani/top-5-deep-learning-frameworks-tutorial)
# 
# ---------------------------------------------------------------------
# you can Fork and Run this kernel on <font color="red">Github</font>:
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
#      1. [Courses](#11)
#      1. [Kaggle kernels](#12)
#      1. [Ebooks](#13)
#      1. [CheatSheet](#14)
# 1. [Machine learning](#2)
#     1. [Machine learning workflow](#21)
#     1. [Real world Application Vs Competitions](#22)
# 1. [Problem Definition](#3)
#     1. [Problem feature](#31)
#         1. [Why am I  using Titanic dataset](#331)
#     1. [Aim](#32)
#     1. [Variables](#33)
#         1. [Types of Features](#331)
#             1. [Categorical](#3311)
#             1. [Ordinal](#3312)
#             1. [Continous](#3313)
# 1. [ Inputs & Outputs](#4)
#     1. [Inputs ](#41)
#     1. [Outputs](#42)
# 1. [Installation](#5)
#     1. [ jupyter notebook](#51)
#         1. [What browsers are supported?](#511)
#     1. [ kaggle kernel](#52)
#     
#     1. [Colab notebook](#53)
#     1. [install python & packages](#54)
#     1. [Loading Packages](#55)
# 1. [Exploratory data analysis](#6)
#     1. [Data Collection](#61)
#     1. [Visualization](#62)
#         1. [Scatter plot](#621)
#         1. [Box](#622)
#         1. [Histogram](#623)
#         1. [Multivariate Plots](#624)
#         1. [Violinplots](#625)
#         1. [Pair plot](#626)
#         1. [Kde plot](#627)
#         1. [Joint plot](#628)
#         1. [Andrews curves](#629)
#         1. [Heatmap](#6210)
#         1. [Radviz](#6211)
#     1. [Data Preprocessing](#63)
#         1. [Features](#631)
#         1. [Explorer Dataset](#632)
#     1. [Data Cleaning](#64)
#         1. [Transforming Features](#641)
#         1. [Feature Encoding](#642)
# 1. [Model Deployment](#7)
#     1. [Families of ML algorithms](#71)
#     1. [Prepare Features & Targets](#72)
#     1. [how to prevent overfitting &  underfitting?](#73)
#     1. [Accuracy and precision](#74)
#     1. [RandomForestClassifier](#74)
#         1. [prediction](#741)
#     1. [XGBoost](#75)
#         1. [prediction](#751)
#     1. [Logistic Regression](#76)
#         1. [prediction](#761)
#     1. [DecisionTreeRegressor ](#77)
#     1. [HuberRegressor](#78)
#     1. [ExtraTreeRegressor](#79)
#     1. [How do I submit?](#710)
# 1. [Conclusion](#8)
# 1. [References](#9)

#  <a id="1"></a> <br>
#  <br>
# ## 1- Introduction
# This is a **comprehensive ML techniques with python** , that I have spent for more than two months to complete it.
# 
# It is clear that everyone in this community is familiar with Titanic dataset but if you need to review your information about the dataset please visit this [link](https://www.kaggle.com/c/titanic/data).
# 
# I have tried to help **beginners**  in Kaggle how to face machine learning problems. and I think it is a great opportunity for who want to learn machine learning workflow with python completely.
# I have covered most of the methods that are implemented for **Titanic** until **2018**, you can start to learn and review your knowledge about ML with a perfect dataset and try to learn and memorize the workflow for your journey in Data science world.

#  <a id="11"></a> <br>
#  <br>
# ## 1-1 Courses
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

#  <a id="12"></a> <br>
#  <br>
# ## 1-2 Kaggle kernels
# I want to thanks **Kaggle team**  and  all of the **kernel's authors**  who develop this huge resources for Data scientists. I have learned from The work of others and I have just listed some more important kernels that inspired my work and I've used them in this kernel:
# 
# 1. [https://www.kaggle.com/ash316/eda-to-prediction-dietanic](https://www.kaggle.com/ash316/eda-to-prediction-dietanic)
# 
# 2. [https://www.kaggle.com/mrisdal/exploring-survival-on-the-titanic](https://www.kaggle.com/mrisdal/exploring-survival-on-the-titanic)
# 
# 3. [https://www.kaggle.com/yassineghouzam/titanic-top-4-with-ensemble-modeling](https://www.kaggle.com/yassineghouzam/titanic-top-4-with-ensemble-modeling)
# 
# 4. [https://www.kaggle.com/ldfreeman3/a-data-science-framework-to-achieve-99-accuracy](https://www.kaggle.com/ldfreeman3/a-data-science-framework-to-achieve-99-accuracy)
# 
# 5. [https://www.kaggle.com/startupsci/titanic-data-science-solutions](https://www.kaggle.com/startupsci/titanic-data-science-solutions)
# 6. [scikit-learn-ml-from-start-to-finish](https://www.kaggle.com/jeffd23/scikit-learn-ml-from-start-to-finish)
# <br>
# [go to top](#top)

#  <a id="13"></a> <br>
#  <br>
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

#  <a id="14"></a> <br>
#  <br>
# ## 1-4 Cheat Sheets
# Data Science is an ever-growing field, there are numerous tools & techniques to remember. It is not possible for anyone to remember all the functions, operations and formulas of each concept. That’s why we have cheat sheets. But there are a plethora of cheat sheets available out there, choosing the right cheat sheet is a tough task.
# 
# [Top 28 Cheat Sheets for Machine Learning](https://www.analyticsvidhya.com/blog/2017/02/top-28-cheat-sheets-for-machine-learning-data-science-probability-sql-big-data/)
# <br>
# ###### [Go to top](#top)

# <a id="2"></a> <br>
# ## 2- Machine Learning
# Machine Learning is a field of study that gives computers the ability to learn without being explicitly programmed.
# 
# **Arthur	Samuel, 1959**

#  <a id="21"></a> <br>
# ## 2-1 Machine Learning Workflow
# 
# If you have already read some [machine learning books](https://github.com/mjbahmani/10-steps-to-become-a-data-scientist/tree/master/Ebooks). You have noticed that there are different ways to stream data into machine learning.
# 
# Most of these books share the following steps:
# 1. Define Problem
# 1. Specify Inputs & Outputs
# 1. Exploratory data analysis
# 1. Data Collection
# 1. Data Preprocessing
# 1. Data Cleaning
# 1. Visualization
# 1. Model Design, Training, and Offline Evaluation
# 1. Model Deployment, Online Evaluation, and Monitoring
# 1. Model Maintenance, Diagnosis, and Retraining
# 
# Of course, the same solution can not be provided for all problems, so the best way is to create a **general framework** and adapt it to new problem.
# 
# **You can see my workflow in the below image** :
# 
#  <img src="http://s8.picofile.com/file/8344100018/workflow3.png" />
# 
# **Data Science has so many techniques and procedures that can confuse anyone.**

#  <a id="22"></a> <br>
# ## 2-1 Real world Application Vs Competitions
# We all know that there are differences between real world problem and competition problem. The following figure that is taken from one of the courses in coursera, has partly made this comparison 
# 
# <img src="http://s9.picofile.com/file/8339956300/reallife.png" height="600" width="500" />
# 
# As you can see, there are a lot more steps to solve  in real problems.
# ###### [Go to top](#top)

# <a id="3"></a> 
# <br>
# ## 3- Problem Definition
# I think one of the important things when you start a new machine learning project is Defining your problem. that means you should understand business problem.( **Problem Formalization**)
# 
# Problem Definition has four steps that have illustrated in the picture below:
# <img src="http://s8.picofile.com/file/8344103134/Problem_Definition2.png" width=400 height=400>

# <a id="31"></a>
# <br>
# ## 3-1 Problem Feature
# The sinking of the Titanic is one of the most infamous shipwrecks in history. **On April 15, 1912**, during her maiden voyage, the Titanic sank after colliding with an iceberg, killing **1502 out of 2224** passengers and crew. That's why the name DieTanic. This is a very unforgetable disaster that no one in the world can forget.
# 
# It took about $7.5 million to build the Titanic and it sunk under the ocean due to collision. The Titanic Dataset is a very good dataset for begineers to start a journey in data science and participate in competitions in Kaggle.
# 
# ٌWe will use the classic titanic data set. This dataset contains information about **11 different variables**:
# <img src="http://s9.picofile.com/file/8340453092/Titanic_feature.png" height="500" width="500">
# 
# 1. Survival
# 1. Pclass
# 1. Name
# 1. Sex
# 1. Age
# 1. SibSp
# 1. Parch
# 1. Ticket
# 1. Fare
# 1. Cabin
# 1. Embarked
# 
# > <font color="red"><b>Note :</b></font>
# You must answer the following question:
# How does your company expact to use and benfit from your model.

# <a id="331"></a> 
# <br>
# ### 3-3-1 Why am I  using Titanic dataset
# 
# 1. This is a good project because it is so well understood.
# 
# 1. Attributes are numeric and categorical so you have to figure out how to load and handle data.
# 
# 1. It is a ML problem, allowing you to practice with perhaps an easier type of supervised learning algorithm.
# 
# 1. We can define problem as clustering(unsupervised algorithm) project too.
# 
# 1. Because we love   **Kaggle** :-) .
# 
# <a id="32"></a> <br>
# ### 3-2 Aim
# It is your job to predict if a **passenger** survived the sinking of the Titanic or not.  For each PassengerId in the test set, you must predict a 0 or 1 value for the Survived variable.

# <a id="33"></a> <br>
# ### 3-3 Variables
# 
# 1. **Age** :
#     1. Age is fractional if less than 1. If the age is estimated, is it in the form of xx.5
# 
# 1. **Sibsp** :
#     1. The dataset defines family relations in this way...
# 
#         a. Sibling = brother, sister, stepbrother, stepsister
# 
#         b. Spouse = husband, wife (mistresses and fiancés were ignored)
# 
# 1. **Parch**:
#     1. The dataset defines family relations in this way...
# 
#         a. Parent = mother, father
# 
#         b. Child = daughter, son, stepdaughter, stepson
# 
#         c. Some children travelled only with a nanny, therefore parch=0 for them.
# 
# 1. **Pclass** :
#     *  A proxy for socio-economic status (SES)
#         * 1st = Upper
#         * 2nd = Middle
#         * 3rd = Lower
# 1. **Embarked** :
#      * nominal datatype 
# 1. **Name**: 
#     * nominal datatype . It could be used in feature engineering to derive the gender from title
# 1. **Sex**: 
#    * nominal datatype 
# 1. **Ticket**:
#     * that have no impact on the outcome variable. Thus, they will be excluded from analysis
# 1. **Cabin**: 
#     * is a nominal datatype that can be used in feature engineering
# 1.  **Fare**:
#     * Indicating the fare
# 1. **PassengerID**:
#     * have no impact on the outcome variable. Thus, it will be excluded from analysis
# 1. **Survival**:
#     * **[dependent variable](http://www.dailysmarty.com/posts/difference-between-independent-and-dependent-variables-in-machine-learning)** , 0 or 1

# <a id="331"></a> <br>
# ### 3-3-1  Types of Features
# <a id="3311"></a> <br>
# ### 3-3-1-1 Categorical
# 
# A categorical variable is one that has two or more categories and each value in that feature can be categorised by them. for example, gender is a categorical variable having two categories (male and female). Now we cannot sort or give any ordering to such variables. They are also known as Nominal Variables.
# 
# 1. **Categorical Features in the dataset: Sex,Embarked.**
# 
# <a id="3312"></a> <br>
# ### 3-3-1-2 Ordinal
# An ordinal variable is similar to categorical values, but the difference between them is that we can have relative ordering or sorting between the values. For eg: If we have a feature like Height with values Tall, Medium, Short, then Height is a ordinal variable. Here we can have a relative sort in the variable.
# 
# 1. **Ordinal Features in the dataset: PClass**
# 
# <a id="3313"></a> <br>
# ### 3-3-1-3 Continous:
# A feature is said to be continous if it can take values between any two points or between the minimum or maximum values in the features column.
# 
# 1. **Continous Features in the dataset: Age**
# 
# 
# <br>
# ###### [Go to top](#top)

# <a id="4"></a> <br>
# ## 4- Inputs & Outputs
# <a id="41"></a> <br>
# ### 4-1 Inputs
# What's our input for this problem:
#     1. train.csv
#     1. test.csv
# <a id="42"></a> <br>
# ### 4-2 Outputs
# 1. Your score is the percentage of passengers you correctly predict. This is known simply as "**accuracy**”.
# 
# 
# The Outputs should have exactly **2 columns**:
# 
#     1. PassengerId (sorted in any order)
#     1. Survived (contains your binary predictions: 1 for survived, 0 for deceased)
# 

# <a id="5"></a> <br>
# ## 5-Installation
# #### Windows:
# 1. Anaconda (from https://www.continuum.io) is a free Python distribution for SciPy stack. It is also available for Linux and Mac.
# 1. Canopy (https://www.enthought.com/products/canopy/) is available as free as well as commercial distribution with full SciPy stack for Windows, Linux and Mac.
# 1. Python (x,y) is a free Python distribution with SciPy stack and Spyder IDE for Windows OS. (Downloadable from http://python-xy.github.io/)
# 
# #### Linux:
# 1. Package managers of respective Linux distributions are used to install one or more packages in SciPy stack.
# 
# 1. For Ubuntu Users:
# sudo apt-get install python-numpy python-scipy python-matplotlibipythonipythonnotebook
# python-pandas python-sympy python-nose

# <a id="51"></a> <br>
# ## 5-1 Jupyter notebook
# I strongly recommend installing **Python** and **Jupyter** using the **[Anaconda Distribution](https://www.anaconda.com/download/)**, which includes Python, the Jupyter Notebook, and other commonly used packages for scientific computing and data science.
# 
# 1. First, download Anaconda. We recommend downloading Anaconda’s latest Python 3 version.
# 
# 2. Second, install the version of Anaconda which you downloaded, following the instructions on the download page.
# 
# 3. Congratulations, you have installed Jupyter Notebook! To run the notebook, run the following command at the Terminal (Mac/Linux) or Command Prompt (Windows):

# > jupyter notebook
# > 

# <a id="52"></a> <br>
# ## 5-2 Kaggle Kernel
# Kaggle kernel is an environment just like you use jupyter notebook, it's an **extension** of the where in you are able to carry out all the functions of jupyter notebooks plus it has some added tools like forking et al.

# <a id="53"></a> <br>
# ## 5-3 Colab notebook
# **Colaboratory** is a research tool for machine learning education and research. It’s a Jupyter notebook environment that requires no setup to use.
# <a id="531"></a> <br>
# ### 5-3-1 What browsers are supported?
# Colaboratory works with most major browsers, and is most thoroughly tested with desktop versions of Chrome and Firefox.
# <a id="532"></a> <br>
# ### 5-3-2 Is it free to use?
# Yes. Colaboratory is a research project that is free to use.
# <a id="533"></a> <br>
# ### 5-3-3 What is the difference between Jupyter and Colaboratory?
# Jupyter is the open source project on which Colaboratory is based. Colaboratory allows you to use and share Jupyter notebooks with others without having to download, install, or run anything on your own computer other than a browser.
# ###### [Go to top](#top)

# <a id="55"></a> <br>
# ## 5-5 Loading Packages
# In this kernel we are using the following packages:

#  <img src="http://s8.picofile.com/file/8338227868/packages.png" width=400  height=400>
# 

# ### 5-5-1 Import

# In[ ]:


from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
from pandas import get_dummies
import matplotlib as mpl
import xgboost as xgb
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib
import warnings
import sklearn
import scipy
import numpy
import json
import sys
import csv
import os

# ### 5-5-2 Version

# In[ ]:


print('matplotlib: {}'.format(matplotlib.__version__))
print('sklearn: {}'.format(sklearn.__version__))
print('scipy: {}'.format(scipy.__version__))
print('seaborn: {}'.format(sns.__version__))
print('pandas: {}'.format(pd.__version__))
print('numpy: {}'.format(np.__version__))
print('Python: {}'.format(sys.version))

# ### 5-5-2 Setup
# 
# A few tiny adjustments for better **code readability**

# In[ ]:


sns.set(style='white', context='notebook', palette='deep')
pylab.rcParams['figure.figsize'] = 12,8
warnings.filterwarnings('ignore')
mpl.style.use('ggplot')
sns.set_style('white')

# <a id="6"></a> <br>
# ## 6- Exploratory Data Analysis(EDA)
#  In this section, you'll learn how to use graphical and numerical techniques to begin uncovering the structure of your data. 
#  
# * Which variables suggest interesting relationships?
# * Which observations are unusual?
# * Analysis of the features!
# 
# By the end of the section, you'll be able to answer these questions and more, while generating graphics that are both **insightful** and **beautiful**.  then We will review analytical and statistical operations:
# 
# *   5-1 Data Collection
# *   5-2 Visualization
# *   5-3 Data Preprocessing
# *   5-4 Data Cleaning
# <img src="http://s9.picofile.com/file/8338476134/EDA.png">
# 
#  ><font color="red"><b>Note:</b></font>
#  You can change the order of the above steps.

# <a id="61"></a> <br>
# ## 6-1 Data Collection
# **Data collection** is the process of gathering and measuring data, information or any variables of interest in a standardized and established manner that enables the collector to answer or test hypothesis and evaluate outcomes of the particular collection.[techopedia]
# <br>
# I start Collection Data by the training and testing datasets into Pandas DataFrames
# ###### [Go to top](#top)

# In[ ]:


# import train and test to play with it
df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')

# ><font color="red"><b>Note: </b></font>
# 
# * Each **row** is an observation (also known as : sample, example, instance, record)
# * Each **column** is a feature (also known as: Predictor, attribute, Independent Variable, input, regressor, Covariate)

# After loading the data via **pandas**, we should checkout what the content is, description and via the following:

# In[ ]:


type(df_train)

# In[ ]:


type(df_test)

# <a id="62"></a> <br>
# ## 6-2 Visualization
# **Data visualization**  is the presentation of data in a pictorial or graphical format. It enables decision makers to see analytics presented visually, so they can grasp difficult concepts or identify new patterns.
# 
# With interactive visualization, you can take the concept a step further by using technology to drill down into charts and graphs for more detail, interactively changing what data you see and how it’s processed.[SAS]
# 
#  In this section I show you  **11 plots** with **matplotlib** and **seaborn** that is listed in the blew picture:
#  <img src="http://s8.picofile.com/file/8338475500/visualization.jpg" width=400 height=400 />
# 
# ###### [Go to top](#top)

# <a id="621"></a> <br>
# ### 6-2-1 Scatter plot
# 
# Scatter plot Purpose To identify the type of relationship (if any) between two quantitative variables
# 
# 
# 

# In[ ]:


# Modify the graph above by assigning each species an individual color.
g = sns.FacetGrid(df_train, hue="Survived", col="Pclass", margin_titles=True,
                  palette={1:"seagreen", 0:"gray"})
g=g.map(plt.scatter, "Fare", "Age",edgecolor="w").add_legend();

# In[ ]:


plt.figure(figsize=(8,6))
plt.scatter(range(df_train.shape[0]), np.sort(df_train['Age'].values))
plt.xlabel('index')
plt.ylabel('Survived')
plt.title('Explore: Age')
plt.show()

# <a id="622"></a> <br>
# ### 6-2-2 Box
# In descriptive statistics, a **box plot** or boxplot is a method for graphically depicting groups of numerical data through their quartiles. Box plots may also have lines extending vertically from the boxes (whiskers) indicating variability outside the upper and lower quartiles, hence the terms box-and-whisker plot and box-and-whisker diagram.[wikipedia]

# In[ ]:


ax= sns.boxplot(x="Pclass", y="Age", data=df_train)
ax= sns.stripplot(x="Pclass", y="Age", data=df_train, jitter=True, edgecolor="gray")
plt.show()

# <a id="623"></a> <br>
# ### 6-2-3 Histogram
# We can also create a **histogram** of each input variable to get an idea of the distribution.
# 
# 

# In[ ]:


# histograms
df_train.hist(figsize=(15,20));
plt.figure();

# It looks like perhaps two of the input variables have a Gaussian distribution. This is useful to note as we can use algorithms that can exploit this assumption.
# 
# 

# In[ ]:


df_train["Age"].hist();

# In[ ]:


f,ax=plt.subplots(1,2,figsize=(20,10))
df_train[df_train['Survived']==0].Age.plot.hist(ax=ax[0],bins=20,edgecolor='black',color='red')
ax[0].set_title('Survived= 0')
x1=list(range(0,85,5))
ax[0].set_xticks(x1)
df_train[df_train['Survived']==1].Age.plot.hist(ax=ax[1],color='green',bins=20,edgecolor='black')
ax[1].set_title('Survived= 1')
x2=list(range(0,85,5))
ax[1].set_xticks(x2)
plt.show()

# In[ ]:


f,ax=plt.subplots(1,2,figsize=(18,8))
df_train['Survived'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True)
ax[0].set_title('Survived')
ax[0].set_ylabel('')
sns.countplot('Survived',data=df_train,ax=ax[1])
ax[1].set_title('Survived')
plt.show()

# In[ ]:


f,ax=plt.subplots(1,2,figsize=(18,8))
df_train[['Sex','Survived']].groupby(['Sex']).mean().plot.bar(ax=ax[0])
ax[0].set_title('Survived vs Sex')
sns.countplot('Sex',hue='Survived',data=df_train,ax=ax[1])
ax[1].set_title('Sex:Survived vs Dead')
plt.show()

# In[ ]:


sns.countplot('Pclass', hue='Survived', data=df_train)
plt.title('Pclass: Sruvived vs Dead')
plt.show()

# <a id="624"></a> <br>
# ### 6-2-4 Multivariate Plots
# Now we can look at the interactions between the variables.
# 
# First, let’s look at scatterplots of all pairs of attributes. This can be helpful to spot structured relationships between input variables.

# In[ ]:



# scatter plot matrix
pd.plotting.scatter_matrix(df_train,figsize=(10,10))
plt.figure();

# Note the diagonal grouping of some pairs of attributes. This suggests a high correlation and a predictable relationship.

# <a id="625"></a> <br>
# ### 6-2-5 violinplots

# In[ ]:


# violinplots on petal-length for each species
sns.violinplot(data=df_train,x="Sex", y="Age")

# In[ ]:


f,ax=plt.subplots(1,2,figsize=(18,8))
sns.violinplot("Pclass","Age", hue="Survived", data=df_train,split=True,ax=ax[0])
ax[0].set_title('Pclass and Age vs Survived')
ax[0].set_yticks(range(0,110,10))
sns.violinplot("Sex","Age", hue="Survived", data=df_train,split=True,ax=ax[1])
ax[1].set_title('Sex and Age vs Survived')
ax[1].set_yticks(range(0,110,10))
plt.show()

# <a id="626"></a> <br>
# ### 6-2-6 pairplot

# In[ ]:


# Using seaborn pairplot to see the bivariate relation between each pair of features
sns.pairplot(df_train, hue="Sex");

# <a id="627"></a> <br>
# ###  6-2-7 kdeplot

# We can also replace the histograms shown in the diagonal of the pairplot by kde.

# In[ ]:


sns.FacetGrid(df_train, hue="Survived", size=5).map(sns.kdeplot, "Fare").add_legend()
plt.show();

# <a id="628"></a> <br>
# ### 6-2-8 jointplot

# In[ ]:


sns.jointplot(x='Fare',y='Age',data=df_train);

# In[ ]:


sns.jointplot(x='Fare',y='Age' ,data=df_train, kind='reg');

# <a id="629"></a> <br>
# ###  6-2-9 Swarm plot

# In[ ]:


sns.swarmplot(x='Pclass',y='Age',data=df_train);

# <a id="6210"></a> <br>
# ### 6-2-10 Heatmap

# In[ ]:


plt.figure(figsize=(7,4)) 
sns.heatmap(df_train.corr(),annot=True,cmap='cubehelix_r') #draws  heatmap with input as the correlation matrix calculted by(iris.corr())
plt.show();

# ###  6-2-11 Bar Plot

# In[ ]:


df_train['Pclass'].value_counts().plot(kind="bar");

# ### 6-2-12 Factorplot

# In[ ]:


sns.factorplot('Pclass','Survived',hue='Sex',data=df_train)
plt.show();

# In[ ]:


sns.factorplot('SibSp','Survived',hue='Pclass',data=df_train)
plt.show()

# In[ ]:


#let's see some others factorplot
f,ax=plt.subplots(1,2,figsize=(20,8))
sns.barplot('SibSp','Survived', data=df_train,ax=ax[0])
ax[0].set_title('SipSp vs Survived in BarPlot')
sns.factorplot('SibSp','Survived', data=df_train,ax=ax[1])
ax[1].set_title('SibSp vs Survived in FactorPlot')
plt.close(2)
plt.show();

# ### 6-2-13 distplot

# In[ ]:


f,ax=plt.subplots(1,3,figsize=(20,8))
sns.distplot(df_train[df_train['Pclass']==1].Fare,ax=ax[0])
ax[0].set_title('Fares in Pclass 1')
sns.distplot(df_train[df_train['Pclass']==2].Fare,ax=ax[1])
ax[1].set_title('Fares in Pclass 2')
sns.distplot(df_train[df_train['Pclass']==3].Fare,ax=ax[2])
ax[2].set_title('Fares in Pclass 3')
plt.show()

# ### 6-2-12 Conclusion
# We have used Python to apply data visualization tools to theTitanic dataset.

# <a id="63"></a> <br>
# ## 6-3 Data Preprocessing
# **Data preprocessing** refers to the transformations applied to our data before feeding it to the algorithm.
#  
# Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
# there are plenty of steps for data preprocessing and **we just listed some of them** :
# * removing Target column (id)
# * Sampling (without replacement)
# * Dealing with Imbalanced Data
# * Introducing missing values and treating them (replacing by average values)
# * Noise filtering
# * Data discretization
# * Normalization and standardization
# * PCA analysis
# * Feature selection (filter, embedded, wrapper)
# 
# ###### [Go to top](#top)

# <a id="631"></a> <br>
# ## 6-3-1 Features
# Features:
# * numeric
# * categorical
# * ordinal
# * datetime
# * coordinates
# 
# ### Find the type of features in titanic dataset:
# <img src="http://s9.picofile.com/file/8339959442/titanic.png" height="700" width="600" />

# <a id="632"></a> <br>
# ## 6-3-2 Explorer Dataset
# 1- Dimensions of the dataset.
# 
# 2- Peek at the data itself.
# 
# 3- Statistical summary of all attributes.
# 
# 4- Breakdown of the data by the class variable.[7]
# 
# Don’t worry, each look at the data is **one command**. These are useful commands that you can use again and again on future projects.
# 
# ###### [Go to top](#top)

# In[ ]:


# shape
print(df_train.shape)

# In[ ]:


#columns*rows
df_train.size

# >  <font color="red"><b>Note:</b></font>
# how many NA elements in every column
# 

# In[ ]:


##df_train.isnull().sum()

# In[ ]:


def check_missing_data(df):
    flag=df.isna().sum().any()
    if flag==True:
        total = df.isnull().sum()
        percent = (df.isnull().sum())/(df.isnull().count()*100)
        output = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
        data_type = []
        # written by MJ Bahmani
        for col in df.columns:
            dtype = str(df[col].dtype)
            data_type.append(dtype)
        output['Types'] = data_type
        return(np.transpose(output))
    else:
        return(False)

# In[ ]:


check_missing_data(df_train)

# In[ ]:


check_missing_data(df_test)

# If you want to remove all the null value, you can uncomment this line

# In[ ]:


# remove rows that have NA's
#train = train.dropna()

# 
# We can get a quick idea of how many instances (rows) and how many attributes (columns) the data contains with the shape property.
# 
# You should see 891 instances and 12 attributes:

# In[ ]:


print(df_train.shape)

# >  <font color="red"><b>Note:</b></font>
# for getting some information about the dataset you can use **info()** command

# In[ ]:


print(df_train.info())

# >  <font color="red"><b>Note:</b></font>
# you see number of unique item for **Age** and **Pclass** with command below:

# In[ ]:


df_train['Age'].unique()

# In[ ]:


df_train["Pclass"].value_counts()


# To check the first 5 rows of the data set, we can use head(5).

# In[ ]:


df_train.head(5) 

# To check out last 5 row of the data set, we use tail() function

# In[ ]:


df_train.tail() 

# To pop up 5 random rows from the data set, we can use **sample(5)**  function

# In[ ]:


df_train.sample(5) 

# To give a statistical summary about the dataset, we can use **describe()

# In[ ]:


df_train.describe() 

# To check out how many null info are on the dataset, we can use **isnull().sum()

# In[ ]:


df_train.isnull().sum()

# In[ ]:


df_train.groupby('Pclass').count()

# To print dataset **columns**, we can use columns atribute

# In[ ]:


df_train.columns

# >  <font color="red"><b>Note:</b></font>
# in pandas's data frame you can perform some query such as "where"

# In[ ]:


df_train.where(df_train ['Age']==30).head(2)

# As you can see in the below in python, it is so easy perform some query on the dataframe:

# In[ ]:


df_train[df_train['Age']<7.2].head(2)

# Seperating the data into dependent and independent variables

# In[ ]:


X = df_train.iloc[:, :-1].values
y = df_train.iloc[:, -1].values

# >  <font color="red"><b>Note:</b></font>
# Preprocessing and generation pipelines depend on a model type

# <a id="64"></a> <br>
# ## 6-4 Data Cleaning 
# 1. When dealing with real-world data,** dirty data** is the norm rather than the exception. 
# 1. We continuously need to predict correct values, impute missing ones, and find links between various data artefacts such as schemas and records. 
# 1. We need to stop treating data cleaning as a piecemeal exercise (resolving different types of errors in isolation), and instead leverage all signals and resources (such as constraints, available statistics, and dictionaries) to accurately predict corrective actions.
# 1. The primary goal of data cleaning is to detect and remove errors and **anomalies** to increase the value of data in analytics and decision making.[8]
# 
# ###### [Go to top](#top)

# <a id="641"></a> <br>
# ## 6-4-1 Transforming Features
# Data transformation is the process of converting data from one format or structure into another format or structure[[wiki](https://en.wikipedia.org/wiki/Data_transformation)] 
# 1. Age
# 1. Cabin
# 1. Fare
# 1. Name

# In[ ]:



def simplify_ages(df):
    df.Age = df.Age.fillna(-0.5)
    bins = (-1, 0, 5, 12, 18, 25, 35, 60, 120)
    group_names = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']
    categories = pd.cut(df.Age, bins, labels=group_names)
    df.Age = categories
    return df

def simplify_cabins(df):
    df.Cabin = df.Cabin.fillna('N')
    df.Cabin = df.Cabin.apply(lambda x: x[0])
    return df

def simplify_fares(df):
    df.Fare = df.Fare.fillna(-0.5)
    bins = (-1, 0, 8, 15, 31, 1000)
    group_names = ['Unknown', '1_quartile', '2_quartile', '3_quartile', '4_quartile']
    categories = pd.cut(df.Fare, bins, labels=group_names)
    df.Fare = categories
    return df

def format_name(df):
    df['Lname'] = df.Name.apply(lambda x: x.split(' ')[0])
    df['NamePrefix'] = df.Name.apply(lambda x: x.split(' ')[1])
    return df    
    
def drop_features(df):
    return df.drop(['Ticket', 'Name', 'Embarked'], axis=1)

def transform_features(df):
    df = simplify_ages(df)
    df = simplify_cabins(df)
    df = simplify_fares(df)
    df = format_name(df)
    df = drop_features(df)
    return df

df_train = transform_features(df_train)
df_test = transform_features(df_test)
df_train.head()

# <a id="642"></a> <br>
# ## 6-4-2 Feature Encoding
# In machine learning projects, one important part is feature engineering. It is very common to see categorical features in a dataset. However, our machine learning algorithm can only read numerical values. It is essential to encoding categorical features into numerical values[28]
# 1. Encode labels with value between 0 and n_classes-1
# 1. LabelEncoder can be used to normalize labels.
# 1. It can also be used to transform non-numerical labels (as long as they are hashable and comparable) to numerical labels.

# In[ ]:


def encode_features(df_train, df_test):
    features = ['Fare', 'Cabin', 'Age', 'Sex', 'Lname', 'NamePrefix']
    df_combined = pd.concat([df_train[features], df_test[features]])
    
    for feature in features:
        le = preprocessing.LabelEncoder()
        le = le.fit(df_combined[feature])
        df_train[feature] = le.transform(df_train[feature])
        df_test[feature] = le.transform(df_test[feature])
    return df_train, df_test

# <a id="7"></a> <br>
# ## 7- Model Deployment
# In this section have been applied plenty of  ** learning algorithms** that play an important rule in your experiences and improve your knowledge in case of ML technique.
# >  <font color="red"><b>Note:</b></font>
# The results shown here may be slightly different for your analysis because, for example, the neural network algorithms use random number generators for fixing the initial value of the weights (starting points) of the neural networks, which often result in obtaining slightly different (local minima) solutions each time you run the analysis. Also note that changing the seed for the random number generator used to create the train, test, and validation samples can change your results.

# 
# <a id="71"></a> <br>
# ## 7-1 Families of ML algorithms
# There are several categories for machine learning algorithms, below are some of these categories:
# * Linear
#     * Linear Regression
#     * Logistic Regression
#     * Support Vector Machines
# * Tree-Based
#     * Decision Tree
#     * Random Forest
#     * GBDT
# * KNN
# * Neural Networks
# 
# -----------------------------
# And if we  want to categorize ML algorithms with the type of learning, there are below type:
# * Classification
# 
#     * k-Nearest Neighbors
#     * LinearRegression
#     * SVM
#     * DT 
#     * NN
#     
# * clustering
# 
#     * K-means
#     * HCA
#     * Expectation Maximization
#     
# * Visualization and	dimensionality 	reduction:
# 
#     * Principal Component Analysis(PCA)
#     * Kernel PCA
#     * Locally -Linear	Embedding 	(LLE)
#     * t-distributed	Stochastic	NeighborEmbedding 	(t-SNE)
#     
# * Association rule learning
# 
#     * Apriori
#     * Eclat
# * Semisupervised learning
# * Reinforcement Learning
#     * Q-learning
# * Batch learning & Online learning
# * Ensemble  Learning
# 
# >  <font color="red"><b>Note:</b></font>
# Here is no method which outperforms all others for all tasks
# 
# ###### [Go to top](#top)

# <a id="72"></a> <br>
# ## 7-2 Prepare Features & Targets
# First of all seperating the data into independent(Feature) and dependent(Target) variables.
# 
# >  <font color="red"><b>Note:</b></font>
# * X==>> Feature - independent
# * y==>> Target    - dependent

# In[ ]:


#Encode Dataset
df_train, df_test = encode_features(df_train, df_test)
df_train.head()

# In[ ]:


df_test.head()

# <a id="73"></a> <br>
# ## 7-3 how to prevent overfitting &  underfitting?
# 
# <img src='https://cdn-images-1.medium.com/max/800/1*JZbxrdzabrT33Yl-LrmShw.png' width=500 height=500>
# 1. graph on the left side:
#     1. we can predict that the line does not cover all the points shown in the graph. Such model tend to cause underfitting of data .It also called High Bias.
# 
# 1. graph on right side:
#     1. shows the predicted line covers all the points in graph. In such condition you can also think that it’s a good graph which cover all the points. But that’s not actually true, the predicted line into the graph covers all points which are noise and outlier. Such model are also responsible to predict poor result due to its complexity.It is also called High Variance.
# 
# 1. middle graph:
#     1. it shows a pretty good predicted line. It covers majority of the point in graph and also maintains the balance between bias and variance.[30]

# Prepare X(features) , y(target)

# In[ ]:


x_all = df_train.drop(['Survived', 'PassengerId'], axis=1)
y_all = df_train['Survived']

# In[ ]:


num_test = 0.3
X_train, X_test, y_train, y_test = train_test_split(x_all, y_all, test_size=num_test, random_state=100)

# <a id="74"></a> <br>
# ## 7-4 Accuracy and precision
# We know that the titanic problem is a binary classification and to evaluate, we just need to calculate the accuracy.
# 
# 1. **accuracy**
# 
#     1. Your score is the percentage of passengers you correctly predict. This is known simply as "accuracy”.
# 
# 1. **precision** : 
# 
#     1. In pattern recognition, information retrieval and binary classification, precision (also called positive predictive value) is the fraction of relevant instances among the retrieved instances, 
# 1. **recall** : 
# 
#     1. recall is the fraction of relevant instances that have been retrieved over the total amount of relevant instances. 
# 1. **F-score** :
# 
#     1. the F1 score is a measure of a test's accuracy. It considers both the precision p and the recall r of the test to compute the score: p is the number of correct positive results divided by the number of all positive results returned by the classifier, and r is the number of correct positive results divided by the number of all relevant samples (all samples that should have been identified as positive). The F1 score is the harmonic average of the precision and recall, where an F1 score reaches its best value at 1 (perfect precision and recall) and worst at 0.
# 
# 1. **What is the difference between accuracy and precision?**
#     1. "Accuracy" and "precision" are general terms throughout science. A good way to internalize the difference are the common "bullseye diagrams". In machine learning/statistics as a whole, accuracy vs. precision is analogous to bias vs. variance.

# In[ ]:


result=None

# <a id="74"></a> <br>
# ## 7-4 RandomForestClassifier
# A random forest is a meta estimator that fits a number of decision tree classifiers on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting. The sub-sample size is always the same as the original input sample size but the samples are drawn with replacement if bootstrap=True (default).

# In[ ]:


# Choose the type of classifier. 
rfc = RandomForestClassifier()

# Choose some parameter combinations to try
parameters = {'n_estimators': [4, 6, 9], 
              'max_features': ['log2', 'sqrt','auto'], 
              'criterion': ['entropy', 'gini'],
              'max_depth': [2, 3, 5, 10], 
              'min_samples_split': [2, 3, 5],
              'min_samples_leaf': [1,5,8]
             }

# Type of scoring used to compare parameter combinations
acc_scorer = make_scorer(accuracy_score)

# Run the grid search
grid_obj = GridSearchCV(rfc, parameters, scoring=acc_scorer)
grid_obj = grid_obj.fit(X_train, y_train)

# Set the clf to the best combination of parameters
rfc = grid_obj.best_estimator_

# Fit the best algorithm to the data. 
rfc.fit(X_train, y_train)

# <a id="741"></a> <br>
# ## 7-4-1 prediction

# In[ ]:


rfc_prediction = rfc.predict(X_test)
rfc_score=accuracy_score(y_test, rfc_prediction)
print(rfc_score)

# <a id="75"></a> <br>
# ## 7-5 XGBoost
# [XGBoost](https://en.wikipedia.org/wiki/XGBoost) is an open-source software library which provides a gradient boosting framework for C++, Java, Python, R, and Julia. it aims to provide a "Scalable, Portable and Distributed Gradient Boosting (GBM, GBRT, GBDT) Library". 

# In[ ]:


xgboost = xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05).fit(X_train, y_train)

# <a id="751"></a> <br>
# ## 7-5-1 prediction

# In[ ]:


xgb_prediction = xgboost.predict(X_test)
xgb_score=accuracy_score(y_test, xgb_prediction)
print(xgb_score)

# <a id="76"></a> <br>
# ## 7-6 Logistic Regression
# the logistic model  is a widely used statistical model that, in its basic form, uses a logistic function to model a binary dependent variable; many more complex extensions exist. In regression analysis, logistic regression (or logit regression) is estimating the parameters of a logistic model

# In[ ]:


logreg = LogisticRegression()
logreg.fit(X_train, y_train)

# <a id="761"></a> <br>
# ## 7-6-1 prediction

# In[ ]:


logreg_prediction = logreg.predict(X_test)
logreg_score=accuracy_score(y_test, logreg_prediction)
print(logreg_score)


# <a id="77"></a> <br>
# ## 7-7 DecisionTreeRegressor
# The function to measure the quality of a split. Supported criteria are “mse” for the mean squared error, which is equal to variance reduction as feature selection criterion and minimizes the L2 loss using the mean of each terminal node, “friedman_mse”, which uses mean squared error with Friedman’s improvement score for potential splits, and “mae” for the mean absolute error, which minimizes the L1 loss using the median of each terminal node.

# In[ ]:


from sklearn.tree import DecisionTreeRegressor

# Define model. Specify a number for random_state to ensure same results each run
dt = DecisionTreeRegressor(random_state=1)



# In[ ]:


# Fit model
dt.fit(X_train, y_train)

# In[ ]:


dt_prediction = dt.predict(X_test)
dt_score=accuracy_score(y_test, dt_prediction)
print(dt_score)

# <a id="79"></a> <br>
# ## 7-9 ExtraTreeRegressor
# Extra Tree Regressor differ from classic decision trees in the way they are built. When looking for the best split to separate the samples of a node into two groups, random splits are drawn for each of the max_features randomly selected features and the best split among those is chosen. When max_features is set 1, this amounts to building a totally random decision tree.

# In[ ]:


from sklearn.tree import ExtraTreeRegressor
# Define model. Specify a number for random_state to ensure same results each run
etr = ExtraTreeRegressor()

# In[ ]:


# Fit model
etr.fit(X_train, y_train)

# In[ ]:


etr_prediction = etr.predict(X_test)
etr_score=accuracy_score(y_test, etr_prediction)
print(etr_score)

# <a id="710"></a> <br>
# ## 7-10 How do I submit?
# 1. Fork and Commit this Kernel.
# 1. Then navigate to the Output tab of the Kernel and "Submit to Competition".

# In[ ]:


X_train = df_train.drop("Survived",axis=1)
y_train = df_train["Survived"]

# In[ ]:


X_train = X_train.drop("PassengerId",axis=1)
X_test  = df_test.drop("PassengerId",axis=1)

# In[ ]:


xgboost = xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05).fit(X_train, y_train)

# In[ ]:


Y_pred = xgboost.predict(X_test)

# You can change your model and submit the results of other models

# In[ ]:


submission = pd.DataFrame({
        "PassengerId": df_test["PassengerId"],
        "Survived": Y_pred
    })
submission.to_csv('submission.csv', index=False)

# -----------------
# <a id="8"></a> <br>
# # 8- Conclusion
# I have tried to cover all the parts related to the process of **Machine Learning** with a variety of Python packages and I know that there are still some problems then I hope to get your feedback to improve it.

# You can Fork and Run this kernel on Github:
# > ###### [ GitHub](https://github.com/mjbahmani/Machine-Learning-Workflow-with-Python)
# 
# --------------------------------------
# 
#  **I hope you find this kernel helpful and some <font color="red"><b>UPVOTES</b></font> would be very much appreciated** 

# <a id="9"></a> <br>
# 
# -----------
# 
# # 9- References
# 1. [https://skymind.ai/wiki/machine-learning-workflow](https://skymind.ai/wiki/machine-learning-workflow)
# 
# 1. [Problem-define](https://machinelearningmastery.com/machine-learning-in-python-step-by-step/)
# 
# 1. [Sklearn](http://scikit-learn.org/)
# 
# 1. [machine-learning-in-python-step-by-step](https://machinelearningmastery.com/machine-learning-in-python-step-by-step/)
# 
# 1. [Data Cleaning](http://wp.sigmod.org/?p=2288)
# 
# 1. [competitive data science](https://www.coursera.org/learn/competitive-data-science/)
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
# 
# 1. [Machine Learning Certification by University of Washington](https://www.coursera.org/specializations/machine-learning)
# 
# 1. [Data Science and Machine Learning Bootcamp with R](https://www.udemy.com/data-science-and-machine-learning-bootcamp-with-r/)
# 
# 1. [Creative Applications of Deep Learning with TensorFlow](https://www.class-central.com/course/kadenze-creative-applications-of-deep-learning-with-tensorflow-6679)
# 
# 1. [Neural Networks for Machine Learning](https://www.class-central.com/mooc/398/coursera-neural-networks-for-machine-learning)
# 
# 1. [Practical Deep Learning For Coders, Part 1](https://www.class-central.com/mooc/7887/practical-deep-learning-for-coders-part-1)
# 
# 1. [Machine Learning](https://www.cs.ox.ac.uk/teaching/courses/2014-2015/ml/index.html)
# 
# 1. [https://www.kaggle.com/ash316/eda-to-prediction-dietanic](https://www.kaggle.com/ash316/eda-to-prediction-dietanic)
# 
# 1. [https://www.kaggle.com/mrisdal/exploring-survival-on-the-titanic](https://www.kaggle.com/mrisdal/exploring-survival-on-the-titanic)
# 
# 1. [https://www.kaggle.com/yassineghouzam/titanic-top-4-with-ensemble-modeling](https://www.kaggle.com/yassineghouzam/titanic-top-4-with-ensemble-modeling)
# 
# 1. [https://www.kaggle.com/ldfreeman3/a-data-science-framework-to-achieve-99-accuracy](https://www.kaggle.com/ldfreeman3/a-data-science-framework-to-achieve-99-accuracy)
# 
# 1. [https://www.kaggle.com/startupsci/titanic-data-science-solutions](https://www.kaggle.com/startupsci/titanic-data-science-solutions)
# 
# 1. [Top 28 Cheat Sheets for Machine Learning](https://www.analyticsvidhya.com/blog/2017/02/top-28-cheat-sheets-for-machine-learning-data-science-probability-sql-big-data/)
# 1. [xenonstack](https://www.xenonstack.com/blog/data-science/preparation-wrangling-machine-learning-deep/)
# 1. [towardsdatascience](https://towardsdatascience.com/encoding-categorical-features-21a2651a065c)
# 1. [train-test-split-and-cross-validation](https://towardsdatascience.com/train-test-split-and-cross-validation-in-python-80b61beca4b6)
# 1. [what-is-underfitting-and-overfitting](https://medium.com/greyatom/what-is-underfitting-and-overfitting-in-machine-learning-and-how-to-deal-with-it-6803a989c76)
# 1. [permutation-importance](https://www.kaggle.com/dansbecker/permutation-importance)
# 1. [partial-plots](https://www.kaggle.com/dansbecker/partial-plots)
# -------------
# 
# ###### [Go to top](#top)

# Go to first step: [**Course Home Page**](https://www.kaggle.com/mjbahmani/10-steps-to-become-a-data-scientist)
# 
# Go to next step : [**Mathematics and Linear Algebra**](https://www.kaggle.com/mjbahmani/linear-algebra-for-data-scientists)
# 
