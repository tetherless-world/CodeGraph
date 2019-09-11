#!/usr/bin/env python
# coding: utf-8

# ## <div style="text-align: center"> +20 ML Algorithms +15 Plot for Beginners</div>
# <div style="text-align: center"><b>Quite Practical and Far from any Theoretical Concepts</b></div>
# <img src='https://image.ibb.co/gbH3ue/iris.png'>
# <div style="text-align: center">[Image-Credit](https://medium.com/@jebaseelanravi96/machine-learning-iris-classification-33aa18a4a983)</div>
# <div style="text-align:center">last update: <b>12/02/2019</b></div>
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
# 
# ---------------------------------------------------------------------
# you can Fork and Run this kernel on Github:
# > ###### [ GitHub](https://github.com/mjbahmani)
# 
# -------------------------------------------------------------------------------------------------------------
# 
#  **I hope you find this kernel helpful and some <font color="red"><b>UPVOTES</b></font> would be very much appreciated**
#  
#  -----------

#  <a id="top"></a> <br>
# ## Notebook  Content
# *   1-  [Introduction](#1)
#     * [1-1 Courses](#11)
#     * [1-2 Ebooks](#12)
#     * [1-3 Cheat Sheets](#13)
# *   2- [Machine learning workflow](#2)
# *       2-1 [Real world Application Vs Competitions](#21)
# *   3- [Problem Definition](#3)
# *       3-1 [Problem feature](#31)
# *       3-2 [Aim](#32)
# *       3-3 [Variables](#33)
# *   4-[ Inputs & Outputs](#4)
# *   4-1 [Inputs ](#41)
# *   4-2 [Outputs](#42)
# *   5- [Loading Packages](#5)
# *   6- [Exploratory data analysis](#6)
# *       6-1 [Data Collection](#61)
# *       6-2 [Visualization](#62)
# *           6-2-1 [Scatter plot](#621)
# *           6-2-2 [Box](#622)
# *           6-2-3 [Histogram](#623)
# *           6-2-4 [Multivariate Plots](#624)
# *           6-2-5 [Violinplots](#625)
# *           6-2-6 [Pair plot](#626)
# *           6-2-7 [Kde plot](#627)
# *           6-2-8 [Joint plot](#628)
# *           6-2-9 [Andrews curves](#629)
# *           6-2-10 [Heatmap](#6210)
# *           6-2-11 [Radviz](#6211)
# *           6-2-12 [Bar Plot](#6212)
# *           6-2-13 [Visualization with Plotly](#6213)
# *           6-2-14 [Conclusion](#6214)
# *       6-3 [Data Preprocessing](#63)
# *           6-3-1 [Features](#631)
# *           6-3-2 [Explorer Dataset](#632)
# *       6-4 [Data Cleaning](#64)
# *   7- [Model Deployment](#7)
# *       7-1[ Families of ML algorithms](#71)
# *       7-2[ Prepare Features & Targets](#72)
# *       7-3[ Accuracy and precision](#73)
# *       7-4[ KNN](#74)
# *       7-5 [Radius Neighbors Classifier](#75)
# *       7-6 [Logistic Regression](#76)
# *       7-7 [Passive Aggressive Classifier](#77)
# *       7-8 [Naive Bayes](#78)
# *       7-9 [MultinomialNB](#79)
# *       7-10 [BernoulliNB](#710)
# *       7-11 [SVM](#711)
# *       7-12 [Nu-Support Vector Classification](#712)
# *       7-13 [Linear Support Vector Classification](#713)
# *       7-14 [Decision Tree](#714)
# *       7-15 [ExtraTreeClassifier](#715)
# *       7-16 [Neural network](#716)
# *            7-16-1 [What is a Perceptron?](#7161)
# *       7-17 [RandomForest](#717)
# *       7-18 [Bagging classifier ](#718)
# *       7-19 [AdaBoost classifier](#719)
# *       7-20 [Gradient Boosting Classifier](#720)
# *   8- [Conclusion](#8)
# *   9- [References](#9)

#  <a id="1"></a> <br>
# ## 1- Introduction
# This is a **comprehensive ML techniques with python** , that I have spent for more than two months to complete it.
# 
# it is clear that everyone in this community is familiar with IRIS dataset but if you need to review your information about the dataset please visit this [link](https://archive.ics.uci.edu/ml/datasets/iris).
# 
# I have tried to help **beginners**  in Kaggle how to face machine learning problems. and I think it is a great opportunity for who want to learn machine learning workflow with python completely.
# I have covered most of the methods that are implemented for iris until **2018**, you can start to learn and review your knowledge about ML with a simple dataset and try to learn and memorize the workflow for your journey in Data science world.
#  <a id="11"></a> <br>
# ## 1-1 Courses
# 
# There are alot of Online courses that can help you develop your knowledge, here I have just  listed some of them:
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
# 
# 
# 5- [https://www.kaggle.com/startupsci/titanic-data-science-solutions](https://www.kaggle.com/startupsci/titanic-data-science-solutions)
# 
#  <a id="12"></a> <br>
# ## 1-2 Ebooks
# So you love reading , here is **10 free machine learning books**
# 
# 1. [Probability and Statistics for Programmers](http://www.greenteapress.com/thinkstats/)
# 
# 1. [Bayesian Reasoning and Machine Learning](http://web4.cs.ucl.ac.uk/staff/D.Barber/textbook/091117.pdf)
# 
# 1. [An Introduction to Statistical Learning](http://www-bcf.usc.edu/~gareth/ISL/)
# 
# 1. [Understanding Machine Learning](http://www.cs.huji.ac.il/~shais/UnderstandingMachineLearning/index.html)
# 
# 1. [A Programmer’s Guide to Data Mining](http://guidetodatamining.com/)
# 
# 1. [Mining of Massive Datasets](http://infolab.stanford.edu/~ullman/mmds/book.pdf)
# 
# 1. [A Brief Introduction to Neural Networks](http://www.dkriesel.com/_media/science/neuronalenetze-en-zeta2-2col-dkrieselcom.pdf)
# 
# 1. [Deep Learning](http://www.deeplearningbook.org/)
# 
# 1. [Natural Language Processing with Python](https://www.researchgate.net/publication/220691633_Natural_Language_Processing_with_Python)
# 
# 1. [Machine Learning Yearning](http://www.mlyearning.org/)
#  
#  <a id="13"></a> <br>
#  
# ## 1-3 Cheat Sheets
# Some perfect cheatsheet [26]:
# 1. [top-28-cheat-sheets-for-machine-learning-data-science-probability-sql-big-data](https://www.analyticsvidhya.com/blog/2017/02/top-28-cheat-sheets-for-machine-learning-data-science-probability-sql-big-data/)
# 
# 
# I am open to getting your feedback for improving this **kernel**
# <br>
# [go to top](#top)

# <a id="2"></a> <br>
# ## 2- Machine Learning Workflow
# Field of 	study 	that 	gives	computers	the	ability 	to	learn 	without 	being
# explicitly 	programmed.
# 
# **Arthur	Samuel, 1959**
# 
# If you have already read some [machine learning books](https://github.com/mjbahmani/10-steps-to-become-a-data-scientist/tree/master/Ebooks). You have noticed that there are different ways to stream data into machine learning.
# 
# most of these books share the following steps (checklist):
# 1. Define the Problem(Look at the big picture)
# 1. Specify Inputs & Outputs
# 1. Data Collection
# 1. Exploratory data analysis
# 1. Data Preprocessing
# 1. Model Design, Training, and Offline Evaluation
# 1. Model Deployment, Online Evaluation, and Monitoring
# 1. Model Maintenance, Diagnosis, and Retraining
# 
# **You can see my workflow in the below image** :
#  <img src="http://s9.picofile.com/file/8338227634/workflow.png" />
# 
# **you should	feel free	to	adapt 	this	checklist 	to	your needs**
# <br>
# [go to top](#top)

# <a id="21"></a> <br>
# ## 2-1 Real world Application Vs Competitions
# Just a simple comparison between real-world apps with competitions based on [coursera courses](https://www.coursera.org/learn/competitive-data-science):
# <img src="http://s9.picofile.com/file/8339956300/reallife.png" height="600" width="500" />

# <a id="3"></a> <br>
# ## 3- Problem Definition
# I think one of the important things when you start a new machine learning project is Defining your problem. that means you should understand business problem.( **Problem Formalization**)
# 
# Problem Definition has four steps that have illustrated in the picture below:
# <img src="http://s8.picofile.com/file/8338227734/ProblemDefination.png">
# <a id="31"></a> <br>
# ### 3-1 Problem Feature
# we will use the classic Iris data set. This dataset contains information about three different types of Iris flowers:
# 
# 1. Iris Versicolor
# 1. Iris Virginica
# 1. Iris Setosa
# 
# The data set contains measurements of four variables :
# 
# 1. sepal length 
# 1. sepal width
# 1. petal length 
# 1. petal width
#  
# The Iris data set has a number of interesting features:
# 
# 1. One of the classes (Iris Setosa) is linearly separable from the other two. However, the other two classes are not linearly separable.
# 
# 2. There is some overlap between the Versicolor and Virginica classes, so it is unlikely to achieve a perfect classification rate.
# 
# 3. There is some redundancy in the four input variables, so it is possible to achieve a good solution with only three of them, or even (with difficulty) from two, but the precise choice of best variables is not obvious.
# 
# **Why am I  using iris dataset:**
# 
# 1- This is a good project because it is so well understood.
# 
# 2- Attributes are numeric so you have to figure out how to load and handle data.
# 
# 3- It is a classification problem, allowing you to practice with perhaps an easier type of supervised learning algorithm.
# 
# 4- It is a multi-class classification problem (multi-nominal) that may require some specialized handling.
# 
# 5- It only has 4 attributes and 150 rows, meaning it is small and easily fits into memory (and a screen or A4 page).
# 
# 6- All of the numeric attributes are in the same units and the same scale, not requiring any special scaling or transforms to get started.[5]
# 
# 7- we can define problem as clustering(unsupervised algorithm) project too.
# <a id="32"></a> <br>
# ### 3-2 Aim
# The aim is to classify iris flowers among three species (setosa, versicolor or virginica) from measurements of length and width of sepals and petals
# <a id="33"></a> <br>
# ### 3-3 Variables
# The variables are :
# **sepal_length**: Sepal length, in centimeters, used as input.
# **sepal_width**: Sepal width, in centimeters, used as input.
# **petal_length**: Petal length, in centimeters, used as input.
# **petal_width**: Petal width, in centimeters, used as input.
# **setosa**: Iris setosa, true or false, used as target.
# **versicolour**: Iris versicolour, true or false, used as target.
# **virginica**: Iris virginica, true or false, used as target.
# 
# **<< Note >>**
# > You must answer the following question:
# How does your company expact to use and benfit from your model.
# <br>
# [go to top](#top)

# <a id="4"></a> <br>
# ## 4- Inputs & Outputs
# <a id="41"></a> <br>
# ### 4-1 Inputs
# **Iris** is a very popular **classification** and **clustering** problem in machine learning and it is such as "Hello world" program when you start learning a new programming language. then I decided to apply Iris on  20 machine learning method on it.
# As a result, **iris dataset is used as the input of all algorithms**.
# <a id="42"></a> <br>
# ### 4-2 Outputs
# the outputs for our algorithms totally depend on the type of classification or clustering algorithms.
# the outputs can be the number of clusters or predict for new input.
# 
# **setosa**: Iris setosa, true or false, used as target.
# **versicolour**: Iris versicolour, true or false, used as target.
# **virginica**: Iris virginica, true or false, used as a target.

# <a id="5"></a> <br>
# ## 5 Loading Packages
# In this kernel we are using the following packages:

#  <img src="http://s8.picofile.com/file/8338227868/packages.png">
# 

# <a id="51"></a> <br>
# ###   5-1 Import

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from pandas import get_dummies
import plotly.graph_objs as go
from sklearn import datasets
import plotly.plotly as py
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

# <a id="52"></a> <br>
# ### 5-2 Version

# In[ ]:


print('matplotlib: {}'.format(matplotlib.__version__))
print('sklearn: {}'.format(sklearn.__version__))
print('scipy: {}'.format(scipy.__version__))
print('seaborn: {}'.format(sns.__version__))
print('pandas: {}'.format(pd.__version__))
print('numpy: {}'.format(np.__version__))
print('Python: {}'.format(sys.version))

# <a id="53"></a> <br>
# ### 5-3 Setup
# 
# A few tiny adjustments for better **code readability**

# In[ ]:


sns.set(style='white', context='notebook', palette='deep')
warnings.filterwarnings('ignore')
sns.set_style('white')
np.random.seed(1337)
#show plot inline

# <a id="6"></a> <br>
# ## 6- Exploratory Data Analysis(EDA)
#  In this section, you'll learn how to use graphical and numerical techniques to begin uncovering the structure of your data. 
#  
# * Which variables suggest interesting relationships?
# * Which observations are unusual?
# 
# By the end of the section, you'll be able to answer these questions and more, while generating graphics that are both insightful and beautiful.  then We will review analytical and statistical operations:
# 
# *   5-1 Data Collection
# *   5-2 Visualization
# *   5-3 Data Preprocessing
# *   5-4 Data Cleaning
# <img src="http://s9.picofile.com/file/8338476134/EDA.png">

# <a id="61"></a> <br>
# ## 6-1 Data Collection
# **Iris dataset**  consists of 3 different types of irises’ (Setosa, Versicolour, and Virginica) petal and sepal length, stored in a 150x4 numpy.ndarray
# 
# The rows being the samples and the columns being: Sepal Length, Sepal Width, Petal Length and Petal Width.[6]
# 

# In[ ]:


print(os.listdir("../input/"))

# In[ ]:


# import Dataset to play with it
dataset = pd.read_csv('../input/Iris.csv')

# **<< Note 1 >>**
# 
# * Each row is an observation (also known as : sample, example, instance, record)
# * Each column is a feature (also known as: Predictor, attribute, Independent Variable, input, regressor, Covariate)

# After loading the data via **pandas**, we should checkout what the content is, description and via the following:

# In[ ]:


type(dataset)

# <a id="62"></a> <br>
# ## 6-2 Visualization
# 
# With interactive visualization, you can take the concept a step further by using technology to drill down into charts and graphs for more detail, interactively changing what data you see and how it’s processed.[SAS]
# 
#  In this section I show you  **+15  plots** with **matplotlib** and **seaborn** that is listed in the blew picture:
#  <img src="http://s8.picofile.com/file/8338475500/visualization.jpg" />
#  </br>
# [go to top](#top)

# <a id="621"></a> <br>
# ### 6-2-1 Scatter plot
# 
# Scatter plot Purpose To identify the type of relationship (if any) between two quantitative variables
# 
# 
# 

# In[ ]:


# Modify the graph above by assigning each species an individual color.
sns.FacetGrid(dataset, hue="Species", size=5) \
   .map(plt.scatter, "SepalLengthCm", "SepalWidthCm") \
   .add_legend()
plt.show()

# <a id="622"></a> <br>
# ### 6-2-2 Box
# In descriptive statistics, a **box plot** or boxplot is a method for graphically depicting groups of numerical data through their quartiles. Box plots may also have lines extending vertically from the boxes (whiskers) indicating variability outside the upper and lower quartiles, hence the terms box-and-whisker plot and box-and-whisker diagram.[wikipedia]

# In[ ]:


dataset.plot(kind='box', subplots=True, layout=(2,3), sharex=False, sharey=False)
plt.figure()
#This gives us a much clearer idea of the distribution of the input attributes:

# In[ ]:


# To plot the species data using a box plot:

sns.boxplot(x="Species", y="PetalLengthCm", data=dataset )
plt.show()

# In[ ]:


ax= sns.boxplot(x="Species", y="PetalLengthCm", data=dataset)
ax= sns.stripplot(x="Species", y="PetalLengthCm", data=dataset, jitter=True, edgecolor="gray")
plt.show()

# In[ ]:


# Tweek the plot above to change fill and border color color using ax.artists.
# Assing ax.artists a variable name, and insert the box number into the corresponding brackets

ax= sns.boxplot(x="Species", y="PetalLengthCm", data=dataset)
ax= sns.stripplot(x="Species", y="PetalLengthCm", data=dataset, jitter=True, edgecolor="gray")

boxtwo = ax.artists[2]
boxtwo.set_facecolor('red')
boxtwo.set_edgecolor('black')
boxthree=ax.artists[1]
boxthree.set_facecolor('yellow')
boxthree.set_edgecolor('black')

plt.show()

# <a id="623"></a> <br>
# ### 6-2-3 Histogram
# We can also create a **histogram** of each input variable to get an idea of the distribution.
# 
# [go to top](#top)

# In[ ]:


# histograms
dataset.hist(figsize=(15,20))
plt.figure()

# It looks like perhaps two of the input variables have a Gaussian distribution. This is useful to note as we can use algorithms that can exploit this assumption.
# 
# 

# In[ ]:


dataset["PetalLengthCm"].hist();

# <a id="624"></a> <br>
# ### 6-2-4 Multivariate Plots
# Now we can look at the interactions between the variables.
# 
# First, let’s look at scatterplots of all pairs of attributes. This can be helpful to spot structured relationships between input variables.

# In[ ]:



# scatter plot matrix
pd.plotting.scatter_matrix(dataset,figsize=(10,10))
plt.figure()

# Note the diagonal grouping of some pairs of attributes. This suggests a high correlation and a predictable relationship.

# <a id="625"></a> <br>
# ### 6-2-5 violinplots

# In[ ]:


# violinplots on petal-length for each species
sns.violinplot(data=dataset,x="Species", y="PetalLengthCm")

# <a id="626"></a> <br>
# ### 6-2-6 pairplot

# In[ ]:


# Using seaborn pairplot to see the bivariate relation between each pair of features
sns.pairplot(dataset, hue="Species")

# From the plot, we can see that the species setosa is separataed from the other two across all feature combinations
# 
# We can also replace the histograms shown in the diagonal of the pairplot by kde.

# In[ ]:


# updating the diagonal elements in a pairplot to show a kde
sns.pairplot(dataset, hue="Species",diag_kind="kde")

# <a id="627"></a> <br>
# ###  6-2-7 kdeplot

# In[ ]:


# seaborn's kdeplot, plots univariate or bivariate density estimates.
#Size can be changed by tweeking the value used
sns.FacetGrid(dataset, hue="Species", size=5).map(sns.kdeplot, "PetalLengthCm").add_legend()
plt.show()

# <a id="628"></a> <br>
# ### 6-2-8 jointplot

# In[ ]:


# Use seaborn's jointplot to make a hexagonal bin plot
#Set desired size and ratio and choose a color.
sns.jointplot(x="SepalLengthCm", y="SepalWidthCm", data=dataset, size=10,ratio=10, kind='hex',color='green')
plt.show()

# <a id="629"></a> <br>
# ###  6-2-9 andrews_curves

# In[ ]:


#In Pandas use Andrews Curves to plot and visualize data structure.
#Each multivariate observation is transformed into a curve and represents the coefficients of a Fourier series.
#This useful for detecting outliers in times series data.
#Use colormap to change the color of the curves

from pandas.tools.plotting import andrews_curves
andrews_curves(dataset.drop("Id", axis=1), "Species",colormap='rainbow')
plt.show()

# In[ ]:


# we will use seaborn jointplot shows bivariate scatterplots and univariate histograms with Kernel density 
# estimation in the same figure
sns.jointplot(x="SepalLengthCm", y="SepalWidthCm", data=dataset, size=6, kind='kde', color='#800000', space=0)

# <a id="6210"></a> <br>
# ### 6-2-10 Heatmap

# In[ ]:


plt.figure(figsize=(7,4)) 
sns.heatmap(dataset.corr(),annot=True,cmap='cubehelix_r') #draws  heatmap with input as the correlation matrix calculted by(iris.corr())
plt.show()

# <a id="6211"></a> <br>
# ### 6-2-11 radviz

# In[ ]:


# A final multivariate visualization technique pandas has is radviz
# Which puts each feature as a point on a 2D plane, and then simulates
# having each sample attached to those points through a spring weighted
# by the relative value for that feature
from pandas.tools.plotting import radviz
radviz(dataset.drop("Id", axis=1), "Species")

# <a id="6212"></a> <br>
# ### 6-2-12 Bar Plot

# In[ ]:


dataset['Species'].value_counts().plot(kind="bar");

# <a id="6213"></a> <br>
# ### 6-2-13 Visualization with Plotly

# In[ ]:


import plotly.offline as py
import plotly.graph_objs as go
py.init_notebook_mode(connected=True)
from plotly import tools
import plotly.figure_factory as ff
iris = datasets.load_iris()
X = iris.data[:, :2]  # we only take the first two features.
Y = iris.target

x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
trace = go.Scatter(x=X[:, 0],
                   y=X[:, 1],
                   mode='markers',
                   marker=dict(color=np.random.randn(150),
                               size=10,
                               colorscale='Viridis',
                               showscale=False))

layout = go.Layout(title='Training Points',
                   xaxis=dict(title='Sepal length',
                            showgrid=False),
                   yaxis=dict(title='Sepal width',
                            showgrid=False),
                  )
 
fig = go.Figure(data=[trace], layout=layout)

# In[ ]:


py.iplot(fig)

# **<< Note >>**
# 
# **Yellowbrick** is a suite of visual diagnostic tools called “Visualizers” that extend the Scikit-Learn API to allow human steering of the model selection process. In a nutshell, Yellowbrick combines scikit-learn with matplotlib in the best tradition of the scikit-learn documentation, but to produce visualizations for your models! 

# <a id="6214"></a> <br>
# ### 6-2-14 Conclusion
# we have used Python to apply data visualization tools to the Iris dataset. Color and size changes were made to the data points in scatterplots. I changed the border and fill color of the boxplot and violin, respectively.

# <a id="6"></a> <br>
# ## 6-3 Data Preprocessing
# **Data preprocessing** refers to the transformations applied to our data before feeding it to the algorithm.[11]
#  
# there are plenty of steps for data preprocessing and we just listed some of them :
# * removing Target column (id)
# * Sampling (without replacement)
# * Making part of iris unbalanced and balancing (with undersampling and SMOTE)
# * Introducing missing values and treating them (replacing by average values)
# * Noise filtering
# * Data discretization
# * Normalization and standardization
# * PCA analysis
# * Feature selection (filter, embedded, wrapper)

# <a id="30"></a> <br>
# ## 6-3-1 Features
# **Features**:
# 1. numeric
# 1. categorical
# 1. ordinal
# 1. datetime
# 1. coordinates
# 
# Now could you find the type of features in titanic dataset?
# <img src="http://s9.picofile.com/file/8339959442/titanic.png" height="700" width="600" />

# <a id="632"></a> <br>
# ### 6-3-2 Explorer Dataset
# 1- Dimensions of the dataset.
# 
# 2- Peek at the data itself.
# 
# 3- Statistical summary of all attributes.
# 
# 4- Breakdown of the data by the class variable.[7]
# 
# Don’t worry, each look at the data is **one command**. These are useful commands that you can use again and again on future projects.

# In[ ]:


# shape
print(dataset.shape)

# In[ ]:


#columns*rows
dataset.size

# How many NA elements in every column
# 

# In[ ]:


dataset.isnull().sum()

# In[ ]:


# remove rows that have NA's
dataset = dataset.dropna()

# 
# We can get a quick idea of how many instances (rows) and how many attributes (columns) the data contains with the shape property.
# 
# You should see 150 instances and 5 attributes:

# For getting some information about the dataset you can use **info()** command

# In[ ]:


print(dataset.info())

# You see number of unique item for Species with command below:

# In[ ]:


dataset['Species'].unique()

# In[ ]:


dataset["Species"].value_counts()


# To check the first 5 rows of the data set, we can use head(5).

# In[ ]:


dataset.head(5) 

# To check out last 5 row of the data set, we use tail() function

# In[ ]:


dataset.tail() 

# To pop up 5 random rows from the data set, we can use **sample(5)**  function

# In[ ]:


dataset.sample(5) 

# To give a statistical summary about the dataset, we can use **describe()

# In[ ]:


dataset.describe() 

# To check out how many null info are on the dataset, we can use **isnull().sum().

# In[ ]:


dataset.isnull().sum()

# In[ ]:


dataset.groupby('Species').count()

# To print dataset **columns**, we can use columns atribute

# In[ ]:


dataset.columns

# **<< Note 2 >>**
# <br>
# > in pandas's data frame you can perform some query such as "where".

# In[ ]:


dataset.where(dataset ['Species']=='Iris-setosa')

# As you can see in the below in python, it is so easy perform some query on the dataframe:

# In[ ]:


dataset[dataset['SepalLengthCm']>7.2]

# In[ ]:


# Seperating the data into dependent and independent variables
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# **<< Note >>**
# >Preprocessing and generation pipelines depend on a model type.

# <a id="64"></a> <br>
# ## 6-4 Data Cleaning
# When dealing with real-world data, dirty data is the norm rather than the exception. We continuously need to predict correct values, impute missing ones, and find links between various data artefacts such as schemas and records. We need to stop treating data cleaning as a piecemeal exercise (resolving different types of errors in isolation), and instead leverage all signals and resources (such as constraints, available statistics, and dictionaries) to accurately predict corrective actions.[12]

# In[ ]:


cols = dataset.columns
features = cols[0:4]
labels = cols[4]
print(features)
print(labels)

# In[ ]:


#Well conditioned data will have zero mean and equal variance
#We get this automattically when we calculate the Z Scores for the data

data_norm = pd.DataFrame(dataset)

for feature in features:
    dataset[feature] = (dataset[feature] - dataset[feature].mean())/dataset[feature].std()

#Show that should now have zero mean
print("Averages")
print(dataset.mean())

print("\n Deviations")
#Show that we have equal variance
print(pow(dataset.std(),2))

# In[ ]:


#Shuffle The data
indices = data_norm.index.tolist()
indices = np.array(indices)
np.random.shuffle(indices)


# In[ ]:


# One Hot Encode as a dataframe
from sklearn.model_selection import train_test_split
y = get_dummies(y)

# Generate Training and Validation Sets
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=.3)

# Convert to np arrays so that we can use with TensorFlow
X_train = np.array(X_train).astype(np.float32)
X_test  = np.array(X_test).astype(np.float32)
y_train = np.array(y_train).astype(np.float32)
y_test  = np.array(y_test).astype(np.float32)

# In[ ]:


#Check to make sure split still has 4 features and 3 labels
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

# <a id="32"></a> <br>
# ## 7- Model Deployment
# In this section have been applied more than **20 learning algorithms** that play an important rule in your experiences and improve your knowledge in case of ML technique.

# <a id="33"></a> <br>
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
#     * k-Nearest 	Neighbors
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
# * Visualization 	and	dimensionality 	reduction:
# 
#     * Principal 	Component 	Analysis(PCA)
#     * Kernel PCA
#     * Locally -Linear	Embedding 	(LLE)
#     * t-distributed	Stochastic	Neighbor	Embedding 	(t-SNE)
#     
# * Association 	rule	learning
# 
#     * Apriori
#     * Eclat
# * Semisupervised learning
# * Reinforcement Learning
#     * Q-learning
# * Batch learning & Online learning
# * Ensemble  Learning
# 
# **<< Note >>**
# > Here is no method which outperforms all others for all tasks
# 
# 

# <a id="34"></a> <br>
# ## 7-2 Prepare Features & Targets
# First of all seperating the data into dependent(Feature) and independent(Target) variables.
# 
# **<< Note 4 >>**
# 1. X==>>Feature
# 1. y==>>Target

# In[ ]:



X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# <a id="35"></a> <br>
# ## 7-3 Accuracy and precision
# 1. **precision** : 
# 
#     1. In pattern recognition, information retrieval and binary classification, precision (also called positive predictive value) is the fraction of relevant instances among the retrieved instances, 
# 1. **recall** : 
# 
#     1. recall is the fraction of relevant instances that have been retrieved over the total amount of relevant instances. 
# 1. **F-score** :
# 
#     1. the F1 score is a measure of a test's accuracy. It considers both the precision p and the recall r of the test to compute the score: p is the number of correct positive results divided by the number of all positive results returned by the classifier, and r is the number of correct positive results divided by the number of all relevant samples (all samples that should have been identified as positive). The F1 score is the harmonic average of the precision and recall, where an F1 score reaches its best value at 1 (perfect precision and recall) and worst at 0.
#     <br>
# 1. **What is the difference between accuracy and precision?**
# <br>
#     1. "Accuracy" and "precision" are general terms throughout science. A good way to internalize the difference are the common "bullseye diagrams". In machine learning/statistics as a whole, accuracy vs. precision is analogous to bias vs. variance.[13]

# <a id="74"></a> <br>
# ## 7-4 K-Nearest Neighbours
# In **Machine Learning**, the **k-nearest neighbors algorithm** (k-NN) is a non-parametric method used for classification and regression. In both cases, the input consists of the k closest training examples in the feature space. The output depends on whether k-NN is used for classification or regression[https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm):
# 
# In k-NN classification, the output is a class membership. An object is classified by a majority vote of its neighbors, with the object being assigned to the class most common among its k nearest neighbors (k is a positive integer, typically small). If k = 1, then the object is simply assigned to the class of that single nearest neighbor.
# In k-NN regression, the output is the property value for the object. This value is the average of the values of its k nearest neighbors.
# k-NN is a type of instance-based learning, or lazy learning, where the function is only approximated locally and all computation is deferred until classification. The k-NN algorithm is among the simplest of all machine learning algorithms.

# In[ ]:


# K-Nearest Neighbours
from sklearn.neighbors import KNeighborsClassifier

Model = KNeighborsClassifier(n_neighbors=8)
Model.fit(X_train, y_train)

y_pred = Model.predict(X_test)

# Summary of the predictions made by the classifier
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
# Accuracy score

print('accuracy is',accuracy_score(y_pred,y_test))

# <a id="75"></a> <br>
# ##  7-5 Radius Neighbors Classifier
# Classifier implementing a **vote** among neighbors within a given **radius**[scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.RadiusNeighborsClassifier.html)
# 
# In scikit-learn **RadiusNeighborsClassifier** is very similar to **KNeighborsClassifier** with the exception of two parameters. First, in RadiusNeighborsClassifier we need to specify the radius of the fixed area used to determine if an observation is a neighbor using radius. Unless there is some substantive reason for setting radius to some value, it is best to treat it like any other hyperparameter and tune it during model selection. The second useful parameter is outlier_label, which indicates what label to give an observation that has no observations within the radius - which itself can often be a useful tool for identifying outliers.

# In[ ]:


from sklearn.neighbors import  RadiusNeighborsClassifier
Model=RadiusNeighborsClassifier(radius=8.0)
Model.fit(X_train,y_train)
y_pred=Model.predict(X_test)
#summary of the predictions made by the classifier
print(classification_report(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))
#Accouracy score
print('accuracy is ', accuracy_score(y_test,y_pred))

# <a id="76"></a> <br>
# ## 7-6 Logistic Regression
# Logistic regression is the appropriate regression analysis to conduct when the dependent variable is **dichotomous** (binary). Like all regression analyses, the logistic regression is a **predictive analysis** [statisticssolutions](https://www.statisticssolutions.com/what-is-logistic-regression/).
# 
# In statistics, the logistic model (or logit model) is a widely used statistical model that, in its basic form, uses a logistic function to model a binary dependent variable; many more complex extensions exist. In regression analysis, logistic regression (or logit regression) is estimating the parameters of a logistic model; it is a form of binomial regression. Mathematically, a binary logistic model has a dependent variable with two possible values, such as pass/fail, win/lose, alive/dead or healthy/sick; these are represented by an indicator variable, where the two values are labeled "0" and "1"

# In[ ]:


# LogisticRegression
from sklearn.linear_model import LogisticRegression
Model = LogisticRegression()
Model.fit(X_train, y_train)

y_pred = Model.predict(X_test)

# Summary of the predictions made by the classifier
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
# Accuracy score
print('accuracy is',accuracy_score(y_pred,y_test))

# <a id="77"></a> <br>
# ##  7-7 Passive Aggressive Classifier

# In[ ]:


from sklearn.linear_model import PassiveAggressiveClassifier
Model = PassiveAggressiveClassifier()
Model.fit(X_train, y_train)

y_pred = Model.predict(X_test)

# Summary of the predictions made by the classifier
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
# Accuracy score
print('accuracy is',accuracy_score(y_pred,y_test))

# <a id="78"></a> <br>
# ## 7-8 Naive Bayes
# In machine learning, naive Bayes classifiers are a family of simple "**probabilistic classifiers**" based on applying Bayes' theorem with strong (naive) independence assumptions between the features.

# In[ ]:


# Naive Bayes
from sklearn.naive_bayes import GaussianNB
Model = GaussianNB()
Model.fit(X_train, y_train)

y_pred = Model.predict(X_test)

# Summary of the predictions made by the classifier
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
# Accuracy score
print('accuracy is',accuracy_score(y_pred,y_test))

# <a id="79"></a> <br>
# ##  7-9 BernoulliNB
# Like MultinomialNB, this classifier is suitable for **discrete data**. The difference is that while MultinomialNB works with occurrence counts, BernoulliNB is designed for binary/boolean features.

# In[ ]:


# BernoulliNB
from sklearn.naive_bayes import BernoulliNB
Model = BernoulliNB()
Model.fit(X_train, y_train)

y_pred = Model.predict(X_test)

# Summary of the predictions made by the classifier
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
# Accuracy score
print('accuracy is',accuracy_score(y_pred,y_test))

# <a id="710"></a> <br>
# ## 7-10 SVM
# 
# The advantages of support vector machines are [simsam](http://www.simsam.us/2017/05/30/support-vector-machine-algorithm/):
# * Effective in high dimensional spaces.
# * Still effective in cases where number of dimensions is greater than the number of samples. 
# * Uses a subset of training points in the decision function (called support vectors), so it is also memory efficient.
# * Versatile: different Kernel functions can be specified for the decision function. Common kernels are provided, but it is also possible to specify custom kernels.
# 
# The disadvantages of support vector machines include:
# 
# * If the number of features is much greater than the number of samples, avoid over-fitting in choosing Kernel functions and regularization term is crucial.
# * SVMs do not directly provide probability estimates, these are calculated using an expensive five-fold cross-validation

# In[ ]:


# Support Vector Machine
from sklearn.svm import SVC

Model = SVC()
Model.fit(X_train, y_train)

y_pred = Model.predict(X_test)

# Summary of the predictions made by the classifier
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
# Accuracy score

print('accuracy is',accuracy_score(y_pred,y_test))

# <a id="711"></a> <br>
# ## 7-11 Nu-Support Vector Classification
# 
# > Similar to SVC but uses a parameter to control the number of support vectors.

# In[ ]:


# Support Vector Machine's 
from sklearn.svm import NuSVC

Model = NuSVC()
Model.fit(X_train, y_train)

y_pred = Model.predict(X_test)

# Summary of the predictions made by the classifier
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
# Accuracy score

print('accuracy is',accuracy_score(y_pred,y_test))

# <a id="712"></a> <br>
# ## 7-12 Linear Support Vector Classification
# 
# Similar to **SVC** with parameter kernel=’linear’, but implemented in terms of liblinear rather than libsvm, so it has more flexibility in the choice of penalties and loss functions and should scale better to large numbers of samples.

# In[ ]:


# Linear Support Vector Classification
from sklearn.svm import LinearSVC

Model = LinearSVC()
Model.fit(X_train, y_train)

y_pred = Model.predict(X_test)

# Summary of the predictions made by the classifier
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
# Accuracy score

print('accuracy is',accuracy_score(y_pred,y_test))

# <a id="713"></a> <br>
# ## 7-13 Decision Tree
# Decision Trees (DTs) are a non-parametric supervised learning method used for **classification** and **regression**. The goal is to create a model that predicts the value of a target variable by learning simple **decision rules** inferred from the data features.

# In[ ]:


# Decision Tree's
from sklearn.tree import DecisionTreeClassifier

Model = DecisionTreeClassifier()

Model.fit(X_train, y_train)

y_pred = Model.predict(X_test)

# Summary of the predictions made by the classifier
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
# Accuracy score
print('accuracy is',accuracy_score(y_pred,y_test))

# <a id="714"></a> <br>
# ## 7-14 ExtraTreeClassifier
# An extremely randomized tree classifier.
# 
# Extra-trees differ from classic decision trees in the way they are built. When looking for the best split to separate the samples of a node into two groups, random splits are drawn for each of the **max_features** randomly selected features and the best split among those is chosen. When max_features is set 1, this amounts to building a totally random decision tree.
# 
# **Warning**: Extra-trees should only be used within ensemble methods.

# In[ ]:


# ExtraTreeClassifier
from sklearn.tree import ExtraTreeClassifier

Model = ExtraTreeClassifier()

Model.fit(X_train, y_train)

y_pred = Model.predict(X_test)

# Summary of the predictions made by the classifier
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
# Accuracy score
print('accuracy is',accuracy_score(y_pred,y_test))

# <a id="715"></a> <br>
# ## 7-15 Neural network
# 
# I have used multi-layer Perceptron classifier.
# This model optimizes the log-loss function using **LBFGS** or **stochastic gradient descent**.

# <a id="7151"></a> <br>
# ## 7-15-1 What is a Perceptron?

# There are many online examples and tutorials on perceptrons and learning. Here is a list of some articles:
# - [Wikipedia on Perceptrons](https://en.wikipedia.org/wiki/Perceptron)
# - Jurafsky and Martin (ed. 3), Chapter 8

# This is an example that I have taken from a draft of the 3rd edition of Jurafsky and Martin, with slight modifications:
# We import *numpy* and use its *exp* function. We could use the same function from the *math* module, or some other module like *scipy*. The *sigmoid* function is defined as in the textbook:
# 

# In[ ]:


import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Our example data, **weights** $w$, **bias** $b$, and **input** $x$ are defined as:

# In[ ]:


w = np.array([0.2, 0.3, 0.8])
b = 0.5
x = np.array([0.5, 0.6, 0.1])

# Our neural unit would compute $z$ as the **dot-product** $w \cdot x$ and add the **bias** $b$ to it. The sigmoid function defined above will convert this $z$ value to the **activation value** $a$ of the unit:

# In[ ]:


z = w.dot(x) + b
print("z:", z)
print("a:", sigmoid(z))

# <a id="7152"></a> <br>
# ### 7-15-2 The XOR Problem
# The power of neural units comes from combining them into larger networks. Minsky and Papert (1969): A single neural unit cannot compute the simple logical function XOR.
# 
# The task is to implement a simple **perceptron** to compute logical operations like AND, OR, and XOR.
# 
# - Input: $x_1$ and $x_2$
# - Bias: $b = -1$ for AND; $b = 0$ for OR
# - Weights: $w = [1, 1]$
# 
# with the following activation function:
# 
# $$
# y = \begin{cases}
#     \ 0 & \quad \text{if } w \cdot x + b \leq 0\\
#     \ 1 & \quad \text{if } w \cdot x + b > 0
#   \end{cases}
# $$

# We can define this activation function in Python as:

# In[ ]:


def activation(z):
    if z > 0:
        return 1
    return 0

# For AND we could implement a perceptron as:

# In[ ]:


w = np.array([1, 1])
b = -1
x = np.array([0, 0])
print("0 AND 0:", activation(w.dot(x) + b))
x = np.array([1, 0])
print("1 AND 0:", activation(w.dot(x) + b))
x = np.array([0, 1])
print("0 AND 1:", activation(w.dot(x) + b))
x = np.array([1, 1])
print("1 AND 1:", activation(w.dot(x) + b))

# For OR we could implement a perceptron as:

# In[ ]:


w = np.array([1, 1])
b = 0
x = np.array([0, 0])
print("0 OR 0:", activation(w.dot(x) + b))
x = np.array([1, 0])
print("1 OR 0:", activation(w.dot(x) + b))
x = np.array([0, 1])
print("0 OR 1:", activation(w.dot(x) + b))
x = np.array([1, 1])
print("1 OR 1:", activation(w.dot(x) + b))

# There is no way to implement a perceptron for XOR this way.

# no see our prediction for iris

# In[ ]:


from sklearn.neural_network import MLPClassifier
Model=MLPClassifier()
Model.fit(X_train,y_train)
y_pred=Model.predict(X_test)
# Summary of the predictions
print(classification_report(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))
#Accuracy Score
print('accuracy is ',accuracy_score(y_pred,y_test))

# <a id="716"></a> <br>
# ## 7-16 RandomForest
# A random forest is a meta estimator that **fits a number of decision tree classifiers** on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting. 
# 
# The sub-sample size is always the same as the original input sample size but the samples are drawn with replacement if bootstrap=True (default).

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
Model=RandomForestClassifier(max_depth=2)
Model.fit(X_train,y_train)
y_pred=Model.predict(X_test)
print(classification_report(y_test,y_pred))
print(confusion_matrix(y_pred,y_test))
#Accuracy Score
print('accuracy is ',accuracy_score(y_pred,y_test))

# <a id="717"></a> <br>
# ## 7-17 Bagging classifier 
# A Bagging classifier is an ensemble **meta-estimator** that fits base classifiers each on random subsets of the original dataset and then aggregate their individual predictions (either by voting or by averaging) to form a final prediction. Such a meta-estimator can typically be used as a way to reduce the variance of a black-box estimator (e.g., a decision tree), by introducing randomization into its construction procedure and then making an ensemble out of it.
# 
# This algorithm encompasses several works from the literature. When random subsets of the dataset are drawn as random subsets of the samples, then this algorithm is known as Pasting . If samples are drawn with replacement, then the method is known as Bagging . When random subsets of the dataset are drawn as random subsets of the features, then the method is known as Random Subspaces . Finally, when base estimators are built on subsets of both samples and features, then the method is known as Random Patches .[http://scikit-learn.org]

# In[ ]:


from sklearn.ensemble import BaggingClassifier
Model=BaggingClassifier()
Model.fit(X_train,y_train)
y_pred=Model.predict(X_test)
print(classification_report(y_test,y_pred))
print(confusion_matrix(y_pred,y_test))
#Accuracy Score
print('accuracy is ',accuracy_score(y_pred,y_test))

# <a id="718"></a> <br>
# ##  7-18 AdaBoost classifier
# 
# An AdaBoost classifier is a meta-estimator that begins by fitting a classifier on the original dataset and then fits additional copies of the classifier on the same dataset but where the weights of incorrectly classified instances are adjusted such that subsequent classifiers focus more on difficult cases.
# This class implements the algorithm known as **AdaBoost-SAMME** .

# In[ ]:


from sklearn.ensemble import AdaBoostClassifier
Model=AdaBoostClassifier()
Model.fit(X_train,y_train)
y_pred=Model.predict(X_test)
print(classification_report(y_test,y_pred))
print(confusion_matrix(y_pred,y_test))
#Accuracy Score
print('accuracy is ',accuracy_score(y_pred,y_test))

# <a id="719"></a> <br>
# ## 7-19 Gradient Boosting Classifier
# GB builds an additive model in a forward stage-wise fashion; it allows for the optimization of arbitrary differentiable loss functions.

# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier
Model=GradientBoostingClassifier()
Model.fit(X_train,y_train)
y_pred=Model.predict(X_test)
print(classification_report(y_test,y_pred))
print(confusion_matrix(y_pred,y_test))
#Accuracy Score
print('accuracy is ',accuracy_score(y_pred,y_test))

# <a id="720"></a> <br>
# ## 7-20 Linear Discriminant Analysis
# Linear Discriminant Analysis (discriminant_analysis.LinearDiscriminantAnalysis) and Quadratic Discriminant Analysis (discriminant_analysis.QuadraticDiscriminantAnalysis) are two classic classifiers, with, as their names suggest, a **linear and a quadratic decision surface**, respectively.
# 
# These classifiers are attractive because they have closed-form solutions that can be easily computed, are inherently multiclass, have proven to work well in practice, and have no **hyperparameters** to tune.

# In[ ]:


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
Model=LinearDiscriminantAnalysis()
Model.fit(X_train,y_train)
y_pred=Model.predict(X_test)
print(classification_report(y_test,y_pred))
print(confusion_matrix(y_pred,y_test))
#Accuracy Score
print('accuracy is ',accuracy_score(y_pred,y_test))

# -----------------
# <a id="8"></a> <br>
# # 8- Conclusion

# In this kernel, I have tried to cover all the parts related to the process of **Machine Learning** with a variety of Python packages and I know that there are still some problems then I hope to get your feedback to improve it.
# <br>
# [go to top](#top)

# Fork and Run this Notebook on GitHub:
# 
# > #### [ GitHub](https://github.com/mjbahmani/10-steps-to-become-a-data-scientist)
# > #### [ Kaggle](https://www.kaggle.com/mjbahmani)
# 
# --------------------------------------
# 
#  **I hope you find this kernel helpful and some <font color="red"><b>UPVOTES</b></font> would be very much appreciated** 

# <a id="9"></a> <br>
# # 9- References & Credits
# 1. [Iris image](https://rpubs.com/wjholst/322258)
# 1. [IRIS](https://archive.ics.uci.edu/ml/datasets/iris)
# 1. [https://skymind.ai/wiki/machine-learning-workflow](https://skymind.ai/wiki/machine-learning-workflow)
# 1. [IRIS-wiki](https://archive.ics.uci.edu/ml/datasets/iris)
# 1. [Problem-define](https://machinelearningmastery.com/machine-learning-in-python-step-by-step/)
# 1. [Sklearn](http://scikit-learn.org/)
# 1. [machine-learning-in-python-step-by-step](https://machinelearningmastery.com/machine-learning-in-python-step-by-step/)
# 1. [Data Cleaning](http://wp.sigmod.org/?p=2288)
# 1. [competitive data science](https://www.coursera.org/learn/competitive-data-science/)
# 1. [Top 28 Cheat Sheets for Machine Learning](https://www.analyticsvidhya.com/blog/2017/02/top-28-cheat-sheets-for-machine-learning-data-science-probability-sql-big-data/)
# 1. [geeksforgeeks](https://www.geeksforgeeks.org/data-preprocessing-machine-learning-python/)
# 1. [https://wp.sigmod.org/?p=2288](https://wp.sigmod.org/?p=2288)
# 1. [https://en.wikipedia.org/wiki/Precision_and_recall](https://en.wikipedia.org/wiki/Precision_and_recall)
# 1. [medium](https://medium.com/@jebaseelanravi96/machine-learning-iris-classification-33aa18a4a983)
# -------------
# 

# Go to first step: [**Course Home Page**](https://www.kaggle.com/mjbahmani/10-steps-to-become-a-data-scientist)
# 
# Go to next step : [**Mathematics and Linear Algebra**](https://www.kaggle.com/mjbahmani/linear-algebra-for-data-scientists)
