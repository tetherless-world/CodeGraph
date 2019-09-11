#!/usr/bin/env python
# coding: utf-8

# **Update:**  
# 11/22/17 Please note, this kernel is currently in progress, but open to feedback. Thanks!  
# 11/23/17 Cleaned up published notebook and updated through step 3.  
# 11/25/17 Added enhancements of published notebook and started step 4.  
# 11/26/17 Skipped ahead to data model, since this is a published notebook. Accuracy with (very) simple data cleaning and logistic regression is **~82%**. Continue to up vote and I will continue to develop this notebook. Thanks!  
# 12/2/17 Updated section 4 with exploratory statistics and section 5 with more classifiers. Improved model to **~85%** accuracy.  
# 12/3/17 Update section 4 with improved graphical statistics.  
# 12/7/17 Updated section 5 with Data Science 101 Le
# 
# 
# > Hello, this is my debut project on Kaggle, using the popular Getting Started Competition. My goal is to add to the Data Science Community, 1) a framework that teaches how-to think like a data scientist vs what to think/code and 2) concise code and clear documentation because simple is better than complex.
# 
# 
# # How a Data Scientist Beat the Odds
#  
# It's the classical problem, predict the outcome of a binary event. In laymen terms this means, it either occurred or did not occur. For example, you won or did not win, you passed the test or did not pass the test, you were accepted or not accepted, and you get the point. A common business application is churn or customer retention. Another popular use case is, healthcare's mortality rate or survival analysis. Binary events create an interesting dynamic, because we know statistically, a random guess should achieve a 50% accuracy rate, without creating one single algorithm or writing one single line of code. However, just like autocorrect spellcheck technology, sometimes we humans can be too smart for our own good and actually underperform a coin flip. In this kernel, I use Kaggle's Getting Started Competition, Titanic: Machine Learning from Disaster, to walk the reader through, how-to use the data science framework to beat the odds.
#    
# *What happens when technology is too smart for its own good?*
# ![Funny Autocorrect](http://15858-presscdn-0-65.pagely.netdna-cdn.com/wp-content/uploads/2016/03/hilarious-autocorrect-fails-20x.jpg)

# # A Data Science Framework
# 
# 1. **What is the problem?** If data science, big data, machine learning, predictive analytics, business intelligence, or any other buzzword is the solution, then what is the problem? As the saying goes, don't put the cart before the horse. Problems before requirements, requirements before solutions, solutions before design, and design before technology. Too often we are quick to jump to the new shiny technology, tool, or algorithm before determining the actual problem we are trying to solve.
# 2. **Where is the dataset?** John Naisbitt wrote in his 1984 book Megatrends, we are “drowning in data, yet staving for knowledge." So, chances are, the dataset(s) already exist somewhere, in some format. It may be external or internal, structured or unstructured, static or streamed, objective or subjective, etc. As the saying goes, you don't have to reinvent the wheel, you just have to know where to find it. In the next step, we will worry about transforming "dirty data" to "clean data."
# 3. **Prepare data for consumption.** This step is often referred to as data wrangling, a required process to turn “wild” data into “manageable” data. Data wrangling includes implementing data architectures for storage and processing, developing data governance standards for quality and control, data extraction (i.e. ETL and web scraping), and data cleaning to identify aberrant, missing, or outlier data points.
# 4. **Perform exploratory analysis.** Anybody who has ever worked with data knows, garbage-in, garbage-out (GIGO). Therefore, it is important to deploy descriptive and graphical statistics to look for potential problems, patterns, classifications, correlations and comparisons in the dataset. Statistical classification is also important to understand the overall datatype structure (i.e. qualitative vs quantitative), in order to select the correct hypothesis test or data model.
# 5. **Model Data** Like descriptive and inferential statistics, data modeling can either summarize the data or predict future outcomes. Your dataset and expected results, will determine the algorithms available for use. It's important to remember, algorithms are tools and not magical wands. An analogy would be asking someone to hand you a Philip screwdriver, and they hand you a flathead screwdriver or worst a hammer. At best, it shows a complete lack of understanding. At worst, it makes completing the project impossible. The same is true in data modelling. So use caution when selecting your tool, because at the end of the day, you are still the master craft (wo)man.
# 6. **Evaluate Data Model** After you've trained your model based on a subset of your data, it's time to test your model. This helps ensure you haven't overfit your model or made it so specific to the selected subset, it does not accurately fit another subset from the same dataset.
# 7. **Implement, Optimize, and Strategize** This is the "bionic man" step, where you iterate back through the process to make it better...stronger...faster than it was before. As a data scientist, your strategy should be to outsource developer operations and application plumbing, so you have more time to focus on recommendations and design. Once you're able to package your ideas, this becomes your “currency exchange" rate.

# # Step 1: What is the problem?
# 
# For this project, the problem statement is given to us on a golden plater, develop an algorithm to predict the survival outcome of passengers on the Titanic.
# 
# ......
# 
# **Project Summary:**
# The sinking of the RMS Titanic is one of the most infamous shipwrecks in history.  On April 15, 1912, during her maiden voyage, the Titanic sank after colliding with an iceberg, killing 1502 out of 2224 passengers and crew. This sensational tragedy shocked the international community and led to better safety regulations for ships.
# 
# One of the reasons that the shipwreck led to such loss of life was that there were not enough lifeboats for the passengers and crew. Although there was some element of luck involved in surviving the sinking, some groups of people were more likely to survive than others, such as women, children, and the upper-class.
# 
# In this challenge, we ask you to complete the analysis of what sorts of people were likely to survive. In particular, we ask you to apply the tools of machine learning to predict which passengers survived the tragedy.
# 
# Practice Skills
# * Binary classification
# * Python and R basics
# 
# # Step 2: Where is the dataset?
# 
# The dataset is also given to us on a golden plater with test and train data at [Kaggle's Titanic: Machine Learning from Disaster](https://www.kaggle.com/c/titanic/data)
# 
# # Step 3: Prepare data for consumption.
# 
# Since step 2 was provided to us on a golden plater, so is step 3. Therefore, normal processes in data wrangling, such as data architecture, governance, and extraction are out of scope. Thus, only data cleaning is in scope.

# ## 3.1 Import Libraries
# The following code is written in Python 3.x. Libraries provide pre-written functionality to perform necessary tasks. The idea is why write ten lines of code, when you can write one line. 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 


import sys #access to system parameters https://docs.python.org/3/library/sys.html
print("Python version: {}". format( sys.version))

import pandas as pd #collection of functions for data processing and analysis modeled after R dataframes with SQL like features
print("pandas version: {}". format( pd.__version__))

import matplotlib #collection of functions for scientific and publication-ready visualization
print("matplotlib version: {}". format( matplotlib.__version__)) 

import numpy as np #foundational package for scientific computing
print("NumPy version: {}". format( np.__version__)) 

import scipy as sp #collection of functions for scientific computing and advance mathematics
print("SciPy version: {}". format( sp.__version__)) 


import IPython 
from IPython import display #pretty printing of dataframes in Jupyter notebook
from IPython.display import Image
print(" IPython version: {}". format( IPython.__version__)) 

import sklearn #collection of machine learning algorithms
print("scikit-learn version: {}". format( sklearn.__version__))

#measure execution of code snippets: https://docs.python.org/3/library/timeit.html
import timeit as t
import random
from time import time

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

print('-'*25)


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

# ## 3.11 Load Data Modelling Libraries
# 
# We will use the popular *scikit-learn* library to develop our machine learning algorithms. In *scikit* algorithms are called Estimators and implemented in their own classes. For data visualization, we will use the *matplotlib* and *seaborn* library. Below are common classes to load.

# In[ ]:


#Common Model Algorithms
from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process

#Common Model Helpers
from sklearn.preprocessing import Imputer , Normalizer , scale, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, GridSearchCV, StratifiedKFold, RandomizedSearchCV
from sklearn.feature_selection import RFECV
from sklearn import metrics

# Visualization
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns
from pandas.tools.plotting import scatter_matrix

# Configure visualizations
#%matplotlib inline = show plots in Jupyter Notebook browser
mpl.style.use( 'ggplot' )
sns.set_style( 'white' )
pylab.rcParams[ 'figure.figsize' ] = 8 , 6

# ## 3.2 Meet and Great Data
# 
# This is the meet and great step. Get to know your data by first name and learn a little bit about it. What does it look like (datatype and values), what makes it tick (independent/feature variables(s)), what's its goals in life (dependent/target variable(s)). Think of it like a first date, before you jump in and start poking it in the bedroom.
# 
# To begin this tasks, we first import our data. Next we use the info() and sample () function, to get a quick and dirty overview of variable datatypes (i.e. qualitative vs quantitative). Click here for the [Source Data Dictionary](https://www.kaggle.com/c/titanic/data).
# 
# 1. The *Survived* variable is our outcome or dependent variable. It is a binary nominal datatype of 1 for survived and 0 for did not survive. All other variables are potential predictor or independent variables. **It's important to note, more predictor variables do not make a better model, but the right variables.**
# 2. The *PassengerID* and *Ticket* variables are assumed to be random unique identifiers, that have no impact on the outcome variable. Thus, they will be excluded from analysis.
# 3. The *Pclass* variable is an ordinal datatype for the ticket class, a proxy for socio-economic status (SES), representing 1 = upper class, 2 = middle class, and 3 = lower class.
# 4. The *Name* variable is a nominal datatype. It could be used for [feature engineering](https://triangleinequality.wordpress.com/2013/09/08/basic-feature-engineering-with-the-titanic-data/) to derive the gender from title, family size from surname, and SES from titles like doctor or master; however these variables already exist. For the first model iteration, this variable will be excluded from analysis. It can be used during subsequent iterations to evaluate if more complexity improves the base model accuracy. 
# 5. The *Sex* and *embarked* variables are a nominal datatype. They will be converted to dummy variables for mathematical calculations.
# 6. The *Age* and *fare* variable are continuous quantitative datatypes.
# 7. The *SibSp* represents number of siblings/spouse aboard and *Parch* represents number of parents or children aboard. Both are dicrete quantitative datatypes. This can be used for feature engineering to create a family size or is alone variable.
# 8. The *cabin* variable is a nominal datatype that can be used in feature engineering for approximate position on ship when the incident occurred and SES from deck levels. However, since there are many null values, it does not add value and thus is excluded from analysis.

# In[ ]:


#load as dataframe
data_raw = pd.read_csv('../input/train.csv')

#Note: The test file is really validation data for competition submission, because we do not know the survival status
#We will create real test data in a later section, so we can evaluate our model before competition submission
validation_raw  = pd.read_csv('../input/test.csv')

#preview data
print (data_raw.info())
#data_raw.head() #https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.head.html
#data_raw.tail() #https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.tail.html
data_raw.sample(10) #https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.sample.html

# ## 3.21 Explore Data with Descriptive Statistics
# 
# Descriptive statistics is a quick and dirty way to identify aberrant, missing, or outlier data. In two lines of code, we can quickly determine we have a little bit of data cleaning ahead of us. In this stage, you will find yourself doing one of three things: completing missing information, correcting outliers, or creating new features for analysis.
# 
# 1. Reviewing the data, there does not appear to be any aberrant or non-acceptable data inputs.
# 2. There are null values or missing data in the age, cabin, and embarked field. Missing values can be bad, because it skews our data model. Thus, it's important to fix before starting analysis. There are two common methods, either delete the record or populate the missing value using a reasonable input. It is not recommended to delete the record, especially a large percentage of records, unless it truly represents an incomplete record. Instead, it's best to impute missing values. A basic methodology for qualitative data is impute using mode. A basic methodology for quantitative data is impute using mean, median, or mean + standard deviation. An intermediate methodology is to use the basic methodology based on specific criteria; like the average age by class or embark port by fare and SES. There are more complex methodologies, however before deploying, it should be compared to the base model to determine if complexity truly adds value. For this dataset, age will be imputed with the median, the cabin attribute will be dropped, and embark will be imputed with mode. Subsequent model iterations will modify this decision to determine if it improves the model’s accuracy.
# 3. At this stage, we see we may have potential outliers in age and fare. However, since they are reasonable values, we will wait until after we complete our graphical statistics to determine if we should include or exclude from analysis. It should be noted, that if they were unreasonable values, for example age = 800 instead of 80, then it's probably a safe decision to fix now. However, we want to use caution when we modify data from its original value, because it may be necessary to create an accurate model.
# 4. Before we move on to the next step, let's deal with formatting. There are no date or currency formats, but datatype formats. Our categorical data imported as objects, which makes it difficult for mathematical calculations. For this dataset, we will convert object datatypes to categorical dummy variables.
# 

# In[ ]:


#Quantitative Descriptive Statistics
print(data_raw.isnull().sum())
print("-"*10)

#Qualitative Descriptive Statistics
print(data_raw['Sex'].value_counts())
print("-"*10)
print(data_raw['Embarked'].value_counts())
data_raw.describe(include = 'all')

# ## 3.22 Clean Data
# 
# Now that we know what to clean, let's execute our code.
# 
# Source Documentation:
# * [pandas.DataFrame](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html)
# * [pandas.DataFrame.info](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.info.html)
# * [pandas.DataFrame.describe](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.describe.html)
# * [Indexing and Selecting Data](https://pandas.pydata.org/pandas-docs/stable/indexing.html)
# * [pandas.isnull](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.isnull.html)
# * [pandas.DataFrame.sum](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.sum.html)
# * [pandas.DataFrame.mode](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.mode.html)
# * [pandas.DataFrame.copy](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.copy.html)
# * [pandas.DataFrame.fillna](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.fillna.html)
# * [pandas.DataFrame.drop](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.drop.html)
# * [pandas.Series.value_counts](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.value_counts.html)

# In[ ]:


#create a copy of data
#remember python assignment or equal passes by reference vs values, so we use the copy function: https://stackoverflow.com/questions/46327494/python-pandas-dataframe-copydeep-false-vs-copydeep-true-vs
data1 = data_raw.copy(deep = True)

# In[ ]:


#cleanup age with median
data1['Age'].fillna(data1['Age'].median(), inplace = True)

#preview data again
print(data1.isnull().sum())
print("-"*10)
#print(data_raw.isnull().sum())

# In[ ]:


#cleanup embarked with mode
data1['Embarked'].fillna(data1['Embarked'].mode()[0], inplace = True)

#preview data again
print(data1.isnull().sum())
print("-"*10)
#print(data_raw.isnull().sum())

# In[ ]:


#delete the cabin feature/column and others previously stated to exclude
drop_column = ['Cabin','PassengerId','Name', 'Ticket']
data1.drop(drop_column, axis=1, inplace = True)

#preview data again
print(data1.isnull().sum())
print("-"*10)
#print(data_raw.isnull().sum())

# ## 3.23 Cleanup Formats
# 
# We will convert *Sex* and *Embarked* from objects to dummy variables for mathematical analysis. In addition, we will explicitly convert *Pclass* to integer categorical variables, so not to be confused with quantitative variables. There are multiple ways to encode categorical variables; we will use the pandas method; *scikit* also has LabelEncoder and OneHotEncoder.
# 
# Source Documentation:
# * [Categorical Encoding](http://pbpython.com/categorical-encoding.html)
# * [Pandas Categorical dtype](https://pandas.pydata.org/pandas-docs/stable/categorical.html)
# * [pandas.get_dummies](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.get_dummies.html)

# In[ ]:


#convert to explicit category data type
#data1['Pclass'] = data1['Pclass'].astype('category')
#data1['Sex'] = data1['Sex'].astype('category')
#data1['Embarked'] = data1['Embarked'].astype('category')

print("Original Features: ", list(data1.columns), '\n')
data1_dummy = pd.get_dummies(data1)
print("Features with Dummies: ", data1_dummy.columns.values, '\n')

print (data1_dummy.dtypes)
data1_dummy.head()

# ## 3.24 Da-Double Check Cleaned Data
# 
# Now that we've cleaned our data, let's do a discount da-double check!

# In[ ]:


#Quantitative Descriptive Statistics
print (data1.info())
print("-"*10)
print(data1.isnull().sum())
print("-"*10)

#Qualitative Descriptive Statistics
print(data1.Sex.value_counts())
print("-"*10)
print(data1.Embarked.value_counts())
print("-"*10)
data1.describe(include = 'all')

# ## 3.25 Split Training and Testing Data
# 
# As mentioned previously, the test file provided is really for the competition submission data and does not provide the outcome/target variable for us to validate our model. So, we will use *scikit* function to split the training data in two datasets; 75/25 split. This is important, so we don't overfit our model. Meaning, the algorithm is so specific to a given subset, it cannot accurately generalize another subset, from the same dataset. It's important our algorithm has not seen the subset we will use to test, so it doesn't "cheat" by memorizing the answers. We will use [*scikit* train_test_split function](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html).
# 
# In this step, we will also define our x (independent/features/explanatory/predictor/etc.) and y (dependent/target/outcome/response/etc.) variables for data modeling.

# In[ ]:


#define x and y variables for original features aka feature selection
data1_x = ['Pclass','Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
data1_y = ['Survived']
data1_xy = data1_x + data1_y
print('Original X Y: ', data1_xy, '\n')

#define x and y variables for dummy features aka feature selection
data1_dummy_x = data1_dummy.iloc[:,1:].columns.tolist()
data1_dummy_y = data1_dummy.iloc[:,0:1].columns.tolist()
data1_dummy_xy = data1_dummy_x + data1_dummy_y
print('Dummy Coding X Y: ', data1_dummy_xy, '\n')

#split train and test data with function defaults
train1_x, test1_x, train1_y, test1_y = train_test_split(data1[data1_x], data1[data1_y])
train1_dummy_x, test1_dummy_x, train1_dummy_y, test1_dummy_y = train_test_split(data1_dummy[data1_dummy_x], data1_dummy[data1_dummy_y])

print("Data Shape: {}".format(data1.shape))
print("Train1 Shape: {}".format(train1_x.shape))
print("Test1 Shape: {}".format(test1_x.shape), '\n')

print("Data1_Dummy Shape: {}".format(data1_dummy.shape))
print("Train1_Dummy Shape: {}".format(train1_dummy_x.shape))
print("Test1_Dummy Shape: {}".format(test1_dummy_x.shape), '\n')

# # Step 4: Perform Exploratory Analysis
# 
# Now that our data is cleaned, let's use descriptive and graphical statistics to describe and summarize our data. In this stage, you will find yourself classifying features and trying to determine their relationship with the target variable and each other.

# In[ ]:


#Correlation by Survival; excluding continuous variables of age and fare
#using group by https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.groupby.html
for x in data1_x:
    if data1[x].dtype != 'float64' :
        print('Survival Correlation by:', x)
        print(data1[[x, data1_y[0]]].groupby(x, as_index=False).mean())
        print('-'*10, '\n')

###Feature Engineering###

#Family Size
data1['FamilySize'] = data1 ['SibSp'] + data1['Parch'] + 1
print ('Survival Correlation by: Family Size \n',
       data1[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean(),
      '\n','-'*10, '\n')

#IsAlone
data1['IsAlone'] = 1 #create a new feature and initialize to yes/1 is alone
data1['IsAlone'].loc[data1['FamilySize'] > 1] = 0 # now update to no/0 if family size is greater than 1
print ('Survival Correlation by: IsAlone \n',
       data1[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean(),
      '\n','-'*10, '\n')

##Handling continuous data##
#qcut or frequency bins: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.qcut.html
#cut or value bins: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.cut.html
#qcut vs cut: https://stackoverflow.com/questions/30211923/what-is-the-difference-between-pandas-qcut-and-pandas-cut

#Fare Bins/Buckets using qcut
data1['FareBin'] = pd.qcut(data1['Fare'], 4)
print ('Survival Correlation by: FareBin \n',
       data1[['FareBin', 'Survived']].groupby(['FareBin'], as_index=False).mean(),
      '\n','-'*10, '\n')

#Age Bins/Buckets using cut
data1['AgeBin'] = pd.cut(data1['Age'].astype(int), 5)
print ('Survival Correlation by: AgeBin \n',
       data1[['AgeBin', 'Survived']].groupby(['AgeBin'], as_index=False).mean(),
      '\n','-'*10, '\n')


#simple frequency table of class and sex
#using crosstabs: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.crosstab.html
print(pd.crosstab(data1['Pclass'], data1['Sex']))


# In[ ]:


#optional plotting w/pandas: https://pandas.pydata.org/pandas-docs/stable/visualization.html

#we will use matplotlib.pyplot: https://matplotlib.org/api/pyplot_api.html
#to organize our graphics will use figure: https://matplotlib.org/api/_as_gen/matplotlib.pyplot.figure.html#matplotlib.pyplot.figure
#subplot: https://matplotlib.org/api/_as_gen/matplotlib.pyplot.subplot.html#matplotlib.pyplot.subplot
#and subplotS: https://matplotlib.org/api/_as_gen/matplotlib.pyplot.subplots.html?highlight=matplotlib%20pyplot%20subplots#matplotlib.pyplot.subplots

#graph distribution of quantitative data
plt.figure(figsize=[16,12])

plt.subplot(231)
plt.boxplot(x=data1['Fare'], showmeans = True, meanline = True)
plt.title('Fare Boxplot')
plt.ylabel('Fare ($)')

plt.subplot(232)
plt.boxplot(data1['Age'], showmeans = True, meanline = True)
plt.title('Age Boxplot')
plt.ylabel('Age (Years)')

plt.subplot(233)
plt.boxplot(data1['FamilySize'], showmeans = True, meanline = True)
plt.title('Family Size Boxplot')
plt.ylabel('Family Size (#)')

plt.subplot(234)
plt.hist(x = [data1[data1['Survived']==1]['Fare'], data1[data1['Survived']==0]['Fare']], 
         stacked=True, color = ['g','r'],label = ['Survived','Dead'])
plt.title('Fare Histogram by Survival')
plt.xlabel('Fare ($)')
plt.ylabel('# of Passengers')
plt.legend()

plt.subplot(235)
plt.hist(x = [data1[data1['Survived']==1]['Age'], data1[data1['Survived']==0]['Age']], 
         stacked=True, color = ['g','r'],label = ['Survived','Dead'])
plt.title('Age Histogram by Survival')
plt.xlabel('Age (Years)')
plt.ylabel('# of Passengers')
plt.legend()

plt.subplot(236)
plt.hist(x = [data1[data1['Survived']==1]['FamilySize'], data1[data1['Survived']==0]['FamilySize']], 
         stacked=True, color = ['g','r'],label = ['Survived','Dead'])
plt.title('Family Size Histogram by Survival')
plt.xlabel('Family Size (#)')
plt.ylabel('# of Passengers')
plt.legend()


#we will use seaborn graphics for multi-variable comparison: https://seaborn.pydata.org/api.html

#graph distribution of qualitative data: Pclass
#we know class mattered in survival, now let's compare class and a 2nd feature
fig, (axis1,axis2,axis3) = plt.subplots(1,3,figsize=(16,12))

sns.boxplot(x = 'Pclass', y = 'Fare', hue = 'Survived', data = data1, ax = axis1)
axis1.set_title('Pclass vs Fare Survival Comparison')

sns.violinplot(x = 'Pclass', y = 'Age', hue = 'Survived', data = data1, split = True, ax = axis2)
axis2.set_title('Pclass vs Age Survival Comparison')

sns.boxplot(x = 'Pclass', y ='FamilySize', hue = 'Survived', data = data1, ax = axis3)
axis3.set_title('Pclass vs Family Size Survival Comparison')


#graph distribution of qualitative data: Sex
#we know sex mattered in survival, now let's compare sex and a 2nd feature
fig, qaxis = plt.subplots(1,3,figsize=(16,12))

sns.barplot(x = 'Sex', y = 'Survived', hue = 'Embarked', data=data1, ax = qaxis[0])
axis1.set_title('Sex vs Embarked Survival Comparison')

sns.barplot(x = 'Sex', y = 'Survived', hue = 'Pclass', data=data1, ax  = qaxis[1])
axis1.set_title('Sex vs Pclass Survival Comparison')

sns.barplot(x = 'Sex', y = 'Survived', hue = 'IsAlone', data=data1, ax  = qaxis[2])
axis1.set_title('Sex vs IsAlone Survival Comparison')


#graph individual features by survival
fig, saxis = plt.subplots(2, 3,figsize=(16,12))

sns.barplot(x = 'Embarked', y = 'Survived', data=data1, ax = saxis[0,0])
sns.barplot(x = 'Pclass', y = 'Survived', order=[1,2,3], data=data1, ax = saxis[0,1])
sns.barplot(x = 'IsAlone', y = 'Survived', order=[1,0], data=data1, ax = saxis[0,2])

g = sns.factorplot(x = 'FareBin', y = 'Survived',  data=data1, ax = saxis[1,0])
h = sns.factorplot(x = 'AgeBin', y = 'Survived',  data=data1, ax = saxis[1,1])
j = sns.factorplot(x = 'FamilySize', y = 'Survived', data=data1, ax = saxis[1,2])
#close factor plot facetgrid we don't need: https://stackoverflow.com/questions/33925494/seaborn-produces-separate-figures-in-subplots
plt.close(g.fig)
plt.close(h.fig)
plt.close(j.fig)


#more side-by-side comparisons
fig, (maxis1, maxis2) = plt.subplots(1, 2,figsize=(16,12))

#how does family size factor with sex & survival
sns.pointplot(x="FamilySize", y="Survived", hue="Sex", data=data1,
              palette={"male": "blue", "female": "pink"},
              markers=["*", "o"], linestyles=["-", "--"], ax = maxis1)

#how does class factor with sex & survival
sns.pointplot(x="Pclass", y="Survived", hue="Sex", data=data1,
              palette={"male": "blue", "female": "pink"},
              markers=["*", "o"], linestyles=["-", "--"], ax = maxis2)

#how does embark port factor with class, sex, and survival
plt.figure()
sns.factorplot(x="Pclass", y="Survived", hue="Sex", col="Embarked",
                   data=data1, aspect=0.9, size=3.5, ci=95.0)


#plot distributions of Age of passengers who survived or did not survive
def plot_distribution( df , feature , target , **kwargs ):
    plt.figure()
    row = kwargs.get( 'row', None )
    col = kwargs.get( 'col', None )
    facet = sns.FacetGrid( df, hue=target, aspect=4, row=row ,col=col )
    facet.map( sns.kdeplot, feature, shade= True )
    facet.set( xlim=( 0 , df[feature].max() ) )
    facet.add_legend()

plot_distribution(data1 , feature='Age' , target='Survived' , row='Sex')


#pair plots
plt.figure()
f = sns.pairplot(data1, hue='Survived', palette = 'seismic',size=1.2,diag_kind = 'kde',diag_kws=dict(shade=True),plot_kws=dict(s=10) )
f.set(xticklabels=[])


#correlation heatmap
def correlation_heatmap(df):
    plt.figure()
    _ , ax = plt.subplots(figsize =(14, 12))
    #colormap = plt.cm.viridis
    colormap = sns.diverging_palette(220, 10, as_cmap = True)
    
    _ = sns.heatmap(
        df.corr(), 
        cmap = colormap,
        square=True, 
        cbar_kws={'shrink':.9 }, 
        ax=ax,
        annot=True, 
        linewidths=0.1,vmax=1.0, linecolor='white',
        annot_kws={'fontsize':12 }
    )
    
    plt.title('Pearson Correlation of Features', y=1.05, size=15)

correlation_heatmap(data1)






# # Step 5: Model Data
# 
# Data Science is a multi-disciplinary field between mathematics (i.e. statistics, linear algebra, etc.), computer science (i.e. programming languages, computer systems, etc.) and business management (i.e. communication, subject-matter knowledge, etc.). Most data scientist come from one of the three fields, so they tend to lean towards that disciple. However, data science is like a three-legged stool, with no one leg being more important than the other. With that being said, this step requires advanced knowledge in mathematics. But thanks to computer science, a lot of the heavy lifting is done for you. So, problems that once required graduate mathematic degrees, now only take a few lines of code.
# 
# This is both good and bad. It’s good because these algorithms are now accessible to more people that can solve more problems in the world. It’s bad because a lower barrier to entry means more people will not know the tools they are using and can come to incorrect conclusions. That’s why I focus on teaching you, not just what to do, but why you’re doing it. Previously, I used the analogy of asking someone to hand you a Philip screwdriver, and they hand you a flathead screwdriver or worst a hammer. At best, it shows a complete lack of understanding. At worst, it makes completing the project impossible; or even worst incorrect. So now that I’ve hammered (no pun intended) my point, I’ll show you what to do and most importantly, WHY you do it.
# 
# First, you must understand, that the purpose of machine learning is to solve human problems. Machine learning can be categorized as: supervised learning, unsupervised learning, and reinforced learning. Supervised learning is where you train the model by presenting it a training dataset that includes the correct target response. Unsupervised learning is where you train the model using a training dataset that does not include the correct target response. And reinforced learning is a hybrid of the previous two, where the model is not given the correct target response immediately, but later in a sequence of events to reinforce learning. We are doing supervised machine learning, because we are training our algorithm by presenting it with a set of features and their corresponding target. We then hope to present it a new subset from the same dataset and have similar results in prediction accuracy.
# 
# There are many machine learning algorithms, however they can be reduced to four categories: classification, regression, clustering, or dimensionality reduction, depending on your target variable and data modeling goals. We can generalize that a continuous target variable requires a regression algorithm and a discrete target variable requires a classification algorithm. One side note, logistic regression, while it has regression in the name, is really a classification algorithm. Since our problem is predicting if a passenger survived or did not survive, this is a discrete target variable. We will use a classification algorithm from the *scikit* library to begin our analysis.
# 
# **Machine Learning Selection**
# * [SciKit Estimator Overview](http://scikit-learn.org/stable/user_guide.html)
# * [SciKit Estimator Detail](http://scikit-learn.org/stable/modules/classes.html)
# * [Choosing Estimator Mind Map](http://scikit-learn.org/stable/tutorial/machine_learning_map/index.html)
# * [Choosing Estimator Cheat Sheet](https://s3.amazonaws.com/assets.datacamp.com/blog_assets/Scikit_Learn_Cheat_Sheet_Python.pdf)
# 
# 
# Now that we identified our solution as a supervised learning classification algorithm. We can narrow our list of choices.
# 
# **Machine Learning Classification Algorithms:**
# * [Ensemble Methods](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.ensemble)
# * [Generalized Linear Models (GLM)](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.linear_model)
# * [Naive Bayes](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.naive_bayes)
# * [Nearest Neighbors](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.neighbors)
# * [Support Vector Machines (SVM)](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.svm)
# * [Decision Trees](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.tree)
# 

# In[ ]:


#Machine Learning Algorithm (MLA) Selection and initialization
MLA = [
    #Ensemble Methods
    ensemble.AdaBoostClassifier(),
    ensemble.BaggingClassifier(),
    ensemble.ExtraTreesClassifier(),
    ensemble.GradientBoostingClassifier(),
    ensemble.RandomForestClassifier(n_estimators = 100),
    #ensemble.VotingClassifier(),

    #Gaussian Processes
    gaussian_process.GaussianProcessClassifier(),
    
    #GLM
    linear_model.LogisticRegressionCV(),
    linear_model.PassiveAggressiveClassifier(),
    linear_model. RidgeClassifierCV(),
    linear_model.SGDClassifier(),
    linear_model.Perceptron(),
    
    #Navies Bayes
    naive_bayes.GaussianNB(),
    
    #Nearest Neighbor
    neighbors.KNeighborsClassifier(n_neighbors = 3),
    
    #SVM
    svm.SVC(probability=True),
    svm.LinearSVC(),
    
    #Trees    
    tree.DecisionTreeClassifier(),
    tree.ExtraTreeClassifier(),
    
    #Discriminant Analysis
    discriminant_analysis.LinearDiscriminantAnalysis(),
    discriminant_analysis.QuadraticDiscriminantAnalysis()
    ]

#create table to compare MLA
MLA_columns = ['MLA Name', 'MLA Parameters','MLA Train Accuracy', 'MLA Test Accuracy', 'MLA Best Accuracy', 'MLA Time']
MLA_compare = pd.DataFrame(columns = MLA_columns)


#index through MLA and save to table
row_index = 0
for alg in MLA:
    #set name column
    MLA_compare.loc[row_index, 'MLA Name'] = alg.__class__.__name__
    
    #get and set algorithm, execution time, and accuracy
    start_time = t.default_timer()
    alg.fit(train1_dummy_x,train1_dummy_y)
    run_time = t.default_timer() - start_time     
    
    MLA_compare.loc[row_index, 'MLA Parameters'] = str(alg.get_params())
    MLA_compare.loc[row_index, 'MLA Time'] = run_time
    MLA_compare.loc[row_index, 'MLA Train Accuracy'] = alg.score(train1_dummy_x,train1_dummy_y)*100
    MLA_compare.loc[row_index, 'MLA Test Accuracy'] = alg.score(test1_dummy_x,test1_dummy_y)*100
    
    row_index+=1

#print and sort table: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.sort_values.html
MLA_compare.sort_values(by = ['MLA Test Accuracy'], ascending = False, inplace = True)
print(MLA_compare)
MLA_compare


#logreg = LogisticRegression()
#%timeit logreg.fit(train1_dummy_x,train1_dummy_y) 
#print("LogReg Training w/dummy set score: {:.2f}". format(logreg.score(train1_dummy_x,train1_dummy_y)*100)) 
#print("LogReg Test w/dummy set score: {:.2f}". format(logreg.score(test1_dummy_x,test1_dummy_y)*100))
print('-'*10,)



# In[ ]:


#barplot using https://seaborn.pydata.org/generated/seaborn.barplot.html
sns.barplot(x='MLA Test Accuracy', y = 'MLA Name', data = MLA_compare, color = 'm')

#prettify using pyplot: https://matplotlib.org/api/pyplot_api.html
plt.title('Machine Learning Algorithm Accuracy Score \n')
plt.xlabel('Accuracy Score (%)')
plt.ylabel('Algorithm')

# ## 5.1 Model Optimization
# 
# Let's recap, with some basic data cleaning, analysis, and machine learning algorithms (MLA), we are able to predict passenger survival with ~85% accuracy. If this were a college course, that would be a B-grade. Not bad for a few lines of code. But the question we always ask is, can we do better and more importantly get an ROI (return on investment) for our time invested? For example, if we're only going to increase our accuracy by 1/10th of a percent, is it really worth 3-months of model optimization. If you work in research maybe the answer is yes, but if you work in business mostly the answer is no. So, keep that in mind.
# 
# For model optimization, we have a couple options: 1) find a "better" algorithm, 2) tune our current algorithm parameters, 3) feature engineer new variables to find new signals in the data, or 4) we can go back to the beginning and determine if we asked the right questions, got the right data, and made the right decisions along the process.
# 
# ### Data Science 101: Determine a Baseline Accuracy ###
# Before we decide how-to make our model better, let's determine if ~85% is good enough. To do that, we have to go back to the basics of data science 101. We know this is a binary problem, because there are only two possible outcomes; passengers survived or died. So, think of it like a coin flip. If you have a fair coin and you guessed heads or tail, then you have a 50-50 chance of guessing right. So, let's set 50% as an F grade, because if your model accuracy is any worse than that, then why do I need you when I can just flip a coin?
# 
# Okay, so with no information about the dataset, we can always get 50% with a binary problem. But we have information about the dataset, so we should be able to do better. We know that 1,502/2,224 or 67.5% of people died. Therefore, If I just guessed that 100% of people died, then I would be right 67.5% of the time. So, let's set 68% as a D grade, because again, if your model accuracy is any worst that that, then why do I need you, when I can just assume if you were on the Titanic that day you died and have a 68% accuracy.
# 
# ### Data Science 101: How-to Create Your Own Model ###
# Our accuracy is increasing, but can we do better? Are there any signals in our data? To illustrate this, we're going to build our own decision tree model, because it is the easiest to conceptualize and requires simple addition and multiplication calculations. When creating a decision tree, you want to ask questions that gives you better information about your outcome by segregated the survived/1 from the dead/0. This is part science and part art, so let's just play the 21-question game to show you how it works. If you want to follow along on your own, download the train dataset and import into Excel. Create a pivot table with survival in the columns, count and % of row count in the values, and the features described below in the rows.
# 
# Remember, the name of the game is to create subsets using a decision tree model to get survived/1 in one bucket and dead/0 in another bucket. Our rule of thumb will be the majority rules. Meaning, if the majority or 50% or more survived, then everybody in our subgroup survived/1, but if 50% or less survived then if everybody in our subgroup died/0. Also, we will stop if the subgroup is 10% of our total dataset or 9 cases and/or our model accuracy plateaus or decreases. Got it? Let's go!
# 
# ***Question 1: Were you on the Titanic?*** If Yes, then majority (62%) died. Note our sample survival is different than our population of 68%. Nonetheless, if we assumed everybody died, our sample accuracy is 62%.
# 
# ***Question 2: Are you male or female?*** Male, majority (81%) died. Female, majority (74%) survived. Giving us an accuracy of 79%.
# 
# ***Question 3A (going down the female branch with count = 344): Are you in class 1, 2, or 3?*** Class 1, majority (97%) survived and Class 2, majority (92%) survived. Since are dead group is less than 9, we will stop going down this branch. Class 3, is even at a 50-50 split. No new information to improve our model is gained.
# 
# ***Question 4A (going down the female class 3 branch with count = 144): Did you embark from port C, Q, or S?*** We gain a little information. C and Q, the majority still survived, so no change. Also, the dead subgroup is 9 or less, so we will stop. S, the majority (63%) died. So, we will change females, class 3, embarked S from assuming they survived, to assuming they died. Our model accuracy increases to 81%. 
# 
# ***Question 5A (going down the female class 3 embarked S branch with count = 88):*** So far, it looks like we made good decisions. Adding another level does not seem to gain much more information. This subgroup 55 died and 33 survived, since majority died we need to find a signal to identify the 33 or a subgroup to change them from dead to survived and improve our model accuracy. We can play with our features. One I found was fare 0-8, majority survived. It's a small sample size 11-9, but one often used in statistics. We slightly improve our accuracy, but not much to move us past 82%. So, we'll stop here.
# 
# ***Question 3B (going down the male branch with count = 577):*** Going back to question 2, we know the majority of males died. So, we are looking for a feature that identifies a subgroup that majority survived. Surprisingly, class or even embarked didn't matter, title does and gets us to 82%. Guess and checking other features, none seem to push us past 82%. So, we'll stop here for now.
# 
# So, with very little information, we get to 82% accuracy. We'll give that a grade of a C for average or our baseline. But can we do better? By making a better decision tree, new features, etc. Before we do, let's code what we just wrote above.
# 

# In[ ]:


#create a 2nd copy of our data
data2 = data_raw.copy(deep = True)

#Feature Engineering
#Note: we will not do any imputing missing data at this time
data2['FareBin'] = pd.qcut(data1['Fare'], 4)
data2['AgeBin'] = pd.cut(data1['Age'].astype(int), 5)
data2['Title'] = data2['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]
#data2.sample(10)


#coin flip model with random 1/survived 0/died

#Iterate over DataFrame rows as (index, Series) pairs: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.iterrows.html
for index, row in data2.iterrows(): 
    #random number generator: https://docs.python.org/2/library/random.html
    if random.random() > .5:     # Random float x, 0.0 <= x < 1.0    
        data2.set_value(index, 'Random_Predict', 1) #predict survived/1
    else: 
        data2.set_value(index, 'Random_Predict', 0) #predict died/0
    

#score random guess of survival. Use shortcut 1 = Right Guess and 0 = Wrong Guess
#the mean of the column will then equal the accuracy
data2['Random_Score'] = 0 #assume prediction wrong
data2.loc[(data2['Survived'] == data2['Random_Predict']), 'Random_Score'] = 1 #set to 1 for correct prediction
print('Coin Flip Model Accuracy: {:.2f}%'.format(data2['Random_Score'].mean()*100))

#we can also use scikit's accuracy_score function to save us a few lines of code
#http://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html#sklearn.metrics.accuracy_score
print('Coin Flip Model Accuracy w/SciKit: {:.2f}%'.format(metrics.accuracy_score(data2['Survived'], data2['Random_Predict'])*100))



#group by or pivot table: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.groupby.html
pivot_female = data2[data2.Sex=='female'].groupby(['Sex','Pclass', 'Embarked','FareBin'])['Survived'].mean()
print('\n\nSurvival Decision Tree w/Female Node: \n',pivot_female)

pivot_male = data2[data2.Sex=='male'].groupby(['Sex','Title'])['Survived'].mean()
print('\n\nSurvival Decision Tree w/Male Node: \n',pivot_male)

#Question 1: Were you on the Titanic; majority died
data2['Tree_Predict'] = 0

#Question 2: Are you female; majority survived
data2.loc[(data2['Sex'] == 'female'), 'Tree_Predict'] = 1

#Question 3A Female - Class and Question 4 Embarked gain minimum information

#Question 5B Female - FareBin; set anything less than .5 in female node decision tree back to 0
data2.loc[(data2['Sex'] == 'female') & (data2['Pclass'] == 3) & 
          (data2['Embarked'] == 'C') & (data2['Fare'] > 8) & (data2['Fare'] <15),
          'Tree_Predict'] = 0

data2.loc[(data2['Sex'] == 'female') & (data2['Pclass'] == 3) & 
          (data2['Embarked'] == 'S') & (data2['Fare'] > 8),
          'Tree_Predict'] = 0

#Question 3B Male: Title; set anything greater than .5 to 1 for majority survived
male_title = ['Master', 'Sir']
data2.loc[(data2['Sex'] == 'male') &
          (data2['Title'].isin(male_title)),
          'Tree_Predict'] = 1

#Score Decision Tree Model
print('\n\nDecision Tree Model Accuracy: {:.2f}%\n'.format(metrics.accuracy_score(data2['Survived'], data2['Tree_Predict'])*100))

#Accuracy Summary Report with http://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html#sklearn.metrics.classification_report
#Where recall score = (true positives)/(true positive + false negative) w/1 being best:http://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html#sklearn.metrics.recall_score
#And F1 score = weighted average of precision and recall w/1 being best: http://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
print(metrics.classification_report(data2['Survived'], data2['Tree_Predict']))

#Credit: http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Compute confusion matrix
cnf_matrix = metrics.confusion_matrix(data2['Survived'], data2['Tree_Predict'])
np.set_printoptions(precision=2)

class_names = ['Dead', 'Survived']
# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True, 
                      title='Normalized confusion matrix')


# # Credits
# Programming is all about "borrowing" code, because knife sharpens knife. Nonetheless, I want to give credit, where credit is due. 
# 
# * Müller, Andreas C.; Guido, Sarah. Introduction to Machine Learning with Python: A Guide for Data Scientists. O'Reilly Media.
# 
# 

# 
# 
