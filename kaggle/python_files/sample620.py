#!/usr/bin/env python
# coding: utf-8

# This is my first kernel on Kaggle. Would really appreciate any feedback that I can get.
# 
# Titanic is one of the most popular dataseta on Kaggle in which we have to perform a binary classification to predict the survival of the passengers.
# Since I am fairly new to data science so this kernel will be simple and learner oriented. I have added links for the libraries, methods, etc. for 
# easy reference and will update this kernel further with more information and code for few other tasks that I feel are currently missing from this kernel.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

# In[ ]:


# Used to display the full output without the ...
pd.set_option('display.expand_frame_repr', False)

# Ignore warnings thrown by Seaborn
import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn

# In[ ]:


# Import charting libraries
import matplotlib.pyplot as plt
import seaborn as sns

# We begin by picking the two input files, train and validation

# In[ ]:


train = pd.read_csv("../input/train.csv")
validation = pd.read_csv("../input/test.csv")

# We use Pandas [read_csv](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html) to fetch the two datasets. This method returns new [DataFrame](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html) objects which we then store in train and validation.

# **Step 1:  Poking the Data**
# 
# First we need to get an initial picture of the data available to us.

# In[ ]:


# Check the train dataset.
print(train.columns.values)
train.describe(include='all')

# Here, train.[columns](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.columns.html#pandas.DataFrame.columns) returns all the column labels in the train DataFrame and we then use [values](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.values.html?highlight=values#pandas.Series.values) to get an easy to print numpy ndarray.
# 
# Next we use train.[describe](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.describe.html) to the indicative stats that allow us to quickly glance through the data to see number of null values, spread, outliers, etc. We used include='all' to also include non numeric features in the describe output.
# 
# We have 12 features in the dataset. The details of these features are provided by Kaggle [here](https://www.kaggle.com/c/titanic/data).  The meaning of these features is quite easy to understand but the real trick lies in how we actually use this data to train our models. There are a total of 891 rows in the train data. From the count field we can see that the Age, Cabin and Embarked have null values.
# It is worth noting that the data provided is [tidy](https://en.wikipedia.org/wiki/Tidy_data). Tidy data sets means that each feature is stored as a column and each observation is stored as a row. Unlike the datasets provided here by Kaggle, real world data may need to be converted to this format before we begin. Tidy data makes it much easier for us to perform [data cleaning](https://en.wikipedia.org/wiki/Data_cleansing).

# In[ ]:


# Understand the datatypes
print(train.dtypes)
print()
# Focus first on null values
print(train.isna().sum())

# The [dtypes](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.dtypes.html) method tells us the data types of the features. We follow different preprocessing steps for different types of data. Checking for nulls with [isna](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.isna.html) gives us an indication of how much data is missing and the importance of features. Too many nulls may lead to an unsalvageable feature.
# We have nulls in Age, Cabin and Embarked in the train dataset.

# In[ ]:


# Check the validation dataset also.
print(train.columns.values)
train.describe(include='all')

# In[ ]:


print(validation.dtypes)
print()
print(validation.isna().sum())

# We have nulls in Age, Cabin and Fare in the validation dataset.
# 
# Next we will check the correlation coefficients for the provided data using the [corr](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.corr.html) method. This gives us an initial idea about the relationship amongst the various numerical features. We use this knowledge to direct our exploration and analysis of the data.

# In[ ]:


# Check the correlation for the current numeric feature set.
print(train[['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']].corr())
sns.heatmap(train[['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']].corr(), annot=True, fmt = ".2f", cmap = "coolwarm")

# **Step 2: Exploring the data**
# 
# We will explore the features in a sequential manner since the number of features is not big.

# In[ ]:


# List the features again
print(train.columns.values)

# **Pclass:** It has a negative correlation of -0.33 which means that increase in Pclass leads to decrease in Survived. We group the data by passing the Feature as an attribute to the [groupby](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.groupby.html) method.

# In[ ]:


# Lets see the relation between Pclass and Survived
print(train[['Pclass', 'Survived']].groupby(['Pclass']).mean())
sns.catplot(x='Pclass', y='Survived',  kind='bar', data=train)

# The passengers in a higher class definitely had a much higher chance of survival.

# **Sex:**
# Does gender play a part too?

# In[ ]:


print(train[['Sex', 'Survived']].groupby(['Sex']).mean())
sns.catplot(x='Sex', y='Survived',  kind='bar', data=train)

# Women had a much higher chance of survival than men. Let's see the impact of Pclass and Sex together on Survived.

# In[ ]:


sns.catplot(x='Sex', y='Survived',  kind='bar', data=train, hue='Pclass')

# No wonder Jack died and Rose lived in the movie :)

# **Fare:** Let's check fare next because it is closely correlated to Pclass. 

# In[ ]:


g = sns.FacetGrid(train, col='Survived')
g = g.map(sns.distplot, "Fare")

# We notice that Fare for Survived = 0 is heavily skewed towards the lower end. Also for Survived = 1, we see that Fare has a long trail towards the right which shows that passenegers who paid very high fares were much safer.

# In[ ]:


group = pd.cut(train.Fare, [0,50,100,150,200,550])
piv_fare = train.pivot_table(index=group, columns='Survived', values = 'Fare', aggfunc='count')
piv_fare.plot(kind='bar')

# As the fare increases, so does the chances of survival.

# **Age:** Age had a low inverse correlation with Survived(-0.07). Is it really useful?

# In[ ]:


g = sns.FacetGrid(train, col='Survived')
g = g.map(sns.distplot, "Age")

# We can glean that children had a high probability of Survival.

# In[ ]:


group = pd.cut(train.Age, [0,14,30,60,100])
piv_fare = train.pivot_table(index=group, columns='Survived', values = 'Age', aggfunc='count')
piv_fare.plot(kind='bar')

# Through this bar plot we can confirm that children did have a higher chance of survival even though Age as a whole is not a strongly correlated feature with Survived. 

# **Embarked:** Did the embarkation location play any part?

# In[ ]:


print(train[['Embarked', 'Survived']].groupby(['Embarked']).mean())
sns.catplot(x='Embarked', y='Survived',  kind='bar', data=train)

# In[ ]:


sns.catplot('Pclass', kind='count', col='Embarked', data=train)

# So people who Embarked from C had the highest chance of survival because most of them where of Pclass 1 which has the highest Survival rate.

# **SibSp and Parch**
# Both of these features are related to the family size of the passengers.

# In[ ]:


print(train[['SibSp', 'Survived']].groupby(['SibSp']).mean())
sns.catplot(x='SibSp', y='Survived', data=train, kind='bar')

# People with 0-2 SibSp(Sibling/Spouse) have a higher chance of survival.

# In[ ]:


print(train[['Parch', 'Survived']].groupby(['Parch']).mean())
sns.catplot(x='Parch', y='Survived', data=train, kind='bar')

# We can pull two types of information from this data easily, the family size(FamilySize) and whether the person is travelling alone(IsAlone).

# **Name**
# 
# Although the name data might seem inconsequential but there are useful insights that we can get from this data. The titles from the name data can give us an indication of the social/economic standing of the passenger.

# I have used [str](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.str.html) method to get the StringMethods object over the Series and [split](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.str.split.html) to split the strings around the separators.
# Pandas documentation has an easy to understand page titled "[Working with Text Data](https://pandas.pydata.org/pandas-docs/stable/text.html)" which would bring you upto the speed quickly.

# In[ ]:


# Explanation for the next code block used to get the Titles
# Using this output to explain the string manipulation below
print(train.Name.head(1))
print()
# The above returns a single name for e.g. Braund, Mr. Owen Harris.
# Calling str returns a String object
print(train.Name.head(1).str)
print()
# Next we split the string into a List with a comma as the separator
print(train.Name.head(1).str.split(','))
print()
# Similary we remove the . and then strip the remaining string to get the title.
# We pick the second item of the
print(train.Name.head(1).str.split(',').str[1])
print()

# In[ ]:


# Get the titles
for dataset in [train, validation]:
    # Use split to get only the titles from the name
    dataset['Title'] = dataset['Name'].str.split(',').str[1].str.split('.').str[0].str.strip()
    # Check the initial list of titles.
    print(dataset['Title'].value_counts())
    print()

# In[ ]:


sns.catplot(x='Survived', y='Title', data=train, kind ='bar')

# We again notice that the Titles used for females have a much higher rate of survival.

# **Ticket**
# 
# WIP. We will drop this from the dataset for now.

# **Step 3: Feature Engineering and Processing**

# **Fixing the missing data : **
# First, we will focus on removing the null fields. Lets see all the features with nulls again.

# In[ ]:


for df in [train, validation]:
    print(df.shape)
    print()
    print(df.isna().sum())

# **Embarked:** I decided to drop the passengers who didn't embark since modeling based on their data would act like noise in my opinion. I feel that they can't reliably tell us about the survived/not survived ouptut.
# Cross verification - Using Embarked.mode() in fillna to handle the 2 nulls in train, lowered my accuracy to 0.77 from 0.79.

# In[ ]:


# Drop rows with nulls for Embarked
for df in [train, validation]:
    df.dropna(subset = ['Embarked'], inplace = True)

# **Fare**
# 
# We saw earlier that the validation dataset has a null Fare. The train dataset has no 
# null fares.

# In[ ]:


print(train[train['Fare'].isnull()])
print() 
# 1 row with null Fare in validation
print(validation[validation['Fare'].isnull()])
# We can deduce that Pclass should be related to Fares.
sns.catplot(x='Pclass', y='Fare', data=validation, kind='point')

# In[ ]:


# There is a clear relation between Pclass and Fare. We can use this information to impute the missing fare value.
# We see that the passenger is from Pclass 3. So we take a median value for all the Pclass 3 fares.
validation['Fare'].fillna(validation[validation['Pclass'] == 3].Fare.median(), inplace = True)

# **Age**
# 
# One quick way to impute the missing ages would be to take median of the Age data from the and apply it to all using [fillna](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.fillna.html). An alternative approach that I have used with better results is to impute ages based on passeneger titles. This is because of the inherent age related meaning assigned to titles like Master, Mr, Miss, Mrs etc. This lets us assign age more accurately for individual passengers.

# In[ ]:


print(train[['Age','Title']].groupby('Title').mean())
sns.catplot(x='Age', y='Title', data=train, kind ='bar')

# We can see that different titles have different age means.

# In[ ]:


# Returns titles from the passed in series.
def getTitle(series):
    return series.str.split(',').str[1].str.split('.').str[0].str.strip()
# Prints the count of titles with nulls for the train dataframe.
print(getTitle(train[train.Age.isnull()].Name).value_counts())
# Fill Age median based on Title
mr_mask = train['Title'] == 'Mr'
miss_mask = train['Title'] == 'Miss'
mrs_mask = train['Title'] == 'Mrs'
master_mask = train['Title'] == 'Master'
dr_mask = train['Title'] == 'Dr'
train.loc[mr_mask, 'Age'] = train.loc[mr_mask, 'Age'].fillna(train[train.Title == 'Mr'].Age.mean())
train.loc[miss_mask, 'Age'] = train.loc[miss_mask, 'Age'].fillna(train[train.Title == 'Miss'].Age.mean())
train.loc[mrs_mask, 'Age'] = train.loc[mrs_mask, 'Age'].fillna(train[train.Title == 'Mrs'].Age.mean())
train.loc[master_mask, 'Age'] = train.loc[master_mask, 'Age'].fillna(train[train.Title == 'Master'].Age.mean())
train.loc[dr_mask, 'Age'] = train.loc[dr_mask, 'Age'].fillna(train[train.Title == 'Dr'].Age.mean())
# Prints the count of titles with nulls for the train dataframe. -- Should be empty this time.
print()
print(getTitle(train[train.Age.isnull()].Name).value_counts())

# In[ ]:


# Prints the count of titles with nulls for the validation dataframe.
print(getTitle(validation[validation.Age.isnull()].Name).value_counts())
# Fill Age median based on Title
mr_mask = validation['Title'] == 'Mr'
miss_mask = validation['Title'] == 'Miss'
mrs_mask = validation['Title'] == 'Mrs'
master_mask = validation['Title'] == 'Master'
ms_mask = validation['Title'] == 'Ms'
validation.loc[mr_mask, 'Age'] = validation.loc[mr_mask, 'Age'].fillna(validation[validation.Title == 'Mr'].Age.mean())
validation.loc[miss_mask, 'Age'] = validation.loc[miss_mask, 'Age'].fillna(validation[validation.Title == 'Miss'].Age.mean())
validation.loc[mrs_mask, 'Age'] = validation.loc[mrs_mask, 'Age'].fillna(validation[validation.Title == 'Mrs'].Age.mean())
validation.loc[master_mask, 'Age'] = validation.loc[master_mask, 'Age'].fillna(validation[validation.Title == 'Master'].Age.mean())
validation.loc[ms_mask, 'Age'] = validation.loc[ms_mask, 'Age'].fillna(validation[validation.Title == 'Miss'].Age.mean())
# Prints the count of titles with nulls for the validation dataframe. -- Should be empty this time.
print(getTitle(validation[validation.Age.isnull()].Name).value_counts())

# In[ ]:


# train.Age.fillna(train.Age.median(), inplace=True)
# validation.Age.fillna(validation.Age.median(), inplace=True)
print(train.isna().sum())
print(validation.isna().sum())

# We will begin by dropping the columns that we are not using.

# In[ ]:


train.drop(columns=['PassengerId'], inplace = True)
[df.drop(columns=['Ticket'], inplace = True) for df in [train, validation]]

# Next up, we will encode all the categorical features. One of things to lookout for is the Dummy Trap. We have the option for using "drop_first = True" with get_dummies but here we will manually select the features and intentionally leave one out based on their correlations.

# In[ ]:


[train, validation] = [pd.get_dummies(data = df, columns = ['Pclass', 'Sex', 'Embarked']) for df in [train, validation]]

# We will convert the Cabin data into a flag about whether a passenger had an assigned cabin or not. Also we will use SibSp and Parch to calculate the Family Size and a flag named IsAlone.

# In[ ]:


for df in [train, validation]:
    df['HasCabin'] = df['Cabin'].notna().astype(int)
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] > 1).astype(int)

# In[ ]:


[df.drop(columns=['Cabin', 'SibSp', 'Parch'], inplace = True) for df in [train, validation]]

# In[ ]:


# We see that there are a few non standard titles. Some of them are just French titles
# with the same meaning as in English while others point to people who would probably
# have more privileges or military training etc and can be placed in a separate category.
# French titles - https://en.wikipedia.org/wiki/French_honorifics
# Mlle - https://en.wikipedia.org/wiki/Mademoiselle_(title)
# Mme - https://en.wikipedia.org/wiki/Madam
# Mme was a bit harder to understand as Wikipedia says that its used for adult women
# but doesn't given any pointers towards their marital status.
# Searching up on Google and considering that the title is used for adult women
# we can assume that this title was usually assigned to married women.
# https://www.frenchtoday.com/blog/french-culture/madame-or-mademoiselle-a-delicate-question
# Ms - An alternate abbrevation for Miss
train['Title'] = train['Title'].replace('Mlle', 'Miss').replace('Ms', 'Miss').replace('Mme', 'Mrs').replace(['Dr', 'Major', 'Col', 'Rev', 'Lady', 'Jonkheer', 'Don', 'Sir', 'Dona', 'Capt', 'the Countess'], 'Special')
validation['Title'] = validation['Title'].replace('Mlle', 'Miss').replace('Ms', 'Miss').replace('Mme', 'Mrs').replace(['Dr', 'Major', 'Col', 'Rev', 'Lady', 'Jonkheer', 'Don', 'Sir', 'Dona', 'Capt', 'the Countess'], 'Special')
[df.drop(columns=['Name'], inplace = True) for df in [train, validation]]
[train, validation] = [pd.get_dummies(data = df, columns = ['Title']) for df in [train, validation]]

# In[ ]:


# Check the updated dataset
print(train.columns.values)
print(validation.columns.values)

# In[ ]:


# Check the correlation with the updated datasets
train.corr()

# In[ ]:


# Split the the dataset into train and test sets.
from sklearn.model_selection import train_test_split
# Use only the features with a coeefficient greater than 0.3
X = train[['Age', 'Fare', 'Pclass_2', 'Pclass_3', 'Sex_female', 'Embarked_C',
       'Embarked_S', 'HasCabin', 'FamilySize', 'Title_Master', 'Title_Mr',
       'Title_Mrs', 'Title_Special']]
y = train['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1)
print(X_train.shape, X_test.shape)

# In[ ]:


# We will also create a base model to check the goodness of our model.
# First we see the actual number of survivors
print(y.value_counts())

# In[ ]:


# We will select the larger number and consider that everyone dies to create a baseline.
y_default = pd.Series([0] * train['Survived'].shape[0], name = 'Survived')
print(y_default.value_counts())

# In[ ]:


# Calculate the baseline
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score
print(confusion_matrix(y, y_default))
print()
print(accuracy_score(y, y_default))
# So if we assumed that everyone dies we would be correct 61% of the time.
# So this is the bare minimun level of accuracy our prediciton should aim to improve upon.

# **3.1 Choosing an extimator**
# There are a lot of algorithms for classification and just iterating over all classification models may not be feasible. We need to shortlist a few algorithms that we can then apply and then later tune further for better results. I am going to refer to the [cheat sheet](http://scikit-learn.org/stable/tutorial/machine_learning_map/index.html) provided in sklearn documentation.

# In[ ]:


# First attempt with LinearSVC
from sklearn.svm import LinearSVC
# Looking into the documentation points us to set dual=False for cases with n_samples > n_features.
classifier = LinearSVC(dual=False)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))
scores = cross_val_score(classifier, X_train, y_train, cv=10, scoring='accuracy')
print(scores.mean())

# In[ ]:


# Next up we will try KNN
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 2)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))
scores = cross_val_score(classifier, X_train, y_train, cv=10, scoring='accuracy')
print(scores.mean())

# In[ ]:


# KNN isn't useful for us so we now move to a few popular ensemble estimators
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
print("AdaBoostClassifier")
ada_boost_classifier = AdaBoostClassifier()
ada_boost_classifier.fit(X_train, y_train)
y_pred = ada_boost_classifier.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))
scores = cross_val_score(ada_boost_classifier, X_train, y_train, cv=10, scoring='accuracy')
print(scores.mean())
print("BaggingClassifier")
bagging_classifier = BaggingClassifier()
bagging_classifier.fit(X_train, y_train)
y_pred = bagging_classifier.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))
scores = cross_val_score(bagging_classifier, X_train, y_train, cv=10, scoring='accuracy')
print(scores.mean())
print("ExtraTreesClassifier")
extra_trees_classifier = ExtraTreesClassifier(n_estimators=100)
extra_trees_classifier.fit(X_train, y_train)
y_pred = extra_trees_classifier.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))
scores = cross_val_score(extra_trees_classifier, X_train, y_train, cv=10, scoring='accuracy')
print(scores.mean())
print("GradientBoostingClassifier")
gradient_boosting_classifier = GradientBoostingClassifier()
gradient_boosting_classifier.fit(X_train, y_train)
y_pred = gradient_boosting_classifier.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))
scores = cross_val_score(gradient_boosting_classifier, X_train, y_train, cv=10, scoring='accuracy')
print(scores.mean())
print("RandomForestClassifier")
random_forest_classifier = RandomForestClassifier(n_estimators=100)
random_forest_classifier.fit(X_train, y_train)
y_pred = random_forest_classifier.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))
scores = cross_val_score(random_forest_classifier, X_train, y_train, cv=10, scoring='accuracy')
print(scores.mean())

# In[ ]:


# Will also try XGB based on its popularity and relevance here.
from xgboost import XGBClassifier
xgboost_classifier = XGBClassifier()
xgboost_classifier.fit(X_train, y_train)
y_pred = xgboost_classifier.predict(X_test)
# Print the confusion matrix
# http://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
print(confusion_matrix(y_test, y_pred))
# Print the accuracy score
# http://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html
print(accuracy_score(y_test, y_pred))
scores = cross_val_score(xgboost_classifier, X_train, y_train, cv=10, scoring='accuracy')
print(scores.mean())

# In[ ]:


# X = train.iloc[:, 1:]
# y = train.iloc[:, 0]
# print(X.columns.values)
# xgboost_classifier = XGBClassifier()
# from sklearn.feature_selection import RFECV
# rfecv = RFECV(estimator=xgboost_classifier, cv=10, scoring='accuracy')
# rfecv = rfecv.fit(X, y)
# print(X.columns[rfecv.support_])

# In[ ]:


# Now we will pass the validation set provided for creating our submission
# Pick the same columns as in X_test
X_validation = validation[['Age', 'Fare', 'Pclass_2', 'Pclass_3', 'Sex_female', 'Embarked_C',
       'Embarked_S', 'HasCabin', 'FamilySize', 'Title_Master', 'Title_Mr',
       'Title_Mrs', 'Title_Special']]
# Call the predict from the created classifier
y_valid = xgboost_classifier.predict(X_validation)

# **Step 4: Submission**

# In[ ]:


print(validation.columns.values)

# In[ ]:


# Creating final output file
validation_pId = validation.loc[:, 'PassengerId']
my_submission = pd.DataFrame(data={'PassengerId':validation_pId, 'Survived':y_valid})
print(my_submission['Survived'].value_counts())

# In[ ]:


my_submission.to_csv('submission.csv', index = False)

# We can now commit and open the saved kernel and then click on the "Output" tab to submit the generated output to get our accuracy score.

# Current Acurracy = 0.79425
# 
# Major tasks pending-
# 1. Use Ticket.
# 2. Try other models and model tuning.
# 3. Try using the entire train data for training. This should theoretically improve the accuracy further as more data will be used to train the final model.
#     Using a smaller split for train_test_split seems to confirm this. Changing split size from 0.2 to 0.1 increases the accuracy slightly.
#     
# Failed approaches - Ideas that I thought were good but actually either didn't affect or reduced the accuracy.
# 1. Use Titles to assign missing Age by using medians. For e.g. calculate Age.median() of all passengers that have the title 'Mr' and then use this median to fill null values for Age with Mr in Title.
# This reduced accuracy to little over 77% from 78.47%. Using mean instead of median on the other hand increased accuracy to 79.42%.
# Although median is more "robust"(https://creativemaths.net/blog/median/), mean here was better probably because Age occurs here as a Gaussian distribution(https://www.quora.com/What-is-more-accurate-the-median-or-mean).
# Easy explanation - https://statistics.laerd.com/statistical-guides/measures-central-tendency-mean-mode-median.php
