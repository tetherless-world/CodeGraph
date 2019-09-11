#!/usr/bin/env python
# coding: utf-8

# # Import the dependencies

# In[1]:


# Data manipulation
import numpy as np
import pandas as pd

# Data visualization
import seaborn as sb
import matplotlib.pyplot as plt

# Regex
import re as re

# Model Selection and Evaluation
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import cross_val_score

# Performance
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

# Machine Learning Algorithms
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression


# Preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Base classes
from sklearn.base import BaseEstimator, TransformerMixin


# # Get the data

# In[2]:


titanic_trainSet = pd.read_csv('../input/train.csv')
titanic_testSet = pd.read_csv('../input/test.csv')
full_data = [titanic_trainSet, titanic_testSet]

# 
# # Explore and visualize the data to gain insights

# ### There are 12 features available in the training set.
# ### 'Survived' feature is our target variable, where 0 means the passenger did not survive, while 1 means he/she survived.

# In[3]:


titanic_trainSet.head()

# ### Test set also has the same set of features but with no output label.
# ### To know how well our model has performed, we need to submit our model predictions to the Titanic Kaggle Competition and wait for the result. But we can use cross-validation to have an idea of how good our model is.

# In[4]:


titanic_testSet.head()

# In[5]:


titanic_trainSet.info()
print('-'*40)
titanic_testSet.info()

# In[6]:


titanic_trainSet.describe()

# # Analysing the features.
# ### There are 3 categorical input features in our dataset(Pclass, Sex, Embarked) and 1 categorical label (Survived), hence classification task. 
# ### Lets analyze them first.

# ## Survived

# In[7]:


print(titanic_trainSet["Survived"].value_counts(sort=False))
print('-'*50)
plt.figure(figsize=(10, 6))
sb.set(style="whitegrid")
sb.countplot( x= 'Survived', hue='Sex', data=titanic_trainSet)
plt.title('Survival Distribution')
plt.show()

# ## Passenger Class

# In[8]:


print("Passengers in each class\n")
titanic_trainSet["Pclass"].value_counts(sort=False)

# In[9]:


print(titanic_trainSet[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean())
plt.figure(figsize=(10, 6))
sb.countplot( x= 'Pclass', hue='Survived',data=titanic_trainSet)
plt.title('Survival rate of each class')
plt.show()

# ## Gender

# In[10]:


print('Gender distribution\n')
print(titanic_trainSet["Sex"].value_counts(sort=False))

# In[11]:


print(titanic_trainSet[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean())
plt.figure(figsize=(10, 6))
sb.countplot( x= 'Sex', hue='Survived',data=titanic_trainSet)
plt.title('Survival rate of each Gender')
plt.show()

# ## Embarked
# ### This attribute tells us where the passenger board the Titanic: C=Cherbourg, Q=Queenstown, S=Southampton.

# In[12]:


print('Embarked distribution\n')
print(titanic_trainSet["Embarked"].value_counts(sort=False))

# In[13]:


print(titanic_trainSet[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean())
plt.figure(figsize=(10, 6))
sb.countplot( x= 'Embarked', hue='Survived', data=titanic_trainSet)
plt.title('Survival rate based on boarding point')
plt.show()

# In[ ]:




# # Feature Engineering

# ## Relatives Onboard
# #### 'SibSp': tells how many siblings & spouses of the passenger aboard the Titanic.
# #### 'Parch': tells how many children & parents of the passenger aboard the Titanic.
# #### In whole, these two features are telling about the no. of relatives of a passenger aboard the ship. So we can add these two feature and make one i.e. 'RelativesOnboard'

# In[14]:


for dataset in full_data:
    dataset['RelativesOnboard'] = dataset['SibSp'] + dataset['Parch']

# In[15]:


print('Relatives distribution\n')
print(titanic_trainSet["RelativesOnboard"].value_counts(sort=False))

# In[16]:


print(titanic_trainSet[['RelativesOnboard', 'Survived']].groupby(['RelativesOnboard'], as_index=False).mean())
plt.figure(figsize=(10, 6))
sb.countplot( x= 'RelativesOnboard', hue='Survived', data=titanic_trainSet)
plt.title('Survival rate based on no. of relatives')
plt.show()

# ## Age
# #### 'Age' is a continuous attribute. But we can convert it into a categorical feature based on different age groups. And then we can find out the survival rate of each age group.

# In[17]:


for dataset in full_data:
    dataset['AgeGroup'] = dataset['Age'] // 15 * 15

# In[18]:


titanic_trainSet.isna().sum()

# In[19]:


titanic_copy = titanic_trainSet.copy()

# In[20]:


median = titanic_copy['AgeGroup'].median()
titanic_copy['AgeGroup'] = titanic_copy['AgeGroup'].fillna(median)

# In[21]:


print('Age distribution\n')
print(titanic_copy["AgeGroup"].value_counts(sort=False))

# In[22]:


print(titanic_copy[['AgeGroup', 'Survived']].groupby(['AgeGroup'], as_index=False).mean())
plt.figure(figsize=(10, 6))
sb.countplot( x= 'AgeGroup', hue='Survived', data=titanic_copy)
plt.title('Survival rate based on Age')
plt.show()

# ## Name
# #### We can use name attribute to parse out the 'Title' for each person and then use it as a categorical features.

# In[23]:


titanic_trainSet['Name'].head()

# In[25]:


def parseTitle(name):
	title_search = re.search(' ([A-Za-z]+)\.', name)
  
	# If the title exists, extract and return it.
	if title_search:
		return title_search.group(1)
	return ""

for dataset in full_data:
  dataset['Title'] = dataset['Name'].apply(parseTitle)


# In[26]:


print('Title distribution\n')
print(titanic_trainSet["Title"].value_counts(sort=False))

# In[27]:


for dataset in full_data:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')


# In[28]:


print('Title distribution\n')
print(titanic_trainSet["Title"].value_counts(sort=True))

# In[29]:


print(titanic_trainSet[['Title', 'Survived']].groupby(['Title'], as_index=False).mean())
plt.figure(figsize=(10, 6))
sb.countplot( x= 'Title', hue='Survived', data=titanic_trainSet)
plt.title('Survival rate based on Title')
plt.show()

# # Feature Selection
# ### So, after analysing and processing the features, I have selected the final features for ML algorithms.

# In[30]:


features_testSet = ['Pclass', 'Sex', 'Embarked', 'Fare', 'Title', 'AgeGroup', 'RelativesOnboard']
features_trainSet = features_testSet + ['Survived']

train_set = titanic_trainSet[[*features_trainSet]]
test_set = titanic_testSet[[*features_testSet]]

train_set.head()
  

# In[ ]:


test_set.head()

# # Data Preprocessing
# ## Dealing with missing values and categorical values

# In[31]:


class CategoricalImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.most_frequent_ = pd.Series([X[c].value_counts().index[0] for c in X],
                                        index=X.columns)
        return self
    def transform(self, X, y=None):
        return X.fillna(self.most_frequent_)

# In[32]:


num_pipeline = Pipeline([('imputer', SimpleImputer(strategy="median"))])
cat_pipeline = Pipeline([("imputer", CategoricalImputer()), ("cat_encoder", OneHotEncoder(sparse=False))])

# In[33]:


num_attribs = ['Fare']
cat_attribs = ['Pclass', 'Sex', 'Embarked', 'Title', 'AgeGroup', 'RelativesOnboard']

full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", cat_pipeline, cat_attribs),
    ])


# ## Separate the Features and Label

# In[34]:


X_train = full_pipeline.fit_transform(train_set)
y_train = train_set["Survived"]

# # Select and train a model
# ## `Before directly training on the test set, first train and evaluate on the training set. Also, try different ML algorithms and choose which fits best on the data.`

# In[35]:


classifiers = [
    KNeighborsClassifier(3),
    SVC(probability=True, gamma="auto"),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
	  AdaBoostClassifier(),
    GradientBoostingClassifier(),
    GaussianNB(),
    LogisticRegression()]

# In[ ]:


classifiers_Kscores = []
classifiers_accuracy = []
for clf in classifiers:
  clf_scores = cross_val_score(clf, X_train, y_train, cv=10)
  classifiers_Kscores.append(clf_scores)
  model_name = type(clf).__name__
  classifiers_accuracy.append(model_name+': '+str(format(clf_scores.mean()*100,'.2f')))

# In[40]:


plt.figure(figsize=(10, 6))
plt.boxplot(classifiers_Kscores, labels=("KNN","SVC","Trees","Forest","Ada","Gradient","NB","Logistic"))
plt.ylabel("Accuracy", fontsize=14)
plt.show()

print("\n\nClassifiers Accuracy:")
classifiers_accuracy


# ## Accuracy Score is generally not the preferred performance measure for classifiers, especially when you are dealing with skewed datasets (i.e., when some classes are much more frequent than others).
# ## We will also calculate the F1_score for each classifier, which is the weighted average of the precision and recall.
# ## F1 = 2 x (precision x recall) / (precision + recall)

# In[41]:


splits = StratifiedShuffleSplit(n_splits=1, test_size=0.4, random_state=42)

for train_index, test_index in splits.split(X_train, y_train):
  X_train1, X_test1 = X_train[train_index], X_train[test_index]
  y_train1, y_test1 = y_train[train_index], y_train[test_index]

# In[ ]:



f1_scores = []
for clf in classifiers:
  clf.fit(X_train1, y_train1)
  pred = clf.predict(X_test1)
  model_name = type(clf).__name__
  f1_scores.append((model_name+': '+str(format(f1_score(y_test1, pred)*100,'.2f'))))

# In[43]:


print("F1 Scores:")
f1_scores

# # Evaluate the model on the Test Set

# In[44]:


X_test = full_pipeline.fit_transform(test_set)

# In[45]:


ada_clf = AdaBoostClassifier()
ada_clf.fit(X_train,y_train)
y_pred = ada_clf.predict(X_test)

# In[47]:


passengerID =np.array(titanic_testSet["PassengerId"]).astype(int)
titanicSurvival_predictions = pd.DataFrame(y_pred, passengerID, columns = ["Survived"])

titanicSurvival_predictions.to_csv("Titanic_Survival_Predictions_ada.csv", index_label = ["PassengerId"])
