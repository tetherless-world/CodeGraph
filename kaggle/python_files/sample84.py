#!/usr/bin/env python
# coding: utf-8

# # Classifying multi-label comments with Logistic Regression
# #### Rhodium Beng
# Started on 20 December 2017
# 
# This kernel is inspired by:
# - kernel by Jeremy Howard : _NB-SVM strong linear baseline + EDA (0.052 lb)_
# - kernel by Issac : _logistic regression (0.055 lb)_
# - _Solving Multi-Label Classification problems_, https://www.analyticsvidhya.com/blog/2017/08/introduction-to-multi-label-classification/

# In[ ]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import re

# ## Load training and test data

# In[ ]:


train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')

# ## Examine the data (EDA)

# In[ ]:


train_df.sample(5)

# In the training data, the comments are labelled as one or more of the six categories; toxic, severe toxic, obscene, threat, insult and identity hate. This is essentially a multi-label classification problem.

# In[ ]:


cols_target = ['obscene','insult','toxic','severe_toxic','identity_hate','threat']

# In[ ]:


# check missing values in numeric columns
train_df.describe()

# There are no missing numeric values. Based on the mean values, it also looks like there are many comments which are not labelled in any of the six categories.

# In[ ]:


# check for any 'null' comment
no_comment = train_df[train_df['comment_text'].isnull()]
len(no_comment)

# In[ ]:


test_df.head()

# In[ ]:


no_comment = test_df[test_df['comment_text'].isnull()]
no_comment

# There is a row in the test data which does not contain any comment, so let's put 'unknown' in its place.

# In[ ]:


# fill NaN with string "unknown"
test_df.fillna('unknown',inplace=True)

# In[ ]:


# let's see the total rows in train, test data and the numbers for the various categories
print('Total rows in train is {}'.format(len(train_df)))
print('Total rows in test is {}'.format(len(test_df)))

# In[ ]:


print(train_df[cols_target].sum())

# Majority of the comments are not labelled in one or more of these categories.

# In[ ]:


# Let's look at the character length for the rows and record these
train_df['char_length'] = train_df['comment_text'].str.len()

# In[ ]:


# look at the histogram plot for text length
sns.set()
train_df['char_length'].hist()
plt.show()

# Most of the text length are within 500 characters, with some up to 5,000 characters long.

# Next, let's examine the correlations among the target variables.

# In[ ]:


data = train_df[cols_target]

# In[ ]:


colormap = plt.cm.magma
plt.figure(figsize=(7,7))
plt.title('Correlation of features & targets',y=1.05,size=14)
sns.heatmap(data.astype(float).corr(),linewidths=0.1,vmax=1.0,square=True,cmap=colormap,
           linecolor='white',annot=True)

# Indeed, it looks like the various labels are correlated, e.g. insult-obscene has the highest at 0.74, followed by toxic-obscene and toxic-insult.

# What about the character length & distribution of the comment text in the test data?

# In[ ]:


test_df['char_length'] = test_df['comment_text'].str.len()

# In[ ]:


plt.figure(figsize=(20,5))
plt.hist(test_df['char_length'])
plt.show()

# Looks like there are several very long comments in the test data. Let's see what they are.

# In[ ]:


test_df[test_df['char_length']>5000]

# Let's truncate char length in test_df to 5,000 characters and see if the distribution would be similar to train_df.

# In[ ]:


test_comment = test_df['comment_text'].apply(lambda x: x[:5000])
char_length = test_comment.str.len()

# In[ ]:


plt.figure()
plt.hist(char_length)
plt.show()

# Now, the shape of character length distribution looks similar to the train data. I guess the train data were clipped to 5,000 characters to facilitate the folks who did the labelling of comment categories.

# ## Clean up the comment text

# In[ ]:


def clean_text(text):
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"\'scuse", " excuse ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub('\W', ' ', text)
    text = re.sub('\s+', ' ', text)
    text = text.strip()
    return text

# In[ ]:


# clean the comment_text in train_df
cleaned_train_comment = []
for i in range(0,len(train_df)):
    cleaned_comment = clean_text(train_df['comment_text'][i])
    cleaned_train_comment.append(cleaned_comment)
train_df['comment_text'] = pd.Series(cleaned_train_comment).astype(str)

# In[ ]:


# clean the comment_text in test_df
cleaned_test_comment = []
for i in range(0,len(test_df)):
    cleaned_comment = clean_text(test_df['comment_text'][i])
    cleaned_test_comment.append(cleaned_comment)
test_df['comment_text'] = pd.Series(cleaned_test_comment).astype(str)

# ## Define X from entire train & test data for use in tokenization by Vectorizer

# In[ ]:


train_df = train_df.drop('char_length',axis=1)

# In[ ]:


X = train_df.comment_text
test_X = test_df.comment_text

# In[ ]:


print(X.shape, test_X.shape)

# ## Vectorize the data

# In[ ]:


# import and instantiate CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
vect = TfidfVectorizer(max_features=20000,min_df=2)
vect

# In[ ]:


# learn the vocabulary in the training data, then use it to create a document-term matrix
X_dtm = vect.fit_transform(X)
# examine the document-term matrix created from X_train
X_dtm

# In[ ]:


# transform the test data using the earlier fitted vocabulary, into a document-term matrix
test_X_dtm = vect.transform(test_X)
# examine the document-term matrix from X_test
test_X_dtm

# ## Solving a multi-label classification problem
# One way to approach a multi-label classification problem is to transform the problem into separate single-class classifier problems. This is known as 'problem transformation'. There are three methods:
# * _**Binary Relevance.**_ This is probably the simplest which treats each label as a separate single classification problems. The key assumption here though, is that there are no correlation among the various labels.
# * _**Classifier Chains.**_ In this method, the first classifier is trained on the input X. Then the subsequent classifiers are trained on the input X and all previous classifiers' predictions in the chain. This method attempts to draw the signals from the correlation among preceding target variables.
# * _**Label Powerset.**_ This method transforms the problem into a multi-class problem  where the multi-class labels are essentially all the unique label combinations. In our case here, where there are six labels, Label Powerset would in effect turn this into a six-factorial or 720-class problem!

# ## Binary Relevance - build a multi-label classifier using Logistic Regression

# In[ ]:


# import and instantiate the Logistic Regression model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
logreg = LogisticRegression(C=6.0,random_state=123)

# create submission file
submission_binary = pd.read_csv('../input/sample_submission.csv')

for label in cols_target:
    print('... Processing {}'.format(label))
    y = train_df[label]
    # train the model using X_dtm & y
    logreg.fit(X_dtm, y)
    # compute the training accuracy
    y_pred_X = logreg.predict(X_dtm)
    print('Training accuracy is {}'.format(accuracy_score(y, y_pred_X)))
    # compute the predicted probabilities for X_test_dtm
    test_y_prob = logreg.predict_proba(test_X_dtm)[:,1]
    submission_binary[label] = test_y_prob

# ### Create submission file

# In[ ]:


submission_binary.head()

# In[ ]:


# generate submission file
submission_binary.to_csv('submission_binary.csv',index=False)

# #### Binary Relevance with Logistic Regression classifier scored 0.062 on the public leaderboard.

# ## Classifier Chains - build a multi-label classifier using Logistic Regression

# In[ ]:


# create submission file
submission_chains = pd.read_csv('../input/sample_submission.csv')

# create a function to add features
def add_feature(X, feature_to_add):
    '''
    Returns sparse feature matrix with added feature.
    feature_to_add can also be a list of features.
    '''
    from scipy.sparse import csr_matrix, hstack
    return hstack([X, csr_matrix(feature_to_add).T], 'csr')

# In[ ]:


for label in cols_target:
    print('... Processing {}'.format(label))
    y = train_df[label]
    # train the model using X_dtm & y
    logreg.fit(X_dtm,y)
    # compute the training accuracy
    y_pred_X = logreg.predict(X_dtm)
    print('Training Accuracy is {}'.format(accuracy_score(y,y_pred_X)))
    # make predictions from test_X
    test_y = logreg.predict(test_X_dtm)
    test_y_prob = logreg.predict_proba(test_X_dtm)[:,1]
    submission_chains[label] = test_y_prob
    # chain current label to X_dtm
    X_dtm = add_feature(X_dtm, y)
    print('Shape of X_dtm is now {}'.format(X_dtm.shape))
    # chain current label predictions to test_X_dtm
    test_X_dtm = add_feature(test_X_dtm, test_y)
    print('Shape of test_X_dtm is now {}'.format(test_X_dtm.shape))

# ### Create submission file

# In[ ]:


submission_chains.head()

# In[ ]:


# generate submission file
submission_chains.to_csv('submission_chains.csv', index=False)

# ### That's all for now. Would like to work on the last problem transformation method Label Powerset next, but right now, I can't think of how I could generate the prediction probability numbers in the format required for submission.
# ### Tips and comments are most welcomed & appreciated.
# ### Please upvote if you find it useful. Happy Holidays!

# In[ ]:



