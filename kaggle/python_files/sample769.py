#!/usr/bin/env python
# coding: utf-8

# ## Table of contents
# 
# 1. Prepare to start
# 2. Sneak a peek
#     * Take a look
#     * Size of the data
#     * Dtypes
#     * A magic feature?
#     * Missing values
# 3. Basic Exploratory Analysis
#     * Feature correlations
#     * Distances between class medians
#     * What logistic regression wants to tell us
#     * The magic turtle again 
#     * What about the column names?

# ## Prepare to start

# In[4]:


import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()

import os
print(os.listdir("../input"))

from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go

# ## Sneak a peek

# ### Take a look

# In[5]:


train = pd.read_csv("../input/train.csv")
train.head()

# In[6]:


test = pd.read_csv("../input/test.csv")

# ### Insights
# 
# * Hey, we are given crazy column names! Can we find a meaning behind them? Hmm... ;-)
# * Very interesting id-column, isn't it?

# ### Size of the data

# In[7]:


train.shape[0] / test.shape[0]

# Roughly we have twice as much train data than test data.

# ### Dtypes

# In[ ]:


train.info()

# Hmm... two int columns?

# In[ ]:


test.info()

# ### A magic feature?

# In[ ]:


column_types = train.dtypes
column_types[column_types==np.int64]

# Ohhh! :-D Cool! The wheezy-copper-turtle-magic oooohhh magic! What does that mean for the comp?

# ### Missing values
# 
# Let's start with obvious once:

# In[ ]:


train.isnull().sum().sum()

# In[ ]:


test.isnull().sum().sum()

# No obvious missing values in train and test. Perhaps there are some non-obvious once but let's move this topic to exploratory data analysis.

# ## Basic exploratory analysis
# 
# Ok, now the colorful part starts. We have already found an interesting feature by peeking at the data. Perhaps we can find some more during basic EDA. Let's stay curious and critical! Do we know if test and train behave the same? No! For this reason, I don't like to combine train and test right now... :-)

# ### Class balance

# In[ ]:


sns.countplot(train.target, palette="Set2");

# Very balanced in train!

# ### Feature correlation

# In[ ]:


magic = "wheezy-copper-turtle-magic"
train_corr = train.drop(["target", magic], axis=1).corr()
test_corr = test.drop(magic, axis=1).corr()

# In[ ]:


train_corr_flat = train_corr.values.flatten()
train_corr_flat = train_corr_flat[train_corr_flat != 1]

test_corr_flat = test_corr.values.flatten()
test_corr_flat = test_corr_flat[test_corr_flat != 1]

fig, ax = plt.subplots(1,2,figsize=(20,5))
sns.distplot(train_corr_flat, ax=ax[0], color="tomato")
sns.distplot(test_corr_flat, ax=ax[1], color="limegreen");
ax[0].set_title("Off-diagonal train corr \n distribution")
ax[1].set_title("Off-diagonal test corr \n distribution");
ax[0].set_xlabel("feature correlation value")
ax[1].set_xlabel("feature correlation value");

# ### Insights
# 
# * Looks almost decorrelated. But don't be too fast. Maybe this slight values are still something fruitful. 
# * Do you see what is missing in train? We have +- 0.015 in test on both sides but only -0.01 and +0.015 in train. There must be some differences in feature distributions between train and test.

# In[ ]:


plt.figure(figsize=(25,25))
sns.heatmap(train_corr, vmin=-0.016, vmax=0.016, cmap="RdYlBu_r");

# In[ ]:


plt.figure(figsize=(25,25))
sns.heatmap(test_corr, vmin=-0.016, vmax=0.016, cmap="RdYlBu_r");

# ### Insights
# 
# * If you compare correlations carefully, you can observe that some of the nearby features have some slight correlations in train and test. This is especially interesting for some subsequent features. Can we find some sense in the way of column order?
# * This is not always true. Take a look at the lower right corner. The correlation between the last two features in train looks strong enough to "be" something, but in test this correlation has gone! As we have already seen by the correlation distributions we can expect some differences in the feature distributions of train and test. 

# In[ ]:


target_medians = train.groupby("target").median()
sorted_target_distance = np.abs(target_medians.iloc[0]-target_medians.iloc[1]).sort_values(ascending=False)

# In[ ]:


sorted_target_distance.head()

# In[ ]:


sorted_target_distance.tail()

# :-o What's that? Wheezy-copper-turtle-magic... again!

# In[ ]:


fig, ax = plt.subplots(2,2,figsize=(20,10))
sns.distplot(train.loc[train.target==0, "wheezy-myrtle-mandrill-entropy"], color="Blue", ax=ax[0,0])
sns.distplot(train.loc[train.target==1, "wheezy-myrtle-mandrill-entropy"], color="Red", ax=ax[0,0])
sns.distplot(train.loc[train.target==0, "wheezy-copper-turtle-magic"], color="Blue", ax=ax[0,1])
sns.distplot(train.loc[train.target==1, "wheezy-copper-turtle-magic"], color="Red", ax=ax[0,1])
ax[1,0].scatter(train["wheezy-myrtle-mandrill-entropy"].values,
                train["skanky-carmine-rabbit-contributor"].values, c=train.target.values,
                cmap="coolwarm", s=1, alpha=0.5)
ax[1,0].set_xlabel("wheezy-myrtle-mandrill-entropy")
ax[1,0].set_ylabel("skanky-carmine-rabbit-contributor")
ax[1,1].scatter(train["wheezy-myrtle-mandrill-entropy"].values,
                train["wheezy-copper-turtle-magic"].values, c=train.target.values,
                cmap="coolwarm", s=1, alpha=0.5)
ax[1,1].set_xlabel("wheezy-myrtle-mandrill-entropy")
ax[1,1].set_ylabel("wheezy-copper-turtle-magic");

# ### Insights
# 
# Crazy turtle! :-D
# 
# * By taking the feature mean per target and computing distances between the classes I wanted to find features that show clear separating forces. But... even the feature with highest difference in classes medians looks like "no big deal" at all. 
# * Our crazy magic turtle feature has no difference at all! And it's distribution is almost uniform even though the small bin peaks look interesting (just an artifact of plotting?). 
# * By plotting two features with highest distances in class medians I wanted to see if we can see something like in don't overfit (overlapping gaussians)... but this time it doesn't look like that. Let's see what logistic regression tells us with its weights in the next step.
# * The scatter plot of wheezy-copper-turtle-magic and our highest class median distance feature, "wheezy_myrtle-mandrill-entropy", looks indeed crazy! It's not that the turtle magic spreads smoothly with lower density over the whole space of the mandrills tails. No! It has some subsequent "peak" tails depending on the value of magic turtle. This again leads to the question: What role does the magic turtle play in this game?

# ### What logistic regression wants to tell us...

# In[ ]:


n_splits=5
n_repeats=3

X=train.drop(["target", "id"], axis=1).values
y=train.target.values
XTest = test.drop("id", axis=1).values

scaler = StandardScaler()
X = scaler.fit_transform(X)
XTest = scaler.transform(XTest)

# In[ ]:


X, x_val, y, y_val = train_test_split(X,y,test_size=0.2, stratify=y, random_state=2019)

# In[ ]:


skf = RepeatedStratifiedKFold(n_repeats=n_repeats,
                              n_splits=n_splits,
                              random_state=2019)

p_val = np.zeros(y_val.shape)
pTest = np.zeros(XTest.shape[0])
for train_idx, test_idx in skf.split(X,y):
    
    x_train, x_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    lr=LogisticRegression(penalty="l1", C=1, solver="saga")
    lr.fit(x_train, y_train)
    p_test = lr.predict_proba(x_test)[:,1]
    p_val += lr.predict_proba(x_val)[:,1]
    pTest += lr.predict_proba(XTest)[:,1]
    print(roc_auc_score(y_test, p_test))

p_val /= (n_splits*n_repeats)
pTest /= (n_splits*n_repeats)


# In[ ]:


print(roc_auc_score(y_val, p_val))

# As Chris already pointed out, we won't get far by ignoring the pattern found with our magic turtle. 

# ### The magic turtle again
# 
# Can't get enough! :-) Let's look at a 3D-Scatterplot with magic turtle:

# In[1]:


feat1 = "wheezy-myrtle-mandrill-entropy"
feat2 = "skanky-carmine-rabbit-contributor"
feat3 = "wheezy-copper-turtle-magic"

# In[8]:


N = 10000

trace1 = go.Scatter3d(
    x=train[feat1].values[0:N], 
    y=train[feat2].values[0:N],
    z=train[feat3].values[0:N],
    mode='markers',
    marker=dict(
        color=train.target.values[0:N],
        colorscale = "Jet",
        opacity=0.3,
        size=2
    )
)

figure_data = [trace1]
layout = go.Layout(
    title = 'The turtle place',
    scene = dict(
        xaxis = dict(title=feat1),
        yaxis = dict(title=feat2),
        zaxis = dict(title=feat3),
    ),
    margin=dict(
        l=0,
        r=0,
        b=0,
        t=0
    ),
    showlegend=True
)

fig = go.Figure(data=figure_data, layout=layout)
py.iplot(fig, filename='simple-3d-scatter')

# ### What about the column names?
# 
# Have you asked yourself as well who had this nice phantasy to build up the column names? Is there a logic behind it? Can we find connections given the column names? Well I don't know it and we all have to work on it but perhaps a sketch helps...

# In[16]:


names = list(train.drop(["id", "target"], axis=1).columns.values)
names

# In[18]:


first_names = []
second_names = []
third_names = []
fourth_names = []

for name in names:
    words = name.split("-")
    first_names.append(words[0])
    second_names.append(words[1])
    third_names.append(words[2])
    fourth_names.append(words[3])

# In[25]:


print(len(first_names), len(np.unique(first_names)))
print(len(second_names), len(np.unique(second_names)))
print(len(third_names), len(np.unique(third_names)))
print(len(fourth_names), len(np.unique(fourth_names)))

# In[ ]:



