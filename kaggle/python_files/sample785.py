#!/usr/bin/env python
# coding: utf-8

# ## Our goal
# 
# In this competition we are asked to predict if a customer will make a transaction or not regardless of the amount of money transacted. Hence our goal is to solve a binary classification problem. In the data description you can see that the features given are numeric and anonymized. Furthermore the data seems to be artificial as they state that "the data has the same structure as our real data". 
# 
# ### Table of contents
# 
# 1. [Loading packages](#load) (complete)
# 2. [Sneak a peek at the data](#data) (complete)
# 2. [What can we say about the target?](#target) (complete)
# 3. [Can we find relationships between features?](#correlation) (complete)
# 4. [Baseline submission](#baselines) (complete)
# 5. [Basic feature engineering](#engineering) (complete)
# 6. [Gaussian Mixture Clustering](#clustering) (complete)

# ## Kernel settings

# In[ ]:


fit_gaussians = False
use_plotly=True

# ## Loading packages <a class="anchor" id="load"></a>

# In[ ]:


# data manipulation
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go

# sklearn models & tools
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import roc_auc_score
from sklearn.metrics import make_scorer
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA

# ignore warnings
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

import os
print(os.listdir("../input"))

# In[ ]:


train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
submission = pd.read_csv("../input/sample_submission.csv")

# ## Sneak a peek at the data <a class="anchor" id="data"></a>

# ### Train

# In[ ]:


train.shape

# Ok, 200.000 rows and 202 features. 

# In[ ]:


train.head(10)

# In[ ]:


train.target.dtype

# In[ ]:


org_vars = train.drop(["target", "ID_code"], axis=1).columns.values
len(org_vars)

# In[ ]:


train["Id"] = train.index.values
original_trainid = train.ID_code.values

train.drop("ID_code", axis=1, inplace=True)

# The target as well as the ID-Code of a sample are 2 special variables. Consequently 200 features are left. Browsing through the columns we can see that they look really numeric. It seems that there are no counter or integer variables. In addition it looks like if there are no missing values. Let's check it out:

# In[ ]:


train.isnull().sum().sum()

# ### Test

# In[ ]:


test.head(10)

# In[ ]:


test.isnull().sum().sum()

# At a first glance this looks similar to train except from the missing target.

# In[ ]:


test.shape

# In[ ]:


test["Id"] = test.index.values
original_testid = test.ID_code.values

test.drop("ID_code", axis=1, inplace=True)

# ### Submission
# 
# Before we start, let's look at the sample submission as well:

# In[ ]:


submission.head()

# Ok, not much to say about it.

# ## What can we say about the target? <a class="anchor" id="target"></a>

# In[ ]:


fig, ax = plt.subplots(1,2,figsize=(20,5))
sns.countplot(train.target.values, ax=ax[0], palette="husl")
sns.violinplot(x=train.target.values, y=train.index.values, ax=ax[1], palette="husl")
sns.stripplot(x=train.target.values, y=train.index.values,
              jitter=True, ax=ax[1], color="black", size=0.5, alpha=0.5)
ax[1].set_xlabel("Target")
ax[1].set_ylabel("Index");
ax[0].set_xlabel("Target")
ax[0].set_ylabel("Counts");

# In[ ]:


train.loc[train.target==1].shape[0] / train.loc[train.target==0].shape[0]

# ### Take Away
# 
# * We have to solve an imbalanced class problem. The number of customers that will not make a transaction is much higher than those that will. 
# * It seem that there is no relationship of the target with the index of the train dataframe. This is more empressend by the zero targets than for the ones. 
# * Take a look at the jitter plots within the violinplots. We can see that the targets look uniformly distributed over the indexes. It seems that the competitors were careful during the process of ordering the data. Once more this indicates that the data is simulated.

# ## Can we find relationships between features? <a class="anchor" id="correlation"></a>

# ### Linear correlations
# 
# I have already seen some correlation heatmaps in public kernels and it seems as if there is almost no correlation between features. Let's check this out by computing all correlation values and plotting the overall distribution:

# In[ ]:


train_correlations = train.drop(["target"], axis=1).corr()
train_correlations = train_correlations.values.flatten()
train_correlations = train_correlations[train_correlations != 1]

test_correlations = test.corr()
test_correlations = test_correlations.values.flatten()
test_correlations = test_correlations[test_correlations != 1]

plt.figure(figsize=(20,5))
sns.distplot(train_correlations, color="Red", label="train")
sns.distplot(test_correlations, color="Green", label="test")
plt.xlabel("Correlation values found in train (except 1)")
plt.ylabel("Density")
plt.title("Are there correlations between features?"); 
plt.legend();

# Woooow! :-O All features seem to have no linear correlation!!! Neither in train nor in test. Very strange. We know that they are anonymized and perhaps they are decorrelated by some transformation as well. 

# ### Random Forest Top Features
# 
# To start easy, let's use a random forest to select top 10 features. They can serve as a starting point to discover their nature and for trying to understand the data. In addition they may yield some ideas on how to generate new features. I am going to use stratified KFold as a cross validation strategy. It's somehow arbitrary to use KFold as we don't know if we have time series data given, but it may serve as a good starting point. 
# 
# To start simple I like to use a random forest that helps us to select important features. As there are no linear correlations it's a good idea to start with a nonlinear model that allows us to discover features, their importances as well as interactions. Let's start! :-)

# In[ ]:


parameters = {'min_samples_leaf': [20, 25]}
forest = RandomForestClassifier(max_depth=15, n_estimators=15)
grid = GridSearchCV(forest, parameters, cv=3, n_jobs=-1, verbose=2, scoring=make_scorer(roc_auc_score))

# In[ ]:


grid.fit(train.drop("target", axis=1).values, train.target.values)

# In[ ]:


grid.best_score_

# You can see that the score is not as good as some other scores of public kernels but nontheless my attempt is to understand the data by improving this score. We can use more powerful models later on.

# In[ ]:


grid.best_params_

# Let's take a look at n_top features of your choice:

# In[ ]:


n_top = 5 

# In[ ]:


importances = grid.best_estimator_.feature_importances_
idx = np.argsort(importances)[::-1][0:n_top]
feature_names = train.drop("target", axis=1).columns.values

plt.figure(figsize=(20,5))
sns.barplot(x=feature_names[idx], y=importances[idx]);
plt.title("What are the top important features to start with?");

# Ok, that's enough to start with the "data-understanding-journey".

# ### Exploring top features
# 
# First of all: How do the distributions of the variables look like with respect to the targets in train? Can we observe discrepancies between train and test features for selected top features?

# In[ ]:


fig, ax = plt.subplots(n_top,2,figsize=(20,5*n_top))

for n in range(n_top):
    sns.distplot(train.loc[train.target==0, feature_names[idx][n]], ax=ax[n,0], color="Orange", norm_hist=True)
    sns.distplot(train.loc[train.target==1, feature_names[idx][n]], ax=ax[n,0], color="Red", norm_hist=True)
    sns.distplot(test.loc[:, feature_names[idx][n]], ax=ax[n,1], color="Mediumseagreen", norm_hist=True)
    ax[n,0].set_title("Train {}".format(feature_names[idx][n]))
    ax[n,1].set_title("Test {}".format(feature_names[idx][n]))
    ax[n,0].set_xlabel("")
    ax[n,1].set_xlabel("")

# ### Take Away
# 
# * Interestingly there are some peeks inside the distributions, especially for variables 81, 12 and 53. Why do these data points accumulate on these values?
# * We can observe that the accumulations are less dense in the test data. 
# * Variable 174 seem to miss the bulb on the right hand side of the distribution in the test data. 

# In[ ]:


top = train.loc[:, feature_names[idx]]
top.describe()

# ### How do the scatter plots look like?

# In[ ]:


top = top.join(train.target)
sns.pairplot(top, hue="target")

# Crazy! Can you see the sharp limits of several variables where the samples with target 1 suddenly accumulate and seldomly pass over. Look at var 81 and 12 for example. You can see that there are limits close to 10 (var 81) and 13.5 (var 12). This finding could be a nice entry point for further feature engineering.

# ## Baseline submissions 
# 
# ### What score does the forest yield on public LB?

# In[ ]:


y_proba = grid.predict_proba(test.values)
y_proba_train = grid.predict_proba(train.drop("target", axis=1).values)

# In[ ]:


fig, ax = plt.subplots(2,1,figsize=(20,8))
sns.distplot(y_proba_train[train.target==1,1], norm_hist=True, color="mediumseagreen",
             ax=ax[0], label="1")
sns.distplot(y_proba_train[train.target==0,1], norm_hist=True, color="coral",
             ax=ax[0], label="0")
sns.distplot(y_proba[:,1], norm_hist=True,
             ax=ax[1], color="purple")
ax[1].set_xlabel("Predicted probability for test data");
ax[1].set_ylabel("Density");
ax[0].set_xlabel("Predicted probability for train data");
ax[0].set_ylabel("Density");
ax[0].legend();

# In[ ]:


submission["target"] = y_proba

# In[ ]:


submission.to_csv("submission_baseline_forest.csv", index=False)

# Yields 0.662 on public LB.

# ## Feature engineering 
# 
# Let's do some basic feature engineering. Perhaps it helps to improve:

# In[ ]:


original_features = train.drop(["target", "Id"], axis=1).columns.values
original_features

# ### Rounding & quantile based binning

# In[ ]:


encoder = LabelEncoder()
for your_feature in top.drop("target", axis=1).columns.values:
    train[your_feature + "_qbinned"] = pd.qcut(
        train.loc[:, your_feature].values,
        q=10,
        labels=False
    )
    train[your_feature + "_qbinned"] = encoder.fit_transform(
        train[your_feature + "_qbinned"].values.reshape(-1, 1)
    )
    
    
    train[your_feature + "_rounded"] = np.round(train.loc[:, your_feature].values)
    train[your_feature + "_rounded_10"] = np.round(10*train.loc[:, your_feature].values)
    train[your_feature + "_rounded_100"] = np.round(100*train.loc[:, your_feature].values)

# ### New feature importances

# In[ ]:


cv = StratifiedKFold(n_splits=3, random_state=0)
forest = RandomForestClassifier(max_depth=15, n_estimators=15, min_samples_leaf=20,
                                n_jobs=-1)

scores = []
X = train.drop("target", axis=1).values
y = train.target.values

for train_idx, test_idx in cv.split(X, y):
    x_train = X[train_idx]
    x_test = X[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]
    
    forest.fit(x_train, y_train)
    y_proba = forest.predict_proba(x_test)
    y_pred = np.zeros(y_proba.shape[0])
    y_pred[y_proba[:,1] >= 0.166] = 1
    
    score = roc_auc_score(y_test, y_pred)
    print(score)
    scores.append(score)

print(np.round(np.mean(scores),4))
print(np.round(np.std(scores), 4))

# In[ ]:


importances = forest.feature_importances_
feature_names = train.drop("target", axis=1).columns.values
idx = np.argsort(importances)[::-1][0:30]

plt.figure(figsize=(20,5))
sns.barplot(x=feature_names[idx], y=importances[idx]);
plt.xticks(rotation=90);

# ## Gaussian Mixture Clustering  <a class="anchor" id="clustering"></a>
# 
# The majority of the data looks like a big gaussian distribution. Besides that there seems to be at least one or two more gaussians that could explain the second and third mode that we can find for important features. Let's motivate this even further by looking at scatter and kde-plots of some top-features:

# In[ ]:


col1 = "var_81"
col2 = "var_12"
N=70000

# In[ ]:


fig, ax = plt.subplots(1,1, figsize=(20,10))
sns.kdeplot(train[col1].values[0:N], train[col2].values[0:N])
ax.scatter(train[col1].values[0:N], train[col2].values[0:N],
           s=2, c=train.target.values[0:N], cmap="coolwarm", alpha=0.5)
ax.set_xlabel(col1)
ax.set_xlabel(col2);

# * At least one big gaussians with one or two small, very thin but long gaussians.
# * It's very interesting that we can still find outliers beside sharp lines. 
# 
# Let's assume now that the data was generated using a mixture of gaussians and let's try to cluster them. Perhaps we can see that some clusters occupy more hot targets than others.

# In[ ]:


combined = train.drop(["target", "Id"], axis=1).append(test.drop("Id", axis=1))
combined.shape

# In[ ]:


max_components = 10
start_components = 3
n_splits = 3
K = train.shape[0]

X = train.loc[:, original_features].values[0:K]
y = train.target.values[0:K]

# In[ ]:


seeds = np.random.RandomState(0).randint(0,100, size=(max_components-start_components))
seeds

# In[ ]:


scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

# In[ ]:


if fit_gaussians:
    components = np.arange(start_components, max_components, 1)
    kf = StratifiedKFold(random_state=0, n_splits=n_splits)
    
    scores = np.zeros(shape=(max_components-start_components, n_splits))

    for m in components:
        split=0
        print("Components " + str(m))
        for train_index, test_index in kf.split(X_scaled, y):
            print("Split " + str(split))
            x_train, x_test = X_scaled[train_index], X_scaled[test_index]
            gm = GaussianMixture(n_components=m, random_state=seeds[m-start_components])
            gm.fit(x_train)
            score = gm.score(x_test)
            scores[m-start_components,split] = score
            split +=1
    
    print(np.round(np.mean(scores, axis=1), 2))
    print(np.round(np.std(scores, axis=1), 2))
    best_idx = np.argmax(np.mean(scores, axis=1))
    best_component = components[best_idx]
    best_seed = seeds[best_idx]
    print("Best component found " + str(best_component))
    
else:
    best_seed = seeds[0]
    best_component = 3

# In[ ]:


X = train.loc[:, original_features].values

gm = GaussianMixture(n_components=best_component, random_state=best_seed)
X_scaled = scaler.transform(X)
gm.fit(X_scaled)

# In[ ]:


train["cluster"] = gm.predict(X_scaled)
train["logL"] = gm.score_samples(X_scaled)
test["cluster"] = gm.predict(test.loc[:, original_features].values)
test["logL"] = gm.score_samples(test.loc[:, original_features].values)

# In[ ]:


fig, ax = plt.subplots(1,2,figsize=(20,5))
sns.countplot(train.cluster, palette="Set2", ax=ax[0])
sns.distplot(train.logL, color="Dodgerblue", ax=ax[1]);

# ### Take-Away
# 
# * By fitting the gaussian mixture model we are maximizing the log likelihood. The higher, the better the gaussians suite to our data. As it's difficult to choose the right number of components (gaussians) I decided to use a stratified k fold of the train data. This way we can fit gaussians to a train subset, and test how big the log likelihood is on the test subset. By doing so three times for each selected component, we gain some more information about the stability of our solution. **We can see that 3 gaussians seem to be sufficient as the log likelihood values decrease with more components**. 
# * This need not be true as the **solution depends on the initialization of the gaussians (the seeds I used) and with more data, the result may be different**. 
# * But we can say: There are **at least 3 gaussians**. This is what we have already found by visual exploration of the data. 
# * The individual score per data spot can be understood as a measure of density. If it's low, the data spot lives in a region with other data points far away. If it's high, it should have a lot of neighbors. Consequently the individual logL-score can tell us something about outliers in the data.

# In[ ]:


cluster_occupation = train.groupby("cluster").target.value_counts() / train.groupby("cluster").size() * 100
cluster_occupation = cluster_occupation.loc[:, 1]

target_occupation = train.groupby("target").cluster.value_counts() / train.groupby("target").size() * 100
target_occupation = target_occupation.loc[1, :]
target_occupation.index = target_occupation.index.droplevel("target")

fig, ax = plt.subplots(1,2,figsize=(20,5))
ax[0].set_title("How many % of the data per cluster has hot targets?")
sns.barplot(cluster_occupation.index, cluster_occupation.values, ax=ax[0], color="cornflowerblue")
ax[0].set_ylabel("% of cluster data")
ax[0].set_ylim([0,100])

ax[1].set_title("How many % of total hot targets are in one cluster?")
sns.barplot(target_occupation.index, target_occupation.values, ax=ax[1], color="tomato")
ax[1].set_ylabel("% of hot targets")
ax[1].set_ylim([0,100]);

# * As we have much more cold-targets (zero) that hot (ones), I'm not surprised that hot targets occupy only a small part of the data per cluster. Nonetheless we can see that cluster 1 has significantly more hot targets than the others.
# * The second plot shows that most hot targets are located in cluster 1 followed by cluster 2. This confirms our assumption that the big gaussian in the middle (cluster 0) has the smallest amount of hot targets and that the small, thin side distributions are more likely to have hot targets. 

# In[ ]:


plt.figure(figsize=(20,5))
for n in range(gm.means_.shape[0]):
    plt.plot(gm.means_[n,:], 'o')
plt.title("How do the gaussian means look like?")
plt.ylabel("Cluster mean value")
plt.xlabel("Feature")

# Only some features are important to separate the structure of the data. 

# Good luck for the last days! :-)
