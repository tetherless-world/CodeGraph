#!/usr/bin/env python
# coding: utf-8

# # PUBG Data Exploration + Random Forest (+ Funny GIFs)

# [](http://)Hi fellow Kagglers! 
# 
# In this Kernel we ([Dalton Harmsen](https://www.kaggle.com/daltonharmsen), [Lourens Touwen](https://www.kaggle.com/lourenst) and [Carlo Lepelaars](https://www.kaggle.com/carlolepelaars)) will show you how we explore the [PUBG dataset](https://www.kaggle.com/c/pubg-finish-placement-prediction/data), detect outliers and recognize important features. We also implement a random forest model and optimize it.
# 
# If you like this Kaggle kernel, feel free to give an upvote and leave a comment.
# 
# A lot of inspiration for this kernel came from [fast.ai](https://www.fast.ai/)'s "[Machine Learning for Coders](https://course.fast.ai/ml)" course.
# 
# ![alt text](https://o.aolcdn.com/images/dims?quality=100&image_uri=http%3A%2F%2Fo.aolcdn.com%2Fhss%2Fstorage%2Fmidas%2Fb0be09f425cc5175fb413bc03c32dd0d%2F206235889%2Fpubg-ed.jpg&client=amp-blogside-v2&signature=88c6b77342cbeb0d25c0dc9d909018136aec1971 "Logo Title Text 1")

# # Table of Contents

# * [Preparation](#1)
# * [Extra Data (Coming Soon)](#2)
# * [Initial Exploration](#3)
# * [Illegal Match](#4)
# * [Feature Engineering](#5)
# * [Outlier Detection](#6)
# * [Categorical Variables](#7)
# * [Preparation for Machine Learning](#8)
# * [Feature Importance](#9)
# * [Final Random Forest Model](#10)
# * [Kaggle Submission](#11)
# 

# # Let's Go!

# ![Alt Text](https://media.giphy.com/media/xT9IgnOQS8e8uKkflK/giphy.gif)

# # Preparation <a id="1"></a>

# First we import the dependencies needed for handling data, visualization and training our model. 
# 
# Important dependencies are:
# * [Pandas](https://pandas.pydata.org) for their dataframe structures and easy visualization.
# * [Matplotlib](https://matplotlib.org) for visualization.
# * [Scikit-learn](https://scikit-learn.org/stable) for machine learning.
# * [fastai](https://www.fast.ai) for machine learning and feature importance.

# In[ ]:


## Something went wrong when importing fastai.structured.
## We fixed this by put the whole source code of fastai.structured in the notebook.
## This was copied from: https://github.com/anandsaha/fastai.part1.v2/blob/master/fastai/structured.py

from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import LabelEncoder, Imputer, StandardScaler
from pandas.api.types import is_string_dtype, is_numeric_dtype
from sklearn.ensemble import forest
from sklearn.tree import export_graphviz

def get_sample(df,n):
    """ Gets a random sample of n rows from df, without replacement.
    Parameters:
    -----------
    df: A pandas data frame, that you wish to sample from.
    n: The number of rows you wish to sample.
    Returns:
    --------
    return value: A random sample of n rows of df.
    Examples:
    ---------
    >>> df = pd.DataFrame({'col1' : [1, 2, 3], 'col2' : ['a', 'b', 'a']})
    >>> df
       col1 col2
    0     1    a
    1     2    b
    2     3    a
    >>> get_sample(df, 2)
       col1 col2
    2     3    a
    1     2    b
    """
    idxs = sorted(np.random.permutation(len(df))[:n])
    return df.iloc[idxs].copy()

def proc_df(df, y_fld, skip_flds=None, do_scale=False, na_dict=None,
            preproc_fn=None, max_n_cat=None, subset=None, mapper=None):

    """ proc_df takes a data frame df and splits off the response variable, and
    changes the df into an entirely numeric dataframe.
    Parameters:
    -----------
    df: The data frame you wish to process.
    y_fld: The name of the response variable
    skip_flds: A list of fields that dropped from df.
    do_scale: Standardizes each column in df,Takes Boolean Values(True,False)
    na_dict: a dictionary of na columns to add. Na columns are also added if there
        are any missing values.
    preproc_fn: A function that gets applied to df.
    max_n_cat: The maximum number of categories to break into dummy values, instead
        of integer codes.
    subset: Takes a random subset of size subset from df.
    mapper: If do_scale is set as True, the mapper variable
        calculates the values used for scaling of variables during training time(mean and standard deviation).
    Returns:
    --------
    [x, y, nas, mapper(optional)]:
        x: x is the transformed version of df. x will not have the response variable
            and is entirely numeric.
        y: y is the response variable
        nas: returns a dictionary of which nas it created, and the associated median.
        mapper: A DataFrameMapper which stores the mean and standard deviation of the corresponding continous
        variables which is then used for scaling of during test-time.
    Examples:
    ---------
    >>> df = pd.DataFrame({'col1' : [1, 2, 3], 'col2' : ['a', 'b', 'a']})
    >>> df
       col1 col2
    0     1    a
    1     2    b
    2     3    a
    note the type of col2 is string
    >>> train_cats(df)
    >>> df
       col1 col2
    0     1    a
    1     2    b
    2     3    a
    now the type of col2 is category { a : 1, b : 2}
    >>> x, y, nas = proc_df(df, 'col1')
    >>> x
       col2
    0     1
    1     2
    2     1
    >>> data = DataFrame(pet=["cat", "dog", "dog", "fish", "cat", "dog", "cat", "fish"],
                 children=[4., 6, 3, 3, 2, 3, 5, 4],
                 salary=[90, 24, 44, 27, 32, 59, 36, 27])
    >>> mapper = DataFrameMapper([(:pet, LabelBinarizer()),
                          ([:children], StandardScaler())])
    >>>round(fit_transform!(mapper, copy(data)), 2)
    8x4 Array{Float64,2}:
    1.0  0.0  0.0   0.21
    0.0  1.0  0.0   1.88
    0.0  1.0  0.0  -0.63
    0.0  0.0  1.0  -0.63
    1.0  0.0  0.0  -1.46
    0.0  1.0  0.0  -0.63
    1.0  0.0  0.0   1.04
    0.0  0.0  1.0   0.21
    """
    if not skip_flds: skip_flds=[]
    if subset: df = get_sample(df,subset)
    df = df.copy()
    if preproc_fn: preproc_fn(df)
    y = df[y_fld].values
    df.drop(skip_flds+[y_fld], axis=1, inplace=True)

    if na_dict is None: na_dict = {}
    for n,c in df.items(): na_dict = fix_missing(df, c, n, na_dict)
    if do_scale: mapper = scale_vars(df, mapper)
    for n,c in df.items(): numericalize(df, c, n, max_n_cat)
    res = [pd.get_dummies(df, dummy_na=True), y, na_dict]
    if do_scale: res = res + [mapper]
    return res

def rf_feat_importance(m, df):
    return pd.DataFrame({'cols':df.columns, 'imp':m.feature_importances_}
                       ).sort_values('imp', ascending=False)

def set_rf_samples(n):
    """ Changes Scikit learn's random forests to give each tree a random sample of
    n random rows.
    """
    forest._generate_sample_indices = (lambda rs, n_samples:
        forest.check_random_state(rs).randint(0, n_samples, n))

def reset_rf_samples():
    """ Undoes the changes produced by set_rf_samples.
    """
    forest._generate_sample_indices = (lambda rs, n_samples:
        forest.check_random_state(rs).randint(0, n_samples, n_samples))

# In[ ]:


# For autoreloading modules
# For notebook plotting

# Standard libraries
import os
import numpy as np
import pandas as pd

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from pdpbox import pdp
from plotnine import *
from pandas_summary import DataFrameSummary
from sklearn.ensemble import RandomForestRegressor
from IPython.display import display

# Machine Learning
import sklearn
from sklearn import metrics
from scipy.cluster import hierarchy as hc
from fastai.imports import *

# Directories
KAGGLE_DIR = '../input/'

# Info about dataset
print('Files and directories: \n{}\n'.format(os.listdir("../input")))

print('\n# File sizes')
for file in os.listdir(KAGGLE_DIR):
    print('{}| {} MB'.format(file.ljust(30), 
                             str(round(os.path.getsize(KAGGLE_DIR + file) / 1000000, 2))))

# And of course, we import our data from the Kaggle kernel directory and load it into two different DataFrames. one for the training data and one for the test data.

# In[ ]:


# Import dataset
train = pd.read_csv(KAGGLE_DIR + 'train_V2.csv')
test = pd.read_csv(KAGGLE_DIR + 'test_V2.csv')

# ![API Img](http://media.comicbook.com/2018/03/pubg-api-1093349.jpeg)
# 

# # Initial Exploration <a id="3"></a>

# Let's look at the DataFrame from head to tail.

# In[ ]:


# First five rows (From Head)
print('First 5 rows: ')
display(train.head())

# Last five rows (To Tail)
print('Last 5 rows: ')
display(train.tail())

# Summary Statistics of the training data.

# In[ ]:


# Stats
train.describe()

# Data types, memory usage, shape, etc.

# In[ ]:


# Types, Data points, memory usage, etc.
train.info()

# Check dataframe's shape
print('Shape of training set: ', train.shape)

# ### Feature descriptions (From Kaggle)
# 
# * DBNOs - Number of enemy players knocked.
# * assists - Number of enemy players this player damaged that were killed by teammates.
# * boosts - Number of boost items used.
# * damageDealt - Total damage dealt. Note: Self inflicted damage is subtracted.
# * headshotKills - Number of enemy players killed with headshots.
# * heals - Number of healing items used.
# * Id - Player’s Id
# * killPlace - Ranking in match of number of enemy players killed.
# * killPoints - Kills-based external ranking of player. (Think of this as an Elo ranking where only kills matter.) If there is a value other than -1 in rankPoints, then any 0 in killPoints should be treated as a “None”.
# * killStreaks - Max number of enemy players killed in a short amount of time.
# * kills - Number of enemy players killed.
# * longestKill - Longest distance between player and player killed at time of death. This may be misleading, as downing a player and driving away may lead to a large longestKill stat.
# * matchDuration - Duration of match in seconds.
# * matchId - ID to identify match. There are no matches that are in both the training and testing set.
# * matchType - String identifying the game mode that the data comes from. The standard modes are “solo”, “duo”, “squad”, “solo-fpp”, “duo-fpp”, and “squad-fpp”; other modes are from events or custom matches.
# * rankPoints - Elo-like ranking of player. This ranking is inconsistent and is being deprecated in the API’s next version, so use with caution. Value of -1 takes place of “None”.
# * revives - Number of times this player revived teammates.
# * rideDistance - Total distance traveled in vehicles measured in meters.
# * roadKills - Number of kills while in a vehicle.
# * swimDistance - Total distance traveled by swimming measured in meters.
# * teamKills - Number of times this player killed a teammate.
# * vehicleDestroys - Number of vehicles destroyed.
# * walkDistance - Total distance traveled on foot measured in meters.
# * weaponsAcquired - Number of weapons picked up.
# * winPoints - Win-based external ranking of player. (Think of this as an Elo ranking where only winning matters.) If there is a value other than -1 in rankPoints, then any 0 in winPoints should be treated as a “None”.
# * groupId - ID to identify a group within a match. If the same group of players plays in different matches, they will have a different groupId each time.
# * numGroups - Number of groups we have data for in the match.
# * maxPlace - Worst placement we have data for in the match. This may not match with numGroups, as sometimes the data skips over placements.
# * winPlacePerc - The target of prediction. This is a percentile winning placement, where 1 corresponds to 1st place, and 0 corresponds to last place in the match. It is calculated off of maxPlace, not numGroups, so it is possible to have missing chunks in a match.
# 
# [Source](https://www.kaggle.com/c/pubg-finish-placement-prediction/data)

# In[ ]:


# Use this code if you want to store and read DataFrames in a feather format
# os.makedirs('tmp', exist_ok=True)
# train.to_feather('tmp/PUBG')
# df_raw = pd.read_feather('tmp/PUBG')

# # Illegal Match <a id="4"></a>

# Fellow Kaggler '[averagemn](https://www.kaggle.com/donkeys)' brought to our attention that there is one particular player with a 'winPlacePerc' of NaN. The case was that this match had only one player. We will delete this row from our dataset.

# In[ ]:


# Check row with NaN value
train[train['winPlacePerc'].isnull()]

# Let's delete this entry:

# In[ ]:


# Drop row with NaN 'winPlacePerc' value
train.drop(2744604, inplace=True)

# And he's gone!

# In[ ]:


# The row at index 2744604 will be gone
train[train['winPlacePerc'].isnull()]

# # Feature Engineering <a id="5"></a>

# Earlier in this kernel we created the new features ''totalDistance'' and  ''headshot_rate". In this section we add more interesting features to improve the predictive quality of our machine learning models.
# 
# Initial ideas for this section come from [this amazing kernel](https://www.kaggle.com/deffro/eda-is-fun).
# 
# Note: It is important with feature engineering that you also add the engineered features to your test set!

# ### Players Joined

# This is likely a very valuable feature for our model. If we know how many people are in a match we can normalize other features and get stronger predictions on individual players.

# In[ ]:


# playersJoined
train['playersJoined'] = train.groupby('matchId')['matchId'].transform('count')
plt.figure(figsize=(15,10))
sns.countplot(train[train['playersJoined']>=75]['playersJoined'])
plt.title('playersJoined')
plt.show()

# There are a few matches with fewer than 75 players that are not displayed here. As you can see most of the matches are nearly packed a have nearly 100 players. It is nevertheless interesting to take these features into our analysis.

# ### Normalized features

# Now that we have a feature 'playersJoined' we can normalize other features based on the amount of players. Features that can be valuable to normalize are:
# 1. kills
# 2. damageDealt
# 3. maxPlace
# 4. matchDuration
# 
# Let's try out some things!

# **Normalize features**

# In[ ]:


# Create normalized features
train['killsNorm'] = train['kills']*((100-train['playersJoined'])/100 + 1)
train['damageDealtNorm'] = train['damageDealt']*((100-train['playersJoined'])/100 + 1)
train['maxPlaceNorm'] = train['maxPlace']*((100-train['playersJoined'])/100 + 1)
train['matchDurationNorm'] = train['matchDuration']*((100-train['playersJoined'])/100 + 1)
# Compare standard features and normalized features
to_show = ['Id', 'kills','killsNorm','damageDealt', 'damageDealtNorm', 'maxPlace', 'maxPlaceNorm', 'matchDuration', 'matchDurationNorm']
train[to_show][0:11]

# ### Heals and Boosts

# We create a feature called 'healsandboosts' by adding heals and boosts. (duh!) We are not sure if this has additional predictive value, but we can always delete it later if the feature importance according to our random forest model is too low.

# In[ ]:


# Create new feature healsandboosts
train['healsandboosts'] = train['heals'] + train['boosts']
train[['heals', 'boosts', 'healsandboosts']].tail()

# ### Killing without moving

# We try to identify cheaters by checking if people are getting kills without moving. We first identify the totalDistance travelled by a player and then set a boolean value to True if someone got kills without moving a single inch. We will remove cheaters in our outlier detection section.

# In[ ]:


# Create feature totalDistance
train['totalDistance'] = train['rideDistance'] + train['walkDistance'] + train['swimDistance']
# Create feature killsWithoutMoving
train['killsWithoutMoving'] = ((train['kills'] > 0) & (train['totalDistance'] == 0))

# The feature headshot_rate will also help us to catch cheaters.

# In[ ]:


# Create headshot_rate feature
train['headshot_rate'] = train['headshotKills'] / train['kills']
train['headshot_rate'] = train['headshot_rate'].fillna(0)

# # Outlier Detection <a id="6"></a>

# Some rows in our dataset have weird characteristics. The players could be cheaters, maniacs or just anomalies. Removing these outliers will likely improve results.
# 
# Inspiration for this section comes from [this amazing Kaggle Kernel.](https://www.kaggle.com/rejasupotaro/cheaters-and-zombies)

# ![Alt Text](https://media.giphy.com/media/OPRbXcsGctvZC/giphy.gif)

# **Kills without movement**

# This is perhaps the most obvious sign of cheating in the game. It is already fishy if a player hasn't moved during the whole game, but the player could be AFK and got killed. However, if the player managed to get kills without moving it is most likely a cheater.

# In[ ]:


# Check players who kills without moving
display(train[train['killsWithoutMoving'] == True].shape)
train[train['killsWithoutMoving'] == True].head(10)

# Got the suckers! 

# In[ ]:


# Remove outliers
train.drop(train[train['killsWithoutMoving'] == True].index, inplace=True)

# **Anomalies in roadKills**

# In[ ]:


# Players who got more than 10 roadKills
train[train['roadKills'] > 10]

# In[ ]:


# Drop roadKill 'cheaters'
train.drop(train[train['roadKills'] > 10].index, inplace=True)

# Note that player c3e444f7d1289d drove 5 meters but killed 14 people with it. Sounds insane doesn't it?
# 
# ![Alt Text](https://media.giphy.com/media/3o7aD85usFbbbrCR3i/giphy.gif)

# **Anomalies in aim (More than 45 kills)**

# Let's plot the total kills for every player first. It doesn't look like there are too many outliers.

# In[ ]:


# Plot the distribution of kills
plt.figure(figsize=(12,4))
sns.countplot(data=train, x=train['kills']).set_title('Kills')
plt.show()

# Let's take a closer look.

# In[ ]:


# Players who got more than 30 kills
display(train[train['kills'] > 30].shape)
train[train['kills'] > 30].head(10)

# In[ ]:


# Remove outliers
train.drop(train[train['kills'] > 30].index, inplace=True)

# What do you think? Should we remove all these outliers from our dataset?

# **Anomalies in aim part 2 (100% headshot rate)**

# Again, we first take a look at the whole dataset and create a new feature 'headshot_rate'.
# We see that the most players score in the 0 to 10% region. However, there are a few anomalies that have a headshot_rate of 100% percent with more than 9 kills!

# In[ ]:


# Plot the distribution of headshot_rate
plt.figure(figsize=(12,4))
sns.distplot(train['headshot_rate'], bins=10)
plt.show()

# In[ ]:


# Players who made a minimum of 10 kills and have a headshot_rate of 100%
display(train[(train['headshot_rate'] == 1) & (train['kills'] > 9)].shape)
train[(train['headshot_rate'] == 1) & (train['kills'] > 9)].head(10)

# It is unclear if these players are cheating so we are probably not deleting these players from the dataset.
# If they are legitimate players, they are probably really crushing the game!
# 
# ![Alt Text](https://media.giphy.com/media/l3mZrOajz5VCZf7Hy/giphy.gif)
# 

# **Anomalies in aim part 3 (Longest kill)**

# Most kills are made from a distance of 100 meters or closer. There are however some outliers who make a kill from more than 1km away. This is probably done by cheaters.

# In[ ]:


# Plot the distribution of longestKill
plt.figure(figsize=(12,4))
sns.distplot(train['longestKill'], bins=10)
plt.show()

# Let's take a look at the players who make these shots.

# In[ ]:


# Check out players who made kills with a distance of more than 1 km
display(train[train['longestKill'] >= 1000].shape)
train[train['longestKill'] >= 1000].head(10)

# In[ ]:


# Remove outliers
train.drop(train[train['longestKill'] >= 1000].index, inplace=True)

# There is something fishy going on with these players. We are probably better off removing them from our dataset.
# 
# ![Alt Text](https://media.giphy.com/media/RHJkLqcdvMQF4GI3P7/giphy.gif)

# **Anomalies in travelling (rideDistance, walkDistance and swimDistance)**
# 
# Let's check out anomalies in Distance travelled.

# In[ ]:


# Summary statistics for the Distance features
train[['walkDistance', 'rideDistance', 'swimDistance', 'totalDistance']].describe()

# **walkDistance**

# In[ ]:


# Plot the distribution of walkDistance
plt.figure(figsize=(12,4))
sns.distplot(train['walkDistance'], bins=10)
plt.show()

# In[ ]:


# walkDistance anomalies
display(train[train['walkDistance'] >= 10000].shape)
train[train['walkDistance'] >= 10000].head(10)

# In[ ]:


# Remove outliers
train.drop(train[train['walkDistance'] >= 10000].index, inplace=True)

# **rideDistance**

# In[ ]:


# Plot the distribution of rideDistance
plt.figure(figsize=(12,4))
sns.distplot(train['rideDistance'], bins=10)
plt.show()

# In[ ]:


# rideDistance anomalies
display(train[train['rideDistance'] >= 20000].shape)
train[train['rideDistance'] >= 20000].head(10)

# In[ ]:


# Remove outliers
train.drop(train[train['rideDistance'] >= 20000].index, inplace=True)

# Cheaters or do they just like to ride like these guys?
# 
# ![Alt Text](https://media.giphy.com/media/qlCFjkSruesco/giphy.gif)

# **swimDistance**

# In[ ]:


# Plot the distribution of swimDistance
plt.figure(figsize=(12,4))
sns.distplot(train['swimDistance'], bins=10)
plt.show()

# In[ ]:


# Players who swam more than 2 km
train[train['swimDistance'] >= 2000]

# In[ ]:


# Remove outliers
train.drop(train[train['swimDistance'] >= 2000].index, inplace=True)

# Do you think these guys are legit?
# 
# ![Alt Text](https://thumbs.gfycat.com/EvenSpiffyFerret-size_restricted.gif)

# **Anomalies in supplies (weaponsAcquired)**
# 
# Most people acquire between 0 and 10 weapons in a game, but you also see some people acquire more than 80 weapons! Let's check these guys out.

# In[ ]:


# Plot the distribution of weaponsAcquired
plt.figure(figsize=(12,4))
sns.distplot(train['weaponsAcquired'], bins=100)
plt.show()

# In[ ]:


# Players who acquired more than 80 weapons
display(train[train['weaponsAcquired'] >= 80].shape)
train[train['weaponsAcquired'] >= 80].head()

# In[ ]:


# Remove outliers
train.drop(train[train['weaponsAcquired'] >= 80].index, inplace=True)

# We should probably remove these outliers from our model. Do you agree?
# 
# Note that player 3f2bcf53b108c4 acquired 236 weapons in one game!
# 
# ![Alt Text](https://media.giphy.com/media/69lWR6c8Afx9qeg2Tu/giphy.gif)

# **Anomalies in supplies part 2 (heals)**
# 
# Most players us 5 healing items or less. We can again recognize some weird anomalies

# In[ ]:


# Distribution of heals
plt.figure(figsize=(12,4))
sns.distplot(train['heals'], bins=10)
plt.show()

# In[ ]:


# 40 or more healing items used
display(train[train['heals'] >= 40].shape)
train[train['heals'] >= 40].head(10)

# In[ ]:


# Remove outliers
train.drop(train[train['heals'] >= 40].index, inplace=True)

# **Outlier conclusions**

# We removed about 2000 players from our dataset. Do you think this is too much? Please let us know in the comments.

# In[ ]:


# Remaining players in the training set
train.shape

# # Categorical Variables <a id="7"></a>

# We will one hot encode the 'matchType' feature to use it in our Random Forest model.

# In[ ]:


print('There are {} different Match types in the dataset.'.format(train['matchType'].nunique()))

# In[ ]:


# One hot encode matchType
train = pd.get_dummies(train, columns=['matchType'])

# Take a look at the encoding
matchType_encoding = train.filter(regex='matchType')
matchType_encoding.head()

# There are a lot of groupId's and matchId's so one-hot encoding them is computational suicide.
# We will turn them into category codes. That way we can still benefit from correlations between groups and matches in our Random Forest algorithm.

# In[ ]:


# Turn groupId and match Id into categorical types
train['groupId'] = train['groupId'].astype('category')
train['matchId'] = train['matchId'].astype('category')

# Get category coding for groupId and matchID
train['groupId_cat'] = train['groupId'].cat.codes
train['matchId_cat'] = train['matchId'].cat.codes

# Get rid of old columns
train.drop(columns=['groupId', 'matchId'], inplace=True)

# Lets take a look at our newly created features
train[['groupId_cat', 'matchId_cat']].head()

# In[ ]:


# Drop Id column, because it probably won't be useful for our Machine Learning algorithm,
# because the test set contains different Id's
train.drop(columns = ['Id'], inplace=True)

# **voilà!**

# # Preparation for Machine Learning <a id="8"></a>

# ## Sampling

# We will take a sample of 500000 rows from our training set for easy debugging and exploration.

# In[ ]:


# Take sample for debugging and exploration
sample = 500000
df_sample = train.sample(sample)

# ## Split target variable, validation data, etc.

# In[ ]:


# Split sample into training data and target variable
df = df_sample.drop(columns = ['winPlacePerc']) #all columns except target
y = df_sample['winPlacePerc'] # Only target variable

# In[ ]:


# Function for splitting training and validation data
def split_vals(a, n : int): 
    return a[:n].copy(), a[n:].copy()
val_perc = 0.12 # % to use for validation set
n_valid = int(val_perc * sample) 
n_trn = len(df)-n_valid
# Split data
raw_train, raw_valid = split_vals(df_sample, n_trn)
X_train, X_valid = split_vals(df, n_trn)
y_train, y_valid = split_vals(y, n_trn)

# Check dimensions of samples
print('Sample train shape: ', X_train.shape, 
      'Sample target shape: ', y_train.shape, 
      'Sample validation shape: ', X_valid.shape)

# ## Set metrics (MAE)

# [Mean Absolute Error (MAE)](https://en.wikipedia.org/wiki/Mean_absolute_error) is the metric that is used for this competition. The scikit-learn library already programmed this metric for us so we don't have to implement it from scratch.

# In[ ]:


# Metric used for the PUBG competition (Mean Absolute Error (MAE))
from sklearn.metrics import mean_absolute_error

# Function to print the MAE (Mean Absolute Error) score
# This is the metric used by Kaggle in this competition
def print_score(m : RandomForestRegressor):
    res = ['mae train: ', mean_absolute_error(m.predict(X_train), y_train), 
           'mae val: ', mean_absolute_error(m.predict(X_valid), y_valid)]
    if hasattr(m, 'oob_score_'): res.append(m.oob_score_)
    print(res)

# ## First basic Random Forest Model

# In[ ]:


# Train basic model
m1 = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, max_features='sqrt',
                          n_jobs=-1)
m1.fit(X_train, y_train)
print_score(m1)

# # Feature Importance <a id="9"></a>

# The [fastai](https://www.fast.ai/) library gives us an easy way to analyze feature importances from a random forest algorithm with just one line of code!

# In[ ]:


# What are the most predictive features according to our basic random forest model
fi = rf_feat_importance(m1, df); fi[:10]

# In[ ]:


# Plot a feature importance graph for the 20 most important features
plot1 = fi[:20].plot('cols', 'imp', figsize=(14,6), legend=False, kind = 'barh')
plot1

# In[ ]:


# Use this code if you want to save the figure
#fig = plot1.get_figure()
#fig.savefig("Feature_importances(AllFeatures).png")

# In[ ]:


# Keep only significant features
to_keep = fi[fi.imp>0.005].cols
print('Significant features: ', len(to_keep))
to_keep

# In[ ]:


# Make a DataFrame with only significant features
df_keep = df[to_keep].copy()
X_train, X_valid = split_vals(df_keep, n_trn)

# ## Second Random Forest Model

# This time we use only the top features to train a random forest model. This often improves results a little bit.

# In[ ]:


# Train model on top features
m2 = RandomForestRegressor(n_estimators=80, min_samples_leaf=3, max_features='sqrt',
                          n_jobs=-1)
m2.fit(X_train, y_train)
print_score(m2)

# **Feature importance for top features**

# In[ ]:


# Get feature importances of our top features
fi_to_keep = rf_feat_importance(m2, df_keep)
plot2 = fi_to_keep.plot('cols', 'imp', figsize=(14,6), legend=False, kind = 'barh')
plot2

# Use this code if you want to save the figure
#fig = plot2.get_figure()
#fig.savefig("Feature_importances(TopFeatures).png")

# ## Correlations

# **Dendrogram (to view correlation of features)**

# In[ ]:


# Create a Dendrogram to view highly correlated features
corr = np.round(scipy.stats.spearmanr(df_keep).correlation, 4)
corr_condensed = hc.distance.squareform(1-corr)
z = hc.linkage(corr_condensed, method='average')
fig = plt.figure(figsize=(14,10))
dendrogram = hc.dendrogram(z, labels=df_keep.columns, orientation='left', leaf_font_size=16)
plt.plot()

# In[ ]:


# Use this code if you want to save the figure
#plt.savefig('Dendrogram.png')

# **Correlation Heatmap**

# In[ ]:


# Correlation heatmap
corr = df_keep.corr()

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Create heatmap
heatmap = sns.heatmap(corr)

# In[ ]:


# Use this code if you want to save the figure
#fig = heatmap.get_figure()
#fig.savefig("Heatmap(TopFeatures).png")

# **Predictive quality of kills**

# In[ ]:


# Plot the predictive quality of kills 
x_all = get_sample(train, 100000)
ggplot(x_all, aes('kills','winPlacePerc'))+stat_smooth(se=True, colour='red', method='mavg')

# **Predictive quality of walkDistance**

# In[ ]:


# Plot the predictive quality of walkDistance
x_all = get_sample(train, 100000)
ggplot(x_all, aes('walkDistance','winPlacePerc'))+stat_smooth(se=True, colour='red', method='mavg')

# # Final Random Forest Model <a id="10"></a>

# In[ ]:


# Prepare data
val_perc_full = 0.12 # % to use for validation set
n_valid_full = int(val_perc_full * len(train)) 
n_trn_full = len(train)-n_valid_full
df_full = train.drop(columns = ['winPlacePerc']) # all columns except target
y = train['winPlacePerc'] # target variable
df_full = df_full[to_keep] # Keep only relevant features
X_train, X_valid = split_vals(df_full, n_trn_full)
y_train, y_valid = split_vals(y, n_trn_full)

# Check dimensions of data
print('Sample train shape: ', X_train.shape, 
      'Sample target shape: ', y_train.shape, 
      'Sample validation shape: ', X_valid.shape)

# In[ ]:


# Train final model
# You should get better results by increasing n_estimators
# and by playing around with the parameters
m3 = RandomForestRegressor(n_estimators=70, min_samples_leaf=3, max_features=0.5,
                          n_jobs=-1)
m3.fit(X_train, y_train)
print_score(m3)

# # Kaggle Submission <a id="11"></a>

# In[ ]:


# Add engineered features to the test set
test['headshot_rate'] = test['headshotKills'] / test['kills']
test['headshot_rate'] = test['headshot_rate'].fillna(0)
test['totalDistance'] = test['rideDistance'] + test['walkDistance'] + test['swimDistance']
test['playersJoined'] = test.groupby('matchId')['matchId'].transform('count')
test['killsNorm'] = test['kills']*((100-test['playersJoined'])/100 + 1)
test['damageDealtNorm'] = test['damageDealt']*((100-test['playersJoined'])/100 + 1)
test['maxPlaceNorm'] = test['maxPlace']*((100-train['playersJoined'])/100 + 1)
test['matchDurationNorm'] = test['matchDuration']*((100-test['playersJoined'])/100 + 1)
test['healsandboosts'] = test['heals'] + test['boosts']
test['killsWithoutMoving'] = ((test['kills'] > 0) & (test['totalDistance'] == 0))

# Turn groupId and match Id into categorical types
test['groupId'] = test['groupId'].astype('category')
test['matchId'] = test['matchId'].astype('category')

# Get category coding for groupId and matchID
test['groupId_cat'] = test['groupId'].cat.codes
test['matchId_cat'] = test['matchId'].cat.codes

# Remove irrelevant features from the test set
test_pred = test[to_keep].copy()

# Fill NaN with 0 (temporary)
test_pred.fillna(0, inplace=True)
test_pred.head()

# In[ ]:


# Make submission ready for Kaggle
# We use our final Random Forest model (m3) to get the predictions
predictions = np.clip(a = m3.predict(test_pred), a_min = 0.0, a_max = 1.0)
pred_df = pd.DataFrame({'Id' : test['Id'], 'winPlacePerc' : predictions})

# Create submission file
pred_df.to_csv("submission.csv", index=False)

# **Check of submission file**

# It is always nice to take a look at few of your predictions to make sure that the structure is right for a Kaggle submission.

# In[ ]:


# Last check of submission
print('Head of submission: ')
display(pred_df.head())
print('Tail of submission: ')
display(pred_df.tail())
