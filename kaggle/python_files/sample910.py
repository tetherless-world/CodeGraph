#!/usr/bin/env python
# coding: utf-8

# # Cheaters???
# 
# My friend who plays PUBG said that there are cheaters in this game.
# If it's true, we should probably remove such players from the training dataset not to get the model confused.
# 
# I learned that there are some types of cheats.
# 
# <div align="center">
#     <img src="https://cdn.mos.cms.futurecdn.net/36pdCgyXDgKmbqSpxnJ6Ue-650-80.png" width="640">
#     <a href="https://www.gamesradar.com/pubg-cheats-explained/">PlayerUnknown's Battlegrounds cheats explained | GamesRadar+</a>
# </div>
# 
# ## Aim Hacks
# 
# > They will take control of a players aim and automatically target it towards opponents. 
# 
# ## Speed Hacks
# 
# > They usually give the player a massive speed increase, meaning they can go from one side of the map to the other in seconds.
# 
# ## Recoil Hacks
# 
# > automatically manage the recoil. This means all they have to do is press the fire button and don’t have to adjust their mouse to account for the recoil, as the script will do it all for them and every shot will go exactly where they want it to.
# 
# ## Wall Hacks
# 
# > Wall hacks basically allow cheaters to see other players through walls, or add extra UI elements to reveal a players location.
# 
# # How cheaters look like
# 
# Now I have some ideas about cheaters.
# 
# - Acquiring 100 weapons without moving
# - Killing 100 players without moving
# - 100/100 kills are headshots
# - Reviving 100 times
# - ...
# 
# Let's take a look at the actual data.

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.simplefilter(action='ignore', category=Warning)

# In[ ]:


train = pd.read_csv('../input/train.csv')
display(train.head())
display(train.describe())

# In[ ]:


def show_countplot(column):
    plt.figure(figsize=(12,4))
    sns.countplot(data=train, x=column).set_title(column)
    plt.show()
    
def show_distplot(column):
    plt.figure(figsize=(12, 4))
    sns.distplot(train[column], bins=50)
    plt.show()

# # Aim Hacks
# 
# ## kills
# 
# - ID: 299122 got 57 weapons, killed 40 players, but his total distance is 50.26m ???
# - ID: 94553 killed 48 players without healing ???
# - ID: 4303492 killed 42 players and all of them were run over by his vehicle ???

# In[ ]:


show_countplot('kills')

# In[ ]:


train[train['kills'] >= 40]

# ## Headshot rate
# 
# "headshot rate = 100%" doesn''t look cheaters to me. They look good players and actually they won the game!

# In[ ]:


train['headshot_rate'] = train['headshotKills'] / train['kills']
train['headshot_rate'] = train['headshot_rate'].fillna(0)
show_distplot('headshot_rate')

# In[ ]:


train[(train['headshot_rate'] >= 1) & (train['kills'] >= 10)]

# ## longestKill
# 
# Lucky or Cheater? My friend told me that 1km sniper shot is not impossible in this game.

# In[ ]:


show_distplot('longestKill')

# In[ ]:


train[train['longestKill'] >= 1000]

# ## teamKills
# 
# Hmm... they are just madmen?

# In[ ]:


show_countplot('teamKills')

# In[ ]:


train[train['teamKills'] >= 5]

# # Speed Hacks
# 
# I think the map is 8*8 km and each yellow square is 1km.
# 
# <div align="center">
#     <img src="https://d1u5p3l4wpay3k.cloudfront.net/battlegrounds_gamepedia_en/thumb/e/ea/Map.jpg/1600px-Map.jpg?version=24add1e17865d696a24f50bcc3f27da5" width="640">
#     <a href="https://pubg.gamepedia.com/MapsMaps">PLAYERUNKNOWN'S BATTLEGROUNDS Wiki</a>
# </div>
# 
#  I want to measure how long it takes to run 1km in this game. Looking at the data, I have no idea how they ran long distance without being killed.
#  
# - walkDistance max:  17km 300m
# - rideDistance max: 48km 390m
# - swimDistance max: 5km 286m
# 
# ## *Distance
# 
# But they didn't kill players so much = It's okay to ignore?

# In[ ]:


train[['walkDistance', 'rideDistance', 'swimDistance']].describe()

# In[ ]:


show_distplot('walkDistance')

# In[ ]:


train[train['walkDistance'] >= 13000]

# In[ ]:


show_distplot('rideDistance')

# In[ ]:


train[train['rideDistance'] >= 30000]

# # Items/Supplies Hacks
# 
# Item/Supplies make a game advantageous. Let's see how many items/supplies players got in a game.

# ## weaponsAcquired
# 
# Is `weaponsAcquired > 60` possible? If the player moved long distance...?

# In[ ]:


show_countplot('weaponsAcquired')

# In[ ]:


train[train['weaponsAcquired'] >= 60]

# ## heals
# 
# Can we say "they were lucky" for players who are `heals >= 40 && kills >= 40` ???

# In[ ]:


show_countplot('heals')

# In[ ]:


train[train['heals'] >= 50]

# Even if you successfully find a cheater, you still have several options about how to deal with cheaters.
# 
# - Remove the player
# - Remove the group
# - Remove the match
# - Leave it as it is
# 
# Do we need to leave it as it is if there are cheaters in the test dataset as well?

# # Zombies!!!
# 
# Kyle Beck pointed out that there is a zombie mode in PUBG! (Thank you for your comment!)
# 
# <div align="center">
#     <img src="http://cdn.gamer-network.net/2018/metabomb/pubghowtoplayzombiemode.JPG" width="640">
#     <a href="https://www.metabomb.net/pubg/gameplay-guides/pubg-how-to-play-zombie-mode">PUBG: How to play zombie mode | Metabomb</a>
# </div>
#     
# 
# The zombie mode is a match like "A few humans VS A large amount of zombies". It makes sense that a few players killed a large amount of players in this game. Let's see if such matches are included in this dataset.
# 
# First, let's calculate the number of players in each team.

# In[ ]:


agg = train.groupby(['groupId']).size().to_frame('players_in_team')
train = train.merge(agg, how='left', on=['groupId'])
train[['matchId', 'groupId', 'players_in_team']].head()

# Next, aggregate the number to see min, max, mean, and variance.

# In[ ]:


agg = train.groupby(['matchId']).agg({'players_in_team': ['min', 'max', 'mean']})
agg.columns = ['_'.join(x) for x in agg.columns.ravel()]
train = train.merge(agg, how='left', on=['matchId'])
train['players_in_team_var'] = train.groupby(['matchId'])['players_in_team'].var()
display(train[['matchId', 'players_in_team', 'players_in_team_min', 'players_in_team_max', 'players_in_team_mean', 'players_in_team_var']].head())
display(train[['matchId', 'players_in_team', 'players_in_team_min', 'players_in_team_max', 'players_in_team_mean', 'players_in_team_var']].describe())

# What if the variance is high...?

# In[ ]:


plt.figure(figsize=(10,4))
for i, match_id in enumerate(train.nlargest(2, 'players_in_team_var')['matchId'].values):
    plt.subplot(1, 2, i + 1)
    sns.distplot(train[train['matchId'] == match_id]['players_in_team'])
plt.show()

# I found **"A few humans VS A large amount of zombies"** !!!
# However, I can see few matches that look like the zombie mode = can be outliers.
# 
# There are more modes other than that according to <a href="https://pubg.gamepedia.com/Game_Modes">Game Modes - PLAYERUNKNOWN'S BATTLEGROUNDS Wiki</a>
# 
# - Solos
# - Duos
# - Squads
# 
# ## Solos
# 
#  Complete free for all, kill everyone, be the last one alive (`players_in_team_max = 1`).

# In[ ]:


plt.figure(figsize=(20,5))
for i, match_id in enumerate(train[train['players_in_team_max'] == 1]['matchId'].values[:4]):
    plt.subplot(1, 4, i + 1)
    sns.distplot(train[train['matchId'] == match_id]['players_in_team'])
plt.show()

# ## Duos
# 
# You will be paired up with another individual and will compete to be the last ones alive (`players_in_team_max = 2 & variance ≈ 0`).

# In[ ]:


plt.figure(figsize=(20,5))
for i, match_id in enumerate(train[(train['players_in_team_max'] == 2) & (train['players_in_team_var'] == 0)]['matchId'].values[:4]):
    plt.subplot(1, 4, i + 1)
    sns.distplot(train[train['matchId'] == match_id]['players_in_team'])
plt.show()

# ## Squads
# 
# You can team up in groups of 2, 3 or 4 players, or if you prefer, you can still play solo and take on everyone alone in the match (`players_in_team_max = 4 & variance ≈ 0`).

# In[ ]:


plt.figure(figsize=(20,5))
for i, match_id in enumerate(train[(train['players_in_team_max'] == 4) & (train['players_in_team_var'] > 0)]['matchId'].values[:4]):
    plt.subplot(1, 4, i + 1)
    sns.distplot(train[train['matchId'] == match_id]['players_in_team'])
plt.show()

# This can be an important feature!
# 
# # Predict game mode
# 
# (I'm working on clustering matches now...)

# In[ ]:


from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.cluster import KMeans

# In[ ]:


train['players_in_team_var'] = train['players_in_team_var'].fillna(-1)
columns = ['players_in_team', 'players_in_team_min', 'players_in_team_max', 'players_in_team_mean', 'players_in_team_var']
data = train.groupby(['matchId']).first()[columns].reset_index()
preprocessor = make_pipeline(StandardScaler(), PCA(n_components=2))
reduced_data = preprocessor.fit_transform(data)
model = KMeans(n_clusters=8)
model.fit(reduced_data)
data['game_mode'] = model.predict(reduced_data)

plt.figure(figsize=(6,6))
plt.scatter(reduced_data[:,0], reduced_data[:,1], c=data['game_mode'])
plt.show()

# In[ ]:


data.groupby(['game_mode']).mean().reset_index()[
    ['game_mode', 'players_in_team', 'players_in_team_min', 'players_in_team_max', 'players_in_team_mean', 'players_in_team_var']
].merge(data.groupby(['game_mode']).size().to_frame('count'), how='left', on=['game_mode'])

# In[ ]:


plt.figure(figsize=(20,10))
for i, match_id in enumerate(data.groupby(['game_mode']).first()['matchId'].values):
    plt.subplot(2, 4, i + 1)
    sns.distplot(train[train['matchId'] == match_id]['players_in_team']).set_title(f'game_mode: {i}')
plt.show()
