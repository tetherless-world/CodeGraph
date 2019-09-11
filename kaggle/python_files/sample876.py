#!/usr/bin/env python
# coding: utf-8

# [Hyun woo kim]  - 2018-11-08
# 
# I changed the kernel to match the changed dataset. The changes are summarized in the kernel below.
# https://www.kaggle.com/chocozzz/updated-what-is-difference-before-data

# ![](https://storage.googleapis.com/kaggle-media/competitions/PUBG/PUBG%20Inlay.jpg)
# 
# ## Notebook Outline
# - Competition Description
# - Game Description
# - Variable Description
# - Simple EDA 
# - Feature Engineering
# - LightGBM

# ## 1. Competiton Description
# 
# Description : So, where we droppin' boys and girls?
# 
# Battle Royale-style video games have taken the world by storm. 100 players are dropped onto an island empty-handed and must explore, scavenge, and eliminate other players until only one is left standing, all while the play zone continues to shrink.
# 
# PlayerUnknown's BattleGrounds (PUBG) has enjoyed massive popularity. With over 50 million copies sold, it's the fifth best selling game of all time, and has millions of active monthly players.
# 
# The team at [PUBG](https://www.pubg.com/) has made official game data available for the public to explore and scavenge outside of "The Blue Circle." This competition is not an official or affiliated PUBG site - Kaggle collected data made possible through the [PUBG Developer API.](https://developer.pubg.com/)
# 
# You are given over 65,000 games' worth of anonymized player data, split into training and testing sets, and asked to predict final placement from final in-game stats and initial player ratings.
# 
# What's the best strategy to win in PUBG? Should you sit in one spot and hide your way into victory, or do you need to be the top shot? Let's let the data do the talking!

# ## 2. Game Description

# I will try this game and know what it is but I will introduce the game simply for those who do not know. This game is a kind of Survival game that goes around a certain field and makes a fight. I have attached the game video to the link below. So it would be better to understand data
# 
# Video Link : https://www.youtube.com/watch?v=rmyyeqQpHQc

# In[ ]:


import pandas as pd #Analysis 
import matplotlib.pyplot as plt #Visulization
import seaborn as sns #Visulization
import numpy as np #Analysis 
from scipy.stats import norm #Analysis 
from sklearn.preprocessing import StandardScaler #Analysis 
from scipy import stats #Analysis 
import warnings 
warnings.filterwarnings('ignore')
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import plotly.figure_factory as ff
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
import gc

# In[ ]:


#If you run all dataset, you change debug False
debug = False
if debug == True:
    df_train = pd.read_csv('../input/train_V2.csv', nrows=10000)
    df_test  = pd.read_csv('../input/test_V2.csv')
else:
    df_train = pd.read_csv('../input/train_V2.csv')
    df_test  = pd.read_csv('../input/test_V2.csv')

# In[ ]:


print("Train : ",df_train.shape)
print("Test : ",df_test.shape)

# In[ ]:


df_train.head()

# #### reduce memory

# In[ ]:


# Memory saving function credit to https://www.kaggle.com/gemartin/load-data-reduce-memory-usage
def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    #start_mem = df.memory_usage().sum() / 1024**2
    #print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    #end_mem = df.memory_usage().sum() / 1024**2
    #print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    #print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df
df_train = reduce_mem_usage(df_train)
df_test = reduce_mem_usage(df_test)

# ## 3. Variable Description

# ### What is difference Id, groupId, matchId ?

# In the data description,   
# - matchId - ID to identify match. There are no matches that are in both the training and testing set.
# - groupId - ID to identify a group within a match. If the same group of players plays in different matches, they will have a different groupId each time. 
# 

# In[ ]:


df_train[df_train['groupId']=='4d4b580de459be']

# In[ ]:


len(df_train[df_train['matchId']=='a10357fd1a4a91'])

# Consider the example above. In both cases Id is different, but groupId and matchId are the same. To illustrate this, a person A with an Id 7f96b2f878858a and a person B with an ID 7516514fbd1091 are friends and have a team together (groupId). Then the same match is done, so you can assume that they entered the game with the same matchId.
# 
# To put it another way, Battlegrounds (PBUGs) have a total about 100 people per game. These 100 players have the same matchId. Among them, groupId are same as 4d4b580de459be, so you can think that they are friends  and joined the team and played together. (There are about 100 people, not necessarily 100 people.)

# In[ ]:


temp = df_train[df_train['matchId']=='a10357fd1a4a91']['groupId'].value_counts().sort_values(ascending=False)
#print("Total number of states : ",len(temp))
trace = go.Bar(
    x = temp.index,
    y = (temp)
)
data = [trace]
layout = go.Layout(
    title = "GroupId of Match Id: a10357fd1a4a91",
    xaxis=dict(
        title='groupId',
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
    ),
    yaxis=dict(
        title='Count of groupId of type of MatchId a10357fd1a4a91',
        titlefont=dict(
            size=16,
            color='rgb(107, 107, 107)'
        ),
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
)
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='schoolStateNames')

# **Note :**  You can see something strange in value counts. Four people are maximum team member and I do not know what it means more than four people. 
# 
# **michaelapers** commented 
# 
# I do want to get ahead of one inevitable question. You will notice that there are frequently more than the supposed max number of players in a group regardless of mode. For example, you might have more than 4 people in a group with matchType == 'squad'. This is caused by disconnections in the game. When disconnections occur, players of multiple groups are stored in the API's database as having the same final placement. This has the consequence that when I make the groupId feature from final placements, we have too large of groups. Please take groupId to mean "players that have the same final placement" and not "players that definitely played in a group together." 
# 
# https://www.kaggle.com/c/pubg-finish-placement-prediction/discussion/68965#406275

# ### Data description detail

# This game is simple. Pick up your weapons, walk around, kill enemies and survive until the end. So if you look at the variables, kill and ride will come out and if you stay alive you will win.

# - assists : The assists means that i don't kill enemy but help kill enemy. So when you look at the variable, there is also a kill.
# In other words, if I kill the enemy? `kill +1`. but if I did not kill the enemy but helped kill the enemy?` assists + 1.`

# In[ ]:


temp = df_train['assists'].value_counts().sort_values(ascending=False)

#print("Total number of states : ",len(temp))
trace = go.Bar(
    x = temp.index,
    y = (temp)
)
data = [trace]
layout = go.Layout(
    title = "",
    xaxis=dict(
        title='assists',
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
    ),
    yaxis=dict(
        title='Count of assists',
        titlefont=dict(
            size=16,
            color='rgb(107, 107, 107)'
        ),
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
)
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='schoolStateNames')

# ### Related variables with kills

# - kills : Number of enemy players killed.

# In[ ]:


temp = df_train['kills'].value_counts().sort_values(ascending=False)

#print("Total number of states : ",len(temp))
trace = go.Bar(
    x = temp.index,
    y = (temp)
)
data = [trace]
layout = go.Layout(
    title = "",
    xaxis=dict(
        title='kills',
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
    ),
    yaxis=dict(
        title='Count of kills',
        titlefont=dict(
            size=16,
            color='rgb(107, 107, 107)'
        ),
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
)
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='schoolStateNames')

# - killStreaks : Max number of enemy players killed in a short amount of time.

# In[ ]:


temp = df_train['killStreaks'].value_counts().sort_values(ascending=False)

#print("Total number of states : ",len(temp))
trace = go.Bar(
    x = temp.index,
    y = (temp)
)
data = [trace]
layout = go.Layout(
    title = "",
    xaxis=dict(
        title='killStreaks',
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
    ),
    yaxis=dict(
        title='Count of killStreaks',
        titlefont=dict(
            size=16,
            color='rgb(107, 107, 107)'
        ),
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
)
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='schoolStateNames')

# - roadKills : Number of kills while in a vehicle.

# In[ ]:


temp = df_train['roadKills'].value_counts().sort_values(ascending=False)

#print("Total number of states : ",len(temp))
trace = go.Bar(
    x = temp.index,
    y = (temp)
)
data = [trace]
layout = go.Layout(
    title = "",
    xaxis=dict(
        title='roadKills',
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
    ),
    yaxis=dict(
        title='Count of roadKills',
        titlefont=dict(
            size=16,
            color='rgb(107, 107, 107)'
        ),
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
)
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='schoolStateNames')

# In[ ]:


df_train['roadKills'].value_counts()

# I've explained it in more detail below, but it's hard to kill if you're in a car. So I do not understand the number 42. If you die in a car, there are usually
# 
# - The player plays the game well. ( I don't it... haha )

# - teamKills : Number of times this player killed a teammate.

# In[ ]:


temp = df_train['teamKills'].value_counts().sort_values(ascending=False)

#print("Total number of states : ",len(temp))
trace = go.Bar(
    x = temp.index,
    y = (temp),
)
data = [trace]
layout = go.Layout(
    title = "",
    xaxis=dict(
        title='teamKills',
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
    ),
    yaxis=dict(
        title='Count of teamKills',
        titlefont=dict(
            size=16,
            color='rgb(107, 107, 107)'
        ),
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
)
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='schoolStateNames')

# In[ ]:


df_train['teamKills'].value_counts()

# Rationally, teamkill is hard to understand. Still, I will explain the case I saw while playing YouTube or the game.
# 
# - A team member is a friend and kills for fun.
# - The team member is not  played well. so killing the member. 
# - In the case of a squad, even if they are not friends, they automatically become a team. At that time, I saw that my nationality was different or I was not a friend and I was killed.
# - Only act irrational for fun.

# - longestKill : Longest distance between player and player killed at time of death. This may be misleading, as downing a player and driving away may lead to a large longestKill stat.

# In[ ]:


#histogram
f, ax = plt.subplots(figsize=(18, 8))
sns.distplot(df_train['longestKill'])

# There are many kinds of guns in the game. So, as you can see in the picture below, the number of times you pick up a gun is several times.

# In[ ]:


temp = df_train['weaponsAcquired'].value_counts().sort_values(ascending=False)

#print("Total number of states : ",len(temp))
trace = go.Bar(
    x = temp.index,
    y = (temp),
)
data = [trace]
layout = go.Layout(
    title = "",
    xaxis=dict(
        title='weaponsAcquired',
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
    ),
    yaxis=dict(
        title='Count of weaponsAcquired',
        titlefont=dict(
            size=16,
            color='rgb(107, 107, 107)'
        ),
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
)
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='schoolStateNames')

# Among the guns are special guns aimed at fighters at close range, and there are sniper guns that are specially designed to match enemies at long distances (The range is fixed for each gun ). So over a certain distance, all were shot by a sniper rifle.

# ### headshotKills - not knocked.
# HeadshotKills means that a bullet hit his head and  he died `right away`.  it is important that he died right away. 
# - DBNOs : Number of enemy players knocked.
# 
# DBNOs variable means Number of enemy players `knocked`. Knocked is not dead, but can not act. so if you are knocked, your colleagues can save you (`revives` variable) but if you died? you don't save... :(

# - Died picture
# ![](https://i.ytimg.com/vi/0qSFX2SBUho/maxresdefault.jpg)
# 
# 
# - Knocked picture
# ![](https://i.ytimg.com/vi/EgLRYtUqxn4/maxresdefault.jpg)
# 
# [original picture link - above](https://www.google.co.kr/url?sa=i&rct=j&q=&esrc=s&source=images&cd=&cad=rja&uact=8&ved=2ahUKEwi2pp7i4-_dAhXGE7wKHdsJBNQQjRx6BAgBEAU&url=https%3A%2F%2Fwww.youtube.com%2Fwatch%3Fv%3D0qSFX2SBUho&psig=AOvVaw1JcDsctlYqqKvW_IyzuEue&ust=1538845327420883)
# 
# [original picture link - below](https://www.google.co.kr/url?sa=i&rct=j&q=&esrc=s&source=images&cd=&cad=rja&uact=8&ved=2ahUKEwj2ybrJ4-_dAhUJwbwKHcfCDlIQjRx6BAgBEAU&url=https%3A%2F%2Fwww.youtube.com%2Fwatch%3Fv%3DEgLRYtUqxn4&psig=AOvVaw27IBxucFCW7i3Dd55GSlSM&ust=1538845290684243)

# In[ ]:


temp = df_train['headshotKills'].value_counts().sort_values(ascending=False)

#print("Total number of states : ",len(temp))
trace = go.Bar(
    x = temp.index,
    y = (temp),
)
data = [trace]
layout = go.Layout(
    title = "",
    xaxis=dict(
        title='headshotKills',
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
    ),
    yaxis=dict(
        title='Count of headshotKills',
        titlefont=dict(
            size=16,
            color='rgb(107, 107, 107)'
        ),
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
)
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='schoolStateNames')

# In[ ]:


df_train['headshotKills'].value_counts()

# In[ ]:


temp = df_train['DBNOs'].value_counts().sort_values(ascending=False)

#print("Total number of states : ",len(temp))
trace = go.Bar(
    x = temp.index,
    y = (temp),
)
data = [trace]
layout = go.Layout(
    title = "",
    xaxis=dict(
        title='DBNOs',
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
    ),
    yaxis=dict(
        title='Count of DBNOs',
        titlefont=dict(
            size=16,
            color='rgb(107, 107, 107)'
        ),
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
)
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='schoolStateNames')

# 

# ### what is difference boost vs heal?
# Both of these variables are items that restore health. but the boosts immediately show the effect, and the heals show the effect slowly. 
# 
# - boosts : Number of boost items used.
# 
# ![boosts](https://cdn.appuals.com/wp-content/uploads/2017/10/Battlegrounds-Healing-and-Boost-Items-Guide-5.png)
# 
# 
# - heals : Number of healing items used.
# 
# ![heals](https://cdn.appuals.com/wp-content/uploads/2017/10/Battlegrounds-Healing-and-Boost-Items-Guide-1.png)
# 
# [original picture Link](https://appuals.com/playerunknowns-battlegrounds-healing-boost-items-guide/)
# 

# In[ ]:


temp = df_train['boosts'].value_counts().sort_values(ascending=False)

#print("Total number of states : ",len(temp))
trace = go.Bar(
    x = temp.index,
    y = (temp),
)
data = [trace]
layout = go.Layout(
    title = "",
    xaxis=dict(
        title='boosts',
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
    ),
    yaxis=dict(
        title='Count of boosts',
        titlefont=dict(
            size=16,
            color='rgb(107, 107, 107)'
        ),
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
)
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='schoolStateNames')

# In[ ]:


temp = df_train['heals'].value_counts().sort_values(ascending=False)

#print("Total number of states : ",len(temp))
trace = go.Bar(
    x = temp.index,
    y = (temp),
)
data = [trace]
layout = go.Layout(
    title = "",
    xaxis=dict(
        title='heals',
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
    ),
    yaxis=dict(
        title='Count of heals',
        titlefont=dict(
            size=16,
            color='rgb(107, 107, 107)'
        ),
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
)
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='schoolStateNames')

# ### others

# - damageDealt : Total damage dealt. Note: Self inflicted damage is subtracted. If it is not headshot, it does not die in one shot. So restores health by using` boosts` or `heals`. `damageDealt` means how many bullets have ever been hit.
# 

# In[ ]:


#histogram
f, ax = plt.subplots(figsize=(8, 6))
sns.distplot(df_train['damageDealt'])

# - revives : Number of times this player revived teammates. I said above,  if you knock, your teammates can save you. If a team member is saved, the revives are +1.

# In[ ]:


f, ax = plt.subplots(figsize=(8, 6))
df_train['revives'].value_counts().sort_values(ascending=False).plot.bar()
plt.show()

# - walkDistance : Total distance traveled on foot measured in meters.
# 
# ![](https://i.ytimg.com/vi/Ig_KOUqrSH8/maxresdefault.jpg)
# 
# [original picture link](https://www.google.co.kr/url?sa=i&rct=j&q=&esrc=s&source=images&cd=&cad=rja&uact=8&ved=2ahUKEwignNbp4u_dAhWFxbwKHVGICYAQjRx6BAgBEAU&url=https%3A%2F%2Fwww.youtube.com%2Fwatch%3Fv%3DIg_KOUqrSH8&psig=AOvVaw06O9ien8kWzTdVEG0Fki7e&ust=1538845052210260)

# In[ ]:


#histogram
f, ax = plt.subplots(figsize=(18, 8))
sns.distplot(df_train['walkDistance'])

# - rideDistance : Total distance traveled in vehicles measured in meters. The PUBG game is so wide that it is hard to walk around. So I ride around VEHICLE as shown in the picture below.

# ![](http://file.gamedonga.co.kr/files/2017/04/04/pkbg089.jpg)
# 
# [original picture link](https://www.google.co.kr/url?sa=i&rct=j&q=&esrc=s&source=images&cd=&cad=rja&uact=8&ved=2ahUKEwjl7p784u_dAhVG6LwKHceWB3wQjRx6BAgBEAU&url=http%3A%2F%2Fgame.donga.com%2F86877%2F&psig=AOvVaw01B4xxH_3KBE8QpqBsFwmH&ust=1538845127758218)

# In[ ]:


#histogram
f, ax = plt.subplots(figsize=(18, 8))
sns.distplot(df_train['rideDistance'])

# - swimDistance: Total distance traveled by swimming measured in meters. The map is wide, and there are some kind of river. 
# 
# ![](https://i.ytimg.com/vi/heUNpk8XaRU/maxresdefault.jpg)
# 
# [original picture link](https://www.google.co.kr/url?sa=i&rct=j&q=&esrc=s&source=images&cd=&cad=rja&uact=8&ved=2ahUKEwjT1e3d2e_dAhWMabwKHYh2CsAQjRx6BAgBEAU&url=https%3A%2F%2Fwww.youtube.com%2Fwatch%3Fv%3DheUNpk8XaRU&psig=AOvVaw2FnVjz97_tbgYfeHuGycut&ust=1538842587462044)
# 

# - vehicleDestroys: Number of vehicles destroyed.

# In[ ]:


temp = df_train['vehicleDestroys'].value_counts().sort_values(ascending=False)

#print("Total number of states : ",len(temp))
trace = go.Bar(
    x = temp.index,
    y = (temp),
)
data = [trace]
layout = go.Layout(
    title = "",
    xaxis=dict(
        title='vehicleDestroys',
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
    ),
    yaxis=dict(
        title='Count of vehicleDestroys',
        titlefont=dict(
            size=16,
            color='rgb(107, 107, 107)'
        ),
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
)
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='schoolStateNames')

# In[ ]:


df_train['vehicleDestroys'].value_counts()

# If you look at the above values, you will rarely destroy a vehicle. In fact, it is very natural. It is difficult to destroy the car. And there is no profit by destroying the car. Even so, the destruction of a car can be thought of in the following sense. 
# 
# - The enemy was in the car and shot the car.
# - The enemy hid the car in cover and shot the car.
# - He broke a car with no meaning.
# 
# The third reason is very important. When you play games, you can meet a lot of strange people.

# - weaponsAcquired : Number of weapons picked up. This game is a using gun , but it does not give a gun from the beginning. So you have to go around the map and look for weapons. In the process, you can also have heals, boosts and vehicles.
# 
# ![](https://t1.daumcdn.net/cfile/tistory/9904AA3359B201AA0C)
# 
# [original picture link](https://www.google.co.kr/url?sa=i&rct=j&q=&esrc=s&source=images&cd=&cad=rja&uact=8&ved=2ahUKEwjtq9eM4-_dAhUK7bwKHeOjBPMQjRx6BAgBEAU&url=http%3A%2F%2Fhogod.tistory.com%2F23&psig=AOvVaw1SqPa1ImkjsfcThfY5nfgW&ust=1538845160945955)

# In[ ]:


temp = df_train['weaponsAcquired'].value_counts().sort_values(ascending=False)

#print("Total number of states : ",len(temp))
trace = go.Bar(
    x = temp.index,
    y = (temp),
)
data = [trace]
layout = go.Layout(
    title = "",
    xaxis=dict(
        title='weaponsAcquired',
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
    ),
    yaxis=dict(
        title='Count of weaponsAcquired',
        titlefont=dict(
            size=16,
            color='rgb(107, 107, 107)'
        ),
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
)
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='schoolStateNames')

# ## 4. Simple EDA

# ### 4.1 Missing Values

# In[ ]:


#missing data
total = df_train.isnull().sum().sort_values(ascending=False)
percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)

#histogram
#missing_data = missing_data.head(20)
percent_data = percent.head(20)
percent_data.plot(kind="bar", figsize = (8,6), fontsize = 10)
plt.xlabel("Columns", fontsize = 20)
plt.ylabel("Count", fontsize = 20)
plt.title("Total Missing Value (%) in Train", fontsize = 20)

# In[ ]:


#missing data
total = df_test.isnull().sum().sort_values(ascending=False)
percent = (df_test.isnull().sum()/df_test.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)

#histogram
#missing_data = missing_data.head(20)
percent_data = percent.head(20)
percent_data.plot(kind="bar", figsize = (8,6), fontsize = 10)
plt.xlabel("Columns", fontsize = 20)
plt.ylabel("Count", fontsize = 20)
plt.title("Total Missing Value (%) in Test", fontsize = 20)

# There is not Missing Value

# ### 4.2 winPlacePerc (Target Value)

# In[ ]:


#winPlacePerc correlation matrix
k = 10 #number of variables for heatmap
corrmat = df_train.corr() 
cols = corrmat.nlargest(k, 'winPlacePerc').index # nlargest : Return this many descending sorted values
cm = np.corrcoef(df_train[cols].values.T) # correlation 
sns.set(font_scale=1.25)
f, ax = plt.subplots(figsize=(8, 6))
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 8}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()

# ### 4.2.2 others

# In[ ]:


df_train.plot(x="walkDistance",y="winPlacePerc", kind="scatter", figsize = (8,6))

# In[ ]:


f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x='boosts', y="winPlacePerc", data=df_train)
fig.axis(ymin=0, ymax=1);

# In[ ]:


df_train.plot(x="weaponsAcquired",y="winPlacePerc", kind="scatter", figsize = (8,6))

# In[ ]:


df_train.plot(x="damageDealt",y="winPlacePerc", kind="scatter", figsize = (8,6))

# In[ ]:


df_train.plot(x="heals",y="winPlacePerc", kind="scatter", figsize = (8,6))

# In[ ]:


df_train.plot(x="longestKill",y="winPlacePerc", kind="scatter", figsize = (8,6))

# In[ ]:


df_train.plot(x="kills",y="winPlacePerc", kind="scatter", figsize = (8,6))

# In[ ]:


f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x='killStreaks', y="winPlacePerc", data=df_train)
fig.axis(ymin=0, ymax=1);

# In[ ]:


f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x='assists', y="winPlacePerc", data=df_train)
fig.axis(ymin=0, ymax=1);

# ### what is changed ?
# https://www.kaggle.com/chocozzz/updated-what-is-difference-before-data

# ## 5. Feature Engineering

# ### 5.1 headshot rate

# In[ ]:


df_train = df_train[df_train['Id']!='f70c74418bb064']

# In[ ]:


headshot = df_train[['kills','winPlacePerc','headshotKills']]
headshot['headshotrate'] = headshot['kills'] / headshot['headshotKills']

# In[ ]:


headshot.corr()

# In[ ]:


del headshot

# In[ ]:


df_train['headshotrate'] = df_train['kills']/df_train['headshotKills']
df_test['headshotrate'] = df_test['kills']/df_test['headshotKills']

# ### 5.2 killStreak rate

# In[ ]:


killStreak = df_train[['kills','winPlacePerc','killStreaks']]
killStreak['killStreakrate'] = killStreak['killStreaks']/killStreak['kills']
killStreak.corr()

# - minus killStreakrate is better than killStreaks. so i delete killStreaks and use killStreakrate

# ### 5.3 health Items

# In[ ]:


healthitems = df_train[['heals','winPlacePerc','boosts']]
healthitems['healthitems'] = healthitems['heals'] + healthitems['boosts']
healthitems.corr()

# In[ ]:


del healthitems

# This is a bad variable. so don't use it

# ### 5.4 kills & assists 

# In[ ]:


kills = df_train[['assists','winPlacePerc','kills']]
kills['kills_assists'] = (kills['kills'] + kills['assists'])
kills.corr()

# so it is good. i use kills_assists and drop kills because of high corr

# In[ ]:


del df_train,df_test;
gc.collect()

# ### 5.5 statisticals feature
# I am using features of 2 kernels   
# - anycode : https://www.kaggle.com/anycode/simple-nn-baseline-3  
# - harsit : https://www.kaggle.com/harshitsheoran/mlp-and-fe
# 

# In[ ]:


def feature_engineering(is_train=True,debug=True):
    test_idx = None
    if is_train: 
        print("processing train.csv")
        if debug == True:
            df = pd.read_csv('../input/train_V2.csv', nrows=10000)
        else:
            df = pd.read_csv('../input/train_V2.csv')           

        df = df[df['maxPlace'] > 1]
    else:
        print("processing test.csv")
        df = pd.read_csv('../input/test_V2.csv')
        test_idx = df.Id
    
    # df = reduce_mem_usage(df)
    #df['totalDistance'] = df['rideDistance'] + df["walkDistance"] + df["swimDistance"]
    
    # df = df[:100]
    
    print("remove some columns")
    target = 'winPlacePerc'

    print("Adding Features")
 
    df['headshotrate'] = df['kills']/df['headshotKills']
    df['killStreakrate'] = df['killStreaks']/df['kills']
    df['healthitems'] = df['heals'] + df['boosts']
    df['totalDistance'] = df['rideDistance'] + df["walkDistance"] + df["swimDistance"]
    df['killPlace_over_maxPlace'] = df['killPlace'] / df['maxPlace']
    df['headshotKills_over_kills'] = df['headshotKills'] / df['kills']
    df['distance_over_weapons'] = df['totalDistance'] / df['weaponsAcquired']
    df['walkDistance_over_heals'] = df['walkDistance'] / df['heals']
    df['walkDistance_over_kills'] = df['walkDistance'] / df['kills']
    df['killsPerWalkDistance'] = df['kills'] / df['walkDistance']
    df["skill"] = df["headshotKills"] + df["roadKills"]

    df[df == np.Inf] = np.NaN
    df[df == np.NINF] = np.NaN
    
    print("Removing Na's From DF")
    df.fillna(0, inplace=True)

    
    features = list(df.columns)
    features.remove("Id")
    features.remove("matchId")
    features.remove("groupId")
    features.remove("matchType")
    
    # matchType = pd.get_dummies(df['matchType'])
    # df = df.join(matchType)    
    
    y = None
    
    
    if is_train: 
        print("get target")
        y = np.array(df.groupby(['matchId','groupId'])[target].agg('mean'), dtype=np.float64)
        features.remove(target)

    print("get group mean feature")
    agg = df.groupby(['matchId','groupId'])[features].agg('mean')
    agg_rank = agg.groupby('matchId')[features].rank(pct=True).reset_index()
    
    if is_train: df_out = agg.reset_index()[['matchId','groupId']]
    else: df_out = df[['matchId','groupId']]

    df_out = df_out.merge(agg.reset_index(), suffixes=["", ""], how='left', on=['matchId', 'groupId'])
    df_out = df_out.merge(agg_rank, suffixes=["_mean", "_mean_rank"], how='left', on=['matchId', 'groupId'])
    
    # print("get group sum feature")
    # agg = df.groupby(['matchId','groupId'])[features].agg('sum')
    # agg_rank = agg.groupby('matchId')[features].rank(pct=True).reset_index()
    # df_out = df_out.merge(agg.reset_index(), suffixes=["", ""], how='left', on=['matchId', 'groupId'])
    # df_out = df_out.merge(agg_rank, suffixes=["_sum", "_sum_rank"], how='left', on=['matchId', 'groupId'])
    
    # print("get group sum feature")
    # agg = df.groupby(['matchId','groupId'])[features].agg('sum')
    # agg_rank = agg.groupby('matchId')[features].agg('sum')
    # df_out = df_out.merge(agg.reset_index(), suffixes=["", ""], how='left', on=['matchId', 'groupId'])
    # df_out = df_out.merge(agg_rank.reset_index(), suffixes=["_sum", "_sum_pct"], how='left', on=['matchId', 'groupId'])
    
    print("get group max feature")
    agg = df.groupby(['matchId','groupId'])[features].agg('max')
    agg_rank = agg.groupby('matchId')[features].rank(pct=True).reset_index()
    df_out = df_out.merge(agg.reset_index(), suffixes=["", ""], how='left', on=['matchId', 'groupId'])
    df_out = df_out.merge(agg_rank, suffixes=["_max", "_max_rank"], how='left', on=['matchId', 'groupId'])
    
    print("get group min feature")
    agg = df.groupby(['matchId','groupId'])[features].agg('min')
    agg_rank = agg.groupby('matchId')[features].rank(pct=True).reset_index()
    df_out = df_out.merge(agg.reset_index(), suffixes=["", ""], how='left', on=['matchId', 'groupId'])
    df_out = df_out.merge(agg_rank, suffixes=["_min", "_min_rank"], how='left', on=['matchId', 'groupId'])
    
    print("get group size feature")
    agg = df.groupby(['matchId','groupId']).size().reset_index(name='group_size')
    df_out = df_out.merge(agg, how='left', on=['matchId', 'groupId'])
    
    print("get match mean feature")
    agg = df.groupby(['matchId'])[features].agg('mean').reset_index()
    df_out = df_out.merge(agg, suffixes=["", "_match_mean"], how='left', on=['matchId'])
    
    # print("get match type feature")
    # agg = df.groupby(['matchId'])[matchType.columns].agg('mean').reset_index()
    # df_out = df_out.merge(agg, suffixes=["", "_match_type"], how='left', on=['matchId'])
    
    print("get match size feature")
    agg = df.groupby(['matchId']).size().reset_index(name='match_size')
    df_out = df_out.merge(agg, how='left', on=['matchId'])
    
    df_out.drop(["matchId", "groupId"], axis=1, inplace=True)

    X = df_out
    
    feature_names = list(df_out.columns)

    del df, df_out, agg, agg_rank
    gc.collect()

    return X, y, feature_names, test_idx

# In[ ]:


x_train, y_train, train_columns, _ = feature_engineering(True,False)
x_test, _, _ , test_idx = feature_engineering(False,False)

# In[ ]:


x_train.shape

# In[ ]:


x_train = reduce_mem_usage(x_train)
x_test = reduce_mem_usage(x_test)

# ## 6. LightGBM

# In[ ]:


import os
import time
import gc
import warnings
warnings.filterwarnings("ignore")
# data manipulation

# model
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb

# ### Predict 

# In[ ]:


#excluded_features = []
#use_cols = [col for col in df_train.columns if col not in excluded_features]

train_index = round(int(x_train.shape[0]*0.7))
dev_X = x_train[:train_index] 
val_X = x_train[train_index:]
dev_y = y_train[:train_index] 
val_y = y_train[train_index:] 
gc.collect();

# custom function to run light gbm model
def run_lgb(train_X, train_y, val_X, val_y, x_test):
    params = {"objective" : "regression", "metric" : "mae", 'n_estimators':20000, 'early_stopping_rounds':200,
              "num_leaves" : 31, "learning_rate" : 0.05, "bagging_fraction" : 0.7,
               "bagging_seed" : 0, "num_threads" : 4,"colsample_bytree" : 0.7
             }
    
    lgtrain = lgb.Dataset(train_X, label=train_y)
    lgval = lgb.Dataset(val_X, label=val_y)
    model = lgb.train(params, lgtrain, valid_sets=[lgtrain, lgval], early_stopping_rounds=200, verbose_eval=1000)
    
    pred_test_y = model.predict(x_test, num_iteration=model.best_iteration)
    return pred_test_y, model

# Training the model #
pred_test, model = run_lgb(dev_X, dev_y, val_X, val_y, x_test)

# ### Below trick in https://www.kaggle.com/anycode/simple-nn-baseline-4 , very nice kernel [version41]
# LB 0.0011 better
# 
# ### Updated trick in https://www.kaggle.com/ceshine/a-simple-post-processing-trick-lb-0237-0204 [version42]
# LB 0.002x better

# In[ ]:


del dev_X, dev_y, val_X, val_y, x_test;
gc.collect()

# In[ ]:


df_sub = pd.read_csv("../input/sample_submission_V2.csv")
df_test = pd.read_csv("../input/test_V2.csv")
df_sub['winPlacePerc'] = pred_test
# Restore some columns
df_sub = df_sub.merge(df_test[["Id", "matchId", "groupId", "maxPlace", "numGroups"]], on="Id", how="left")

# Sort, rank, and assign adjusted ratio
df_sub_group = df_sub.groupby(["matchId", "groupId"]).first().reset_index()
df_sub_group["rank"] = df_sub_group.groupby(["matchId"])["winPlacePerc"].rank()
df_sub_group = df_sub_group.merge(
    df_sub_group.groupby("matchId")["rank"].max().to_frame("max_rank").reset_index(), 
    on="matchId", how="left")
df_sub_group["adjusted_perc"] = (df_sub_group["rank"] - 1) / (df_sub_group["numGroups"] - 1)

df_sub = df_sub.merge(df_sub_group[["adjusted_perc", "matchId", "groupId"]], on=["matchId", "groupId"], how="left")
df_sub["winPlacePerc"] = df_sub["adjusted_perc"]

# Deal with edge cases
df_sub.loc[df_sub.maxPlace == 0, "winPlacePerc"] = 0
df_sub.loc[df_sub.maxPlace == 1, "winPlacePerc"] = 1

# Align with maxPlace
# Credit: https://www.kaggle.com/anycode/simple-nn-baseline-4
subset = df_sub.loc[df_sub.maxPlace > 1]
gap = 1.0 / (subset.maxPlace.values - 1)
new_perc = np.around(subset.winPlacePerc.values / gap) * gap
df_sub.loc[df_sub.maxPlace > 1, "winPlacePerc"] = new_perc

# Edge case
df_sub.loc[(df_sub.maxPlace > 1) & (df_sub.numGroups == 1), "winPlacePerc"] = 0
assert df_sub["winPlacePerc"].isnull().sum() == 0

df_sub[["Id", "winPlacePerc"]].to_csv("submission_adjusted.csv", index=False)

# ### If there is any part of the data that you do not understand, I will answer with a comment. I will continue to add content.
