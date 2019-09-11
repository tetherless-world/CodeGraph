#!/usr/bin/env python
# coding: utf-8

# # Introduction
# Recommender Systems:
# 1. User Based Recommender Systems
# 1. Item Based Recommender Systems
# 
# <br>What is recommender System?
#    * Based on previous(past) behaviours, it predicts the likelihood that a user would prefer an item.
#    * For example, Netflix uses recommendation system. It suggest people new movies according to their past activities that are like watching and voting movies.
#    * The purpose of recommender systems is recommending new things that are not seen before from people.
#    
# <br>
# 1. User Based Collaborative Filtering
#     * Collaborative filtering is making recommend according to combination of your experience and experiences of other people.
#     * First we need to make user vs item matrix.
#         * Each row is users and each columns are items like movie, product or websites
#     * Secondly, computes similarity scores between users.
#         * Each row is users and each row is vector.
#         * Compute similarity of these rows (users).
#     * Thirdly, find users who are similar to you based on past behaviours
#     * Finally, it suggests that you are not experienced before.
#     * Lets make an example of user based collaborative filtering
#         * Think that there are two people
#         * First one watched 2 movies that are lord of the rings and hobbit
#         * Second one watched only lord of the rings movie
#         * User based collaborative filtering computes similarity of these two people and sees both are watched a lord of the rings.
#         * Then it recommends hobbit movie to second one as it can be seen picture
#         *<a href="https://ibb.co/droZMy"><img src="https://preview.ibb.co/feq3EJ/resim_a.jpg" alt="resim_a" border="0"></a>
#         
#     * User based collaborative filtering has some problems
#         * In this system, each row of matrix is user. Therefore, comparing and finding similarity between of them is computationaly hard and spend too much computational power.
#         * Also, habits of people can be changed. Therefore making correct and useful recommendation can be hard in time.
#     * In order to solve these problems, lets look at another recommender system that is item based collaborative filtering
# 1. Item Based Collaborative Filtering
#     * In this system, instead of finding relationship between users, used items like movies or stuffs are compared with each others.
#     * In user based recommendation systems, habits of users can be changed. This situation makes hard to recommendation. However, in item based recommendation systems, movies or stuffs does not change. Therefore recommendation is easier.
#     * On the other hand, there are almost 7 billion people all over the world. Comparing people increases the computational power. However, if items are compared, computational power is less.
#     * In item based recommendation systems, we need to make user vs item matrix that we use also in user based recommender systems.
#         * Each row is user and each column is items like movie, product or websites.
#         * However, at this time instead of calculating similarity between rows, we need to calculate similarity between columns that are items like movies or stuffs.
#     * Lets look at how it is works.
#         * Firstly, there are similarities between lord of the rings and hobbit movies because both are liked by three different people. There is a similarity point between these two movies.
#         * If the similarity is high enough, we can recommend hobbit to other people who only watched lord of the rings movie as it can be seen in figure below.
#         *<a href="https://imgbb.com/"><img src="https://image.ibb.co/maEQdd/resim_b.jpg" alt="resim_b" border="0"></a>
# 
# 
# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

# In[ ]:


# import movie data set and look at columns
movie = pd.read_csv("../input/movie.csv")
movie.columns

# In[ ]:


# what we need is that movie id and title
movie = movie.loc[:,["movieId","title"]]
movie.head(10)

# In[ ]:


# import rating data and look at columsn
rating = pd.read_csv("../input/rating.csv")
rating.columns

# In[ ]:


# what we need is that user id, movie id and rating
rating = rating.loc[:,["userId","movieId","rating"]]
rating.head(10)

# In[ ]:


# then merge movie and rating data
data = pd.merge(movie,rating)

# In[ ]:


# now lets look at our data 
data.head(10)

# * As it can be seen data frame that is above, we have 4 features that are movie id, title user id and rating
# * According to these data frame, we will make item based recommendation system
# * Lets look at shape of the data. The number of sample in data frame is 20 million that is too much. There can be problem in kaggle even if their own desktop ide's like spyder or pycharm.
# * Therefore, in order to learn item based recommendation system lets use 1 million of sample in data.

# In[ ]:


data.shape

# In[ ]:


data = data.iloc[:1000000,:]

# In[ ]:


# lets make a pivot table in order to make rows are users and columns are movies. And values are rating
pivot_table = data.pivot_table(index = ["userId"],columns = ["title"],values = "rating")
pivot_table.head(10)

# * As it can be seen from table above, rows are users, columns are movies and values are ratings
# * For example user 11 gives 3.5 rating to movie "Ace Ventura: When Nature Calls (1995)" and gives 3.0 rating to movie "Bad Boys (1995)".
# * Now lets make a scenario, we have movie web site and "Bad Boys (1995)" movie are watched and rated by people. The question is that which movie do we recommend these people who watched "Bad Boys (1995)" movie.
# * In order to answer this question we will find similarities between "Bad Boys (1995)" movie and other movies.

# In[ ]:


movie_watched = pivot_table["Bad Boys (1995)"]
similarity_with_other_movies = pivot_table.corrwith(movie_watched)  # find correlation between "Bad Boys (1995)" and other movies
similarity_with_other_movies = similarity_with_other_movies.sort_values(ascending=False)
similarity_with_other_movies.head()

# * It can be concluded that we need to recommend "Headless Body in Topless Bar (1995)" movie to people who watched "Bad Boys (1995)".
# * On the other hand even if we do not consider, number of rating for each movie is also important.

# # Conclusion
# What we learn is that
# * User based recommentation systems
# * Item based recommentation systems
# * How to find correlation or similarity between two vectors
# * Then we make very basic movie recommendation system.
# * **If you have any question I will be happy to hear it.**

# In[ ]:




# In[ ]:



