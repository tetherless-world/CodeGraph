#!/usr/bin/env python
# coding: utf-8

# In this notebook I'll try to implement the scoring metric. I am still not 100% sure about my interpretation, so please leave me a comment if you find something wrong.
# 
# -------------------------------

# # Mean Average Precision (MAP)
# Submissions are evaluated according to the Mean Average Precision @ 5 (MAP@5):
# 
# $$MAP@5 = {1 \over U} \sum_{u=1}^{U} \sum_{k=1}^{min(n,5)}P(k)$$
# 
# where `U` is the number of images, `P(k)` is the precision at cutoff `k` and `n` is the number of predictions per image.
# 

# ## Precision
# 
# 
# Precision (also called positive predictive value) is the fraction of relevant instances among the retrieved instances.
# 
# In a classification task, the precision for a class is the number of true positives (i.e. the number of items correctly labeled as belonging to the positive class) divided by the total number of elements labeled as belonging to the positive class (i.e. the sum of true positives and false positives, which are items incorrectly labeled as belonging to the class).
# 
# 
# $$ P = { \#\ of\ correct\ predictions\over \#\ of\ all\ predictions  } = {TP \over (TP + FP)}$$
# 

# ## Precision @k
# Precision at cutoff `k`, `P(k)`, is simply the precision calculated by considering only the subset of your predictions from rank 1 through `k`.
# 
# For example:
# 
# | true  | predicted   | k  | P(k) |
# |:-:|:-:|:-:|:-:|
# | [x]  | [x, ?, ?, ?, ?]   | 1  | 1.0  |
# | [x]  | [?, x, ?, ?, ?]   | 1  | 0.0  |
# | [x]  | [?, x, ?, ?, ?]   | 2  | $$1\over2$$  |
# | [x]  | [?, ?, x, ?, ?]   | 2  | 0.0  |
# | [x]  | [?, ?, x, ?, ?]   | 3  | $$1\over3$$  |
# 
# where `x` is the correct and `?` is incorrect prediction. 

# ## Precision @5 per image
# I think the evaluation metric in the competition's description is a bit confusing. According to @inversion's [answer](https://www.kaggle.com/c/humpback-whale-identification/discussion/73303#431164) in [this discussion](https://www.kaggle.com/c/humpback-whale-identification/discussion/73303):
# > the calculation would stop after the first occurrence of the correct whale, so `P(1) = 1`. So, a prediction that is `correct` `incorrect` `incorrect` `incorrect` `incorrect` also scores `1`.
# 
# So we don't have to sum up to 5, only up to the first correct answer. In this competition there is only one correct (`TP`) answer per image, so the possible precision scores per image are either `0` or `P(k)=1/k`.
# 
# | true  | predicted   | k  | Image score |
# |:-:|:-:|:-:|:-:|:-:|
# | [x]  | [x, ?, ?, ?, ?]   | 1  | 1.0  |
# | [x]  | [?, x, ?, ?, ?]   | 2  | 0 + 1/2 = 0.5 |
# | [x]  | [?, ?, x, ?, ?]   | 3  | 0/1 + 0/2 + 1/3  = 0.33 |
# | [x]  | [?, ?, ?, x, ?]   | 4  | 0/1 + 0/2 + 0/3 + 1/4  = 0.25 |
# | [x]  | [?, ?, ?, ?, x]   | 5  | 0/1 + 0/2 + 0/3 + 0/4 + 1/5  = 0.2 |
# | [x]  | [?, ?, ?, ?, ?]   | 5  | 0/1 + 0/2 + 0/3 + 0/4 + 0/5  = 0.0 |
# 
# where `x` is the correct and `?` is incorrect prediction. 
# 
# 

# ## Leaderboard score
# The final score is simply the average over the scores of the images.

# # Implementation

# In[ ]:


import numpy as np
import pandas as pd

# In[ ]:


def map_per_image(label, predictions):
    """Computes the precision score of one image.

    Parameters
    ----------
    label : string
            The true label of the image
    predictions : list
            A list of predicted elements (order does matter, 5 predictions allowed per image)

    Returns
    -------
    score : double
    """    
    try:
        return 1 / (predictions[:5].index(label) + 1)
    except ValueError:
        return 0.0

def map_per_set(labels, predictions):
    """Computes the average over multiple images.

    Parameters
    ----------
    labels : list
             A list of the true labels. (Only one true label per images allowed!)
    predictions : list of list
             A list of predicted elements (order does matter, 5 predictions allowed per image)

    Returns
    -------
    score : double
    """
    return np.mean([map_per_image(l, p) for l,p in zip(labels, predictions)])


# In[ ]:


#                   (true, [predictions])
assert map_per_image('x', []) == 0.0
assert map_per_image('x', ['y']) == 0.0
assert map_per_image('x', ['x']) == 1.0
assert map_per_image('x', ['x', 'y', 'z']) == 1.0
assert map_per_image('x', ['y', 'x']) == 0.5
assert map_per_image('x', ['y', 'x', 'x']) == 0.5
assert map_per_image('x', ['y', 'z']) == 0.0
assert map_per_image('x', ['y', 'z', 'x']) == 1/3
assert map_per_image('x', ['y', 'z', 'a', 'b', 'c']) == 0.0
assert map_per_image('x', ['x', 'z', 'a', 'b', 'c']) == 1.0
assert map_per_image('x', ['y', 'z', 'a', 'b', 'x']) == 1/5
assert map_per_image('x', ['y', 'z', 'a', 'b', 'c', 'x']) == 0.0

assert map_per_set(['x'], [['x', 'y']]) == 1.0
assert map_per_set(['x', 'z'], [['x', 'y'], ['x', 'y']]) == 1/2
assert map_per_set(['x', 'z'], [['x', 'y'], ['x', 'y', 'z']]) == 2/3
assert map_per_set(['x', 'z', 'k'], [['x', 'y'], ['x', 'y', 'z'], ['a', 'b', 'c', 'd', 'e']]) == 4/9

# In[ ]:


train_df = pd.read_csv("../input/train.csv")
train_df.head()

# In[ ]:


labels = train_df['Id'].values
labels

# In[ ]:


# 5 most common Id
# sample_pred = train_df['Id'].value_counts().nlargest(5).index.tolist()
sample_pred = ['new_whale', 'w_23a388d', 'w_9b5109b', 'w_9c506f6', 'w_0369a5c']
predictions = [sample_pred for i in range(len(labels))]
sample_pred

# In[ ]:


map_per_set(labels, predictions)

# **Thanks for reading.** 
