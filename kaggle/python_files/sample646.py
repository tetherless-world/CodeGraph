#!/usr/bin/env python
# coding: utf-8

# **Edit:** Quadratic Kappa Metric is the same as cohen kappa metric in Sci-kit learn @ sklearn.metrics.cohen_kappa_score when weights are set to 'Quadratic'. Thanks to Johannes for figuring that out. 

# ## What is Quadratic Weighted Kappa?  

# Submissions are scored based on the quadratic weighted kappa, which measures the agreement between two ratings. This metric typically varies from 0 (random agreement between raters) to 1 (complete agreement between raters). In the event that there is less agreement between the raters than expected by chance, the metric may go below 0. The quadratic weighted kappa is calculated between the scores which are expected/known and the predicted scores. <br>
# 
# 
# Results have 5 possible ratings, 0,1,2,3,4.  The quadratic weighted kappa is calculated as follows. First, an N x N histogram matrix O is constructed, such that Oi,j corresponds to the number of adoption records that have a rating of i (actual) and received a predicted rating j. An N-by-N matrix of weights, w, is calculated based on the difference between actual and predicted rating scores.
# 
# An N-by-N histogram matrix of expected ratings, E, is calculated, assuming that there is no correlation between rating scores.  This is calculated as the outer product between the actual rating's histogram vector of ratings and the predicted rating's histogram vector of ratings, normalized such that E and O have the same sum.
# 
# From these three matrices, the quadratic weighted kappa is calculated.

# ### Breaking down the formula into parts

# #### 5 step breakdown for Weighted Kappa Metric

# - First, create a multi class confusion matrix `O` between predicted and actual ratings. 
# - Second, construct a weight matrix `w` which calculates the weight between the actual and predicted ratings. 
# - Third, calculate `value_counts()` for each rating in preds and actuals. 
# - Fourth, calculate `E`, which is the outer product of two value_count vectors 
# - Fifth, normalise the `E` and `O` matrix
# - Caclulate, weighted kappa as per formula

# #### Each Step Explained

# **Step-1:** Under Step-1, we shall be calculating a `confusion_matrix` between the Predicted and Actual values. <a href="https://www.dataschool.io/simple-guide-to-confusion-matrix-terminology/">Here</a> is a great resource to know more about `confusion_matrix`. <br>
# **Step-2:** Under Step-2, under step-2 each element is weighted. Predictions that are further away from actuals are marked harshly than predictions that are closer to actuals. We will have a less score if our prediction is 5 and actual is 3 as compared to a prediction of 4 in the same case. <br>
# **Step-3:** We create two vectors, one for preds and one for actuals, which tells us how many values of each rating exist in both vectors. <br>
# **Step-4:**`E` is the Expected Matrix which is the outer product of the two vectors calculated in step-3.<br>
# **Step-5:** Normalise both matrices to have same sum. Since, it is easiest to get sum to be '1', we will simply divide each matrix by it's sum to normalise the data. <br>
# **Step-6:** Calculated numerator and denominator of Weighted Kappa and return the Weighted Kappa metric as `1-(num/den)`

# ### Interpreting the Quadratic Weighted Kappa Metric 

# A weighted Kappa is a metric which is used to calculate the amount of similarity between predictions and actuals. A perfect score of `1.0` is granted when both the predictions and actuals are the same. <br>
# Whereas, the least possible score is `-1` which is given when the predictions are furthest away from actuals. In our case, consider all actuals were 0's and all predictions were 4's. This would lead to a `QWKP` score of `-1`.<br>
# The aim is to get as close to 1 as possible. Generally a score of 0.6+ is considered to be a really good score. 

# ## Create our own Quadratic Weighted Kappa Metric

# In[ ]:



# In[ ]:


import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

# For the purpose of explaination, we will assume are actual and preds vectors to be the following. 

# In[ ]:


actuals = np.array([4, 4, 3, 4, 4, 4, 1, 1, 2, 1])
preds   = np.array([0, 2, 1, 0, 0, 0, 1, 1, 2, 1])

# In[ ]:


actuals.shape

# ### Step-1: Confusion Matrix

# In[ ]:


O = confusion_matrix(actuals, preds); O

# In[ ]:


confusion_matrix(actuals, preds)

# ### Step-2: Weighted Matrix

# An N-by-N matrix of weights, w, is calculated based on the difference between actual and predicted rating scores.

# In[ ]:


w = np.zeros((5,5)); w

# In[ ]:


for i in range(len(w)):
    for j in range(len(w)):
        w[i][j] = float(((i-j)**2)/16) #as per formula, for this competition, N=5

# In[ ]:


w

# Note that all values lying on the diagonal are penalised the least with a penalty of 0, whereas predictions and actuals furthest away from each other are penalised the most.

# ### Step-3: Histogram

# In[ ]:


N=5
act_hist=np.zeros([N])
for item in actuals: 
    act_hist[item]+=1
    
pred_hist=np.zeros([N])
for item in preds: 
    pred_hist[item]+=1

# In[ ]:


print(f'Actuals value counts:{act_hist}, Prediction value counts:{pred_hist}')

# Therefore, we have 3 values with adoption rating 1, 1 value with adoption rating 2, 1 value with adoption rating 1 an 5 values with adoption rating of 5 in the actuals. 

# ### Step-4: Expected Value (Outer product of histograms) 

# Expected matrix is calculated as the outer product between the actual rating's histogram vector of ratings and the predicted rating's histogram vector of ratings

# In[ ]:


E = np.outer(act_hist, pred_hist); E

# ### Step-5: Normalise E and O matrix

# `E` and `O` are normalized such that E and O have the same sum.

# In[ ]:


E = E/E.sum(); E.sum()

# In[ ]:


O = O/O.sum(); O.sum()

# In[ ]:


E

# In[ ]:


O

# ### Step-6: Calculate Weighted Kappa

# In[ ]:


num=0
den=0
for i in range(len(w)):
    for j in range(len(w)):
        num+=w[i][j]*O[i][j]
        den+=w[i][j]*E[i][j]
 
weighted_kappa = (1 - (num/den)); weighted_kappa

# ### Compare Result with Existing Metric

# The following code to calculate the Weighted Kappa Metric was used by Abhishek in his kernel https://www.kaggle.com/abhishek/maybe-something-interesting-here. 

# In[ ]:


# The following 3 functions have been taken from Ben Hamner's github repository
# https://github.com/benhamner/Metrics
def Cmatrix(rater_a, rater_b, min_rating=None, max_rating=None):
    """
    Returns the confusion matrix between rater's ratings
    """
    assert(len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(rater_a + rater_b)
    if max_rating is None:
        max_rating = max(rater_a + rater_b)
    num_ratings = int(max_rating - min_rating + 1)
    conf_mat = [[0 for i in range(num_ratings)]
                for j in range(num_ratings)]
    for a, b in zip(rater_a, rater_b):
        conf_mat[a - min_rating][b - min_rating] += 1
    return conf_mat


def histogram(ratings, min_rating=None, max_rating=None):
    """
    Returns the counts of each type of rating that a rater made
    """
    if min_rating is None:
        min_rating = min(ratings)
    if max_rating is None:
        max_rating = max(ratings)
    num_ratings = int(max_rating - min_rating + 1)
    hist_ratings = [0 for x in range(num_ratings)]
    for r in ratings:
        hist_ratings[r - min_rating] += 1
    return hist_ratings


def quadratic_weighted_kappa(y, y_pred):
    """
    Calculates the quadratic weighted kappa
    axquadratic_weighted_kappa calculates the quadratic weighted kappa
    value, which is a measure of inter-rater agreement between two raters
    that provide discrete numeric ratings.  Potential values range from -1
    (representing complete disagreement) to 1 (representing complete
    agreement).  A kappa value of 0 is expected if all agreement is due to
    chance.
    quadratic_weighted_kappa(rater_a, rater_b), where rater_a and rater_b
    each correspond to a list of integer ratings.  These lists must have the
    same length.
    The ratings should be integers, and it is assumed that they contain
    the complete range of possible ratings.
    quadratic_weighted_kappa(X, min_rating, max_rating), where min_rating
    is the minimum possible rating, and max_rating is the maximum possible
    rating
    """
    rater_a = y
    rater_b = y_pred
    min_rating=None
    max_rating=None
    rater_a = np.array(rater_a, dtype=int)
    rater_b = np.array(rater_b, dtype=int)
    assert(len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(min(rater_a), min(rater_b))
    if max_rating is None:
        max_rating = max(max(rater_a), max(rater_b))
    conf_mat = Cmatrix(rater_a, rater_b,
                                min_rating, max_rating)
    num_ratings = len(conf_mat)
    num_scored_items = float(len(rater_a))

    hist_rater_a = histogram(rater_a, min_rating, max_rating)
    hist_rater_b = histogram(rater_b, min_rating, max_rating)

    numerator = 0.0
    denominator = 0.0

    for i in range(num_ratings):
        for j in range(num_ratings):
            expected_count = (hist_rater_a[i] * hist_rater_b[j]
                              / num_scored_items)
            d = pow(i - j, 2.0) / pow(num_ratings - 1, 2.0)
            numerator += d * conf_mat[i][j] / num_scored_items
            denominator += d * expected_count / num_scored_items

    return (1.0 - numerator / denominator)

# In[ ]:


quadratic_weighted_kappa(actuals, preds)

# Our result matches the existic quadratic weighted kappa metric. 

# ## Rewrite the Quadratic Kappa Metric function

# In[ ]:


def quadratic_kappa(actuals, preds, N=5):
    """This function calculates the Quadratic Kappa Metric used for Evaluation in the PetFinder competition
    at Kaggle. It returns the Quadratic Weighted Kappa metric score between the actual and the predicted values 
    of adoption rating."""
    w = np.zeros((N,N))
    O = confusion_matrix(actuals, preds)
    for i in range(len(w)): 
        for j in range(len(w)):
            w[i][j] = float(((i-j)**2)/(N-1)**2)
    
    act_hist=np.zeros([N])
    for item in actuals: 
        act_hist[item]+=1
    
    pred_hist=np.zeros([N])
    for item in preds: 
        pred_hist[item]+=1
                         
    E = np.outer(act_hist, pred_hist);
    E = E/E.sum();
    O = O/O.sum();
    
    num=0
    den=0
    for i in range(len(w)):
        for j in range(len(w)):
            num+=w[i][j]*O[i][j]
            den+=w[i][j]*E[i][j]
    return (1 - (num/den))

# In[ ]:


actuals

# In[ ]:


preds

# In[ ]:


quadratic_kappa(actuals, preds)

# **What if both actuals and predictions match 100%?**

# In[ ]:


actuals = np.array([4, 4, 3, 4, 4, 4, 1, 1, 2, 0])
preds   = np.array([4, 4, 3, 4, 4, 4, 1, 1, 2, 0])
quadratic_kappa(actuals, preds)

# In[ ]:



