#!/usr/bin/env python
# coding: utf-8

# ## Recap
# So far, you have loaded your data and reviewed it with the following code. Run this cell to set up your coding environment where the previous step left off.

# In[3]:


# Code you have previously used to load data
import pandas as pd

# Path of the file to read
iowa_file_path = '../input/home-data-for-ml-course/train.csv'

home_data = pd.read_csv(iowa_file_path)

# Set up code checking
from learntools.core import binder
binder.bind(globals())
from learntools.machine_learning.ex3 import *

print("Setup Complete")

# # Exercises
# 
# ## Step 1: Specify Prediction Target
# Select the target variable, which corresponds to the sales price. Save this to a new variable called `y`. You'll need to print a list of the columns to find the name of the column you need.
# 

# In[ ]:


# print the list of columns in the dataset to find the name of the prediction target


# In[ ]:


#y = _

step_1.check()

# In[ ]:


# The lines below will show you a hint or the solution.
# step_1.hint() 
# step_1.solution()


# ## Step 2: Create X
# Now you will create a DataFrame called `X` holding the predictive features.
# 
# Since you want only some columns from the original data, you'll first create a list with the names of the columns you want in `X`.
# 
# You'll use just the following columns in the list (you can copy and paste the whole list to save some typing, though you'll still need to add quotes):
#     * LotArea
#     * YearBuilt
#     * 1stFlrSF
#     * 2ndFlrSF
#     * FullBath
#     * BedroomAbvGr
#     * TotRmsAbvGrd
# 
# After you've created that list of features, use it to create the DataFrame that you'll use to fit the model.

# In[ ]:


# Create the list of features below
# feature_names = ___

# select data corresponding to features in feature_names
#X = _

step_2.check()

# In[ ]:


# step_2.hint()
# step_2.solution()

# ## Review Data
# Before building a model, take a quick look at **X** to verify it looks sensible

# In[ ]:


# Review data
# print description or statistics from X
#print(_)

# print the top few lines
#print(_)


# ## Step 3: Specify and Fit Model
# Create a `DecisionTreeRegressor` and save it iowa_model. Ensure you've done the relevant import from sklearn to run this command.
# 
# Then fit the model you just created using the data in `X` and `y` that you saved above.

# In[ ]:


# from _ import _
#specify the model. 
#For model reproducibility, set a numeric value for random_state when specifying the model
iowa_model = _

# Fit the model
_

step_3.check()

# In[ ]:


# step_3.hint()
# step_3.solution()

# ## Step 4: Make Predictions
# Make predictions with the model's `predict` command using `X` as the data. Save the results to a variable called `predictions`.

# In[ ]:


predictions = _
print(predictions)
step_4.check()

# In[ ]:


# step_4.hint()
# step_4.solution()

# ## Think About Your Results
# 
# Use the `head` method to compare the top few predictions to the actual home values (in `y`) for those same homes. Anything surprising?
# 
# You'll understand why this happened if you keep going.

# 
# ## Keep Going
# You've built a decision tree model.  It's natural to ask how accurate the model's predictions will be and how you can improve that. Learn how to do that with **[Model Validation](https://www.kaggle.com/dansbecker/model-validation)**.
# 
# ---
# **[Course Home Page](https://www.kaggle.com/learn/machine-learning)**
# 
# 
# 
