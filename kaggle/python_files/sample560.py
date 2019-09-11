#!/usr/bin/env python
# coding: utf-8

# **Objective:**
# 
# This is an introductory notebook for people wanting to get started with [H2O](https://www.h2o.ai/products/h2o/) (the open source machine learning package by H2O.ai) 
# 
# **What is H2O?:**
# 
# H2O is a Java-based software for data modeling and general computing. There are many different perceptions of the H2O software, but the primary purpose of H2O is as a distributed (many machines), parallel (many CPUs), in memory (several hundred GBs Xmx) processing engine. 
# 
# Wait, we as Data Scientists do not need to know Java for using H2O to build models. We can use our favorite language (Python or R) :) 
# 
# ** H2O - Key Features:**
# 
# Some of the key features 
# 1. Access from both R and Python
# 2. Access from  web-based interface named Flow. By means of Flow, data scientists are able to import, explore, and modify datasets, play with models, verify models performances, and much more. (This is not accessible here in Kaggle Kernels)
# 3. AutoML : automatic training and tuning of many models within a user-specified time-limit. 
# 4. Distributed, In-memory processing : In-memory processing with fast serialization between nodes and clusters to support massive datasets. 
# 5. Simple Deployment : Easy to deploy POJOs and MOJOs to deploy models for fast and accurate scoring in any environment, including with very large models.
# 
# Let us first import the necessary modules.

# In[ ]:


import h2o
import time
import seaborn
import itertools
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from h2o.estimators.glm import H2OGeneralizedLinearEstimator
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.estimators.random_forest import H2ORandomForestEstimator


# Once the module in imported, the first step is to initialize the h2o module. 
# 
# The *h2o.init()* command is pretty smart and does a lot of things. First, an attempt is made to search for an existing H2O instance being started already, before starting a new one. When none is found automatically or specified manually with argument available, a new instance of H2O is started. 
# 
# During startup, H2O is going to print some useful information. Version of the Python it is running on, H2O’s version, how to connect to H2O’s Flow interface or where error logs reside, just to name a few.

# In[ ]:


h2o.init()

# **Data Exploration:**
# 
# Now that the initialization is done, let us first import the dataset. The command is very similar to *pandas.read_csv* and the data is stored in memory as [H2OFrame](http://docs.h2o.ai/h2o/latest-stable/h2o-py/docs/frame.html)
# 
# H2O supports various file formats and data sources. More detailed information can be seen [here](http://docs.h2o.ai/h2o/latest-stable/h2o-docs/getting-data-into-h2o.html).

# In[ ]:


diabetes_df = h2o.import_file("../input/diabetes.csv", destination_frame="diabetes_df")

# We can also see the progress when the loading happens. This will be very helpful while dealing with larger datasets.
# 
# As a first step, let us have a look at the dataset. 

# In[ ]:


diabetes_df.describe()

# *describe()* gives out a lot of information. 
# 1. Number of rows and columns in the dataset
# 2. A number of summary statistics about the dataset such as 
#     * Data type of the column such as integer, categorical etc
#     * Minimum value
#     * Mean value
#     * Maximum value
#     * Standard deviation value
#     * Number of zeros in the column
#     * Number of missing values in the column
# 3. A look at the top few rows
# 
# Now let us look at the distribution of the individual features using *hist()* command 

# In[ ]:


for col in diabetes_df.columns:
    diabetes_df[col].hist()

# Now let us also look at the correlation of the individual features. We can use the *cor()* function in H2OFrame for the same.

# In[ ]:


plt.figure(figsize=(10,10))
corr = diabetes_df.cor().as_data_frame()
corr.index = diabetes_df.columns
sns.heatmap(corr, annot = True, cmap='RdYlGn', vmin=-1, vmax=1)
plt.title("Correlation Heatmap", fontsize=16)
plt.show()

# Let us now split the data into three parts - train, valid and test datasets - at a ratio of 60%, 20% and 20% respectively. We could use *split_frame()* function for the same.

# In[ ]:


train, valid, test = diabetes_df.split_frame(ratios=[0.6,0.2], seed=1234)
response = "Outcome"
train[response] = train[response].asfactor()
valid[response] = valid[response].asfactor()
test[response] = test[response].asfactor()
print("Number of rows in train, valid and test set : ", train.shape[0], valid.shape[0], test.shape[0])

# **Modeling and Model Tuning : **
# 
# Now, let us build a baseline model using these splits. There are [multiple algorithms](http://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science.html) available in the H2O module. We can start with the Kaggler's favorite - GBM.
# 
# Please note that the target variable is "Outcome" and rest of the features are as input features. Since this is a baseline model, let us use the default parameters.

# In[ ]:


predictors = diabetes_df.columns[:-1]
gbm = H2OGradientBoostingEstimator()
gbm.train(x=predictors, y=response, training_frame=train)

# Again, it is good to see the model building progress.
# 
# Now let us do a *print(model_name)* to understand more about the model.

# In[ ]:


print(gbm)

# Now that is quite a bit of information. We can look at them individually.
# 1. First, we get the name of the model and a key to acces the model ( key is not much useful for us I guess )
# 2. Error metrics on the train data like log-loss, mean per class error, AUC, Gini, MSE, RMSE
# 3. Confusion matrix for max F1 threshold
# 4. Threshold values for different metrics
# 5. Gains / Lift table 
# 6. Scoring history - information on how the metrics changed in each of the epochs
# 7. Feature importance
# 
# Okay. I heard you. How can we use the metrics of train set (as we actually trained on this dataset). We need to evaluate them from the valid set. We can use the *model_performance()* function for the same. We can then print the metrics. 

# In[ ]:


perf = gbm.model_performance(valid)
print(perf)

# So using our baseline model, we are getting about 0.8 auc in valid set and 0.98 auc in train set. Similarly, log loss is 0.53 in valid set and 0.21 in train set. 
# 
# Now we can use the validation set to tune our parameters. We can use the early stopping to find the number of iterations to train similar to other GBM implementations. We can set some random values for the parameters to start with. Please note that, we have added a new *validation_frame* parameter in this one compared to the previous one while training. 

# In[ ]:


gbm_tune = H2OGradientBoostingEstimator(
    ntrees = 3000,
    learn_rate = 0.01,
    stopping_rounds = 20,
    stopping_metric = "AUC",
    col_sample_rate = 0.7,
    sample_rate = 0.7,
    seed = 1234
)      
gbm_tune.train(x=predictors, y=response, training_frame=train, validation_frame=valid)

# Now let us check the validation auc to check the performance.

# In[ ]:


gbm_tune.model_performance(valid).auc()

# We are getting similar performance (0.8 valid AUC) using this new model with early stopping too. 
# 
# **Grid Search:**
# 
# Now let us do grid search to find the best paramters for GBM model. 

# In[ ]:


from h2o.grid.grid_search import H2OGridSearch

gbm_grid = H2OGradientBoostingEstimator(
    ntrees = 3000,
    learn_rate = 0.01,
    stopping_rounds = 20,
    stopping_metric = "AUC",
    col_sample_rate = 0.7,
    sample_rate = 0.7,
    seed = 1234
) 

hyper_params = {'max_depth':[4,6,8,10,12]}
grid = H2OGridSearch(gbm_grid, hyper_params,
                         grid_id='depth_grid',
                         search_criteria={'strategy': "Cartesian"})
#Train grid search
grid.train(x=predictors, 
           y=response,
           training_frame=train,
           validation_frame=valid)

# In[ ]:


print(grid)

# So this has printed the log loss performance at various depths. If we want to look at the validation AUC, then we can use the following.

# In[ ]:


sorted_grid = grid.get_grid(sort_by='auc',decreasing=True)
print(sorted_grid)

# Interestingly, there is not much change in the AUC for the top two results. Since we train on a very small sample, we might be getting this.
# 
# Also please note that, we just searched for the *max_depth* parameter. Please do a more comprehensive search for better results. Please refer to this [notebook](https://github.com/h2oai/h2o-3/blob/master/h2o-docs/src/product/tutorials/gbm/gbmTuning.ipynb) for more comprehensive details on finetuning. 
# 
# **K-Fold cross validation:**
# 
# Most of the times, we will just do K-fold cross valdiation. So now let us do the same using H2O. Just setting the *nfolds* parameter in the model will do the k-fold cross validation.

# In[ ]:


cv_gbm = H2OGradientBoostingEstimator(
    ntrees = 3000,
    learn_rate = 0.05,
    stopping_rounds = 20,
    stopping_metric = "AUC",
    nfolds=4, 
    seed=2018)
cv_gbm.train(x = predictors, y = response, training_frame = train, validation_frame=valid)
cv_summary = cv_gbm.cross_validation_metrics_summary().as_data_frame()
cv_summary

# Now let us test the performance on the valid set just like before. 

# In[ ]:


cv_gbm.model_performance(valid).auc()

# **XGBoost:**
# 
# Recently H2O has also added the XGBoost version of GBM into its kitty. Now let us see how to use the XGBoost model in H2O. We follow the same code convention as that of GBM except that we will use *H2OXGBoostEstimator* function. 

# In[ ]:


from h2o.estimators import H2OXGBoostEstimator

cv_xgb = H2OXGBoostEstimator(
    ntrees = 3000,
    learn_rate = 0.05,
    stopping_rounds = 20,
    stopping_metric = "AUC",
    nfolds=4, 
    seed=2018)
cv_xgb.train(x = predictors, y = response, training_frame = train, validation_frame=valid)
cv_xgb.model_performance(valid).auc()

# Not much improvement in the performance. Probably we need to tune the parameters more. 
# 
# Getting the variable importances plot from a model is simple too. *varimp_plot()* will get us that. Now let us check the variable importance of the XGBoost model.

# In[ ]:


cv_xgb.varimp_plot()

# **AutoML : Automatic Machine Learning:**
# 
# From the [H2O AutoML page](http://docs.h2o.ai/h2o/latest-stable/h2o-docs/automl.html),
# 
# *H2O’s AutoML can be used for automating the machine learning workflow, which includes automatic training and tuning of many models within a user-specified time-limit. Stacked Ensembles will be automatically trained on collections of individual models to produce highly predictive ensemble models which, in most cases, will be the top performing models in the AutoML Leaderboard.*
# 
# So let us use the *H2OAutoML* function to do automatic machine learning. We can specify the *max_models* parameter which indicates the number of individual (or "base") models, and does not include the two ensemble models that are trained at the end.

# In[ ]:


from h2o.automl import H2OAutoML

aml = H2OAutoML(max_models = 10, max_runtime_secs=100, seed = 1)
aml.train(x=predictors, y=response, training_frame=train, validation_frame=valid)

# Now let us look at the automl leaderboard.

# In[ ]:


lb = aml.leaderboard
lb

# AutoML has built variety of models inlcuding GBM, GLM, Deep Learning and XRT (Extremely Randomized Trees) and then build two stacked ensemble models (the first two in the leaderboard) on top of them and the best model is a stacked ensemble. 
# 
# Now let us look at the contribution of the individual models for this meta learner. 

# In[ ]:


metalearner = h2o.get_model(aml.leader.metalearner()['name'])
metalearner.std_coef_plot()

# So GBM is the topmost contributor to the ensemble followed by GLM and DL. 

# **References:**
# 
# 1. [GBM tuning tutorial for python](https://github.com/h2oai/h2o-3/blob/master/h2o-docs/src/product/tutorials/gbm/gbmTuning.ipynb)
# 2. [Machine Learning with H2O](https://dzone.com/articles/machine-learning-with-h2o-hands-on-guide-for-data)
# 3. [XGBoost in H2O platform](https://blog.h2o.ai/2017/06/xgboost-in-h2o-machine-learning-platform/)
# 4. [AutoML H2O Demo](https://github.com/h2oai/h2o-tutorials/blob/master/h2o-world-2017/automl/Python/automl_binary_classification_product_backorders.ipynb)
# 5. [AutoML Docs](http://docs.h2o.ai/h2o/latest-stable/h2o-docs/automl.html)
# 

# **More to come. Stay tuned.!**
# 
# Disclaimer : I am currently working as a Data Scientist at H2O.ai
