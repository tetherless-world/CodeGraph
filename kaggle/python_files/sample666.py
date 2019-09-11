#!/usr/bin/env python
# coding: utf-8

#  # <div style="text-align: center"> Tutorial on Ensemble Learning (Don't Overfit) </div>
#  <img src='https://upload.wikimedia.org/wikipedia/commons/thumb/1/19/Overfitting.svg/800px-Overfitting.svg.png' width=400 height=400 >
# ### <div style="text-align: center"> CLEAR DATA. MADE MODEL </div>
# <div style="text-align:center">last update: <b>06/03/2019</b></div>
# 
# 
# >You are reading **10 Steps to Become a Data Scientist** and are now in the 8th step : 
# 
# 1. [Leren Python](https://www.kaggle.com/mjbahmani/the-data-scientist-s-toolbox-tutorial-1)
# 2. [Python Packages](https://www.kaggle.com/mjbahmani/the-data-scientist-s-toolbox-tutorial-2)
# 3. [Mathematics and Linear Algebra](https://www.kaggle.com/mjbahmani/linear-algebra-for-data-scientists)
# 4. [Programming &amp; Analysis Tools](https://www.kaggle.com/mjbahmani/20-ml-algorithms-15-plot-for-beginners)
# 5. [Big Data](https://www.kaggle.com/mjbahmani/a-data-science-framework-for-quora)
# 6. [Data visualization](https://www.kaggle.com/mjbahmani/top-5-data-visualization-libraries-tutorial)
# 7. [Data Cleaning](https://www.kaggle.com/mjbahmani/machine-learning-workflow-for-house-prices)
# 8. <font color="red">You are in the 8th step</font>
# 9. [A Comprehensive ML  Workflow with Python](https://www.kaggle.com/mjbahmani/a-comprehensive-ml-workflow-with-python)
# 10. [Deep Learning](https://www.kaggle.com/mjbahmani/top-5-deep-learning-frameworks-tutorial)
# 
# 
# you can Fork and Run this kernel on <font color="red">Github</font>:
# 
# > [ GitHub](https://github.com/mjbahmani/10-steps-to-become-a-data-scientist)
# 
#  **I hope you find this kernel helpful and some <font color='red'> UPVOTES</font> would be very much appreciated**
#  
#  -----------

# <a id="top"></a> <br>
# ## Notebook  Content
# 1. [Introduction](#1)
# 1. [Import packages](#2)
#     1. [Version](#21)
#     1. [Setup](#22)
#     1. [Data Collection](#23)
# 1. [Exploratory Data Analysis(EDA)](#3)
# 1. [What's Ensemble Learning?](#4)
#     1. [Why Ensemble Learning?](#41)
# 1. [Ensemble Techniques](#5)
#     1. [what-is-the-difference-between-bagging-and-boosting?](#51)
# 1. [Model Deployment](#6)
#     1. [Prepare Features & Targets](#61)
#     1. [RandomForest](#62)
#     1. [Bagging classifier ](#63)
#     1. [AdaBoost](#64)
#     1. [Gradient Boosting Classifier](#65)
#     1. [Linear Discriminant Analysis](#66)
#     1. [Quadratic Discriminant Analysis](#67)
# 1. [Don't Overfit](#7)
#     1. [Feature Importance](#71)
#     1. [Partial Dependence Plots](#72)
#     1. [pdpbox](#73)
#     1. [SHAP Values](#74)
# 1. [Model Development](#8)
#     1. [lightgbm](#81)
#     1. [RandomForestClassifier](#82)
#     1. [ DecisionTreeClassifier](#83)
#     1. [CatBoostClassifier](#84)
# 1. [References & Credits](#9)

# <a id="1"></a> <br>
# #  1- Introduction
# In this kernel, I want to start explorer everything about **Ensemble modeling** with focus on **Overfit**. I will run plenty of algorithms on Don't Overfit dataset. I hope you enjoy and give me feedback.

# <a id="2"></a> <br>
# ## 2- Import packages

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from catboost import CatBoostClassifier,Pool
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from pandas import get_dummies
import plotly.graph_objs as go
from sklearn import datasets
import plotly.plotly as py
import seaborn as sns
import xgboost as xgb
import lightgbm as lgb
import pandas as pd
import numpy as np
import matplotlib
import warnings
import sklearn
import scipy
import numpy
import time
import json
import sys
import csv
import os

# <a id="21"></a> <br>
# ### 2-1 Version

# In[ ]:


print('matplotlib: {}'.format(matplotlib.__version__))
print('sklearn: {}'.format(sklearn.__version__))
print('scipy: {}'.format(scipy.__version__))
print('seaborn: {}'.format(sns.__version__))
print('pandas: {}'.format(pd.__version__))
print('numpy: {}'.format(np.__version__))
print('Python: {}'.format(sys.version))

# <a id="22"></a> <br>
# ### 2-2 Setup
# 
# A few tiny adjustments for better **code readability**

# In[ ]:


# for get better result chage fold_n to 5
fold_n=5
folds = StratifiedKFold(n_splits=fold_n, shuffle=True, random_state=10)
warnings.filterwarnings('ignore')
sns.set(color_codes=True)
plt.style.available

# <a id="3"></a> 
# ## 3- Exploratory Data Analysis(EDA)
#  In this section, we'll analysis how to use graphical and numerical techniques to begin uncovering the structure of your data. 
# *  Data Collection
# *  Visualization
# *  Data Preprocessing
# *  Data Cleaning

#  <a id="31"></a> <br>
# ## 3-1 Data Collection

# In[ ]:


import os
print([filename for filename in os.listdir('../input') if '.csv' in filename])

# In[ ]:


# import Dataset to play with it
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

# In[ ]:


sample_submission = pd.read_csv('../input/sample_submission.csv')
sample_submission.head()

# **<< Note 1 >>**
# 
# * Each row is an observation (also known as : sample, example, instance, record)
# * Each column is a feature (also known as: Predictor, attribute, Independent Variable, input, regressor, Covariate)

# In[ ]:


train.shape, test.shape, sample_submission.shape

# In[ ]:


train.head()

# In[ ]:


test.head()

# In[ ]:


train.tail()

# In[ ]:


train.columns

# In[ ]:


print(train.info())

#  <a id="32"></a> <br>
# ## 3-2 Visualization

# <a id="321"></a> 
# ### 3-2-1 hist

# In[ ]:


train['target'].value_counts().plot.bar();

# <a id="323"></a> 
# ### 3-2-3 countplot

# In[ ]:


f,ax=plt.subplots(1,2,figsize=(18,8))
train['target'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True)
ax[0].set_title('target')
ax[0].set_ylabel('')
sns.countplot('target',data=train,ax=ax[1])
ax[1].set_title('target')
plt.show()

# <a id="324"></a> 
# ### 3-2-4 hist
# if you check histogram for all feature, you will find that most of them are so similar

# In[ ]:


train["1"].hist();

# In[ ]:


train["2"].hist();

# In[ ]:


train["3"].hist();

# ### 3-2-5 Mean Frequency

# In[ ]:


train[train.columns[2:]].mean().plot('hist');plt.title('Mean Frequency');

# <a id="326"></a> 
# ### 3-2-6 distplot
#  the target in data set is **imbalance**

# In[ ]:


sns.set(rc={'figure.figsize':(9,7)})
sns.distplot(train['target']);

# <a id="327"></a> 
# ### 3-2-7 violinplot

# In[ ]:


sns.violinplot(data=train,x="target", y="1");

# In[ ]:


sns.violinplot(data=train,x="target", y="20");

# <a id="4"></a> <br>
# ## 4- What's Ensemble Learning?
# let us, review some defination on Ensemble Learning:
# 
# 1. **Ensemble learning** is the process by which multiple models, such as classifiers or experts, are strategically generated and combined to solve a particular computational intelligence problem[9]
# 1. **Ensemble Learning** is a powerful way to improve the performance of your model. It usually pays off to apply ensemble learning over and above various models you might be building. Time and again, people have used ensemble models in competitions like Kaggle and benefited from it.[6]
# 1. **Ensemble methods** are techniques that create multiple models and then combine them to produce improved results. Ensemble methods usually produces more accurate solutions than a single model would.[10]
# <img src='https://hub.packtpub.com/wp-content/uploads/2018/02/ensemble_machine_learning_image_1-600x407.png'  width=400 height=400>
# [img-ref](https://hub.packtpub.com/wp-content/uploads/2018/02/ensemble_machine_learning_image_1-600x407.png)
# 
# > <font color="red"><b>Note</b></font>
# Ensemble Learning is a Machine Learning concept in which the idea is to train multiple models using the same learning algorithm. The ensembles take part in a bigger group of methods, called multiclassifiers, where a set of hundreds or thousands of learners with a common objective are fused together to solve the problem.[11]
# 
# > <font color="red"><b>Note</b></font>
# This Kernel assumes a basic understanding of Machine Learning algorithms. I would recommend going through this [**kernel**](https://www.kaggle.com/mjbahmani/a-comprehensive-ml-workflow-with-python)  to familiarize yourself with these concepts.
# 
# [go to top](#top)

# <a id="41"></a> <br>
# ## 4-1 Why Ensemble Learning?
# 1. Difference in population
# 1. Difference in hypothesis
# 1. Difference in modeling technique
# 1. Difference in initial seed
# <br>
# [go to top](#top)

# <a id="5"></a> <br>
# # 5- Ensemble Techniques
# The goal of any machine learning problem is to find a single model that will best predict our wanted outcome. Rather than making one model and hoping this model is the best/most accurate predictor we can make, ensemble methods take a myriad of models into account, and average those models to produce one final model.[12]
# <img src='https://uploads.toptal.io/blog/image/92062/toptal-blog-image-1454584029018-cffb1b601292e8d328556e355ed4f7e0.jpg' width=300 height=300>
# [img-ref](https://www.toptal.com/machine-learning/ensemble-methods-machine-learning)
# 1. Voting
# 1. Weighted Average 
# 1. Stacking
# 1. Blending
# 1. Bagging  
# 1. Boosting 

# <a id="51"></a> <br>
# ## 5-1 What is the difference between bagging and boosting?
# 1. **Bagging**: It is the method to decrease the variance of model by generating additional data for training from your original data set using combinations with repetitions to produce multisets of the same size as your original data.
#     1. Bagging meta-estimator
#     1. Random forest
# 1. **Boosting**: It helps to calculate the predict the target variables using different models and then average the result( may be using a weighted average approach).
#     1. AdaBoost
#     1. GBM
#     1. XGBM
#     1. Light GBM
#     1. CatBoost
#     
# <img src='https://www.globalsoftwaresupport.com/wp-content/uploads/2018/02/ds33ggg.png'>
# [Image-Credit](https://www.globalsoftwaresupport.com/boosting-adaboost-in-machine-learning/)
# <br>
# [go to top](#top)

# <a id="6"></a> <br>
# ## 6- Some Ensemble  Model
# In this section have been applied more than **8 learning algorithms** that play an important rule in your experiences and improve your knowledge in case of ML technique.
# 
# > **<< Note 3 >>** : The results shown here may be slightly different for your analysis because, for example, the neural network algorithms use random number generators for fixing the initial value of the weights (starting points) of the neural networks, which often result in obtaining slightly different (local minima) solutions each time you run the analysis. Also note that changing the seed for the random number generator used to create the train, test, and validation samples can change your results.
# <br>
# [go to top](#top)

# <a id="61"></a> <br>
# ## 6-1 Prepare Features & Targets
# First of all seperating the data into dependent(Feature) and independent(Target) variables.
# 
# **<< Note 4 >>**
# * X==>>Feature
# * y==>>Target

# In[ ]:


train['target'].value_counts()

# In[ ]:


cols=["target","id"]
X = train.drop(cols,axis=1)
y = train["target"]

# In[ ]:


X_test  = test.drop("id",axis=1)

# In[ ]:


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.2, random_state = 0)

# After loading the data via **pandas**, we should checkout what the content is, description and via the following:
# <br>
# [go to top](#top)

# <a id="62"></a> <br>
# ## 6-2 RandomForest
# A random forest is a meta estimator that **fits a number of decision tree classifiers** on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting.[RandomForestClassifie](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) 
# 
# The sub-sample size is always the same as the original input sample size but the samples are drawn with replacement if bootstrap=True (default).

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
Model=RandomForestClassifier(max_depth=2)
Model.fit(X_train,y_train)
y_pred=Model.predict(X_val)
print(classification_report(y_pred,y_val))
print(confusion_matrix(y_pred,y_val))
#Accuracy Score
print('accuracy is ',accuracy_score(y_pred,y_val))

# <a id="63"></a> <br>
# ## 6-3 Bagging classifier 
# A Bagging classifier is an ensemble **meta-estimator** that fits base classifiers each on random subsets of the original dataset and then aggregate their individual predictions (either by voting or by averaging) to form a final prediction. Such a meta-estimator can typically be used as a way to reduce the variance of a black-box estimator (e.g., a decision tree), by introducing randomization into its construction procedure and then making an ensemble out of it.
# 
# This algorithm encompasses several works from the literature. When random subsets of the dataset are drawn as random subsets of the samples, then this algorithm is known as Pasting . If samples are drawn with replacement, then the method is known as Bagging . When random subsets of the dataset are drawn as random subsets of the features, then the method is known as Random Subspaces . Finally, when base estimators are built on subsets of both samples and features, then the method is known as Random Patches .[http://scikit-learn.org]
# <br>
# [go to top](#top)

# In[ ]:


from sklearn.ensemble import BaggingClassifier
bag_Model=BaggingClassifier()
Model.fit(X_train,y_train)
y_pred=Model.predict(X_val)
print(classification_report(y_pred,y_val))
print(confusion_matrix(y_pred,y_val))
#Accuracy Score
print('accuracy is ',accuracy_score(y_pred,y_val))

# <a id="64"></a> <br>
# ##  6-4 AdaBoost classifier
# 
# An AdaBoost classifier is a meta-estimator that begins by fitting a classifier on the original dataset and then fits additional copies of the classifier on the same dataset but where the weights of incorrectly classified instances are adjusted such that subsequent classifiers focus more on difficult cases.
# This class implements the algorithm known as **AdaBoost-SAMME** .

# In[ ]:


from sklearn.ensemble import AdaBoostClassifier
Model=AdaBoostClassifier()
Model.fit(X_train,y_train)
y_pred=Model.predict(X_val)
print(classification_report(y_pred,y_val))
print(confusion_matrix(y_pred,y_val))
#Accuracy Score
print('accuracy is ',accuracy_score(y_pred,y_val))

# <a id="65"></a> <br>
# ## 6-5 Gradient Boosting Classifier
# GB builds an additive model in a forward stage-wise fashion; it allows for the optimization of arbitrary differentiable loss functions.

# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier
Model=GradientBoostingClassifier()
Model.fit(X_train,y_train)
y_pred=Model.predict(X_val)
print(classification_report(y_pred,y_val))
print(confusion_matrix(y_pred,y_val))
#Accuracy Score
print('accuracy is ',accuracy_score(y_pred,y_val))

# <a id="66"></a> <br>
# ## 6-6 Linear Discriminant Analysis
# Linear Discriminant Analysis (discriminant_analysis.LinearDiscriminantAnalysis) and Quadratic Discriminant Analysis (discriminant_analysis.QuadraticDiscriminantAnalysis) are two classic classifiers, with, as their names suggest, a **linear and a quadratic decision surface**, respectively.
# 
# These classifiers are attractive because they have closed-form solutions that can be easily computed, are inherently multiclass, have proven to work well in practice, and have no **hyperparameters** to tune.

# In[ ]:


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
Model=LinearDiscriminantAnalysis()
Model.fit(X_train,y_train)
y_pred=Model.predict(X_val)
print(classification_report(y_pred,y_val))
print(confusion_matrix(y_pred,y_val))
#Accuracy Score
print('accuracy is ',accuracy_score(y_pred,y_val))

# <a id="67"></a> <br>
# ## 6-7 Quadratic Discriminant Analysis
# A classifier with a quadratic decision boundary, generated by fitting class conditional densities to the data and using Bayes’ rule.
# 
# The model fits a **Gaussian** density to each class.

# In[ ]:


from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
Model=QuadraticDiscriminantAnalysis()
Model.fit(X_train,y_train)
y_pred=Model.predict(X_val)
print(classification_report(y_pred,y_val))
print(confusion_matrix(y_pred,y_val))
#Accuracy Score
print('accuracy is ',accuracy_score(y_pred,y_val))

# <a id="7"></a> <br>
# # 7- Don't Overfit
# To solve Don't Overfit problem, I would like to suggest that we first extract some insights  with using this [great course](https://www.kaggle.com/learn/machine-learning-explainability) by [dansbecker](https://www.kaggle.com/dansbecker). To understand this section, it's good idea to first read this course.

# <a id="71"></a> <br>
# ## 7-1 Feature Importance

# In[ ]:


train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
rfc_model = RandomForestClassifier(random_state=0).fit(train_X, train_y)

# Here is how to calculate and show importances with the [eli5](https://eli5.readthedocs.io/en/latest/) library:

# In[ ]:


import eli5
from eli5.sklearn import PermutationImportance

perm = PermutationImportance(rfc_model, random_state=1).fit(val_X, val_y)
eli5.show_weights(perm, feature_names = val_X.columns.tolist(),top=300)

# <a id="7-1-1"></a> <br>
# ## 7-1-1 What can be inferred from the above?
# 1. As you move down the top of the graph, the importance of the feature decreases.
# 1. The features that are shown in green indicate that they have a positive impact on our prediction
# 1. The features that are shown in white indicate that they have no effect on our prediction
# 1. The features shown in red indicate that they have a negative impact on our prediction

# <a id="72"></a> <br>
# ## 7-2 Partial Dependence Plots
# While feature importance shows what variables most affect predictions, partial dependence plots show how a feature affects predictions.[Credit](https://www.kaggle.com/learn/machine-learning-explainability)

# In[ ]:


train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
tree_model = DecisionTreeClassifier(random_state=0, max_depth=5, min_samples_split=5).fit(train_X, train_y)

# For the sake of explanation, I use a Decision Tree which you can see below.

# In[ ]:


features = [c for c in train.columns if c not in ['id', 'target']]

# In[ ]:


from sklearn import tree
import graphviz
tree_graph = tree.export_graphviz(tree_model, out_file=None, feature_names=features)
display(graphviz.Source(tree_graph))

# <a id="73"></a> <br>
# ## 7-3 pdpbox

# In[ ]:


from matplotlib import pyplot as plt
from pdpbox import pdp, get_dataset, info_plots

# Create the data that we will plot
pdp_goals = pdp.pdp_isolate(model=tree_model, dataset=val_X, model_features=features, feature='26')

# plot it
pdp.pdp_plot(pdp_goals, '26')
plt.show()

# In[ ]:


# Create the data that we will plot
pdp_goals = pdp.pdp_isolate(model=rfc_model, dataset=val_X, model_features=features, feature='264')

# plot it
pdp.pdp_plot(pdp_goals, '264')
plt.show()

# <a id="74"></a> <br>
# ## 7-4 SHAP Values
# SHAP (SHapley Additive exPlanations) is a unified approach to explain the output of any machine learning model. SHAP connects game theory with local explanations, uniting several previous methods [1-7] and representing the only possible consistent and locally accurate additive feature attribution method based on expectations (see the SHAP NIPS paper for details).
# 
# 
# <img src='https://raw.githubusercontent.com/slundberg/shap/master/docs/artwork/shap_diagram.png' width=400 height=400  >
# 
# [image-Credits](https://github.com/slundberg/shap)

# This section is based on this [course](https://www.kaggle.com/dansbecker/shap-values) on kaggle.
# 
# 1. SHAP Values (an acronym from SHapley Additive exPlanations) break down a prediction to show the impact of each feature. Where could you use this?
# 1. We 'll use SHAP Values to explain individual predictions in this kernel.

# In[ ]:


import shap  # package used to calculate Shap values

# Create object that can calculate shap values
explainer = shap.TreeExplainer(rfc_model)

# Calculate Shap values
shap_values = explainer.shap_values(X_train)

# In[ ]:


shap.summary_plot(shap_values, X_train)

# In[ ]:


shap.initjs()
shap.force_plot(explainer.expected_value[1], shap_values[1], X_train.iloc[:,1:10])

# <a id="8"></a> <br>
# # 8- Model Development
# So far, we have used two models, and at this point we add another model and we'll be expanding it soon. in this section you will see following model:
# 
# 1. lightgbm
# 1. RandomForestClassifier
# 1. DecisionTreeClassifier
# 1. CatBoostClassifier

# <a id="81"></a> <br>
# ## 8-1 lightgbm

# In[ ]:


# based on following kernel https://www.kaggle.com/dromosys/sctp-working-lgb
params = {'num_leaves': 9,
         'min_data_in_leaf': 42,
         'objective': 'binary',
         'max_depth': 16,
         'learning_rate': 0.0123,
         'boosting': 'gbdt',
         'bagging_freq': 5,
         'bagging_fraction': 0.8,
         'feature_fraction': 0.8201,
         'bagging_seed': 11,
         'reg_alpha': 1.728910519108444,
         'reg_lambda': 4.9847051755586085,
         'random_state': 42,
         'metric': 'auc',
         'verbosity': -1,
         'subsample': 0.81,
         'min_gain_to_split': 0.01077313523861969,
         'min_child_weight': 19.428902804238373,
         'num_threads': 4}

# In[ ]:


## based on following kernel https://www.kaggle.com/dromosys/sctp-working-lgb
y_pred_lgb = np.zeros(len(X_test))
for fold_n, (train_index, valid_index) in enumerate(folds.split(X,y)):
    print('Fold', fold_n, 'started at', time.ctime())
    X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
    y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]
    
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_valid, label=y_valid)
        
    lgb_model = lgb.train(params,train_data,num_boost_round=2000,#change 20 to 2000
                    valid_sets = [train_data, valid_data],verbose_eval=300,early_stopping_rounds = 200)##change 10 to 200
            
    y_pred_lgb += lgb_model.predict(X_test, num_iteration=lgb_model.best_iteration)/5

# <a id="82"></a> <br>
# ## 8-2 RandomForestClassifier

# In[ ]:


y_pred_rfc = rfc_model.predict(X_test)

# <a id="83"></a> <br>
# ## 8-3 DecisionTreeClassifier

# In[ ]:


y_pred_tree = tree_model.predict(X_test)

# <a id="84"></a> <br>
# ## 8-4 CatBoostClassifier

# In[ ]:


train_pool = Pool(train_X,train_y)
cat_model = CatBoostClassifier(
                               iterations=3000,# change 25 to 3000 to get best performance 
                               learning_rate=0.03,
                               objective="Logloss",
                               eval_metric='AUC',
                              )
cat_model.fit(train_X,train_y,silent=True)
y_pred_cat = cat_model.predict(X_test)

# <a id="85"></a> <br>
# ## 8-5 Gradient Boosting Classifier

# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier
Model=GradientBoostingClassifier()
Model.fit(X_train,y_train)
y_pred=Model.predict(X_test)

# Now you can change your model and submit the results of other models.

# In[ ]:


submission_rfc = pd.DataFrame({
        "id": test["id"],
        "target": y_pred_rfc
    })
submission_rfc.to_csv('submission_rfc.csv', index=False)

# In[ ]:


submission_tree = pd.DataFrame({
        "id": test["id"],
        "target": y_pred_tree
    })
submission_tree.to_csv('submission_tree.csv', index=False)

# In[ ]:


submission_cat = pd.DataFrame({
        "id": test["id"],
        "target": y_pred_cat
    })
submission_cat.to_csv('submission_cat.csv', index=False)

# In[ ]:


submission_lgb = pd.DataFrame({
        "id": test["id"],
        "target": y_pred_lgb
    })
submission_lgb.to_csv('submission_lgb.csv', index=False)

# you can follow me on:
# > ###### [ GitHub](https://github.com/mjbahmani)
# > ###### [Kaggle](https://www.kaggle.com/mjbahmani/)
# 
#   **I hope you find this kernel helpful and some <font color='red'> UPVOTES</font> would be very much appreciated**
#  

# <a id="9"></a> <br>
# # 9-References & Credits

# 1. [datacamp](https://www.datacamp.com/community/tutorials/xgboost-in-python)
# 1. [Github](https://github.com/mjbahmani)
# 1. [analyticsvidhya](https://www.analyticsvidhya.com/blog/2015/08/introduction-ensemble-learning/)
# 1. [ensemble-learning-python](https://www.datacamp.com/community/tutorials/ensemble-learning-python)
# 1. [image-header-reference](https://data-science-blog.com/blog/2017/12/03/ensemble-learning/)
# 1. [scholarpedia](http://www.scholarpedia.org/article/Ensemble_learning)
# 1. [toptal](https://www.toptal.com/machine-learning/ensemble-methods-machine-learning)
# 1. [quantdare](https://quantdare.com/what-is-the-difference-between-bagging-and-boosting/)
# 1. [towardsdatascience](https://towardsdatascience.com/ensemble-methods-in-machine-learning-what-are-they-and-why-use-them-68ec3f9fef5f)
# 1. [scikit-learn](https://scikit-learn.org/stable/modules/ensemble.html)
# 1. [https://www.kaggle.com/dansbecker/permutation-importance](https://www.kaggle.com/dansbecker/permutation-importance)
# 1. [https://www.kaggle.com/dansbecker/partial-plots](https://www.kaggle.com/dansbecker/partial-plots)
# 1. [https://www.kaggle.com/dansbecker/shap-values](https://www.kaggle.com/dansbecker/shap-values)
# 1. [https://www.kaggle.com/miklgr500/catboost-with-gridsearch-cv](https://www.kaggle.com/miklgr500/catboost-with-gridsearch-cv)

# Go to first step: [**Course Home Page**](https://www.kaggle.com/mjbahmani/10-steps-to-become-a-data-scientist)
# 
# Go to next step : [**Mathematics and Linear Algebra**](https://www.kaggle.com/mjbahmani/linear-algebra-for-data-scientists)
