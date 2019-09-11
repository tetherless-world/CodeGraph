#!/usr/bin/env python
# coding: utf-8

# > **Problem overview**
# 
# Long ago, in the distant, fragrant mists of time, there was a competition...
# 
# It was not just any competition. It was a competition that challenged mere mortals to model a 20,000x200 matrix of continuous variables using only 250 training samples... without overfitting. Data scientists ― including Kaggle's very own Will Cukierski ― competed by the hundreds. Legends were made. (Will took 5th place, and eventually ended up working at Kaggle!) People overfit like crazy. It was a Kaggle-y, data science-y madhouse.
# 
# So... we're doing it again.
# 
# This is the next logical step in the evolution of weird competitions. Once again we have 20,000 rows of continuous variables, and a mere handful of training samples. Once again, we challenge you not to overfit. Do your best, model without overfitting, and add, perhaps, to your own legend. In addition to bragging rights, the winner also gets swag. Enjoy!
# 
# Interesting article:
# * https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-classification-in-python/

# In[ ]:


# import data manipulation library
import numpy as np
import pandas as pd

# import data visualization library
import matplotlib.pyplot as plt
import seaborn as sns

# import pystan model class
import pystan

# import sklearn data preprocessing
from sklearn.preprocessing import RobustScaler

# import sklearn model class
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.kernel_ridge import KernelRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# import sklearn model selection
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

# import sklearn model evaluation classification metrics
from sklearn.metrics import accuracy_score, auc, classification_report, confusion_matrix, f1_score, fbeta_score, precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve

# > **Acquiring training and testing data**
# 
# We start by acquiring the training and testing datasets into Pandas DataFrames.

# In[ ]:


# acquiring training and testing data
df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')

# In[ ]:


# visualize head of the training data
df_train.head(n=5)

# In[ ]:


# visualize tail of the testing data
df_test.tail(n=5)

# In[ ]:


# combine training and testing dataframe
df_train['datatype'], df_test['datatype'] = 'training', 'testing'
df_test.insert(1, 'target', np.nan)
df_data = pd.concat([df_train, df_test], ignore_index=True)

# > **Feature exploration, engineering and cleansing**
# 
# Here we generate descriptive statistics that summarize the central tendency, dispersion and shape of a dataset’s distribution together with exploring some data.

# In[ ]:


# countplot function plot - categorical variable (x-axis) vs. categorical variable (y-axis)
def countplot(x = None, y = None, data = None, ncols = 5, nrows = 3):
    fig, axes = plt.subplots(figsize=(4*ncols , 3*nrows), ncols=ncols, nrows=nrows)
    axes = axes.flatten()
    for i, v in enumerate(x): sns.countplot(x=v, hue=y, data=data, ax=axes[i])

# In[ ]:


# boxplot function plot - categorical variable (x-axis) vs. numerical variable (y-axis)
def boxplot(cat = None, num = None, data = None, ncols = 5, nrows = 3):
    fig, axes = plt.subplots(figsize=(4*ncols , 3*nrows), ncols=ncols, nrows=nrows)
    axes = axes.flatten()
    if type(cat) == list:
        for i, v in enumerate(cat): sns.boxplot(x=v, y=num, data=data, ax=axes[i])
    else:
        for i, v in enumerate(num): sns.boxplot(x=cat, y=v, data=data, ax=axes[i])

# In[ ]:


# swarmplot function plot - categorical variable (x-axis) vs. numerical variable (y-axis)
def swarmplot(cat = None, num = None, data = None, ncols = 5, nrows = 3):
    fig, axes = plt.subplots(figsize=(4*ncols , 3*nrows), ncols=ncols, nrows=nrows)
    axes = axes.flatten()
    if type(cat) == list:
        for i, v in enumerate(cat): sns.swarmplot(x=v, y=num, data=data, ax=axes[i])
    else:
        for i, v in enumerate(num): sns.swarmplot(x=cat, y=v, data=data, ax=axes[i])

# In[ ]:


# violinplot function plot - categorical variable (x-axis) vs. numerical variable (y-axis)
def violinplot(cat = None, num = None, data = None, ncols = 5, nrows = 3):
    fig, axes = plt.subplots(figsize=(4*ncols , 3*nrows), ncols=ncols, nrows=nrows)
    axes = axes.flatten()
    if type(cat) == list:
        for i, v in enumerate(cat): sns.violinplot(x=v, y=num, data=data, ax=axes[i])
    else:
        for i, v in enumerate(num): sns.violinplot(x=cat, y=v, data=data, ax=axes[i])

# In[ ]:


# scatterplot function plot - numerical variable (x-axis) vs. numerical variable (y-axis)
def scatterplot(x = None, y = None, data = None, ncols = 5, nrows = 3):
    fig, axes = plt.subplots(figsize=(4*ncols , 3*nrows), ncols=ncols, nrows=nrows)
    axes = axes.flatten()
    for i, xi in enumerate(x): sns.scatterplot(x=xi, y=y, data=data, ax=axes[i])

# In[ ]:


# describe training and testing data
df_data.describe(include='all')

# In[ ]:


# convert dtypes numeric to object
col_convert = ['target']
df_data[col_convert] = df_data[col_convert].astype('object')

# In[ ]:


# list all features type number
col_number = df_data.select_dtypes(include=['number']).columns.tolist()
print('features type number:\n items %s\n length %d' %(col_number, len(col_number)))

# list all features type object
col_object = df_data.select_dtypes(include=['object']).columns.tolist()
print('features type object:\n items %s\n length %d' %(col_object, len(col_object)))

# In[ ]:


# feature exploration: histogram of all numeric features
_ = df_data.hist(bins=20, figsize=(200, 150))

# After extracting all features, it is required to convert category features to numerics features, a format suitable to feed into our Machine Learning models.

# In[ ]:


# feature extraction: target
df_data['target'] = df_data['target'].fillna(-1)

# In[ ]:


# convert category codes for data dataframe
df_data = pd.get_dummies(df_data, columns=['datatype'], drop_first=True)

# In[ ]:


# convert dtypes object to numeric for data dataframe
col_convert = ['target']
df_data[col_convert] = df_data[col_convert].astype(int)

# In[ ]:


# describe data dataframe
df_data.describe(include='all')

# In[ ]:


# verify dtypes object
df_data.info()

# > **Analyze and identify patterns by visualizations**
# 
# Let us generate some correlation plots of the features to see how related one feature is to the next. To do so, we will utilize the Seaborn plotting package which allows us to plot very conveniently as follows.
# 
# The Pearson Correlation plot can tell us the correlation between features with one another. If there is no strongly correlated between features, this means that there isn't much redundant or superfluous data in our training data. This plot is also useful to determine which features are correlated to the observed value.
# 
# The pairplots is also useful to observe the distribution of the training data from one feature to the other.
# 
# The pivot table is also another useful method to observe the impact between features.

# > **Model, predict and solve the problem**
# 
# Now, it is time to feed the features to Machine Learning models.

# In[ ]:


# select all features to evaluate the feature importances
x = df_data[df_data['datatype_training'] == 1].drop(['id', 'target', 'datatype_training'], axis=1)
y = df_data.loc[df_data['datatype_training'] == 1, 'target']

# In[ ]:


# set up lasso regression to find the feature importances
lassoreg = Lasso(alpha=1e-5).fit(x, y)
feat = pd.DataFrame(data=lassoreg.coef_, index=x.columns, columns=['feature_importances']).sort_values(['feature_importances'], ascending=False)

# In[ ]:


# plot the feature importances
feat[(feat['feature_importances'] < -1e-3) | (feat['feature_importances'] > 1e-3)].dropna().plot(y='feature_importances', figsize=(20, 5), kind='bar')
plt.axhline(-0.05, color="grey")
plt.axhline(0.05, color="grey")

# In[ ]:


# list feature importances
model_feat = feat[(feat['feature_importances'] < -0.05) | (feat['feature_importances'] > 0.05)].index

# In[ ]:


# select the important features
x = df_data.loc[df_data['datatype_training'] == 1, model_feat]
y = df_data.loc[df_data['datatype_training'] == 1, 'target']

# In[ ]:


# create scaler to the features
scaler = RobustScaler()
x = scaler.fit_transform(x)

# In[ ]:


# perform train-test (validate) split
x_train, x_validate, y_train, y_validate = train_test_split(x, y, random_state=58, test_size=0.25)

# In[ ]:


# linear regression model setup
model_linreg = LinearRegression()

# linear regression model fit
model_linreg.fit(x_train, y_train)

# linear regression model prediction
model_linreg_ypredict = model_linreg.predict(x_validate)

# linear regression model metrics
model_linreg_rocaucscore = roc_auc_score(y_validate, model_linreg_ypredict)
model_linreg_cvscores = cross_val_score(model_linreg, x, y, cv=20, scoring='roc_auc')
print('linear regression\n  roc auc score: %0.4f, cross validation score: %0.4f (+/- %0.4f)' %(model_linreg_rocaucscore, model_linreg_cvscores.mean(), 2 * model_linreg_cvscores.std()))

# With linear regression submission, the LB score is 0.629. It's seem overfitting.

# In[ ]:


# lasso regression model setup
model_lassoreg = Lasso(alpha=0.01)

# lasso regression model fit
model_lassoreg.fit(x_train, y_train)

# lasso regression model prediction
model_lassoreg_ypredict = model_lassoreg.predict(x_validate)

# lasso regression model metrics
model_lassoreg_rocaucscore = roc_auc_score(y_validate, model_lassoreg_ypredict)
model_lassoreg_cvscores = cross_val_score(model_lassoreg, x, y, cv=20, scoring='roc_auc')
print('lasso regression\n  roc auc score: %0.4f, cross validation score: %0.4f (+/- %0.4f)' %(model_lassoreg_rocaucscore, model_lassoreg_cvscores.mean(), 2 * model_lassoreg_cvscores.std()))

# In[ ]:


# specify the hyperparameter space
params = {
    'alpha': np.logspace(-4, -2, base=10, num=50),
}

# lasso regression grid search model setup
model_lassoreg_cv = GridSearchCV(model_lassoreg, params, iid=False, cv=5)

# lasso regression grid search model fit
model_lassoreg_cv.fit(x_train, y_train)

# lasso regression grid search model prediction
model_lassoreg_cv_ypredict = model_lassoreg_cv.predict(x_validate)

# lasso regression grid search model metrics
model_lassoreg_cv_rocaucscore = roc_auc_score(y_validate, model_lassoreg_cv_ypredict)
model_lassoreg_cv_cvscores = cross_val_score(model_lassoreg_cv, x, y, cv=20, scoring='roc_auc')
print('lasso regression grid search\n  roc auc score: %0.4f, cross validation score: %0.4f (+/- %0.4f)' %(model_lassoreg_cv_rocaucscore, model_lassoreg_cv_cvscores.mean(), 2 * model_lassoreg_cv_cvscores.std()))
print('  best parameters: %s' %model_lassoreg_cv.best_params_)

# With lasso regression submission, the LB score is 0.704. It's seem overfitting.

# In[ ]:


# ridge regression model setup
model_ridgereg = Ridge(alpha=35)

# ridge regression model fit
model_ridgereg.fit(x_train, y_train)

# ridge regression model prediction
model_ridgereg_ypredict = model_ridgereg.predict(x_validate)

# ridge regression model metrics
model_ridgereg_rocaucscore = roc_auc_score(y_validate, model_ridgereg_ypredict)
model_ridgereg_cvscores = cross_val_score(model_ridgereg, x, y, cv=20, scoring='roc_auc')
print('ridge regression\n  roc auc score: %0.4f, cross validation score: %0.4f (+/- %0.4f)' %(model_ridgereg_rocaucscore, model_ridgereg_cvscores.mean(), 2 * model_ridgereg_cvscores.std()))

# In[ ]:


# specify the hyperparameter space
params = {'alpha': np.logspace(-4, 4, base=10, num=50)}

# ridge regression grid search model setup
model_ridgereg_cv = GridSearchCV(model_ridgereg, params, iid=False, cv=5)

# ridge regression grid search model fit
model_ridgereg_cv.fit(x_train, y_train)

# ridge regression grid search model prediction
model_ridgereg_cv_ypredict = model_ridgereg_cv.predict(x_validate)

# ridge regression grid search model metrics
model_ridgereg_cv_rocaucscore = roc_auc_score(y_validate, model_ridgereg_cv_ypredict)
model_ridgereg_cv_cvscores = cross_val_score(model_ridgereg_cv, x, y, cv=20, scoring='roc_auc')
print('ridge regression grid search\n  roc auc score: %0.4f, cross validation score: %0.4f (+/- %0.4f)' %(model_ridgereg_cv_rocaucscore, model_ridgereg_cv_cvscores.mean(), 2 * model_ridgereg_cv_cvscores.std()))
print('  best parameters: %s' %model_ridgereg_cv.best_params_)

# With ridge regression submission, the LB score is 0.690. It's seem overfitting.

# In[ ]:


# elastic net regression model setup
model_elasticnetreg = ElasticNet(alpha=0.01, l1_ratio=0.9)

# elastic net regression model fit
model_elasticnetreg.fit(x_train, y_train)

# elastic net regression model prediction
model_elasticnetreg_ypredict = model_elasticnetreg.predict(x_validate)

# elastic net regression model metrics
model_elasticnetreg_rocaucscore = roc_auc_score(y_validate, model_elasticnetreg_ypredict)
model_elasticnetreg_cvscores = cross_val_score(model_elasticnetreg, x, y, cv=20, scoring='roc_auc')
print('elastic net regression\n  roc auc score: %0.4f, cross validation score: %0.4f (+/- %0.4f)' %(model_elasticnetreg_rocaucscore, model_elasticnetreg_cvscores.mean(), 2 * model_elasticnetreg_cvscores.std()))

# In[ ]:


# specify the hyperparameter space
params = {'alpha': np.logspace(-4, -2, base=10, num=10),
          'l1_ratio': np.linspace(0.1, 0.9, num=5),
}

# elastic net regression grid search model setup
model_elasticnetreg_cv = GridSearchCV(model_elasticnetreg, params, iid=False, cv=5)

# elastic net regression grid search model fit
model_elasticnetreg_cv.fit(x_train, y_train)

# elastic net regression grid search model prediction
model_elasticnetreg_cv_ypredict = model_elasticnetreg_cv.predict(x_validate)

# elastic net regression grid search model metrics
model_elasticnetreg_cv_rocaucscore = roc_auc_score(y_validate, model_elasticnetreg_cv_ypredict)
model_elasticnetreg_cv_cvscores = cross_val_score(model_elasticnetreg_cv, x, y, cv=20, scoring='roc_auc')
print('elastic net regression grid search\n  roc auc score: %0.4f, cross validation score: %0.4f (+/- %0.4f)' %(model_elasticnetreg_cv_rocaucscore, model_elasticnetreg_cv_cvscores.mean(), 2 * model_elasticnetreg_cv_cvscores.std()))
print('  best parameters: %s' %model_elasticnetreg_cv.best_params_)

# In[ ]:


# kernel ridge regression model setup
model_kernelridgereg = KernelRidge(alpha=0.0001, degree=4, kernel='polynomial')

# kernel ridge regression model fit
model_kernelridgereg.fit(x_train, y_train)

# kernel ridge regression model prediction
model_kernelridgereg_ypredict = model_kernelridgereg.predict(x_validate)

# kernel ridge regression model metrics
model_kernelridgereg_rocaucscore = roc_auc_score(y_validate, model_kernelridgereg_ypredict)
model_kernelridgereg_cvscores = cross_val_score(model_kernelridgereg, x, y, cv=20, scoring='roc_auc')
print('kernel ridge regression\n  roc auc score: %0.4f, cross validation score: %0.4f (+/- %0.4f)' %(model_kernelridgereg_rocaucscore, model_kernelridgereg_cvscores.mean(), 2 * model_kernelridgereg_cvscores.std()))

# In[ ]:


# specify the hyperparameter space
params = {'alpha': np.logspace(-4, -2, base=10, num=10),
          'degree': [1, 2, 3, 4, 5],
}

# kernel ridge regression grid search model setup
model_kernelridgereg_cv = GridSearchCV(model_kernelridgereg, params, iid=False, cv=5)

# kernel ridge regression grid search model fit
model_kernelridgereg_cv.fit(x_train, y_train)

# kernel ridge regression grid search model prediction
model_kernelridgereg_cv_ypredict = model_kernelridgereg_cv.predict(x_validate)

# kernel ridge regression grid search model metrics
model_kernelridgereg_cv_rocaucscore = roc_auc_score(y_validate, model_kernelridgereg_cv_ypredict)
model_kernelridgereg_cv_cvscores = cross_val_score(model_kernelridgereg_cv, x, y, cv=20, scoring='roc_auc')
print('kernel ridge regression grid search\n  roc auc score: %0.4f, cross validation score: %0.4f (+/- %0.4f)' %(model_kernelridgereg_cv_rocaucscore, model_kernelridgereg_cv_cvscores.mean(), 2 * model_kernelridgereg_cv_cvscores.std()))
print('  best parameters: %s' %model_kernelridgereg_cv.best_params_)

# In[ ]:


# decision tree regression model setup
model_treereg = DecisionTreeRegressor(splitter='best', min_samples_split=5)

# decision tree regression model fit
model_treereg.fit(x_train, y_train)

# decision tree regression model prediction
model_treereg_ypredict = model_treereg.predict(x_validate)

# decision tree regression model metrics
model_treereg_rocaucscore = roc_auc_score(y_validate, model_treereg_ypredict)
model_treereg_cvscores = cross_val_score(model_treereg, x, y, cv=20, scoring='roc_auc')
print('decision tree regression\n  roc auc score: %0.4f, cross validation score: %0.4f (+/- %0.4f)' %(model_treereg_rocaucscore, model_treereg_cvscores.mean(), 2 * model_treereg_cvscores.std()))

# In[ ]:


# random forest regression model setup
model_forestreg = RandomForestRegressor(n_estimators=100, min_samples_split=3, random_state=58)

# random forest regression model fit
model_forestreg.fit(x_train, y_train)

# random forest regression model prediction
model_forestreg_ypredict = model_forestreg.predict(x_validate)

# random forest regression model metrics
model_forestreg_rocaucscore = roc_auc_score(y_validate, model_forestreg_ypredict)
model_forestreg_cvscores = cross_val_score(model_forestreg, x, y, cv=20, scoring='roc_auc')
print('random forest regression\n  roc auc score: %0.4f, cross validation score: %0.4f (+/- %0.4f)' %(model_forestreg_rocaucscore, model_forestreg_cvscores.mean(), 2 * model_forestreg_cvscores.std()))

# In[ ]:


# stan model setup
model_code = """
    data {
        int N; // the number of training data
        int N2; // the number of testing data
        int K; // the number of features
        int y[N]; // the response variable
        matrix[N,K] X; // the training matrix
        matrix[N2,K] X_test; // the testing matrix
    }
    parameters {
        vector[K] alpha;
        real beta;
    }
    transformed parameters {
        vector[N] y_linear;
        y_linear = beta + X * alpha;
    }
    model {
        alpha ~ cauchy(0, 10); // cauchy distribution
        for (i in 1:K)
            alpha[i] ~ student_t(1, 0, 0.03); // student t distribution
        y ~ bernoulli_logit(y_linear); // bernoulli distribution, logit parameterization
    }
    generated quantities {
        vector[N2] y_pred;
        y_pred = beta + X_test * alpha;
    }
"""

model_data = {
    'N': 250,
    'N2': 19750,
    'K': 300,
    'y': df_data.loc[df_data['datatype_training'] == 1, 'target'],
    'X': df_data[df_data['datatype_training'] == 1].drop(['id', 'target', 'datatype_training'], axis=1),
    'X_test': df_data[df_data['datatype_training'] == 0].drop(['id', 'target', 'datatype_training'], axis=1),
}

model_stan = pystan.StanModel(model_code=model_code)

# stan model fit
model_stan_fitted = model_stan.sampling(data=model_data, seed=58)

# With pystan bernoulli distribution, logit parameterization submission, the LB score is 0.859.

# > **Supply or submit the results**
# 
# Our submission to the competition site Kaggle is ready. Any suggestions to improve our score are welcome.

# In[ ]:


# prepare testing data and compute the observed value
x_test = df_data[df_data['datatype_training'] == 0]
y_test = pd.DataFrame(np.mean(model_stan_fitted.extract(permuted=True)['y_pred'], axis=0),
                      columns=['target'], index=df_data.loc[df_data['datatype_training'] == 0, 'id'])

# In[ ]:


# summit the results
out = pd.DataFrame({'id': y_test.index, 'target': y_test['target']})
out.to_csv('submission.csv', index=False)

# In[ ]:



