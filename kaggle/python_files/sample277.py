#!/usr/bin/env python
# coding: utf-8

# In[ ]:


## Import the required python utilities
from plotly.offline import init_notebook_mode, iplot
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import pandas as pd
import numpy as np

## Import sklearn important modules
from sklearn.decomposition import PCA, SparsePCA, MiniBatchSparsePCA, KernelPCA, IncrementalPCA
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection
from sklearn.decomposition import TruncatedSVD, FastICA, NMF, FactorAnalysis
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import lightgbm as lgb

init_notebook_mode(connected=True)
path = "../input/"

# ## Dataset Decomposition Techniques  
# 
# ### Problem Statement: Santander Value Prediction
# 
# Santander Group wants to identify the value of transactions for each potential customer. This is a first step that Santander needs to nail in order to personalize their services at scale. The dataset can be downloaded from this [link](https://www.kaggle.com/c/santander-value-prediction-challenge/data). In this kernel I have explained different approaches for dataset decomposition. 
# 
# 
# ### Introduction
# 
# 
# The purpose of this kernel is to walkthrough different dataset decomposition techniques and their implementations.   Decomposition of dataset into lower dimensions often becomes an important task while deailing with datasets having larger number of features. Dimensionality Reduction refers to the process of converting a dataset having vast dimensions into a dataset with lesser number of dimensions. This process is done by ensuring that the information conveyed by the original dataset is not lost. 
# 
# Credits:
# - https://www.kaggle.com/arthurtok/interactive-intro-to-dimensionality-reduction
# 
# ### Contents
# 
# 1. Dataset Preparation    
# 2. Feature Statistics    
# 3. Eigen Values and Eigen Vectors   
# 4. Principal Components Analysis   
# &nbsp;&nbsp;&nbsp;&nbsp; 4.1 Finding Right Number of Components   
# &nbsp;&nbsp;&nbsp;&nbsp; 4.2 PCA Implementation    
# &nbsp;&nbsp;&nbsp;&nbsp; 4.3 Variants of PCA  
# 5. Truncated SVD   
# 6. Fast ICA   
# 7. Factor Analysis   
# 8. Non-Negative Matrix Factorization  
# 9. Gaussian Random Projection  
# 10. Sparse Random Projection  
# 11. tSNE Visualization  
# 12. Baseline Model with Decomposition Features  

# ### 1. Dataset Preparation  
# 
# As the first step, load the required dataset. Also separate out the target variable and remove it from the original dataset. This step is done so that entire dataframe can be used directly in decomposition. 

# In[ ]:


train = pd.read_csv(path+'train.csv')

target = train['target']
train = train.drop(["target", "ID"], axis=1)

print ("Rows: " + str(train.shape[0]) + ", Columns: " + str(train.shape[1]))
train.head()

# There are 4459 rows and 4992 features in the dataset which means that this dataset consists of more number of columns than number of rows. Normalize the dataset so that every value is in the same range 

# In[ ]:


standardized_train = StandardScaler().fit_transform(train.values)

# ### 2. Feature Statistics  
# 
# Computing the basic statistics about features such as mean, variance, standard deviation can help to understand about features. In this part, we will compute the following details about the features. 

# In[ ]:


feature_df = train.describe().T
feature_df = feature_df.reset_index().rename(columns = {'index' : 'columns'})
feature_df['distinct_vals'] = feature_df['columns'].apply(lambda x : len(train[x].value_counts()))
feature_df['column_var'] = feature_df['columns'].apply(lambda x : np.var(train[x]))
feature_df['column_std'] = feature_df['columns'].apply(lambda x : np.std(train[x]))
feature_df['column_mean'] = feature_df['columns'].apply(lambda x : np.mean(train[x]))
feature_df['target_corr'] = feature_df['columns'].apply(lambda x : np.corrcoef(target, train[x])[0][1])
feature_df.head()

# #### Variable Variance 
# 
# Variance defines how the data is spread across the mean. It is calulcated by taking the square of difference of every value from the mean value for a variable. One of the statistical intution is that if the feature variance is very less, then the feature will add less contribution to the model. However, I donot follow this blindly as most of the deep learning and boosting modles are robust to such issues. But, variance can give an idea about the features which can be discarded. Atleast the features having zero variance can be discarded because they are essently constant features. (again, these features might not be significant when considered individually, but can be useful in the row wise aggregated features) 

# In[ ]:


len(feature_df[feature_df['column_var'].astype(float) == 0.0])

# So there are 256 columns in the dataset having zero variance ie. they have constant values. 
# 
# Lets plot the variance of the variables. 

# In[ ]:


feature_df = feature_df.sort_values('column_var', ascending = True)
feature_df['column_var'] = (feature_df['column_var'] - feature_df['column_var'].min()) / (feature_df['column_var'].max() - feature_df['column_var'].min())
trace1 = go.Scatter(x=feature_df['columns'], y=feature_df['column_var'], opacity=0.75, marker=dict(color="red"))
layout = dict(height=400, title='Feature Variance', legend=dict(orientation="h"));
fig = go.Figure(data=[trace1], layout=layout);
iplot(fig);

# In[ ]:


trace1 = go.Histogram(x=feature_df[feature_df['column_var'] <= 0.01]['column_var'], opacity=0.45, marker=dict(color="red"))
layout = dict(height=400, title='Distribution of Variable Variance <= 0.01', legend=dict(orientation="h"));
fig = go.Figure(data=[trace1], layout=layout);
iplot(fig);

trace1 = go.Histogram(x=feature_df[feature_df['column_var'] > 0.01]['column_var'], opacity=0.45, marker=dict(color="red"))
layout = dict(height=400, title='Distribution of Variable Variance > 0.01', legend=dict(orientation="h"));
fig = go.Figure(data=[trace1], layout=layout);
iplot(fig);

# So we can see that a large number of variables has variable less than 0.01 and fewer variables has variance greater than 0.01 .
# 
# #### Correlation with Target Variable 
# 
# Pearson’s correlation coefficient is the test statistics that measures the statistical relationship, or association, between two continuous variables. 
# 
# Another statistics test which which can be helpful about the columns is the correlation of the feature with the target variable. High correlated features are good for models for the reverse may not be true. Lets look at what is the distribution of correlations with the target variable in this dataset. 

# In[ ]:


trace1 = go.Histogram(x=feature_df['target_corr'], opacity=0.45, marker=dict(color="green"))
layout = dict(height=400, title='Distribution of correlation with target', legend=dict(orientation="h"));
fig = go.Figure(data=[trace1], layout=layout);
iplot(fig);

# As we can see that most of the variables are not very highly correlated with the target variable, and a majority of the variable have exteremely low correlation with the target. 
# 
# So use of basic statistics is one of the method through which one can get idea about features having statistical significance. The same can be used to handpick important features and discard others.  
# 
# ### 3. Decomposition into EigenVectors and EigenValues  
# 
# In linear algebra, an eigenvector of a linear transformation is a non-zero vector that changes by only a scalar factor when that linear transformation is applied to it. If T is a linear transformation from a vector space V over a field F into itself and v is a vector in V that is not the zero vector, then v is an eigenvector of T if T(v) is a scalar multiple of v. 
# 
# > T(v)= λv
# 
# 
# where λ is a scalar value known as the eigenvalue or characteristic root associated with the eigenvector v. In terms of decomposition, eigen vectors are the principal components for any dataset. Lets visualize the individual and cumulative variance explained by eigen vectors. 

# In[ ]:


# Calculating Eigenvectors and eigenvalues of Cov matirx
mean_vec = np.mean(standardized_train, axis=0)
cov_matrix = np.cov(standardized_train.T)
eig_vals, eig_vecs = np.linalg.eig(cov_matrix)

# Create a list of (eigenvalue, eigenvector) tuples
eig_pairs = [ (np.abs(eig_vals[i]),eig_vecs[:,i]) for i in range(len(eig_vals))]

# Sort the eigenvalue, eigenvector pair from high to low
eig_pairs.sort(key = lambda x: x[0], reverse= True)

# Calculation of Explained Variance from the eigenvalues
tot = sum(eig_vals)

# Individual explained variance
var_exp = [(i/tot)*100 for i in sorted(eig_vals, reverse=True)] 
var_exp_real = [v.real for v in var_exp]

# Cumulative explained variance
cum_var_exp = np.cumsum(var_exp) 
cum_exp_real = [v.real for v in cum_var_exp]

## plot the variance and cumulative variance 
trace1 = go.Scatter(x=train.columns, y=var_exp_real, name="Individual Variance", opacity=0.75, marker=dict(color="red"))
trace2 = go.Scatter(x=train.columns, y=cum_exp_real, name="Cumulative Variance", opacity=0.75, marker=dict(color="blue"))
layout = dict(height=400, title='Variance Explained by Variables', legend=dict(orientation="h", x=0, y=1.2));
fig = go.Figure(data=[trace1, trace2], layout=layout);
iplot(fig);

# So from the above graph we can observe the following points: About 500 features of this dataset can describe about 75% of the explained variance, About 1000 features of this dataset can describe about 90% of the explained variance and About 1500 features of this dataset can describe about 95% of the explained variance   
# 
# 
# ### 4. PCA - Principal Component Analysis    
# 
# Principal Component Analysis is the technique for finding most informative vectors of a high-dimensional datasets. In other words, PCA extracts the important variables in form of components from a datasets containing large number of features. The important features are extracted with the goal to capture maximum possible information from the dataset.  
# 
# The first principal component is a linear combination of dataset features having maximum variance. It determines the direction of highest variability in the data. If the components are uncorrelated, their directions should be orthogonal. This suggests the correlation b/w the components in zero. All succeeding principal component follows the similar concept i.e. they capture the remaining variation without being correlated with the previous component. 
# 
# 
# ### 4.1 Finding Right Number of Components
# 
# We observed from the cumulative frequency graph of eigen vectors that about 1000 variables can give upto 90% explained variance of the dataset. We can use PCA with N number of components and obtain the right number which matches a threshold value of explained variance. 

# In[ ]:


def _get_number_components(model, threshold):
    component_variance = model.explained_variance_ratio_
    explained_variance = 0.0
    components = 0

    for var in component_variance:
        explained_variance += var
        components += 1
        if(explained_variance >= threshold):
            break
    return components

### Get the optimal number of components
pca = PCA()
train_pca = pca.fit_transform(standardized_train)
components = _get_number_components(pca, threshold=0.85)
components

# So, for a threshold value = 0.85, we can choose 993 components. These components will explain about 85% of the variance of the dataset

# In[ ]:


def plot_3_components(x_trans, title):
    trace = go.Scatter3d(x=x_trans[:,0], y=x_trans[:,1], z = x_trans[:,2],
                          name = target, mode = 'markers', text = target, showlegend = False,
                          marker = dict(size = 8, color=x_trans[:,1], 
                          line = dict(width = 1, color = '#f7f4f4'), opacity = 0.5))
    layout = go.Layout(title = title, showlegend= True)
    fig = dict(data=[trace], layout=layout)
    iplot(fig)

def plot_2_components(x_trans, title):
    trace = go.Scatter(x=x_trans[:,0], y=x_trans[:,1], name=target, mode='markers',
        text = target, showlegend = False,
        marker = dict(size = 8, color=x_trans[:,1], line = dict(width = 1, color = '#fefefe'), opacity = 0.7))
    layout = go.Layout(title = title, hovermode= 'closest',
        xaxis= dict(title= 'First Component',
            ticklen = 5, zeroline= False, gridwidth= 2),
        yaxis=dict(title= 'Second Component',
            ticklen = 5, gridwidth = 2), showlegend= True)
    fig = dict(data=[trace], layout=layout)
    iplot(fig)

# ### 4.2 Implementing PCA
# 
# Lets implement the PCA and visualize the first three and two components. Sklearn provides a good implementation of PCA and its variants.  

# In[ ]:


### Implement PCA 
obj_pca = model = PCA(n_components = components)
X_pca = obj_pca.fit_transform(standardized_train)

## Visualize the Components 
plot_3_components(X_pca, 'PCA - First Three Component')
plot_2_components(X_pca, 'PCA - First Two Components')

# We can observe one main property of the components that they are orthogonal to each other, which means that they are uncorrelated. 
# 
# ### 4.3 PCA Variants 
# 
# Sklearn provides different Variants of PCA which can be helpful as well. 
# 
# **4.3.1 Kernel PCA:** 
# 
# KernelPCA is an extension of PCA which achieves non-linear dimensionality reduction through the use of kernels. It has many applications including denoising, compression and structured prediction (kernel dependency estimation). 
# 
# **4.3.2 Incremental PCA:** 
# 
# Incremental PCA works similar to PCA but Depending on the size of the input data, incremental PCA is much more memory efficient. This technique allows for partial computations which almost exactly match the results of PCA while processing the data in a minibatch fashion.   
# 
# **4.3.3 Sparse PCA:**   
# 
# Sparse PCA finds out the set of sparse components that can optimally reconstruct the data. The amount of sparseness is tunable parameter. 
# 
# **4.3.4 Mini Batch Sparse PCA:**    
# 
# Mini Batch Sparse PCA is similar to sparse PCA but it computes the components by taking mini batches at onces from the data. It is faster but less accurate. 

# ### 5. Truncated SVD
# 
# The Singular-Value Decomposition or SVD is a matrix decomposition method for reducing a matrix to its constituent parts in order to make certain subsequent matrix calculations simpler. Truncated SVD is the variant of SVD which is also used for dimentionality reduction. Contrary to PCA, this estimator does not center the data before computing the singular value decomposition. This means it can work very well with sparse matrices efficiently. 

# In[ ]:


### Implement Truncated SVD 
obj_svd = TruncatedSVD(n_components = components)
X_svd = obj_svd.fit_transform(standardized_train)

## Visualize the Components 
plot_3_components(X_svd, 'Truncated SVD - First three components')
plot_2_components(X_svd, 'Truncated SVD - First two components')

# We can observe that results of PCA and Truncated SVD are pretty much similar. This is because PCA is (truncated) SVD on centered data (by per-feature mean substraction). If the data is already centered, those two classes will do the same, which we can observe in the graphs. TruncatedSVD is very useful on large sparse datasets which cannot be centered without making use of too much of memory.
# 
# ### 6. Independent Component Analysis - ICA
# 
# Independent component analysis separates the dataset containing multivariate features into additive subcomponents that are maximally independent. Typically, ICA is not used for dimentionality reduction but for separating the individual components. 

# In[ ]:


### Implement ICA 
obj_ica = FastICA(n_components = 30)
X_ica = obj_ica.fit_transform(standardized_train)

## Visualize the Components 
plot_3_components(X_ica, 'ICA - First three components')
plot_2_components(X_ica, 'ICA - First two components')

# ### 7. Factor Analysis 
# 
# A simple linear generative model with Gaussian latent variables.

# In[ ]:


### Implement Factor Analysis 
obj_fa = FactorAnalysis(n_components = 30)
X_fa = obj_fa.fit_transform(standardized_train)

## Visualize the Components 
plot_3_components(X_fa, 'Factor Analysis - First three components')
# plot_2_components(X, 'Factor Analysis - First two components')

# ### 8. Non Negative Matrix Factorization
# 
# NNMF is the technique which is used to find two non-negative matrices (W, H) whose product approximates the non- negative matrix X. 

# In[ ]:


### Implement NonNegative Matrix Factorization
obj = NMF(n_components = 2)
X_nmf = obj.fit_transform(train)

## Visualize the Components 
# plot_3_components(X, 'NNMF - First three components')
plot_2_components(X_nmf, 'NNMF - First two components')

# ### 9. Gaussian Random Projection  
# 

# In[ ]:


### Implement Gaussian Random Projection
obj_grp = GaussianRandomProjection(n_components = 30, eps=0.1)
X_grp = obj_grp.fit_transform(standardized_train)

## Visualize the Components 
plot_3_components(X_grp, 'Gaussian Random Projection - First three components')
plot_2_components(X_grp, 'Gaussian Random Projection - First two components')

# ### 10. Sparse Random Projection  

# In[ ]:


### Implement Sparse Random Projection
obj_srp = SparseRandomProjection(n_components = 30, eps=0.1)
X_srp = obj_srp.fit_transform(standardized_train)

## Visualize the Components 
plot_3_components(X_srp, 'Sparse Random Projection - First three components')
plot_2_components(X_srp, 'Sparse Random Projection - First two components')

# ### 11. Lets further decompose the dataset into two components:  t- SNE 
# 
# t-SNE was introduced in 2008 as the method for dataset decomposition using non-linear relations.  (t-SNE) t-Distributed Stochastic Neighbor Embedding is a non-linear dimensionality reduction algorithm used for exploring high-dimensional data. It maps multi-dimensional data to two or more dimensions suitable for human observation.  t-SNE is based on probability distributions with random walk on neighborhood graphs to find the structure within the data. The goal is to take a set of points in a high-dimensional space and find a representation of those points in a lower-dimensional space, typically the 2D plane. Lets apply it on the truncated svd components and further decompose the data into two components.
# 
# 

# In[ ]:


tsne_model = TSNE(n_components=2, verbose=1, random_state=42, n_iter=500)
tsne_results = tsne_model.fit_transform(X_svd)

traceTSNE = go.Scatter(
    x = tsne_results[:,0],
    y = tsne_results[:,1],
    name = target,
     hoveron = target,
    mode = 'markers',
    text = target,
    showlegend = True,
    marker = dict(
        size = 8,
        color = '#c94ff2',
        showscale = False,
        line = dict(
            width = 2,
            color = 'rgb(255, 255, 255)'
        ),
        opacity = 0.8
    )
)
data = [traceTSNE]

layout = dict(title = 'TSNE (T-Distributed Stochastic Neighbour Embedding)',
              hovermode= 'closest',
              yaxis = dict(zeroline = False),
              xaxis = dict(zeroline = False),
              showlegend= False,

             )

fig = dict(data=data, layout=layout)
iplot(fig)

# ### 12. Baseline Model with Decomposed Features
# 
# Lets create the baseline models using the decomposed features

# In[ ]:


## add the decomposed features in the train dataset
def _add_decomposition(df, decomp, ncomp, flag):
    for i in range(1, ncomp+1):
        df[flag+"_"+str(i)] = decomp[:, i - 1]

_add_decomposition(train, X_pca, 30, 'pca')
_add_decomposition(train, X_svd, 30, 'svd')
_add_decomposition(train, X_ica, 30, 'ica')
_add_decomposition(train, X_fa, 30, 'fa')
_add_decomposition(train, X_grp, 30, 'grp')
_add_decomposition(train, X_srp, 30, 'srp')

# Prepare the dataset - Create train, test splits and obtain the feature names

# In[ ]:


## create the lists of decomposed and non decomposed features 
all_features = [x for x in train.columns if x not in ["ID", "target"]]
all_features = [x for x in all_features if "_" not in x]
decomposed_features = [x for x in train.columns if "_" in x]

## split the dataset into train test validation
target_log = np.log1p(target.values)
train_x, val_x, train_y, val_y = train_test_split(train, target_log, test_size=0.20, random_state=2018)

# Train the lightgbm model without decomposed features.

# In[ ]:


## create a baseline model with all features 
params = {'learning_rate': 0.01, 
          'max_depth': 16, 
          'boosting': 'gbdt', 
          'objective': 'regression', 
          'metric': 'rmse', 
          'is_training_metric': True, 
          'num_leaves': 144, 
          'feature_fraction': 0.9, 
          'bagging_fraction': 0.7, 
          'bagging_freq': 5, 
          'seed':2018}

## model without decomposed features 
train_X = lgb.Dataset(train_x[all_features], label=train_y)
val_X = lgb.Dataset(val_x[all_features], label=val_y)
model1 = lgb.train(params, train_X, 1000, val_X, verbose_eval=100, early_stopping_rounds=100)

# Lets now train another model which uses only decomposed features 

# In[ ]:


## create a model with decomposed features 
train_X = lgb.Dataset(train_x[decomposed_features], label=train_y)
val_X = lgb.Dataset(val_x[decomposed_features], label=val_y)
model2 = lgb.train(params, train_X, 3000, val_X, verbose_eval=100, early_stopping_rounds=100)

# Lets use all the features in the model, but lets select the important features using random forests 

# In[ ]:


## Find important features using Random Forests 
complete_features = all_features + decomposed_features
model = RandomForestRegressor(n_jobs=-1, random_state=2018)
model.fit(train[complete_features], target)
importances = model.feature_importances_

## get list of important features 
importances_df = pd.DataFrame({'importance': importances, 'feature': complete_features})
importances_df = importances_df.sort_values(by=['importance'], ascending=[False])
important_features = importances_df[:750]['feature'].values

# Create the model with important features 

# In[ ]:


## create a model with important features   
train_X = lgb.Dataset(train_x[important_features], label=train_y)  
val_X = lgb.Dataset(val_x[important_features], label=val_y)  
model3 = lgb.train(params, train_X, 3000, val_X, verbose_eval=100, early_stopping_rounds=100)  

# Predit the output on test data

# In[ ]:


test = pd.read_csv(path+"test.csv")
testid = test.ID.values
test = test.drop('ID', axis = 1)

# Obtain the decomposed components for test data

# In[ ]:


## obtain the components from test data
standardized_test = StandardScaler().fit_transform(test[all_features].values)
tsX_pca = obj_pca.transform(standardized_test)
tsX_svd = obj_svd.transform(standardized_test)
tsX_ica = obj_ica.transform(standardized_test)
tsX_fa  = obj_fa.transform(standardized_test)
tsX_grp = obj_grp.transform(standardized_test)
tsX_srp = obj_srp.transform(standardized_test)

# Add the components to test data

# In[ ]:


## add the components in test data
_add_decomposition(test, tsX_pca, 30, 'pca')
_add_decomposition(test, tsX_svd, 30, 'svd')
_add_decomposition(test, tsX_ica, 30, 'ica')
_add_decomposition(test, tsX_fa, 30, 'fa')
_add_decomposition(test, tsX_grp, 30, 'grp')
_add_decomposition(test, tsX_srp, 30, 'srp')

# Predict the output

# In[ ]:


## create submission file 
pred = np.expm1(model3.predict(test[important_features], num_iteration=model3.best_iteration))
sub = pd.DataFrame()
sub['ID'] = testid
sub['target'] = pred
sub.to_csv('submission.csv', index=False)
sub.head()

# This is the baseline model. To improve this one can perform the following ideas: 
#     
# - extensive feature engineering: aggregated features, feature group statistics, mini batch kmeans clustering etc.  
# - models fine tuning 
# - stacking / ensembling  
# 

# In[ ]:



