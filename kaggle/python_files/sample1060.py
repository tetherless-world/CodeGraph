#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# In this kernel, we will apply Bayesian inference on Santander Customer Transaction data, which has a binary target and 200 continuous features. We model the target as unknown $Y$ and the features as observation $X$. The prior $p_Y(y)$ reflects our knowledge about the unknown before observation. In this problem, $Y$ is Bernoulli (only two classes) so it can be specified by setting the positive probability, which is usually set as the proportion of the positive class in the data. The likelihood $f_{X|Y}(x|y)$ models the distribution of the observation given that we know the class. The posterior $p_{Y|X}(y|x)$ is our updated knowledge about the unknown after observation.
# 
# The MAP (Maximum A Posteriori) estimator picks the class with the highest posterior probability. For binary classification, it has the same effect as setting a threshold of $0.5$ for the positive posterior probability. The LMS (Least Mean Squares) estimator $\mathbf E[Y|X]$ picks the mean of the posterior distribution. For binary classification, this is just the positive posterior probability $p_{Y|X}(1|x)$, which is what we need to submit for the competition.
# 
# The Bayes rule for this problem is of the form
# $$p_{Y|X}(y|x)=\frac{p_Y(y)f_{X|Y}(x|y)}{\sum_{y'}p_Y(y')f_{X|Y}(x|y')}$$
# 
# Here $X$ represents a sequence of 200 observations $X_0,X_1,\ldots,X_{199}$. We assume that the likelihood distributions are normal and independent. This gives us the Gaussian naive Bayes classifier (Gaussian means normal and naive means independent):
# $$p_{Y|X_0,X_1,\ldots,X_{199}}(y|x_0,x_1,\ldots,x_{199})=\frac{p_Y(y)\prod_{i=0}^{199}f_{X_i|Y}(x_i|y)}{\sum_{y'=0}^1p_Y(y')\prod_{i=0}^{199}f_{X_i|Y}(x_i|y')}$$
# 
# Note that we only require 1 number for the prior and 800 numbers for the likelihood (200 sample means and variances for each of the two classes). "Fitting" is just computing those numbers, and "predicting" is carried out according to the above formula (although we need to operate on the log scale because multiplying many small numbers poses a problem when our machine has limited precision). It is a very simple and efficient model.

# # Checking Assumptions
# 
# The classifier has already been implemented by scikit-learn, so we can use it right away. But we have to make sure that our assumptions hold, i.e., the likelihood distributions are normal and independent.

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# In[ ]:


plt.style.use('bmh')
plt.rcParams['figure.figsize'] = (10, 10)
title_config = {'fontsize': 20, 'y': 1.05}

# In[ ]:


train = pd.read_csv('../input/train.csv')

# In[ ]:


train.head()

# In[ ]:


X_train = train.iloc[:, 2:].values.astype('float64')
y_train = train['target'].values

# We will look at the likelihood distributions by plotting the KDE (Kernel Density Estimates) using the [pandas.DataFrame.plot.kde](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.plot.kde.html). Note that KDE is a similarity-based method so it gets slower with more data. We can speed up by reducing the number of evaluation points (`ind` parameter), but this also decreases the resolution of the plot.

# In[ ]:


pd.DataFrame(X_train[y_train == 0]).plot.kde(ind=100, legend=False)
plt.title('Likelihood KDE Plots for the Negative Class', **title_config);

# In[ ]:


pd.DataFrame(X_train[y_train == 1]).plot.kde(ind=100, legend=False)
plt.title('Likelihood KDE Plots for the Positive Class', **title_config);

# The KDE plots above suggest that the likelihood distributions have different centers and spread. We will standardize them (subtract mean and divide by standard deviation) so that they have zero mean and unit variance. We can use [sklearn.preprocessing.StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) for standardization.

# In[ ]:


from sklearn.preprocessing import StandardScaler

scaled = pd.DataFrame(StandardScaler().fit_transform(X_train))

# In[ ]:


scaled[y_train == 0].plot.kde(ind=100, legend=False)
plt.title('Likelihood KDE Plots for the Negative Class after Standardization', **title_config);

# In[ ]:


scaled[y_train == 1].plot.kde(ind=100, legend=False)
plt.title('Likelihood KDE Plots for the Positive Class after Standardization', **title_config);

# Now the KDE plots above look approximately normal, but some have small bumps on the left or right. We can proceed without doing anything, or we can use quantile transformation to remove the small bumps. It turns out that the transformation provides only marginal improvement in performance (0.001 in cross-validation AUC) despite requiring significantly more computation. In practice, we might choose to skip the transformation. In this competition, however, we will do the transformation for that tiny improvement.
# 
# Ideally, we need to apply the transformation to the features separately for the positive and negative classes. However, we cannot because it becomes a trouble when we are predicting the test data (we do not know the target value). We will instead apply it to the features as a whole so what we really get are normal unconditional distributions $f_{X_i}$ instead of normal conditional distributions $f_{X_i|Y}$, but we hope that the conditional distributions will become more normal as well. We can use [sklearn.preprocessing.QuantileTransformer](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.QuantileTransformer.html) for quantile transformation.

# In[ ]:


from sklearn.preprocessing import QuantileTransformer

transformed = pd.DataFrame(QuantileTransformer(output_distribution='normal').fit_transform(X_train))

# In[ ]:


transformed[y_train == 0].plot.kde(ind=100, legend=False)
plt.title('Likelihood KDE Plots for the Negative Class after Quantile Transformation', **title_config);

# In[ ]:


transformed[y_train == 1].plot.kde(ind=100, legend=False)
plt.title('Likelihood KDE Plots for the Positive Class after Quantile Transformation', **title_config);

# In the KDE plots above, the likelihood distributions have become normal as we desire.
# 
# Independence is difficult to check, but we can check the sample correlation coefficients. Small correlation coefficients mean that there is a weak linear pattern. We visualize the correlation matrix by using [matplotlib.pyplot.imshow](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.imshow.html).

# In[ ]:


plt.imshow(transformed.corr())
plt.colorbar()
plt.title('Correlation Matrix Plot of the Features', **title_config);

# The correlation matrix plot above shows very small correlation coefficients between the features.
# 
# Finally, it is important that $Y$ is dependent on $X$. If $X$ and $Y$ were independent, then the posterior would be equal to the prior  $p_{Y|X}(y|x)=p_Y(y)$, and we would not need to do any calculation! We have already seen above that the positive and negative likelihood distributions are slightly different. Let us look at how the sample means and sample variances differ.

# In[ ]:


plt.hist(transformed[y_train == 0].mean() - transformed[y_train == 1].mean())
plt.title('Histogram of Sample Mean Differences between Two Classes', **title_config);

# In[ ]:


plt.hist(transformed[y_train == 0].var() - transformed[y_train == 1].var())
plt.title('Histogram of Sample Variance Differences between Two Classes', **title_config);

# While the sample mean differences are more or less balanced around zero, the sample variance differences are almost entirely on the negative side. This means that the negative likelihood distributions are more concentrated around their means than the positive ones. These differences add to the discriminative power of the model. The further away the centers of the distributions or the greater the difference in the spread of the distributions, the more it can tell about which class the point is coming from.
# 
# If there are features $X_i$ such that the likelihood distributions are equal — $f_{X_i|Y}(x_i|0)=f_{X_i|Y}(x_i|1)$, their densities will cancel in the numerator and the denominator. These features do not help in classification. So, in some sense, the Bayes classifier performs automatic feature selection.
# 
# Now I have the following puzzle. The plot below shows two features with the least sample variance difference (greatest absolute difference where the variance of the positive class is higher). Surprisingly, the negative class looks more spread out despite having lower sample variance than the positive class.

# In[ ]:


select = (transformed[y_train == 0].var() - transformed[y_train == 1].var()).nsmallest(2).index
plt.scatter(transformed.loc[y_train == 0, select[0]], transformed.loc[y_train == 0, select[1]], alpha=0.5, label='Negative')
plt.scatter(transformed.loc[y_train == 1, select[0]], transformed.loc[y_train == 1, select[1]], alpha=0.5, label='Positive')
plt.xlabel(f'Transformed var_{select[0]}')
plt.ylabel(f'Transformed var_{select[1]}')
plt.title('Positive Class Looks More Concentrated Despite Higher Sample Variance', **title_config)
plt.legend();

# Why? Let us look at the sample mean differences.

# In[ ]:


transformed.loc[y_train == 0, select[0]].mean() - transformed.loc[y_train == 1, select[0]].mean()

# In[ ]:


transformed.loc[y_train == 0, select[1]].mean() - transformed.loc[y_train == 1, select[1]].mean()

# The center of the negative class is above and to the right of that of the positive class, but in the above plot, we see straight lines on the lower and left edges. The bounds have remained even after quantile transformation. It looks like these bounds have prevented the positive class from expanding to the lower and left sides. The bounds are more obvious when you look at the original data.

# In[ ]:


plt.scatter(X_train[y_train == 0, select[0]], X_train[y_train == 0, select[1]], alpha=0.5, label='Negative')
plt.scatter(X_train[y_train == 1, select[0]], X_train[y_train == 1, select[1]], alpha=0.5, label='Positive')
plt.xlabel(f'var_{select[0]}')
plt.ylabel(f'var_{select[1]}')
plt.title('Bounds in Data', **title_config)
plt.legend();

# Despite the presence of bounds, we are going to assume that the transformed data is normal and proceed anyway. We can sample data from normal distributions using [np.random.normal](https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.normal.html) and plot them for comparison.

# In[ ]:


size0 = (y_train == 0).sum()
size1 = y_train.size - size0
x0 = np.random.normal(transformed.loc[y_train == 0, select[0]].mean(),
                      transformed.loc[y_train == 0, select[0]].std(), size=size0)
y0 = np.random.normal(transformed.loc[y_train == 0, select[1]].mean(),
                      transformed.loc[y_train == 0, select[1]].std(), size=size0)
x1 = np.random.normal(transformed.loc[y_train == 1, select[0]].mean(),
                      transformed.loc[y_train == 1, select[0]].std(), size=size1)
y1 = np.random.normal(transformed.loc[y_train == 1, select[1]].mean(),
                      transformed.loc[y_train == 1, select[1]].std(), size=size1)
plt.scatter(x0, y0, alpha=0.5, label='Negative')
plt.scatter(x1, y1, alpha=0.5, label='Positive')
plt.xlabel(f'Simulated var_{select[0]}')
plt.ylabel(f'Simulated var_{select[1]}')
plt.title('Simulated Data for the Puzzle', **title_config)
plt.legend();

# We see above that the positive class spreads more to the lower and left sides than the negative class. Another reason for the illusion is that we have far fewer positive points than negative points.

# # Training and Evaluating the Model
# 
# Now we are ready to train our model. We combine the quantile transformer and Gaussian naive Bayes classifer, [sklearn.naive_bayes.GaussianNB](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html), into a pipeline using [sklearn.pipeline.make_pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.make_pipeline.html).

# In[ ]:


from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import GaussianNB

pipeline = make_pipeline(QuantileTransformer(output_distribution='normal'), GaussianNB())
pipeline.fit(X_train, y_train)

# After training the model, we plot the ROC curve on training data and evaluate the model by computing the training AUC and cross-validation AUC. We can use [sklearn.metrics.roc_curve](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html) to obtain the values for plotting the curve and [sklearn.metrics.auc](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html) for computing the AUC.

# In[ ]:


from sklearn.metrics import roc_curve, auc

fpr, tpr, thr = roc_curve(y_train, pipeline.predict_proba(X_train)[:,1])
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic Plot', **title_config)
auc(fpr, tpr)

# We compute the 10-fold cross-validation score by using [sklearn.model_selection.cross_val_score](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html).

# In[ ]:


from sklearn.model_selection import cross_val_score

cross_val_score(pipeline, X_train, y_train, scoring='roc_auc', cv=10).mean()

# We achieved good AUC on both training and cross-validation. But is this the best that this model can achieve? Let us use simulation to get an estimate of the optimal AUC that this model can achieve. We will draw samples from the normal distribution with the 800 parameters of the likelihood. The amount of samples to draw from each class will be determined by the prior so that the classes have the same proportions as the training data.

# In[ ]:


from sklearn.metrics import roc_auc_score

pipeline.fit(X_train, y_train)
model = pipeline.named_steps['gaussiannb']
size = 1000000
size0 = int(size * model.class_prior_[0])
size1 = size - size0
sample0 = np.concatenate([[np.random.normal(i, j, size=size0)]
                          for i, j in zip(model.theta_[0], np.sqrt(model.sigma_[0]))]).T
sample1 = np.concatenate([[np.random.normal(i, j, size=size1)]
                          for i, j in zip(model.theta_[1], np.sqrt(model.sigma_[1]))]).T
X_sample = np.concatenate([sample0, sample1])
y_sample = np.concatenate([np.zeros(size0), np.ones(size1)])
roc_auc_score(y_sample, model.predict_proba(X_sample)[:,1])

# We see that the optimal AUC under the model is not much different from the cross-validation AUC.

# # Submitting the Test Predictions
# 
# Let us use this model to predict the test data.

# In[ ]:


test = pd.read_csv('../input/test.csv')

# In[ ]:


test.head()

# In[ ]:


X_test = test.iloc[:, 1:].values.astype('float64')
submission = pd.read_csv('../input/sample_submission.csv')
submission['target'] = pipeline.predict_proba(X_test)[:,1]
submission.to_csv('submission.csv', index=False)

# # Conclusion
# 
# The Gaussian naive Bayes classifier performs quite well on Santander Customer Trasaction data. This is because the normality and independence assumptions are closely followed by the data. We have seen that even if the data have been generated by independent normal distributions (according to the model trained on transformed data), we cannot get a better AUC. However, there may still be some other transformation that can improve our model. In my opinion, the normality assumption is not very realistic since some features seem to have lower and upper bounds.
# 
# One can also remove the assumptions and try to use density estimation techniques to model the likelihood distributions. KDE is not very tractable on data of this size. We can also use a Gaussian mixture model or a multivariate normal distribution with the sample covariance matrix from the data. In my experience, they give better training AUC but worse cross-validation AUC. The Gaussian naive Bayes classifier (improved a little bit by quantile transformation) is currently the best Bayesian model for the data. Please tell me if you have found a better one!
