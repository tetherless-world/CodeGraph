#!/usr/bin/env python
# coding: utf-8

# # EDA: What does Mktres mean?
# By @marketneutral
# 
# There have been a few Discussion Forum questions about what transformation is used to go from `Raw` to `Mktres` returns. Using just the data, let's see if we can figure this out. 

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# In[ ]:


plt.style.use('seaborn')
plt.rcParams['figure.figsize'] = 12, 7

# In[ ]:


from kaggle.competitions import twosigmanews
env = twosigmanews.make_env()

# In[ ]:


(market_train_df, news_train_df) = env.get_training_data()

# In[ ]:


del news_train_df

# ## Is it just the raw minus the market return?
# One possibility is that the `Raw` return simply subtracts the "market return" for the day. If that's true then the implied market return will be the same for each stock for each day. We can check this easily as:

# In[ ]:


df = (
    market_train_df.
    reset_index().
    sort_values(['assetCode', 'time']).
    set_index(['assetCode','time'])
)

df['implied_mkt_return'] = (
    df.
    groupby('assetCode').
    apply(lambda x: x.returnsClosePrevRaw1 - x.returnsClosePrevMktres1).
    reset_index(0, drop=True)
)

# We can see that the implied market returns here for AAPL and NFLX are **not the same**. As such, the adjustment is **not** simply the difference to market return. If the implied market returns were the same for each stock, then this plot would just be a straight line.

# In[ ]:


plt.scatter(
    df.loc['AAPL.O'].implied_mkt_return,
    df.loc['NFLX.O'].implied_mkt_return,
    alpha=0.6
);
plt.xlabel('Implied Market Return from AAPL');
plt.ylabel('Implied Market Return from NFLX');

# ## Is it the CAPM beta-adjusted return?
# Alternatively, I suspect that this adjustment is just a [CAPM $\beta$](https://en.wikipedia.org/wiki/Beta_(finance). In other words,
# 
# $$mktres = raw - \beta r_{\text{market}}$$
# 
# How can we check this? First, let's extract dataframes, T rows of dates by N columns of stocks, for the `returnsClosePrevRaw1` and `returnsClosePrevMktres`.

# In[ ]:


returnsClosePrevRaw1 = (
    df['returnsClosePrevRaw1'].
    swaplevel().
    unstack()
)

returnsClosePrevMktres1 = (
    df['returnsClosePrevMktres1'].
    swaplevel().
    unstack()
)

# Because not all assets exist for all days, there will be many columns with `NaN` in them for long periods. For this EDA, I need complete returns. As such, I pick a segment of time, and only take stocks that have complete data over that time.

# In[ ]:


num_days = 260*5  # Take 5 years
num_stocks = 200  # Try for 200 stocks but we will get many less due to NaNs

returnsClosePrevRaw1 = \
    returnsClosePrevRaw1.iloc[-num_days:, 0:num_stocks].dropna(axis=1)
num_stocks = len(returnsClosePrevRaw1.columns)
print(num_stocks)

# I'll sync up these two DataFrames and clip outliers at some large numbers (large for what daily returns should be).

# In[ ]:


returnsClosePrevMktres1 = (
    returnsClosePrevMktres1.
    loc[returnsClosePrevRaw1.index][returnsClosePrevRaw1.columns].
    clip(lower=-0.15, upper=0.15)
)

# ### Extracting the hidden market return
# 
# We can use `scikit-learn` *Principal Components Analysis* to extract the latent features in the returns data. One accepted stylized fact about stock markets is the first PC (i.e., the PC which explains the most variance) is the **market**. For the following we will
# 
# - Fit the PCA model to the `returnsClosePrevRaw1`
# - Project the PCs on the returns to extract the time series of the hidden market factor
# - Calcualte *our own* residual return, `residual`
# - And lasty, plot the *heirarchical correlation matrix* of `returnsClosePrevMktres1` and compare that to the hierarchical correlation matrix of `residual`
# - If these two are similar, then we can conclude that the `returnsClosePrevMktres1` is just the single factor beta-adjusted return.
# 
# ### Sidebar: why does PCA apply here?
# 
# PCA finds an axis rotation that has maximum variance explanation. In this case, we want to find out if there are significant common drivers of stock returns across the universe of stocks. There is a great StackOverflow post [here](https://stats.stackexchange.com/questions/2691/making-sense-of-principal-component-analysis-eigenvectors-eigenvalues) that talks about PCA in general and incudes one of my favorite all time GIFs: searching and finding PCA factors in 2 dimensions:
# 
# <img src="https://i.stack.imgur.com/lNHqt.gif">

# In[ ]:


from sklearn.decomposition import PCA
pca = PCA(n_components=15, svd_solver='full')
pca.fit(returnsClosePrevRaw1)

# Let's take a look and see what the PCA reduction found. As we see, there is one big feature and we will assume this is the **market** factor. The first PC explains almost 30% of the total variance.

# In[ ]:


plt.bar(range(15),pca.explained_variance_ratio_);
plt.title('Principal Components Sorted by Variance Explain');
plt.ylabel('% of Total Variance Explained');
plt.xlabel('PC factor number');

# Now let's extract the implied market return and then calculate **our own residualized returns**. We will compare these to `returnsCloseMktres1`.

# In[ ]:


pcs = pca.transform(returnsClosePrevRaw1)

# In[ ]:


# It's always tricky keeping the dimensions right, so I am going to print them for reference.
print(num_stocks)
print(num_days)
print(np.shape(pca.components_))
print(np.shape(pcs))

# In[ ]:


# the market return is the first PC
mkt_return = pcs[:,0].reshape(num_days,1)

# the betas of each stock to the market return are in
# the first column of the components
mkt_beta = pca.components_[0,:].reshape(num_stocks,1)

# the market portion of returns is the projection of one onto the other
mkt_portion = mkt_beta.dot(mkt_return.T).T

# ...and the residual is just the difference
residual = returnsClosePrevRaw1 - mkt_portion

# In[ ]:


print(mkt_return.shape)
print(mkt_portion.shape)

# And lastly for the EDA fun. Of course our residuals will not match `returnsCloseMktres1` exactly even if it were known for sure that there is a one factor model at play. There are many ways to calculate a market beta and many assumptions to make (not the least of which is the lookback period in days for the regression). We are after a general view though of what's going on. Hence we will do the following: compare the correlation matrices of `returnsCloseMktres1` with our residuals and see if they **have the same structure**. If they do, I would argue that the `Mktres` adjustment is just a single factor beta adjustment. If they don't, what else could be happening? Well, perhaps there is a multi-factor beta adjustment (e.g., in our case we could use more than 1 PC). Let's see.
# 
# To calculate the correlation matrices, I use the `LedoitWolf` estimator in `sklearn.covariance` which applies Bayesian shrinkage to reduce esitmation error. The extraction of the correlation matrix from the covariance matrix is an exercise in linear algebra.

# In[ ]:


from sklearn.covariance import LedoitWolf

def get_corr_from_cov(covmat):
    d = np.diag(np.sqrt(np.diag(lw.covariance_)))
    return np.linalg.inv(d).dot(lw.covariance_).dot(np.linalg.inv(d))

lw = LedoitWolf()

lw.fit(returnsClosePrevMktres1)
corr = get_corr_from_cov(lw.covariance_)

lw.fit(residual)
corr2 = get_corr_from_cov(lw.covariance_)

# ### Hierarchical Plot

# In[ ]:


from scipy.spatial import distance
from scipy.cluster import hierarchy

def plot_side_by_side_hm(corr, corr2, title1, title2):
    row_linkage = hierarchy.linkage(
        distance.pdist(corr), method='average')
    row_order = list(map(int, hierarchy.dendrogram(row_linkage, no_plot=True)['ivl']))
    
    col_linkage = hierarchy.linkage(
        distance.pdist(corr.T), method='average')
    col_order = list(map(int, hierarchy.dendrogram(col_linkage, no_plot=True)['ivl']))
    
    corr_swapped = np.copy(corr)
    corr_swapped[:, :] = corr_swapped[row_order, :]
    corr_swapped[:, :] = corr_swapped[:, col_order]

    corr_swapped2 = np.copy(corr2)
    corr_swapped2[:, :] = corr_swapped2[row_order, :]
    corr_swapped2[:, :] = corr_swapped2[:, col_order]

    fig, (ax1, ax2) = plt.subplots(ncols=2)
    fig.tight_layout()
    cs1 = sns.heatmap(corr_swapped, square=True, xticklabels=False, yticklabels=False, cbar=False, ax=ax1, cmap='OrRd')
    cs1.set_title(title1)
    cs2 = sns.heatmap(corr_swapped2, square=True, xticklabels=False, yticklabels=False, cbar=False, ax=ax2, cmap='OrRd')
    cs2.set_title(title2);

plot_side_by_side_hm(
    corr,
    corr2,
    'Hierarchical Correlation Matrix: returnsClosePrevMktres1',
    'Correlation Matrix: Our Residual Est (mapped to <-- hierarchy)'
)

# 
# **The two correlations matrices appear to be roughly equivalent.** Conclusion: the `Mktres` adustment is the subtraction of a per-stock beta adjusted market return from the `raw` return.
# 
# 
# You might wonder what the correlation matrices would look like if they were **not** implying equivalence. For fun, we can compare the correlation matrices of the `returnsPrevCloseRaw1` and `returnsPrevCloseMktres` and see.[](http://)

# In[ ]:


lw.fit(returnsClosePrevRaw1)
corr = get_corr_from_cov(lw.covariance_)

lw.fit(returnsClosePrevMktres1)
corr2 = get_corr_from_cov(lw.covariance_)

# In[ ]:


plot_side_by_side_hm(
    corr,
    corr2,
    'Hierarchical Correlation Matrix: returnsClosePrevRaw1',
    'Correlation Matrix: returnsClosePrevMktres1 (mapped to <-- hierarchy)'
)

# This is my first Kaggle kernel! I hope you liked it. 
