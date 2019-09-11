#!/usr/bin/env python
# coding: utf-8

# # Introduction: Random vs Bayesian Optimization Model Tuning
# 
# In this notebook, we will compare random search and Bayesian optimization hyperparameter tuning methods implemented in two previous notebooks.
# 
# * [Intro to Model Tuning: Grid and Random Search](https://www.kaggle.com/willkoehrsen/intro-to-model-tuning-grid-and-random-search)
# * [Automated Model Tuning](https://www.kaggle.com/willkoehrsen/automated-model-tuning)
# 
# In those notebooks we saw results of the methods applied to a limited dataset (10000 observations) but here we will explore results on a complete dataset with 700 + features.  The results in this notebook are from 500 iterations of random search and 400 iterations of Bayesian Optimization (these took about 5 days to run each). We will thoroughly explore the results both visually and statistically, and then implement the best hyperparameter values on a full set of features. After all the hard work in the random search and Bayesian optimization notebooks, now we get to have some fun! 
# 
# # Roadmap
# 
# Our plan of action is as follows:
# 
# 1. High Level Overview
#     * Which method did best? 
# 2. Examine distribution of scores
#     * Are there trends over the course of the search?
# 3. Explore hyperparameter values
#     * Look at values over the course of the search
#     * Identify correlations between hyperparameters and the score
# 4. Perform "meta" machine learning using these results
#     * Fit a linear regression to results and look at coefficients
# 5. Train a model on the full set of features using the best performing values
#     * Try best results from both random search and bayesian optimization
# 6.  Lay out next steps
#     * How can we use these results for this _and other_ problems? 
#     * Are there better methods for hyperparameter optimization
#     
# At each step, we will use plenty of figures and statistics to explore the data. This will be a fun notebook (even though it may not land you at the top of the leaderboard)! 
# 
# ## Recap 
# 
# In the respective notebooks, we examined we performed 1000 iterations of random search and Bayesian optimization on a reduced sample of the dataset (10000 rows). We compared the cross-validation ROC AUC on the training data, the score on a "testing set" (6000 observations) and the score on the real test set when submitted to the competition leaderboard. Results are below:
# 
# | Method                               | Cross Validation Score | Test Score (on 6000 Rows) | Submission to Leaderboard | Iterations to best score |
# |--------------------------------------|------------------------|---------------------------|---------------------------|--------------------------|
# | Random Search                        | 0.73110                | 0.73274                   | 0.782                     | 996                      |
# | Bayesian Hyperparameter Optimization | 0.73448                | 0.73069                   | 0.792                     | 596                      

# __Take these with some skepticism because they were performed on a very small subset of the data!__ 
# 
# For more rigorous results, we will turn to the evaluation metrics from running __500 iterations (with random search)__ and __400+ iterations (with Bayesian Optimization)__ on a full training dataset with about 700 features (the features are from [this notebook](https://www.kaggle.com/jsaguiar/updated-0-792-lb-lightgbm-with-simple-features) by [Aguiar](https://www.kaggle.com/jsaguiar)). These iterations took around 6 days on a machine with 128 GB of RAM so they will not run in a kernel! The Bayesian Optimization method is still running and I will update the results as they finish.
# 
# __In this notebook  we will focus only on the results and building the best model, so for the explanations of the methods, refer to the previous notebooks! __
# 
# # Overall Results
# 
# First, let's start with the most basic question: which model produced the highest cross validation ROC AUC score (using 5 folds) on the training dataset?

# In[ ]:


# pandas and numpy for data manipulation
import pandas as pd
import numpy as np

# Visualizations!
import matplotlib.pyplot as plt
import seaborn as sns

# Plot formatting and default style
plt.rcParams['font.size'] = 18
plt.style.use('fivethirtyeight')

# In[ ]:


# Read in data and sort
random = pd.read_csv('../input/home-credit-model-tuning/random_search_simple.csv').sort_values('score', ascending = False).reset_index()
opt = pd.read_csv('../input/home-credit-model-tuning/bayesian_trials_simple.csv').sort_values('score', ascending = False).reset_index()

print('Best score from random search:         {:.5f} found on iteration: {}.'.format(random.loc[0, 'score'], random.loc[0, 'iteration']))
print('Best score from bayesian optimization: {:.5f} found on iteration: {}.'.format(opt.loc[0, 'score'], opt.loc[0, 'iteration']))

# Well, there you go! __Random search slightly outperformed  Bayesian optimization and found a higher cross validation model in far fewer iterations.__ However, as we will shortly see, this does not mean random search is the better hyperparameter optimization method. 
# 
# When submitted to the competition (at the end of this notebook):
# 
# * __Random search results scored 0.790__
# * __Bayesian optimization results scored 0.791__
# 
# What were the best model hyperparameters from both methods?
# 
# ####  Random Search best Hyperparameters

# In[ ]:


import pprint
import ast

keys = []
for key, value in ast.literal_eval(random.loc[0, 'hyperparameters']).items():
    print(f'{key}: {value}')
    keys.append(key)

# #### Bayesian Optimization best Hyperparameters

# In[ ]:


for key in keys:
    print('{}: {}'.format(key, ast.literal_eval(opt.loc[0, 'hyperparameters'])[key]))

# If we compare the individual values, we actually see that they are fairly close together when we consider the entire search grid! 
# 
# ## Distribution of Scores
# 
# Let's plot the distribution of scores for both models in a kernel density estimate plot.

# In[ ]:


# Kdeplot of model scores
plt.figure(figsize = (10, 6))
sns.kdeplot(opt['score'], label = 'Bayesian Opt')
sns.kdeplot(random['score'], label = 'Random Search')
plt.xlabel('Score (5 Fold Validation ROC AUC)'); plt.ylabel('Density');
plt.title('Random Search and Bayesian Optimization Results');

# Bayesian optimization did not produce the highest individual score, but it did tend to spend more time evaluating "better" values of hyperparameters. __Random search got lucky and found the best values but Bayesian optimization tended to "concentrate" on better-scoring values__. That's pretty much what we expect: random search does a good job of exploring the search space which means it will probably happen upon a high-scoring set of values (if the space is not extremely high-dimensional) while Bayesian optimization will tend to focus on a set of values that yield higher scores. __If all you wanted was the conclusion, then you're probably good to go. If you really enjoy making plots and doing exploratory data analysis and want to gain a better understanding of how these methods work, then read on!__ In the next few sections, we will thoroughly explore these results.
# 
# Our plan for going through the results is as follows:
# 
# * Distribution of scores
#     * Overall distribution
#     * Score versus the iteration (did scores improve as search progressed)
# * Distribution of hyperparameters
#     * Overall distribution including the hyperparameter grid for a reference
#     * Hyperparameters versus iteration to look at _evolution_ of values
# * Hyperparameter values versus the score
#     * Do scores improve with certain values of hyperparameters (correlations)
#     * 3D plots looking at effects of 2 hyperparameters at a time on the score
# * Additional Plots
#     * Time to run each evaluation for Bayesian optimization
#     * Correlation heatmaps of hyperparameters with score
#     
# There will be all sorts of plots: heatmaps, 3D scatterplots, density plots, bar charts (hey even bar charts can be helpful!)
# 
# After going through the results, we will do a little meta-machine learning, and implement the best model on the full set of features.

# # Distribution of Scores
# 
# We already saw the kernel density estimate plot, so let's go on to a bar plot. First we'll get the data in a long format.

# In[ ]:


random['set'] = 'random'
scores = random[['score', 'iteration', 'set']]

opt['set'] = 'opt'
scores = scores.append(opt[['set', 'iteration', 'score']], sort = True)
scores.head()

# In[ ]:


plt.figure(figsize = (12, 6))

plt.subplot(121)
plt.hist(random['score'], bins = 20, color = 'blue', edgecolor = 'k')
plt.xlim((0.72, 0.80))
plt.xlabel('Score'); plt.ylabel("Count"); plt.title('Random Search Distribution of Scores');

plt.subplot(122)
plt.hist(opt['score'], bins = 20, color = 'blue', edgecolor = 'k')
plt.xlim((0.72, 0.80))
plt.xlabel('Score'); plt.ylabel("Count"); plt.title('Bayes Opt Search Distribution of Scores');

# Keep in mind that random search ran for more iterations (as of now). Even so, we can see that Bayesian Optimization tended to produce much more higher cross validation scores. Let's look at the statistical averages:

# In[ ]:


scores.groupby('set')['score'].agg(['mean', 'max', 'min', 'std', 'count'])

# If we are going by mean, then Bayesian optimization is the clear winner. If we go by high score, then random search just wins out. 

# ## Score versus Iteration
# 
# Now, to see if either method improves over the course of the search, we need to plot the score as a function of the iteration. 

# In[ ]:


plt.rcParams['font.size'] = 16

best_random_score = random.loc[0, 'score']
best_random_iteration = random.loc[0, 'iteration']

best_opt_score = opt.loc[0, 'score']
best_opt_iteration = opt.loc[0, 'iteration']

sns.lmplot('iteration', 'score', hue = 'set', data = scores, size = 8)
plt.scatter(best_random_iteration, best_random_score, marker = '*', s = 400, c = 'blue', edgecolor = 'k')
plt.scatter(best_opt_iteration, best_opt_score, marker = '*', s = 400, c = 'red', edgecolor = 'k')
plt.xlabel('Iteration'); plt.ylabel('ROC AUC'); plt.title("Validation ROC AUC versus Iteration");

# Again keeping in mind that Bayesian optimization has not yet finished, we can see a clear upward trend for this method and no trend whatsoever for random search. 
# 
# ### Linear Regression of Scores versus Iteration
# 
# To show that Bayesian optimization improves over time, we can regress the score by the iteration. Then, we can use this to extrapolate into the future, __a wildly inappropriate technique in this case, but fun nonetheless!__
# 
# Here we use `np.polyfit` with a degree of 1 for the linear regression (you can compare the results with `LinearRegression`  from `sklearn.linear_model`.

# In[ ]:


random_fit = np.polyfit(random['iteration'], random['score'], 1)
print('Random search slope: {:.8f}'.format(random_fit[0]))

# The random search slope is basically zero. 

# In[ ]:


opt_fit = np.polyfit(opt['iteration'], opt['score'], 1)
print('opt search slope: {:.8f}'.format(opt_fit[0]))

# In[ ]:


opt_fit[0] / random_fit[0]

# The Bayesian slope is about 15 times greater than that of random search! What happens if we say run these methods for 10,000 iterations?

# In[ ]:


print('After 10,000 iterations, the random score is: {:.5f}.'.format(
random_fit[0] * 1e5 + random_fit[1]))

# In[ ]:


print('After 10,000 iterations, the bayesian score is: {:.5f}.'.format(
opt_fit[0] * 1e5 + opt_fit[1]))

# Incredible! I told you this was wildly inappropriate. Nonetheless, the slope does indicate that Bayesian optimization "learns" the hyperparameter values that do better over time. It then concentrates on evaluating these rather than spending time exploring other values as does random search. This means it can get stuck in a local optimum and can tend to __exploit__ values rather than continue to __explore__.
# 
# Now we will move on to the actual values of the hyperparameters.

# # Hyperparameter Values
# 
# For each hyperparameter, we will plot the values tried by both searches as well as the reference distribution (which was the same in both cases, just a grid for random and distributions for Bayesian). We would expect the random search to almost exactly match the reference - it will converge on the reference given enough iterations.
# 
# First, we will process the results into a dataframe where each column is one hyperparameter. Saving the file converted the dictionary into a string, so we use `ast.literal_eval` to convert back to a dictionary before adding as a row in the dataframe.

# In[ ]:


import ast
def process(results):
    """Process results into a dataframe with one column per hyperparameter"""
    
    results = results.copy()
    results['hyperparameters'] = results['hyperparameters'].map(ast.literal_eval)
    
    # Sort with best values on top
    results = results.sort_values('score', ascending = False).reset_index(drop = True)
    
     # Create dataframe of hyperparameters
    hyp_df = pd.DataFrame(columns = list(results.loc[0, 'hyperparameters'].keys()))

    # Iterate through each set of hyperparameters that were evaluated
    for i, hyp in enumerate(results['hyperparameters']):
        hyp_df = hyp_df.append(pd.DataFrame(hyp, index = [0]), 
                               ignore_index = True, sort= True)
        
    # Put the iteration and score in the hyperparameter dataframe
    hyp_df['iteration'] = results['iteration']
    hyp_df['score'] = results['score']
    
    return hyp_df

# In[ ]:


random_hyp = process(random)
opt_hyp = process(opt)

random_hyp.head()

# Next we define the hyperparameter grid that was used (the same ranges applied in both searches).

# In[ ]:


# Hyperparameter grid
param_grid = {
    'is_unbalance': [True, False],
    'boosting_type': ['gbdt', 'goss', 'dart'],
    'num_leaves': list(range(20, 150)),
    'learning_rate': list(np.logspace(np.log10(0.005), np.log10(0.5), base = 10, num = 1000)),
    'subsample_for_bin': list(range(20000, 300000, 20000)),
    'min_child_samples': list(range(20, 500, 5)),
    'reg_alpha': list(np.linspace(0, 1)),
    'reg_lambda': list(np.linspace(0, 1)),
    'colsample_bytree': list(np.linspace(0.6, 1, 10)),
    'subsample': list(np.linspace(0.5, 1, 100))
}

# # Distributions of Search Values
# 
# Below are the kernel density estimate plots for each hyperparameter. The dashed vertical lines indicate the "optimal" value found in the respective searches. 
# 
# We start with the learning rate:

# In[ ]:


best_random_hyp = random_hyp.loc[0, :]
best_opt_hyp = opt_hyp.loc[0, :]

# In[ ]:


plt.figure(figsize = (20, 8))
plt.rcParams['font.size'] = 18

# Density plots of the learning rate distributions 
sns.kdeplot(param_grid['learning_rate'], label = 'Sampling Distribution', linewidth = 4, color = 'k')
sns.kdeplot(random_hyp['learning_rate'], label = 'Random Search', linewidth = 4, color = 'blue')
sns.kdeplot(opt_hyp['learning_rate'], label = 'Bayesian', linewidth = 4, color = 'green')
plt.vlines([best_random_hyp['learning_rate']],
           ymin = 0.0, ymax = 50.0, linestyles = '--', linewidth = 4, colors = ['blue'])
plt.vlines([best_opt_hyp['learning_rate']],
           ymin = 0.0, ymax = 50.0, linestyles = '--', linewidth = 4, colors = ['green'])
plt.legend()
plt.xlabel('Learning Rate'); plt.ylabel('Density'); plt.title('Learning Rate Distribution');

print('Best value from random search: {:.5f}.'.format(best_random_hyp['learning_rate']))
print('Best value from Bayesian: {:.5f}.'.format(best_opt_hyp['learning_rate']))

# Even though the search domain extended from 0.005 to 0.2, both optimal values clustered around a lower value. Perhaps this tells us we should concentrate further searches in this area below 0.02?
# 
# That code was a little tedious, so let's write a function that makes the same code for any hyperparameter (feel free to pick your own colors!).

# In[ ]:


def plot_hyp_dist(hyp):
    """Plots distribution of hyp along with best values of hyp as vertical line"""
    plt.figure(figsize = (16, 6))
    plt.rcParams['font.size'] = 18

    # Density plots of the learning rate distributions 
    sns.kdeplot(param_grid[hyp], label = 'Sampling Distribution', linewidth = 4, color = 'k')
    sns.kdeplot(random_hyp[hyp], label = 'Random Search', linewidth = 4, color = 'blue')
    sns.kdeplot(opt_hyp[hyp], label = 'Bayesian', linewidth = 4, color = 'green')
    plt.vlines([best_random_hyp[hyp]],
               ymin = 0.0, ymax = 50.0, linestyles = '--', linewidth = 4, colors = ['blue'])
    plt.vlines([best_opt_hyp[hyp]],
               ymin = 0.0, ymax = 50.0, linestyles = '--', linewidth = 4, colors = ['green'])
    plt.legend()
    plt.xlabel(hyp); plt.ylabel('Density'); plt.title('{} Distribution'.format(hyp));

    print('Best value from random search: {:.5f}.'.format(best_random_hyp[hyp]))
    print('Best value from Bayesian: {:.5f}.'.format(best_opt_hyp[hyp]))
    plt.show()

# In[ ]:


plot_hyp_dist('min_child_samples')

# We can do this for all of the hyperparameters. These results can be used to inform further searches. They can even be used to define a grid search over a concentrated region. The problem with grid search is the insane compuational and time costs involved, and a smaller hyperparameter grid will help immensely! 

# In[ ]:


plot_hyp_dist('num_leaves')

# In[ ]:


plot_hyp_dist('reg_alpha')

# In[ ]:


plot_hyp_dist('reg_lambda')

# The `reg_alpha` and `reg_lambda` best scores seem to complement one another for Bayesian optimization. In other words, if either `reg_lambda` or `reg_alpha` is high (say greater than 0.5), then the other should be low (below 0.5). These hyperparameters control a penalty placed on the weights of the trees and thus are meant to control overfitting. It might make sense if only one needs to be high then.

# In[ ]:


plot_hyp_dist('subsample_for_bin')

# In[ ]:


plot_hyp_dist('colsample_bytree')

# ### Boosting Type
# 
# The boosting type deserves its own section because it is a categorical variable, and because as we will see, it has an outsized effect on model performance. First, let's calculate statistics grouped by boosting type for each search method.

# In[ ]:


random_hyp.groupby('boosting_type')['score'].agg(['mean', 'max', 'min', 'std', 'count'])

# In[ ]:


opt_hyp.groupby('boosting_type')['score'].agg(['mean', 'max', 'min', 'std', 'count'])

# In both search methods, the `gbdt` (gradient boosted decision tree) and `dart` (dropout meets additive regression tree) do much better than `goss` (gradient based one-sided sampling). `gbdt` does the best on average (and for the max), so it might make sense to use that method in the future! Let's view the results as a barchart:

# In[ ]:


plt.figure(figsize = (16, 6))

plt.subplot(121)
random_hyp.groupby('boosting_type')['score'].agg(['mean', 'max', 'min', 'std'])['mean'].plot.bar(color = 'b')
plt.ylabel('Score'); plt.title('Random Search Boosting Type Scores', size = 14);

plt.subplot(122)
opt_hyp.groupby('boosting_type')['score'].agg(['mean', 'max', 'min', 'std'])['mean'].plot.bar(color = 'b')
plt.ylabel('Score'); plt.title('Bayesian Boosting Type Scores', size = 14);

# __`gbdt` (or `dart`) it should be! Notice that random search tried `gbdt` about the same number of times as the other two (since it selected with no reasoning) while Bayesian optimization tried `gbdt` much more often. __
# 
# Since `gbdt` supports `subsample` (using on a sample of the observations to train on in every tree) we can plot the distribution of `subsample` where `boosting_type=='gbdt'`. We also show the reference distribution.

# In[ ]:


plt.figure(figsize = (20, 8))
plt.rcParams['font.size'] = 18

# Density plots of the learning rate distributions 
sns.kdeplot(param_grid['subsample'], label = 'Sampling Distribution', linewidth = 4, color = 'k')
sns.kdeplot(random_hyp[random_hyp['boosting_type'] == 'gbdt']['subsample'], label = 'Random Search', linewidth = 4, color = 'blue')
sns.kdeplot(opt_hyp[opt_hyp['boosting_type'] == 'gbdt']['subsample'], label = 'Bayesian', linewidth = 4, color = 'green')
plt.vlines([best_random_hyp['subsample']],
           ymin = 0.0, ymax = 50.0, linestyles = '--', linewidth = 4, colors = ['blue'])
plt.vlines([best_opt_hyp['subsample']],
           ymin = 0.0, ymax = 50.0, linestyles = '--', linewidth = 4, colors = ['green'])
plt.legend()
plt.xlabel('Subsample'); plt.ylabel('Density'); plt.title('Subsample Distribution');

print('Best value from random search: {:.5f}.'.format(best_random_hyp['subsample']))
print('Best value from Bayesian: {:.5f}.'.format(best_opt_hyp['subsample']))

# There is a significant disagreement between the two methods on the optimal value for `subsample`. Perhaps we would want to leave this as a wide distribution in any further searches (although some subsampling does look to be beneficial).

# Finally, we can look at the instance of `is_unbalance`, a hyperparameter that tells LightGBM whether or not to treat the problem as unbalance classification.

# In[ ]:


random_hyp.groupby('is_unbalance')['score'].agg(['mean', 'max', 'min', 'std', 'count'])

# In[ ]:


opt_hyp.groupby('is_unbalance')['score'].agg(['mean', 'max', 'min', 'std', 'count'])

# __According to the average score, it pretty much does not matter if this hyperparameter is `True` or `False`.__ To be honest, I'm not sure what difference this is supposed to make, so anyone who wants can fill me in!

# # Hyperparameters versus Iteration
# 
# Next we will take a look at the __evolution__ of the Bayesian search (random search shows no pattern as expected) by graphing the values versus the iteration. This can inform us the direction in which the search was heading in terms of where the values tended to cluster. Given these graphs, we might then be able to extrapolate values that lead to even higher scores (or maybe not, _extrapolation is dangerous_!)
# 
# The black star in the plots below signifies the best scoring value.

# In[ ]:


fig, axs = plt.subplots(1, 4, figsize = (24, 6))
i = 0

# Plot of four hyperparameters
for i, hyper in enumerate(['colsample_bytree', 'learning_rate', 'min_child_samples', 'num_leaves']):
        opt_hyp[hyper] = opt_hyp[hyper].astype(float)
        # Scatterplot
        sns.regplot('iteration', hyper, data = opt_hyp, ax = axs[i])
        axs[i].scatter(best_opt_hyp['iteration'], best_opt_hyp[hyper], marker = '*', s = 200, c = 'k')
        axs[i].set(xlabel = 'Iteration', ylabel = '{}'.format(hyper), title = '{} over Search'.format(hyper));

plt.tight_layout()

# In[ ]:


fig, axs = plt.subplots(1, 3, figsize = (18, 6))
i = 0

# Plot of four hyperparameters
for i, hyper in enumerate(['reg_alpha', 'reg_lambda', 'subsample_for_bin']):
        opt_hyp[hyper] = opt_hyp[hyper].astype(float)
        # Scatterplot
        sns.regplot('iteration', hyper, data = opt_hyp, ax = axs[i])
        axs[i].scatter(best_opt_hyp['iteration'], best_opt_hyp[hyper], marker = '*', s = 200, c = 'k')
        axs[i].set(xlabel = 'Iteration', ylabel = '{}'.format(hyper), title = '{} over Search'.format(hyper));

plt.tight_layout()

# We want to be careful about placing too much value in these results, because remember, the Bayesian optimization could have found a local minimum of the cross validation loss that it is exploting. Moreover, the trends here are generally pretty small. It is encouraging that the best value was found close to the end of the search indicating cross validation scores were continuing to improve. 
# 
# Next, we can look at the values of the score as a function of the hyperparameter values. This is again a dangerous area! 

# # Plots of Hyperparameters vs Score
# 
# ![](http://)These next plots show the value of a single hyperparameter versus the score. We want to avoid placing too much emphasis on these graphs, because we are not changing one hyperparameter at a time. Therefore, if there are trends, it might not be solely due to the single hyperparameter we show. A truly accurate grid would be 10-dimensional and show the values of __all__ hyperparameters and the resulting score. If we could understand a __10-dimensional__ graph, then we might be able to figure out the optimal combination of hyperparameters! 

# In[ ]:


random_hyp['set'] = 'Random Search'
opt_hyp['set'] = 'Bayesian'

# Append the two dataframes together
hyp = random_hyp.append(opt_hyp, ignore_index = True, sort = True)
hyp.head()

# In[ ]:


fig, axs = plt.subplots(1, 4, figsize = (24, 6))
i = 0

# Plot of four hyperparameters
for i, hyper in enumerate(['colsample_bytree', 'learning_rate', 'min_child_samples', 'num_leaves']):
        random_hyp[hyper] = random_hyp[hyper].astype(float)
        # Scatterplot
        sns.regplot(hyper, 'score', data = random_hyp, ax = axs[i], color = 'b', scatter_kws={'alpha':0.6})
        axs[i].scatter(best_random_hyp[hyper], best_random_hyp['score'], marker = '*', s = 200, c = 'b', edgecolor = 'k')
        axs[i].set(xlabel = '{}'.format(hyper), ylabel = 'Score', title = 'Score vs {}'.format(hyper));
        
        opt_hyp[hyper] = opt_hyp[hyper].astype(float)
        # Scatterplot
        sns.regplot(hyper, 'score', data = opt_hyp, ax = axs[i], color = 'g', scatter_kws={'alpha':0.6})
        axs[i].scatter(best_opt_hyp[hyper], best_opt_hyp['score'], marker = '*', s = 200, c = 'g', edgecolor = 'k')

plt.legend()
plt.tight_layout()



# __The only clear distinction is that the score decreases as the learning rate increases.__ Of course, we cannot say whether that is due to the learning rate itself, or some other factor (we will look at the interplay between the learning rate and the number of esimators shortly). The learning rate domain was on a logarithmic scale, so it's most accurate for the plot to be as well (unfortunately I cannot get this to work yet).

# In[ ]:


# hyper = 'learning_rate'

# fig, ax = plt.subplots(1, 1, figsize = (6, 6))

# random_hyp[hyper] = random_hyp[hyper].astype(float)
# # Scatterplot
# sns.regplot(hyper, 'score', data = random_hyp, ax = ax, color = 'b', scatter_kws={'alpha':0.6})
# ax.scatter(best_random_hyp[hyper], best_random_hyp['score'], marker = '*', s = 200, c = 'b', edgecolor = 'k')

# opt_hyp[hyper] = opt_hyp[hyper].astype(float)
# # Scatterplot
# sns.regplot(hyper, 'score', data = opt_hyp, ax = ax, color = 'g', scatter_kws={'alpha':0.6})
# ax.scatter(best_opt_hyp[hyper], best_opt_hyp['score'], marker = '*', s = 200, c = 'g', edgecolor = 'k')

# ax.set(xlabel = '{}'.format(hyper), ylabel = 'Score', title = 'Score vs {}'.format(hyper))
# ax.set(xscale = 'log');

# Now for the next four hyperparameters versus the score.

# In[ ]:


fig, axs = plt.subplots(1, 4, figsize = (24, 6))
i = 0

# Plot of four hyperparameters
for i, hyper in enumerate(['reg_alpha', 'reg_lambda', 'subsample_for_bin', 'subsample']):
        random_hyp[hyper] = random_hyp[hyper].astype(float)
        # Scatterplot
        sns.regplot(hyper, 'score', data = random_hyp, ax = axs[i], color = 'b', scatter_kws={'alpha':0.6})
        axs[i].scatter(best_random_hyp[hyper], best_random_hyp['score'], marker = '*', s = 200, c = 'b', edgecolor = 'k')
        axs[i].set(xlabel = '{}'.format(hyper), ylabel = 'Score', title = 'Score vs {}'.format(hyper));
        
        opt_hyp[hyper] = opt_hyp[hyper].astype(float)
        # Scatterplot
        sns.regplot(hyper, 'score', data = opt_hyp, ax = axs[i], color = 'g', scatter_kws={'alpha':0.6})
        axs[i].scatter(best_opt_hyp[hyper], best_opt_hyp['score'], marker = '*', s = 200, c = 'g', edgecolor = 'k')

plt.legend()
plt.tight_layout()



# There are not any strong trends here. Next we will try to look at two hyperparameters simultaneously versus the score in a 3-dimensional plot. This makes sense for hyperparameters that work in concert, such as the learning rate and the number of esimators or the two regularization values.

# ## 3D Plots 
# 
# To try and examine the simultaneous effects of hyperparameters, we can make 3D plots with 2 hyperparameters and the score. A truly accurate plot would be 10-D (one for each hyperparameter) but in this case we will stick to 3 dimensions. 3D plots can be made in matplotlib by import `Axes3D` and specifying the `3d` projection in a call to `.add_subplot`

# In[ ]:


from mpl_toolkits.mplot3d import Axes3D
plt.rcParams['axes.labelpad'] = 12

# First up is `reg_alpha` and `reg_lambda`. These control the amount of regularization on each decision tree and help to prevent overfitting to the training data.

# In[ ]:


fig = plt.figure(figsize = (10, 10))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(random_hyp['reg_alpha'], random_hyp['reg_lambda'],
           random_hyp['score'], c = random_hyp['score'], 
           cmap = plt.cm.seismic_r, s = 40)

ax.set_xlabel('Reg Alpha')
ax.set_ylabel('Reg Lambda')
ax.set_zlabel('Score')

plt.title('Score as Function of Reg Lambda and Alpha');

# In[ ]:


best_random_hyp

# It's a little difficult to tell much from this plot. If we look at the best values and then look at the plot, we can see that scores do tend to be higher around 0.9 for `reg_alpha`and 0.2 for `reg_lambda`.  Later, we'll make the same plot for the Bayesian Optimization for comparison.

# The next plot is learning rate and number of estimators versus the score. __Remember that the number of estimators was selected using early stopping for 100 rounds with 5-fold cross validation__. The number of estimators __was not__ a hyperparameter in the grid that we searched over. Early stopping is a more efficient method of finding the best number of estimators than including it in a search (based on my limited experience)!

# In[ ]:


fig = plt.figure(figsize = (10, 10))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(random_hyp['learning_rate'], random_hyp['n_estimators'],
           random_hyp['score'], c = random_hyp['score'], 
           cmap = plt.cm.seismic_r, s = 40)

ax.set_xlabel('Learning Rate')
ax.set_ylabel('Number of Estimators')
ax.set_zlabel('Score')

plt.title('Score as Function of Learning Rate and Estimators', size = 16);

# Here there appears to be a clear trend: a lower learning rate leads to higher values! What does the plot of just learning rate versus number of estimators look like?

# In[ ]:


plt.figure(figsize = (8, 7))
plt.plot(random_hyp['learning_rate'], random_hyp['n_estimators'], 'ro')
plt.xlabel('Learning Rate'); plt.ylabel('N Estimators'); plt.title('Number of Estimators vs Learning Rate');

# This plot is very easy to interpret: the lower the learning rate, the more estimators that will be trained. From our knowledge of the model, this makes sense: each individual decision trees contribution is lessened as the learning rate is decreased leading to a need for more decision trees in the ensemble. Moreover, from the previous graphs, it appears that decreasing the learning rate increases the model score.

# ### Function for 3D plotting
# 
# Any time you write code more than twice, it should be encoded into a function! That's what the next code block is for: putting this code into a function that we can use many times! This function can be used for __any__ 3d plotting needs.

# In[ ]:


from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def plot_3d(x, y, z, df, cmap = plt.cm.seismic_r):
    """3D scatterplot of data in df"""

    fig = plt.figure(figsize = (10, 10))
    
    ax = fig.add_subplot(111, projection='3d')
    
    # 3d scatterplot
    ax.scatter(df[x], df[y],
               df[z], c = df[z], 
               cmap = cmap, s = 40)

    # Plot labeling
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_zlabel(z)

    plt.title('{} as function of {} and {}'.format(
               z, x, y), size = 18);
    
plot_3d('learning_rate', 'n_estimators', 'score', opt_hyp)

# The bayesian optimization results are close in trend to those from random search: lower learning rate leads to higher cross validation scores.

# In[ ]:


plt.figure(figsize = (8, 7))
plt.plot(opt_hyp['learning_rate'], opt_hyp['n_estimators'], 'ro')
plt.xlabel('Learning Rate'); plt.ylabel('N Estimators'); plt.title('Number of Estimators vs Learning Rate');

# In[ ]:


plot_3d('reg_alpha', 'reg_lambda', 'score', opt_hyp)

# In[ ]:


best_opt_hyp

# Again, we probably want one of the regularization values to be high and the other to be low. This must help to "balance" the model between bias and variance. 

# # Correlations between Hyperparameters and Score
# 
# Time for another dangerous act: finding correlations between the hyperparameters and the score. These are not going to be accurate because again, we are not varying one value at a time! Nonetheless, we may discover useful insight about the Gradient Boosting Machine model.

# ### Correlations for Random Search

# In[ ]:


random_hyp['n_estimators'] = random_hyp['n_estimators'].astype(np.int32)
random_hyp.corr()['score']

# As expected, the `learning_rate` has one of the greatest correlations with the score. The `subsample` rate might be affected by the fact that 1/3 of the time this was set to 1.0.

# In[ ]:


random_hyp[random_hyp['boosting_type'] == 'gbdt'].corr()['score']['subsample']

# ### Correlations for Bayesian Optimization

# In[ ]:


opt_hyp['n_estimators'] = opt_hyp['n_estimators'].astype(np.int32)
opt_hyp.corr()['score']

# The `learning_rate` again appears to be moderately correlated with the score. This should tell us again that a lower learning rate tends to co-occur with a higher cross-validation score, but not that this is nexessarily the cause of the higher score. 

# In[ ]:


opt_hyp[opt_hyp['boosting_type'] == 'gbdt'].corr()['score']['subsample']

# ## Correlation Heatmap
# 
# Now we can make a heatmap of the correlations. I enjoy heatmaps and thankfully, they are not very difficult to make in `seaborn`.

# In[ ]:


plt.figure(figsize = (12, 12))

# Heatmap of correlations
sns.heatmap(random_hyp.corr().round(2), cmap = plt.cm.gist_heat_r, vmin = -1.0, annot = True, vmax = 1.0)
plt.title('Correlation Heatmap');

# That's a lot of plot for not very much code! We can see that the number of estimators and the learning rate have the greatest magnitude correlation (ignoring subsample which is influenced by the boosting type).

# In[ ]:


plt.figure(figsize = (12, 12))

# Heatmap of correlations
sns.heatmap(opt_hyp.corr().round(2), cmap = plt.cm.gist_heat_r, vmin = -1.0, annot = True, vmax = 1.0)
plt.title('Correlation Heatmap');

# Feel free to use this code for your own heatmaps! (Also send me color recommendations because I am not great at picking out a palette).

# # Meta-Machine Learning
# 
# So we have a labeled set of data: the hyperparameter values and the resulting score. Clearly, the next step is to use these for machine learning? Yes, here we will perform _meta-machine learning_ by fitting an estimator on top of the hyperparameter values and the scores. This is a supervised regression problem, and although we can use any method for learning the data, here we will stick to a linear regression. This will let us examine the coefficients on each hyperparameter and will help reduce overfitting. 

# In[ ]:


# Create training data and labels
hyp = hyp.drop(columns = ['metric', 'set', 'verbose'])
hyp['n_estimators'] = hyp['n_estimators'].astype(np.int32)
hyp['min_child_samples'] = hyp['min_child_samples'].astype(np.int32)
hyp['num_leaves'] = hyp['num_leaves'].astype(np.int32)
hyp['subsample_for_bin'] = hyp['subsample_for_bin'].astype(np.int32)
hyp = pd.get_dummies(hyp)

train_labels = hyp.pop('score')
train = np.array(hyp.copy())

# In[ ]:


from sklearn.linear_model import LinearRegression

# Create the lasso regression with cv
lr = LinearRegression()

# Train on the data
lr.fit(train, train_labels)

# In[ ]:


x = list(hyp.columns)
x_values = lr.coef_

coefs = {variable: coef for variable, coef in zip(x, x_values)}
coefs

# If we wanted, we could treat this as _another optimization problem_ and try to maximize the linear regression in terms of the score! However, for now I think we have done enough optimization. 
# 
# It's time to move on to implementing the best hyperparameter values from random and Bayesian optimization on the full dataset.

# # Implementation
# 
# The full set of features on which these results come are from [this notebook](https://www.kaggle.com/jsaguiar/updated-0-792-lb-lightgbm-with-simple-features) by [Aguiar](https://www.kaggle.com/jsaguiar)). Here, we will load in the same features, train on the full training features and make predictions on the testing data. These can then be uploaded to the competition.

# In[ ]:


import lightgbm as lgb

# In[ ]:


train = pd.read_csv('../input/home-credit-simple-featuers/simple_features_train.csv')
print('Full Training Features Shape: ', train.shape)
test = pd.read_csv('../input/home-credit-simple-featuers/simple_features_test.csv')
print('Full Testing Features Shape: ', test.shape)

# First we need to format the data and extract the labels.

# In[ ]:


train_labels = np.array(train['TARGET'].astype(np.int32)).reshape((-1, ))
train = train.drop(columns = ['SK_ID_CURR', 'TARGET'])

test_ids = list(test['SK_ID_CURR'])
test = test.drop(columns = ['SK_ID_CURR'])

# We can also save the features to later use for plotting feature importances.

# In[ ]:


features = list(train.columns)

# ### Random Search

# In[ ]:


random_best = ast.literal_eval(random.loc[0, 'hyperparameters'])

rmodel = lgb.LGBMClassifier(**random_best)
rmodel.fit(train, train_labels)

# In[ ]:


rpreds = rmodel.predict_proba(test)[:, 1]
rsub = pd.DataFrame({'SK_ID_CURR': test_ids, 'TARGET': rpreds})
rsub.to_csv('submission_random_search.csv', index = False)

# ### Bayesian Optimization

# In[ ]:


bayes_best = ast.literal_eval(opt.loc[0, 'hyperparameters'])

bmodel = lgb.LGBMClassifier(**bayes_best)
bmodel.fit(train, train_labels)

# In[ ]:


bpreds = bmodel.predict_proba(test)[:, 1]
bsub = pd.DataFrame({'SK_ID_CURR': test_ids, 'TARGET': bpreds})
bsub.to_csv('submission_bayesian_optimization.csv', index = False)

# ### Competition Results
# 
# * __Random search results scored 0.790__
# * __Bayesian optimization results scored 0.791__
# 
# If we go by best score on the public leaderboard, Bayesian Optimization wins! However, the public leaderboard is based only on 10% of the test data, so it's possible this is a result of overfitting to this particular subset of the testing data. Overall, I would say the complete results suggest that both methods produce similar outcomes especially when run for enough iterations. Either method is better than hand-tuning! 

# #### Feature Importances
# 
# As a final step, we can compare the feature importances between the models from the best hyperparameters. It would be interesting to see if the hyperparameter values has an effect on the feature importances.

# In[ ]:


random_fi = pd.DataFrame({'feature': features, 'importance': rmodel.feature_importances_})
bayes_fi = pd.DataFrame({'feature': features, 'importance': bmodel.feature_importances_})

# In[ ]:


def plot_feature_importances(df):
    """
    Plots 15 most important features and returns a sorted feature importance dataframe.
    
    Parameters
    --------
    df : dataframe
        Dataframe of feature importances. Columns must be feature and importance

        
    Return
    --------
    df : dataframe
        Dataframe ordered by feature importances with a normalized column (sums to 1)
        and a cumulative importance column
    
    """
    
    plt.rcParams['font.size'] = 18
    
    # Sort features according to importance
    df = df.sort_values('importance', ascending = False).reset_index()
    
    # Normalize the feature importances to add up to one
    df['importance_normalized'] = df['importance'] / df['importance'].sum()
    df['cumulative_importance'] = np.cumsum(df['importance_normalized'])

    # Make a horizontal bar chart of feature importances
    plt.figure(figsize = (10, 6))
    ax = plt.subplot()
    
    # Need to reverse the index to plot most important on top
    ax.barh(list(reversed(list(df.index[:15]))), 
            df['importance_normalized'].head(15), 
            align = 'center', edgecolor = 'k')
    
    # Set the yticks and labels
    ax.set_yticks(list(reversed(list(df.index[:15]))))
    ax.set_yticklabels(df['feature'].head(15))
    
    # Plot labeling
    plt.xlabel('Normalized Importance'); plt.title('Feature Importances')
    plt.show()

    
    return df

# In[ ]:


norm_randomfi = plot_feature_importances(random_fi)
norm_randomfi.head(10)

# In[ ]:


norm_bayesfi = plot_feature_importances(bayes_fi)
norm_bayesfi.head(10)

# The feature importances look to be relatively stable across hyperparameter values. This is what I expected, but at the same time, we can see that the _absolute magnitude_ of the importances differs significantly but not the _relative ordering_.

# In[ ]:


random.loc[0, 'hyperparameters']

# # Conclusions
# 
# Random search narrowly beat out Bayesian optimization in terms of finding the hyperparameter values that resulted in the highest cross validation ROC AUC. That single number does not tell the whole story though as the Bayesian method average ROC AUC was much higher than that of random search. We expect this to be the case because Bayesian optimization should focus on higher scoring values based on the surrogate model of the objective function it constructs. Morevoer, this tells us Bayesian optimization is a valuable technique, but random search can still happen upon better values in fewer search iterations if we are lucky. 
# 
# * Random search slightly outperformed Bayesian optimization in terms of cv ROC AUC 
# * Bayesian optimization average scores were much higher than random search indicating it spends more time evaluating "better" hyperparameters
# * Bayesian scored 0.791 when submitted and random search scored 0.790 indicating that with enough iterations, the methods deliver similar results
# * Boosting type "gdbt" did much better than "goss" with "dart" nearly as good
# * A lower learning rate resulted in higher model scores: lower than 0.02 looks to be optimal
# * `reg_alpha` and `reg_lambda` should complement one another: if one is high (above 0.5), than the other should be lower (below 0.5)
# * Some subsampling appears to increase the model scores
# * The other hyperparameters either did not have a significant effect, or their effects are intertwined and hence could not be disentangled in this study
# 
# Feel free to build upon these results! I'm curious if the best hyperparameters for this dataset will translate to other datasets, either for this problem, or for vastly different data science problems. The best way to find out is to try them! 
# 
# If you're looking for more work on this problem, I have a series of notebooks documenting my work:
# 
# __Additional Notebooks__ 
# 
# * [A Gentle Introduction](https://www.kaggle.com/willkoehrsen/start-here-a-gentle-introduction)
# * [Manual Feature Engineering Part One](https://www.kaggle.com/willkoehrsen/introduction-to-manual-feature-engineering)
# * [Manual Feature Engineering Part Two](https://www.kaggle.com/willkoehrsen/introduction-to-manual-feature-engineering-p2)
# * [Introduction to Automated Feature Engineering](https://www.kaggle.com/willkoehrsen/automated-feature-engineering-basics)
# * [Advanced Automated Feature Engineering](https://www.kaggle.com/willkoehrsen/tuning-automated-feature-engineering-exploratory)
# * [Feature Selection](https://www.kaggle.com/willkoehrsen/introduction-to-feature-selection)
# * [Intro to Model Tuning: Grid and Random Search](https://www.kaggle.com/willkoehrsen/intro-to-model-tuning-grid-and-random-search)
# * [Automated Model Tuning](https://www.kaggle.com/willkoehrsen/automated-model-tuning)
# 
# Thanks for reading and feel free to share any constructive criticism or feedback. 
# 
# Best,
# 
# Will

# In[ ]:



