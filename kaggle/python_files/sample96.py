#!/usr/bin/env python
# coding: utf-8

# High all,
# 
# I figured it would be interesting to compare the top scoring public kernels.
# 
# The analysis is also available(and may be updated) [here](https://ui.neptune.ml/jakub-czakon/santander/wiki/1-public_kernel_comparison).
# 
# # Comparison Setup
# 
# ## Kernels considered:
# - https://www.kaggle.com/mhviraf/santander-compact-solution-14-lines-will-do
# - https://www.kaggle.com/kamalchhirang/simple-lightgbm-with-good-parameters
# - https://www.kaggle.com/sandeepkumar121995/magic-parameters
# - https://www.kaggle.com/jesucristo/30-lines-starter-solution-fast
# - https://www.kaggle.com/brandenkmurray/randomly-shuffled-data-also-works
# - https://www.kaggle.com/lavanyadml/santander-ls
# - https://www.kaggle.com/jesucristo/santander-magic-lgb
# - https://www.kaggle.com/gpreda/santander-fast-compact-solution
# - https://www.kaggle.com/jesucristo/40-lines-starter-solution-fast DELETED
# 
# I hope I didn't forget to upvote/fork any of those kernels. Good job people!
# 
# ## Validation:
# I adjusted all of them so that they would have the same validation schema:
# 
#     N_SPLITS = 15
#     SEED = 2319   
#     folds = StratifiedKFold(n_splits=N_SPLITS, shuffle=False, random_state=SEED)
#     
# ## Tracking
# I logged all the experiment information to [Neptune](http://bit.ly/2FndEZO) to be able to easily compare it later:
#   - hyperparameters
#   - lightgbm training curves
#   - roc_auc metric on out of fold predictions
#   - confusion matrix
#   - ROC AUC curve
#   - prediction distribution plot
#  
# **Note** 
# You don't have to know how to track stuff with Neptune to follow this post. 
# 
# If you are interested, [here is an example neptune kernel](https://www.kaggle.com/jakubczakon/example-lightgbm-neptune-tracking).  
# [Neptune](http://bit.ly/2FndEZO) is free for non-organizations and you can easily use it from inside kaggle kernels (or any other place for that matter) to track your experiments. 
#   
# # Results
# Ok, let's explore the results.
# 
# ## Scores
# First, lets have a look at the validation results of those models:
# 
# [Results dashboard](https://ui.neptune.ml/jakub-czakon/santander/experiments?filterId=288495a5-c965-4896-815c-7474fd95234f)
# 
# ![image](https://gist.githubusercontent.com/jakubczakon/f754769a39ea6b8fa9728ede49b9165c/raw/2507ca2f4ff706b4a358decf0ba714ead7010a0b/comparison8.png)
# 
# All the top kernels, which is no surprise, perform quite well and get us `Local CV &gt; 0.9`.
# The results vary from `0.900464` to `0.901075`, which may not be a lot in a large picture of things, but for this particular competition could mean a difference of 1000+ places.
# It seems that the `Random Shuffled Data Also Works` not only "also works" but works the best, which was surprising to me. 
# 
# ## Hyperparameters
# 
# Looking at the parameters in the table above it seems that every kernel apart from the best one used `num_leaves=13`. Interesting.
# To get more insights I decided to use the `plot_evaluations` function from the `skopt.plots` package: 
# 
# [Exploration experiment](https://ui.neptune.ml/jakub-czakon/santander/e/SAN1-170/channels)
# 
# ![image](https://gist.githubusercontent.com/jakubczakon/f754769a39ea6b8fa9728ede49b9165c/raw/2507ca2f4ff706b4a358decf0ba714ead7010a0b/comparison2.png)
# 
# It's a pretty useful tool, that lets you see what were the hyperparameter values that were explored, and which values were explored the most. The red dot shows the best run. 
# 
# Sometimes, you can see clear feature interactions (in the hyperparameter space) or the over/underexplored areas in the hyperparameter space. 
# In this particular case my conclusions are:
# - explore other `num_leaves` values between `3` and `13`
# - higher bagging fraction perform better
# - there seems to be a sweet spot when it comes to `early_stopping_rounds=3000`
# 
# ## Learning curves
# Since I logged all the metrics during training with the `neptune_monitor` callback we can try and get some insights here. If you want to see how to create a custom callback for your tracking tool/setup [go to this example kernel](https://www.kaggle.com/jakubczakon/example-lightgbm-neptune-tracking).
# Let's start with the comparison of the learning curves for all the experiments:
# 
# [Learning curves comparison](https://ui.neptune.ml/jakub-czakon/santander/compare?shortId=%5B%22SAN1-84%22%2C%22SAN1-59%22%2C%22SAN1-124%22%2C%22SAN1-142%22%2C%22SAN1-56%22%2C%22SAN1-140%22%2C%22SAN1-141%22%2C%22SAN1-58%22%2C%22SAN1-57%22%5D)
# 
# ![image](https://gist.githubusercontent.com/jakubczakon/f754769a39ea6b8fa9728ede49b9165c/raw/2507ca2f4ff706b4a358decf0ba714ead7010a0b/comparison9.png)
# 
# Apart from the `Randomly Shuffled Data Also Works` all those curves are quite similar. We can see that the "shuffled" gets to 0.9 slower but is overfitting considerably less. That is the result of the `num_leaves=3` but maybe shuffling features somehow plays a role here as well. It is probably worth exploring.
# 
# Let's now take a closer look at the learning curves of the top two kernels `Randomly Shuffled Data Also Works` and `Magic Parameters`.
# 
# **`Randomly Shuffled Data Also Works`**
# 
# [Experiment](https://ui.neptune.ml/jakub-czakon/santander/e/SAN1-84/charts)
# 
# ![image](https://gist.githubusercontent.com/jakubczakon/f754769a39ea6b8fa9728ede49b9165c/raw/2507ca2f4ff706b4a358decf0ba714ead7010a0b/comparison12_random_shuffle.png)
# 
# We can see that there is some overfitting but not a lot. Training curves are almost identical for each fold, but when we look at the validation curves there is quite a lot of difference:
# 
# ![image](https://gist.githubusercontent.com/jakubczakon/f754769a39ea6b8fa9728ede49b9165c/raw/2507ca2f4ff706b4a358decf0ba714ead7010a0b/comparison13_random_shuffle.png)
# 
# The worst result was just 0.889 for fold 4 while the best one got 0.9111 on fold 13. 
# 
# **`Magic Parameters`**
# Looking at this kernel:
# 
# [Experiment](https://ui.neptune.ml/jakub-czakon/santander/e/SAN1-59/charts)
# 
# ![image](https://gist.githubusercontent.com/jakubczakon/f754769a39ea6b8fa9728ede49b9165c/raw/2507ca2f4ff706b4a358decf0ba714ead7010a0b/comparison10_magic_lgb.png)
# 
# Overfitting is significantly more visible here.
# 
# ![image](https://gist.githubusercontent.com/jakubczakon/f754769a39ea6b8fa9728ede49b9165c/raw/2507ca2f4ff706b4a358decf0ba714ead7010a0b/comparison11_magic_lgb.png)
# 
# The difference between best and worst folds is pretty much the same `fold 4 0.889` vs `fold 13 0.910`.
# 
# My conclusions are:
#  - `num_leaves=3` gives better results and controls overfitting, maybe other regularization params like l1/l2 could push the score up a bit more.
#  - the difference between best and worst folds is large.  Maybe the investigation of this problem can reveal insights about the underlying structure of the problem.
# 
# ## Prediction correlations
# Let's look at the prediction correlations for out of fold train predictions and averaged test predictions respectively:
# 
# [Exploration experiment](https://ui.neptune.ml/jakub-czakon/santander/e/SAN1-170/channels)
# 
# ![image](https://gist.githubusercontent.com/jakubczakon/f754769a39ea6b8fa9728ede49b9165c/raw/2507ca2f4ff706b4a358decf0ba714ead7010a0b/comparison3.png)
# ![image](https://gist.githubusercontent.com/jakubczakon/f754769a39ea6b8fa9728ede49b9165c/raw/2507ca2f4ff706b4a358decf0ba714ead7010a0b/comparison4.png)
# 
# Both for the train oof and test predictions, the correlations between public kernels are extremely high.
# It seems that it will be very difficult to squeeze extra juice from blending/stacking those models.
# 
# My conclusions are:
# - we may need to add diversity to gain anything by blending
# 
# ## Predictions exploration
# 
# Let's start by looking at the standard model diagnostics for the best model:
# 
# [Best experiment](https://ui.neptune.ml/jakub-czakon/santander/e/SAN1-84/channels)
# ![image](https://gist.githubusercontent.com/jakubczakon/f754769a39ea6b8fa9728ede49b9165c/raw/4d52325ae3ac0f2aeabf406a6c62a650b7f1b0c3/comparsion15.png)
# 
# Ok, it looks like it has trouble with giving a very low score to positive examples. 
# 
# Let's take a look at the predictions that public kernels produce. 
# 
# I will begin by looking at the error distribution plot. I calculated it simply by substructing `prediction` from `target`.
# 
# [Exploration experiment](https://ui.neptune.ml/jakub-czakon/santander/e/SAN1-170/channels)
# 
# ![image](https://gist.githubusercontent.com/jakubczakon/f754769a39ea6b8fa9728ede49b9165c/raw/2507ca2f4ff706b4a358decf0ba714ead7010a0b/comparison1.png)
# 
# Ok, it seems that our model is pretty good when it comes to predicting negative cases but has a lot of problems when it comes to predicting positive cases. Not only that, but it is quite confidently wrong, giving very low scores for some positive cases. 
# 
# Now, I would like to take a look at the predictions themselves.
# [Exploration experiment](https://ui.neptune.ml/jakub-czakon/santander/e/SAN1-170/channels)
# 
# ![image](https://gist.githubusercontent.com/jakubczakon/f754769a39ea6b8fa9728ede49b9165c/raw/2507ca2f4ff706b4a358decf0ba714ead7010a0b/comparsion7.png)
# 
# And if we split it into positive and negative examples respectively:
# [Exploration experiment](https://ui.neptune.ml/jakub-czakon/santander/e/SAN1It is not super important to know how-170/channels)
# 
# ![image](https://gist.githubusercontent.com/jakubczakon/f754769a39ea6b8fa9728ede49b9165c/raw/2507ca2f4ff706b4a358decf0ba714ead7010a0b/comparison6.png)
# 
# [Exploration experiment](https://ui.neptune.ml/jakub-czakon/santander/e/SAN1-170/channels)
# 
# ![image](https://gist.githubusercontent.com/jakubczakon/f754769a39ea6b8fa9728ede49b9165c/raw/2507ca2f4ff706b4a358decf0ba714ead7010a0b/comparison5.png)
# 
# Even though the prediction distributions at the first sight look very similar there could still be benefits to blending.
# Especially those positive predictions differ from model to model quite a bit.
# 
# My conclusions are:
# - maybe there is some LB sauce that can be squeezed from blending after all
# - we should investigate how to make our model less sure when it is wrong about the positive cases (potentially add some models that do)
# 
# # Final thoughts:
# - There is probably room to explore on the hyperparameter front
# - Difference between worst and best fold results is huge. Exploring it could bring insights
# - There is probably to blending/stacking public kernels but adding diversity may be important.
# 
# What do you think?
# 
# 
# # Edits:
# 
# Thank you @tilii7 for cleaning my thought.
# Added: "Not only that, it is quite confidently wrong (giving very low scores) for some positive cases."  
# Dropped: "In simple words, some positive cases are given very low predictions by our models. "

# In[ ]:



