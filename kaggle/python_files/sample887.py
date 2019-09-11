#!/usr/bin/env python
# coding: utf-8

# # Getting Started with LightGBM
# ---
# 
# This kernel is designed for both beginners and more experienced Kagglers as a quick introduction to this intriguing challenge. It will cover:
# 
# * A brief overview of the data and the target
# * A quick explanation of the AUC metric used for scoring in this competition
# * A baseline gradient-boosting trees model
# * Some experiments to improve model performance
# * Two baseline submissions
# 
# First we load in the dataset. The training data in this competition includes the target, and the row ID. We'll separate those out and take a look at the first few rows of train.

# In[ ]:


import numpy as np 
import pandas as pd 
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
y = train['target'].values.flatten()
ids = train['id']
train.drop(['target', 'id'], axis=1, inplace=True)
test.drop('id', axis=1, inplace=True)
train.head()

# A few things stand out. Firstly, the Kaggle team obviously had some fun when choosing the column names! They presumably have left some clues in here for us. The last word for each name in particular look like they are hinting at something. Words like 'sorted', 'noise' and 'gaussian' are all pertinent to machine learning. For the benefit of non-native English speakers who may be confused by the vocabulary, the four words in each column names mostly consist of:
# 
# * An adjective
# * A colour
# * An animal
# * A machine-learning term, Kaggle term, or nonsense word ('pembus', fepid')
# 
# Finding exceptions to this general rule may help you identify valuable features!
# 
# The variables mostly lie within a small range around 0, implying they have been already scaled with a tool like sklearn's `StandardScaler()`.
# 
# There don't seem to be any missing values in the first few rows. How large is the total training data, and how much of it is missing?

# In[ ]:


print('train consists of {} rows and {} columns.'.format(train.shape[0], train.shape[1]))
print('train contains {} missing values.'.format(train.isna().sum().sum()))
print('train is {}% incomplete.'.format(100*train.isna().sum().sum()/(train.shape[0]*train.shape[1])))

# There are no missing values to be worked around or [imputed](https://en.wikipedia.org/wiki/Imputation_(statistics). That simplifies things a little. Now we can make a baseline model to see how accurately we can make predictions without changing the data in any way. But first, we should understand how the scoring for this competition works.
# 
# ---
# # AUC: Area-Under-the-Curve
# 
# ![](https://developers.google.com/machine-learning/crash-course/images/ROCCurve.svg)
# 
# Receiver-Operating-Characteristic: Area-Under-the-Curve, ROC-AUC or just AUC for short, is a common metric for assessing the performance of classification models. Note that it's improper to say AUC measures the 'accuracy' of a model, as 'accuracy' has a specific definition in statistics. The basic thing to know about AUC is that it scores your overall predictions between 0 and 1. 1 means your model predicted everything perfectly. 0.5 means it predicted no better than random selection. Anything below 0.5 and something is seriously wrong with your model, as it could be predicting opposite classes!
# 
# There are many explanations of AUC online and [this is a good place to start](https://towardsdatascience.com/understanding-auc-roc-curve-68b2303cc9c5). It can be easier to understand when you look at how classification prediction works.
# 
# When we make a prediction with a model, we output a number between 0 and 1. Say our model predicts 0.4872 - what does this number mean? Our output has to be a zero or one, so the prediction has to be evaluated against a threshold, where the prediction becomes 1 if it is equal to or larger than the threshold. You might typically expect this to be 0.5, which makes mathematical sense. In this case, the final prediction would be 0, not 0.4872. However, this may not be best threshold for our model. If the actual value for the row was 1, we have predicted a false negative. Had we used a threshold of 0.4 to identify positive cases, our prediction would have been successful. So what is the best threshold to use overall?
# 
# This actually depends on the nature of the task, and whether you consider false positives (type 1 errors) or false negatives (type 2 errors) to be worse. For example, say you were working on a model that uses ground radar signatures to estimate the probability that a buried object is a land mine. Which type of error would have more serious consequences, a false positive or a false negative? For most situations it isn't so evident which type of error is worse so we use a metric that combines the rates of both, like AUC. AUC measures the ratio of the True Positive Rate and the False Positive Rate over the entire threshold range and computes the total area under this curve, hence the name.
# 
# ---
# # Starter Model
# 
# Let's see how well we can model the data without modification and using default model parameters. This will give us a quick baseline score we can use to evaluate any new models against. The Out-Of-Fold (OOF) score will tell us how accurate the model was on the validation data, averaged across the 5 folds.

# In[ ]:


import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

N_FOLD = 5
folds = StratifiedKFold(n_splits=N_FOLD, shuffle=True, random_state=42)

oof = np.zeros(len(train))
importances = np.zeros(train.shape[1])
X = train.values
preds = np.zeros(len(test))

for train_idx, valid_idx in folds.split(X, y):

    X_train, X_valid = X[train_idx, :], X[valid_idx, :]
    y_train, y_valid = y[train_idx], y[valid_idx]
    
    model = lgb.LGBMClassifier(n_estimators=10000, eval_metric='auc')
    model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_valid, y_valid)], verbose=200,
                      early_stopping_rounds=250, eval_metric='auc')
    val_preds = model.predict(X_valid)
    importances += model.feature_importances_/N_FOLD
    oof[valid_idx] = val_preds

AUC_OOF = round(roc_auc_score(y, oof), 4)
print('Model ensemble OOF AUC score: {}'.format(AUC_OOF))

# As stated above, an AUC of 0.5 indicates that a model performs as well as random guessing. So our model performance isn't very good at all! Perhaps we should take a closer look at the features.
# 

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns

#make lgb feature importance df
feature_df = pd.DataFrame({'feature' : train.columns,
                             'importance' : importances})
feature_df = feature_df.sort_values('importance', ascending=False)
    
#plot feature importances
N_ROWS = 50
plot_height = int(np.floor(len(feature_df.iloc[:N_ROWS, :])/5))
plt.figure(figsize=(12, plot_height));
sns.barplot(x='importance', y='feature', data=feature_df.iloc[:N_ROWS, :]);
plt.title('LightGBM Feature Importance');
plt.show()

# The column `'wheezy-copper-turtle-magic'` is our most important feature but nothing else immediately stands out. 
# 
# # Cardinality
# ---
# 
# It was mentioned earlier that the data seem to have been scaled before they were given to us. What if some of the variables were orginally integer values which indicated categorical data? We can explore this by examining the **cardinality** of our data. Cardinality is simply the number of unique values that are present in a variable. For a truly continuous variable we would expect the cardinality to be very high, equal or almost equal to the number of total rows in the dataframe. If a continuous-looking variable only has, say, 10 unique values, it would make more sense to treat it as a categorical variable and see if this improves the model.

# In[ ]:


cards = []
for i in range(0, train.shape[1]):
    cards.append(len(np.unique(train.iloc[:, i].values)))
cards = np.asarray(cards)

card_df = pd.DataFrame({'feature' : train.columns,
                       'cardinality' : cards})

card_df.sort_values('cardinality', inplace=True)
card_df.head()        

# Only one feature, `'wheezy-copper-turtle-magic'` displays low cardinality. So we can try encoding it as an integer variable with `pd.factorize()`, supply it to our model as a categorical variable and see if it improves performance. It's important to combine the train and test data before factorising it so that the corresponding numbers match up in the two datasets.

# In[ ]:


temp = pd.concat([train['wheezy-copper-turtle-magic'], test['wheezy-copper-turtle-magic']])
temp = pd.factorize(temp)[0]
train['wheezy-copper-turtle-magic'] = temp[:len(train)]
test['wheezy-copper-turtle-magic'] = temp[len(train):]
train['wheezy-copper-turtle-magic'] = train['wheezy-copper-turtle-magic'].astype('category')
test['wheezy-copper-turtle-magic'] = test['wheezy-copper-turtle-magic'].astype('category')
cat_feature_index = [train.columns.get_loc('wheezy-copper-turtle-magic')]

# Now compare the model performance:

# In[ ]:


oof = np.zeros(len(train))
importances = np.zeros(train.shape[1])
X = train.values

for train_idx, valid_idx in folds.split(X, y):

    X_train, X_valid = X[train_idx, :], X[valid_idx, :]
    y_train, y_valid = y[train_idx], y[valid_idx]
    
    model = lgb.LGBMClassifier(n_estimators=10000, eval_metric='auc')
    model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_valid, y_valid)], verbose=200,
                      early_stopping_rounds=250, eval_metric='auc', categorical_feature=cat_feature_index)
    val_preds = model.predict(X_valid)
    importances += model.feature_importances_/N_FOLD
    oof[valid_idx] = val_preds

AUC_OOF = round(roc_auc_score(y, oof), 4)
print('Model ensemble OOF AUC score: {}'.format(AUC_OOF))

# ---
# This has improved the model performance significantly. How can we further optimise the model?
# 
# # Parameter Tuning
# 
# An important part of optimising model performance is selecting the best parameters for it, such as the learning rate and maximum tree depth. You can try modifying these individually but there are a number of tools to help adjust model parameter settings. For this kernel, I'm recycling a function of mine that uses the Hyperopt package to automatically tune parameters. The code is in the code block below.

# In[ ]:


#import required packages
from hyperopt import hp, tpe, Trials, STATUS_OK
from hyperopt.fmin import fmin
from hyperopt.pyll.stochastic import sample
import gc #garbage collection
#optional but advised
import warnings
warnings.filterwarnings('ignore')

#GLOBAL HYPEROPT PARAMETERS
NUM_EVALS = 1000 #number of hyperopt evaluation rounds
N_FOLDS = 5 #number of cross-validation folds on data in each evaluation round

#LIGHTGBM PARAMETERS
LGBM_MAX_LEAVES = 2**9 #maximum number of leaves per tree for LightGBM
LGBM_MAX_DEPTH = 25 #maximum tree depth for LightGBM
EVAL_METRIC_LGBM_REG = 'mae' #LightGBM regression metric. Note that 'rmse' is more commonly used 
EVAL_METRIC_LGBM_CLASS = 'auc'#LightGBM classification metric

def quick_hyperopt(data, labels, num_evals=NUM_EVALS, Class=True, cat_features=None):
    
    print('Running {} rounds of LightGBM parameter optimisation:'.format(num_evals))
    #clear space
    gc.collect()

    integer_params = ['max_depth',
                     'num_leaves',
                      'max_bin',
                     'min_data_in_leaf',
                     'min_data_in_bin']

    def objective(space_params):

        #cast integer params from float to int
        for param in integer_params:
            space_params[param] = int(space_params[param])

        #extract nested conditional parameters
        if space_params['boosting']['boosting'] == 'goss':
            top_rate = space_params['boosting'].get('top_rate')
            other_rate = space_params['boosting'].get('other_rate')
            #0 <= top_rate + other_rate <= 1
            top_rate = max(top_rate, 0)
            top_rate = min(top_rate, 0.5)
            other_rate = max(other_rate, 0)
            other_rate = min(other_rate, 0.5)
            space_params['top_rate'] = top_rate
            space_params['other_rate'] = other_rate

        subsample = space_params['boosting'].get('subsample', 1.0)
        space_params['boosting'] = space_params['boosting']['boosting']
        space_params['subsample'] = subsample

        if Class:                
            if cat_features is not None:
                cv_results = lgb.cv(space_params, train, nfold = N_FOLDS, stratified=True, categorical_feature=cat_features,
                                    early_stopping_rounds=100, metrics=EVAL_METRIC_LGBM_CLASS, seed=42)
                best_loss = 1 - cv_results['auc-mean'][-1]
            else:
                cv_results = lgb.cv(space_params, train, nfold = N_FOLDS, stratified=True,
                                    early_stopping_rounds=100, metrics=EVAL_METRIC_LGBM_CLASS, seed=42)
                best_loss = 1 - cv_results['auc-mean'][-1]

        else:
            if cat_features is not None:
                cv_results = lgb.cv(space_params, train, nfold = N_FOLDS, stratified=False,
                                    early_stopping_rounds=100, metrics=EVAL_METRIC_LGBM_REG, seed=42)
                best_loss = cv_results['l1-mean'][-1] #'l2-mean' for rmse
            else:
                cv_results = lgb.cv(space_params, train, nfold = N_FOLDS, stratified=True,
                                    early_stopping_rounds=100, metrics=EVAL_METRIC_LGBM_REG, seed=42)
                best_loss = 1 - cv_results['auc-mean'][-1]

        return{'loss':best_loss, 'status': STATUS_OK }

    if cat_features is not None:
        train = lgb.Dataset(data, labels, categorical_feature=cat_features)
    else:
         train = lgb.Dataset(data, labels)

    #integer and string parameters, used with hp.choice()
    boosting_list = [{'boosting': 'gbdt',
                      'subsample': hp.uniform('subsample', 0.5, 1)},
                     {'boosting': 'goss',
                      'subsample': 1.0,
                     'top_rate': hp.uniform('top_rate', 0, 0.5),
                     'other_rate': hp.uniform('other_rate', 0, 0.5)}] #if including 'dart', make sure to set 'n_estimators'

    if Class:
        metric_list = ['auc'] #modify as required for other classification metrics
        objective_list = ['binary', 'cross_entropy']

    else:
        metric_list = ['MAE', 'RMSE'] 
        objective_list = ['huber', 'gamma', 'fair', 'tweedie']


    space ={'boosting' : hp.choice('boosting', boosting_list),
            'num_leaves' : hp.quniform('num_leaves', 2, LGBM_MAX_LEAVES, 1),
            'max_depth': hp.quniform('max_depth', 2, LGBM_MAX_DEPTH, 1),
            'max_bin': hp.quniform('max_bin', 32, 255, 1),
            'min_data_in_leaf': hp.quniform('min_data_in_leaf', 1, 256, 1),
            'min_data_in_bin': hp.quniform('min_data_in_bin', 1, 256, 1),
            'min_gain_to_split' : hp.quniform('min_gain_to_split', 0.1, 5, 0.01),
            'lambda_l1' : hp.uniform('lambda_l1', 0, 5),
            'lambda_l2' : hp.uniform('lambda_l2', 0, 5),
            'learning_rate' : hp.loguniform('learning_rate', np.log(0.005), np.log(0.2)),
            'metric' : hp.choice('metric', metric_list),
            'objective' : hp.choice('objective', objective_list),
            'feature_fraction' : hp.quniform('feature_fraction', 0.5, 1, 0.01),
            'bagging_fraction' : hp.quniform('bagging_fraction', 0.5, 1, 0.01)
        }

    trials = Trials()
    best = fmin(fn=objective,
                space=space,
                algo=tpe.suggest,
                max_evals=num_evals, 
                trials=trials)

    #fmin() will return the index of values chosen from the lists/arrays in 'space'
    #to obtain actual values, index values are used to subset the original lists/arrays
    best['boosting'] = boosting_list[best['boosting']]['boosting']#nested dict, index twice
    best['metric'] = metric_list[best['metric']]
    best['objective'] = objective_list[best['objective']]

    #cast floats of integer params to int
    for param in integer_params:
        best[param] = int(best[param])

    print('{' + '\n'.join('{}: {}'.format(k, v) for k, v in best.items()) + '}')
    return(best)    

# In[ ]:


lgbm_params = quick_hyperopt(train, y, 25, cat_features=cat_feature_index)

# With the optimised parameters we can train our final models, make the predictions on test and submit.

# In[ ]:


oof = np.zeros(len(train))
X = train.values
preds = np.zeros(len(test))

for train_idx, valid_idx in folds.split(X, y):

    X_train, X_valid = X[train_idx, :], X[valid_idx, :]
    y_train, y_valid = y[train_idx], y[valid_idx]
    
    train_dataset = lgb.Dataset(X_train, y_train, categorical_feature=cat_feature_index)
    val_dataset = lgb.Dataset(X_valid, y_valid, categorical_feature=cat_feature_index)

    model = lgb.train(lgbm_params, train_dataset, valid_sets=[train_dataset, val_dataset], verbose_eval=200,
                      num_boost_round=10000, early_stopping_rounds=250, categorical_feature=cat_feature_index)
    val_preds = model.predict(X_valid, num_iteration=model.best_iteration)
    oof[valid_idx] = val_preds
    preds += model.predict(test, num_iteration=model.best_iteration)/N_FOLD

AUC_OOF = round(roc_auc_score(y, oof), 4)
print('Model ensemble OOF AUC score: {}'.format(AUC_OOF))

# In[ ]:


sub = pd.read_csv('../input/sample_submission.csv')
sub['target'] = preds
sub.to_csv('initial_sub.csv', index=False)

# # Wordplay Experiment
# 
# As a last idea, let's assume that Kaggle's columns names are designed to give us obvious hints. What happens if we drop all the columns with the words 'noise', 'distraction' and 'discard'?

# In[ ]:


keep_cols = train.columns
remove_words = ['noise', 'distraction', 'discard']
for keyword in remove_words:
    keep_cols = [x for x in keep_cols if keyword not in keep_cols]
    
train_2 = train[keep_cols]
test_2 = test[keep_cols]
cat_feature_index = [train_2.columns.get_loc('wheezy-copper-turtle-magic')]

#get new optimised parameters
final_params = quick_hyperopt(train_2, y, 25, cat_features=cat_feature_index)

oof = np.zeros(len(train))
X = train_2.values
preds = np.zeros(len(test))

for train_idx, valid_idx in folds.split(X, y):

    X_train, X_valid = X[train_idx, :], X[valid_idx, :]
    y_train, y_valid = y[train_idx], y[valid_idx]
    
    train_dataset = lgb.Dataset(X_train, y_train, categorical_feature=cat_feature_index)
    val_dataset = lgb.Dataset(X_valid, y_valid, categorical_feature=cat_feature_index)

    model = lgb.train(final_params, train_dataset, valid_sets=[train_dataset, val_dataset], verbose_eval=200,
                      num_boost_round=10000, early_stopping_rounds=250, categorical_feature=cat_feature_index)
    val_preds = model.predict(X_valid, num_iteration=model.best_iteration)
    oof[valid_idx] = val_preds
    preds += model.predict(test_2, num_iteration=model.best_iteration)/N_FOLD

AUC_OOF = round(roc_auc_score(y, oof), 4)
print('Model ensemble OOF AUC score: {}'.format(AUC_OOF))

# Apparently the clues aren't that obvious! So what happens if we remove the columns with positive words like 'important', 'grandmaster' and 'expert'?

# In[ ]:


keep_cols = train.columns
remove_words = ['important', 'grandmaster', 'expert']
for keyword in remove_words:
    keep_cols = [x for x in keep_cols if keyword not in keep_cols]
    
train_3 = train[keep_cols]
test_3 = test[keep_cols]
cat_feature_index = [train_3.columns.get_loc('wheezy-copper-turtle-magic')]

#get new optimised parameters
final_params = quick_hyperopt(train_3, y, 25, cat_features=cat_feature_index)

oof = np.zeros(len(train))
X = train_3.values
preds = np.zeros(len(test))

for train_idx, valid_idx in folds.split(X, y):

    X_train, X_valid = X[train_idx, :], X[valid_idx, :]
    y_train, y_valid = y[train_idx], y[valid_idx]
    
    train_dataset = lgb.Dataset(X_train, y_train, categorical_feature=cat_feature_index)
    val_dataset = lgb.Dataset(X_valid, y_valid, categorical_feature=cat_feature_index)

    model = lgb.train(final_params, train_dataset, valid_sets=[train_dataset, val_dataset], verbose_eval=200,
                      num_boost_round=10000, early_stopping_rounds=250, categorical_feature=cat_feature_index)
    val_preds = model.predict(X_valid, num_iteration=model.best_iteration)
    oof[valid_idx] = val_preds
    preds += model.predict(test_3, num_iteration=model.best_iteration)/N_FOLD

AUC_OOF = round(roc_auc_score(y, oof), 4)
print('Model ensemble OOF AUC score: {}'.format(AUC_OOF))

# In[ ]:


sub = pd.read_csv('../input/sample_submission.csv')
sub['target'] = preds
sub.to_csv('second_sub.csv', index=False)

# Looks like removing the positive descriptions actually improved results! Hopefully you can find some clues of your own in the variable names. 
# 
# I hope this kernel is of use to other Kagglers, all comments and questions are welcome. Good luck with the competition!
