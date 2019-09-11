#!/usr/bin/env python
# coding: utf-8

# <h1 align="center">Mean (likelihood) encoding for categorical variables with high cardinality and feature interactions: a comprehensive study with Python </h1>
# 
# 
# <div align="center">By Viacheslav Prokopev (<a href='https://www.kaggle.com/vprokopev'>Kaggle</a>, <a href='https://www.facebook.com/vprokopev.agi'>Facebook</a>, <a href='https://github.com/vprokopev'>GitHub</a>)</div>
# 

# <h2> Abstract </h2>
# 
# Encoding categorical variables with a mean target value (and also other target statistics) is a popular method for working with high cardinality features, especially useful for tree-based models. The method is simple: count mean target value for each category (for regression tasks) or likelihood of a data point to belong to one of the classes (for classification tasks) and use it as a label for a class.
# 
# This method (in plain, not regularised implementation) is really similar to label encoding, in a way that we also just assign labels to each category, but these labels are not random with mean encoding, they are correlated with target variable, which helps ML models to use labels more efficiently
# 
# In this study, I will show how mean encoding methods with different regularisations compare to methods like one-hot encoding, label and frequency encoding and to each other using 6 datasets with high cardinality features. I will also give my explanations, intuitions, and recommendations on osage of those methods.

# ## Key takeaways of the study
# 
# - Mean encodings with different regularisations were tested on 6 datasets along with one-hot, label and frequency encodings. Datasets mostly consisted of high-cardinality categorical features.
# 
# - k-fold regularisation using 4 or 5 folds and $\alpha = 5$ should be your "to go" regularisation for mean encoding, it almost always shows good results.
# 
# - Most of the time, there is no point to even try to use mean encoding without a prior ($\alpha$). It always performs worst then the one with a prior.
# 
# - Expanding mean regularisation works well only on bigger datasets (>100000 samples).
# 
# - Frequency encoding works surprisingly good on many datasets, I suggest you to always give it a try.
# 
# - Mean encodings let models converge faster (in terms of a number of iterations) then the frequency and label encodings.
# 
# - Performance of a particular encoding depends a lot on a dataset you work with. Any encoding can strongly outperform all other encodings for a number of different reasons.

# <h2> Navigation </h2>
# 
# <ol style="list-style: none;">
#   <li>1.  &nbsp;&nbsp;<a href="#1">Imports and helper function definitions (skip this)</a></li>
#   <li>2.  &nbsp;&nbsp;<a href="#2">High cardinality categorical features</a></li>
#   <li>2.1 <a href="#21">Feature interactions usually cause high cardinality</a></li>
#   <li>3.  &nbsp;&nbsp;<a href="#3">Popular encodings for categorical variables, advantages and disadvantages for tree models</a></li>
#   <li>3.1 <a href="#31">One-hot encoding</a></li>
#   <li>3.2 <a href="#32">Label and frequency encoding</a></li>
#   <li>4.  &nbsp;&nbsp;<a href="#4">Mean target encoding</a></li>
#   <li>4.1 <a href="#41">Overfitting with mean target encoding</a></li>
#   <li>4.2 <a href="#42">Using the prior probability for regularisation</a></li>
#   <li>4.3 <a href="#43">K-fold regularisation for mean encodings</a></li>
#   <li>4.4 <a href="#44">Expanding mean regularisation</a></li>
#   <li>5.  &nbsp;&nbsp;<a href="#5">Datasets</a></li>
#   <li>6.  &nbsp;&nbsp;<a href="#6">Encoding tests</a></li>
#   <li>6.1 <a href="#61">Testing procedure</a></li>
#   <li>6.2 <a href="#62">Encoding statistics</a></li>
#   <li>6.3 <a href="#63">IMDB movie revenue</a></li>
#   <li>6.4 <a href="#64">Video Game Sales</a></li>
#   <li>6.5 <a href="#65">World Bank poverty prediction</a></li>
#   <li>6.6 <a href="#66">World Bank poverty prediction with feature interactions</a></li>
#   <li>6.7 <a href="#67">Home Credit Default Risk</a></li>
#   <li>6.8 <a href="#68">Avazu Click-Through Rate Prediction</a></li>
#   <li>7.  &nbsp;&nbsp;<a href="#7">Summary and Conclusions</a></li>
#   <li>7.1 <a href="#71">Summary</a></li>
#   <li>7.2 <a href="#72">Conclusions</a></li>
# </ol>
# 
# 

# <div id="1"><h2>1. Imports and helper function definitions (skip this)</h2></div>

# In[ ]:


import pandas as pd
import numpy as np
import math
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.cross_validation import StratifiedKFold
from numpy.random import normal
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import os
print(os.listdir("../input"))

# In[ ]:


def one_hot_encode(train_data, test_data, columns):
    '''Returns a DataFrame with encoded columns'''
    conc = pd.concat([train_data, test_data], axis=0)
    encoded_cols = []
    for col in columns:
        encoded_cols.append(pd.get_dummies(conc[col], prefix='one_hot_'+col, 
                                      drop_first=True))
    all_encoded = pd.concat(encoded_cols, axis=1)
    return (all_encoded.iloc[:train_data.shape[0],:], 
            all_encoded.iloc[train_data.shape[0]:,:])


def one_hot_encode(train_data, test_data, columns):
    conc = pd.concat([train_data, test_data], axis=0)
    encoded = pd.get_dummies(conc.loc[:, columns], drop_first=True,
                             sparse=True) 
    return (encoded.iloc[:train_data.shape[0],:], 
            encoded.iloc[train_data.shape[0]:,:])


def label_encode(train_data, test_data, columns):
    'Returns a DataFrame with encoded columns'
    encoded_cols = []
    for col in columns:
        factorised = pd.factorize(train_data[col])[1]
        labels = pd.Series(range(len(factorised)), index=factorised)
        encoded_col_train = train_data[col].map(labels) 
        encoded_col_test = test_data[col].map(labels)
        encoded_col = pd.concat([encoded_col_train, encoded_col_test], axis=0)
        encoded_col[encoded_col.isnull()] = -1
        encoded_cols.append(pd.DataFrame({'label_'+col:encoded_col}))
    all_encoded = pd.concat(encoded_cols, axis=1)
    return (all_encoded.loc[train_data.index,:], 
            all_encoded.loc[test_data.index,:])

def freq_encode(train_data, test_data, columns):
    '''Returns a DataFrame with encoded columns'''
    encoded_cols = []
    nsamples = train_data.shape[0]
    for col in columns:    
        freqs_cat = train_data.groupby(col)[col].count()/nsamples
        encoded_col_train = train_data[col].map(freqs_cat)
        encoded_col_test = test_data[col].map(freqs_cat)
        encoded_col = pd.concat([encoded_col_train, encoded_col_test], axis=0)
        encoded_col[encoded_col.isnull()] = 0
        encoded_cols.append(pd.DataFrame({'freq_'+col:encoded_col}))
    all_encoded = pd.concat(encoded_cols, axis=1)
    return (all_encoded.loc[train_data.index,:], 
            all_encoded.loc[test_data.index,:])

def mean_encode(train_data, test_data, columns, target_col, reg_method=None,
                alpha=0, add_random=False, rmean=0, rstd=0.1, folds=1):
    '''Returns a DataFrame with encoded columns'''
    encoded_cols = []
    target_mean_global = train_data[target_col].mean()
    for col in columns:
        # Getting means for test data
        nrows_cat = train_data.groupby(col)[target_col].count()
        target_means_cats = train_data.groupby(col)[target_col].mean()
        target_means_cats_adj = (target_means_cats*nrows_cat + 
                                 target_mean_global*alpha)/(nrows_cat+alpha)
        # Mapping means to test data
        encoded_col_test = test_data[col].map(target_means_cats_adj)
        # Getting a train encodings
        if reg_method == 'expanding_mean':
            train_data_shuffled = train_data.sample(frac=1, random_state=1)
            cumsum = train_data_shuffled.groupby(col)[target_col].cumsum() - train_data_shuffled[target_col]
            cumcnt = train_data_shuffled.groupby(col).cumcount()
            encoded_col_train = cumsum/(cumcnt)
            encoded_col_train.fillna(target_mean_global, inplace=True)
            if add_random:
                encoded_col_train = encoded_col_train + normal(loc=rmean, scale=rstd, 
                                                               size=(encoded_col_train.shape[0]))
        elif (reg_method == 'k_fold') and (folds > 1):
            kfold = StratifiedKFold(train_data[target_col].values, folds, shuffle=True, random_state=1)
            parts = []
            for tr_in, val_ind in kfold:
                # divide data
                df_for_estimation, df_estimated = train_data.iloc[tr_in], train_data.iloc[val_ind]
                # getting means on data for estimation (all folds except estimated)
                nrows_cat = df_for_estimation.groupby(col)[target_col].count()
                target_means_cats = df_for_estimation.groupby(col)[target_col].mean()
                target_means_cats_adj = (target_means_cats*nrows_cat + 
                                         target_mean_global*alpha)/(nrows_cat+alpha)
                # Mapping means to estimated fold
                encoded_col_train_part = df_estimated[col].map(target_means_cats_adj)
                if add_random:
                    encoded_col_train_part = encoded_col_train_part + normal(loc=rmean, scale=rstd, 
                                                                             size=(encoded_col_train_part.shape[0]))
                # Saving estimated encodings for a fold
                parts.append(encoded_col_train_part)
            encoded_col_train = pd.concat(parts, axis=0)
            encoded_col_train.fillna(target_mean_global, inplace=True)
        else:
            encoded_col_train = train_data[col].map(target_means_cats_adj)
            if add_random:
                encoded_col_train = encoded_col_train + normal(loc=rmean, scale=rstd, 
                                                               size=(encoded_col_train.shape[0]))

        # Saving the column with means
        encoded_col = pd.concat([encoded_col_train, encoded_col_test], axis=0)
        encoded_col[encoded_col.isnull()] = target_mean_global
        encoded_cols.append(pd.DataFrame({'mean_'+target_col+'_'+col:encoded_col}))
    all_encoded = pd.concat(encoded_cols, axis=1)
    return (all_encoded.loc[train_data.index,:], 
            all_encoded.loc[test_data.index,:])

def test_clf(X_train, y_train, X_test, y_test, iterations):
    train_scores = []
    val_scores = []
    for i in iterations:
        model = GradientBoostingRegressor(n_estimators=i, learning_rate=1, max_depth=3, 
                                           min_samples_leaf=3, random_state=0)
        model.fit(X_train, y_train)
        y_train_pred_scores = model.predict(X_train)
        y_test_pred_scores = model.predict(X_test)
        train_scores.append(mean_absolute_error(y_train, y_train_pred_scores))
        val_scores.append(mean_absolute_error(y_test, y_test_pred_scores))
    return train_scores, val_scores

def test_reg(X_train, y_train, X_test, y_test, iterations):
    train_scores = []
    val_scores = []
    for i in n_estimators_list:   
        model = GradientBoostingClassifier(n_estimators=i, learning_rate=1, max_depth=3, 
                                           min_samples_leaf=3, random_state=0, max_features=max_features)
        model.fit(X_train, y_train)
        y_train_pred_scores = model.predict_proba(X_clf_train)[:,1]
        y_test_pred_scores = model.predict_proba(X_clf_test)[:,1]
        train_scores.append(roc_auc_score(y_clf_train, y_train_pred_scores))
        val_scores.append(roc_auc_score(y_clf_test, y_test_pred_scores))
    return train_scores, val_scores

def scoring_gbr_sklern(X_train, y_train, X_test, y_test, n_estimators=100, 
                       learning_rate=1, max_depth=3, random_state=0, max_features=None,
                       min_samples_leaf=1, verbose=False):
    scores_train = []
    scores_test = []
    iterations = []
    log_iters = list(set((np.logspace(math.log(1, 8), math.log(400, 8), 
                                      num=50, endpoint=True, base=8, 
                                      dtype=np.int))))
    log_iters.sort()
    for i in log_iters:
        model = GradientBoostingRegressor(n_estimators=i, learning_rate=learning_rate, 
                                          max_depth=max_depth, random_state=random_state,
                                          min_samples_leaf=min_samples_leaf, max_features=max_features)
        model.fit(X_train, y_train)
        y_train_pred_scores = model.predict(X_train)
        y_test_pred_scores = model.predict(X_test)
        scores_train.append(mean_squared_error(y_train, y_train_pred_scores))
        scores_test.append(mean_squared_error(y_test, y_test_pred_scores))
        iterations.append(i)
        if verbose:
            print(i, scores_train[-1], scores_test[-1])
    best_score = min(scores_test)
    best_iter = iterations[scores_test.index(best_score)]
    if verbose:
        print('Best score: {}\nBest iter: {}'.format(best_score, best_iter))
    return scores_train, scores_test, iterations, model

def scoring_gbc_sklern(X_train, y_train, X_test, y_test, n_estimators=100, 
                       learning_rate=1, max_depth=3, random_state=0, max_features=None,
                       min_samples_leaf=1, verbose=False):
    scores_train = []
    scores_test = []
    iterations = []
    weight_0 = 1
    weight_1 = (len(y_train) - y_train.sum())/y_train.sum()
    sample_weights = [weight_1 if i else weight_0 for i in y_train]
    log_iters = list(set((np.logspace(math.log(1, 8), math.log(500, 8), 
                                      num=50, endpoint=True, base=8, 
                                      dtype=np.int))))
    log_iters.sort()
    for i in log_iters:
        model = GradientBoostingClassifier(n_estimators=i, learning_rate=learning_rate, 
                                          max_depth=max_depth, random_state=random_state,
                                          min_samples_leaf=min_samples_leaf, max_features=max_features)
        model.fit(X_train, y_train, sample_weight=sample_weights)
        y_train_pred_scores = model.predict_proba(X_train)
        y_test_pred_scores = model.predict_proba(X_test)
        scores_train.append(roc_auc_score(y_train, y_train_pred_scores[:,1]))
        scores_test.append(roc_auc_score(y_test, y_test_pred_scores[:,1]))
        iterations.append(i)
        if verbose:
            print(iterations[-1], scores_train[-1], scores_test[-1])
    best_score = max(scores_test)
    best_iter = iterations[scores_test.index(best_score)]
    if verbose:
        print('Best score: {}\nBest iter: {}'.format(best_score, best_iter))
    return scores_train, scores_test, iterations, model

def test_encoding(train_data, test_data, cols_to_encode, target_col, encoding_funcs, 
                  scoring_func, scoring_func_params={}, other_cols_to_use=None,
                  alpha=0):
    y_train = train_data[target_col]
    y_test = test_data[target_col]
    X_train_cols = []
    X_test_cols = []
    for encoding_func in encoding_funcs:  
        if (encoding_func==mean_encode) or (encoding_func==mean_and_freq_encode):
            encoded_features = encoding_func(train_data, test_data, cols_to_encode, 
                                             target_col=target_col, alpha=alpha)
        else:
            encoded_features = encoding_func(train_data, test_data, cols_to_encode)
        X_train_cols.append(encoded_features[0]), 
        X_test_cols.append(encoded_features[1])
    X_train = pd.concat(X_train_cols, axis=1)
    X_test = pd.concat(X_test_cols, axis=1)
    if other_cols_to_use:
        X_train = pd.concat([X_train, train_data.loc[:, other_cols_to_use]], axis=1)
        X_test = pd.concat([X_test, test_data.loc[:, other_cols_to_use]], axis=1)
    return scoring_func(X_train, y_train, X_test, y_test, **scoring_func_params)

def describe_dataset(data, target_col):
    ncats = []
    ncats10 = []
    ncats100 = []
    nsamples_median = []
    X_col_names = list(data.columns)
    X_col_names.remove(target_col)
    print('Number of samples: ', data.shape[0])
    for col in X_col_names:
        counts = data.groupby([col])[col].count()
        ncats.append(len(counts))
        ncats10.append(len(counts[counts<10]))
        ncats100.append(len(counts[counts<100]))
        nsamples_median.append(counts.median())
    data_review_df = pd.DataFrame({'Column':X_col_names, 'Number of categories':ncats, 
                                   'Categories with < 10 samples':ncats10,
                                   'Categories with < 100 samples':ncats100,
                                   'Median samples in category':nsamples_median})
    data_review_df = data_review_df.loc[:, ['Column', 'Number of categories',
                                             'Median samples in category',
                                             'Categories with < 10 samples',
                                             'Categories with < 100 samples']]
    return data_review_df.sort_values(by=['Number of categories'], ascending=False)

def make_vgsales():
    vgsales = pd.read_csv('../input/vgsales1.csv')
    vgsales = vgsales.loc[(vgsales['Year'].notnull()) & (vgsales['Publisher'].notnull()), 
                         ['Platform', 'Genre', 'Publisher', 'Year', 'Global_Sales']]
    vgsales['Year'] = vgsales.loc[:,['Year']].astype('str')
    vgsales['Platform x Genre'] = vgsales['Platform'] + '_' + vgsales['Genre']
    vgsales['Platform x Year'] = vgsales['Platform'] + '_' + vgsales['Year']
    vgsales['Genre x Year'] = vgsales['Genre'] + '_' + vgsales['Year']
    return vgsales

def make_poverty():
    poverty = pd.read_csv('../input/A_indiv_train.csv')
    poverty_cols_to_use = ['HeUgMnzF', 'gtnNTNam', 'XONDGWjH', 'hOamrctW', 'XacGrSou', 
                           'ukWqmeSS', 'SGeOiUlZ', 'RXcLsVAQ', 'poor']
    poverty['poor'] = poverty['poor'].astype(int)
    poverty = poverty.loc[:, poverty_cols_to_use]
    return poverty

def make_poverty_interaction():
    poverty = pd.read_csv('../input/A_indiv_train.csv')
    poverty_cols_to_use = ['HeUgMnzF', 'gtnNTNam', 'XONDGWjH', 'hOamrctW', 'XacGrSou', 
                            'ukWqmeSS', 'SGeOiUlZ', 'RXcLsVAQ', 'poor']
    poverty = poverty.loc[:, poverty_cols_to_use]
    poverty.loc[:, poverty_cols_to_use[:-1]] = poverty.loc[:, poverty_cols_to_use[:-1]].astype(str)
    poverty['poor'] = poverty['poor'].astype(int)
    poverty['interaction_1'] = poverty['HeUgMnzF'] + poverty['XONDGWjH']
    poverty['interaction_2'] = poverty['gtnNTNam'] + poverty['hOamrctW']
    poverty['interaction_3'] = poverty['XONDGWjH'] + poverty['XacGrSou']
    poverty['interaction_4'] = poverty['hOamrctW'] + poverty['ukWqmeSS']
    poverty['interaction_5'] = poverty['XacGrSou'] + poverty['SGeOiUlZ']
    poverty['interaction_6'] = poverty['ukWqmeSS'] + poverty['RXcLsVAQ']
    poverty['interaction_7'] = poverty['SGeOiUlZ'] + poverty['RXcLsVAQ']
    poverty['interaction_8'] = poverty['HeUgMnzF'] + poverty['gtnNTNam']
    poverty['interaction_9'] = poverty['ukWqmeSS'] + poverty['hOamrctW']
    poverty['interaction_10'] = poverty['XONDGWjH'] + poverty['RXcLsVAQ']
    return poverty

def make_poverty_interaction_only():
    poverty = pd.read_csv('../input/A_indiv_train.csv')
    poverty_cols_to_use = ['HeUgMnzF', 'gtnNTNam', 'XONDGWjH', 'hOamrctW', 'XacGrSou', 
                            'ukWqmeSS', 'SGeOiUlZ', 'RXcLsVAQ', 'poor']
    poverty = poverty.loc[:, poverty_cols_to_use]
    poverty.loc[:, poverty_cols_to_use[:-1]] = poverty.loc[:, poverty_cols_to_use[:-1]].astype(str)
    poverty['poor'] = poverty['poor'].astype(int)
    poverty_interactions = poverty.loc[:,['poor']]
    poverty_interactions['interaction_1'] = poverty['HeUgMnzF'] + poverty['XONDGWjH']
    poverty_interactions['interaction_2'] = poverty['gtnNTNam'] + poverty['hOamrctW']
    poverty_interactions['interaction_3'] = poverty['XONDGWjH'] + poverty['XacGrSou']
    poverty_interactions['interaction_4'] = poverty['hOamrctW'] + poverty['ukWqmeSS']
    poverty_interactions['interaction_5'] = poverty['XacGrSou'] + poverty['SGeOiUlZ']
    poverty_interactions['interaction_6'] = poverty['ukWqmeSS'] + poverty['RXcLsVAQ']
    poverty_interactions['interaction_7'] = poverty['SGeOiUlZ'] + poverty['RXcLsVAQ']
    poverty_interactions['interaction_8'] = poverty['HeUgMnzF'] + poverty['gtnNTNam']
    poverty_interactions['interaction_9'] = poverty['ukWqmeSS'] + poverty['hOamrctW']
    poverty_interactions['interaction_10'] = poverty['XONDGWjH'] + poverty['RXcLsVAQ']
    return poverty_interactions

def make_ctr():
    ctr = pd.read_csv('../input/ctr_data.csv', nrows=100000)
    ctr = ctr.astype('str')
    ctr['click'] = ctr['click'].astype('int')
    ctr['interaction_1'] = (ctr['site_category'] + ctr['C15'] + ctr['C16'] + ctr['C20'] + ctr['C17'])
    ctr['interaction_2'] = (ctr['site_category'] + ctr['C18'] + ctr['C19'] + ctr['device_model'])
    ctr_cols_to_use = ['site_id', 'app_id', 'device_id',
                      'device_model', 'C14', 'interaction_1', 'interaction_2', 'click']
    ctr = ctr.loc[:, ctr_cols_to_use]
    return ctr

def make_movie():
    movie = pd.read_csv('../input/IMDB-Movie-Data1.csv').loc[:, ['Genre', 'Year', 'Rating', 
                                                       'Revenue (Millions)']]
    movie = movie.loc[movie['Revenue (Millions)'].notnull(), :]
    movie['Year x Rating'] = movie['Year'] + movie['Rating']
    return movie

def make_credit():
    credit = pd.read_csv('../input/credit1.csv')
    cols = list(credit.columns)
    cols.remove('Unnamed: 0')
    return credit.loc[:, cols]

def encoding_stats(train_data, test_data, X_train, X_test, target_col, encoding_function,
                  feature_cols_to_use):
    if encoding_function.__name__ == 'one_hot_encode':
        return np.nan, np.nan, np.nan, np.nan
    if encoding_function.__name__ == 'mean_encode':
        enc_suffix = 'mean_'+target_col+'_'
    if encoding_function.__name__ == 'freq_encode':    
        enc_suffix = 'freq_'
    if encoding_function.__name__ == 'label_encode':
        enc_suffix = 'label_'
    cols_to_encoded_mapping = {}
    for col in feature_cols_to_use:
        for col_enc in X_train.columns:
            if col == col_enc[len(enc_suffix):]:
                cols_to_encoded_mapping[col] = col_enc
    train_conc = pd.concat([train_data, X_train], axis=1)
    test_conc = pd.concat([test_data, X_test], axis=1)
    mean_stds_train = []
    std_means_train = []
    mean_stds_test = []
    std_means_test = []
    for key in cols_to_encoded_mapping.keys():
        #how much randomisation added
        mean_stds_train.append(train_conc.groupby(key)[cols_to_encoded_mapping[key]].std().mean())
        mean_stds_test.append(test_conc.groupby(key)[cols_to_encoded_mapping[key]].std().mean())
        # how distinguishable are categories with that encoding
        std_means_train.append(train_conc.groupby(key)[cols_to_encoded_mapping[key]].mean().std())
        std_means_test.append(test_conc.groupby(key)[cols_to_encoded_mapping[key]].mean().std())
    
    encoding_stats = (np.mean(mean_stds_train), np.mean(std_means_train),
                      np.mean(mean_stds_test), np.mean(std_means_test))
    return encoding_stats

def test_all_encodings(train_data, test_data, target_col, testing_params, 
                       test_one_hot=False, regression=False, skip_first_iters_graph=0,
                      max_features_one_hot=0.01):
    encoding_settings = [[label_encode, {}, 'Label encoding', '#960000'],
                         [freq_encode, {}, 'Frequency encoding', '#FF2F02'],
                         [mean_encode, {'alpha':0, 'folds':None, 'reg_method':None, 
                                        'add_random':False, 'rmean':0, 'rstd':0.0,
                                        'target_col':target_col},
                         'Mean encoding, alpha=0', '#A4C400'],
                         [mean_encode, {'alpha':2, 'folds':None, 'reg_method':None, 
                                        'add_random':False, 'rmean':0, 'rstd':0.0,
                                        'target_col':target_col}, 
                         'Mean encoding, alpha=2', '#73B100'],
                         [mean_encode, {'alpha':5, 'folds':None, 'reg_method':None, 
                                        'add_random':False, 'rmean':0, 'rstd':0.0,
                                        'target_col':target_col}, 
                         'Mean encoding, alpha=5', '#2B8E00'],
                         [mean_encode, {'alpha':5, 'folds':3, 'reg_method':'k_fold', 
                                        'add_random':False, 'rmean':0, 'rstd':0.0,
                                        'target_col':target_col}, 
                         'Mean encoding, alpha=5, 4 folds', '#00F5F2'],
                         [mean_encode, {'alpha':5, 'folds':5, 'reg_method':'k_fold', 
                                        'add_random':False, 'rmean':0, 'rstd':0.0,
                                        'target_col':target_col}, 
                         'Mean encoding, alpha=5, 7 folds', '#00BAD3'],
                         [mean_encode, {'alpha':5, 'folds':None, 'reg_method':'expanding_mean', 
                                        'add_random':False, 'rmean':0, 'rstd':0.0,
                                        'target_col':target_col}, 
                         'Mean encoding, alpha=5, expanding mean', '#B22BFA']]
    review_rows = []
    if test_one_hot:
        oh_settings = [[one_hot_encode, {}, 'One hot encoding', '#E7E005']]
        encoding_settings = oh_settings + encoding_settings
    feature_cols_to_use = list(train_data.columns)
    feature_cols_to_use.remove(target_col)
    if regression:
        scoring_function = scoring_gbr_sklern
        best_score_function = min
    else:
        scoring_function = scoring_gbc_sklern
        best_score_function = max     
    plt.figure(figsize=(10,7))
    for encoding_function, encoding_params, str_name, color in encoding_settings:
        if encoding_function.__name__ == 'one_hot_encode':
            testing_params['max_features'] = max_features_one_hot
        else:
            testing_params['max_features'] = None
        X_train, X_test = encoding_function(train_data, test_data, feature_cols_to_use,
                                            **encoding_params)
        scores = scoring_function(X_train, train_data[target_col], X_test, 
                                    test_data[target_col], 
                                    min_samples_leaf=1, max_depth=3, **testing_params)
        skip_it = int(skip_first_iters_graph)
        train_scores, test_scores, iters, model_ = scores
        plt.plot(iters[skip_it:], 
                 test_scores[skip_it:], 
                 label='Test, ' + str_name, linewidth=1.5, color=color)
        best_score_test = best_score_function(test_scores)
        best_iter_test = iters[test_scores.index(best_score_test)]
        best_score_train = best_score_function(train_scores[:best_iter_test])
        print('Best score for {}: is {}, on iteration {}'.format(str_name, 
                                                                 best_score_test, 
                                                                 best_iter_test,
                                                                 best_score_train))
        enc_stats = encoding_stats(train_data, test_data, X_train, X_test, 
                                   target_col, encoding_function, feature_cols_to_use)
        review_rows.append([str_name, best_score_train, best_score_test, best_iter_test] + list(enc_stats))
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    if regression:
        columns=['Encoding', 'Train RMSE score on best iteration', 
             'Best RMSE score (test)', 'Best iteration (test)',
             'EV (train)', 'ED (train)', 'EV (test)', 'ED (test)']
    else:
        columns=['Encoding', 'Train AUC score on best iteration', 
             'Best AUC score (test)', 'Best iteration (test)',
             'EV (train)', 'ED (train)', 'EV (test)', 'ED (test)']
    return pd.DataFrame(review_rows, columns=columns)

# <div id="2"><h2>2. High cardinality categorical features</h2></div>
# 
# Features with a lot of different categories require different approaches to encoding then low cardinality features. One-hot encoding will create a huge amount of columns and harm the column sampling process in tree methods: one-hot encoded columns will overcrowd all other predictors and make high cardinality feature to be way too disproportionally important for a model. Label encoding will be hard to process for a model, because of how random it will look and how many splits will be needed.
# 
# <div id="21"><h3>2.1 Feature interactions usually cause high cardinality</h3></div>
# 
# In many settings, hand-picked feature interactions can be quite useful. They help a model to extract information from the data with fewer splits and improve its performance. By combining of several categorical variables, or binned continuous variables, we usually create a new high-cardinality feature. That creates even more need for a proper encoding scheme for those.

# <div id="3"><h2>3. Popular encodings for categorical variables, advantages and disadvantages for tree models</h2></div>
# 
# <div id="31"><h3>3.1 One-hot encoding</h3></div>
# 
# One-hot encoding maps each category to a vector in $R^{(n-1)}$ or $R^{(n-k)}$ where $n>k>1$ if we want to skip some categories. Each vector in a vector contains one '1' and all the rest of its values are '0'. That encoding is usually used in linear models and is not the best choice for tree models.
# 
# __Advantages__:
# 
# - Each category is properly separated from others. We put no assumptions about the relationships between categories with that encoding, so the model is less biased in that sense.
# 
# - Very easy to implement.
# 
# __Disadvantages__:
# 
# - For high cardinality features, one hot encoding produces a lot of columns. It slows down learning significantly and if we the model randomly samples a fraction of features for each tree or split, then the chances of one-hot encoded feature to be present in a sample are artificially increased, and chances of other (not one-hot encoded) variables to be considered on a split/tree are reduced. That makes a model treat one-hot encoded features as more useful than not one-hot encoded.
# 
# - On each split, trees can only separate one category from the others. Trees have to put every category in a separate bin, there is no other way to split a one hot encoded column but between 0 and 1. That leads to more splits needed to achieve same train accuracy as other, more compact encodings. That again, slowers learning and prevents trees from putting similar categories together in one bin, which might reduce the quality of the model.
# 
# <div id="32"><h3>3.2 Label and frequency encoding</h3></div>
# 
# Label encoding is a mapping of each category to some number in $R^1$. Numbers (labels) are usually chosen in a way that has no or almost no meaning in terms of relationships between categories. So, categories encoded with numbers that close are to each other (usually) are not more related then categories encoded with numbers that far away from each other.
# 
# Frequency encoding is a way to utilize frequencies of categories as labels. It can help if frequency correlates with the target and also, it can help the model to understand that smaller categories are less trustworthy then bigger ones, especially when frequency encoding is used parallel with other type of encoding.
# 
# __Advantages (compared to one-hot encoding)__:
# 
# - Faster learning than with one-hot representations. Numbers in $R^1$ are way more compact representations then vectors in $R^{n-1}$ used in one-hot encoding, that leads to fewer features for trees, which leads to faster learning.
# 
# - Less splits needed, means a more robust model. Unlike with one hot encoding, here trees can separate several categories at a time (with one-hot it is always 1). 
# 
# - Easy to implement.
# 
# __Disadvantages__:
# 
# - Bias. Label encoding is biased in a way that assumes a particular ordered relationship between categories. In reality, due to randomness in assigning labels, that relationship does not exist. 
# 
# - Nonlinearity towards target variable, more splits. At each binary split, tree models need to find a value of a variable that separates the target as good as possible. It is way harder to achieve when feature and target have almost zero linear dependencies. Trees have to do a lot of splits to put individual categories in separate buckets to continue to improve train loss.

# <div id="4"><h2>4. Mean target encoding</h2></div>
# 
# Plain mean target encoding can be viewed as a variation of label encoding, that is used to make labels correlate with the target. For each category, we set it's label to the mean value of a target variable on a training data. 
# 
# <div align='center'> <font size="4"> $label_c = p_c$</font></div>
# 
# Where $p_c$ is a mean target value for the category $c$.
# 
# We __do not__ use test data to estimate encodings, for the obvious reason: we should treat test data like we do not know target for it.
# 
# __Advantages (compared to label encoding)__:
# 
# - Fewer splits, faster learning. Trees can utilize the linear relationships between labels and target. This is especially useful when working with high cardinality categorical features: it is hard for a model to put every small category into a separate bucket, but if a lot of small categories can be put in one bin together based on their mean target value, then trees can learn way faster.
# 
# - Less bias, since now labels have more meaning: closer labels mean closer categories in terms of the mean target variable.
# 
# __Disadvantages__:
# 
# - Harder to construct and do validation.
# - Easy to overfit if regularisation is not used.
# 
# <div id="41"><h3>4.1 Overfitting with mean target encoding</h3></div>
# 
# When we dealing with high cardinality features, a lot of categories have a small number of samples in them. Many of those categories will look like great predictors for a model, when in fact they are not. 
# 
# Consider a binary target $Y$, and let's suppose that it's distribution for some category is completely random:
# </br>
# <div align='center'>$P(Y|category_1) = Bernoulli(0.5)$</div>
# 
# That implies that the category has 0 predicting power (when considered by itself). 
# 
# But, let's say now we have 5 examples in train data with that category (equivalent to sampling 5 times from the distribution Bernoulli(0.5)), what is the probability category will look like a good predictor? Probability to get all five 1's or all five 0's is 0.0625. So, if we have 100 categories like this one, with 5 samples and 0 predicting power, we expect at least 6 of them to have a target of all 0'r or target of all 1's. 
# 
# Then, if we add combinations where four values of the target are the same (still looks like a great predictor), the probability of that happening equals 0.375! More than every third category with 0 predicting power, on a small sample size of 5 samples will look like a decent predictor.
# 
# Trees will put those categories in a separate leafs and will learn to predict extreme values for those categories. Then, when we get the same categories in test data, most of them will not have the same target distribution, and model predictions will be wrong.
# 
# <div id="42"><h3>4.2 Using the prior probability for regularisation</h3></div>
# 
# Let's keep using a binary response variable as an example. Regression tasks will all have the same intuitions as binary classification behind them. 
# 
# Simpliest regularisation technique is to move encodings of rare categories closer to the dataset target mean, both on train and test data. That way we hope that the model will be less likely learn very high or very low probabilities for small categories: their encodings will now be closer to the middle and kind of blend with encodings for categories that have a smaller mean target value. 
# 
# Now encoding for each category in both train and test data will be estimated as:
# 
# <div align='center'> <font size="4"> $label_c = \frac{(p_c*n_c + p_{global}*\alpha)}{(n_c+\alpha)}$</font></div>
# 
# Where $p_c$ is a target mean for a category, $n_c$ is a number of samples in a category, $p_{global}$ is a global target mean and $\alpha$ is a regularisation parameter that can be viewed as a size of a category you can trust.
# 
# Drawbacks are obvious, encoding is still just pseudo continuous, every category will be encoded with a distinct real number, that still allows the model to put any category in a distinct leaf and set an extreme target probability for the leaf. It's is just a bit harder for the model to do it now, because the encodings for those categories are closer to the middle, instead of also having extreme values, so the model needs more splits to overfit.
# 
# What we need is to somehow randomize encodings inside a category, so that those small categories with extreme target values will be hard to put in a distinct bin, but bigger categories that we cat trust will still have its encodings close the mean target value for them. Randomise, but still, let model learn that there is a correlation between category encoding and the target variable.
# 
# <div id="43"><h3>4.3 K-fold regularisation for mean encodings</h3></div>
# 
# The idea behind k-fold regularisation is to use just a part of examples of a category to estimate the encoding for that category. Split the data into k-folds and for each example, estimate it's encoding using all folds except the one that the example belongs to. We can use the global mean regularisation in combination with k-fold for more robust encoding.
# 
# Now the encoding estimation is a 3-step procedure:
# 
# 1. Split train data into k folds.
# 
# Then, for each sample:
# 
# 2. Exclude the fold that contains a sample.
# 
# 3. Estimate encoding for the sample with the data that left using equation above
# 
# Notice that k-fold regularisation is only performed for training data, test labels are still estimated from all training examples, since we do not supposed to know the labels to the test data and definitely cannot use them for estimating encodings.
# 
# I believe that the best number of folds is between 3 and 6, depending on how much randomization you want to achieve. Fewer folds for stronger randomization in terms of bigger variance of labels for each category.
# 
# <div id="44"><h3>4.4 Expanding mean regularisation</h3></div>
# 
# Process:
# 1. Fix some random permutation of rows (samples)
# 2. Moving from top to bottom, for each example, estimate the encoding using target mean of all the examples before the estimated one. An estimated example is not used.
# 
# That way we create even more randomization of encoding inside each category. Every train sample in each category now estimated from a different subsample of train data. Smaller categories got more randomization, bigger categories got less randomization and more samples encoded with numbers closer to the category mean, by the law of large numbers. That is a very useful property because overfitting arises mostly because of those smaller categories. 
# 
# The main drawback is that sometimes this regularisation can add to much noise and make a lot of categories useless. Examples of that happening will be shown later in this article.

# <div id="5"><h2>5. Datasets</h2></div>
# 
# More datasets will be extended in the future, but for now I used those six to study the effectiveness of mean encoding compared to other encoding methods and compare different regularisation schemes:
# 
# 1. <a href='https://www.kaggle.com/PromptCloudHQ/imdb-data'>IMDB movie revenue</a>. 872 samples. Regression task, a small dataset, mixed high and low cardinality features.
# 1. <a href='https://www.kaggle.com/gregorut/videogamesales'>Video Game Sales</a>. 16291 samples. Regression task, bigger dataset, mixed high and low cardinality features.
# 1. <a href='https://www.drivendata.org/competitions/50/worldbank-poverty-prediction/'>World Bank poverty prediction</a>. 37560 samples. Classification, big dataset, features of a lower cardinality then other datasets.
# 1. <a href='https://www.drivendata.org/competitions/50/worldbank-poverty-prediction/'>World Bank poverty  prediction (interactions)</a>. 37560 samples. Classification, big dataset, high cardinality features are created by feature interactions.
# 1. <a href='https://www.kaggle.com/c/home-credit-default-risk'>Home Credit Default Risk</a>. 109742 samples. Classification, big dataset, a mix of high and low cardinality features. Low cardinality features are created by feature interactions.
# 1. <a href='https://www.kaggle.com/c/avazu-ctr-prediction/data'>Avazu Click-Through Rate Prediction</a>. 100000 samples.  Classification, big dataset, extremely high cardinality features.
# 
# I changed a bit in every dataset, created some new features using feature interactions so that I have more high cardinality categorical variables to work with. Also, I deleted all numerical columns (or binned them), for clearer experiment results and in some, used just a part of the data.

# <div id="6"><h2>6. Encoding tests</h2></div>
# 
# On every dataset, I trained a sklearn implementation of gradient boosted trees, made a train/test split, capture the best test score the model achieved and the number of iteration that score was achieved at.
# 
# I also provide two important statistics about the mean encodings that will help us understand how much randomness each regularisation adds to the train data. 
# 
# <div id="61"><h3>6.1 Testing procedure</h3></div>
# 
# 1. Train/test split
# 2. Encode all columns using the same method. When dealing with mean encoding regularisations, regularise only train data.
# 3. Test a gradient boosted trees model from sklearn several times, with a different number of learners.
# 4. Summarise the results
# 
# <div id="62"><h3>6.2 Encoding statistics</h3></div>
# 
# 1. __Encoding variability inside categories (EV)__. Describes the variation of encodings of individual samples inside one category. Obviously, when we do not use regularisation and encode all samples with one number (adjusted mean), this score is equals to 0. When we use k-fold or expanding mean methods, the score is > 0. <div align='center'> <font size="4"> $EV = \frac{1}{ncols}\cdot\left(\sum_{col}^{ }Var\left(E\left[p_c\right]\right)^{\frac{1}{2}}\right)$ </font> </div> Where $p_c$ is an encoding for a particular category of a particular column, $E\left[p_c\right]$ is an expected value of that encoding (mean of the encodings for individual samples, $Var$ is a variance and $ncols$ is a number of columns in that were analysed.
# 
# 2. __Encoding distinguishability between categories (ED)__. How much variation there are between the average of encoded samples in each category. So to be clear: we encode each category, then count the mean of encodings inside each category and then count standard deviation of those means. That shows how distinguishable encoded categories are, are the far away from each other or they concentrated in close areas. 
# 
# <div align='center'> <font size="4"> $ED = \frac{1}{ncols}\cdot\left(\sum_{col}^{ }E\left[Var\left(p_c\right)^{\frac{1}{2}}\right]\right)$ </font> </div>

# <div id="63"><h3>6.3 IMDB movie revenue</h3></div>
# 
# #### Dataset summary
# 
# Using genre, release year and rating, predict the movie sales.
# 
# 4 columns, one of which is a interaction. Data is just 872 rows. All columns except for the 'Year' can be considered a high cardinality features. 

# In[ ]:


movie = make_movie()
target_col = 'Revenue (Millions)'
describe_dataset(movie, target_col)

# #### Testing encodings

# In[ ]:


train_data, test_data = train_test_split(movie, test_size=0.3, random_state=4)
testing_params = {'learning_rate':0.2}

test_all_encodings(train_data, test_data, target_col, testing_params, 
                   test_one_hot=True, regression=True)

# #### Takeaways
# 
# - One-hot encoding showed way better train score than other methods.
# - Mean encodings converge to the best score way faster then one-hot and label encodings.
# - Mean encoding, alpha=5, 4 folds is the best regularisation for that dataset
# - Expanding mean regularisation added too much noise and the model was not able to get enough valuable information from data.

# <div id="64"><h3>6.4 Video Game Sales</h3></div>
# 
# #### Dataset summary
# 
# Predict video game sales using information about release year, publisher, genre and platform.
# 
# 7 columns, 3 of which are interactions. Data has 16291 rows. 1 very high cardinality column, 3 columns are high cardinality features and 3 are lower cardinality categoricals.

# In[ ]:


vgsales = make_vgsales()
target_col = 'Global_Sales'
describe_dataset(vgsales, target_col)

# #### Testing encodings

# In[ ]:


train_data, test_data = train_test_split(vgsales, test_size=0.3, random_state=4)
testing_params = {'learning_rate':0.35}

test_all_encodings(train_data, test_data, target_col, testing_params, 
                   test_one_hot=True, regression=True)

# #### Takeaways
# 
# - 'Mean encoding, alpha=5, 4 folds' showed the best result by far, even compared to 'Mean encoding, alpha=5, 7 folds'
# - Best regularisation for mean encoding ('Mean encoding, alpha=5, 4 folds') did not actually converge faster then label encoding and just slightly faster then frequency encoding.
# - Mean encodings with no regularisation and mean encoding with expanding mean regularisation showed very bad results, even compared to label and frequency encodings.
# - One-hot and mean encoding with expanding mean regularisation showed the worst results by far.

# <div id="65"><h3>6.5 World Bank poverty prediction</h3></div>
# #### Dataset summary
# 
# 8 columns, no interactions. Data has 37560 rows. All columns are low or average cardinality.

# In[ ]:


poverty = make_poverty()
target_col = 'poor'
describe_dataset(poverty, target_col)

# #### Testing encodings

# In[ ]:


train_data, test_data = train_test_split(poverty, test_size=0.3, random_state=4)
testing_params = {'learning_rate':0.4}

test_all_encodings(train_data, test_data, target_col, testing_params, 
                   test_one_hot=True, regression=False)

# #### Takeaways
# 
# - Frequency encoding showed the best result on the dataset, followed by unregularised mean encoding with alpha=5.
# - More encoding randomization clearly does not help the model on this dataset. We can clearly see a negative correlation between the randomization of encodings and AUC score.
# - Frequency encoding is actually the less randomized one because some categories have the same frequencies.
# - Expanding mean regularisation added way too much noise, the score is way behind scores with other encodings, overfitting is fast and severe.

# <div id="66"><h3>6.6 World Bank poverty prediction with feature interactions</h3></div>
# #### Dataset summary
# 18 columns, 10 interactions. Data has 37560 rows. All columns are different cardinality.

# In[ ]:


poverty_interactions = make_poverty_interaction()
target_col = 'poor'
describe_dataset(poverty_interactions, target_col)

# #### Testing encodings

# In[ ]:


train_data, test_data = train_test_split(poverty_interactions, test_size=0.3, random_state=4)
testing_params = {'learning_rate':0.4}

test_all_encodings(train_data, test_data, target_col, testing_params, 
                   test_one_hot=True, regression=False)

# #### Takeaways
# 
# - All that applies to the dataset above (without feature interactions) is valid here.
# - Frequency encoding worked by far the best, supposedly because it helps a model understand which categories are very small and have no predictive power. Most likely, small categories are bad predictors in that dataset, so the model learned not to use them and got a better generalization (this needs further investigation).
# - Mean encoding with 4-fold regularisation showed a second best result, but it is far from frequency encoding score.
# - Expanding mean encoding overfits fast and severe, again.
# - What changed is that now mean encodings with no regularisation overfit more, and k-fold regularides one work better.

# <div id="67"><h3>6.7 Home Credit Default Risk</h3></div>
# #### Dataset summary
# 15 columns, 10 of which are interactions of other columns (some might not be included here). Data has 109742 rows. Columns are high or average cardinality.

# In[ ]:


credit = make_credit()
target_col = 'TARGET'
describe_dataset(credit, target_col)

# #### Testing encodings

# In[ ]:


train_data, test_data = train_test_split(credit, test_size=0.3, random_state=4)
testing_params = {'learning_rate':0.07}

test_all_encodings(train_data, test_data, target_col, testing_params, 
                   test_one_hot=False, regression=False)

# #### Takeaways
# 
# - Expaning mean regularisation finally performs really well, when the number of samples is big enough.
# - Frequency encoding takes twice as many trees to got to a good result copmared to regularised mean encodings.
# - Mean encodings with no regularisation performed very badly, espetially the one with $\alpha=0$, overfitting is hude there with 0.972738 as a train score and just 0.595126 on the same iterarion.

# <div id="68"><h3>6.8 Avazu Click-Through Rate Prediction</h3></div>
# 
# #### Dataset summary
# 
# All columns are very high crdinality. 100000 rows.

# In[ ]:


ctr = make_ctr()
target_col = 'click'
describe_dataset(ctr, target_col)

# #### Testing encodings

# In[ ]:


train_data, test_data = train_test_split(ctr, test_size=0.3, random_state=4)
testing_params = {'learning_rate':0.2}

test_all_encodings(train_data, test_data, target_col, testing_params, 
                   test_one_hot=False, regression=False)

# #### Takeaways
# 
# - Regularised mean encodings showed the best results on this data with very high cardinality features. That was expected, this case is actually a perfect case for mean encoding.
# - Frequency encoding showed a very good result, again, but again with a huge amount of trees needed to get to a good score.
# - Unregularised mean encodings all cause overfitting and lead to a bad score.

# <div id="7"><h2>7. Summary and conclusions</h2></div>
# 

# <div id="71"><h3>7.1 Summary</h3></div>

# In[ ]:


agg_results = {'Encodings':['Samples', 'Cardinality', 'One-hot','Label','Frequency','Mean encoding, alpha=0',
                            'Mean encoding, alpha=2','Mean encoding, alpha=5',
                            'Mean encoding, alpha=5, 4 folds','Mean encoding, alpha=5, 7 folds',
                            'Mean encoding, alpha=5, expanding mean'],
              'IMDB movie revenue':['872', 'High/Average', 1, 6, 9, 5, 4, 7, 2, 3, 8],
              'Video Game Sales':['16291', 'High/average/low', 8, 3, 4, 7, 5, 6, 1, 2, 9],
              'World Bank poverty prediction':['37560', 'Average/low', 6, 5, 1, 3, 4, 2, 8, 7, 9],
              'World Bank poverty prediction with feature interactions':['37560', 'High/average/low', 
                                                                         6, 2, 1, 7, 8, 4, 3, 5, 9], 
              'Home Credit Default Risk':['109742', 'High/average', 'not tested', 7, 4, 8, 6, 5, 1, 2, 3],
              'Avazu Click-Through Rate Prediction':['100000', 'High', 'not tested', 5, 4, 8, 7, 6, 1, 2, 3],
              'Median position':[np.nan, np.nan, 6, 5, 4, 7, 5, 5, 1, 2, 8]}
agg_results = pd.DataFrame(agg_results)
agg_results = agg_results.loc[:,['Encodings', 'Median position', 'IMDB movie revenue', 'Video Game Sales',
                                 'World Bank poverty prediction', 
                                 'World Bank poverty prediction with feature interactions',
                                 'Home Credit Default Risk', 'Avazu Click-Through Rate Prediction']]
agg_results

# <div id="72"><h3>7.2 Conclusions</h3></div>

# For features with hagh cardinality:
# 
# 1. I suggest to try mean encoding with 4 or 5 folds and $\alpha = 5$ first, and then try a frequency encoding. Maybe using both encodings will benefit the model.
# 2. Then if the dataset is big enough (>50000 examples) and the score is still unsutisfying, try expanding mean regularisation.
# 3. Try one-hot encodings for high cardinality features only in case of very small datasets, or if you just really, really want o try it, your gut just screams that it will work best, then do it.
# 4. There is no point in using mean encoding with $\alpha = 0$ and no regularisation. At least use $\alpha = 5$ and even that will be not as good as k-fold regularisation on a bigger datasets, but can sometimes work better on smaller ones (<50000).
# 
# For features with low cardinality:
# 
# 1. I suggest you to try label encoding and frequency encoding first.
# 2. If the results are unsutisfcing, try mean encoding with 4 or 5 folds and $\alpha = 5$.
# 3. Then you can try to create some feature interactions and try different mean encodings for them.
# 
# For datasets with < 50000 rows:
# 
# 1. Try label encoding and frequency encoding first. If the dataset is very small, try one-hot encoding, but remember, it will affect feature samppling process, so it is probably better to use all features for every tree/split, the data is small anyway, so the speed shoyld not be a problem.
# 2. Do not use too much regularisation with mean encodings, 4 or 5 folds and $\alpha = 5$ is the first choice again, expanding mean, most likely, will not work here.
