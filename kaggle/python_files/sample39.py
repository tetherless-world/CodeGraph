#!/usr/bin/env python
# coding: utf-8

# # Santander Customer Transaction Prediction
# Hi, this kernel produces a top 5 submission with public/private LB of 0.92569/0.92446 running in 1.5 hours on kaggle servers. It was written in a very short time thus containing some mistakes which aren't edited. This achievement wouldn't have been possible without the huge insights and inspiration I gained from many great kernels / discussions, thank you! I also want to thank my teammates interneuron and Chua Cheng Hong for helping me getting the most out of this kernel and collectively achieving 3rd place in this competition. 

# # Imports

# In[1]:


import numpy as np
import pandas as pd
import lightgbm as lgb

from sklearn.metrics import roc_auc_score, log_loss
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold,KFold
from scipy.stats import norm, skew

from tqdm import tqdm_notebook as tqdm
from copy import copy
from multiprocessing import Pool

import warnings
warnings.filterwarnings('ignore')
np.random.seed(42)

# # Loading Data
# At this point I filter out the fakes (shoutout to YaG320) and concatenate train and test for future FE. Setting `use_experimental = True` splits the Train data into train / test which was useful for later NN training indicating whether a model is overfitting. I wasn't sure if the fakes are going to be used for final score evaluation, so I also applied all the transformations to them and kept them in a separate dataframe. 

# In[2]:


use_experimental = False

train_df = pd.read_csv('../input/santander-customer-transaction-prediction/train.csv')
test_df = pd.read_csv('../input/santander-customer-transaction-prediction/test.csv')

indices_fake = np.load('../input/list-of-fake-samples-and-public-private-lb-split/synthetic_samples_indexes.npy')
indices_pub = np.load('../input/list-of-fake-samples-and-public-private-lb-split/public_LB.npy')
indices_pri = np.load('../input/list-of-fake-samples-and-public-private-lb-split/private_LB.npy')
indices_real = np.concatenate([indices_pub, indices_pri])

features = [c for c in train_df.columns if c not in ['ID_code', 'target']]
target_train = train_df['target']
X_train = train_df
X_test = test_df.loc[indices_real,:]
X_test['target'] = np.zeros(X_test.shape[0])
X_fake = test_df.loc[indices_fake,:]
X_fake['target'] = np.zeros(X_test.shape[0])
train_length = X_train.shape[0]
target_test = X_test['target']
target_fake = X_fake['target']

if use_experimental:
    np.random.seed(42)    
    indices = np.arange(train_length)
    train_length = 150000
    np.random.shuffle(indices)
    indices_train = indices[:train_length]
    indices_test = indices[train_length:]
    # Swapped order to not overwrite X_train to soon
    X_test = X_train.iloc[indices_test,:]
    X_fake = X_train.iloc[indices_test,:]
    target_fake = X_fake['target']
    X_train = X_train.iloc[indices_train,:]
    target_train = X_train['target']
    target_test = X_test['target']

X_all = pd.concat([X_train, X_test])
print(X_all.shape)

# # Feature Engineering

# ## Counts, Density, Deviation
# Here I calculate the unique counts of each faeture seaparately. Based on that I also calculate the density by smoothing the counts and also the deviation as counts/density.

# In[3]:


import scipy.ndimage

sigma_fac = 0.001
sigma_base = 4

eps = 0.00000001

def get_count(X_all, X_fake):
    features_count = np.zeros((X_all.shape[0], len(features)))
    features_density = np.zeros((X_all.shape[0], len(features)))
    features_deviation = np.zeros((X_all.shape[0], len(features)))

    features_count_fake = np.zeros((X_fake.shape[0], len(features)))
    features_density_fake = np.zeros((X_fake.shape[0], len(features)))
    features_deviation_fake = np.zeros((X_fake.shape[0], len(features)))
    
    sigmas = []

    for i,var in enumerate(tqdm(features)):
        X_all_var_int = (X_all[var].values * 10000).round().astype(int)
        X_fake_var_int = (X_fake[var].values * 10000).round().astype(int)
        lo = X_all_var_int.min()
        X_all_var_int -= lo
        X_fake_var_int -= lo
        hi = X_all_var_int.max()+1
        counts_all = np.bincount(X_all_var_int, minlength=hi).astype(float)
        zeros = (counts_all == 0).astype(int)
        before_zeros = np.concatenate([zeros[1:],[0]])
        indices_all = np.arange(counts_all.shape[0])
        # Geometric mean of twice sigma_base and a sigma_scaled which is scaled to the length of array 
        sigma_scaled = counts_all.shape[0]*sigma_fac
        sigma = np.power(sigma_base * sigma_base * sigma_scaled, 1/3)
        sigmas.append(sigma)
        counts_all_smooth = scipy.ndimage.filters.gaussian_filter1d(counts_all, sigma)
        deviation = counts_all / (counts_all_smooth+eps)
        indices = X_all_var_int
        features_count[:,i] = counts_all[indices]
        features_density[:,i] = counts_all_smooth[indices]
        features_deviation[:,i] = deviation[indices]
        indices_fake = X_fake_var_int
        features_count_fake[:,i] = counts_all[indices_fake]
        features_density_fake[:,i] = counts_all_smooth[indices_fake]
        features_deviation_fake[:,i] = deviation[indices_fake]
        
    features_count_names = [var+'_count' for var in features]
    features_density_names = [var+'_density' for var in features]
    features_deviation_names = [var+'_deviation' for var in features]

    X_all_count = pd.DataFrame(columns=features_count_names, data = features_count)
    X_all_count.index = X_all.index
    X_all_density = pd.DataFrame(columns=features_density_names, data = features_density)
    X_all_density.index = X_all.index
    X_all_deviation = pd.DataFrame(columns=features_deviation_names, data = features_deviation)
    X_all_deviation.index = X_all.index
    X_all = pd.concat([X_all,X_all_count, X_all_density, X_all_deviation], axis=1)
    
    X_fake_count = pd.DataFrame(columns=features_count_names, data = features_count_fake)
    X_fake_count.index = X_fake.index
    X_fake_density = pd.DataFrame(columns=features_density_names, data = features_density_fake)
    X_fake_density.index = X_fake.index
    X_fake_deviation = pd.DataFrame(columns=features_deviation_names, data = features_deviation_fake)
    X_fake_deviation.index = X_fake.index
    X_fake = pd.concat([X_fake,X_fake_count, X_fake_density, X_fake_deviation], axis=1)    

    features_count = features_count_names
    features_density = features_density_names
    features_deviation = features_deviation_names
    return X_all, features_count, features_density, features_deviation, X_fake

X_all, features_count, features_density, features_deviation, X_fake = get_count(X_all, X_fake)
print(X_all.shape)

# ## Target encoding (unused)
# 

# In[ ]:


# # Also try to encode counts themselves

# weighting = 500
# n_splits = 2

# features_to_encode = features_count

# def get_encoding(X_all):
#     arr_all_int = (X_all[features_to_encode].values * 10000).round().astype(int)
#     arr_target_train = target_train.values 
#     arr_target_test = target_test.values
    
#     preds_oof = np.zeros((train_length, len(features)))
#     preds_test = np.zeros((arr_all_int.shape[0] - train_length, len(features)))
#     preds_train = np.zeros((train_length, len(features)))
    
#     for v ,var in enumerate(tqdm(features_to_encode)):
#         lo = arr_all_int[:,v].min()
#         arr_all_var = arr_all_int[:,v] - lo
#         hi = arr_all_var.max() + 1
#         arr_train_var = arr_all_var[:train_length]
#         arr_test_var = arr_all_var[train_length:]
#         folds = StratifiedKFold(n_splits=n_splits, shuffle=True)
#         for train_idx, val_idx in folds.split(arr_train_var, arr_target_train):
#             X_tr = arr_train_var[train_idx]
#             y_tr = arr_target_train[train_idx]
#             X_val = arr_train_var[val_idx]
#             y_val = arr_target_train[val_idx]
#             X_ts = arr_test_var
#             arr1 = X_tr[y_tr == 1]        
#             mean = arr1.shape[0] / X_tr.shape[0]
#             hits = np.bincount(arr1, minlength=hi).astype(float)
#             base = np.bincount(X_tr, minlength=hi).astype(float)
#             lamb = base / (base + weighting)
#             expected_target = (hits / (base+0.00000001)) * lamb + (1-lamb) * mean              

#             prediction_oof = expected_target[X_val]
#             prediction_test = expected_target[X_ts]
#             prediction_train = expected_target[X_tr]
#             preds_oof[val_idx,v] += prediction_oof 
#             preds_test[:,v] += prediction_test 
#             preds_train[train_idx,v] += prediction_train 
#         score_running_oof = roc_auc_score(arr_target_train, preds_oof.mean(axis=1))
#         score_running_test = roc_auc_score(arr_target_test, preds_test.mean(axis=1))
#         score_running_train = roc_auc_score(arr_target_train, preds_train.mean(axis=1))
        
#     preds_all = np.concatenate([preds_oof, preds_test], axis=0)
#     feature_names = [var+'_encoding' for var in features_to_encode]
#     X_all_encoding = pd.DataFrame(columns=feature_names, data = preds_all)
#     X_all_encoding.index = X_all.index
#     X_all = pd.concat([X_all,X_all_encoding], axis=1)
#     return X_all, feature_names

# X_all, features_encoding = get_encoding(X_all)

# print(X_all.shape)

# ## NB predictor (unused)

# In[ ]:


# def compress(arr, indices, thresh = 100000):
#     old_length = arr.shape[0] 
#     fac = old_length // thresh
#     if fac < 1:
#         return arr, indices
#     new_length = int(np.ceil(old_length / fac))+2
#     new_arr = np.zeros(new_length * fac)
#     new_arr[fac:old_length+fac] = arr
#     new_arr = new_arr.reshape(new_length, fac).sum(axis=1)
#     index_shift = indices[0]-0.5*(fac-1)-1
#     new_indices = np.arange(new_length) * fac + index_shift
#     return new_arr, new_indices    

# def transform_grouper(counts_orig, indices_orig, min_elems=10):
#     first_ind, first_count, last_ind, last_count = indices_orig[0], counts_orig[0], indices_orig[-1], counts_orig[-1]
#     indices = indices_orig[1:-1]
#     counts = counts_orig[1:-1]
#     new_indices = [np.array([first_ind])]
#     cur_indices = []
#     cur_counts = 0
#     for i, count in enumerate(counts):
#         cur_indices.append(indices[i])
#         cur_counts += count
#         if cur_counts >= min_elems:
#             new_indices.append(np.array(cur_indices))
#             cur_counts = 0
#             cur_indices = []
#     if cur_indices:
#         new_indices.append(np.array(cur_indices))
#     new_indices.append(np.array([last_ind]))
#     return new_indices

# def transform_space(counts, indices_grouped):
#     new_counts = []
#     new_indices = []
#     for arr_index in indices_grouped:
#         count = counts[arr_index]
#         count_sum = np.sum(count)
#         if count_sum > 0:
#             new_indices.append(np.sum(arr_index*count)/count_sum)
#         else:
#             new_indices.append(np.mean(arr_index))
#         new_counts.append(count_sum)
#     return np.array(new_counts), np.array(new_indices)

# def get_hist(arr, indices_grouped):
#     hi = indices_grouped[-1][0] + 1
#     counts = np.bincount(arr, minlength=hi)
#     indices = np.arange(counts.shape[0])
#     counts, indices = transform_space(counts, indices_grouped)
#     return counts / counts.sum(), indices

# def get_density_func(kde, indices, sigma=0.0001, resolution=2000):
#     kde_smooth = scipy.ndimage.filters.gaussian_filter1d(kde,sigma*len(indices))
#     return scipy.interpolate.interp1d(indices, kde_smooth, kind='linear')

# def get_p_x_t(x, t, density_funcs):
#     return density_funcs[t](x) 

# def get_p_1_x(x, density_funcs, prior=0.1, eps=0.00000000000000001):
#     p_x_0 = get_p_x_t(x,0,density_funcs)
#     p_x_1 = get_p_x_t(x,1,density_funcs)
#     p_x = (p_x_1*prior + p_x_0*(1-prior)+eps)
#     return p_x_1*prior / p_x, p_x

# import scipy.ndimage
# import scipy

# n_splits = 2

# def get_NB_predictor(X_all):   
#     arr_all_int = (X_all[features].values * 10000).round().astype(int)
#     arr_target_train = target_train.values 
#     arr_target_test = target_test.values
#     preds_oof = np.zeros((train_length, len(features)))
#     preds_test = np.zeros((arr_all_int.shape[0] - train_length, len(features)))
#     preds_train = np.zeros((train_length, len(features)))
#     for v, var in enumerate(features):
#         lo = arr_all_int[:,v].min()
#         arr_all_var = arr_all_int[:,v] - lo
#         hi = arr_all_var.max() + 1
#         arr_train_var = arr_all_var[:train_length]
#         arr_test_var = arr_all_var[train_length:]
#         counts_all = np.bincount(arr_all_var, minlength=hi)
#         indices_all = np.arange(counts_all.shape[0])
#         min_elems = [20]
#         sigmas = [0.05,0.02,0.005,0.002]
#         preds_oof_temp = np.zeros((preds_oof.shape[0], len(sigmas), len(min_elems)))
#         preds_test_temp = np.zeros((preds_test.shape[0], len(sigmas), len(min_elems)))
#         preds_train_temp = np.zeros((preds_train.shape[0], len(sigmas), len(min_elems)))
#         for j,min_elem in enumerate(min_elems):
#             indices_grouped = transform_grouper(counts_all, indices_all, min_elems=min_elem)            
#             kfold = StratifiedKFold(n_splits=n_splits, shuffle=False, random_state = np.random.randint(10000))
#             for train_idx, val_idx in kfold.split(arr_train_var, arr_target_train):
#                 X_tr = arr_train_var[train_idx]
#                 y_tr = arr_target_train[train_idx]
#                 X_val = arr_train_var[val_idx]
#                 y_val = arr_target_train[val_idx]
#                 X_ts = arr_test_var
#                 arr0 = X_tr[y_tr == 0]
#                 arr1 = X_tr[y_tr == 1]
#                 prior = arr1.shape[0] / (arr1.shape[0] + arr0.shape[0]) 
#                 kde0, indices0 = get_hist(arr0, indices_grouped)
#                 kde1, indices1 = get_hist(arr1, indices_grouped)
#                 for i,sigma in enumerate(sigmas): 
#                     density_func0 = get_density_func(kde0, indices0, sigma=sigma)
#                     density_func1 = get_density_func(kde1, indices1, sigma=sigma)
#                     prediction_oof, confidence = get_p_1_x(X_val, [density_func0, density_func1], prior=prior)
#                     prediction_test, confidence = get_p_1_x(X_ts, [density_func0, density_func1], prior=prior)
#                     prediction_train, confidence = get_p_1_x(X_tr, [density_func0, density_func1], prior=prior)
#                     preds_oof_temp[val_idx,i,j] += prediction_oof 
#                     preds_test_temp[:,i,j] += prediction_test 
#                     preds_train_temp[train_idx,i,j] += prediction_train 
#         scores = np.zeros((len(sigmas), len(min_elems)))
#         for i in range(len(sigmas)):
#             for j in range(len(min_elems)):
#                 scores[i,j] = log_loss(arr_target_train, preds_oof_temp[:,i,j])
#         sorted_indices = np.dstack(np.unravel_index(np.argsort(scores.ravel()), (len(sigmas), len(min_elems))))[0]
#         preds_oof[:,v] = preds_oof_temp[:,sorted_indices[0,0], sorted_indices[0,1]]
#         preds_test[:,v] = preds_test_temp[:,sorted_indices[0,0], sorted_indices[0,1]]
#         preds_train[:,v] = preds_train_temp[:,sorted_indices[0,0], sorted_indices[0,1]]
#         score_running_oof = roc_auc_score(arr_target_train, preds_oof.mean(axis=1))
#         score_running_test = roc_auc_score(arr_target_test, preds_test.mean(axis=1))
#         score_running_train = roc_auc_score(arr_target_train, preds_train.mean(axis=1))
#         print(var, sorted_indices[0], score_running_oof, score_running_test, score_running_train)
#     preds_all = np.concatenate([preds_oof, preds_test], axis=0)
#     feature_names = [var+'_pred' for var in features]
#     X_all_pred = pd.DataFrame(columns=feature_names, data = preds_all)
#     X_all_pred.index = X_all.index
#     X_all = pd.concat([X_all,X_all_pred], axis=1)
#     return X_all, feature_names
        
# X_all, features_pred = get_NB_predictor(X_all)

# print(X_all.shape)

# ## Standardize
# I standardize all the features (or supposedly so, apparently I forgot density and deviation being in time trouble). Which is important for later NN usage.

# In[4]:


features_to_scale = [features, features_count]

from sklearn.preprocessing import StandardScaler

def get_standardized(X_all, X_fake):
    scaler = StandardScaler()
    features_to_scale_flatten = [var for sublist in features_to_scale for var in sublist]
    scaler.fit(X_all[features_to_scale_flatten])
    features_scaled = scaler.transform(X_all[features_to_scale_flatten])
    features_scaled_fake = scaler.transform(X_fake[features_to_scale_flatten])
    X_all[features_to_scale_flatten] = features_scaled
    X_fake[features_to_scale_flatten] = features_scaled_fake
    return X_all, X_fake

X_all, X_fake = get_standardized(X_all, X_fake)

print(X_all.shape)

# ## Rotated features (unused)
# 

# In[12]:


# features_to_rot = [features, features_count]
# angles = [np.pi/4]

# def get_rotated(X_all):
#     list_features_rot = []
#     list_X_all_rot = [] 
#     for j ,angle in enumerate(angles):
#         list_rot_0 = []
#         list_rot_1 = []
#         feature_names_0 = []
#         feature_names_1 = []
#         c,s = np.cos(angle), np.sin(angle)
#         rot_mat = np.array([[c,-s],[s,c]])
#         for i in tqdm(range(len(features))):
#             vars_to_rot = [feat[i] for feat in features_to_rot]
#             arr = X_all[vars_to_rot].values
#             arr_rot = np.dot(arr, rot_mat)
#             list_rot_0.append(arr_rot[:,0])
#             list_rot_1.append(arr_rot[:,1])
#             feature_names_0.append('var_%d_angle_%d_rot_0' %(i,j))        
#             feature_names_1.append('var_%d_angle_%d_rot_1' %(i,j))        
#         arr_rot_0 = np.stack(list_rot_0).transpose()
#         arr_rot_1 = np.stack(list_rot_1).transpose()
#         list_features_rot.append(feature_names_0)
#         list_features_rot.append(feature_names_1)
#         X_all_rot_0 = pd.DataFrame(columns=feature_names_0, data = arr_rot_0)
#         X_all_rot_0.index = X_all.index
#         X_all_rot_1 = pd.DataFrame(columns=feature_names_1, data = arr_rot_1)
#         X_all_rot_1.index = X_all.index
#         list_X_all_rot.append(X_all_rot_0)
#         list_X_all_rot.append(X_all_rot_1)

#     X_all_rot = pd.concat(list_X_all_rot, axis=1)
#     X_all = pd.concat([X_all, X_all_rot], axis=1)
#     return X_all, feature_names_0, feature_names_1

# X_all, feature_names_rot_0, feature_names_rot_1 = get_rotated(X_all)

# print(X_all.shape)

# ## PCA (unused)

# In[6]:


# features_to_pca = [features, features_count]

# from sklearn.decomposition import PCA

# def get_pca(X_all):
#     list_X_all_pca = [] 
#     list_pca_0 = []
#     list_pca_1 = []
#     feature_names_0 = []
#     feature_names_1 = []
#     for i in tqdm(range(len(features))):
#         vars_to_pca = [feat[i] for feat in features_to_rot]
#         arr = X_all[vars_to_pca].values
#         pca = PCA(n_components = 2)
#         arr_pca = pca.fit_transform(arr)
#         list_pca_0.append(arr_pca[:,0])
#         list_pca_1.append(arr_pca[:,1])
#         feature_names_0.append('var_%d_pca_0' %i)        
#         feature_names_1.append('var_%d_pca_1' %i)        
#     arr_pca_0 = np.stack(list_pca_0).transpose()
#     arr_pca_1 = np.stack(list_pca_1).transpose()
#     X_all_pca_0 = pd.DataFrame(columns=feature_names_0, data = arr_pca_0)
#     X_all_pca_0.index = X_all.index
#     X_all_pca_1 = pd.DataFrame(columns=feature_names_1, data = arr_pca_1)
#     X_all_pca_1.index = X_all.index
#     list_X_all_pca.append(X_all_pca_0)
#     list_X_all_pca.append(X_all_pca_1)

#     X_all_pca = pd.concat(list_X_all_pca, axis=1)
#     X_all = pd.concat([X_all, X_all_pca], axis=1)
#     return X_all, feature_names_0, feature_names_1

# X_all, feature_names_pca_0, feature_names_pca_1 = get_pca(X_all)

# print(X_all.shape)

# ## Setting up Dataframes
# After performing FE on `X_all`, I split it back into train/test and delete the obsolete dataframe. The latter is a reoccuring theme in this kernel and was necessary as I often experienced memory overflow. This is also the reason why I wrote most of the code inside of functions. Shoutout to kaggle however for providing fast GPUs!

# In[7]:


X_train = X_all.iloc[:train_length,:]
X_test = X_all.iloc[train_length:,:]
del X_all
import gc
gc.collect()
print(X_train.shape, X_test.shape)

# # LGBM 
# Many public kernels indicated that the features are independent, conditional on the target. For this reason I train seperate trees for each feature and their respective counts. Using a simple average (of the square root) of all tree predictors achieves around 0.9225 / 0.9205 on public/private LB.

# In[ ]:


features_used = [features, features_count]

# ## Params
# Parameters of the LGBM model. I choose l1 regularization / max_bin / learning rate and num_leaves seaprately for each of the 200 var_x through earlier hyperparam search.

# In[9]:


params = {
    'boost_from_average':'false',
    'boost': 'gbdt',
    'feature_fraction': 1,
    'learning_rate': 0.08,
    'max_depth': -1,
    'metric':'binary_logloss',
    'num_leaves': 4,
    'num_threads': 8,
    'tree_learner': 'serial',
    'objective': 'binary',
    'reg_alpha': 2,
    'reg_lambda': 0,
    'verbosity': 1,
    'max_bin':256,
}

# reg_alpha
reg_alpha_values = [0.75, 1, 2, 3]
reg_alpha_var = [3, 0, 2, 3, 2, 0, 1, 1, 3, 2, 2, 0, 2, 0, 2, 2, 2, 1, 1, 2, 1, 2, 3, 3, 2, 1, 3, 1, 3, 2, 2, 3, 1, 1, 3, 2, 0, 1, 0, 2, 1, 1, 2, 3, 0, 3, 3, 3, 2, 0, 3, 1, 3, 1, 1, 0, 2, 2, 0, 0, 0, 1, 2, 1, 0, 1, 3, 2, 0, 2, 1, 2, 0, 0, 1, 3, 3, 1, 2, 3, 3, 2, 0, 1, 2, 3, 3, 2, 3, 3, 0, 0, 3, 0, 1, 0, 1, 0, 2, 3, 1, 0, 3, 1, 3, 2, 3, 1, 3, 3, 3, 1, 3, 2, 3, 2, 1, 0, 1, 2, 0, 3, 0, 3, 0, 3, 2, 1, 0, 0, 2, 2, 2, 0, 1, 0, 0, 2, 3, 2, 2, 1, 1, 0, 1, 2, 2, 2, 1, 0, 2, 3, 2, 3, 1, 1, 3, 1, 1, 2, 1, 2, 0, 3, 1, 3, 3, 2, 0, 1, 3, 3, 0, 1, 0, 3, 1, 3, 1, 3, 0, 3, 0, 3, 1, 0, 0, 0, 3, 0, 3, 0, 0, 2, 0, 3, 1, 0, 3, 2]

# max_bin
max_bin_values = [256, 512, 1024]
max_bin_var = [0, 0, 1, 0, 0, 0, 2, 0, 0, 2, 0, 2, 0, 0, 1, 1, 1, 0, 0, 0, 0, 2, 2, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 2, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 2, 1, 1, 1, 1, 0, 0, 0, 0, 1, 2, 1, 0, 0, 1, 2, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 2, 1, 0, 0, 0, 2, 0, 1, 1, 0, 1, 0, 0, 1, 2, 1, 2, 1, 0, 0, 1, 0, 2, 0, 1, 0, 0, 2, 1, 1, 1, 0, 0, 0, 2, 0, 0, 2, 1, 0, 0, 1, 0, 1, 2, 0, 0, 0, 0, 0, 2, 2, 2, 2, 1, 1, 2, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 2, 0, 1, 0, 1, 1, 0, 2, 1, 1, 1, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 2, 0, 1, 0, 1, 2, 0, 0, 0, 0, 2, 0, 0, 2, 0, 1, 1, 0, 2, 0, 0, 0, 1, 2, 0, 0, 1, 0, 2]

# learning_rate
learning_rate_values = [0.06, 0.08, 0.12]
learning_rate_var = [2, 2, 2, 1, 2, 2, 2, 0, 1, 2, 0, 2, 2, 2, 0, 2, 2, 0, 2, 1, 2, 2, 2, 2, 2, 0, 1, 0, 2, 0, 0, 2, 0, 2, 2, 2, 1, 2, 0, 0, 2, 0, 0, 1, 2, 1, 2, 0, 0, 2, 1, 2, 2, 2, 2, 0, 0, 2, 1, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 0, 2, 2, 2, 0, 2, 2, 1, 0, 1, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 1, 1, 1, 2, 0, 2, 0, 2, 0, 2, 1, 0, 0, 1, 2, 0, 2, 2, 2, 0, 2, 2, 2, 2, 1, 0, 2, 1, 2, 2, 1, 2, 0, 2, 0, 2, 2, 2, 2, 2, 2, 1, 2, 1, 0, 2, 1, 1, 2, 2, 2, 2, 0, 2, 1, 2, 1, 2, 2, 2, 2, 2, 2, 2, 1, 1, 0, 1, 2, 0, 2, 2, 0, 1, 2, 2, 2, 1, 0, 1, 2, 1, 2, 1, 1, 1, 2, 1, 2, 1, 0, 0, 2, 0, 1, 2, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1]

# num_leaves
num_leaves_values = [3, 4, 5]
num_leaves_var = [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 2, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 2, 2, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 2, 1, 2, 0, 0, 0, 0, 0, 1, 1, 0, 0, 2, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 2, 0, 0, 0, 0, 1, 0, 1, 2, 1, 1, 1, 0, 2, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 1, 2, 0, 1, 0, 2, 2, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 2, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 2, 0, 0, 0, 0, 1, 0, 0, 2, 1, 0, 1, 2, 1, 1, 0, 0, 0, 2, 1, 2]



# ## Training
# 
# A little discussion I had with Chua in this markdown cell. Thats probably not the most efficient way of communicating :D.
# 
# Chua:
# 
# One interesting finding is that any var with AUC lower than 0.504 does not help the model in any way. I checked the naive bayes kernel and they were the 'saw-like' probability curves. Went but to magic ecdf kernel and they were those that have ecdf that lined up a bit too perfectly. Considering that we are training features one by one. I think it may make sense to actually remove them all together.
# 
# Nawid:
# 
# I think your explanation is generally correct, however I would not exclude them already during training especially because you do it separately for every fold, I believe this leads to some sort of overfitting thus leading to differences in CV and LB, I also think that it is better practice to remove them **after** the trees are trained for example with a Lasso model. Moreover I believe that using AUC as validation metric for early stopping leads to overfitting CV. I therefore removed this part of the code and only train a single model. Similarly a very high number of early stopping rounds can also lead to overfitting.

# In[11]:


n_folds = 5
early_stopping_rounds=10
settings = [4]
np.random.seed(47)

settings_best_ind = []

def train_trees():
    preds_oof = np.zeros((len(X_train), len(features)))
    preds_test = np.zeros((len(X_test), len(features)))
    preds_train = np.zeros((len(X_train), len(features)))
    preds_fake = np.zeros((len(X_fake), len(features)))

    features_used_flatten = [var for sublist in features_used for var in sublist]
    X_train_used = X_train[features_used_flatten]
    X_test_used = X_test[features_used_flatten]
    X_fake_used = X_fake[features_used_flatten]

    for i in range(len(features)):
        params['max_bin'] = max_bin_values[max_bin_var[i]]
        params['learning_rate'] = learning_rate_values[learning_rate_var[i]]
        params['reg_alpha'] = reg_alpha_values[reg_alpha_var[i]]
        params['num_leaves'] = num_leaves_values[num_leaves_var[i]]
        features_train = [feature_set[i] for feature_set in features_used] 
        print(f'Training on: {features_train}')
        folds = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=np.random.randint(100000))
        list_folds = list(folds.split(X_train_used.values, target_train.values))
        preds_oof_temp = np.zeros((preds_oof.shape[0], len(settings)))
        preds_test_temp = np.zeros((preds_test.shape[0], len(settings)))
        preds_train_temp = np.zeros((preds_train.shape[0], len(settings)))
        preds_fake_temp = np.zeros((preds_fake.shape[0], len(settings)))

        scores = []
        for j, setting in enumerate(settings):
            # setting is used for hyperparameter tuning, here you can add sometinh like params['num_leaves'] = setting
            print('\nsetting: ', setting)
            for k, (trn_idx, val_idx) in enumerate(list_folds):
                print("Fold: {}".format(k+1), end="")
                trn_data = lgb.Dataset(X_train_used.iloc[trn_idx][features_train], label=target_train.iloc[trn_idx])
                val_data = lgb.Dataset(X_train_used.iloc[val_idx][features_train], label=target_train.iloc[val_idx])

                # Binary Log Loss
                clf = lgb.train(params, trn_data, 2000, valid_sets=[trn_data, val_data], verbose_eval=False, early_stopping_rounds=early_stopping_rounds) 

                prediction_val1 = clf.predict(X_train_used.iloc[val_idx][features_train])
                prediction_test1 = clf.predict(X_test_used[features_train])
                prediction_train1 = clf.predict(X_train_used.iloc[trn_idx][features_train])
                prediction_fake1 = clf.predict(X_fake_used[features_train])

                # Predictions
                s1 = roc_auc_score(target_train.iloc[val_idx], prediction_val1)
                s1_log = log_loss(target_train.iloc[val_idx], prediction_val1)
                print(' - val AUC: {:<8.4f} - loss: {:<8.3f}'.format(s1, s1_log*1000), end='')

                # Predictions Test
                if use_experimental:
                    s1_test = roc_auc_score(target_test, prediction_test1)
                    s1_log_test = log_loss(target_test, prediction_test1)
                    print(' - test AUC: {:<8.4f} - loss: {:<8.3f}'.format(s1_test, s1_log_test*1000), end='')

                # Predictions Train
                s1_train = roc_auc_score(target_train.iloc[trn_idx], prediction_train1)
                s1_log_train = log_loss(target_train.iloc[trn_idx], prediction_train1)
                print(' - train AUC: {:<8.4f} - loss: {:<8.3f}'.format(s1_train, s1_log_train*1000), end='')
                if use_experimental:
                    print('',clf.feature_importance(), end='')

                print('')


                preds_oof_temp[val_idx,j] += np.sqrt(prediction_val1 - prediction_val1.mean() + 0.1) 
                preds_test_temp[:,j] += np.sqrt(prediction_test1 - prediction_test1.mean() + 0.1) / n_folds
                preds_train_temp[trn_idx,j] += np.sqrt(prediction_train1 - prediction_train1.mean() + 0.1) / (n_folds-1)
                preds_fake_temp[:,j] += np.sqrt(prediction_fake1 - prediction_fake1.mean() + 0.1) / n_folds

            score_setting = roc_auc_score(target_train, preds_oof_temp[:,j])
            score_setting_log = 1000*log_loss(target_train, np.exp(preds_oof_temp[:,j]))
            scores.append(score_setting_log)
            print("Score:  - val AUC: {:<8.4f} - loss: {:<8.3f}".format(score_setting, score_setting_log), end='')
            if use_experimental:
                score_setting_test = roc_auc_score(target_test, preds_test_temp[:,j])
                score_setting_log_test = 1000*log_loss(target_test, np.exp(preds_test_temp[:,j]))  
                print(" - test AUC: {:<8.4f} - loss: {:<8.3f}".format(score_setting_test, score_setting_log_test), end='')

            score_setting_train = roc_auc_score(target_train, preds_train_temp[:,j])
            score_setting_log_train = 1000*log_loss(target_train, np.exp(preds_train_temp[:,j]))
            print(" - train AUC: {:<8.4f} - loss: {:<8.3f}".format(score_setting_train, score_setting_log_train))

        best_ind = np.argmin(scores)
        settings_best_ind.append(best_ind)
        preds_oof[:,i] = preds_oof_temp[:,best_ind]
        preds_test[:,i] = preds_test_temp[:,best_ind]
        preds_train[:,i] = preds_train_temp[:,best_ind]
        preds_fake[:,i] = preds_fake_temp[:,best_ind]


        print('\nbest setting: ', settings[best_ind])
        preds_oof_cum = preds_oof[:,:i+1].mean(axis=1)
        print("Cum CV val  : {:<8.4f} - loss: {:<8.3f}".format(roc_auc_score(target_train, preds_oof_cum), 1000*log_loss(target_train, np.exp(preds_oof_cum))))
        if use_experimental:        
            preds_test_cum = preds_test[:,:i+1].mean(axis=1)
            print("Cum CV test : {:<8.4f} - loss: {:<8.3f}".format(roc_auc_score(target_test, preds_test_cum), 1000*log_loss(target_test, np.exp(preds_test_cum))))
        preds_train_cum = preds_train[:,:i+1].mean(axis=1)
        print("Cum CV train: {:<8.4f} - loss: {:<8.3f}".format(roc_auc_score(target_train, preds_train_cum), 1000*log_loss(target_train, np.exp(preds_train_cum))))
        print('*****' * 10 + '\n')
        
    return preds_oof, preds_test, preds_train, preds_fake

preds_oof, preds_test, preds_train, preds_fake = train_trees()

# ## Training Summary

# In[ ]:


preds_oof_cum = np.zeros(preds_oof.shape[0])
if use_experimental:
    preds_test_cum = np.zeros(preds_test.shape[0])
preds_train_cum = np.zeros(preds_train.shape[0])
for i in range(len(features)):
    preds_oof_cum += preds_oof[:,i]
    preds_train_cum += preds_train[:,i]
    print("var_{} Cum val: {:<8.5f}".format(i,roc_auc_score(target_train, preds_oof_cum)), end="")
    if use_experimental:
        preds_test_cum += preds_test[:,i]
        print(" - test : {:<8.5f}".format(roc_auc_score(target_test, preds_test_cum)), end="")
    print(" - train: {:<8.5f}".format(roc_auc_score(target_train, preds_train_cum)))

# In[ ]:


print(settings)
print(settings_best_ind)

# ## EDA on predictors
# I plotted the predictions (sorted by feature), of the trees seaparately for the first 20 vars (x-axis corresponds to the z-score). The predictions are very noisy at the tails of the distributions, therefore I also tried using smoothed predictions (orange line) with no success however.

# In[15]:


from scipy.interpolate import interp1d
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt 

features_to_show = np.arange(20)
plt.figure(figsize = (20,20))

for i in features_to_show:
    var = 'var_'+str(i)
    signal = X_test[var].values
    logits = preds_test[:,i]
    func = interp1d(signal, logits)
    space = np.linspace(signal.min(), signal.max(), 4000)
    activations = func(space)
    activations_smooth = gaussian_filter(activations, 10)
    
    func_smooth = interp1d(space, activations_smooth)
    logits_smooth = func_smooth(signal)
    plt.subplot(5,4,i+1)
    plt.plot(space, activations)
    plt.plot(space, activations_smooth)

# # CNN 
# The CNN model is the main reason for our high placement as we hit a wall using solely trees and couldn't improve upon 0.922 LB. At some point while playing around with the trees I noticed two things:
# 1. Averaging the predictors might not be the best solution. At first I just averaged the predictors, then I observed that averaging the logits of predictors improved the final score massively, however i couldn't really explain why. Averaging logits here is equivalent with multiplying the probabilities p(t|x) which is not what happens in Naive Bayes, there you multiply p(x|t). I concluded that the boost comes from the concativity of the logarithm, which kind of translates to the predictors having lower confidence predicting t=1 and higher confidence predicting t=0, I further investigated concave functions and found the square root to work even better. 
# 
# 2. The predictors are very noisy around the tails of the feature distributions and this pattern is likely related to the counts or the features themselves. Also the pattern seems to be similar across different features. 
# 
# Given these observations I tried using a CNN to learn these patterns. The original idea here was that it should learn when to trust the tree predictors given the feature and its counts. After reading some other top team solutions I think the CNN maybe picked up something else from the features.
# 
# I choose the architecure in a way which would ensure feature independence up until the last dense layer. In order to minimize overfitting and utilize the similarity of patterns across different var_x I used convolutional layers. The convolutions are performed across different var_x and at any point the filters only have a single var and their respective features in their field of view. Batch normalization is a great regularizer here and very crucial for the success of the model. The model has a total of 2.8K trainable parameters which is sufficiently low to prevent overfitting. I verified this by splitting train data into train / test with `use_experimental = True` at the top of the kernel and using test AUC as a gauge. The final prediction is the average of the 7 CNNs trained on every fold.

# ## Training

# In[ ]:


import keras

n_splits = 7
num_preds = 5
epochs = 60
learning_rate_init = 0.02
batch_size = 4000

num_features = len(features)

def get_features(preds, df):
    list_features = [preds, df[features].values, df[features_count].values, df[features_deviation], df[features_density]]
    list_indices = []
    for i in range(num_features):
        indices = np.arange(num_preds)*num_features + i
        list_indices.append(indices)
    indices = np.concatenate(list_indices)
    feats = np.concatenate(list_features, axis=1)[:,indices]
    return feats 

def get_model_3():
    inp = keras.layers.Input((num_features*num_preds,))
    x = keras.layers.Reshape((num_features*num_preds,1))(inp)
    x = keras.layers.Conv1D(32,num_preds,strides=num_preds, activation='elu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv1D(24,1, activation='elu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv1D(16,1, activation='elu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv1D(4,1, activation='elu')(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Reshape((num_features*4,1))(x)
    x = keras.layers.AveragePooling1D(2)(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.BatchNormalization()(x)
    out = keras.layers.Dense(1, activation='sigmoid')(x)
    return keras.Model(inputs=inp, outputs=out)


def lr_scheduler(epoch):
    if epoch <= epochs*0.8:
        return learning_rate_init
    else:
        return learning_rate_init * 0.1

def train_NN(features_oof, features_test, features_train, features_fake):
    
    folds = StratifiedKFold(n_splits=n_splits)

    preds_nn_oof = np.zeros(features_oof.shape[0])
    preds_nn_test = np.zeros(features_test.shape[0])
    preds_nn_fake = np.zeros(features_fake.shape[0])

    for trn_idx, val_idx in folds.split(features_oof, target_train):
        features_oof_tr = features_oof[trn_idx, :]
        target_oof_tr = target_train.values[trn_idx]
        features_oof_val = features_oof[val_idx, :]
        target_oof_val = target_train.values[val_idx]

        optimizer = keras.optimizers.Adam(lr = learning_rate_init, decay = 0.00001)
        model = get_model_3()
        callbacks = []
        callbacks.append(keras.callbacks.LearningRateScheduler(lr_scheduler))
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(features_oof_tr, target_oof_tr, validation_data=(features_oof_val, target_oof_val), epochs=epochs, verbose=2, batch_size=batch_size, callbacks=callbacks)

        preds_nn_oof += model.predict(features_oof, batch_size=2000)[:,0]
        preds_nn_test += model.predict(features_test, batch_size=2000)[:,0]
        preds_nn_fake += model.predict(features_fake, batch_size=2000)[:,0]

        print(roc_auc_score(target_train, preds_nn_oof))
        if use_experimental:
            print(roc_auc_score(target_test, preds_nn_test))
            print(roc_auc_score(target_test, preds_test.mean(axis=1)))

    preds_nn_oof /= n_splits
    preds_nn_test /= n_splits
    preds_nn_fake /= n_splits
    return preds_nn_oof, preds_nn_test, preds_nn_fake


features_oof = get_features(preds_oof, X_train)
features_test = get_features(preds_test, X_test)
if not use_experimental:
    del X_test
features_train = get_features(preds_train, X_train)
if not use_experimental:
    del X_train
features_fake = get_features(preds_fake, X_fake)
if not use_experimental:
    del X_fake
    del preds_oof
    del preds_fake
    del preds_train
    del preds_test

print(get_model_3().summary())
    
preds_nn_oof, preds_nn_test, preds_nn_fake = train_NN(features_oof, features_test, features_train, features_fake)

print(roc_auc_score(target_train, preds_nn_oof))
if use_experimental:
    print('test AUC: ', roc_auc_score(target_test, preds_nn_test))

# ## Generating submission

# In[ ]:


preds_oof_final = preds_nn_oof
preds_test_final = preds_nn_test
preds_fake_final = preds_nn_fake

print('oof  : ', roc_auc_score(target_train, preds_oof_final))
if use_experimental:
    print('test : ', roc_auc_score(target_test, preds_test_final))
    print('train: ', roc_auc_score(target_fake, preds_fake_final))

if not use_experimental:
    sub = pd.DataFrame({"ID_code": test_df.ID_code.values})
    predictions_all = np.zeros(test_df.shape[0])
    predictions_all[indices_real] = preds_test_final
    predictions_all[indices_fake] = preds_fake_final
    sub["target"] = predictions_all
    sub.to_csv("submission.csv", index=False)
    print(sub.head(20))
