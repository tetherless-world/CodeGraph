#!/usr/bin/env python
# coding: utf-8

# # Basic Data Augmentation & Feature Reduction
# ---
# A common problem with small datasets is that models will identify patterns from random variance in the data rather than statistically reliable trends. This problem is exacerbated when we have a large number of features which can interact with eachother. Without oversampling, there are approximately 4194 distinct rows that can be generated with the LANL Earthquake Prediction dataset. How can we increase this number while avoiding oversampling and the consequent risk of leakage?
# 
# This kernel presents a rudimentary approach to data augmentation, so that new data can be generated which will diminish the effect of spurious feature interactions. We can then use this to identify the features in our dataset that are most vulnerable to this problem, along with the features that are more statistically robust.
# 
# The augmented data will use a proportion of features from the original data and retain their values. The remaining features will be values that have been sampled at random from each feature's overall distribution. 

# In[ ]:


import numpy as np
import pandas as pd
import numpy.random
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm_notebook as tqdm
import matplotlib.pyplot as plt
import seaborn as sns
train = pd.read_csv('../input/eq-100-features/train_100.csv')
test = pd.read_csv('../input/eq-100-features/test_100.csv')
y = pd.read_csv('../input/eq-100-features/y.csv').values.flatten()

# First let's quickly establish a baseline CV score using the function from [one of my other kernels](https://www.kaggle.com/bigironsphere/parameter-tuning-in-one-function-with-hyperopt), `quick_hyperopt()`.

# In[ ]:


#import required packages
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
import gc
from hyperopt import hp, tpe, Trials, STATUS_OK
from hyperopt.fmin import fmin
from hyperopt.pyll.stochastic import sample
#optional but advised
import warnings
warnings.filterwarnings('ignore')

#GLOBAL HYPEROPT PARAMETERS
NUM_EVALS = 1000 #number of hyperopt evaluation rounds
N_FOLDS = 5 #number of cross-validation folds on data in each evaluation round

#LIGHTGBM PARAMETERS
LGBM_MAX_LEAVES = 2**10 #maximum number of leaves per tree for LightGBM
LGBM_MAX_DEPTH = 25 #maximum tree depth for LightGBM
EVAL_METRIC_LGBM_REG = 'mae' #LightGBM regression metric. Note that 'rmse' is more commonly used 
EVAL_METRIC_LGBM_CLASS = 'auc'#LightGBM classification metric

#XGBOOST PARAMETERS
XGB_MAX_LEAVES = 2**12 #maximum number of leaves when using histogram splitting
XGB_MAX_DEPTH = 25 #maximum tree depth for XGBoost
EVAL_METRIC_XGB_REG = 'mae' #XGBoost regression metric
EVAL_METRIC_XGB_CLASS = 'auc' #XGBoost classification metric

#CATBOOST PARAMETERS
CB_MAX_DEPTH = 8 #maximum tree depth in CatBoost
OBJECTIVE_CB_REG = 'MAE' #CatBoost regression metric
OBJECTIVE_CB_CLASS = 'Logloss' #CatBoost classification metric

#OPTIONAL OUTPUT
BEST_SCORE = 0

def quick_hyperopt(data, labels, package='lgbm', num_evals=NUM_EVALS, diagnostic=False, Class=False):
    
    #==========
    #LightGBM
    #==========
    
    if package=='lgbm':
        
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
                cv_results = lgb.cv(space_params, train, nfold = N_FOLDS, stratified=True,
                                    early_stopping_rounds=100, metrics=EVAL_METRIC_LGBM_CLASS, seed=42)
                best_loss = 1 - cv_results['auc-mean'][-1]
                
            else:
                cv_results = lgb.cv(space_params, train, nfold = N_FOLDS, stratified=False,
                                    early_stopping_rounds=100, metrics=EVAL_METRIC_LGBM_REG, seed=42)
                best_loss = cv_results['l1-mean'][-1] #'l2-mean' for rmse
            
            return{'loss':best_loss, 'status': STATUS_OK }
        
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
        
        #optional: activate GPU for LightGBM
        #follow compilation steps here:
        #https://www.kaggle.com/vinhnguyen/gpu-acceleration-for-lightgbm/
        #then uncomment lines below:
        #space['device'] = 'gpu'
        #space['gpu_platform_id'] = 0,
        #space['gpu_device_id'] =  0

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
        if diagnostic:
            return(best, trials)
        else:
            return(best)
    
    #==========
    #XGBoost
    #==========
    
    if package=='xgb':
        
        print('Running {} rounds of XGBoost parameter optimisation:'.format(num_evals))
        #clear space
        gc.collect()
        
        integer_params = ['max_depth']
        
        def objective(space_params):
            
            for param in integer_params:
                space_params[param] = int(space_params[param])
                
            #extract multiple nested tree_method conditional parameters
            #libera te tutemet ex inferis
            if space_params['tree_method']['tree_method'] == 'hist':
                max_bin = space_params['tree_method'].get('max_bin')
                space_params['max_bin'] = int(max_bin)
                if space_params['tree_method']['grow_policy']['grow_policy']['grow_policy'] == 'depthwise':
                    grow_policy = space_params['tree_method'].get('grow_policy').get('grow_policy').get('grow_policy')
                    space_params['grow_policy'] = grow_policy
                    space_params['tree_method'] = 'hist'
                else:
                    max_leaves = space_params['tree_method']['grow_policy']['grow_policy'].get('max_leaves')
                    space_params['grow_policy'] = 'lossguide'
                    space_params['max_leaves'] = int(max_leaves)
                    space_params['tree_method'] = 'hist'
            else:
                space_params['tree_method'] = space_params['tree_method'].get('tree_method')
                
            #for classification replace EVAL_METRIC_XGB_REG with EVAL_METRIC_XGB_CLASS
            cv_results = xgb.cv(space_params, train, nfold=N_FOLDS, metrics=[EVAL_METRIC_XGB_REG],
                             early_stopping_rounds=100, stratified=False, seed=42)
            
            best_loss = cv_results['test-mae-mean'].iloc[-1] #or 'test-rmse-mean' if using RMSE
            #for classification, comment out the line above and uncomment the line below:
            #best_loss = 1 - cv_results['test-auc-mean'].iloc[-1]
            #if necessary, replace 'test-auc-mean' with 'test-[your-preferred-metric]-mean'
            return{'loss':best_loss, 'status': STATUS_OK }
        
        train = xgb.DMatrix(data, labels)
        
        #integer and string parameters, used with hp.choice()
        boosting_list = ['gbtree', 'gblinear'] #if including 'dart', make sure to set 'n_estimators'
        metric_list = ['MAE', 'RMSE'] 
        #for classification comment out the line above and uncomment the line below
        #metric_list = ['auc']
        #modify as required for other classification metrics classification
        
        tree_method = [{'tree_method' : 'exact'},
               {'tree_method' : 'approx'},
               {'tree_method' : 'hist',
                'max_bin': hp.quniform('max_bin', 2**3, 2**7, 1),
                'grow_policy' : {'grow_policy': {'grow_policy':'depthwise'},
                                'grow_policy' : {'grow_policy':'lossguide',
                                                  'max_leaves': hp.quniform('max_leaves', 32, XGB_MAX_LEAVES, 1)}}}]
        
        #if using GPU, replace 'exact' with 'gpu_exact' and 'hist' with
        #'gpu_hist' in the nested dictionary above
        
        objective_list_reg = ['reg:linear', 'reg:gamma', 'reg:tweedie']
        objective_list_class = ['reg:logistic', 'binary:logistic']
        #for classification change line below to 'objective_list = objective_list_class'
        objective_list = objective_list_reg
        
        space ={'boosting' : hp.choice('boosting', boosting_list),
                'tree_method' : hp.choice('tree_method', tree_method),
                'max_depth': hp.quniform('max_depth', 2, XGB_MAX_DEPTH, 1),
                'reg_alpha' : hp.uniform('reg_alpha', 0, 5),
                'reg_lambda' : hp.uniform('reg_lambda', 0, 5),
                'min_child_weight' : hp.uniform('min_child_weight', 0, 5),
                'gamma' : hp.uniform('gamma', 0, 5),
                'learning_rate' : hp.loguniform('learning_rate', np.log(0.005), np.log(0.2)),
                'eval_metric' : hp.choice('eval_metric', metric_list),
                'objective' : hp.choice('objective', objective_list),
                'colsample_bytree' : hp.quniform('colsample_bytree', 0.1, 1, 0.01),
                'colsample_bynode' : hp.quniform('colsample_bynode', 0.1, 1, 0.01),
                'colsample_bylevel' : hp.quniform('colsample_bylevel', 0.1, 1, 0.01),
                'subsample' : hp.quniform('subsample', 0.5, 1, 0.05),
                'nthread' : -1
            }
        
        trials = Trials()
        best = fmin(fn=objective,
                    space=space,
                    algo=tpe.suggest,
                    max_evals=num_evals, 
                    trials=trials)
        
        best['tree_method'] = tree_method[best['tree_method']]['tree_method']
        best['boosting'] = boosting_list[best['boosting']]
        best['eval_metric'] = metric_list[best['eval_metric']]
        best['objective'] = objective_list[best['objective']]
        
        #cast floats of integer params to int
        for param in integer_params:
            best[param] = int(best[param])
        if 'max_leaves' in best:
            best['max_leaves'] = int(best['max_leaves'])
        if 'max_bin' in best:
            best['max_bin'] = int(best['max_bin'])
        
        print('{' + '\n'.join('{}: {}'.format(k, v) for k, v in best.items()) + '}')
        
        if diagnostic:
            return(best, trials)
        else:
            return(best)
    
    #==========
    #CatBoost
    #==========
    
    if package=='cb':
        
        print('Running {} rounds of CatBoost parameter optimisation:'.format(num_evals))
        
        #clear memory 
        gc.collect()
            
        integer_params = ['depth',
                          #'one_hot_max_size', #for categorical data
                          'min_data_in_leaf',
                          'max_bin']
        
        def objective(space_params):
                        
            #cast integer params from float to int
            for param in integer_params:
                space_params[param] = int(space_params[param])
                
            #extract nested conditional parameters
            if space_params['bootstrap_type']['bootstrap_type'] == 'Bayesian':
                bagging_temp = space_params['bootstrap_type'].get('bagging_temperature')
                space_params['bagging_temperature'] = bagging_temp
                
            if space_params['grow_policy']['grow_policy'] == 'LossGuide':
                max_leaves = space_params['grow_policy'].get('max_leaves')
                space_params['max_leaves'] = int(max_leaves)
                
            space_params['bootstrap_type'] = space_params['bootstrap_type']['bootstrap_type']
            space_params['grow_policy'] = space_params['grow_policy']['grow_policy']
                           
            #random_strength cannot be < 0
            space_params['random_strength'] = max(space_params['random_strength'], 0)
            #fold_len_multiplier cannot be < 1
            space_params['fold_len_multiplier'] = max(space_params['fold_len_multiplier'], 1)
                       
            #for classification set stratified=True
            cv_results = cb.cv(train, space_params, fold_count=N_FOLDS, 
                             early_stopping_rounds=25, stratified=False, partition_random_seed=42)
           
            best_loss = cv_results['test-MAE-mean'].iloc[-1] #'test-RMSE-mean' for RMSE
            #for classification, comment out the line above and uncomment the line below:
            #best_loss = cv_results['test-Logloss-mean'].iloc[-1]
            #if necessary, replace 'test-Logloss-mean' with 'test-[your-preferred-metric]-mean'
            
            return{'loss':best_loss, 'status': STATUS_OK}
        
        train = cb.Pool(data, labels.astype('float32'))
        
        #integer and string parameters, used with hp.choice()
        bootstrap_type = [{'bootstrap_type':'Poisson'}, 
                           {'bootstrap_type':'Bayesian',
                            'bagging_temperature' : hp.loguniform('bagging_temperature', np.log(1), np.log(50))},
                          {'bootstrap_type':'Bernoulli'}] 
        LEB = ['No', 'AnyImprovement', 'Armijo'] #remove 'Armijo' if not using GPU
        #score_function = ['Correlation', 'L2', 'NewtonCorrelation', 'NewtonL2']
        grow_policy = [{'grow_policy':'SymmetricTree'},
                       {'grow_policy':'Depthwise'},
                       {'grow_policy':'Lossguide',
                        'max_leaves': hp.quniform('max_leaves', 2, 32, 1)}]
        eval_metric_list_reg = ['MAE', 'RMSE', 'Poisson']
        eval_metric_list_class = ['Logloss', 'AUC', 'F1']
        #for classification change line below to 'eval_metric_list = eval_metric_list_class'
        eval_metric_list = eval_metric_list_reg
                
        space ={'depth': hp.quniform('depth', 2, CB_MAX_DEPTH, 1),
                'max_bin' : hp.quniform('max_bin', 1, 32, 1), #if using CPU just set this to 254
                'l2_leaf_reg' : hp.uniform('l2_leaf_reg', 0, 5),
                'min_data_in_leaf' : hp.quniform('min_data_in_leaf', 1, 50, 1),
                'random_strength' : hp.loguniform('random_strength', np.log(0.005), np.log(5)),
                #'one_hot_max_size' : hp.quniform('one_hot_max_size', 2, 16, 1), #uncomment if using categorical features
                'bootstrap_type' : hp.choice('bootstrap_type', bootstrap_type),
                'learning_rate' : hp.uniform('learning_rate', 0.05, 0.25),
                'eval_metric' : hp.choice('eval_metric', eval_metric_list),
                'objective' : OBJECTIVE_CB_REG,
                #'score_function' : hp.choice('score_function', score_function), #crashes kernel - reason unknown
                'leaf_estimation_backtracking' : hp.choice('leaf_estimation_backtracking', LEB),
                'grow_policy': hp.choice('grow_policy', grow_policy),
                #'colsample_bylevel' : hp.quniform('colsample_bylevel', 0.1, 1, 0.01),# CPU only
                'fold_len_multiplier' : hp.loguniform('fold_len_multiplier', np.log(1.01), np.log(2.5)),
                'od_type' : 'Iter',
                'od_wait' : 25,
                'task_type' : 'GPU',
                'verbose' : 0
            }
        
        #optional: run CatBoost without GPU
        #uncomment line below
        #space['task_type'] = 'CPU'
            
        trials = Trials()
        best = fmin(fn=objective,
                    space=space,
                    algo=tpe.suggest,
                    max_evals=num_evals, 
                    trials=trials)
        
        #unpack nested dicts first
        best['bootstrap_type'] = bootstrap_type[best['bootstrap_type']]['bootstrap_type']
        best['grow_policy'] = grow_policy[best['grow_policy']]['grow_policy']
        best['eval_metric'] = eval_metric_list[best['eval_metric']]
        
        #best['score_function'] = score_function[best['score_function']] 
        #best['leaf_estimation_method'] = LEM[best['leaf_estimation_method']] #CPU only
        best['leaf_estimation_backtracking'] = LEB[best['leaf_estimation_backtracking']]        
        
        #cast floats of integer params to int
        for param in integer_params:
            best[param] = int(best[param])
        if 'max_leaves' in best:
            best['max_leaves'] = int(best['max_leaves'])
        
        print('{' + '\n'.join('{}: {}'.format(k, v) for k, v in best.items()) + '}')
        
        if diagnostic:
            return(best, trials)
        else:
            return(best)
    
    else:
        print('Package not recognised. Please use "lgbm" for LightGBM, "xgb" for XGBoost or "cb" for CatBoost.')              

# In[ ]:


lgbm_params = quick_hyperopt(train, y, 'lgbm', 150)

# With some parameter tuning these features can obtain a CV score of just under 2. Now we can use another quick function to isolate the feature split and gain importance scores from a shuffled KFold. 

# In[ ]:


def quick_kfold_imp(X, y, test=None, params=None, n_fold=5, random_state=42):
    
    MAE = 0
    folds = KFold(n_splits=n_fold, shuffle=True, random_state=random_state)
    test_preds = np.zeros(len(test))
    #obtain both split and gain importance
    imp_s = np.zeros(X.shape[1])
    imp_g = np.zeros(X.shape[1])

    if type(y) is not np.ndarray:
        y = y.values.flatten()
        
    for train_idx, valid_idx in folds.split(y):
         
        X_train, X_valid = X.iloc[train_idx, :], X.iloc[valid_idx, :]
        y_train, y_valid = y[train_idx], y[valid_idx]
        
        model = lgb.LGBMRegressor(**params, n_estimators = 50000, n_jobs = -1, eval_metric='mae', importance_type='split')
        model.fit(X_train, y_train,eval_set=[(X_train, y_train), (X_valid, y_valid)],
                  verbose=0, early_stopping_rounds=200)
        val_preds = model.predict(X_valid)
        imp_s += model.feature_importances_/n_fold
        
        model = lgb.LGBMRegressor(**params, n_estimators = 50000, n_jobs = -1, eval_metric='mae', importance_type='gain')
        model.fit(X_train, y_train,eval_set=[(X_train, y_train), (X_valid, y_valid)],
                  verbose=0, early_stopping_rounds=200)
        val_preds = model.predict(X_valid)
        MAE += mean_absolute_error(y_valid, val_preds)/n_fold
        imp_g += model.feature_importances_/n_fold
        test_preds += model.predict(test)/n_fold
                
    print('OOF MAE: {}'.format(MAE))
    
    return(imp_s, imp_g, test_preds)

# We'll run it in a loop with different seeds to get a more accurate picture of the feature importance.

# In[ ]:


train_features = train.columns
imp_split = np.zeros(train.shape[1])
imp_gain = np.zeros(train.shape[1])
test_preds = np.zeros(len(test))

N=25
for i in range(0, N):
    imp_s, imp_g, preds = quick_kfold_imp(train, y, test, lgbm_params, random_state=i)
    imp_split += imp_s/N
    imp_gain += imp_g/N
    test_preds += preds/N

initial_imp = pd.DataFrame({'feature':train_features,
                          'importance_split':imp_split,
                          'importance_gain':imp_gain,
                          'importance_score':np.log((imp_split*imp_gain))})

initial_imp.sort_values('importance_score', ascending=False, inplace=True)
initial_imp.head(10)

# In[ ]:


initial_imp.head(10)

# The importance score was calculated as the natural log of the gain score multiplied by the split score.

# In[ ]:


plot_height = int(np.floor(len(initial_imp)/5))
plt.figure(figsize=(12, plot_height));
sns.barplot(x='importance_score', y='feature', data=initial_imp);
plt.title('Original Feature Scores');
plt.show()

# Now we have our baseline feature scores and predictions for the original data.

# In[ ]:


sub_1 = pd.read_csv('../input/LANL-Earthquake-Prediction/sample_submission.csv')
sub_1['time_to_failure'] = test_preds
sub_1.to_csv('sub_orginal_data.csv', index=False)
sub_1['time_to_failure'].hist()

# # Generate Augmented Data
# ---
# Now that we have established a baseline, we can produce our augmented dataset. In this version I will be substituting 50% of the features in each row with randomly sampled values from that feature's actual distribution.

# In[ ]:


a = np.arange(0, train.shape[1])
#initialise aug dataframe - remember to set dtype!
train_aug = pd.DataFrame(index=train.index, columns=train.columns, dtype='float64')

for i in tqdm(range(0, len(train))):
    #ratio of features to be randomly sampled
    AUG_FEATURE_RATIO = 0.5
    #to integer count
    AUG_FEATURE_COUNT = np.floor(train.shape[1]*AUG_FEATURE_RATIO).astype('int16')
    
    #randomly sample half of columns that will contain random values
    aug_feature_index = np.random.choice(train.shape[1], AUG_FEATURE_COUNT, replace=False)
    aug_feature_index.sort()
    
    #obtain indices for features not in aug_feature_index
    feature_index = np.where(np.logical_not(np.in1d(a, aug_feature_index)))[0]
        
    #first insert real values for features in feature_index
    train_aug.iloc[i, feature_index] = train.iloc[i, feature_index]
              
    #random row index to randomly sampled values for each features
    rand_row_index = np.random.choice(len(train), len(aug_feature_index), replace=True)
        
    #for each feature being randomly sampled, extract value from random row in train
    for n, j in enumerate(aug_feature_index):
        train_aug.iloc[i, j] = train.iloc[rand_row_index[n], j]

# Please note that `pandas` will set the datatype of its columns as `'object'` unless you specify otherwise.  I mention this because the above code, which takes less than 1 minute to process 4194 rows of 100 features, will take around an hour if `dtype` isn't set to `'float64'`!
# 
# Comparing the first few rows of the regular data and the augmented data:

# In[ ]:


train_aug.head(3)

# In[ ]:


train.head(3)

# We can see, for a basic sanity check, that for each row in `train_aug` that half the values are the same as in `train`, and the remaining half are seemingly random. The random values for each feature were sampled from that feature's original distribution, and the overall distributions for each variable in `train` shouldn't be very different as a result. We can do another quick sanity check for this:

# In[ ]:


train['var_0'].hist(bins=25)

# In[ ]:


train_aug['var_0'].hist(bins=25)

# The distributions look almost identical and now we can examine the MAE for the augmented data, along with the feature importances.
# 
# # Augmented Data Evaluation
# ---
# 
# The corresponding y-values for the augmented rows will be the same for the original data. We can run `quick_hyperopt()` again to get an idea of the optimal CV score.

# In[ ]:


train_all = pd.concat([train, train_aug])
y_all = np.append(y, y)

print('Original train data shape: {}'.format(train.shape))
print('Augmented train data shape: {}'.format(train_all.shape))

params_all = quick_hyperopt(train_all, y_all, 'lgbm', 150)

# The combined data has a higher CV, indicating that LightGBM has been less eager to identify predictive feature interactions. Now we can examine the feature importances, and how they have changed relative to the original data. We'll run the augmentation process within a loop to ensure an even distribution of random features.

# In[ ]:


imp_split_all = np.zeros(train_aug.shape[1])
imp_gain_all = np.zeros(train_aug.shape[1])
test_preds_aug = np.zeros(len(test)) 

N = 25
for i in tqdm(range(0, N)):
    
    a = np.arange(0, train.shape[1])
    np.random.seed(i)
    train_aug = pd.DataFrame(index=train.index, columns=train.columns, dtype='float64')

    for i in range(0, len(train)):
        #ratio of features to be randomly sampled
        AUG_FEATURE_RATIO = 0.5
        #to integer count
        AUG_FEATURE_COUNT = np.floor(train.shape[1]*AUG_FEATURE_RATIO).astype('int16')
    
        #randomly sample half of columns that will contain random values
        aug_feature_index = np.random.choice(train.shape[1], AUG_FEATURE_COUNT, replace=False)
        aug_feature_index.sort()
    
        #obtain indices for features not in aug_feature_index
        feature_index = np.where(np.logical_not(np.in1d(a, aug_feature_index)))[0]
        
        #first insert real values for features in feature_index
        train_aug.iloc[i, feature_index] = train.iloc[i, feature_index]
              
        #random row index to randomly sampled values for each features
        rand_row_index = np.random.choice(len(train), len(aug_feature_index), replace=True)
        
        #for each feature being randomly sampled, extract value from random row in train
        for n, j in enumerate(aug_feature_index):
            train_aug.iloc[i, j] = train.iloc[rand_row_index[n], j]
    
    
    train_all = pd.concat([train, train_aug])
        
    imp_s, imp_g, preds = quick_kfold_imp(train_all, y_all, test, params=params_all, random_state=i)
    imp_split_all += imp_s/N
    imp_gain_all += imp_g/N
    test_preds_aug += preds/N


# In[ ]:


aug_imp = pd.DataFrame({'feature':train_features,
                          'importance_split':imp_split_all,
                          'importance_gain':imp_gain_all,
                          'importance_score':np.log((imp_split_all*imp_gain_all))})

aug_imp.sort_values('importance_score', ascending=False, inplace=True)
aug_imp.head(10)

# From inspection, the most important features appear to have changed. 

# In[ ]:


plot_height = int(np.floor(len(aug_imp)/5))
plt.figure(figsize=(12, plot_height));
sns.barplot(x='importance_score', y='feature', data=aug_imp);
plt.title('Augmented Data Feature Scores');
plt.show()

# We can then prepare a submission made with the augmented data and its different feature importances for comparison.

# In[ ]:


sub_aug = pd.read_csv('../input/LANL-Earthquake-Prediction/sample_submission.csv')
sub_aug['time_to_failure'] = test_preds_aug
sub_aug.to_csv('sub_aug_data.csv', index=False)
sub_aug['time_to_failure'].hist()

# # Feature Importance Change
# ---
# 
# So what is the relative change in feature importance for each feature? LightGBM can be inconsistent with feature importance scores so they'll have to be scaled.

# In[ ]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
initial_imp.iloc[:, 1:] = scaler.fit_transform(initial_imp.iloc[:, 1:])

scaler = MinMaxScaler()
aug_imp.iloc[:, 1:] = scaler.fit_transform(aug_imp.iloc[:, 1:])

# In[ ]:


aug_cols = ['feature'] + [x + '_all' for x in aug_imp.columns if x != 'feature']
aug_imp.columns = aug_cols
feature_df = initial_imp.merge(aug_imp, on='feature', how='inner')
feature_df['score_change'] = feature_df['importance_score_all'] - feature_df['importance_score']
feature_df.sort_values('score_change', ascending=False, inplace=True)
feature_df.head(10)

# In[ ]:


plot_height = int(np.floor(len(feature_df)/5))
plt.figure(figsize=(12, plot_height));
sns.barplot(x='score_change', y='feature', data=feature_df);
plt.title('Difference in Feature Scores');
plt.show()

# This has clearly identified some features that became *less* valuable when more randomness is introduced into the dataset.  This indicates that these particular features may have been attributed weight by the model due to random variance instead of a genuine relationship with the target. As a last experiment, we can isolate the features whose change in feature importance was in the bottom two quintiles and remove them. Their predictions can then be evaluated on the test set.

# In[ ]:


MIN_SCORE = np.percentile(feature_df.score_change.values.flatten(), 20)
features = feature_df.loc[feature_df.score_change >= MIN_SCORE, :].feature.values.flatten()
train = train[features]
test = test[features]

params_final = quick_hyperopt(train, y, 'lgbm', 150)

test_preds_final = np.zeros(len(test)) 

N = 30
for i in tqdm(range(0, N)):
    
    a = np.arange(0, train.shape[1])
    np.random.seed(i)
    train_aug = pd.DataFrame(index=train.index, columns=train.columns, dtype='float64')

    for i in range(0, len(train)):
        #ratio of features to be randomly sampled
        AUG_FEATURE_RATIO = 0.5
        #to integer count
        AUG_FEATURE_COUNT = np.floor(train.shape[1]*AUG_FEATURE_RATIO).astype('int16')
    
        #randomly sample half of columns that will contain random values
        aug_feature_index = np.random.choice(train.shape[1], AUG_FEATURE_COUNT, replace=False)
        aug_feature_index.sort()
    
        #obtain indices for features not in aug_feature_index
        feature_index = np.where(np.logical_not(np.in1d(a, aug_feature_index)))[0]
        
        #first insert real values for features in feature_index
        train_aug.iloc[i, feature_index] = train.iloc[i, feature_index]
              
        #random row index to randomly sampled values for each features
        rand_row_index = np.random.choice(len(train), len(aug_feature_index), replace=True)
        
        #for each feature being randomly sampled, extract value from random row in train
        for n, j in enumerate(aug_feature_index):
            train_aug.iloc[i, j] = train.iloc[rand_row_index[n], j]
    
    
    train_all = pd.concat([train, train_aug])
        
    _,  _, preds = quick_kfold_imp(train_all, y_all, test, params=params_final, random_state=i)
    test_preds_final += preds/N

sub_final = pd.read_csv('../input/LANL-Earthquake-Prediction/sample_submission.csv')
sub_final['time_to_failure'] = test_preds_final
sub_final.to_csv('sub_quintile_features_removed.csv', index=False)
sub_final['time_to_failure'].hist()

# # Data Score Evaluation
# ---
# 
# The original features were a random selectin of features I had generated. The augmented data used all these features, but with the augmentation process detailed above to double the number of rows. The 'refined' data was the original data, minus bottom quartile of the features isolated in the Feature Importance Change section above. The three submissions scored:
# 
# * **original features** Hyperopt CV: 1.987 LB: 1.454
# * **augmented data** Hyperopt CV: 2.012 LB 1.445
# * **refined data** Hyperopt CV: 2.005 **LB 1.435**
# 
# 
# Data augmentation, and feature selection via data augmentation can clearly yield positive results. This is a basic introduction to data augmentation and doubtless you can improve on the methods in this kernel. Hopefully you will find a way to make your models more accurate on the test set.
