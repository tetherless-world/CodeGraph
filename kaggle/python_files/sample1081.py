#!/usr/bin/env python
# coding: utf-8

# # Hyperopt Made Simple!
# ## Automated Parameter Tuning in One Function: quick_hyperopt()
# ---
# ![](https://i.imgur.com/JpaUi5T.png)
# 
# Parameter tuning can be a chore. Which parameters should be changed? And by how much? There are an almost infinite combination, so how can we find the best ones? Simple methods like GridSearch can manually run through some preset combinations but will not necessarily pick the best possible parameters, just the best in the set you provided. And this grid will have to be changed when you add new features or otherwise modify your data. A more nuanced approach is to use Bayesian optimisation, probabilistically selecting the optimum values for each parameter after every round of evaluation. 
# 
# This kernel provides a single function for automated Bayesian hyperparameter optimisation with LightGBM, XGBoost and CatBoost via the Hyperopt package. Simply enter your data, target, and the type of model, and it will output a parameter dictionary you can then supply to your model-training kernels. While Hyperopt is a powerful tool, it can be difficult to implement and utilise. This function does the hard(-ish) work for you!
# 
# I owe a great debt to Will Koehrsen for his [exhaustive kernel on automated parameter tuning](https://www.kaggle.com/willkoehrsen/automated-model-tuning). While I have streamlined, modified and generalised much of his work for easy use with different model frameworks, I strongly recommend reading his kernel from top to bottom. It contains a thorough examination of how Bayesian parameter tuning works better than random selection, along with a step-by-step guide on outputting the Hyperopt progress logs. If your aim is to truly understand the *how* and *why* of parameter tuning, his kernels are some of the best resources you'll find.
# 
# The basic form of the function in this kernel is as follows:
# 
# `optimum_parameter_dictionary = quick_hyperopt(data, labels, 'model_type', NUM_EVALS)`
# 
# where `'model_type'` is either `'lgbm'` for LightGBM, `'xgb'` for XGBoost or `'cb'` for CatBoost, and `NUM_EVALS` is the number of parameter optimisation rounds. `optimum_paramater_dictionary` is the parameter dictionary you can supply to the relevant model.
# 
# # Usage Notes
# ---
# 
# * Parameter tuning can be a lengthy and intensive process. If the kernel is crashing or failing to finish, you may be using too much of your data; consider sampling it and running multiple kernels with each sample. You can then take the average/majority vote of each parameter that hyperopt selects. You can also reduce the number of evaluation rounds with the global parameter `NUM_EVALS`. 
# 
# * This function is written for regression tasks, but includes instructions on what must be modified if you are performing a classification task instead. I have tried to make this as simple as possible. By default it will use [ROC-AUC](http://gim.unmc.edu/dxtests/roc3.htm) as its objective metric for LightGBM/XGBoost and [Log Loss](https://www.kaggle.com/dansbecker/what-is-log-loss) for CatBoost.
# 
# * For speed, when optimising a CatBoost model this function is designed to work with GPU. Some CatBoost parameters aren't yet written to work with GPU so I've left them out. LightGBM and XGBoost work quickly with Hyperopt on CPU alone but I've left options for activating them in the code - simply uncomment the relevant lines. For LightGBM you'll have to [compile the GPU version in your kernel by following the steps here.](https://www.kaggle.com/vinhnguyen/gpu-acceleration-for-lightgbm/) 
# 
# * You may want to include other parameters in the search space or remove others; hopefully the code is transparent enough for you to work out how to modify it yourself according to your needs. For example, you may want to use different evaluation metrics, learning rate ranges or model objectives.
# 
# * The default objective metric for this function is Mean Absolute Error (MAE). For many regression tasks, Root Mean Squared Error (RMSE) is the preferred metric. Be sure to change this in the code below if required.
# 
# # Data Preparation
# ---
# 
# Your data does not need any special preperation before using `quick_hyperopt()` beyond the requirements of the desired model (CatBoost for example may complain about NaNs and infinite values). Simply separate the features from the labels/targets. The training features can be either a Pandas DataFrame or 2-D NumPy array. The labels can be either a Pandas Series, a single-column Pandas DataFrame or a flattened 1-D NumPy array. 

# In[1]:


import pandas as pd, numpy as np
data = pd.read_csv('../input/andrews-features/train_X.csv') #pandas DataFrame or numpy 2-D array
labels = pd.read_csv('../input/andrews-features/train_y.csv') #pandas Series, 1-column DataFrame or flattened numpy 1-D array

# In[2]:


def col_replace(df, df2):
    df_cols = df.columns.tolist()
    df2_cols = df2.columns.tolist()
    replace_cols = [x for x in df2_cols if x in df_cols]
    add_cols = [x for x in df2_cols if x not in df_cols]
    df[replace_cols] = df2[replace_cols]
    df = pd.concat([df, df2[add_cols]], axis=1)
    return(df.reset_index(drop=True))

data2 = pd.read_csv('../input/lanl-features/train_features_denoised.csv')
data = col_replace(data, data2).iloc[:-1, :].drop(['seg_id', 'target'], axis=1)

# # quick_hyperopt()
# ---
# Copy and paste the code below into your kernel and you're good to go! All questions, corrections and recommendations are welcome.

# In[3]:


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
LGBM_MAX_LEAVES = 2**11 #maximum number of leaves per tree for LightGBM
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

def quick_hyperopt(data, labels, package='lgbm', num_evals=NUM_EVALS, diagnostic=False):
    
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
            
            #for classification, set stratified=True and metrics=EVAL_METRIC_LGBM_CLASS
            cv_results = lgb.cv(space_params, train, nfold = N_FOLDS, stratified=False,
                                early_stopping_rounds=100, metrics=EVAL_METRIC_LGBM_REG, seed=42)
            
            best_loss = cv_results['l1-mean'][-1] #'l2-mean' for rmse
            #for classification, comment out the line above and uncomment the line below:
            #best_loss = 1 - cv_results['auc-mean'][-1]
            #if necessary, replace 'auc-mean' with '[your-preferred-metric]-mean'
            return{'loss':best_loss, 'status': STATUS_OK }
        
        train = lgb.Dataset(data, labels)
                
        #integer and string parameters, used with hp.choice()
        boosting_list = [{'boosting': 'gbdt',
                          'subsample': hp.uniform('subsample', 0.5, 1)},
                         {'boosting': 'goss',
                          'subsample': 1.0,
                         'top_rate': hp.uniform('top_rate', 0, 0.5),
                         'other_rate': hp.uniform('other_rate', 0, 0.5)}] #if including 'dart', make sure to set 'n_estimators'
        metric_list = ['MAE', 'RMSE'] 
        #for classification comment out the line above and uncomment the line below
        #metric_list = ['auc'] #modify as required for other classification metrics
        objective_list_reg = ['huber', 'gamma', 'fair', 'tweedie']
        objective_list_class = ['binary', 'cross_entropy']
        #for classification set objective_list = objective_list_class
        objective_list = objective_list_reg

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

# # Examples
# ---
# Here are some example usages. For reference, the data in this kernel is approximately 4200 rows by 136 columns. Participants of the LANL Earthquake Prediction challenge will know this dataset as "Andrew's Features". Simply run the function `quick_hyperopt()` with your training data & labels along with the string parameter for your model framework. 
# 
# ---
# # LightGBM
# ![](https://i.imgur.com/Jqw0FGz.jpg)
# 
# 
# LightGBM works quickly with Hyperopt and you can easily expect to run thousands of evaluations, conditional on the size of your dataset. The default parameter space I've defined covers a very broad range and will likely not need modifying unless you're running a classification task. Instructions for doing so are in the code.
# 
# The `num_leaves` parameter should be, at a maximum, equal to `2**max_depth`. In my experience this is overkill for the vast majority of cases when `max_depth` is greater than twelve and will quickly lead to overfitting. I've manually set it to `2**11` but you can easily alter it with the constant `LGBM_MAX_LEAVES` for your own purposes. Set it too high and it'll quickly exhaust the kernel memory - you can increase your available RAM by disabling the GPU disabled if desired.
# 
# For speed I haven't specified `'dart'` as one of the boosting options. If you want to include it, bear in mind that DART boosting cannot use early stopping so you'll have to specify `n_estimators` instead.

# In[ ]:


#obtain optimised parameter dictionary
lgbm_params = quick_hyperopt(data, labels, 'lgbm', 2500)

# The parameter dictionary can then be saved as a kernel output. Remember to use `.item()` when loading it in a new kernel.

# In[ ]:


#open with: lgbm_params = np.load('../input/YOUR_FILEPATH/lgbm_params.npy').item()
np.save('lgbm_params.npy', lgbm_params)

# ---
# # XGBoost
# ![](https://raw.githubusercontent.com/dmlc/dmlc.github.io/master/img/logo-m/xgboost.png)
# 
# My experience is that XGBoost takes longer than LightGBM to reach its optimum parameters. With less than 1000 evaluation rounds it can struggle to beat its default parameters! While it does eventually perform better than its defaults, to save time you might want to identify a set of parameters that already works well and then run a more intensive optimisation in a smaller range around them.
# 
# However, XGBoost works quickly with Hyperopt and the parameter space I've set up should cover most use cases. Again, `'dart'` has been left out for speed.
# 
# This kernel only demonstrates LightGBM parameter tuning; you can view the XGBoost equivalent in earlier versions.
# 

# In[ ]:


#xgb_params = quick_hyperopt(data, labels, 'xgb', 2000)

# And save the dictionary to kernel output:

# In[ ]:


#np.save('xgb_params.npy', xgb_params)

# ---
# # CatBoost 
# ![](https://siliconangle.com/wp-content/blogs.dir/1/files/2017/07/Yandex-CatBoost.png)
# CatBoost is more complicated when it comes to parameter tuning. The main issue is that defining too broad a parameter space will make it impractically slow or request too much space from the RAM, leading to a kernel crash. Some optimisation parameters work with the GPU, some do not. Frustratingly, the parameter `'rsm'`, alias `'colsample_bylevel'` which can dramatically speed up training, does not work with the GPU. I've tried to make the initial code as appropriate for general use as possible, but you will find you need to manually configure this process more than you would for LightGBM or XGBoost in accordance with the size and complexity of your data. Here are some official guidelines for CatBoost parameter tuning:
# 
# * [Parameter tuning guidelines](https://catboost.ai/docs/concepts/parameter-tuning.html)
# * [Speeding up training](https://catboost.ai/docs/concepts/speed-up-training.html)
# * [Full CatBoost training parameter list](https://catboost.ai/docs/concepts/python-reference_train.html)
# * [CatBoost cross-validation guide](https://catboost.ai/docs/concepts/python-reference_cv.html)
# 
# If this process is working too slowly, here are some measures you can try:
# 
# * Instead of using early stopping via `od_type` and `od_wait`, set a fixed number of trees using `iterations`. You may want to use a range of larger values for `learning_rate` when doing this, as low values may prevent model convergence.
# * minimise the range of `border_count`/`max_bin`. If training on GPU, the CatBoost docs do not recommend going over 32.
# * decrease the range of `max_leaves`. The CatBoost docs do not recommend going over 64. For speed, the current limit is 32.
# * `max_depth` increases the CV time exponentially - decreasing its maximum value in the code is probably the most important single change for accelerating cross-validation. The CatBoost docs state there is infrequently any benefit from values over 10. For the sake of speed in these example cases, I've set the maximum to 8. If you wish to adjust this, change the value of `CB_MAX_DEPTH`.
# 
# This implementation does not include any categorical features. If these are present in your training data you will have to add the 'cat_features' argument manually, and uncomment the 'one_hot_max_size' parameter in `quick_hyperopt()` if desired.
# 
# Note that `eval_metric` is the metric that determines early stopping and it is not necessarily the loss metric/objective.
# 
# This kernel only demonstrates LightGBM parameter tuning; you can view the CatBoost equivalent in earlier versions.

# In[ ]:


#cb_params = quick_hyperopt(data, labels, 'cb', 100)

# Again, when training is complete, save the dictionary to kernel output:

# In[ ]:


#np.save('cb_params.npy', cb_params)#

# # Diagnostic Mode
# ---
# If you wish to examine the individual evaluations, for example to see if there are any particular factors slowing down cross-validation, set `diagnostic=True` in `quick_hyperopt()`. This will output a tuple of the best iteration parameter dictionary, along with a Hyperopt trial list object of every round.

# In[ ]:


example_params, example_trials = quick_hyperopt(data, labels, 'lgbm', 5, diagnostic=True)

# To examine the nth round, extract the trial dictionary with `your_trial_output.trials[n]`. The `misc['vals']` entry in this dictionary will display the parameters tried during that round. Each entry also records the start and end time for that iteration. 

# In[ ]:


example_trials.trials[3]

# # LANL Earthquake Prediction
# ---
# As an experiment, let's compare the CV and LB scores obtained from the `quick_hyperopt()` parameters, and those from [this popular LANL Earthquake Prediction kernel.](https://www.kaggle.com/artgor/earthquakes-fe-more-features-and-samples)
# 
# The new and expanded Andrew's Features are used in this version.

# In[ ]:


lanl_params = {'num_leaves': 128,
          'min_data_in_leaf': 79,
          'objective': 'gamma',
          'max_depth': -1,
          'learning_rate': 0.01,
          "boosting": "gbdt",
          "bagging_freq": 5,
          "bagging_fraction": 0.9,
          "bagging_seed": 11,
          "metric": 'mae',
          "verbosity": -1,
          'reg_alpha': 0.1302650970728192,
          'reg_lambda': 0.3603427518866501,
          'feature_fraction': 0.1
         }

# In[ ]:


from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error

sub_1 = pd.read_csv('../input/LANL-Earthquake-Prediction/sample_submission.csv')
sub_2 = pd.read_csv('../input/LANL-Earthquake-Prediction/sample_submission.csv')
test = pd.read_csv('../input/andrews-features/test_X.csv')
test2 = pd.read_csv('../input/lanl-features/test_features_denoised.csv')
test = col_replace(test, test2).drop(['seg_id', 'target'], axis=1)

preds1 = np.zeros(len(sub_1))
preds2 = np.zeros(len(sub_2))

MAE_val1 = 0
MAE_val2 = 0

n_fold = 5
folds = KFold(n_splits=n_fold, shuffle=True, random_state=42)
X = data
y = labels.time_to_failure.values

for train_idx, valid_idx in folds.split(y):
    X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
    y_train, y_valid = y[train_idx], y[valid_idx]
        
    model1 = lgb.LGBMRegressor(**lanl_params, n_estimators = 50000, n_jobs = -1, eval_metric='mae')
    model1.fit(X_train, y_train,eval_set=[(X_train, y_train), (X_valid, y_valid)],
              verbose=0, early_stopping_rounds=200)
    val_preds1 = model1.predict(X_valid)
    MAE_val1 += mean_absolute_error(y_valid, val_preds1)/n_fold
    preds1 += model1.predict(test)/n_fold
    
    model2 = lgb.LGBMRegressor(**lgbm_params, n_estimators = 50000, n_jobs = -1)
    model2.fit(X_train, y_train,eval_set=[(X_train, y_train), (X_valid, y_valid)],
              verbose=0, early_stopping_rounds=200)
    val_preds2 = model2.predict(X_valid)
    MAE_val2 += mean_absolute_error(y_valid, val_preds2)/n_fold
    preds2 += model2.predict(test)/n_fold
    
sub_1['time_to_failure'] = preds1
sub_2['time_to_failure'] = preds2
sub_1.to_csv('submission_lanl_params.csv', index=False)
sub_2.to_csv('submission_hyperopt_params.csv', index=False)

print('CV score - public kernel parameters: {}'.format(MAE_val1))
print('CV score - quick_hyperopt parameters: {}'.format(MAE_val2))

# The Hyperopt parameters clearly improve the cross-validation score, and the leaderboard scores for the previous kernel were as follows:
# 
# * LANL kernel parameters: CV 2.047 LB 1.516
# * `quick_hyperopt` parameters: CV 2.030 LB **1.493**
# 
# # Summary
# ---
# 
# It's as simple as that! I've designed this function so that it will be 'plug-in-and-play' for the majority of regression cases. For LightGBM and XGBoost it will work without a GPU but for CatBoost some parameters will need to be removed from the optimisation process if you want to run on CPU alone. I don't recommend this since parameter tuning CatBoost with a GPU is already very slow. It is likely that the only parameters you will need to change are the number of optimisation rounds via `NUM_EVALS`, and the respective tree depths/maximum leaf counts for the different model frameworks. If you need to modify `quick_hyperopt()` for classification, instructions are in the code. Bear in mind that you may need to consider parameters that are unique to your data, such as `scale_pos_weight` if the classes are imbalanced. For other modifications please consult the Hyperopt docs, along with those for LightGBM, XGBoost and CatBoost:
# 
# * [Hyperopt documentation](https://github.com/hyperopt/hyperopt/wiki/FMin)
# * [LightGBM documentation](https://lightgbm.readthedocs.io/en/latest/)
# * [XGBoost documentation](https://xgboost.readthedocs.io/en/latest/index.html)
# * [CatBoost documentation](https://catboost.ai/docs/)
# 
# I hope this kernel will be of use to you. 
