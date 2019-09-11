#!/usr/bin/env python
# coding: utf-8

# # Time Split Validation - Adversarial EDA - Microsoft Malware
# Is anyone else frustrated with the discrepency between CV and LB?! In this kernel, I show an example of two models that both have CV 0.730 but one has LB 0.670 while the other has LB 0.680. Why?! We will explore this and suggest ideas to more accurately estimate LB.
# 
# # The Culprit
# 
# The Microsoft Malware data has 82 explanatory variables which can all be intrepreted as categorical variables. The reason for CV and LB gap is the difference between TRAIN and TEST variable value distributions. In this kernel, we compare the variable distributions between TRAIN and TEST. If TRAIN and TEST were random samples from the same population then their distributions would be nearly identical (since sample size is a massive 8 million). But this is not what we see! 
# 
# Why? Because TRAIN is sampled from the population of August September 2018 while TEST is sampled from the different population October November 2018 (shown [here][1]). Furthermore we believe that PUBLIC TEST is October 2018 and PRIVATE TEST is November 2018 (shown [here][2]).  
#   
# [1]: https://www.kaggle.com/cdeotte/time-series-eda-malware-0-64
# [2]: https://www.kaggle.com/rquintino/2-months-train-1-month-public-1-day-private

# In[ ]:


import numpy as np, pandas as pd, gc, random
import matplotlib.pyplot as plt

def load(x):
    ignore = ['MachineIdentifier']
    if x in ignore: return False
    else: return True

# LOAD TRAIN AND TEST
df_train = pd.read_csv('../input/microsoft-malware-prediction/train.csv',dtype='category',usecols=load)
df_train['HasDetections'] = df_train['HasDetections'].astype('int8')
if 5244810 in df_train.index:
    df_train.loc[5244810,'AvSigVersion'] = '1.273.1144.0'
    df_train['AvSigVersion'].cat.remove_categories('1.2&#x17;3.1144.0',inplace=True)
#df_train = df_train.sample(8000000).reset_index(drop=True)

df_test = pd.read_csv('../input/microsoft-malware-prediction/test.csv',dtype='category',usecols=load)
#df_test = df_test.sample(1000000).reset_index(drop=True)

# # Test Full versus Train Full
# To compare the distributions of TEST.csv and TRAIN.csv, I wrote a special Python function. (If interested, click the button 'see code' to view.) The visual may be confusing at first, so let me explain. Basically the plot is two histograms on top of each other. Instead of using bars, it uses a line which follows where the tops of the bars would be. The x-axis are the category variable's possible values. They have been ordered from most frequent in TRAIN to less frequent in TRAIN. Then values containing less than 0.1% of data are removed. Then they are relabeled 0, 1, 2, ... n.
# 
# The solid blue line is TRAIN and the solid green line is TEST. If the distributions are the same then the blue and green line would coincide. The dotted blue lines indicate 4x more than TRAIN and 4x less than TRAIN. Therefore if the green line crosses outside the dotted lines, then TEST has a value that is 4x more or 4x less than TRAIN. Let's plot `CountryIdentifier` for TRAIN.csv versus TEST.csv. 

# In[ ]:


# COMPARE VALUE DENSITIES FROM TWO DIFFERENT DATAFRAMES
#
# PARAMETERS
# df1: pandas.DataFrame containing variable
# df2: pandas.DataFrame containing variable
# col: column to compare between df1 and df2
# override: set to False to prevent display when variables similar
# verbose: display text summary
# scale: zooms y-axis
# title: plot title
# lab1: legend label for df1
# lab2: legend label for df2
# prefix: pre text for verbose summary
#
def comparePlot(df1, df2, col, factor=4, override=True, verbose=True, scale=0.5, title='',
                lab1='', lab2='', prefix=''):
    cv1 = pd.DataFrame(df1[col].value_counts(normalize=True).reset_index().rename({col:'train'},axis=1))
    cv2 = pd.DataFrame(df2[col].value_counts(normalize=True).reset_index().rename({col:'test'},axis=1))
    cv3 = pd.merge(cv1,cv2,on='index',how='outer')
    cv3['train'].fillna(0,inplace=True)
    cv3['test'].fillna(0,inplace=True)
    cv3 = cv3.iloc[np.lexsort((cv3['test'], -cv3['train']))]
    cv3['total'] = cv3['train']+cv3['test']
    cv3['trainMX'] = cv3['train']*factor
    cv3['trainMN'] = cv3['train']/factor
    cv3 = cv3[cv3['total']>0.0001]
    if (len(cv3)<5): return
    cv3.reset_index(inplace=True)
    MX = (cv3['test'] > cv3['trainMX'])
    mxSum = round(100*cv3.loc[MX,'test'].sum(),1)
    MN = (cv3['test'] < cv3['trainMN'])
    mnSum = round(100*cv3.loc[MN,'test'].sum(),1)
    #if override | (MX.sum()+MN.sum()>0):
    if override | (mxSum + mnSum > 1):
        plt.figure(figsize=(15,5))
        if lab1=='': lab1='Train'
        if lab2=='': lab2='Test'
        plt.plot(cv3.index,cv3['train'],linewidth=3,alpha=0.7,color='b',label=lab1)
        plt.plot(cv3.index,cv3['trainMX'],linewidth=2,alpha=1.0,linestyle=':',color='b',label=str())
        plt.plot(cv3.index,cv3['trainMN'],linewidth=2,alpha=1.0,linestyle=':',color='b',label=str())
        #plt.bar(cv3.index,cv3['test'],linewidth=3,alpha=0.7,color='g', label='Test.csv')
        plt.plot(cv3.index,cv3['test'],linewidth=3,alpha=0.7,color='g',label=lab2)
        plt.legend()
        if title=='': plt.title(col)
        else: plt.title(col+' - '+title)
        plt.xlabel(col+' values (ordered by train frequency and relabeled)')
        plt.ylabel('Frequency')
        mx = max(cv3['train'].max(),cv3['test'].max())
        #plt.ylim(0,mx*1.05)
        plt.ylim(0,mx*scale)
        plt.show()
        tempMX = cv3.loc[MX.values,['index','test']].sort_values('test',ascending=False)['index']
        tempMN = cv3.loc[MN.values,['index','test']].sort_values('test',ascending=False)['index']
        if verbose:
            if MX.sum()>0:    
                print(prefix+'Test.csv',col,'has',MX.sum(),'values 4x MORE freq than Train.csv. (',mxSum,'% of data)')
            if MX.sum()>10: print('  Top 10 by test freq:',list(tempMX)[:10])
            elif MX.sum()>0: print(list(tempMX)[:10])
            if MN.sum()>0:
                print(prefix+'Test.csv',col,'has',MN.sum(),'values 4x LESS freq than Train.csv. (',mnSum,'% of data)')
            if MN.sum()>10: print('  Top 10 by test freq:',list(tempMN)[:10])
            elif MN.sum()>0: print(list(tempMN)[:10])
    return

# In[ ]:


comparePlot(df_train, df_test, 'CountryIdentifier', verbose=False, title='Test vs. Train')

# Notice how the green line wiggles and does not stay on top of the blue line. Also notice that it stays within the dotted blue lines. This means that the distribution of `CountryIdentifier` values in TEST.csv are different but pretty similar to TRAIN.csv. It is concerning that they differ. For comparsion, see below. 
# # Random Validation Set versus Train Subset
# Below I split TRAIN.csv into a randomly selected validation set and training subset. Notice how the distributions are nearly identical. This is what random samples from the same population look like.

# In[ ]:


df_trainA = df_train.sample(frac=0.5)
df_trainB = df_train[ ~df_train.index.isin(df_trainA.index)]
comparePlot(df_trainA, df_trainB, 'CountryIdentifier', verbose=False,
            title='Random Validation Set vs. Train Subset', lab1='Train', lab2='Validation')

# Comparing the two plots above, we see that a random validation set does not mimic the relationship between TRAIN and TEST. Therefore it is not a good validation scheme to predict LB score.
# # Time Split Validation Set versus Train Subset
# To make a more useful validation set, lets split TRAIN roughly in half ordered by time using `AvSigVerion` (which is a proxy for time). Then we will compare this Time Split Validation set versus Train subset. All `AvSigVersion` with second number `<275` are before August 23, 2018 while `>=275` are after. This splits TRAIN roughly in half and avoids `AvSigVersion` overlap in the second number (which mimics TEST / TRAIN). We notice that this plot looks more like the TEST TRAIN plot above.

# In[ ]:


df_train['AvSigVersion2'] = df_train['AvSigVersion'].map(lambda x: np.int(x.split('.')[1]))
df_trainC = df_train[ df_train['AvSigVersion2']<275 ]
df_trainD = df_train[ df_train['AvSigVersion2']>=275 ]
comparePlot(df_trainC, df_trainD, 'CountryIdentifier', verbose=False,
            title='Time-split Validation vs. Train', lab1='Train', lab2='Validation')

# # Private Test versus Train Full
# The biggest difference in distribution is going from TRAIN to PRIVATE TEST. It is believed that PRIVATE TEST is the month of November 2018. And TRAIN's most recent observation is September 2018. In that large time gap, the distribution of variables will change. Specially it is believed that PRIVATE TEST is after October 25, 2018. Below is the plot of PRIVATE TEST versus TRAIN, and afterward is the plot of PUBLIC TEST versus TRAIN.

# In[ ]:


from datetime import datetime
datedictAS = np.load('../input/malware-timestamps/AvSigVersionTimestamps.npy')[()]
df_train['DateAS'] = df_train['AvSigVersion'].map(datedictAS)
df_test['DateAS'] = df_test['AvSigVersion'].map(datedictAS)

df_testA = df_test[ df_test['DateAS']<datetime(2018,10,25) ]
df_testB = df_test[ df_test['DateAS']>datetime(2018,10,25) ]
comparePlot(df_train, df_testB, 'CountryIdentifier', verbose=False,
           title='Private Test vs. Train', lab1='Train', lab2='Private Test')
comparePlot(df_train, df_testA, 'CountryIdentifier', verbose=False,
           title='Public Test vs. Train', lab1='Train', lab2='Public Test')

# # LB Score versus CV Score
# We now understand how the distribution of TEST differs from TRAIN. Let's now compare a Time Split Validation scheme to a Random Validation scheme. Let's see which one more accurately estimates LB. In order to compare, we will need two sets of variables (two models). 
# 
# ## Model One
# Our first set of variables will be the entire TRAIN.csv converted into categories with no further processing.
# ## Model Two
# Our second set of variables will be the entire TRAIN.csv with certain values removed. Notice I say values not variables. For each of the 82 variables, we will look at all their possible values in TRAIN. If a value does not appear in TEST, we will remove it. If a value appears less than 0.01% of total TRAIN, we will remove it. If a value appears 4x more frequent in TRAIN than TEST, we will remove it. If a value appears 4x less frequent in TRAIN than TEST, we will remove it. (This idea is from [Vladislav Bogorod's][2] brillant kernel [here][1].)
# 
# For example, the variable `AppVersion` contains the value `4.18.1807.18075`. This value appears in TRAIN over 4x as frequent as it appears in TEST. Therefore we will remove this value's category from `AppVersion`. All values that appear 4x more or 4x less are indicated in the compare plots whenever the green solid line crosses over the blue dotted line in. (Note: It has been shown that model two can score 0.692 LB [here][1])
# 
# ## Result
# Below we will see that  
#   
#     Model One scores: 0.732 Random, 0.695 TimeSplit, 0.672 LB  
#     Model Two scores: 0.729 Random, 0.704 TimeSplit, 0.681 LB
# 
# This shows that Time Split Validation was better able to estimate LB score than Random Validation (typical CV). It's a relief to find a test that distinguishes models with different LB scores!
#   
# [1]: https://www.kaggle.com/bogorodvo/upd-lightgbm-baseline-model-using-sparse-matrix
# [2]: https://www.kaggle.com/bogorodvo

# In[ ]:


# FACTORIZE DATA
def factor_data(df_train, df_test, col):
    df_comb = pd.concat([df_train[col],df_test[col]],axis=0)
    df_comb,_ = df_comb.factorize(sort=True)
    # MAKE SMALLEST LABEL 1, RESERVE 0
    df_comb += 1
    # MAKE NAN LARGEST LABEL
    df_comb = np.where(df_comb==0, df_comb.max()+1, df_comb)
    df_train[col] = df_comb[:len(df_train)]
    df_test[col] = df_comb[len(df_train):]
    del df_comb
    
# OPTIMIZE MEMORY
def reduce_memory(df,col):
    mx = df[col].max()
    if mx<256:
            df[col] = df[col].astype('uint8')
    elif mx<65536:
        df[col] = df[col].astype('uint16')
    else:
        df[col] = df[col].astype('uint32')

# REDUCE CATEGORY CARDINALITY
def relax_data(df_train, df_test, col):
    cv1 = pd.DataFrame(df_train[col].value_counts().reset_index().rename({col:'train'},axis=1))
    cv2 = pd.DataFrame(df_test[col].value_counts().reset_index().rename({col:'test'},axis=1))
    cv3 = pd.merge(cv1,cv2,on='index',how='outer')
    cv3['train'].fillna(0,inplace=True)
    cv3['test'].fillna(0,inplace=True)
    factor = len(df_test)/len(df_train)
    cv3['remove'] = False
    cv3['remove'] = cv3['remove'] | (cv3['train'] < len(df_train)/9000)
    cv3['remove'] = cv3['remove'] | (factor*cv3['train'] < cv3['test']/4)
    cv3['remove'] = cv3['remove'] | (factor*cv3['train'] > 4*cv3['test'])
    cv3['new'] = cv3.apply(lambda x: x['index'] if x['remove']==False else 0,axis=1)
    cv3['new'],_ = cv3['new'].factorize(sort=True)
    cv3.set_index('index',inplace=True)
    cc = cv3['new'].to_dict()
    df_train[col] = df_train[col].map(cc)
    reduce_memory(df_train,col)
    df_test[col] = df_test[col].map(cc)
    reduce_memory(df_test,col)
    
# DISPLAY MEMORY STATISTICS
def display_memory(df_train, df_test):
    print(len(df_train),'rows of training data use',df_train.memory_usage(deep=True).sum()//1e6,'Mb memory!')
    print(len(df_test),'rows of test data use',df_test.memory_usage(deep=True).sum()//1e6,'Mb memory!')

# CONVERT DTYPES TO CATEGORIES
def categorize(df_train, df_test, cols):
    for col in cols:
        df_train[col] = df_train[col].astype('category')
        df_test[col] = df_test[col].astype('category')

# In[ ]:


del df_trainA, df_trainB, df_trainC, df_trainD
del df_train['DateAS'], df_test['DateAS']; x=gc.collect()
cols = [x for x in df_train.columns if x not in ['HasDetections','AvSigVersion2']]
    
print('Factorizing...')
for col in cols: factor_data(df_train, df_test, col)
print('Reducing memory...')
for col in cols: reduce_memory(df_train, col)
for col in cols: reduce_memory(df_test, col)
categorize(df_train, df_test, cols)
display_memory(df_train, df_test)

# # Model One: using Random Validation

# In[ ]:


import lightgbm as lgb
df_trainA = df_train.sample(frac=0.5)
df_trainB = df_train[ ~df_train.index.isin(df_trainA.index)]
model = lgb.LGBMClassifier(n_estimators=3000, colsample_bytree=0.2, objective='binary', num_leaves=16,
          max_depth=-1, learning_rate=0.1)
h=model.fit(df_trainA[cols], df_trainA['HasDetections'], eval_metric='auc',
          eval_set=[(df_trainB[cols], df_trainB['HasDetections'])], verbose=250,
          early_stopping_rounds=100)

# In[ ]:


del df_trainA, df_trainB; x=gc.collect()
idx = 0; chunk = 2000000
pred_val = np.zeros(len(df_test))
while idx < len(df_test):
    idx2 = min(idx + chunk, len(df_test) )
    idx = range(idx, idx2)
    pred_val[idx] = model.predict_proba(df_test.iloc[idx][cols])[:,1]
    idx = idx2
submit = pd.read_csv('../input/microsoft-malware-prediction/sample_submission.csv')
submit['HasDetections'] = pred_val
submit.to_csv('ModelOne.csv', index=False)

# ![image](http://playagricola.com/Kaggle/one21319.png)

# # Model One: using Time Split Validation

# In[ ]:


df_trainC = df_train[ df_train['AvSigVersion2']<275 ]
df_trainD = df_train[ df_train['AvSigVersion2']>=275 ]
model = lgb.LGBMClassifier(n_estimators=3000, colsample_bytree=0.2, objective='binary', num_leaves=16,
          max_depth=-1, learning_rate=0.1)
h=model.fit(df_trainC[cols], df_trainC['HasDetections'], eval_metric='auc',
          eval_set=[(df_trainD[cols], df_trainD['HasDetections'])], verbose=250,
          early_stopping_rounds=100)

# # Model Two: using Time Split Validation

# In[ ]:


print('Converting data to Model Two...')
df_trainC = df_trainC.copy()
df_trainD = df_trainD.copy()
for col in cols: relax_data(df_trainC, df_trainD, col)
categorize(df_trainC, df_trainD, cols)
model = lgb.LGBMClassifier(n_estimators=3000, colsample_bytree=0.2, objective='binary', num_leaves=16,
          max_depth=-1, learning_rate=0.1)
h=model.fit(df_trainC[cols], df_trainC['HasDetections'], eval_metric='auc',
          eval_set=[(df_trainD[cols], df_trainD['HasDetections'])], verbose=250,
          early_stopping_rounds=100)

# # Model Two: using Random Validation

# In[ ]:


print('Converting data to Model Two...')
del df_trainC, df_trainD; x=gc.collect()
for col in cols: relax_data(df_train, df_test, col)
categorize(df_train, df_test, cols)
df_trainA = df_train.sample(frac=0.5)
df_trainB = df_train[ ~df_train.index.isin(df_trainA.index)]
model = lgb.LGBMClassifier(n_estimators=3000, colsample_bytree=0.2, objective='binary', num_leaves=16,
          max_depth=-1, learning_rate=0.1)
h=model.fit(df_trainA[cols], df_trainA['HasDetections'], eval_metric='auc',
          eval_set=[(df_trainB[cols], df_trainB['HasDetections'])], verbose=250,
          early_stopping_rounds=100)

# In[ ]:


del df_trainA, df_trainB, df_train; x=gc.collect()
idx = 0; chunk = 2000000
pred_val = np.zeros(len(df_test))
while idx < len(df_test):
    idx2 = min(idx + chunk, len(df_test) )
    idx = range(idx, idx2)
    pred_val[idx] = model.predict_proba(df_test.iloc[idx][cols])[:,1]
    idx = idx2
submit = pd.read_csv('../input/microsoft-malware-prediction/sample_submission.csv')
submit['HasDetections'] = pred_val
submit.to_csv('ModelTwo.csv', index=False)

# ![image](http://playagricola.com/Kaggle/two21319.png)

# # Conclusion
# In conclusion, we see how a time split validation scheme approximates LB better than a random validation scheme. Additionally, using compare plots EDA helps us choose time stable variables for our models. Below I show an alternative method, adversarial validation. After that I present all the compare plots.
# 
# # Adversarial Validation
# Alternatively, some people use adversarial validation to discover troublesome values and/or test their models. Above, we used Compare Plots to discover troublesome values and then created our Model One and Model Two. We won't use adversarial validation here to find variables, but we will apply adversarial validation on Model One and Model Two to confirm that we did a good job. (Note since I use a small tree below, I use training AUC to approximate validation AUC.)
# ## What is Adversarial Validation?
# Normallly a model classifies which computers have Malware and which computers do not. In adversarial validation, we mix all the TRAIN and TEST data together. We add a new variable `IsTest`. For all computers from TEST, we set `IsTest=1` and all computers from TRAIN, we set `IsTest=0`. We then build a model that attempts to classify whether a computer is `IsTest` or not. If the distribution of variables are the same in TEST and TRAIN (were randomly drawn from the sample population), then an adversarial model can not distinguish TEST from TRAIN. But when the distributions are different, it can. Below we witness how an adversarial model can distinquish TEST from TRAIN when using the variables from Model One but cannot distinguish when using the variables from Model Two. (With the exception that I need to remove `SMode` since it is skewed but didn't exceed 4x.)

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
from sklearn import tree
import graphviz

# LOAD TRAIN AND TEST
df_train = pd.read_csv('../input/microsoft-malware-prediction/train.csv',dtype='category',usecols=load, nrows=10000)
df_train['HasDetections'] = df_train['HasDetections'].astype('int8')
if 5244810 in df_train.index:
    df_train.loc[5244810,'AvSigVersion'] = '1.273.1144.0'
    df_train['AvSigVersion'].cat.remove_categories('1.2&#x17;3.1144.0',inplace=True)
#df_train = df_train.sample(1000000).reset_index(drop=True)
df_test = pd.read_csv('../input/microsoft-malware-prediction/test.csv',dtype='category',usecols=load, nrows=10000)
#df_test = df_test.sample(1000000).reset_index(drop=True)

# FACTORIZE
cols = [x for x in df_train.columns if x not in ['HasDetections','AvSigVersion2']]
for col in cols: factor_data(df_train, df_test, col)
for col in cols: reduce_memory(df_train, col)
for col in cols: reduce_memory(df_test, col)
categorize(df_train, df_test, cols)
# COMBINE TRAIN AND TEST
df_train['HasDetections'] = 0
df_test['HasDetections'] = 1
df_comb = pd.concat([df_train,df_test],axis=0)

# VALIDATION
model = DecisionTreeClassifier(max_leaf_nodes=5)
model.fit(df_comb[cols], df_comb['HasDetections'])
pred_val = model.predict_proba(df_comb[cols])[:,1]
print('Model One: Adversarial Training AUC = ',round( roc_auc_score(df_comb['HasDetections'],pred_val),4 ) )
#print('Adversarial Model has tree depth =',model.tree_.max_depth,'and node count =',model.tree_.node_count)
print('Adversarial Model has max_leaf_nodes=5')
# PLOT TREE                    
tree_graph = tree.export_graphviz(model, out_file=None, max_depth = 10,
        impurity = False, feature_names = cols, class_names = ['No', 'Yes'],
        rounded = True, filled= True )
graphviz.Source(tree_graph)

# In[ ]:


# CONVERT VARIABLES TO MODEL TWO
for col in cols: relax_data(df_train, df_test, col)
categorize(df_train, df_test, cols)
df_comb = pd.concat([df_train,df_test],axis=0)
# REMOVE TROUBLESOME SMODE
cols2 = cols.copy()
cols2.remove('SMode')

#VALIDATION
model = DecisionTreeClassifier(max_leaf_nodes=5)
model.fit(df_comb[cols2], df_comb['HasDetections'])
pred_val = model.predict_proba(df_comb[cols2])[:,1]
print('Model Two: Adversarial Training AUC = ',round( roc_auc_score(df_comb['HasDetections'],pred_val),4 ) )
#print('Adversarial Model has tree depth =',model.tree_.max_depth,'and node count =',model.tree_.node_count)
print('Adversarial Model has max_leaf_nodes=5')
# PLOT TREE          
tree_graph = tree.export_graphviz(model, out_file=None, max_depth = 10,
        impurity = False, feature_names = cols2, class_names = ['No', 'Yes'],
        rounded = True, filled= True )
graphviz.Source(tree_graph)

# # EDA Compare Plots
# Below are all variables containing 5 or more categories (after removing values with less than 0.1% overall data). The most troublesome variables are: `EngineVersion`, `AvSigVersion`, `AppVersion`, `Census_OSVersion`, `Census_OSBuildRevision`, `DefaultBrowsersIdentifier`, `OsBuildLab`, `IeVerIdentifier`, `OsBuild`, `OsPlatformSubRelease`, `Census_OSBranch`, `Census_OSBuildNumber`. These variables have 89%, 84%, 70%, 38%, 37%, 9%, 7%, 3%, 2%, 2%, 2%, 2% differences between TRAIN and PUBLIC TEST respectively. And 100%, 100%, 84%, 60%, 60%, 11%, 12%, 6%, 6%, 6%, 6%, 6% differences between TRAIN and PRIVATE TEST. 
#   
# In order to make the most accurate model, you need to do something with these variables.  Recall that Model Two above was created by removing all values from these variables where the green solid line crosses the blue dotted line. To create a better model, transform these values into new values, whose TEST TRAIN distribution is similar. Use creative feature engineering and/or encoding techniques.

# In[ ]:


cols2 = {'AVProductStatesIdentifier':0.01, 'CountryIdentifier':0.4, 'LocaleEnglishNameIdentifier':0.3, 'SmartScreen':0.4,
         'Census_OEMNameIdentifier':0.1,'Census_TotalPhysicalRAM':0.5,'Census_InternalPrimaryDiagonalDisplaySizeInInches':0.05,
        'Census_OSInstallTypeName':0.75,'Census_OSInstallLanguageIdentifier':0.3,'Census_FirmwareManufacturerIdentifier':0.1,
        'EngineVersion':1.0, 'AppVersion':0.7, 'OsBuildLab':0.2, 'Census_OEMModelIdentifier':0.15,
        'Census_InternalBatteryNumberOfCharges':0.05}

df_train = pd.read_csv('../input/microsoft-malware-prediction/train.csv',dtype='category',usecols=load)
df_test = pd.read_csv('../input/microsoft-malware-prediction/test.csv',dtype='category',usecols=load)
df_test['DateAS'] = df_test['AvSigVersion'].map(datedictAS)
df_testA = df_test[ df_test['DateAS']<datetime(2018,10,25) ]
df_testB = df_test[ df_test['DateAS']>datetime(2018,10,25) ]

for x in df_train.columns[:-2]:
    s = 0.5
    if x in cols2: s = cols2[x] 
    comparePlot(df_train,df_testA,x,scale=s, title='Public Test vs. Train', prefix='Public ')
    comparePlot(df_train,df_testB,x,scale=s, title='Private Test vs. Train', prefix='Private ')
