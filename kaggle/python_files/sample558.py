#!/usr/bin/env python
# coding: utf-8

# Here I describe how I got from around 2000th place to 10th. As there are already many great kernels, I will try to make that description short. 
# 
# First I took all the data which was there, threw it into random forest and got 0.13825 on the leaderboard. Next I describe how I improved that... 
# 
# **Step 0: log-transform of sales.** Log-transform of the dependent variable. There are two reasons to predict log of sales, not sales: first, evaluation metric depends on log of sales and, second, many monetary entities have distribution close to log-normal, i.e. heavy-tailed. Usually it is better to predict smt which is not heavy tailed (I honestly tried to predict sales per se, and results were worse).   
# 
# **Step 1: re-coding predictors / missings / outliers.** Some of the predictors which are ordinal (i.e. there is a natural order in their values) initially are stored as factors. I re-coded them in a sensible way (there are many great kernels which describe in detail how to do it; examples of such predictors are *ExterQual* and *BsmtCond*). Obviously I had to fill in missing values - missings in numeric predictors I filled in with zeros when it made sense (e.g. if there are no baths in the basement, BsmtFullBath should be equal to zero), missings in categorical predictors I filled in with modes or other typical values. Last, many users advise removing observations with outliers in predictors *OverallQual* and *GrLivArea* - it is worth doing.
# 
# **Step 2: non-linearities.** As I planned to use linear methods (lasso, ridge, svm), I *replaced* all heavy-tailed predictors with their logs and for some of the predictors *added* their squares (i.e. we have predictor X and we add predictor X^2). Replacing heavy-tailed predictors with their logs is motivated by the fact that a) linear methods might fit such predictors with very small weights and most of the information contained in the values might be lost b) predictions when such predictors take very high values might be also very high or misleading. Adding squares is motivated by non-linearities in scatterplots "predictor vs. log of sales" - we assume that similar non-linearities will also hold when we add predictor to multi-dimensional model). Check, for example that scatterplot of BsmtQual vs log of Sales:
# 
# ![BsmtQual](https://i.imgur.com/K2HLZtbm.png)
# 
# **Step 3: new predictors.** For some of the predictors helped adding indicator variables which are equal to one if respective predictor take certain value. I chose predictors and their values to indicate based on scatterplots - if on scatterplot "predictor vs. log of sales" for the certain value of the predictor there is a spike in values of log-sales or non-linearity, such value of predictor is a good candidate. Example of such non-linearity can be found below. Scatterlot shows re-coded values of BsmtFinType1 vs. log of sales. In the model I added indicator which is equal to one when BsmtFinType1 = 1 (which corresponds to BsmtFinType1 equal to *Unf*):
# 
# ![BsmtFinType1](https://i.imgur.com/MCosvgDm.png)
# 
# **Step 4: stacking.** As a next  step I used 10-fold cross-validation and stacking: for each "run" of cross-validation, I fit 5 models on 9 out of 10 folds (lasso, ridge, elastic net, GBM and LGB), make predictions on the left-out fold and use these five sets of predictions as an input into another lasso model to forecast log of sales on that left-out fold (such lasso model is called meta-model). In total we have 6\*10=60 models - 10 sets of 6 models. All these models I used to make final predictions: we take test dataset, make predictions using 5 sub-models and then use outputs of these models as an input into respective meta-model to get set of predictions for given set of models; we repeat that process 10 times to get 10 sets of predictions and then average them using arithmetic mean to get data used for submission. Steps 0-4 gave me smt around 0.120 on the LB. 
# 
# **Step 5: tuning.** For stacking I used 6 different models and each needed tuning (for each "run" of cross-validation I fit 6 models with always the same parameters, e.g. we also fit ridge with regularization parameter equal to 9.0). I spent a lot of time and submissions to fine-tune parameters (best improvement was by tuning *min_samples_leaf* and *min_samples_split* for GradientBoostingRegressor). Eventually I got smt around 0.1188 and was confident I won't be able to improve it by tuning.
# 
# **Step 6: more with missing values.** Next I tried different strategies of filling in missing values (modes/means/medians/etc.). Best thing which worked was to some extent unexpected: it was based on mice package in R and is described here - https://www.kaggle.com/couyang/svm-benchmark-approach-0-11820-lb-top-13. First I tried using dataset I got from R/mice in my python code for stacking, then I tried pre-filling in some missing values in sensible way and using mice on top, but smt else gave best results. I took training dataset without any transformations or cleaning, used mice on top and then fitted SVM. Basically, I could just re-use code by https://www.kaggle.com/couyang per se - no outlier cleaning, no feature transformations, just mice and svm, nothing else. Using R/mice/svm and python/stacking and taking geometric mean of these predictions gave me 0.11505 on LB, which was already great. It was unexpected that such simple and not-business-driven approach to fill in outliers worked. Note, that mice might fill in missings in numeric predictors with smt not equal to zero when there should be zero. I suppose that such approach helped me as it was quite different from what I got in python.
# 
# **Step 7: brutal force.** I was already happy with 0.11505 but there was last thing I wanted to try. Regression often works bad for edge cases - for small or big values of predictors. I took train, ran R/mice/svm, python/stacking, averaged results using geometric mean, got final predictions for log of sales and plotted them against logs of real values: 
# 
# ![scatter](https://i.imgur.com/5Ky8u7pm.png)
# 
# It was obvious from that picture that for small final predictions we are overestimating sales, for big values of predictions we are underestimating. I tried fitting splines (natural/smoothing) and local regression in R and using them on a test set. It immediately gave me improvement. Final and best submit was based on smt which is quite brutal: we take predicted sales (not log-, but sales as they are), take top and bottom percentiles and manually increase/decrease forecasts. Which eventually gave me 0.10943 on the LB. Also, obviously brute force approach of manual change in forecasted values is more overfitting then anything else as there is no so much business motivation behind it (even using splines would be better for productionized model). It is important to note that such brute force approach improves forecasts for small number of observations, but it seems that is is enough to improve score on LB.
# 
# Some other things I tried which didn't help include a) PCA  b) adding more and more predictors c) adding random forest, KRR, SVM, xgboost to stacking d) using keras on tensorflow and scaling + dense neural nets.
# 
# **Summary:** I would say that productionized version would have evaluation metric of around 0.12, as it is possible to build simple model which will give such result. Interesting things which helped: adding squares of some of the predictors, using mice+svm in R on top of the raw data, working with edge-predictions (very low of very high ones).
# 
# **NB** It seems that kaggle does not support having both R and python inside one notebook, so I had to comment R code. Also I use stacking on 30-fold cross-validation (it takes around 10 minutes on my laptop to fit all models), but here it gets stopped due to timeout.
# 

# In[10]:


import numpy as np
import pandas as pd
import datetime

import random
import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.kernel_ridge import KernelRidge

from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet 

from sklearn.svm import SVR

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler

import lightgbm as lgb

from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

pd.set_option('display.max_columns', None)

# In[11]:


# ------------ Reading and cleaning data ------------ 

# In[12]:


mydata = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

# As suggested by many participants, we remove several outliers
mydata.drop(mydata[(mydata['OverallQual']<5) & (mydata['SalePrice']>200000)].index, inplace=True)
mydata.drop(mydata[(mydata['GrLivArea']>4000) & (mydata['SalePrice']<300000)].index, inplace=True)
mydata.reset_index(drop=True, inplace=True)

# Some of the non-numeric predictors are stored as numbers; we convert them into strings 
mydata['MSSubClass'] = mydata['MSSubClass'].apply(str)
mydata['YrSold'] = mydata['YrSold'].astype(str)
mydata['MoSold'] = mydata['MoSold'].astype(str)


# In[13]:


# ------------ Function to fill in missings ------------ 

# In[14]:


# Here we create funtion which fills all the missing values
# Pay attention that some of the missing values of numeric predictors first are filled in with zeros and then 
# small values are filled in with median/average (and indicator variables are created to account for such change: 
# for each variable we create  which are equal to one);

def fill_missings(res):

    res['Alley'] = res['Alley'].fillna('missing')
    res['PoolQC'] = res['PoolQC'].fillna(res['PoolQC'].mode()[0])
    res['MasVnrType'] = res['MasVnrType'].fillna('None')
    res['BsmtQual'] = res['BsmtQual'].fillna(res['BsmtQual'].mode()[0])
    res['BsmtCond'] = res['BsmtCond'].fillna(res['BsmtCond'].mode()[0])
    res['FireplaceQu'] = res['FireplaceQu'].fillna(res['FireplaceQu'].mode()[0])
    res['GarageType'] = res['GarageType'].fillna('missing')
    res['GarageFinish'] = res['GarageFinish'].fillna(res['GarageFinish'].mode()[0])
    res['GarageQual'] = res['GarageQual'].fillna(res['GarageQual'].mode()[0])
    res['GarageCond'] = res['GarageCond'].fillna('missing')
    res['Fence'] = res['Fence'].fillna('missing')
    res['Street'] = res['Street'].fillna('missing')
    res['LotShape'] = res['LotShape'].fillna('missing')
    res['LandContour'] = res['LandContour'].fillna('missing')
    res['BsmtExposure'] = res['BsmtExposure'].fillna(res['BsmtExposure'].mode()[0])
    res['BsmtFinType1'] = res['BsmtFinType1'].fillna('missing')
    res['BsmtFinType2'] = res['BsmtFinType2'].fillna('missing')
    res['CentralAir'] = res['CentralAir'].fillna('missing')
    res['Electrical'] = res['Electrical'].fillna(res['Electrical'].mode()[0])
    res['MiscFeature'] = res['MiscFeature'].fillna('missing')
    res['MSZoning'] = res['MSZoning'].fillna(res['MSZoning'].mode()[0])    
    res['Utilities'] = res['Utilities'].fillna('missing')
    res['Exterior1st'] = res['Exterior1st'].fillna(res['Exterior1st'].mode()[0])
    res['Exterior2nd'] = res['Exterior2nd'].fillna(res['Exterior2nd'].mode()[0])    
    res['KitchenQual'] = res['KitchenQual'].fillna(res['KitchenQual'].mode()[0])
    res["Functional"] = res["Functional"].fillna("Typ")
    res['SaleType'] = res['SaleType'].fillna(res['SaleType'].mode()[0])
    res['SaleCondition'] = res['SaleCondition'].fillna('missing')
    
    flist = ['LotFrontage','LotArea','MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF',
                     'TotalBsmtSF','1stFlrSF','2ndFlrSF','LowQualFinSF','GrLivArea',
                     'BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr','KitchenAbvGr',
                     'TotRmsAbvGrd','Fireplaces','GarageCars','GarageArea','WoodDeckSF','OpenPorchSF',
                     'EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea','MiscVal']
    for fl in flist:
        res[fl] = res[fl].fillna(0)
        
    res['TotalBsmtSF'] = res['TotalBsmtSF'].apply(lambda x: np.exp(6) if x <= 0.0 else x)
    res['2ndFlrSF'] = res['2ndFlrSF'].apply(lambda x: np.exp(6.5) if x <= 0.0 else x)
    res['GarageArea'] = res['GarageArea'].apply(lambda x: np.exp(6) if x <= 0.0 else x)
    res['GarageCars'] = res['GarageCars'].apply(lambda x: 0 if x <= 0.0 else x)
    res['LotFrontage'] = res['LotFrontage'].apply(lambda x: np.exp(4.2) if x <= 0.0 else x)
    res['MasVnrArea'] = res['MasVnrArea'].apply(lambda x: np.exp(4) if x <= 0.0 else x)
    res['BsmtFinSF1'] = res['BsmtFinSF1'].apply(lambda x: np.exp(6.5) if x <= 0.0 else x)
    
      
    return res

# In[15]:


# ------------- Filling in missing values, re-coding ordinal variables -------------

# In[16]:


# Running function to fill in missings
mydata = fill_missings(mydata)
mydata['TotalSF'] = mydata['TotalBsmtSF'] + mydata['1stFlrSF'] + mydata['2ndFlrSF']

# Working with ordinal predictors
def QualToInt(x):
    if(x=='Ex'):
        r = 0
    elif(x=='Gd'):
        r = 1
    elif(x=='TA'):
        r = 2
    elif(x=='Fa'):
        r = 3
    elif(x=='missing'):
        r = 4
    else:
        r = 5
    return r

mydata['ExterQual'] = mydata['ExterQual'].apply(QualToInt)
mydata['ExterCond'] = mydata['ExterCond'].apply(QualToInt)
mydata['KitchenQual'] = mydata['KitchenQual'].apply(QualToInt)
mydata['HeatingQC'] = mydata['HeatingQC'].apply(QualToInt)
mydata['BsmtQual'] = mydata['BsmtQual'].apply(QualToInt)
mydata['BsmtCond'] = mydata['BsmtCond'].apply(QualToInt)
mydata['FireplaceQu'] = mydata['FireplaceQu'].apply(QualToInt)
mydata['GarageQual'] = mydata['GarageQual'].apply(QualToInt)
mydata['PoolQC'] = mydata['PoolQC'].apply(QualToInt)

def SlopeToInt(x):
    if(x=='Gtl'):
        r = 0
    elif(x=='Mod'):
        r = 1
    elif(x=='Sev'):
        r = 2
    else:
        r = 3
    return r

mydata['LandSlope'] = mydata['LandSlope'].apply(SlopeToInt)
mydata['CentralAir'] = mydata['CentralAir'].apply( lambda x: 0 if x == 'N' else 1) 
mydata['Street'] = mydata['Street'].apply( lambda x: 0 if x == 'Pave' else 1) 
mydata['PavedDrive'] = mydata['PavedDrive'].apply( lambda x: 0 if x == 'Y' else 1)

def GFinishToInt(x):
    if(x=='Fin'):
        r = 0
    elif(x=='RFn'):
        r = 1
    elif(x=='Unf'):
        r = 2
    else:
        r = 3
    return r

mydata['GarageFinish'] = mydata['GarageFinish'].apply(GFinishToInt)

def BsmtExposureToInt(x):
    if(x=='Gd'):
        r = 0
    elif(x=='Av'):
        r = 1
    elif(x=='Mn'):
        r = 2
    elif(x=='No'):
        r = 3
    else:
        r = 4
    return r
mydata['BsmtExposure'] = mydata['BsmtExposure'].apply(BsmtExposureToInt)

def FunctionalToInt(x):
    if(x=='Typ'):
        r = 0
    elif(x=='Min1'):
        r = 1
    elif(x=='Min2'):
        r = 1
    else:
        r = 2
    return r

mydata['Functional_int'] = mydata['Functional'].apply(FunctionalToInt)


def HouseStyleToInt(x):
    if(x=='1.5Unf'):
        r = 0
    elif(x=='SFoyer'):
        r = 1
    elif(x=='1.5Fin'):
        r = 2
    elif(x=='2.5Unf'):
        r = 3
    elif(x=='SLvl'):
        r = 4
    elif(x=='1Story'):
        r = 5
    elif(x=='2Story'):
        r = 6  
    elif(x==' 2.5Fin'):
        r = 7          
    else:
        r = 8
    return r

mydata['HouseStyle_int'] = mydata['HouseStyle'].apply(HouseStyleToInt)
mydata['HouseStyle_1st'] = 1*(mydata['HouseStyle'] == '1Story')
mydata['HouseStyle_2st'] = 1*(mydata['HouseStyle'] == '2Story')
mydata['HouseStyle_15st'] = 1*(mydata['HouseStyle'] == '1.5Fin')

def FoundationToInt(x):
    if(x=='PConc'):
        r = 3
    elif(x=='CBlock'):
        r = 2
    elif(x=='BrkTil'):
        r = 1        
    else:
        r = 0
    return r

mydata['Foundation_int'] = mydata['Foundation'].apply(FoundationToInt)

def MasVnrTypeToInt(x):
    if(x=='Stone'):
        r = 3
    elif(x=='BrkFace'):
        r = 2
    elif(x=='BrkCmn'):
        r = 1        
    else:
        r = 0
    return r

mydata['MasVnrType_int'] = mydata['MasVnrType'].apply(MasVnrTypeToInt)

def BsmtFinType1ToInt(x):
    if(x=='GLQ'):
        r = 6
    elif(x=='ALQ'):
        r = 5
    elif(x=='BLQ'):
        r = 4
    elif(x=='Rec'):
        r = 3   
    elif(x=='LwQ'):
        r = 2
    elif(x=='Unf'):
        r = 1        
    else:
        r = 0
    return r

mydata['BsmtFinType1_int'] = mydata['BsmtFinType1'].apply(BsmtFinType1ToInt)
mydata['BsmtFinType1_Unf'] = 1*(mydata['BsmtFinType1'] == 'Unf')
mydata['HasWoodDeck'] = (mydata['WoodDeckSF'] == 0) * 1
mydata['HasOpenPorch'] = (mydata['OpenPorchSF'] == 0) * 1
mydata['HasEnclosedPorch'] = (mydata['EnclosedPorch'] == 0) * 1
mydata['Has3SsnPorch'] = (mydata['3SsnPorch'] == 0) * 1
mydata['HasScreenPorch'] = (mydata['ScreenPorch'] == 0) * 1
mydata['YearsSinceRemodel'] = mydata['YrSold'].astype(int) - mydata['YearRemodAdd'].astype(int)
mydata['Total_Home_Quality'] = mydata['OverallQual'] + mydata['OverallCond']




# In[17]:


# --------------- Adding log-transformed predictors to raw data --------------- 

# In[18]:


def addlogs(res, ls):
    m = res.shape[1]
    for l in ls:
        res = res.assign(newcol=pd.Series(np.log(1.01+res[l])).values)   
        res.columns.values[m] = l + '_log'
        m += 1
    return res

loglist = ['LotFrontage','LotArea','MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF',
                 'TotalBsmtSF','1stFlrSF','2ndFlrSF','LowQualFinSF','GrLivArea',
                 'BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr','KitchenAbvGr',
                 'TotRmsAbvGrd','Fireplaces','GarageCars','GarageArea','WoodDeckSF','OpenPorchSF',
                 'EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea','MiscVal','YearRemodAdd','TotalSF']

mydata = addlogs(mydata, loglist)

# In[19]:


# ----------------- Creating dataset for training: adding dummies, adding numeric predictors -----------------

# In[20]:


def getdummies(res, ls):
    def encode(encode_df):
        encode_df = np.array(encode_df)
        enc = OneHotEncoder()
        le = LabelEncoder()
        le.fit(encode_df)
        res1 = le.transform(encode_df).reshape(-1, 1)
        enc.fit(res1)
        return pd.DataFrame(enc.transform(res1).toarray()), le, enc

    decoder = []
    outres = pd.DataFrame({'A' : []})

    for l in ls:
        cat, le, enc = encode(res[l])
        cat.columns = [l+str(x) for x in cat.columns]
        outres.reset_index(drop=True, inplace=True)
        outres = pd.concat([outres, cat], axis = 1)
        decoder.append([le,enc])     
    
    return (outres, decoder)

catpredlist = ['MSSubClass','MSZoning','LotShape','LandContour','LotConfig',
               'Neighborhood','Condition1','Condition2','BldgType',
               'RoofStyle','RoofMatl','Exterior1st','Exterior2nd',
               'BsmtFinType2','Heating','HouseStyle','Foundation','MasVnrType','BsmtFinType1',
               'Electrical','Functional','GarageType','Alley','Utilities',
               'GarageCond','Fence','MiscFeature','SaleType','SaleCondition','LandSlope','CentralAir',
               'GarageFinish','BsmtExposure','Street']

# Applying function to get dummies
# Saving decoder - function which can be used to transform new data  
res = getdummies(mydata[catpredlist],catpredlist)
df = res[0]
decoder = res[1]

# Adding real valued features
floatpredlist = ['LotFrontage_log',
                 'LotArea_log',
                 'MasVnrArea_log','BsmtFinSF1_log','BsmtFinSF2_log','BsmtUnfSF_log',
                 'TotalBsmtSF_log','1stFlrSF_log','2ndFlrSF_log','LowQualFinSF_log','GrLivArea_log',
                 'BsmtFullBath_log','BsmtHalfBath_log','FullBath_log','HalfBath_log','BedroomAbvGr_log','KitchenAbvGr_log',
                 'TotRmsAbvGrd_log','Fireplaces_log','GarageCars_log','GarageArea_log',
                 'PoolArea_log','MiscVal_log',
                 'YearRemodAdd','TotalSF_log','OverallQual','OverallCond','ExterQual','ExterCond','KitchenQual',
                 'HeatingQC','BsmtQual','BsmtCond','FireplaceQu','GarageQual','PoolQC','PavedDrive',
                 'HasWoodDeck', 'HasOpenPorch','HasEnclosedPorch', 'Has3SsnPorch', 'HasScreenPorch']
df = pd.concat([df,mydata[floatpredlist]],axis=1)



# In[21]:


# ----------------- Creating dataset for training: using function which creates squared predictors and adding them to the dataset -----------------

# In[22]:


def addSquared(res, ls):
    m = res.shape[1]
    for l in ls:
        res = res.assign(newcol=pd.Series(res[l]*res[l]).values)   
        res.columns.values[m] = l + '_sq'
        m += 1
    return res 

sqpredlist = ['YearRemodAdd', 'LotFrontage_log', 
              'TotalBsmtSF_log', '1stFlrSF_log', '2ndFlrSF_log', 'GrLivArea_log',
              'GarageCars_log', 'GarageArea_log',
              'OverallQual','ExterQual','BsmtQual','GarageQual','FireplaceQu','KitchenQual']
df = addSquared(df, sqpredlist)


# In[23]:


# ------------- Converting data to numpy array ------------- 

# In[24]:


X = np.array(df)
X = np.delete(X, 0, axis=1)
y = np.log(1+np.array(mydata['SalePrice']))

# In[ ]:


# ------------- Modelling -------------
# 30-fold cross-validation
# Stacking: on each run of cross-validation I fit 5 models (l2, l1, GBR, ENet and LGB)
# Then we make 5 predictions using these models on left-out fold and add geometric mean of these predictions
# Finally, use lasso on these six predictors to forecast values on the left-out fold
# Save all the models (in total we have 30*6=180 models)

# In[27]:


nF = 20

kf = KFold(n_splits=nF, random_state=241, shuffle=True)

test_errors_l2 = []
train_errors_l2 = []
test_errors_l1 = []
train_errors_l1 = []
test_errors_GBR = []
train_errors_GBR = []
test_errors_ENet = []
test_errors_LGB = []
test_errors_stack = []
test_errors_ens = []
train_errors_ens = []

models = []

pred_all = []

ifold = 1

for train_index, test_index in kf.split(X):
    print('fold: ',ifold)
    ifold = ifold + 1
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # ridge
    l2Regr = Ridge(alpha=9.0, fit_intercept = True)
    l2Regr.fit(X_train, y_train)
    pred_train_l2 = l2Regr.predict(X_train)
    pred_test_l2 = l2Regr.predict(X_test)
    
    # lasso
    l1Regr = make_pipeline(RobustScaler(), Lasso(alpha = 0.0003, random_state=1, max_iter=50000))
    l1Regr.fit(X_train, y_train)
    pred_train_l1 = l1Regr.predict(X_train)
    pred_test_l1 = l1Regr.predict(X_test)
    
    # GBR      
    myGBR = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.02,
                                      max_depth=4, max_features='sqrt',
                                      min_samples_leaf=15, min_samples_split=50,
                                      loss='huber', random_state = 5) 
    
    myGBR.fit(X_train,y_train)
    pred_train_GBR = myGBR.predict(X_train)

    pred_test_GBR = myGBR.predict(X_test)
    
    # ENet
    ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=4.0, l1_ratio=0.005, random_state=3))
    ENet.fit(X_train, y_train)
    pred_train_ENet = ENet.predict(X_train)
    pred_test_ENet = ENet.predict(X_test) 
    
    # LGB
    myLGB = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=600,
                              max_bin = 50, bagging_fraction = 0.6,
                              bagging_freq = 5, feature_fraction = 0.25,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf = 6, min_sum_hessian_in_leaf = 11)
    myLGB.fit(X_train, y_train)
    pred_train_LGB = myLGB.predict(X_train)
    pred_test_LGB = myLGB.predict(X_test)      
    
    # Stacking
    stackedset = pd.DataFrame({'A' : []})
    stackedset = pd.concat([stackedset,pd.DataFrame(pred_test_l2)],axis=1)
    stackedset = pd.concat([stackedset,pd.DataFrame(pred_test_l1)],axis=1)
    stackedset = pd.concat([stackedset,pd.DataFrame(pred_test_GBR)],axis=1)
    stackedset = pd.concat([stackedset,pd.DataFrame(pred_test_ENet)],axis=1)
    stackedset = pd.concat([stackedset,pd.DataFrame(pred_test_LGB)],axis=1)
    prod = (pred_test_l2*pred_test_l1*pred_test_GBR*pred_test_ENet*pred_test_LGB) ** (1.0/5.0)
    stackedset = pd.concat([stackedset,pd.DataFrame(prod)],axis=1)
    Xstack = np.array(stackedset)
    Xstack = np.delete(Xstack, 0, axis=1)
    l1_staked = Lasso(alpha = 0.0001,fit_intercept = True)
    l1_staked.fit(Xstack, y_test)
    pred_test_stack = l1_staked.predict(Xstack)
    
    models.append([l2Regr,l1Regr,myGBR,ENet,myLGB,l1_staked])
    
    test_errors_l2.append(np.square(pred_test_l2 - y_test).mean() ** 0.5)
    test_errors_l1.append(np.square(pred_test_l1 - y_test).mean() ** 0.5)
    test_errors_GBR.append(np.square(pred_test_GBR - y_test).mean() ** 0.5)
    test_errors_ENet.append(np.square(pred_test_ENet - y_test).mean() ** 0.5)
    test_errors_LGB.append(np.square(pred_test_LGB - y_test).mean() ** 0.5)
    test_errors_stack.append(np.square(pred_test_stack - y_test).mean() ** 0.5)  
    


# In[28]:


# Output of test set errors; they should be lower then 

# In[29]:


print(np.mean(test_errors_l2))
print(np.mean(test_errors_l1))
print(np.mean(test_errors_GBR))
print(np.mean(test_errors_ENet))
print(np.mean(test_errors_LGB))
print(np.mean(test_errors_stack))

# In[30]:


# 
# ----------------------- Scoring: predictions on the test set -------------------------------
# 

# In[31]:


# reading data
scoredata = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

scoredata['MSSubClass'] = scoredata['MSSubClass'].apply(str)
scoredata['YrSold'] = scoredata['YrSold'].astype(str)
scoredata['MoSold'] = scoredata['MoSold'].astype(str)


# In[32]:


# ------------- Filling in missing values, re-coding ordinal variables -------------

# In[33]:


scoredata = fill_missings(scoredata)

scoredata['ExterQual'] = scoredata['ExterQual'].apply(QualToInt)
scoredata['ExterCond'] = scoredata['ExterCond'].apply(QualToInt)
scoredata['KitchenQual'] = scoredata['KitchenQual'].apply(QualToInt)
scoredata['HeatingQC'] = scoredata['HeatingQC'].apply(QualToInt)
scoredata['BsmtQual'] = scoredata['BsmtQual'].apply(QualToInt)
scoredata['BsmtCond'] = scoredata['BsmtCond'].apply(QualToInt)
scoredata['FireplaceQu'] = scoredata['FireplaceQu'].apply(QualToInt)
scoredata['GarageQual'] = scoredata['GarageQual'].apply(QualToInt)
scoredata['PoolQC'] = scoredata['PoolQC'].apply(QualToInt)
scoredata['LandSlope'] = scoredata['LandSlope'].apply(SlopeToInt)
scoredata['CentralAir'] = scoredata['CentralAir'].apply( lambda x: 0 if x == 'N' else 1) 
scoredata['Street'] = scoredata['Street'].apply( lambda x: 0 if x == 'Grvl' else 1) 
scoredata['GarageFinish'] = scoredata['GarageFinish'].apply(GFinishToInt)
scoredata['BsmtExposure'] = scoredata['BsmtExposure'].apply(BsmtExposureToInt)

scoredata['TotalSF'] = scoredata['TotalBsmtSF'] + scoredata['1stFlrSF'] + scoredata['2ndFlrSF']
scoredata['TotalSF'] = scoredata['TotalSF'].fillna(0)

scoredata['Functional_int'] = scoredata['Functional'].apply(FunctionalToInt)
scoredata['HouseStyle_int'] = scoredata['HouseStyle'].apply(HouseStyleToInt)
scoredata['HouseStyle_1st'] = 1*(scoredata['HouseStyle'] == '1Story')
scoredata['HouseStyle_2st'] = 1*(scoredata['HouseStyle'] == '2Story')
scoredata['HouseStyle_15st'] = 1*(scoredata['HouseStyle'] == '1.5Fin')
scoredata['Foundation_int'] = scoredata['Foundation'].apply(FoundationToInt)
scoredata['MasVnrType_int'] = scoredata['MasVnrType'].apply(MasVnrTypeToInt)
scoredata['BsmtFinType1_int'] = scoredata['BsmtFinType1'].apply(BsmtFinType1ToInt)
scoredata['BsmtFinType1_Unf'] = 1*(scoredata['BsmtFinType1'] == 'Unf')
scoredata['PavedDrive'] = scoredata['PavedDrive'].apply( lambda x: 0 if x == 'Y' else 1)

scoredata['HasWoodDeck'] = (scoredata['WoodDeckSF'] == 0) * 1
scoredata['HasOpenPorch'] = (scoredata['OpenPorchSF'] == 0) * 1
scoredata['HasEnclosedPorch'] = (scoredata['EnclosedPorch'] == 0) * 1
scoredata['Has3SsnPorch'] = (scoredata['3SsnPorch'] == 0) * 1
scoredata['HasScreenPorch'] = (scoredata['ScreenPorch'] == 0) * 1
scoredata['Total_Home_Quality'] = scoredata['OverallQual'] + scoredata['OverallCond']


# In[34]:


# --------------- Changing newly appeared values for some predictors --------------- 

# In[35]:



scoredata['MSSubClass'] = scoredata['MSSubClass'].apply(lambda x: '20' if x == '150' else x)
scoredata['MSZoning'] = scoredata['MSZoning'].apply(lambda x: 'RL' if x == 'missing' else x)
scoredata['Utilities'] = scoredata['Utilities'].apply(lambda x: 'AllPub' if x == 'missing' else x)
scoredata['Exterior1st'] = scoredata['Exterior1st'].apply(lambda x: 'VinylSd' if x == 'missing' else x)
scoredata['Exterior2nd'] = scoredata['Exterior2nd'].apply(lambda x: 'VinylSd' if x == 'missing' else x)
scoredata['Functional'] = scoredata['Functional'].apply(lambda x: 'Typ' if x == 'missing' else x)
scoredata['SaleType'] = scoredata['SaleType'].apply(lambda x: 'WD' if x == 'missing' else x)
scoredata['SaleCondition'] = scoredata['SaleCondition'].apply(lambda x: 'Normal' if x == 'missing' else x)



# In[36]:


# --------------- Adding log-transformed predictors to raw data --------------- 

# In[37]:


scoredata = addlogs(scoredata, loglist)

# In[38]:


# ----------------- Creating dataset for training: dummies, adding numeric variables, adding squared predictors ------

# In[39]:


def getdummies_transform(res, ls, decoder):
    def encode(encode_df, le_df, enc_df):
        encode_df = np.array(encode_df)
        res1 = le_df.transform(encode_df).reshape(-1, 1)
        return pd.DataFrame(enc_df.transform(res1).toarray())
    
    L = len(ls)
    outres = pd.DataFrame({'A' : []})

    for j in range(L):
        l = ls[j]
        le = decoder[j][0]
        enc = decoder[j][1]
        cat = encode(res[l], le, enc)
        cat.columns = [l+str(x) for x in cat.columns]
        outres.reset_index(drop=True, inplace=True)
        outres = pd.concat([outres, cat], axis = 1)
    
    return outres

df_scores = getdummies_transform(scoredata, catpredlist, decoder)
df_scores = pd.concat([df_scores,scoredata[floatpredlist]],axis=1)
df_scores = addSquared(df_scores, sqpredlist)



# In[40]:


# Converting data into numpy array

# In[41]:


X_score = np.array(df_scores)
X_score = np.delete(X_score, 0, axis=1)

# In[42]:


# Scoring data

# In[43]:


M = X_score.shape[0]
scores_fin = 1+np.zeros(M)

for md in models:
    l2 = md[0]
    l1 = md[1]
    GBR = md[2]
    ENet = md[3]
    LGB = md[4]
    l1_stacked = md[5]
    
    l2_scores = l2.predict(X_score)
    l1_scores = l1.predict(X_score)
    GBR_scores = GBR.predict(X_score)
    ENet_scores = ENet.predict(X_score)
    LGB_scores = LGB.predict(X_score)
    
    stackedsets = pd.DataFrame({'A' : []})
    stackedsets = pd.concat([stackedsets,pd.DataFrame(l2_scores)],axis=1)
    stackedsets = pd.concat([stackedsets,pd.DataFrame(l1_scores)],axis=1)
    stackedsets = pd.concat([stackedsets,pd.DataFrame(GBR_scores)],axis=1)
    stackedsets = pd.concat([stackedsets,pd.DataFrame(ENet_scores)],axis=1)
    stackedsets = pd.concat([stackedsets,pd.DataFrame(LGB_scores)],axis=1)
    prod = (l2_scores*l1_scores*GBR_scores*ENet_scores*LGB_scores) ** (1.0/5.0)
    stackedsets = pd.concat([stackedsets,pd.DataFrame(prod)],axis=1)    
    Xstacks = np.array(stackedsets)
    Xstacks = np.delete(Xstacks, 0, axis=1)
    scores_fin = scores_fin * l1_stacked.predict(Xstacks)
scores_fin = scores_fin ** (1/nF)
    

# In[44]:


# Reading predictions obtained from running MICE and SVM in R  
# Use R code provided below to get predictions 
# That is not my code, all the credit goes to https://www.kaggle.com/couyang/svm-benchmark-approach-0-11820-lb-top-13.

# In[45]:


#   library(tidyverse)
#   library(mice)
#   library(e1071)
#   library(Metrics)
#   library(randomForest)
#   library(glmnet)
  
#   # Reading data
#     train <- read.csv("D:/python/House_Prices_train.csv", stringsAsFactors = F, sep=',')
#     test <- read.csv("D:/python/House_Prices_test.csv", stringsAsFactors = F, sep=',')
    
#   # Combining test and train data 
#     full <- bind_rows(train,test)
#     SalePrice <- train$SalePrice
#     N <- length(SalePrice)
#     Id <- test$Id
#     full[,c('Id','SalePrice')] <- NULL
#     rm(train,test)
    
#   # Converting predictors to factor or integer
#     chr <- full[,sapply(full,is.character)]
#     int <- full[,sapply(full,is.integer)]
#     fac <- chr %>% lapply(as.factor) %>% as.data.frame()
#     full <- bind_cols(fac,int)
#   # Running MICE based on random forest 
#     micemod <- full %>% mice(method='rf')
#     full <- complete(micemod)
#   # Saving train and test sets
#     train <- full[1:N,]
#     test<-full[(N+1):nrow(full),]
#   # Adding dependent variable
#     train <- cbind(train,SalePrice)

#   # Modelling: SVM
#     svm_model <- svm(SalePrice~., data=train, cost = 3.2)
#     svm_pred_train <- predict(svm_model,newdata = train)
#     sqrt(mean((log(svm_pred_train)-log(train$SalePrice))^2))
#     svm_pred <- predict(svm_model,newdata = test)
    
#   # Writing final predictions to CSV file
#     solution <- data.frame(Id=Id,SalePrice=svm_pred)
#     write.csv(solution,"D:/python/svm_solution_32.csv",row.names = F)


# In[46]:


svm_solution = pd.read_csv('../input/svm-solution-32/svm_solution_32.csv')
svm_solution_ln = np.log(svm_solution['SalePrice'])

# Averaging stacked and SVM predictions
fin_score = np.sqrt(scores_fin * svm_solution_ln)

Id = scoredata['Id']
fin_score = pd.DataFrame({'SalePrice': np.exp(fin_score)-1})
fin_data = pd.concat([Id,fin_score],axis=1)

# In[47]:


# Brutal approach to deal with predictions close to outer range 

# In[48]:


q1 = fin_data['SalePrice'].quantile(0.0042)
q2 = fin_data['SalePrice'].quantile(0.99)

fin_data['SalePrice'] = fin_data['SalePrice'].apply(lambda x: x if x > q1 else x*0.77)
fin_data['SalePrice'] = fin_data['SalePrice'].apply(lambda x: x if x < q2 else x*1.1)


# In[49]:


# Writing dataset for submission 

# In[50]:


fin_data.to_csv('House_Prices_submit.csv', sep=',', index = False)

# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



