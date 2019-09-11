#!/usr/bin/env python
# coding: utf-8

# ##### Libraries

# In[1]:





import pandas as pd
import numpy as np


import time
import seaborn as sns

#from scipy.stats import entropy
from matplotlib import pyplot as plt




from sklearn.metrics import confusion_matrix, matthews_corrcoef, make_scorer, f1_score, accuracy_score, roc_auc_score

from sklearn.model_selection import train_test_split , GridSearchCV, StratifiedKFold


from sklearn.ensemble import  RandomForestClassifier

from time import gmtime, strftime

from datetime import datetime, timedelta

from sklearn.linear_model import LinearRegression, LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from scipy.stats import skew

#from pandas.tools.plotting import scatter_matrix

import scipy.stats as stats

from sklearn import preprocessing

from sklearn.feature_selection import SelectKBest, chi2, f_classif

import lightgbm as lgb

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials


import os

mingw_path = 'C:\\Program Files\\mingw-w64\\x86_64-8.1.0-posix-seh-rt_v6-rev0\\mingw64\\bin'#mandatory for xgboost


os.environ['PATH'] = mingw_path + ';' + os.environ['PATH']

import xgboost as xgb

from xgboost import plot_importance

from sklearn.decomposition import PCA

import gc 

# ###### Some functions to make life easier

# In[2]:



    
    
def duplicated_col(df):
    
    # Duplicates user id
    dupcol = df.columns
    
    for col in dupcol:
    
        print(df.shape[0] -df[col].drop_duplicates().shape[0], 'duplicated  in column %s'%col)
  
def missing(df):
    # Missing values

    missing = df.isnull().sum()

    missing = missing[missing > 0]
    missing.sort_values(inplace=True)

    print('missing values', missing*100/df.shape[0], '*******************')
    
    if not missing.empty:
        (missing*100/df.shape[0]).plot.bar(title = 'Percentage of Missing Values')
    
def unique_val(df):
    #distinct values
    for col in df.columns: 
    
        print( col, '----->',  df[col].nunique(), 'unique values')
        
 
def univariate(df,col,vartype,hue =None):
    
    '''
    Univariate function will plot the graphs based on the parameters.
    df      : dataframe name
    col     : Column name
    vartype : variable type : continuos or categorical
                Continuos(0)   : Distribution, Violin & Boxplot will be plotted.
                Categorical(1) : Countplot will be plotted.
    hue     : It's only applicable for categorical analysis.
    
    '''
    sns.set(style="darkgrid")
    
    if vartype == 0:
        fig, ax=plt.subplots(nrows =1,ncols=3,figsize=(20,8))
        ax[0].set_title("Distribution Plot")
        sns.distplot(df[col],ax=ax[0])
        ax[1].set_title("Violin Plot")
        sns.violinplot(data =df, x=col,ax=ax[1], inner="quartile")
        ax[2].set_title("Box Plot")
        sns.boxplot(data =df, x=col,ax=ax[2],orient='v')
    
    if vartype == 1:
        temp = pd.Series(data = hue)
        fig, ax = plt.subplots(figsize = (12, 4))
        width = len(df[col].unique()) + 6 + 4*len(temp.unique())
        #fig.set_size_inches(width , 7)
        ax = sns.countplot(data = df, x= col, order=df[col].value_counts().iloc[:15].index,hue = hue) 
        if len(temp.unique()) > 0:
            for p in ax.patches:
                ax.annotate('{:1.1f}%'.format((p.get_height()*100)/float(len(df))), (p.get_x()+0.05, p.get_height()+20))  
        else:
            for p in ax.patches:
                ax.annotate(p.get_height(), (p.get_x()+0.32, p.get_height()+20)) 
        del temp
    else:
        exit
        
    plt.show()
        

# 

# ###### Reading the data

# In[3]:




train_df = pd.read_csv('../input/train.csv')



test_df = pd.read_csv('../input/test.csv')



# In[ ]:




# In[4]:


train_df.head(3)

# In[5]:


test_df.head(3)

# In[6]:


train_df.columns

# In[ ]:




# In[7]:


train_df.describe()

# In[ ]:




# In[8]:


train_df.dtypes

# 
# 
# 

# ###### Assignimg the right data type

# Categorical variables

# In[9]:


catList = ['id', 'wheezy-copper-turtle-magic' ,'target'
          ]

for col in catList:
    
    train_df[col] = train_df[col].astype('str')
    
    if col =='target':
        continue
    test_df[col] = test_df[col].astype('str')


# 

# 

# In[ ]:




# Countinuous variable names 

# In[10]:


countList =  list (set(train_df.columns) - set(catList))

# In[11]:


len(catList), len(countList)

# ###### Target variable distribution 

# In[12]:




fig  =  plt.figure(figsize=(15,  7))

train_df.target.value_counts().plot(kind="bar")

plt.title('Histogram of target')

# In[13]:


train_df.target.value_counts()/train_df.target.count()

# 
# 
# The target variable is fairly well balanced

# ###### Exploring  Data for Preprocessing (missing, duplicates, distinct values)

# ###### missing values

# In[14]:


#missing values
missing(train_df)

# In[15]:


missing(test_df)

# No missing values found in the training and test set

# ###### Duplicated values

# In[16]:


print(train_df.shape,  train_df.drop_duplicates().shape)

# In[17]:


duplicated_col(train_df)

# In[18]:


#df.drop_duplicates(subset= colist, inplace=True)

# ###### Distinct values

# In[19]:


unique_val(train_df)

# In[20]:


#for col in colList:

#df.drop(columns= [col], axis=1, inplace= True)

# ###### Exploring continuous variables

# ######  Variable Summary

# In[21]:


train_df[countList].describe()

# ###### Variable distribution and skewness

# In[22]:


"""



skew_col = []

for col in countList:
    

   if skew(train_df.dropna(subset=[col])[col]) > 0.75: 
      
        skew_col.append(col)
        
  


   plt.figure(figsize=(7,5))
   plt.title("Distribution of %s with %0.02f skewness"%(col, skew(train_df.dropna(subset=[col])[col])))      
   sns.distplot(train_df.dropna(subset=[col])[col])
"""

# In[24]:


#skew_col

# The countinuous variables are not skewed 

# In[ ]:




# ###### Variable correlation (Skipped too expensive to plot)

# In[25]:


"""
#correlation map
f,ax = plt.subplots(figsize=(30, 30))
sns.heatmap(train_df[countList].corr(method='spearman'), annot=True, linewidths=.5, fmt= '.2f', cmap='coolwarm', ax=ax)
"""


# 

# Continuous variable relationship (also skipped)

# In[26]:



"""
tic = time.clock()

fig = plt.figure(1, figsize=(15, 15))

ax = fig.gca()
scatter_matrix(df, alpha=0.3,diagonal='kde', ax = ax)
plt.show()

plt.close()
print ('elapsed time', time.time() -tic)
"""


# ###### Exploring categorical variables

# In[27]:


catList

# In[28]:


fig = plt.figure(1, figsize=(15, 6))

train_df['wheezy-copper-turtle-magic'].value_counts().plot(kind = 'bar')

plt.xlabel('wheezy-copper-turtle-magic')
plt.ylabel('count')
plt.title('wheezy-copper-turtle-magic count plot')

plt.close

# In[29]:


nws_un = train_df['wheezy-copper-turtle-magic'].nunique()

print(nws_un)

# In[30]:




fig = plt.figure(1, figsize=(15, 6))

train_df['wheezy-copper-turtle-magic'].value_counts()[:50].plot(kind = 'bar')

plt.xlabel('wheezy-copper-turtle-magic')
plt.ylabel('count')
plt.title('wheezy-copper-turtle-magic count plot (top 50)')

plt.close

# In[31]:




fig = plt.figure(1, figsize=(15, 6))

train_df['wheezy-copper-turtle-magic'].value_counts()[nws_un-50:].plot(kind = 'bar')

plt.xlabel('wheezy-copper-turtle-magic')
plt.ylabel('count')
plt.title('wheezy-copper-turtle-magic count plot (bottom 50)')

plt.close

# wheezy-copper-turtle-magic is fairly balanced

# Checking if test categories are not in train categories 

# In[32]:


[i for i in test_df['wheezy-copper-turtle-magic'].unique() if i not in train_df['wheezy-copper-turtle-magic'].unique()]

# ###### Investigating variable importance

# Continuous variables
# 
# 

# Plotting the distribution of each continuous variable against each target category

# In[33]:


"""
for col in countList:

    g = sns.FacetGrid(train_df[[col]+ ['target']],  hue ='target', size = 4, aspect = 2.0) #, hist = False, kde= True,   label = 'churn')
    g.map(sns.distplot, col, hist = False, 
          kde_kws = {'shade': True, 'linewidth': 3}).set_axis_labels(col,"density").add_legend()
          
"""          

#  ANOVA test whether the mean of some numeric variable differs across the levels of one categorical variable. It essentially answers the question: do any of the group means differ from one another?

# Top nf=50 features 

# In[34]:


nf = 50 
Fvalue_selector=SelectKBest(f_classif,k=len(countList))
feature_kbest=Fvalue_selector.fit_transform(train_df[countList],train_df['target'])
df_Fvalue=pd.DataFrame(Fvalue_selector.scores_,columns=['F-value'])
df_Fvalue['columns']=countList


# In[35]:


df_Fvalue_s=df_Fvalue.sort_values(by='F-value',ascending=False)[:nf]

plt.figure(figsize=(40,10))
plt.title("F-value for continuous features (top %s features)"%nf,fontsize=30)
plt.xlabel("Continuous Features",fontsize=30)
plt.ylabel("F-value statistics",fontsize=30)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.bar(range(len(df_Fvalue_s)),df_Fvalue_s['F-value'],align='edge',color='rgbkymc')
plt.xticks(range(len(df_Fvalue_s)),df_Fvalue_s['columns'],rotation=90,color='g')
plt.show()

# Bottom nf=50 features

# In[36]:


df_Fvalue_s=df_Fvalue.sort_values(by='F-value',ascending=False)[len(df_Fvalue)-nf:]

plt.figure(figsize=(40,10))
plt.title("F-value for continuous features (bottom %s features)"%nf,fontsize=30)
plt.xlabel("Continuous Features",fontsize=30)
plt.ylabel("F-value statistics",fontsize=30)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.bar(range(len(df_Fvalue_s)),df_Fvalue_s['F-value'],align='edge',color='rgbkymc')
plt.xticks(range(len(df_Fvalue_s)),df_Fvalue_s['columns'],rotation=90,color='g')
plt.show()

# High test statistics means the variable is important

# 
# 

# In[ ]:




# One hot encoding of the categorical variable 

# In[37]:


catList

# In[38]:


catList[1]

# In[39]:


#converting  categorical features into dummies 


cat_train=pd.DataFrame()
cat_test =pd.DataFrame() 
col = catList[1]


dummy=pd.get_dummies(train_df[col].append(test_df[col]), prefix= 'whz')

cat_train=pd.concat([cat_train,dummy[:train_df.shape[0]]],axis=1)

cat_test=pd.concat([cat_test,dummy[train_df.shape[0]:]],axis=1)




# In[40]:


train_df.shape, test_df.shape, cat_train.shape, cat_test.shape

# In[41]:


cat_train.head()

# 

# In[42]:


kf =30
chi2_selector=SelectKBest(chi2,k=kf)
feature_kbest=chi2_selector.fit_transform(cat_train, train_df['target'])

df_chi=pd.DataFrame(chi2_selector.scores_, columns=['chi_score'])
df_chi['columns']=cat_train.columns


# Top kf= 30 categories ranked by importance

# In[43]:


kf = 30 

df_chi_s=df_chi.sort_values(by='chi_score')[:kf] 

fig,ax=plt.subplots(figsize=(20,40))
plt.title("Chi-squared statistics for categorical features (top %s)"%kf,fontsize=30)
plt.ylabel("Categorical Features",fontsize=30)
plt.xlabel("Chi-squared statistic",fontsize=30)
plt.barh(range(len(df_chi_s['chi_score'])),df_chi_s['chi_score'],align='edge',color='rgbkymc')
plt.yticks(range(len(df_chi_s['chi_score'])),df_chi_s['columns'],color='g',fontsize=15)
for i in range(0,kf,2):
    ax.get_yticklabels()[i].set_color("red")
plt.show()

# In[44]:


train_df =pd.concat([train_df, cat_train], axis = 1).drop(columns=['wheezy-copper-turtle-magic'], axis = 1)

test_df =pd.concat([test_df, cat_test], axis = 1).drop(columns=['wheezy-copper-turtle-magic'], axis = 1)

# In[45]:


train_df.shape, test_df.shape

# Dimensionality reduction (PCA)

# In[46]:


countList = list(cat_train.columns) + countList

# In[47]:



gc.collect() 

# ##### Dimensionality reduction PCA

# In[ ]:



pca = PCA(n_components=500, random_state= 1234) #200-0.5, 400-0.67

pca.fit(train_df[countList])
pc = pca.transform(train_df[countList])

pctest = pca.transform(test_df[countList])


pc_new = pd.DataFrame(pc)
pc_new.index = train_df.index

pct_new = pd.DataFrame(pctest)
pct_new.index = test_df.index

train_df_pca = pd.concat([train_df[['id', 'target']], pc_new], axis = 1)

test_df_pca = pd.concat([test_df[['id']], pct_new], axis = 1)

train_df_pca.shape, test_df_pca.shape



# In[ ]:


train_df_pca.head()


# In[ ]:


del pc, pc_new, pctest, pct_new

gc.collect()

# In[ ]:


len(train_df_pca)

# ######Splitting the data train and val

# In[ ]:



VALID_SIZE = 0.1 #keeping 10 percent of the data for testing 

# Extract the train and valid (used for validation) dataframes from the train_df

SEED =1234

train, valid = train_test_split( train_df_pca, stratify = train_df_pca.target, test_size=VALID_SIZE, random_state=SEED) 


mylist = ['target', 'id']

train_features = train[train_df_pca.drop(mylist, axis=1).columns]

valid_features = valid[train_df_pca.drop(mylist, axis=1).columns]


y_train = train['target']

y_valid = valid['target']

# In[ ]:




# In[ ]:



num_folds = 5

folds = StratifiedKFold(n_splits=num_folds, shuffle=False, random_state=2319)


# In[ ]:


predictions_lgb = np.zeros(len(valid_features))

tr_predictions_lgb = np.zeros(len(train_features))

test_df['target_lgb'] = 0 

# In[ ]:


params_lgb = {'nthread': 4, 'metric': 'auc', 'colsample_bytree': 0.8500000000000001,
            'max_depth': 5, 'seed': 1, 'silent': 1, 'objective': 'binary',
            'min_child_weight': 6.0, 'n_estimators': 5000.0, 'eta': 0.1,
            'boosting_type': 'gbdt', 'subsample': 0.9500000000000001}

# In[ ]:


num_rounds = int(params_lgb['n_estimators']) 

for  fold_, (train_index, test_index )  in enumerate(folds.split(train_features, y_train)): 
    

       evals_results = {}
    
       dtrain = lgb.Dataset(train_features.iloc[train_index], label=y_train.iloc[train_index])
       dvalid =  lgb.Dataset(train_features.iloc[test_index], label=y_train.iloc[test_index])
    
       bst1 = lgb.train(params_lgb, 

                     dtrain, 

                     valid_sets=[dtrain ,dvalid], 

                     valid_names=['train', 'valid'], 

                     evals_result=evals_results, 

                     num_boost_round= num_rounds,

                     early_stopping_rounds=10,

                     verbose_eval=10 

                       )

        

       predictions_lgb =  predictions_lgb + bst1.predict(valid_features, num_iteration=bst1.best_iteration) /num_folds
    
       tr_predictions_lgb =  tr_predictions_lgb + bst1.predict(train_features, num_iteration=bst1.best_iteration) /num_folds
       
       test_df['target_lgb']  = test_df['target_lgb'] + bst1.predict(test_df_pca.drop(['id'], axis=1),num_iteration=bst1.best_iteration) /num_folds
        
        
        

# In[ ]:


print('Overall accuracy',
             'test_auc', 
             roc_auc_score(y_valid, predictions_lgb), 
             'train_auc',  roc_auc_score(y_train,tr_predictions_lgb))

# In[ ]:




ax = lgb.plot_importance(bst1, max_num_features=25) 
fig = ax.figure
fig.set_size_inches(10, 10)

# 
# 
# 

# #### Make submission

# In[ ]:


submit = pd.read_csv('../input/sample_submission.csv')
submit["target"] = test_df['target_lgb']
submit.to_csv("submission.csv", index=False)
