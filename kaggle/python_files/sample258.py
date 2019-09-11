#!/usr/bin/env python
# coding: utf-8

# - <a href='#1'>Prepare</a>  
# - <a href='#2'>Feature Selection</a>
#     - <a href='#2-1'>1. Filter</a>
#         - <a href='#2-1-1'>1.1 Pearson Correlation</a>
#         - <a href='#2-1-2'>1.2 Chi-2</a>
#     - <a href='#2-2'>2. Wrapper</a>
#     - <a href='#2-3'>3. Embeded</a>
#         - <a href='#2-3-1'>3.1 Logistics Regression L1</a>
#         - <a href='#2-3-2'>3.2 Random Forest</a>
#         - <a href='#2-3-3'>3.3 LightGBM</a>
# - <a href='#3'>Summary</a>

# # <a id='1'>Prepare</a>

# In[ ]:


import pandas as pd
import numpy as np
import gc
import warnings
warnings.filterwarnings("ignore")
application_train = pd.read_csv('../input/application_train.csv')

# ### Stratified Sampling (ratio = 0.1)

# In[ ]:


application_sample1 = application_train.loc[application_train.TARGET==1].sample(frac=0.1, replace=False)
print('label 1 sample size:', str(application_sample1.shape[0]))
application_sample0 = application_train.loc[application_train.TARGET==0].sample(frac=0.1, replace=False)
print('label 0 sample size:', str(application_sample0.shape[0]))
application = pd.concat([application_sample1, application_sample0], axis=0).sort_values('SK_ID_CURR')

# ### Impute missing values

# In[ ]:


categorical_list = []
numerical_list = []
for i in application.columns.tolist():
    if application[i].dtype=='object':
        categorical_list.append(i)
    else:
        numerical_list.append(i)
print('Number of categorical features:', str(len(categorical_list)))
print('Number of numerical features:', str(len(numerical_list)))

# In[ ]:


from sklearn.preprocessing import Imputer
application[numerical_list] = Imputer(strategy='median').fit_transform(application[numerical_list])

# ### Deal with Categorical features: OneHotEncoding

# In[ ]:


del application_train; gc.collect()
application = pd.get_dummies(application, drop_first=True)
print(application.shape)

# ### Feature matrix and target

# In[ ]:


X = application.drop(['SK_ID_CURR', 'TARGET'], axis=1)
y = application.TARGET
feature_name = X.columns.tolist()

# # <a id='2'>Feature Selection</a>
# - select **100** features from 226
# - **xxx_support**: list to represent select this feature or not
# - **xxx_feature**: the name of selected features

# ## <a id='2-1'>1 Filter</a>
# - documentation for **SelectKBest**: http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html
# 
# ###  <a id='2-1-1'>1.1 Pearson Correlation</a>
# **Note**
# - Normalization: no
# - Impute missing values: yes

# In[ ]:


def cor_selector(X, y):
    cor_list = []
    # calculate the correlation with y for each feature
    for i in X.columns.tolist():
        cor = np.corrcoef(X[i], y)[0, 1]
        cor_list.append(cor)
    # replace NaN with 0
    cor_list = [0 if np.isnan(i) else i for i in cor_list]
    # feature name
    cor_feature = X.iloc[:,np.argsort(np.abs(cor_list))[-100:]].columns.tolist()
    # feature selection? 0 for not select, 1 for select
    cor_support = [True if i in cor_feature else False for i in feature_name]
    return cor_support, cor_feature

# In[ ]:


cor_support, cor_feature = cor_selector(X, y)
print(str(len(cor_feature)), 'selected features')

# ###  <a id='2-1-2'>1.2 Chi-2</a>
# 
# **Note**
# - Normalization: MinMaxScaler (values should be bigger than 0)
# - Impute missing values: yes

# In[ ]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import MinMaxScaler
X_norm = MinMaxScaler().fit_transform(X)
chi_selector = SelectKBest(chi2, k=100)
chi_selector.fit(X_norm, y)

# In[ ]:


chi_support = chi_selector.get_support()
chi_feature = X.loc[:,chi_support].columns.tolist()
print(str(len(chi_feature)), 'selected features')

# ## <a id='2-2'>2. Wrapper</a>
# - documentation for **RFE**: http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html
# 
# **Note**
# - Normalization: depend on the used model; yes for LR
# - Impute missing values: depend on the used model; yes for LR
# 

# In[ ]:


from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
rfe_selector = RFE(estimator=LogisticRegression(), n_features_to_select=100, step=10, verbose=5)
rfe_selector.fit(X_norm, y)

# In[ ]:


rfe_support = rfe_selector.get_support()
rfe_feature = X.loc[:,rfe_support].columns.tolist()
print(str(len(rfe_feature)), 'selected features')

# ## <a id='2-3'>3. Embeded</a>
# - documentation for **SelectFromModel**: http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFromModel.html
# ###  <a id='2-3-1'>3.1 Logistics Regression L1</a>
# **Note**
# - Normalization: Yes
# - Impute missing values: Yes

# In[ ]:


from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression

embeded_lr_selector = SelectFromModel(LogisticRegression(penalty="l1"), '1.25*median')
embeded_lr_selector.fit(X_norm, y)

# In[ ]:


embeded_lr_support = embeded_lr_selector.get_support()
embeded_lr_feature = X.loc[:,embeded_lr_support].columns.tolist()
print(str(len(embeded_lr_feature)), 'selected features')

# ###  <a id='2-3-2'>3.2 Random Forest</a>
# **Note**
# - Normalization: No
# - Impute missing values: Yes

# In[ ]:


from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier

embeded_rf_selector = SelectFromModel(RandomForestClassifier(n_estimators=100), threshold='1.25*median')
embeded_rf_selector.fit(X, y)

# In[ ]:


embeded_rf_support = embeded_rf_selector.get_support()
embeded_rf_feature = X.loc[:,embeded_rf_support].columns.tolist()
print(str(len(embeded_rf_feature)), 'selected features')

# ###  <a id='2-3-3'>3.3 LightGBM</a>
# **Note**
# - Normalization: No
# - Impute missing values: No

# In[ ]:


from sklearn.feature_selection import SelectFromModel
from lightgbm import LGBMClassifier

lgbc=LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=32, colsample_bytree=0.2,
            reg_alpha=3, reg_lambda=1, min_split_gain=0.01, min_child_weight=40)

embeded_lgb_selector = SelectFromModel(lgbc, threshold='1.25*median')
embeded_lgb_selector.fit(X, y)

# In[ ]:


embeded_lgb_support = embeded_lgb_selector.get_support()
embeded_lgb_feature = X.loc[:,embeded_lgb_support].columns.tolist()
print(str(len(embeded_lgb_feature)), 'selected features')

# # <a id='3'>Summary</a>

# In[ ]:


pd.set_option('display.max_rows', None)
# put all selection together
feature_selection_df = pd.DataFrame({'Feature':feature_name, 'Pearson':cor_support, 'Chi-2':chi_support, 'RFE':rfe_support, 'Logistics':embeded_lr_support,
                                    'Random Forest':embeded_rf_support, 'LightGBM':embeded_lgb_support})
# count the selected times for each feature
feature_selection_df['Total'] = np.sum(feature_selection_df, axis=1)
# display the top 100
feature_selection_df = feature_selection_df.sort_values(['Total','Feature'] , ascending=False)
feature_selection_df.index = range(1, len(feature_selection_df)+1)
feature_selection_df.head(100)
