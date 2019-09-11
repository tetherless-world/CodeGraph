#!/usr/bin/env python
# coding: utf-8

# ## Microsoft Malware Prediction
# 
# The goal of this competition is to predict a Windows machine’s probability of getting infected by various families of malware, based on different properties of that machine. It is really important to find out whether the computer is infected and cure it.
# 
# We have a huge dataset of data, where most features are categorical. I think that correct mean encoding should be important. Also the number of columns is quite high so it could be tempting to make some automatical processing for all columns. I personally think that it is important to analyze each variable and it could help to do a better processing.
# 
# In this kernel I'll do a detailed EDA, feature engineering and modelling.
# 
# ![](https://storage.googleapis.com/kaggle-competitions/kaggle/10683/logos/thumb76_76.png?t=2018-09-19-16-55-15)

# In[ ]:


#libraries
import numpy as np 
import pandas as pd 
import os
import seaborn as sns 
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import lightgbm as lgb
import xgboost as xgb
import time
import datetime

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, KFold, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
import gc
from catboost import CatBoostClassifier
from tqdm import tqdm_notebook
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

import warnings
warnings.filterwarnings("ignore")

import logging

logging.basicConfig(filename='log.txt',level=logging.DEBUG, format='%(asctime)s %(message)s')

pd.set_option('max_colwidth', 500)
pd.set_option('max_columns', 500)
pd.set_option('max_rows', 100)
import os
print(os.listdir("../input/"))


# ## Loading data
# Let's try loading data in a naive way

# In[ ]:


#https://www.kaggle.com/theoviel/load-the-totality-of-the-data
dtypes = {
        'MachineIdentifier':                                    'category',
        'ProductName':                                          'category',
        'EngineVersion':                                        'category',
        'AppVersion':                                           'category',
        'AvSigVersion':                                         'category',
        'IsBeta':                                               'int8',
        'RtpStateBitfield':                                     'float16',
        'IsSxsPassiveMode':                                     'int8',
        'DefaultBrowsersIdentifier':                            'float16',
        'AVProductStatesIdentifier':                            'float32',
        'AVProductsInstalled':                                  'float16',
        'AVProductsEnabled':                                    'float16',
        'HasTpm':                                               'int8',
        'CountryIdentifier':                                    'int16',
        'CityIdentifier':                                       'float32',
        'OrganizationIdentifier':                               'float16',
        'GeoNameIdentifier':                                    'float16',
        'LocaleEnglishNameIdentifier':                          'int8',
        'Platform':                                             'category',
        'Processor':                                            'category',
        'OsVer':                                                'category',
        'OsBuild':                                              'int16',
        'OsSuite':                                              'int16',
        'OsPlatformSubRelease':                                 'category',
        'OsBuildLab':                                           'category',
        'SkuEdition':                                           'category',
        'IsProtected':                                          'float16',
        'AutoSampleOptIn':                                      'int8',
        'PuaMode':                                              'category',
        'SMode':                                                'float16',
        'IeVerIdentifier':                                      'float16',
        'SmartScreen':                                          'category',
        'Firewall':                                             'float16',
        'UacLuaenable':                                         'float32',
        'Census_MDC2FormFactor':                                'category',
        'Census_DeviceFamily':                                  'category',
        'Census_OEMNameIdentifier':                             'float16',
        'Census_OEMModelIdentifier':                            'float32',
        'Census_ProcessorCoreCount':                            'float16',
        'Census_ProcessorManufacturerIdentifier':               'float16',
        'Census_ProcessorModelIdentifier':                      'float16',
        'Census_ProcessorClass':                                'category',
        'Census_PrimaryDiskTotalCapacity':                      'float32',
        'Census_PrimaryDiskTypeName':                           'category',
        'Census_SystemVolumeTotalCapacity':                     'float32',
        'Census_HasOpticalDiskDrive':                           'int8',
        'Census_TotalPhysicalRAM':                              'float32',
        'Census_ChassisTypeName':                               'category',
        'Census_InternalPrimaryDiagonalDisplaySizeInInches':    'float16',
        'Census_InternalPrimaryDisplayResolutionHorizontal':    'float16',
        'Census_InternalPrimaryDisplayResolutionVertical':      'float16',
        'Census_PowerPlatformRoleName':                         'category',
        'Census_InternalBatteryType':                           'category',
        'Census_InternalBatteryNumberOfCharges':                'float32',
        'Census_OSVersion':                                     'category',
        'Census_OSArchitecture':                                'category',
        'Census_OSBranch':                                      'category',
        'Census_OSBuildNumber':                                 'int16',
        'Census_OSBuildRevision':                               'int32',
        'Census_OSEdition':                                     'category',
        'Census_OSSkuName':                                     'category',
        'Census_OSInstallTypeName':                             'category',
        'Census_OSInstallLanguageIdentifier':                   'float16',
        'Census_OSUILocaleIdentifier':                          'int16',
        'Census_OSWUAutoUpdateOptionsName':                     'category',
        'Census_IsPortableOperatingSystem':                     'int8',
        'Census_GenuineStateName':                              'category',
        'Census_ActivationChannel':                             'category',
        'Census_IsFlightingInternal':                           'float16',
        'Census_IsFlightsDisabled':                             'float16',
        'Census_FlightRing':                                    'category',
        'Census_ThresholdOptIn':                                'float16',
        'Census_FirmwareManufacturerIdentifier':                'float16',
        'Census_FirmwareVersionIdentifier':                     'float32',
        'Census_IsSecureBootEnabled':                           'int8',
        'Census_IsWIMBootEnabled':                              'float16',
        'Census_IsVirtualDevice':                               'float16',
        'Census_IsTouchEnabled':                                'int8',
        'Census_IsPenCapable':                                  'int8',
        'Census_IsAlwaysOnAlwaysConnectedCapable':              'float16',
        'Wdft_IsGamer':                                         'float16',
        'Wdft_RegionIdentifier':                                'float16',
        'HasDetections':                                        'int8'
        }

def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage(deep=True).sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage(deep=True).sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

# In[ ]:


numerics = ['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64']
numerical_columns = [c for c,v in dtypes.items() if v in numerics]
categorical_columns = [c for c,v in dtypes.items() if v not in numerics]

# In[ ]:


train = pd.read_csv('../input/train.csv', dtype=dtypes)

# In[ ]:


train = reduce_mem_usage(train)

# In[ ]:


stats = []
for col in train.columns:
    stats.append((col, train[col].nunique(), train[col].isnull().sum() * 100 / train.shape[0], train[col].value_counts(normalize=True, dropna=False).values[0] * 100, train[col].dtype))
    
stats_df = pd.DataFrame(stats, columns=['Feature', 'Unique_values', 'Percentage of missing values', 'Percentage of values in the biggest category', 'type'])
stats_df.sort_values('Percentage of missing values', ascending=False)

# We can see several interesting things here:
# * PuaMode and Census_ProcessorClass have 99%+ missing values, which means that these columns are useless and should be dropped;
# * In DefaultBrowsersIdentifier column 95% values belong to one category, so I think this columns is also useless;
# * Census_IsFlightingInternal is strange, but maybe analysis of it will make sense;
# * There are 26 columns in total in which one category contains 90% values. I think that these imbalanced columns should be removed from the dataset;
# * One more important point is that there are many columns which are considered to be numerical (from this kernel: #https://www.kaggle.com/theoviel/load-the-totality-of-the-data), but I think they are categorical - like CityIdentifier. We'll see more in EDA below;
# 
# 
# I see that all columns except Census_SystemVolumeTotalCapacity are categorical. 
# Also there are 3 columns, where most of the values are missing. Let's drop them.

# In[ ]:


good_cols = list(train.columns)
for col in train.columns:
    rate = train[col].value_counts(normalize=True, dropna=False).values[0]
    if rate > 0.9:
        good_cols.remove(col)

# In[ ]:


train = train[good_cols]

# Now we can read test data.

# In[ ]:


test_dtypes = {k: v for k, v in dtypes.items() if k in good_cols}
test = pd.read_csv('../input/test.csv', dtype=test_dtypes, usecols=good_cols[:-1])
test.loc[6529507, 'OsBuildLab'] = '17134.1.amd64fre.rs4_release.180410-1804'
test['OsBuildLab'] = test['OsBuildLab'].fillna('17134.1.amd64fre.rs4_release.180410-1804')
test = reduce_mem_usage(test)

# ## Data exploration

# In[ ]:


train.head()

# In[ ]:


# function to plot data
def plot_categorical_feature(col, only_bars=False, top_n=10, by_touch=False):
    top_n = top_n if train[col].nunique() > top_n else train[col].nunique()
    print(f"{col} has {train[col].nunique()} unique values and type: {train[col].dtype}.")
    print(train[col].value_counts(normalize=True, dropna=False).head())
    if not by_touch:
        if not only_bars:
            df = train.groupby([col]).agg({'HasDetections': ['count', 'mean']})
            df = df.sort_values(('HasDetections', 'count'), ascending=False).head(top_n).sort_index()
            data = [go.Bar(x=df.index, y=df['HasDetections']['count'].values, name='counts'),
                    go.Scatter(x=df.index, y=df['HasDetections']['mean'], name='Detections rate', yaxis='y2')]

            layout = go.Layout(dict(title = f"Counts of {col} by top-{top_n} categories and mean target value",
                                xaxis = dict(title = f'{col}',
                                             showgrid=False,
                                             zeroline=False,
                                             showline=False,),
                                yaxis = dict(title = 'Counts',
                                             showgrid=False,
                                             zeroline=False,
                                             showline=False,),
                                yaxis2=dict(title='Detections rate', overlaying='y', side='right')),
                           legend=dict(orientation="v"))

        else:
            top_cat = list(train[col].value_counts(dropna=False).index[:top_n])
            df0 = train.loc[(train[col].isin(top_cat)) & (train['HasDetections'] == 1), col].value_counts().head(10).sort_index()
            df1 = train.loc[(train[col].isin(top_cat)) & (train['HasDetections'] == 0), col].value_counts().head(10).sort_index()
            data = [go.Bar(x=df0.index, y=df0.values, name='Has Detections'),
                    go.Bar(x=df1.index, y=df1.values, name='No Detections')]

            layout = go.Layout(dict(title = f"Counts of {col} by top-{top_n} categories",
                                xaxis = dict(title = f'{col}',
                                             showgrid=False,
                                             zeroline=False,
                                             showline=False,),
                                yaxis = dict(title = 'Counts',
                                             showgrid=False,
                                             zeroline=False,
                                             showline=False,),
                                ),
                           legend=dict(orientation="v"), barmode='group')
        
        py.iplot(dict(data=data, layout=layout))
        
    else:
        top_n = 10
        top_cat = list(train[col].value_counts(dropna=False).index[:top_n])
        df = train.loc[train[col].isin(top_cat)]

        df1 = train.loc[train['Census_IsTouchEnabled'] == 1]
        df0 = train.loc[train['Census_IsTouchEnabled'] == 0]

        df0_ = df0.groupby([col]).agg({'HasDetections': ['count', 'mean']})
        df0_ = df0_.sort_values(('HasDetections', 'count'), ascending=False).head(top_n).sort_index()
        df1_ = df1.groupby([col]).agg({'HasDetections': ['count', 'mean']})
        df1_ = df1_.sort_values(('HasDetections', 'count'), ascending=False).head(top_n).sort_index()
        data1 = [go.Bar(x=df0_.index, y=df0_['HasDetections']['count'].values, name='Nontouch device counts'),
                go.Scatter(x=df0_.index, y=df0_['HasDetections']['mean'], name='Detections rate for nontouch devices', yaxis='y2')]
        data2 = [go.Bar(x=df1_.index, y=df1_['HasDetections']['count'].values, name='Touch device counts'),
                go.Scatter(x=df1_.index, y=df1_['HasDetections']['mean'], name='Detections rate for touch devices', yaxis='y2')]

        layout = go.Layout(dict(title = f"Counts of {col} by top-{top_n} categories for nontouch devices",
                            xaxis = dict(title = f'{col}',
                                         showgrid=False,
                                         zeroline=False,
                                         showline=False,
                                         type='category'),
                            yaxis = dict(title = 'Counts',
                                         showgrid=False,
                                         zeroline=False,
                                         showline=False,),
                                    yaxis2=dict(title='Detections rate', overlaying='y', side='right'),
                            ),
                       legend=dict(orientation="v"), barmode='group')

        py.iplot(dict(data=data1, layout=layout))
        layout['title'] = f"Counts of {col} by top-{top_n} categories for touch devices"
        py.iplot(dict(data=data2, layout=layout))

# ### Target

# In[ ]:


train['HasDetections'].value_counts()

# The target is balanced, which is nice.

# ### Census_IsTouchEnabled

# In[ ]:


plot_categorical_feature('Census_IsTouchEnabled', True)

# As expected Microsoft has much more computers that touch devices. The rate of infections is lower for touch devices, but not by much.

# ### Variables with high amount of unique values
# At first I'll have a look at variables  with a lot of categories. Then I'll move to more interesting variables with a limited amount of categories.

# ## EngineVersion

# In[ ]:


plot_categorical_feature('EngineVersion', by_touch=True)

# We can see that two categories take 84% of all values. The difference in detection rates is noticable. Other categories have different detection rate, but it is simply due to the low number of samples in them.
# Patterns for touch and non-touch devices are quite similar.

# ### AppVersion

# In[ ]:


plot_categorical_feature('AppVersion')

# In this case we have one main category with 0.53 detection rate, other categories are much smaller.

# ### AvSigVersion

# In[ ]:


plot_categorical_feature('AvSigVersion')

# This feature has a huge amount of categories, it seems this is a version of an often updated software.

# ### AVProductStatesIdentifier

# In[ ]:


plot_categorical_feature('AVProductStatesIdentifier', True, 10)

# This is definitely a categorical variable.

# In[ ]:


train['AVProductStatesIdentifier'] = train['AVProductStatesIdentifier'].astype('category')
test['AVProductStatesIdentifier'] = test['AVProductStatesIdentifier'].astype('category')

# ### AVProductsInstalled

# In[ ]:


plot_categorical_feature('AVProductsInstalled', True)

# Hm, interesting. If a computer has an antivirus, it is less likely to be infected. But having two antiviruses has an opposite effect. Maybe those who install 2 antiviruses tend to have less experience with working on PC?
# Also other categories have really low samples, let's combine them.

# In[ ]:


# train.loc[train['AVProductsInstalled'].isin([1, 2]) == False, 'AVProductsInstalled'] = 3
# test.loc[test['AVProductsInstalled'].isin([1, 2]) == False, 'AVProductsInstalled'] = 3

# In[ ]:


train['AVProductsInstalled'] = train['AVProductsInstalled'].astype('category')
test['AVProductsInstalled'] = test['AVProductsInstalled'].astype('category')

# In[ ]:


plot_categorical_feature('AVProductsInstalled', True, by_touch=True)

# It is interesting, that even on touch devices people sometimes install 2 antiviruses.

# ### CountryIdentifier

# In[ ]:


plot_categorical_feature('CountryIdentifier', True, 20)

# Once again we have a categorical column which was defined as numerical. While most countries have rate of detections ~50%, there are some countries where there are much more infected devices.

# In[ ]:


train['CountryIdentifier'] = train['CountryIdentifier'].astype('category')
test['CountryIdentifier'] = test['CountryIdentifier'].astype('category')

# ### CityIdentifier
# The same situation for cities.

# In[ ]:


plot_categorical_feature('CityIdentifier', True, 20)

# In[ ]:


train['CityIdentifier'] = train['CityIdentifier'].astype('category')
test['CityIdentifier'] = test['CityIdentifier'].astype('category')

# ### OrganizationIdentifier

# In[ ]:


plot_categorical_feature('OrganizationIdentifier', True)

# This is quuite strange. 2 organizations cover ~66% of all computers, while unknown organizations have 30% more. Maybe these are some specific industries? Let's combine values.

# In[ ]:


# train.loc[train['OrganizationIdentifier'].isin([27, 18]) == False, 'OrganizationIdentifier'] = 48
# test.loc[test['OrganizationIdentifier'].isin([27, 18]) == False, 'OrganizationIdentifier'] = 48

# In[ ]:


train['OrganizationIdentifier'] = train['OrganizationIdentifier'].astype('category')
test['OrganizationIdentifier'] = test['OrganizationIdentifier'].astype('category')

# In[ ]:


plot_categorical_feature('OrganizationIdentifier', True, by_touch=True)

# 

# ### GeoNameIdentifier

# In[ ]:


plot_categorical_feature('GeoNameIdentifier', True)

# In[ ]:


train['GeoNameIdentifier'] = train['GeoNameIdentifier'].astype('category')
test['GeoNameIdentifier'] = test['GeoNameIdentifier'].astype('category')

# ### LocaleEnglishNameIdentifier

# In[ ]:


plot_categorical_feature('LocaleEnglishNameIdentifier', True)

# In[ ]:


train['LocaleEnglishNameIdentifier'] = train['LocaleEnglishNameIdentifier'].astype('category')
test['LocaleEnglishNameIdentifier'] = test['LocaleEnglishNameIdentifier'].astype('category')

# ### OsPlatformSubRelease

# In[ ]:


plot_categorical_feature('OsPlatformSubRelease', True, by_touch=True)

# It is interesting that most computers have Windows 10 (rs*). I suppose Microsoft specifically chose them, so that we would work with modern devices?

# ### OsBuildLab

# In[ ]:


plot_categorical_feature('OsBuildLab', True)

# ### IeVerIdentifier

# In[ ]:


plot_categorical_feature('IeVerIdentifier', True)

# In[ ]:


train['IeVerIdentifier'] = train['IeVerIdentifier'].astype('category')
test['IeVerIdentifier'] = test['IeVerIdentifier'].astype('category')

# ### Census_OEMNameIdentifier

# In[ ]:


plot_categorical_feature('Census_OEMNameIdentifier', True)

# In[ ]:


train['Census_OEMNameIdentifier'] = train['Census_OEMNameIdentifier'].astype('category')
test['Census_OEMNameIdentifier'] = test['Census_OEMNameIdentifier'].astype('category')

# ### Census_OEMModelIdentifier

# In[ ]:


plot_categorical_feature('Census_OEMModelIdentifier', True)

# In[ ]:


train['Census_OEMModelIdentifier'] = train['Census_OEMModelIdentifier'].astype('category')
test['Census_OEMModelIdentifier'] = test['Census_OEMModelIdentifier'].astype('category')

# ### Census_ProcessorCoreCount

# In[ ]:


plot_categorical_feature('Census_ProcessorCoreCount', True, by_touch=True)

# Most computers have 2, 4 or 8 cores. For touch deviced 4 cores are much more common than other configurations. And these 3 variants cover 95% of all samples,

# In[ ]:


# train.loc[train['Census_ProcessorCoreCount'].isin([2, 4, 8, 12]) == False, 'Census_ProcessorCoreCount'] = 1
# test.loc[test['Census_ProcessorCoreCount'].isin([2, 4, 8, 12]) == False, 'Census_ProcessorCoreCount'] = 1

# ### Census_ProcessorModelIdentifier

# In[ ]:


plot_categorical_feature('Census_ProcessorModelIdentifier', True)

# In[ ]:


train['Census_ProcessorModelIdentifier'] = train['Census_ProcessorModelIdentifier'].astype('category')
test['Census_ProcessorModelIdentifier'] = test['Census_ProcessorModelIdentifier'].astype('category')

# ### Census_PrimaryDiskTotalCapacity

# In[ ]:


plot_categorical_feature('Census_PrimaryDiskTotalCapacity', True)

# ### Census_SystemVolumeTotalCapacity

# In[ ]:


plot_categorical_feature('Census_SystemVolumeTotalCapacity', True)

# I think that this is a first truly numerical variable. Census_PrimaryDiskTotalCapacity could be numerical, but it has too little unique values and every category could be considered a separate disk model.

# ### Census_TotalPhysicalRAM

# In[ ]:


plot_categorical_feature('Census_TotalPhysicalRAM', True, by_touch=True)

# Quite interesting that most computers have <=8 Gb RAM.

# In[ ]:


# top_10 = train['Census_TotalPhysicalRAM'].value_counts(dropna=False, normalize=True).cumsum().index[:10]
# train.loc[train['Census_TotalPhysicalRAM'].isin(top_10) == False, 'Census_TotalPhysicalRAM'] = 1000
# test.loc[test['Census_TotalPhysicalRAM'].isin(top_10) == False, 'Census_TotalPhysicalRAM'] = 1000

# ### Census_InternalPrimaryDiagonalDisplaySizeInInches

# In[ ]:


plot_categorical_feature('Census_InternalPrimaryDiagonalDisplaySizeInInches', True, by_touch=True)

# I'm really surprised that most computers have 15 inch screens. And this is a rare situation, when for some categories detection rate on PC is higher than on touch devices.

# ### Census_InternalPrimaryDisplayResolutionHorizontal

# In[ ]:


plot_categorical_feature('Census_InternalPrimaryDisplayResolutionHorizontal', True, by_touch=True)

# ### Census_InternalPrimaryDisplayResolutionVertical

# In[ ]:


plot_categorical_feature('Census_InternalPrimaryDisplayResolutionVertical', True, by_touch=True)

# ### Census_InternalBatteryNumberOfCharges

# In[ ]:


plot_categorical_feature('Census_InternalBatteryNumberOfCharges', True, by_touch=True)

# Interesting and somewhat strange results. It is expected that most PCs have 0 charges. 4.294967e+09 seems to be some kind of technical value. But having several charges for PC is strange. Anyway, this is definitely a categorical variable.

# In[ ]:


train['Census_InternalBatteryNumberOfCharges'] = train['Census_InternalBatteryNumberOfCharges'].astype('category')
test['Census_InternalBatteryNumberOfCharges'] = test['Census_InternalBatteryNumberOfCharges'].astype('category')

# ### Census_OSVersion

# In[ ]:


plot_categorical_feature('Census_OSVersion', True)

# ### Census_OSBranch

# In[ ]:


plot_categorical_feature('Census_OSBranch', True)

# ### Census_OSBuildNumber

# In[ ]:


plot_categorical_feature('Census_OSBuildNumber', True)

# In[ ]:


train['Census_OSBuildNumber'] = train['Census_OSBuildNumber'].astype('category')
test['Census_OSBuildNumber'] = test['Census_OSBuildNumber'].astype('category')

# ### Census_OSBuildRevision

# In[ ]:


plot_categorical_feature('Census_OSBuildRevision', True)

# In[ ]:


train['Census_OSBuildRevision'] = train['Census_OSBuildRevision'].astype('category')
test['Census_OSBuildRevision'] = test['Census_OSBuildRevision'].astype('category')

# ### Census_FirmwareManufacturerIdentifier

# In[ ]:


plot_categorical_feature('Census_FirmwareManufacturerIdentifier', True)

# In[ ]:


train['Census_FirmwareManufacturerIdentifier'] = train['Census_FirmwareManufacturerIdentifier'].astype('category')
test['Census_FirmwareManufacturerIdentifier'] = test['Census_FirmwareManufacturerIdentifier'].astype('category')

# ### Census_FirmwareVersionIdentifier

# In[ ]:


plot_categorical_feature('Census_FirmwareVersionIdentifier', True)

# In[ ]:


train['Census_FirmwareVersionIdentifier'] = train['Census_FirmwareVersionIdentifier'].astype('category')
test['Census_FirmwareVersionIdentifier'] = test['Census_FirmwareVersionIdentifier'].astype('category')

# ### OsBuild

# In[ ]:


plot_categorical_feature('OsBuild', True)

# In[ ]:


train['OsBuild'] = train['OsBuild'].astype('category')
test['OsBuild'] = test['OsBuild'].astype('category')

# ### Census_ChassisTypeName

# In[ ]:


plot_categorical_feature('Census_ChassisTypeName', True, by_touch=True)

# ### Census_InternalBatteryType

# In[ ]:


# https://www.kaggle.com/youhanlee/my-eda-i-want-to-see-all
# grouping battary types by name
def group_battery(x):
    x = x.lower()
    if 'li' in x:
        return 1
    else:
        return 0
    
train['Census_InternalBatteryType'] = train['Census_InternalBatteryType'].apply(group_battery)
test['Census_InternalBatteryType'] = test['Census_InternalBatteryType'].apply(group_battery)

# In[ ]:


plot_categorical_feature('Census_InternalBatteryType', True)

# ### Census_OSEdition
# 
# I'll combine similar versions into one.

# In[ ]:


def rename_edition(x):
    x = x.lower()
    if 'core' in x:
        return 'Core'
    elif 'pro' in x:
        return 'pro'
    elif 'enterprise' in x:
        return 'Enterprise'
    elif 'server' in x:
        return 'Server'
    elif 'home' in x:
        return 'Home'
    elif 'education' in x:
        return 'Education'
    elif 'cloud' in x:
        return 'Cloud'
    else:
        return x

# In[ ]:


train['Census_OSEdition'] = train['Census_OSEdition'].astype(str)
test['Census_OSEdition'] = test['Census_OSEdition'].astype(str)
train['Census_OSEdition'] = train['Census_OSEdition'].apply(rename_edition)
test['Census_OSEdition'] = test['Census_OSEdition'].apply(rename_edition)
train['Census_OSEdition'] = train['Census_OSEdition'].astype('category')
test['Census_OSEdition'] = test['Census_OSEdition'].astype('category')

# In[ ]:


plot_categorical_feature('Census_OSEdition', True, by_touch=True)

# ### Census_OSSkuName
# Almost the same as previous variable

# In[ ]:


train['Census_OSSkuName'] = train['Census_OSSkuName'].astype(str)
test['Census_OSSkuName'] = test['Census_OSSkuName'].astype(str)
train['Census_OSSkuName'] = train['Census_OSSkuName'].apply(rename_edition)
test['Census_OSSkuName'] = test['Census_OSSkuName'].apply(rename_edition)
train['Census_OSSkuName'] = train['Census_OSSkuName'].astype('category')
test['Census_OSSkuName'] = test['Census_OSSkuName'].astype('category')

# In[ ]:


plot_categorical_feature('Census_OSSkuName', True, by_touch=True)

# ### Census_OSInstallLanguageIdentifier
# Sadly we don't know what mean these numbers.

# In[ ]:


plot_categorical_feature('Census_OSInstallLanguageIdentifier', True)

# In[ ]:


train['Census_OSInstallLanguageIdentifier'] = train['Census_OSInstallLanguageIdentifier'].astype('category')
test['Census_OSInstallLanguageIdentifier'] = test['Census_OSInstallLanguageIdentifier'].astype('category')

# ### Census_OSUILocaleIdentifier

# In[ ]:


plot_categorical_feature('Census_OSUILocaleIdentifier', True)

# In[ ]:


train['Census_OSUILocaleIdentifier'] = train['Census_OSUILocaleIdentifier'].astype('category')
test['Census_OSUILocaleIdentifier'] = test['Census_OSUILocaleIdentifier'].astype('category')

# ### Census_OSUILocaleIdentifier

# In[ ]:


plot_categorical_feature('Census_OSUILocaleIdentifier', True)

# ### OsSuite

# In[ ]:


plot_categorical_feature('OsSuite', True)

# In[ ]:


train['OsSuite'] = train['OsSuite'].astype('category')
test['OsSuite'] = test['OsSuite'].astype('category')

# ### Wdft_RegionIdentifier

# In[ ]:


plot_categorical_feature('Wdft_RegionIdentifier', True)

# In[ ]:


train['Wdft_RegionIdentifier'] = train['Wdft_RegionIdentifier'].astype('category')
test['Wdft_RegionIdentifier'] = test['Wdft_RegionIdentifier'].astype('category')

# ### SkuEdition

# In[ ]:


train['SkuEdition'].value_counts(dropna=False, normalize=True)

# Home and Pro editions together give 97.9%+ of all values. Condidering that other categories are for Enterprise mostly, I combine them with Pro.

# In[ ]:


# train.loc[train['SkuEdition'] != 'Home', 'SkuEdition'] = 'Pro'
# test.loc[test['SkuEdition'] != 'Home', 'SkuEdition'] = 'Pro'

# train['SkuEdition'] = train['SkuEdition'].cat.remove_unused_categories()
# test['SkuEdition'] = test['SkuEdition'].cat.remove_unused_categories()

# In[ ]:


pd.crosstab(train['SkuEdition'], train['Census_OSEdition'], normalize='columns')

# It seems that Home Sku edition corresponds to Core OS Edition, and Pro Sku edition corresponts to all other OS editions.

# ### SmartScreen

# In[ ]:


train['SmartScreen'].value_counts(dropna=False, normalize=True).cumsum()

# As per description: If the value exists but is blank, the value "ExistsNotSet" is sent in telemetry.
# So missing values and ExistsNotSet are in fact the same. This category + RequireAdmin + Off + Warn are 99.3% of all values. I'll combine all other values into Prompt.

# In[ ]:


# train.loc[train['SmartScreen'].isnull(), 'SmartScreen'] = 'ExistsNotSet'
# test.loc[test['SmartScreen'].isnull(), 'SmartScreen'] = 'ExistsNotSet'
# train.loc[train['SmartScreen'].isin(['RequireAdmin', 'ExistsNotSet', 'Off', 'Warn']) == False, 'SmartScreen'] = 'Prompt'
# test.loc[test['SmartScreen'].isin(['RequireAdmin', 'ExistsNotSet', 'Off', 'Warn']) == False, 'SmartScreen'] = 'Prompt'

# train['SmartScreen'] = train['SmartScreen'].cat.remove_unused_categories()
# test['SmartScreen'] = test['SmartScreen'].cat.remove_unused_categories()

# In[ ]:


sns.set(rc={'figure.figsize':(15, 8)})
sns.countplot(x="SmartScreen", hue="HasDetections",  palette="PRGn", data=train)
plt.title("SmartScreen counts")
plt.xticks(rotation='vertical')
plt.show()

# ### Census_MDC2FormFactor

# In[ ]:


train['Census_MDC2FormFactor'].value_counts(dropna=False, normalize=True).cumsum()

# In[ ]:


# top_cats = list(train['Census_MDC2FormFactor'].value_counts().index[:5])
# train.loc[train['Census_MDC2FormFactor'].isin(top_cats) == False, 'Census_MDC2FormFactor'] = 'PCOther'
# test.loc[test['Census_MDC2FormFactor'].isin(top_cats) == False, 'Census_MDC2FormFactor'] = 'PCOther'

# train['Census_MDC2FormFactor'] = train['Census_MDC2FormFactor'].cat.remove_unused_categories()
# test['Census_MDC2FormFactor'] = test['Census_MDC2FormFactor'].cat.remove_unused_categories()

# In[ ]:


plot_categorical_feature('Census_MDC2FormFactor', True)

# In[ ]:


sns.catplot(x="Census_PrimaryDiskTypeName", hue="HasDetections", col="Census_MDC2FormFactor",
                data=train, kind="count",col_wrap=3);

# It is nice to see that these two variables interact reasonably.
# Detachable devices usually have SSD, desktop PC have HDD and SDD and Notebooks usually have HDD.

# In[ ]:


train['Census_PrimaryDiskTypeName'].value_counts(dropna=False, normalize=True).cumsum()

# In[ ]:


# train.loc[train['Census_PrimaryDiskTypeName'].isin(['HDD', 'SSD']) == False, 'Census_PrimaryDiskTypeName'] = 'UNKNOWN'
# test.loc[test['Census_PrimaryDiskTypeName'].isin(['HDD', 'SSD']) == False, 'Census_PrimaryDiskTypeName'] = 'UNKNOWN'

# train['Census_PrimaryDiskTypeName'] = train['Census_PrimaryDiskTypeName'].cat.remove_unused_categories()
# test['Census_PrimaryDiskTypeName'] = test['Census_PrimaryDiskTypeName'].cat.remove_unused_categories()

# ### Census_ProcessorManufacturerIdentifier

# In[ ]:


train['Census_ProcessorManufacturerIdentifier'].value_counts(dropna=False, normalize=True).cumsum()

# In[ ]:


# train.loc[train['Census_ProcessorManufacturerIdentifier'].isin([5.0, 1.0]) == False, 'Census_ProcessorManufacturerIdentifier'] = 0.0
# test.loc[test['Census_ProcessorManufacturerIdentifier'].isin([5.0, 1.0]) == False, 'Census_ProcessorManufacturerIdentifier'] = 0.0

# train['Census_ProcessorManufacturerIdentifier'] = train['Census_ProcessorManufacturerIdentifier'].astype('category')
# test['Census_ProcessorManufacturerIdentifier'] = test['Census_ProcessorManufacturerIdentifier'].astype('category')

# In[ ]:


plot_categorical_feature('Census_ProcessorManufacturerIdentifier', True)

# Once again anonymized variable.

# ### Census_PowerPlatformRoleName

# In[ ]:


train['Census_PowerPlatformRoleName'].value_counts(dropna=False, normalize=True).cumsum()

# In[ ]:


# train.loc[train['Census_PowerPlatformRoleName'].isin(['Mobile', 'Desktop', 'Slate']) == False, 'Census_PowerPlatformRoleName'] = 'UNKNOWN'
# test.loc[test['Census_PowerPlatformRoleName'].isin(['Mobile', 'Desktop', 'Slate']) == False, 'Census_PowerPlatformRoleName'] = 'UNKNOWN'

# train['Census_PowerPlatformRoleName'] = train['Census_PowerPlatformRoleName'].cat.remove_unused_categories()
# test['Census_PowerPlatformRoleName'] = test['Census_PowerPlatformRoleName'].cat.remove_unused_categories()

# In[ ]:


plot_categorical_feature('Census_PowerPlatformRoleName', True)

# For some reason slates have quite a low rate of detections. Maybe no hackers are interested in them? :)

# ### Census_OSInstallTypeName

# In[ ]:


plot_categorical_feature('Census_OSInstallTypeName', True)

# In[ ]:


# top_cats = list(train['Census_OSWUAutoUpdateOptionsName'].value_counts().index[:3])
# train.loc[train['Census_OSWUAutoUpdateOptionsName'].isin(top_cats) == False, 'Census_OSWUAutoUpdateOptionsName'] = 'Off'
# test.loc[test['Census_OSWUAutoUpdateOptionsName'].isin(top_cats) == False, 'Census_OSWUAutoUpdateOptionsName'] = 'Off'

# train['Census_OSWUAutoUpdateOptionsName'] = train['Census_OSWUAutoUpdateOptionsName'].cat.remove_unused_categories()
# test['Census_OSWUAutoUpdateOptionsName'] = test['Census_OSWUAutoUpdateOptionsName'].cat.remove_unused_categories()

# In[ ]:


# train.loc[train['Census_GenuineStateName'] == 'UNKNOWN', 'Census_GenuineStateName'] = 'OFFLINE'
# test.loc[test['Census_GenuineStateName'] == 'UNKNOWN', 'Census_GenuineStateName'] = 'OFFLINE'

# train['Census_GenuineStateName'] = train['Census_GenuineStateName'].cat.remove_unused_categories()
# test['Census_GenuineStateName'] = test['Census_GenuineStateName'].cat.remove_unused_categories()

# In[ ]:


# train.loc[train['Census_ActivationChannel'].isin(['Retail', 'OEM:DM']) == False, 'Census_ActivationChannel'] = 'Volume:GVLK'
# test.loc[test['Census_ActivationChannel'].isin(['Retail', 'OEM:DM']) == False, 'Census_ActivationChannel'] = 'Volume:GVLK'

# train['Census_ActivationChannel'] = train['Census_ActivationChannel'].cat.remove_unused_categories()
# test['Census_ActivationChannel'] = test['Census_ActivationChannel'].cat.remove_unused_categories()

# ## Feature engineering and transformation

# In[ ]:


train.head()

# In[ ]:


train['OsBuildLab'] = train['OsBuildLab'].cat.add_categories(['0.0.0.0.0-0'])
train['OsBuildLab'] = train['OsBuildLab'].fillna('0.0.0.0.0-0')
test['OsBuildLab'] = test['OsBuildLab'].cat.add_categories(['0.0.0.0.0-0'])
test['OsBuildLab'] = test['OsBuildLab'].fillna('0.0.0.0.0-0')

# In[ ]:


def fe(df):
    df['EngineVersion_2'] = df['EngineVersion'].apply(lambda x: x.split('.')[2]).astype('category')
    df['EngineVersion_3'] = df['EngineVersion'].apply(lambda x: x.split('.')[3]).astype('category')

    df['AppVersion_1'] = df['AppVersion'].apply(lambda x: x.split('.')[1]).astype('category')
    df['AppVersion_2'] = df['AppVersion'].apply(lambda x: x.split('.')[2]).astype('category')
    df['AppVersion_3'] = df['AppVersion'].apply(lambda x: x.split('.')[3]).astype('category')

    df['AvSigVersion_0'] = df['AvSigVersion'].apply(lambda x: x.split('.')[0]).astype('category')
    df['AvSigVersion_1'] = df['AvSigVersion'].apply(lambda x: x.split('.')[1]).astype('category')
    df['AvSigVersion_2'] = df['AvSigVersion'].apply(lambda x: x.split('.')[2]).astype('category')

    df['OsBuildLab_0'] = df['OsBuildLab'].apply(lambda x: x.split('.')[0]).astype('category')
    df['OsBuildLab_1'] = df['OsBuildLab'].apply(lambda x: x.split('.')[1]).astype('category')
    df['OsBuildLab_2'] = df['OsBuildLab'].apply(lambda x: x.split('.')[2]).astype('category')
    df['OsBuildLab_3'] = df['OsBuildLab'].apply(lambda x: x.split('.')[3]).astype('category')
    # df['OsBuildLab_40'] = df['OsBuildLab'].apply(lambda x: x.split('.')[-1].split('-')[0]).astype('category')
    # df['OsBuildLab_41'] = df['OsBuildLab'].apply(lambda x: x.split('.')[-1].split('-')[1]).astype('category')

    df['Census_OSVersion_0'] = df['Census_OSVersion'].apply(lambda x: x.split('.')[0]).astype('category')
    df['Census_OSVersion_1'] = df['Census_OSVersion'].apply(lambda x: x.split('.')[1]).astype('category')
    df['Census_OSVersion_2'] = df['Census_OSVersion'].apply(lambda x: x.split('.')[2]).astype('category')
    df['Census_OSVersion_3'] = df['Census_OSVersion'].apply(lambda x: x.split('.')[3]).astype('category')

    # https://www.kaggle.com/adityaecdrid/simple-feature-engineering-xd
    df['primary_drive_c_ratio'] = df['Census_SystemVolumeTotalCapacity']/ df['Census_PrimaryDiskTotalCapacity']
    df['non_primary_drive_MB'] = df['Census_PrimaryDiskTotalCapacity'] - df['Census_SystemVolumeTotalCapacity']

    df['aspect_ratio'] = df['Census_InternalPrimaryDisplayResolutionHorizontal']/ df['Census_InternalPrimaryDisplayResolutionVertical']

    df['monitor_dims'] = df['Census_InternalPrimaryDisplayResolutionHorizontal'].astype(str) + '*' + df['Census_InternalPrimaryDisplayResolutionVertical'].astype('str')
    df['monitor_dims'] = df['monitor_dims'].astype('category')

    df['dpi'] = ((df['Census_InternalPrimaryDisplayResolutionHorizontal']**2 + df['Census_InternalPrimaryDisplayResolutionVertical']**2)**.5)/(df['Census_InternalPrimaryDiagonalDisplaySizeInInches'])

    df['dpi_square'] = df['dpi'] ** 2

    df['MegaPixels'] = (df['Census_InternalPrimaryDisplayResolutionHorizontal'] * df['Census_InternalPrimaryDisplayResolutionVertical'])/1e6

    df['Screen_Area'] = (df['aspect_ratio']* (df['Census_InternalPrimaryDiagonalDisplaySizeInInches']**2))/(df['aspect_ratio']**2 + 1)

    df['ram_per_processor'] = df['Census_TotalPhysicalRAM']/ df['Census_ProcessorCoreCount']

    df['new_num_0'] = df['Census_InternalPrimaryDiagonalDisplaySizeInInches'] / df['Census_ProcessorCoreCount']

    df['new_num_1'] = df['Census_ProcessorCoreCount'] * df['Census_InternalPrimaryDiagonalDisplaySizeInInches']
    
    df['Census_IsFlightingInternal'] = df['Census_IsFlightingInternal'].fillna(1)
    df['Census_ThresholdOptIn'] = df['Census_ThresholdOptIn'].fillna(1)
    df['Census_IsWIMBootEnabled'] = df['Census_IsWIMBootEnabled'].fillna(1)
    df['Wdft_IsGamer'] = df['Wdft_IsGamer'].fillna(0)
    
    return df

# In[ ]:


train = fe(train)
test = fe(test)

# In[ ]:


cat_cols = [col for col in train.columns if col not in ['MachineIdentifier', 'Census_SystemVolumeTotalCapacity', 'HasDetections'] and str(train[col].dtype) == 'category']
len(cat_cols)

# In[ ]:


more_cat_cols = []
add_cat_feats = [
 'Census_OSBuildRevision',
 'OsBuildLab',
 'SmartScreen',
'AVProductsInstalled']
for col1 in add_cat_feats:
    for col2 in add_cat_feats:
        if col1 != col2:
            train[col1 + '__' + col2] = train[col1].astype(str) + train[col2].astype(str)
            train[col1 + '__' + col2] = train[col1 + '__' + col2].astype('category')
            
            test[col1 + '__' + col2] = test[col1].astype(str) + test[col2].astype(str)
            test[col1 + '__' + col2] = test[col1 + '__' + col2].astype('category')
            more_cat_cols.append(col1 + '__' + col2)
            
cat_cols = cat_cols + more_cat_cols

# In[ ]:


to_encode = []
for col in cat_cols:
    if train[col].nunique() > 1000:
        print(col, train[col].nunique())
        to_encode.append(col)

# ### Frequency encoding
# 
# From this kernel:  https://www.kaggle.com/fabiendaniel/detecting-malwares-with-lgbm
# 
# I do frequency and label encoding on the full train dataset, because I think I'll get more correct values this way.

# In[ ]:


train = reduce_mem_usage(train)
test = reduce_mem_usage(test)
gc.collect()

# In[ ]:


def frequency_encoding(variable):
    t = pd.concat([train[variable], test[variable]]).value_counts().reset_index()
    t = t.reset_index()
    t.loc[t[variable] == 1, 'level_0'] = np.nan
    t.set_index('index', inplace=True)
    max_label = t['level_0'].max() + 1
    t.fillna(max_label, inplace=True)
    return t.to_dict()['level_0']

# In[ ]:


for col in tqdm_notebook(to_encode):
    freq_enc_dict = frequency_encoding(col)
    train[col] = train[col].map(lambda x: freq_enc_dict.get(x, np.nan))
    test[col] = test[col].map(lambda x: freq_enc_dict.get(x, np.nan))
    cat_cols.remove(col)

# ### Label Encoding

# In[ ]:


indexer = {}
for col in cat_cols:
    # print(col)
    _, indexer[col] = pd.factorize(train[col].astype(str), sort=True)
    
for col in tqdm_notebook(cat_cols):
    # print(col)
    train[col] = indexer[col].get_indexer(train[col].astype(str))
    test[col] = indexer[col].get_indexer(test[col].astype(str))
    
    train = reduce_mem_usage(train, verbose=False)
    test = reduce_mem_usage(test, verbose=False)

# In[ ]:


del indexer

# In[ ]:


train.head()

# ## Idea of training on all data
# Let's try training model on three subsets of train data separately and blend the predictions.

# In[ ]:


y = train['HasDetections']
train = train.drop(['HasDetections', 'MachineIdentifier'], axis=1)
test = test.drop(['MachineIdentifier'], axis=1)
gc.collect()
train.sort_values('AvSigVersion')
train1 = train[:4000000]
train = train[4000000:8000000]

y1 = y[:4000000]
y = y[4000000:8000000]

# ### Modelling
# 
# Added possibility to do rank averaging.

# In[ ]:


n_fold = 5
folds = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=15)
# folds = TimeSeriesSplit(n_splits=5)

# In[ ]:


from numba import jit
# fast roc_auc computation: https://www.kaggle.com/c/microsoft-malware-prediction/discussion/76013
@jit
def fast_auc(y_true, y_prob):
    y_true = np.asarray(y_true)
    y_true = y_true[np.argsort(y_prob)]
    nfalse = 0
    auc = 0
    n = len(y_true)
    for i in range(n):
        y_i = y_true[i]
        nfalse += (1 - y_i)
        auc += y_i * nfalse
    auc /= (nfalse * (n - nfalse))
    return auc

def eval_auc(preds, dtrain):
    labels = dtrain.get_label()
    return 'auc', fast_auc(labels, preds), True

# idea from this kernel: https://www.kaggle.com/fabiendaniel/detecting-malwares-with-lgbm
def predict_chunk(model, test):
    initial_idx = 0
    chunk_size = 1000000
    current_pred = np.zeros(len(test))
    while initial_idx < test.shape[0]:
        final_idx = min(initial_idx + chunk_size, test.shape[0])
        idx = range(initial_idx, final_idx)
        current_pred[idx] = model.predict(test.iloc[idx], num_iteration=model.best_iteration)
        initial_idx = final_idx
    #predictions += current_pred / min(folds.n_splits, max_iter)
    return current_pred


def train_model(X=train, X_test=test, y=y, params=None, folds=folds, model_type='lgb', plot_feature_importance=False, averaging='usual', make_oof=False):
    result_dict = {}
    if make_oof:
        oof = np.zeros(len(X))
    prediction = np.zeros(len(X_test))
    scores = []
    feature_importance = pd.DataFrame()
    for fold_n, (train_index, valid_index) in enumerate(folds.split(X, y)):
        gc.collect()
        print('Fold', fold_n + 1, 'started at', time.ctime())
        X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
        y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]
        
        
        if model_type == 'lgb':
            train_data = lgb.Dataset(X_train, label=y_train, categorical_feature = cat_cols)
            valid_data = lgb.Dataset(X_valid, label=y_valid, categorical_feature = cat_cols)
            
            model = lgb.train(params,
                    train_data,
                    num_boost_round=2000,
                    valid_sets = [train_data, valid_data],
                    verbose_eval=500,
                    early_stopping_rounds = 200,
                    feval=eval_auc)

            del train_data, valid_data
            
            y_pred_valid = model.predict(X_valid, num_iteration=model.best_iteration)
            del X_valid
            gc.collect()
            # print('predicting on test')
            # y_pred = model.predict(X_test, num_iteration=model.best_iteration)
            y_pred = predict_chunk(model, X_test)
            # print('predicted')
            
        if model_type == 'xgb':
            train_data = xgb.DMatrix(data=X_train, label=y_train)
            valid_data = xgb.DMatrix(data=X_valid, label=y_valid)

            watchlist = [(train_data, 'train'), (valid_data, 'valid_data')]
            model = xgb.train(dtrain=train_data, num_boost_round=20000, evals=watchlist, early_stopping_rounds=200, verbose_eval=500, params=params)
            y_pred_valid = model.predict(xgb.DMatrix(X_valid), ntree_limit=model.best_ntree_limit)
            #y_pred = model.predict(xgb.DMatrix(X_test), ntree_limit=model.best_ntree_limit)
            y_pred = predict_chunk(model, xgb.DMatrix(X_test))
            
        if model_type == 'lcv':
            model = LogisticRegressionCV(scoring='roc_auc', cv=3)
            model.fit(X_train, y_train)

            y_pred_valid = model.predict(X_valid)
            # y_pred = model.predict(X_test)
            y_pred = predict_chunk(model, X_test)
            
        if model_type == 'cat':
            model = CatBoostRegressor(iterations=20000,  eval_metric='AUC', **params)
            model.fit(X_train, y_train, eval_set=(X_valid, y_valid), cat_features=[], use_best_model=True, verbose=False)

            y_pred_valid = model.predict(X_valid)
            # y_pred = model.predict(X_test)
            y_pred = predict_chunk(model, X_test)
        
        if make_oof:
            oof[valid_index] = y_pred_valid.reshape(-1,)
            
        scores.append(fast_auc(y_valid, y_pred_valid))
        print('Fold roc_auc:', roc_auc_score(y_valid, y_pred_valid))
        print('')
        
        if averaging == 'usual':
            prediction += y_pred
        elif averaging == 'rank':
            prediction += pd.Series(y_pred).rank().values
        
        if model_type == 'lgb':
            # feature importance
            fold_importance = pd.DataFrame()
            fold_importance["feature"] = X.columns
            fold_importance["importance"] = model.feature_importance()
            fold_importance["fold"] = fold_n + 1
            feature_importance = pd.concat([feature_importance, fold_importance], axis=0)

    prediction /= n_fold
    
    print('CV mean score: {0:.4f}, std: {1:.4f}.'.format(np.mean(scores), np.std(scores)))
    
    if model_type == 'lgb':
        
        if plot_feature_importance:
            feature_importance["importance"] /= n_fold
            cols = feature_importance[["feature", "importance"]].groupby("feature").mean().sort_values(
                by="importance", ascending=False)[:50].index

            best_features = feature_importance.loc[feature_importance.feature.isin(cols)]
            logging.info('Top features')
            for f in best_features.sort_values(by="importance", ascending=False)['feature'].values:
                logging.info(f)

            plt.figure(figsize=(16, 12));
            sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False));
            plt.title('LGB Features (avg over folds)');
            
            result_dict['feature_importance'] = feature_importance
            
    result_dict['prediction'] = prediction
    if make_oof:
        result_dict['oof'] = oof
    
    return result_dict

# In[ ]:


# params from https://www.kaggle.com/fabiendaniel/detecting-malwares-with-lgbm
params = {'num_leaves': 256,
         'min_data_in_leaf': 42,
         'objective': 'binary',
         'max_depth': 7,
         'learning_rate': 0.05,
         "boosting": "gbdt",
         "feature_fraction": 0.8,
         "bagging_freq": 3,
         "bagging_fraction": 0.8,
         "bagging_seed": 11,
         "lambda_l1": 0.15,
         "lambda_l2": 0.15,
         "random_state": 42,          
         "verbosity": -1}

# In[ ]:


del stats_df, freq_enc_dict

# In[ ]:


result_dict1 = train_model(X=train1, X_test=test, y=y1, params=params, model_type='lgb', plot_feature_importance=True, averaging='rank')

# In[ ]:


del train1, y1

# In[ ]:


# params = {'num_leaves': 64,
#          'min_data_in_leaf': 20,
#          'objective': 'binary',
#          'max_depth': 5,
#          'learning_rate': 0.05,
#          "boosting": "gbdt",
#          "feature_fraction": 0.9,
#          "bagging_freq": 5,
#          "bagging_fraction": 0.7,
#          "bagging_seed": 11,
#          "lambda_l1": 0.2,
#          "lambda_l2": 0.2,
#          "random_state": 13,          
#          "verbosity": -1}
# result_dict = train_model(X=train, X_test=test, y=y, params=params, model_type='lgb', plot_feature_importance=False, averaging='rank')

# In[ ]:


submission = pd.read_csv('../input/sample_submission.csv')

# submission['HasDetections'] = (result_dict['prediction'] + result_dict1['prediction'] + result_dict2['prediction']) / 3
# submission['HasDetections'] = (result_dict['prediction'] + result_dict1['prediction']) / 2
submission['HasDetections'] = result_dict1['prediction']
submission.to_csv('lgb_rank.csv', index=False)

# In[ ]:



