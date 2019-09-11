#!/usr/bin/env python
# coding: utf-8

#  # You can remove 17 columns at the Beginning!!
# 
# As the data is highly dimensional in this competition, it is really difficult to do even a little thing. So, before you begin any work, read this kernel and save your time!
# 
# I have tried to reduce the column dimension by eliminating less useful columns and selected 17 columns which you can remove just after loading the data sets.
# 
# * Selected `mostly-missing feaures` which have more than 99% of missing values.
# * Selected `too-skewed features` whose majority categories cover more than 99% of occurences.
# * Selected `hightly-correlated features`. Tested correlations between columns, picked up pairs whose corr is greater than 0.99, compared the distribution of the features in the pairs and corr with `HasDetections`,  and selected the minor column for elimination.
# 
# **You can eliminate 17 columns without worry:**
# 1. (M) PuaMode
# 1. (M) Census_ProcessorClass
# 1. (S) Census_IsWIMBootEnabled
# 1. (S) IsBeta
# 1. (S) Census_IsFlightsDisabled
# 1. (S) Census_IsFlightingInternal
# 1. (S) AutoSampleOptIn
# 1. (S) Census_ThresholdOptIn
# 1. (S) SMode
# 1. (S) Census_IsPortableOperatingSystem
# 1. (S) Census_DeviceFamily
# 1. (S) UacLuaenable
# 1. (S) Census_IsVirtualDevice
# 1. (C) Platform
# 1. (C) Census_OSSkuName
# 1. (C) Census_OSInstallLanguageIdentifier
# 1. (C) Processor
# 
# Here, (M) denotes `mostly-missing feaures`, (S) means  `too-skewed features`, and (C) indicates `hightly-correlated features`.
# 
# 
# 
# **Use this code:**
# 
# > remove_cols = ['PuaMode', 'Census_ProcessorClass', 'Census_IsWIMBootEnabled', 'IsBeta', 'Census_IsFlightsDisabled', 'Census_IsFlightingInternal', 'AutoSampleOptIn', 'Census_ThresholdOptIn', 'SMode', 'Census_IsPortableOperatingSystem',  'Census_DeviceFamily', 'UacLuaenable', 'Census_IsVirtualDevice', 'Platform', 'Census_OSSkuName', 'Census_OSInstallLanguageIdentifier', 'Processor']
# >
# > train.drop(remove_cols, axis=1, inplace=True)
# >
# > test.drop(remove_cols, axis=1, inplace=True)
# 
# 
# ## If you want to see how I got this:
# In this kernel, I used only train dataset but the result was the same when I used train+test dataset.

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# # 1. Load Data

# In[ ]:


# referred https://www.kaggle.com/theoviel/load-the-totality-of-the-data
dtypes = {
        'MachineIdentifier':                                    'category',
        'ProductName':                                          'category',
        'EngineVersion':                                        'category',
        'AppVersion':                                           'category',
        'AvSigVersion':                                         'category',
        'IsBeta':                                               'int8',
        'RtpStateBitfield':                                     'float16',
        'IsSxsPassiveMode':                                     'int8',
        'DefaultBrowsersIdentifier':                            'float32',
        'AVProductStatesIdentifier':                            'float32',
        'AVProductsInstalled':                                  'float16',
        'AVProductsEnabled':                                    'float16',
        'HasTpm':                                               'int8',
        'CountryIdentifier':                                    'int16',
        'CityIdentifier':                                       'float32',
        'OrganizationIdentifier':                               'float16',
        'GeoNameIdentifier':                                    'float16',
        'LocaleEnglishNameIdentifier':                          'int16',
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
        'UacLuaenable':                                         'float64', # was 'float32'
        'Census_MDC2FormFactor':                                'category',
        'Census_DeviceFamily':                                  'category',
        'Census_OEMNameIdentifier':                             'float32', # was 'float16'
        'Census_OEMModelIdentifier':                            'float32',
        'Census_ProcessorCoreCount':                            'float16',
        'Census_ProcessorManufacturerIdentifier':               'float16',
        'Census_ProcessorModelIdentifier':                      'float32', # was 'float16'
        'Census_ProcessorClass':                                'category',
        'Census_PrimaryDiskTotalCapacity':                      'float64', # was 'float32'
        'Census_PrimaryDiskTypeName':                           'category',
        'Census_SystemVolumeTotalCapacity':                     'float64', # was 'float32'
        'Census_HasOpticalDiskDrive':                           'int8',
        'Census_TotalPhysicalRAM':                              'float32',
        'Census_ChassisTypeName':                               'category',
        'Census_InternalPrimaryDiagonalDisplaySizeInInches':    'float32', # was 'float16'
        'Census_InternalPrimaryDisplayResolutionHorizontal':    'float32', # was 'float16'
        'Census_InternalPrimaryDisplayResolutionVertical':      'float32', # was 'float16'
        'Census_PowerPlatformRoleName':                         'category',
        'Census_InternalBatteryType':                           'category',
        'Census_InternalBatteryNumberOfCharges':                'float64', # was 'float32'
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
train = pd.read_csv('../input/train.csv', dtype=dtypes)
train.shape

# In[ ]:


droppable_features = []

# # 2. Feature Engineering
# 
# ## 2.1 mostly-missing Columns

# In[ ]:


(train.isnull().sum()/train.shape[0]).sort_values(ascending=False)

# * There are 2 columns which have more than 99% of missing values and they are useless.

# In[ ]:


droppable_features.append('PuaMode')
droppable_features.append('Census_ProcessorClass')

# ## 2.2 Too skewed columns

# In[ ]:


pd.options.display.float_format = '{:,.4f}'.format
sk_df = pd.DataFrame([{'column': c, 'uniq': train[c].nunique(), 'skewness': train[c].value_counts(normalize=True).values[0] * 100} for c in train.columns])
sk_df = sk_df.sort_values('skewness', ascending=False)
sk_df

# * There are 12 categorical columns whose majority category covers more than 99% of occurences, and they are useless, too.

# In[ ]:


droppable_features.extend(sk_df[sk_df.skewness > 99].column.tolist())
droppable_features

# In[ ]:


# PuaMode is duplicated in the two categories.
droppable_features.remove('PuaMode')

# Drop these columns.
train.drop(droppable_features, axis=1, inplace=True)

# ### Fill missing values for columns that have more than 10% of missing values

# In[ ]:


# Nan Values
null_counts = train.isnull().sum()
null_counts = null_counts / train.shape[0]
null_counts[null_counts > 0.1]

# ### 4 columns above should be filled missing values.

# In[ ]:


train.DefaultBrowsersIdentifier.value_counts().head(5)

# Replace missing values with 0.

# In[ ]:


train.DefaultBrowsersIdentifier.fillna(0, inplace=True)

# In[ ]:


train.SmartScreen.value_counts()

# In[ ]:


trans_dict = {
    'off': 'Off', '&#x02;': '2', '&#x01;': '1', 'on': 'On', 'requireadmin': 'RequireAdmin', 'OFF': 'Off', 
    'Promt': 'Prompt', 'requireAdmin': 'RequireAdmin', 'prompt': 'Prompt', 'warn': 'Warn', 
    '00000000': '0', '&#x03;': '3', np.nan: 'NoExist'
}
train.replace({'SmartScreen': trans_dict}, inplace=True)

# In[ ]:


train.SmartScreen.isnull().sum()

# In[ ]:


train.OrganizationIdentifier.value_counts()

# ### This column has ID numbers and I think 0 can represent unknown/NA values.

# In[ ]:


train.replace({'OrganizationIdentifier': {np.nan: 0}}, inplace=True)

# In[ ]:


pd.options.display.max_rows = 99
train.Census_InternalBatteryType.value_counts()

# ### Census_InternalBatteryType has 75+% of missing values as well as "˙˙˙" and "unkn" values which seem to mean "unknown". So replace these values with "unknown".

# In[ ]:


trans_dict = {
    '˙˙˙': 'unknown', 'unkn': 'unknown', np.nan: 'unknown'
}
train.replace({'Census_InternalBatteryType': trans_dict}, inplace=True)

# ### Remove missing values from the train.

# In[ ]:


train.shape

# In[ ]:


train.dropna(inplace=True)
train.shape

# Nearly 14% of data has been removed. But I have to think about how to deal with missing values of test dataset...

# MachineIdentifier is not useful for prediction of malware detection.

# In[ ]:


train.drop('MachineIdentifier', axis=1, inplace=True)

# ### Label Encoding for category columns

# In[ ]:


train['SmartScreen'] = train.SmartScreen.astype('category')
train['Census_InternalBatteryType'] = train.Census_InternalBatteryType.astype('category')

cate_cols = train.select_dtypes(include='category').columns.tolist()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

for col in cate_cols:
    train[col] = le.fit_transform(train[col])


# Reduce the memory by codes from https://www.kaggle.com/timon88/load-whole-data-without-any-dtypes

# In[ ]:


def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
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
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df

train = reduce_mem_usage(train)

# ## 2.3 Highly correlated features.
# 
# As there are still too many features, it is bad to calculate and look at all the correlations at once. So, I grouped them by 10 columns and considered their correlations, and finally calculated all the correlation of remaining features.

# In[ ]:


cols = train.columns.tolist()

# In[ ]:


import seaborn as sns

plt.figure(figsize=(10,10))
co_cols = cols[:10]
co_cols.append('HasDetections')
sns.heatmap(train[co_cols].corr(), cmap='RdBu_r', annot=True, center=0.0)
plt.title('Correlation between 1 ~ 10th columns')
plt.show()

# There is no columns which have 0.99+ correlation.

# In[ ]:


corr_remove = []

# In[ ]:


co_cols = cols[10:20]
co_cols.append('HasDetections')
plt.figure(figsize=(10,10))
sns.heatmap(train[co_cols].corr(), cmap='RdBu_r', annot=True, center=0.0)
plt.title('Correlation between 11 ~ 20th columns')
plt.show()

# Compare and choose the feature which has less unique values.

# In[ ]:


print(train.Platform.nunique())
print(train.OsVer.nunique())

# * `Platform` vs `OsVer` : remove **`Platform`**

# In[ ]:


corr_remove.append('Platform')

# In[ ]:


co_cols = cols[20:30]
co_cols.append('HasDetections')
plt.figure(figsize=(10,10))
sns.heatmap(train[co_cols].corr(), cmap='RdBu_r', annot=True, center=0.0)
plt.title('Correlation between 21 ~ 30th columns')
plt.show()

# No features whose correlation is 0.99+.

# In[ ]:


co_cols = cols[30:40]
co_cols.append('HasDetections')
plt.figure(figsize=(10,10))
sns.heatmap(train[co_cols].corr(), cmap='RdBu_r', annot=True, center=0.0)
plt.title('Correlation between 31 ~ 40th columns')
plt.show()

# Nothing.

# In[ ]:


co_cols = cols[40:50]
co_cols.append('HasDetections')
plt.figure(figsize=(10,10))
sns.heatmap(train[co_cols].corr(), cmap='RdBu_r', annot=True, center=0.0)
plt.title('Correlation between 41 ~ 50th columns')
plt.show()

# Nothing.

# In[ ]:


co_cols = cols[50:60]
co_cols.append('HasDetections')
plt.figure(figsize=(10,10))
sns.heatmap(train[co_cols].corr(), cmap='RdBu_r', annot=True, center=0)
plt.title('Correlation between 51 ~ 60th columns')
plt.show()

# In[ ]:


print(train.Census_OSEdition.nunique())
print(train.Census_OSSkuName.nunique(), '\n')
print(train.Census_OSInstallLanguageIdentifier.nunique())
print(train.Census_OSUILocaleIdentifier.nunique())


# * `Census_OSEdition` vs `Census_OSSkuName`:  remove **`Census_OSSkuName`**
# * `Census_OSInstallLanguageIdentifier` vs `Census_OSUILocaleIdentifier`: remove **`Census_OSInstallLanguageIdentifier`**

# In[ ]:


corr_remove.append('Census_OSSkuName')
corr_remove.append('Census_OSInstallLanguageIdentifier')

# In[ ]:


co_cols = cols[60:]
#co_cols.append('HasDetections')
plt.figure(figsize=(10,10))
sns.heatmap(train[co_cols].corr(), cmap='RdBu_r', annot=True, center=0)
plt.title('Correlation between from 61th to the last columns')
plt.show()

# Nothing here.

# In[ ]:


corr_remove

# Now we have got 3 columns to remove from correlations of 10-group features.

# In[ ]:


train.drop(corr_remove, axis=1, inplace=True)

# Now, find cross-group correlated features.

# In[ ]:


corr = train.corr()
high_corr = (corr >= 0.99).astype('uint8')
plt.figure(figsize=(15,15))
sns.heatmap(high_corr, cmap='RdBu_r', annot=True, center=0.0)
plt.show()


# In[ ]:


print(train.Census_OSArchitecture.nunique())
print(train.Processor.nunique())

# `Census_OSArchitecture` and `Processor` have the same length of unique values. Then which one? Let's compare their correlation to the `HasDetections`.

# In[ ]:


train[['Census_OSArchitecture', 'Processor', 'HasDetections']].corr()

# They seem to be totally same, so anything is OK to remove.
# 
# * `Census_OSArchitecture` vs `Processor`: remove **`Processor`**

# In[ ]:


corr_remove.append('Processor')

# In[ ]:


droppable_features.extend(corr_remove)
print(len(droppable_features))
droppable_features

# ## 17 columns can be removed at the beginning.
