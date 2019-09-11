#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from sklearn.model_selection import KFold
import warnings
import gc
import time
import sys
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')
from sklearn import metrics

plt.style.use('seaborn')
sns.set(font_scale=2)
pd.set_option('display.max_columns', 500)

# # 1. Read and check dataset

# ## 1.1 Read dataset

# - This parted was taken from the helpful kernel. https://www.kaggle.com/theoviel/load-the-totality-of-the-data

# In[ ]:


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

# In[ ]:



# In[ ]:


print(train.shape, test.shape)

# - You can see that the datasets are large.

# ## 1.2 Check the target

# In[ ]:


train['HasDetections'].value_counts().plot.bar()
plt.title('HasDetections(target)')

# - Wow, very-well balanced target! Fun with this competition :).

# ## 1.2 Check the dataset

# In[ ]:


# checking missing data
total = train.isnull().sum().sort_values(ascending = False)
percent = (train.isnull().sum()/train.isnull().count()*100).sort_values(ascending = False)
missing_train_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])


# In[ ]:


missing_train_data.head(50)

# - PuaMode, Census_ProcessorClass, DefaultBrowsersIdentifier, Census_IsFlightingInternal and Census_InternalBatteryType have over 70% null data.
# - Let's check their distribution regarding to the target.
# - Because datasets are large, let's compare the distributions using 10% of train.[](http://)

# In[ ]:


train_small = train # train.sample(frac=0.2).copy() # not small for now

# ### 1.1.1 PuaMode

# In[ ]:


print(train_small['PuaMode'].dtypes)

# In[ ]:


sns.countplot(x='PuaMode', hue='HasDetections',data=train_small)
plt.show()

# - Some difference exists there. But, the samples are quite few, so remove this feature.

# ### 1.1.2 Census_ProcessorClass

# In[ ]:


print(train_small['Census_ProcessorClass'].dtypes)

# In[ ]:


sns.countplot(x='Census_ProcessorClass', hue='HasDetections',data=train_small)
plt.show()

# - The meaning of 'Census_ProcessorClassr' is 'Number of logical cores in the processor'.
# - You can check that the more logical cores, the more probable infection with malwares.
# - This feature could be a good features only or component for the combinations with other features. Keep this and think it!

# ### 1.1.3 DefaultBrosersIdentifier

# In[ ]:


print(train_small['DefaultBrowsersIdentifier'].dtypes)

# In[ ]:


fig, ax = plt.subplots(1, 2, figsize=(16, 8))
sns.kdeplot(train_small.loc[train['HasDetections'] == 0, 'DefaultBrowsersIdentifier'], ax=ax[0], label='NoDetection(0)')
sns.kdeplot(train_small.loc[train['HasDetections'] == 1, 'DefaultBrowsersIdentifier'], ax=ax[0], label='HasDetection(1)')

train_small.loc[train['HasDetections'] == 0, 'DefaultBrowsersIdentifier'].hist(ax=ax[1])
train_small.loc[train['HasDetections'] == 1, 'DefaultBrowsersIdentifier'].hist(ax=ax[1])
ax[1].legend(['NoDetection(0)', 'HasDetection(1)'])

plt.show()

# - DefaultBrowsersIdentifier means ' ID for the machine's default browser'.
# - Is this feature meaningful?

# ### 1.1.4 Census_IsFightingInternal

# In[ ]:


print(train_small['Census_IsFlightingInternal'].dtypes)

# In[ ]:


fig, ax = plt.subplots(1, 2, figsize=(16, 8))
sns.kdeplot(train_small.loc[train['HasDetections'] == 0, 'Census_IsFlightingInternal'], ax=ax[0], label='NoDetection(0)')
sns.kdeplot(train_small.loc[train['HasDetections'] == 1, 'Census_IsFlightingInternal'], ax=ax[0], label='HasDetection(1)')

train_small.loc[train['HasDetections'] == 0, 'Census_IsFlightingInternal'].hist(ax=ax[1])
train_small.loc[train['HasDetections'] == 1, 'Census_IsFlightingInternal'].hist(ax=ax[1])
ax[1].legend(['NoDetection(0)', 'HasDetection(1)'])

plt.show()

# In[ ]:


train_small.loc[train['HasDetections'] == 1, 'Census_IsFlightingInternal'].value_counts()

# In[ ]:


train_small.loc[train['HasDetections'] == 0, 'Census_IsFlightingInternal'].value_counts()

# - As you can see, almost value of 'Census_IsFlightingInternal' is 0.0. Just remove.

# ### 1.1.5 Census_InternalBatteryType

# In[ ]:


print(train_small['Census_InternalBatteryType'].dtypes)

# In[ ]:


train_small['Census_InternalBatteryType'].value_counts()

# - I think this feature means the type of batteries of each machine.
# - Oh, no....These days, most batteries are lithum-ion battery.
# - So, Let's group them into lithum-batter group and non0-lithum-battery group

# In[ ]:


def group_battery(x):
    x = x.lower()
    if 'li' in x:
        return 1
    else:
        return 0
    
train_small['Census_InternalBatteryType'] = train_small['Census_InternalBatteryType'].apply(group_battery)

# In[ ]:


sns.countplot(x='Census_InternalBatteryType', hue='HasDetections',data=train_small)
plt.show()

# - The difference is quite small. Do you think that some malwares recognize and select machine based on the type of battery?
# - Battery is very important part for life of machine. I think that malware will focus on other hardware and software parts of machine. remove this.

# In[ ]:


null_cols_to_remove = ['DefaultBrowsersIdentifier', 'PuaMode',
                       'Census_IsFlightingInternal', 'Census_InternalBatteryType']

train.drop(null_cols_to_remove, axis=1, inplace=True)
test.drop(null_cols_to_remove, axis=1, inplace=True)

# ## 2. Exploratory data analysis

# - Exploring data is an exciting journey. During this, we can mine some interesting intuition for high AUC!

# ## 2.1 Categorical features

# In[ ]:


categorical_features = [
        'ProductName',                                          
        'EngineVersion',                                        
        'AppVersion',                                           
        'AvSigVersion',                                         
        'Platform',                                             
        'Processor',                                            
        'OsVer',                                                
        'OsPlatformSubRelease',                                 
        'OsBuildLab',                                           
        'SkuEdition',                                           
        'SmartScreen',                                          
        'Census_MDC2FormFactor',                                
        'Census_DeviceFamily',                                  
        'Census_PrimaryDiskTypeName',                           
        'Census_ChassisTypeName',                               
        'Census_PowerPlatformRoleName',                         
        'Census_OSVersion',                                     
        'Census_OSArchitecture',                                
        'Census_OSBranch',                                      
        'Census_OSEdition',                                     
        'Census_OSSkuName',                                     
        'Census_OSInstallTypeName',                             
        'Census_OSWUAutoUpdateOptionsName',                     
        'Census_GenuineStateName',                              
        'Census_ActivationChannel',                             
        'Census_FlightRing',                                    
]

# ### 2.1.1 ProductName - Defender state information e.g. win8defender

# In[ ]:


def plot_category_percent_of_target(col):
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    cat_percent = train_small[[col, 'HasDetections']].groupby(col, as_index=False).mean()
    cat_size = train_small[col].value_counts().reset_index(drop=False)
    cat_size.columns = [col, 'count']
    cat_percent = cat_percent.merge(cat_size, on=col, how='left')
    cat_percent['HasDetections'] = cat_percent['HasDetections'].fillna(0)
    cat_percent = cat_percent.sort_values(by='count', ascending=False)[:20]
    sns.barplot(ax=ax, x='HasDetections', y=col, data=cat_percent, order=cat_percent[col])

    for i, p in enumerate(ax.patches):
        ax.annotate('{}'.format(cat_percent['count'].values[i]), (p.get_width(), p.get_y()+0.5), fontsize=20)

    plt.xlabel('% of HasDetections(target)')
    plt.ylabel(col)
    plt.show()

# In[ ]:


col = categorical_features[0]
plot_category_percent_of_target(col)

# ### 2.1.2 EngineVersion - Defender state information e.g. 1.1.12603.0

# In[ ]:


col = categorical_features[1]
plot_category_percent_of_target(col)

# ### 2.1.3 AppVersion - Defender state information e.g. 4.9.10586.0

# In[ ]:


col = categorical_features[2]
plot_category_percent_of_target(col)

# ### 2.1.4 AvSigVersion - Defender state information e.g. 1.217.1014.0

# In[ ]:


col = categorical_features[3]
plot_category_percent_of_target(col)

# ### 2.1.5 Platform - Calculates platform name (of OS related properties and processor property)

# In[ ]:


col = categorical_features[4]
plot_category_percent_of_target(col)

# ### 2.1.6 Processor - This is the process architecture of the installed operating system

# In[ ]:


col = categorical_features[5]
plot_category_percent_of_target(col)

# ### 2.1.7 Census_OSVersion - Numeric OS version Example - 10.0.10130.0

# In[ ]:


col = categorical_features[6]
plot_category_percent_of_target(col)

# ### 2.1.8 OsPlatformSubRelease - Returns the OS Platform sub-release (Windows Vista, Windows 7, Windows 8, TH1, TH2)

# In[ ]:


col = categorical_features[7]
plot_category_percent_of_target(col)

# ### 2.1.9 OsBuildLab - Build lab that generated the current OS. Example: 9600.17630.amd64fre.winblue_r7.150109-2022

# In[ ]:


col = categorical_features[8]
plot_category_percent_of_target(col)

# ### 2.1.10 SkuEdition - The goal of this feature is to use the Product Type defined in the MSDN to map to a 'SKU-Edition' name that is useful in population reporting. 

# In[ ]:


col = categorical_features[9]
plot_category_percent_of_target(col)

# 

# ### 2.1.11 SmartScreen - This is the SmartScreen enabled string value from registry. This is obtained by checking in order, 

# In[ ]:


col = categorical_features[10]
plot_category_percent_of_target(col)

# ### 2.1.12 Census_MDC2FormFactor - A grouping based on a combination of Device Census level hardware characteristics.**

# In[ ]:


col = categorical_features[11]
plot_category_percent_of_target(col)

# ### 2.1.13 Census_DeviceFamily - AKA DeviceClass. Indicates the type of device that an edition of the OS is intended for. 

# In[ ]:


col = categorical_features[12]
plot_category_percent_of_target(col)

# ### 2.1.14 Census_PrimaryDiskTotalCapacity - Amount of disk space on primary disk of the machine in MB

# In[ ]:


col = categorical_features[13]
plot_category_percent_of_target(col)

# ### 2.1.15 Census_ChassisTypeName - Retrieves a numeric representation of what type of chassis the machine has

# In[ ]:


col = categorical_features[14]
plot_category_percent_of_target(col)

# ### 2.1.16 Census_PowerPlatformRoleName - Indicates the OEM preferred power management profile

# In[ ]:


col = categorical_features[15]
plot_category_percent_of_target(col)

# ### 2.1.17 Census_OSVersion - Numeric OS version Example

# In[ ]:


col = categorical_features[16]
plot_category_percent_of_target(col)

# ### 2.1.18 Census_OSArchitecture - Architecture on which the OS is based

# In[ ]:


col = categorical_features[17]
plot_category_percent_of_target(col)

# ### 2.1.19 Branch of the OS extracted from the OsVersionFull

# In[ ]:


col = categorical_features[18]
plot_category_percent_of_target(col)

# ### 2.1.20 Census_OSEdition - Edition of the current OS

# In[ ]:


col = categorical_features[19]
plot_category_percent_of_target(col)

# ### 2.1.21 Census_OSSkuName - OS edition friendly name

# In[ ]:


col = categorical_features[20]
plot_category_percent_of_target(col)

# ### 2.1.22 Census_PrimaryDiskTypeName - Friendly name of Primary Disk Type - HDD or SSD

# In[ ]:


col = categorical_features[21]
plot_category_percent_of_target(col)

# ### 2.1.23 Census_OSWUAutoUpdateOptionsName - Friendly name of the WindowsUpdate auto-update settings on the machine.

# In[ ]:


col = categorical_features[22]
plot_category_percent_of_target(col)

# ### 2.1.24 Census_GenuineStateName - Friendly name of OSGenuineStateID. 0 = Genuine

# In[ ]:


col = categorical_features[23]
plot_category_percent_of_target(col)

# ### 2.1.25 Census_ActivationChannel - Retail license key or Volume license key for a machine.

# In[ ]:


col = categorical_features[24]
plot_category_percent_of_target(col)

# ### 2.1.26 Census_IsFlightingInternal - NA

# In[ ]:


col = categorical_features[25]
plot_category_percent_of_target(col)

# ## 2.2 numeric features

# In[ ]:


numeric_features = [
        'IsBeta',                                               
        'RtpStateBitfield',                                     
        'IsSxsPassiveMode',                                     
        'DefaultBrowsersIdentifier',                            
        'AVProductStatesIdentifier',                            
        'AVProductsInstalled',                                  
        'AVProductsEnabled',                                    
        'HasTpm',                                               
        'CountryIdentifier',                                    
        'CityIdentifier',                                       
        'OrganizationIdentifier',                               
        'GeoNameIdentifier',                                    
        'LocaleEnglishNameIdentifier',                          
        'OsBuild',                                              
        'OsSuite',                                              
        'IsProtected',                                          
        'AutoSampleOptIn',                                      
        'SMode',                                                
        'IeVerIdentifier',                                      
        'Firewall',                                             
        'UacLuaenable',                                         
        'Census_OEMNameIdentifier',                             
        'Census_OEMModelIdentifier',                            
        'Census_ProcessorCoreCount',                            
        'Census_ProcessorManufacturerIdentifier',               
        'Census_ProcessorModelIdentifier',                      
        'Census_PrimaryDiskTotalCapacity',                      
        'Census_SystemVolumeTotalCapacity',                     
        'Census_HasOpticalDiskDrive',                           
        'Census_TotalPhysicalRAM',                              
        'Census_InternalPrimaryDiagonalDisplaySizeInInches',    
        'Census_InternalPrimaryDisplayResolutionHorizontal',    
        'Census_InternalPrimaryDisplayResolutionVertical',      
        'Census_InternalBatteryNumberOfCharges',                
        'Census_OSBuildNumber',                                 
        'Census_OSBuildRevision',                               
        'Census_OSInstallLanguageIdentifier',                   
        'Census_OSUILocaleIdentifier',                          
        'Census_IsPortableOperatingSystem',                     
        'Census_IsFlightsDisabled',                             
        'Census_ThresholdOptIn',                                
        'Census_FirmwareManufacturerIdentifier',                
        'Census_FirmwareVersionIdentifier',                     
        'Census_IsSecureBootEnabled',                           
        'Census_IsWIMBootEnabled',                              
        'Census_IsVirtualDevice',                               
        'Census_IsTouchEnabled',                                
        'Census_IsPenCapable',                                  
        'Census_IsAlwaysOnAlwaysConnectedCapable',              
        'Wdft_IsGamer',                                         
        'Wdft_RegionIdentifier',                                
]

# ### 2.2.1 IsBeta - Defender state information e.g. false

# In[ ]:


def plot_category_percent_of_target_for_numeric(col):
    fig, ax = plt.subplots(1, 2, figsize=(20, 8))
    cat_percent = train_small[[col, 'HasDetections']].groupby(col, as_index=False).mean()
    cat_size = train_small[col].value_counts().reset_index(drop=False)
    cat_size.columns = [col, 'count']
    cat_percent = cat_percent.merge(cat_size, on=col, how='left')
    cat_percent['HasDetections'] = cat_percent['HasDetections'].fillna(0)
    cat_percent = cat_percent.sort_values(by='count', ascending=False)[:20]
    cat_percent[col] = cat_percent[col].astype('category')
    sns.barplot(ax=ax[0], x='HasDetections', y=col, data=cat_percent,  order=cat_percent[col])

    for i, p in enumerate(ax[0].patches):
        ax[0].annotate('{}'.format(cat_percent['count'].values[i]), (p.get_width(), p.get_y()+0.5), fontsize=20)

    ax[0].set_title('Barplot sorted by count', fontsize=20)

    sns.barplot(ax=ax[1], x='HasDetections', y=col, data=cat_percent)
    for i, p in enumerate(ax[0].patches):
        ax[1].annotate('{}'.format(cat_percent['count'].sort_index().values[i]), (0, p.get_y()+0.6), fontsize=20)
    ax[1].set_title('Barplot sorted by index', fontsize=20)

    plt.xlabel('% of HasDetections(target)')
    plt.ylabel(col)
    plt.subplots_adjust(wspace=0.5, hspace=0)
    plt.show()

def plot_kde_hist_for_numeric(col):
    fig, ax = plt.subplots(1, 2, figsize=(16, 8))
    sns.kdeplot(train_small.loc[train['HasDetections'] == 0, col], ax=ax[0], label='NoDetection(0)')
    sns.kdeplot(train_small.loc[train['HasDetections'] == 1, col], ax=ax[0], label='HasDetection(1)')

    train_small.loc[train['HasDetections'] == 0, col].hist(ax=ax[1], bins=100)
    train_small.loc[train['HasDetections'] == 1, col].hist(ax=ax[1], bins=100)

    plt.suptitle(col, fontsize=30)
    ax[0].set_yscale('log')
    ax[0].set_title('KDE plot')
    
    ax[1].set_title('Histogram')
    ax[1].legend(['NoDetection(0)', 'HasDetection(1)'])
    ax[1].set_yscale('log')
    plt.show()

# In[ ]:


col = numeric_features[0]

plot_kde_hist_for_numeric(col)
plot_category_percent_of_target_for_numeric(col)

# ### 2.2.2 RtpStateBitfield - NA

# In[ ]:


col = numeric_features[1]

plot_kde_hist_for_numeric(col)
plot_category_percent_of_target_for_numeric(col)

# ### 2.2.3 IsSxsPassiveMode - NA

# In[ ]:


col = numeric_features[2]

plot_kde_hist_for_numeric(col)
plot_category_percent_of_target_for_numeric(col)

# ### 2.2.4 DefaultBrowsersIdentifier - ID for the machine's default browser

# In[ ]:


col = numeric_features[3]

plot_kde_hist_for_numeric(col)
# plot_category_percent_of_target_for_numeric(col)

# ### 2.2.5 AVProductStatesIdentifier - ID for the specific configuration of a user's antivirus software

# In[ ]:


col = numeric_features[4]

plot_kde_hist_for_numeric(col)
# plot_category_percent_of_target_for_numeric(col)

# ### 2.2.6 AVProductsInstalled - NA

# In[ ]:


col = numeric_features[5]

plot_kde_hist_for_numeric(col)
plot_category_percent_of_target_for_numeric(col)

# ### 2.2.7 AVProductsEnabled - NA

# In[ ]:


col = numeric_features[6]

plot_kde_hist_for_numeric(col)
plot_category_percent_of_target_for_numeric(col)

# ### 2.2.8 HasTpm - True if machine has tpm

# In[ ]:


col = numeric_features[7]

plot_kde_hist_for_numeric(col)
plot_category_percent_of_target_for_numeric(col)

# ### 2.2.9 CountryIdentifier - ID for the country the machine is located in

# In[ ]:


col = numeric_features[8]

plot_kde_hist_for_numeric(col)
plot_category_percent_of_target_for_numeric(col)

# ### 2.2.10 CityIdentifier - ID for the city the machine is located in

# In[ ]:


col = numeric_features[9]

plot_kde_hist_for_numeric(col)
plot_category_percent_of_target_for_numeric(col)

# ### 2.2.11 OrganizationIdentifier - ID for the organization the machine belongs in, organization ID is mapped to both specific companies and broad industries

# In[ ]:


col = numeric_features[10]

plot_kde_hist_for_numeric(col)
plot_category_percent_of_target_for_numeric(col)

# ### 2.2.12 GeoNameIdentifier - ID for the geographic region a machine is located in

# In[ ]:


col = numeric_features[11]

plot_kde_hist_for_numeric(col)
# plot_category_percent_of_target_for_numeric(col)

# ### 2.2.13 LocaleEnglishNameIdentifier - English name of Locale ID of the current user

# In[ ]:


col = numeric_features[12]

plot_kde_hist_for_numeric(col)
plot_category_percent_of_target_for_numeric(col)

# ### 2.2.14 OsBuild - Build of the current operating system

# In[ ]:


col = numeric_features[13]

plot_kde_hist_for_numeric(col)
plot_category_percent_of_target_for_numeric(col)

# ### 2.2.15 OsSuite - Product suite mask for the current operating system.

# In[ ]:


col = numeric_features[14]

plot_kde_hist_for_numeric(col)
plot_category_percent_of_target_for_numeric(col)

# ### 2.2.16 IsProtected - This is a calculated field derived from the Spynet Report's AV Products field.
# - a. TRUE if there is at least one active and up-to-date antivirus product running on this machine. b. FALSE if there is no active AV product on this machine, or if the AV is active, but is not receiving the latest updates. c. null if there are no Anti Virus Products in the report. Returns: Whether a machine is protected.

# In[ ]:


col = numeric_features[15]

# plot_kde_hist_for_numeric(col)
plot_category_percent_of_target_for_numeric(col)

# ### 2.2.17 AutoSampleOptIn - This is the SubmitSamplesConsent value passed in from the service, available on CAMP 9+

# In[ ]:


col = numeric_features[16]

# plot_kde_hist_for_numeric(col)
plot_category_percent_of_target_for_numeric(col)

# ### 2.2.18 SMode - This field is set to true when the device is known to be in 'S Mode', as in, Windows 10 S mode, where only Microsoft Store apps can be installed

# In[ ]:


col = numeric_features[17]

# plot_kde_hist_for_numeric(col)
plot_category_percent_of_target_for_numeric(col)

# ### 2.2.19 IeVerIdentifier - NA

# In[ ]:


col = numeric_features[18]

plot_kde_hist_for_numeric(col)
plot_category_percent_of_target_for_numeric(col)

# ### 2.2.20 Firewall - This attribute is true (1) for Windows 8.1 and above if windows firewall is enabled, as reported by the service.

# In[ ]:


col = numeric_features[19]

# plot_kde_hist_for_numeric(col)
plot_category_percent_of_target_for_numeric(col)

# ### 2.2.21 UacLuaenable - This attribute reports whether or not the "administrator in Admin Approval Mode" user type is disabled or enabled in UAC. The value reported is obtained by reading the regkey HKLM\SOFTWARE\Microsoft\Windows\CurrentVersion\Policies\System\EnableLUA.

# In[ ]:


col = numeric_features[20]

plot_kde_hist_for_numeric(col)
plot_category_percent_of_target_for_numeric(col)

# ### 2.2.22 Census_OEMNameIdentifier - NA

# In[ ]:


col = numeric_features[21]

plot_kde_hist_for_numeric(col)
plot_category_percent_of_target_for_numeric(col)

# ### 2.2.23 Census_OEMModelIdentifier - NA

# In[ ]:


col = numeric_features[22]

plot_kde_hist_for_numeric(col)
plot_category_percent_of_target_for_numeric(col)

# ### 2.2.24 Census_ProcessorCoreCount - Number of logical cores in the processor

# In[ ]:


col = numeric_features[23]

plot_kde_hist_for_numeric(col)
plot_category_percent_of_target_for_numeric(col)

# ### 2.2.25 Census_ProcessorManufacturerIdentifier - NA

# In[ ]:


col = numeric_features[24]

plot_kde_hist_for_numeric(col)
plot_category_percent_of_target_for_numeric(col)

# ### 2.2.26 Census_ProcessorModelIdentifier - NA

# In[ ]:


col = numeric_features[25]

plot_kde_hist_for_numeric(col)
plot_category_percent_of_target_for_numeric(col)

# ### Census_PrimaryDiskTotalCapacity - Amount of disk space on primary disk of the machine in MB

# In[ ]:


col = numeric_features[26]

plot_kde_hist_for_numeric(col)
plot_category_percent_of_target_for_numeric(col)

# ### 2.2.28 Census_SystemVolumeTotalCapacity - The size of the partition that the System volume is installed on in MB

# In[ ]:


col = numeric_features[27]

plot_kde_hist_for_numeric(col)
plot_category_percent_of_target_for_numeric(col)

# ### 2.2.29 Census_HasOpticalDiskDrive - True indicates that the machine has an optical disk drive (CD/DVD)

# In[ ]:


col = numeric_features[28]

# plot_kde_hist_for_numeric(col)
plot_category_percent_of_target_for_numeric(col)

# ### 2.2.30 Census_TotalPhysicalRAM - Retrieves the physical RAM in MB

# In[ ]:


col = numeric_features[29]

plot_kde_hist_for_numeric(col)
plot_category_percent_of_target_for_numeric(col)

# ### 2.2.31 Census_InternalPrimaryDiagonalDisplaySizeInInches - Retrieves the physical diagonal length in inches of the primary display

# In[ ]:


col = numeric_features[30]

plot_kde_hist_for_numeric(col)
plot_category_percent_of_target_for_numeric(col)

# ### 2.2.32 Census_InternalPrimaryDisplayResolutionHorizontal - Retrieves the number of pixels in the horizontal direction of the internal display.

# In[ ]:


col = numeric_features[31]

plot_kde_hist_for_numeric(col)
plot_category_percent_of_target_for_numeric(col)

# ### 2.2.33 Census_InternalPrimaryDisplayResolutionVertical - Retrieves the number of pixels in the vertical direction of the internal display

# In[ ]:


col = numeric_features[32]

plot_kde_hist_for_numeric(col)
plot_category_percent_of_target_for_numeric(col)

# ### 2.2.35 Census_OSBuildNumber - OS Build number extracted from the OsVersionFull. Example - OsBuildNumber = 10512 or 10240

# In[ ]:


col = numeric_features[34]

plot_kde_hist_for_numeric(col)
plot_category_percent_of_target_for_numeric(col)

# ### 2.2.37 Census_OSInstallLanguageIdentifier - NA

# In[ ]:


col = numeric_features[36]

plot_kde_hist_for_numeric(col)
plot_category_percent_of_target_for_numeric(col)

# ### 2.2.38 Census_OSUILocaleIdentifier - NA

# In[ ]:


col = numeric_features[37]

plot_kde_hist_for_numeric(col)
plot_category_percent_of_target_for_numeric(col)

# ### 2.2.39 Census_IsPortableOperatingSystem - Indicates whether OS is booted up and running via Windows-To-Go on a USB stick.

# In[ ]:


col = numeric_features[38]

# plot_kde_hist_for_numeric(col)
plot_category_percent_of_target_for_numeric(col)

# ### 2.2.40 Census_IsFlightsDisabled - Indicates if the machine is participating in flighting.

# In[ ]:


col = numeric_features[39]

# plot_kde_hist_for_numeric(col)
plot_category_percent_of_target_for_numeric(col)

# ### 2.2.41 Census_ThresholdOptIn - NA

# In[ ]:


col = numeric_features[40]

# plot_kde_hist_for_numeric(col)
plot_category_percent_of_target_for_numeric(col)

# ### 2.2.42 Census_FirmwareManufacturerIdentifier - NA

# In[ ]:


col = numeric_features[41]

plot_kde_hist_for_numeric(col)
plot_category_percent_of_target_for_numeric(col)

# ### 2.2.43 Census_FirmwareVersionIdentifier - NA

# In[ ]:


col = numeric_features[42]

plot_kde_hist_for_numeric(col)
plot_category_percent_of_target_for_numeric(col)

# ### 2.2.44 Census_IsSecureBootEnabled - Indicates if Secure Boot mode is enabled.

# In[ ]:


col = numeric_features[43]

# plot_kde_hist_for_numeric(col)
plot_category_percent_of_target_for_numeric(col)

# ### 2.2.45 Census_IsWIMBootEnabled - NA

# In[ ]:


col = numeric_features[44]

# plot_kde_hist_for_numeric(col)
plot_category_percent_of_target_for_numeric(col)

# ### 2.2.46 Census_IsVirtualDevice - Identifies a Virtual Machine (machine learning model)

# In[ ]:


col = numeric_features[45]

# plot_kde_hist_for_numeric(col)
plot_category_percent_of_target_for_numeric(col)

# ### 2.2.47 Census_IsTouchEnabled - Is this a touch device ?

# In[ ]:


col = numeric_features[46]

# plot_kde_hist_for_numeric(col)
plot_category_percent_of_target_for_numeric(col)

# ### 2.2.48 Census_IsPenCapable - Is the device capable of pen input ?

# In[ ]:


col = numeric_features[47]

# plot_kde_hist_for_numeric(col)
plot_category_percent_of_target_for_numeric(col)

# ### 2.2.49 Census_IsAlwaysOnAlwaysConnectedCapable - Retreives information about whether the battery enables the device to be AlwaysOnAlwaysConnected .

# In[ ]:


col = numeric_features[48]

# plot_kde_hist_for_numeric(col)
plot_category_percent_of_target_for_numeric(col)

# ### 2.2.50 Wdft_IsGamer - Indicates whether the device is a gamer device or not based on its hardware combination.

# In[ ]:


col = numeric_features[49]

# plot_kde_hist_for_numeric(col)
plot_category_percent_of_target_for_numeric(col)

# ### 2.2.51 Wdft_RegionIdentifier - NA

# In[ ]:


col = numeric_features[50]

# plot_kde_hist_for_numeric(col)
plot_category_percent_of_target_for_numeric(col)

# ## 3. Correlations

# In[ ]:


corr = train_small.corr()['HasDetections']

# In[ ]:


abs(corr).sort_values(ascending=False)

# In[ ]:


def corr_heatmap(cols):
    correlations = train_small[cols+['HasDetections']].corr()
    # Create color map ranging between two colors
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(correlations, cmap=cmap, vmax=1.0, center=0, fmt='.2f',
                square=True, linewidths=.5, annot=True, cbar_kws={"shrink": .75})
    plt.show()

# In[ ]:


corr_heatmap(numeric_features[:10])

# In[ ]:


corr_heatmap(numeric_features[10:20])

# In[ ]:


corr_heatmap(numeric_features[20:30])

# In[ ]:


corr_heatmap(numeric_features[30:40])

# In[ ]:


corr_heatmap(numeric_features[40:])

# In[ ]:


corr = train_small.corr()

# In[ ]:


upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(np.bool))
upper.head()

# In[ ]:


threshold = 0.3
for i, df in (upper.iterrows()):
    for ele in df[df.abs() > threshold].items():
        if ele[0] == i:
            break
        else:
            plt.figure(figsize=(7, 7))
            sns.lmplot(x=i, y=ele[0], data=train_small[:100000], hue='HasDetections', palette='Set1', scatter_kws={'alpha':0.3})
            plt.show()
            print('{:50}, {:50} : {}'.format(i, ele[0], ele[1]))

# #### Many features are categorical and the pairs which have high correlations are also composed of categorial features.
# #### I think that the keypoint is to make some features which have categories with high probability of infection from malwares.
# #### some features are redundant.

# In[ ]:




# # To be updated. Please votes if you think this kernel is helpful!

# In[ ]:




# In[ ]:



