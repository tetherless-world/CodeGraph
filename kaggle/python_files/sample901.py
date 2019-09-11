#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import time
print(os.listdir("../input"))

# ## The data given for the MS Malware Prediction challenge is huge!
# #### Training set - 4.08 GB
# #### Test Set - 3.54 GB
# 
# #### When the normal read_csv() function is applied to read the data the processing time is way too long and the kernel eventually dies given the huge size.

# # How to solve this issue??

# ### Let's try to determine the column data types to check whether the data types are upscaled and taking unnecessary space

# ### To understand the datatypes of each column we don't need to load the whole dataframe. So let's load a single row just to understand the datatypes. We can achieve this by using chunksize parameter

# In[ ]:


reader = pd.read_csv('../input/train.csv',chunksize=10)
type(reader)

# In[ ]:


sample_chunk = None
for chunk in reader:
    sample_chunk = chunk
    print(sample_chunk)
    break

# In[ ]:


type(sample_chunk)

# ### Voila! We have a dataframe to inspect now!

# In[ ]:


columns = list(sample_chunk.columns)
print("Number of columns: {}".format(len(columns)))

# In[ ]:


sample_chunk.info()

# ### So here we have the datatypes. Let's have a look at the data itself.

# In[ ]:


sample_chunk

# ### There's a way to optimise for the reading issue
# 
# *  Load objects as categories.
# *  Binary values are switched to int8
# *  Binary values with missing values are switched to float16 (int does not understand nan)
# *  64 bits encoding are all switched to 32, or 16 of possible

# #### This can be achieved using dtype parameter in read_csv  predefine the datatypes for each column.
# ##### Let's create the variable with suitable values in it.

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


t_start = time.clock()
train = pd.read_csv('../input/train.csv',dtype=dtypes)
t_end = time.clock()

print("Start Time: {}".format(t_start))
print("End Time: {}".format(t_end))

# In[ ]:


train.info()

# #### Voila!! Look at the memory usage 1.6 GB . Now all normal functions can be performed on this data and the loading issue is solved. Similarly the test file can be loaded.
# Reference: https://www.kaggle.com/theoviel/load-the-totality-of-the-data/notebook

# In[ ]:



