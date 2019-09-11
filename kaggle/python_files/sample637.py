#!/usr/bin/env python
# coding: utf-8

# In[2]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
fits =os.listdir("../input/subchallenge%201/")
print(fits)

# Any results you write to the current directory are saved as output.

# In[3]:


from astropy.utils.data import download_file
from astropy.io import fits

# In[16]:



for path in os.listdir("../input/subchallenge%201/"):
    kiki = os.path.join('../input/subchallenge%201/',path)
    img = fits.open(kiki)[0].data
    plt.imshow(image,cmap='gray')
    plt.colorbar()
    

# In[6]:


HDUList=fits.open('../input/subchallenge%201/sphere_irdis_psf_3.fits')

# In[7]:


image = HDUList[0].data

# In[10]:


import matplotlib.pyplot as plt

# In[17]:


plt.imshow(image)
plt.colorbar()

# In[ ]:



