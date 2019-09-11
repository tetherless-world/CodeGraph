#!/usr/bin/env python
# coding: utf-8

# # This kernel show how to extract image features using keras and low memory use.

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

# In[2]:


from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from keras.applications import xception
from keras.applications import inception_v3

# Thanks https://www.kaggle.com/gaborfodor/keras-pretrained-models

# In[3]:



# In[4]:


from os import listdir, makedirs
from os.path import join, exists, expanduser

# In[5]:


cache_dir = expanduser(join('~', '.keras'))
if not exists(cache_dir):
    makedirs(cache_dir)
models_dir = join(cache_dir, 'models')
if not exists(models_dir):
    makedirs(models_dir)

# In[6]:



# In[7]:



# 

# In[8]:


from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
import numpy as np

model = VGG16(weights='imagenet', include_top=False)
model.summary()

# In[9]:



# In[10]:


import zipfile

# In[11]:


myzip = zipfile.ZipFile('../input/avito-demand-prediction/train_jpg.zip')
files_in_zip = myzip.namelist()
for idx, file in enumerate(files_in_zip[:5]):
    if file.endswith('.jpg'):
        myzip.extract(file, path=file.split('/')[3])
myzip.close()

# In[12]:



# In[13]:



# In[14]:


img_path = './856e74b8c46edcf0c0e23444eab019bfda63687bb70a3481955cc6ab86e39df2.jpg/data/competition_files/train_jpg/856e74b8c46edcf0c0e23444eab019bfda63687bb70a3481955cc6ab86e39df2.jpg'
img = image.load_img(img_path, target_size=(224, 224))

# In[15]:


img

# In[16]:


x = image.img_to_array(img)  # 3 dims(3, 224, 224)
x = np.expand_dims(x, axis=0)  # 4 dims(1, 3, 224, 224)
x = preprocess_input(x)

# In[17]:


features = model.predict(x)

# In[18]:


features.reshape((25088,))

# 25088 dim feature is much bigger to use, you'd better try to use some reduce dims model like **PCA** etc, then you can merge these features to your regression model.

# In[ ]:



