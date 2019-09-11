#!/usr/bin/env python
# coding: utf-8

# ## This Notebook Demonstrates:
# 1. Reading the data in python, preparing it for analysis, and adjusting the labels to contain underscores
# 2. The code that simplfies a Raw drawing to the Simplified drawing
# 3. How to make a submission file with predictions in the required format

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

# In[ ]:


import warnings
warnings.filterwarnings('ignore') # to suppress some matplotlib deprecation warnings

import ast
import math

# Have you installed your own package in Kernels yet? 
# If you need to, you can use the "Settings" bar on the right to install `simplification`
from simplification.cutil import simplify_coords

import matplotlib.pyplot as plt
import matplotlib.style as style



# ### Let's read some of the training data

# In[ ]:


data = pd.read_csv('../input/train_simplified/roller coaster.csv',
                   index_col='key_id',
                   nrows=100)
data.head()

# ### Fixing labels
# 
# Notice the `word` values for this label have a space. This aligns with the original Quick Draw data set that is already public. The Kaggle metric code for MAP@K requires each label to be separated by a single space. It parses "roller coaster" as two labels, "roller" and "coaster". Thus, `word` values with spaces need to be updated to use underscores. This is easily done, e.g.,

# In[ ]:


data['word'] = data['word'].replace(' ', '_', regex=True)
data.head()

# ### Let's look at some images
# We're going to grab the first 10 images from the `test_raw.csv` file. Since the `word` values are read as a string, we need to convert them to a list using the `ast.literal_eval` function.

# In[ ]:


test_raw = pd.read_csv('../input/test_raw.csv', index_col='key_id')
first_ten_ids = test_raw.iloc[:10].index
raw_images = [ast.literal_eval(lst) for lst in test_raw.loc[first_ten_ids, 'drawing'].values]

# ## From Raw to Simplified
# 
# This code demonstrates how the `simplified` data was generated from the `raw` data.
# 
# (Code by Jonas Jongejan)

# In[ ]:


def resample(x, y, spacing=1.0):
    output = []
    n = len(x)
    px = x[0]
    py = y[0]
    cumlen = 0
    pcumlen = 0
    offset = 0
    for i in range(1, n):
        cx = x[i]
        cy = y[i]
        dx = cx - px
        dy = cy - py
        curlen = math.sqrt(dx*dx + dy*dy)
        cumlen += curlen
        while offset < cumlen:
            t = (offset - pcumlen) / curlen
            invt = 1 - t
            tx = px * invt + cx * t
            ty = py * invt + cy * t
            output.append((tx, ty))
            offset += spacing
        pcumlen = cumlen
        px = cx
        py = cy
    output.append((x[-1], y[-1]))
    return output
  
def normalize_resample_simplify(strokes, epsilon=1.0, resample_spacing=1.0):
    if len(strokes) == 0:
        raise ValueError('empty image')

    # find min and max
    amin = None
    amax = None
    for x, y, _ in strokes:
        cur_min = [np.min(x), np.min(y)]
        cur_max = [np.max(x), np.max(y)]
        amin = cur_min if amin is None else np.min([amin, cur_min], axis=0)
        amax = cur_max if amax is None else np.max([amax, cur_max], axis=0)

    # drop any drawings that are linear along one axis
    arange = np.array(amax) - np.array(amin)
    if np.min(arange) == 0:
        raise ValueError('bad range of values')

    arange = np.max(arange)
    output = []
    for x, y, _ in strokes:
        xy = np.array([x, y], dtype=float).T
        xy -= amin
        xy *= 255.
        xy /= arange
        resampled = resample(xy[:, 0], xy[:, 1], resample_spacing)
        simplified = simplify_coords(resampled, epsilon)
        xy = np.around(simplified).astype(np.uint8)
        output.append(xy.T.tolist())

    return output

# In[ ]:


simplified_drawings = []
for drawing in raw_images:
    simplified_drawing = normalize_resample_simplify(drawing)
    simplified_drawings.append(simplified_drawing)

# ## Viewing  Drawings
# Aren't these fun to plot?!?!

# In[ ]:


for index, raw_drawing in enumerate(raw_images, 0):
    
    plt.figure(figsize=(6,3))
    
    for x,y,t in raw_drawing:
        plt.subplot(1,2,1)
        plt.plot(x, y, marker='.')
        plt.axis('off')

    plt.gca().invert_yaxis()
    plt.axis('equal')

    for x,y in simplified_drawings[index]:
        plt.subplot(1,2,2)
        plt.plot(x, y, marker='.')
        plt.axis('off')

    plt.gca().invert_yaxis()
    plt.axis('equal')
    plt.show()  

# ## Making a Submission

# In[ ]:


submission = pd.read_csv('../input/sample_submission.csv', index_col='key_id')
# Don't forget, your multi-word labels need underscores instead of spaces!
my_favorite_words = ['donut', 'roller_coaster', 'smiley_face']  
submission['word'] = " ".join(my_favorite_words)
submission.to_csv('my_favorite_words.csv')

# In[ ]:


submission.head()
