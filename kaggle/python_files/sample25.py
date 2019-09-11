#!/usr/bin/env python
# coding: utf-8

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



# In[ ]:


from imageai.Detection import ObjectDetection
import os

model = ObjectDetection()
execution_path = os.getcwd()
model.setModelTypeAsRetinaNet()
model.setModelPath('../input/imageai/resnet50_coco_best_v2.0.1.h5') #downloaded the model from https://www.kaggle.com/anu0012/imageai#resnet50_coco_best_v2.0.1.h5
model.loadModel()

# **Detecting objects in an Image**

# In[ ]:


detections = model.detectObjectsFromImage(input_image='../input/testimagesforobjectdetection/image.png', minimum_percentage_probability=50, output_image_path=os.path.join(execution_path , "image_new.png"))

# **Printing out the detected images and their probablities**

# In[ ]:


for eachObjectDetected in detections:
    print(eachObjectDetected)

# **Displaying the output image with bounding boxes and predictions**

# In[ ]:


from IPython.display import Image
Image("image_new.png")
