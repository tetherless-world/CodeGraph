#!/usr/bin/env python
# coding: utf-8

# In[367]:


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

# In[368]:


# Import libraries
from __future__ import absolute_import, division, print_function, unicode_literals


# Import TensorFlow
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

# Helper libraries
import math
import numpy as np
import matplotlib.pyplot as plt

# Improve progress bar display
import tqdm
import tqdm.auto
tqdm.tqdm = tqdm.auto.tqdm


print(tf.__version__)

# This will go away in the future.
# If this gives an error, you might be running TensorFlow 2 or above
# If so, then just comment out this line and run this cell again
tf.enable_eager_execution()  

# In[369]:


train_dataset = pd.read_csv("../input/train.csv")
test_dataset = pd.read_csv("../input/test.csv")

# ## Preprocess the data

# In[370]:


# Split the datasets into features and labels

features_train = train_dataset.iloc[:, 1:]
labels_train = train_dataset.iloc[:, 0:1].values

# In[371]:


training_dataset = (
    tf.data.Dataset.from_tensor_slices(
        (
            tf.cast(features_train.values, tf.float32),
            tf.cast(labels_train, tf.int32)
        )
    )
)

# In[372]:


testing_dataset = (
    tf.data.Dataset.from_tensor_slices(
        (
            tf.cast(test_dataset.values, tf.float32)
        )
    )
)

# In[373]:


# The map function applies the normalize function to each element in the train
# and test datasets
def normalize(images, labels):
  images = tf.cast(images, tf.float32)
  images /= 255
  return images, labels

# The map function applies the normalize function to each element in the train
# and test datasets
training_dataset = training_dataset.map(normalize)

def reshape_it(images, labels):
    images = tf.reshape(images, [28,28,1])
    return images, labels

training_dataset = training_dataset.map(reshape_it)

# In[374]:


print(training_dataset)

# Visualize our input data

# In[375]:


# Take a single image in the training dataset
for image, label in training_dataset.take(1):
  break
image = image.numpy().reshape((28,28))

# Plot the image
plt.figure()
plt.imshow(image, cmap=plt.cm.binary)
plt.colorbar()
plt.grid(False)
plt.show()

# In[376]:


# Verify the data in the training set
class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
plt.figure(figsize=(10,10))
i = 0
for (image, label) in training_dataset.take(25):
    image = image.numpy().reshape((28,28))
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(image, cmap=plt.cm.binary)
    plt.xlabel(class_names[label])
    i += 1
plt.show()

# Create our keras layers.

# In[377]:


model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), padding='same', activation=tf.nn.relu,
                           input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2), strides=2),
    tf.keras.layers.Conv2D(64, (3,3), padding='same', activation=tf.nn.relu),
    tf.keras.layers.MaxPooling2D((2, 2), strides=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(10,  activation=tf.nn.softmax)
])

# Compile the model.

# In[378]:


model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train our model.

# In[379]:


print(training_dataset)

# In[380]:


BATCH_SIZE = 32
TRAIN_DATA_SIZE = len(train_dataset.index)
TEST_DATA_SIZE = len(test_dataset.index)

training_dataset = training_dataset.repeat().shuffle(TRAIN_DATA_SIZE).batch(BATCH_SIZE)

# In[381]:


print(training_dataset)

# In[382]:


model.fit(training_dataset, epochs=5, steps_per_epoch=math.ceil(TRAIN_DATA_SIZE/BATCH_SIZE))

# Make predictions on the test dataset

# In[383]:


def reshape_test(images):
    images = tf.reshape(images, [-1, 28,28,1])
    return images

testing_dataset = testing_dataset.map(reshape_test)

# In[386]:


# Predict the results:

print(testing_dataset)

# In[387]:


output = model.predict(testing_dataset,steps=math.ceil(TEST_DATA_SIZE), verbose=True)
