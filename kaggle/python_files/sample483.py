#!/usr/bin/env python
# coding: utf-8

# *Poonam Ligade*
# 
# *1st Feb 2017*
# 
# 
# ----------
# 
# 
# This notebook is like note to self.
# 
# I am trying to understand various components of Artificial Neural Networks aka Deep Learning.
# 
# Hope it might be useful for someone else here.
# 
# I am designing neural net on MNIST handwritten digits images to identify their correct label i.e number in image.
# 
# You must have guessed its an image recognition task.
# 
# MNIST is called Hello world of Deep learning.
# 
# Lets start!!
# 
# This notebook is inspired from [Jeremy's][1] [Deep Learning][2] mooc and [Deep learning with python][3] book by Keras author [François Chollet][4] .
# 
# 
#   [1]: https://www.linkedin.com/in/howardjeremy/
#   [2]: http://course.fast.ai/
#   [3]: https://www.manning.com/books/deep-learning-with-python
#   [4]: https://research.google.com/pubs/105096.html

# **Import all required libraries**
# ===============================

# In[ ]:


    # This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense , Dropout , Lambda, Flatten
from keras.optimizers import Adam ,RMSprop
from sklearn.model_selection import train_test_split
from keras import  backend as K
from keras.preprocessing.image import ImageDataGenerator

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

# **Load Train and Test data**
# ============================

# In[ ]:


# create the training & test sets, skipping the header row with [1:]
train = pd.read_csv("../input/train.csv")
print(train.shape)
train.head()

# In[ ]:


test= pd.read_csv("../input/test.csv")
print(test.shape)
test.head()

# In[ ]:


X_train = (train.iloc[:,1:].values).astype('float32') # all pixel values
y_train = train.iloc[:,0].values.astype('int32') # only labels i.e targets digits
X_test = test.values.astype('float32')

# In[ ]:


X_train

# In[ ]:


y_train

# The output variable is an integer from 0 to 9. This is a **multiclass** classification problem.

# ## Data Visualization
# Lets look at 3 images from data set with their labels.

# In[ ]:


#Convert train datset to (num_images, img_rows, img_cols) format 
X_train = X_train.reshape(X_train.shape[0], 28, 28)

for i in range(6, 9):
    plt.subplot(330 + (i+1))
    plt.imshow(X_train[i], cmap=plt.get_cmap('gray'))
    plt.title(y_train[i]);

# In[ ]:


#expand 1 more dimention as 1 for colour channel gray
X_train = X_train.reshape(X_train.shape[0], 28, 28,1)
X_train.shape

# In[ ]:


X_test = X_test.reshape(X_test.shape[0], 28, 28,1)
X_test.shape

# **Preprocessing the digit images**
# ==================================

# **Feature Standardization**
# -------------------------------------
# 
# It is important preprocessing step.
# It is used to centre the data around zero mean and unit variance.

# In[ ]:


mean_px = X_train.mean().astype(np.float32)
std_px = X_train.std().astype(np.float32)

def standardize(x): 
    return (x-mean_px)/std_px

# *One Hot encoding of labels.*
# -----------------------------
# 
# A one-hot vector is a vector which is 0 in most dimensions, and 1 in a single dimension. In this case, the nth digit will be represented as a vector which is 1 in the nth dimension. 
# 
# For example, 3 would be [0,0,0,1,0,0,0,0,0,0].

# In[ ]:


from keras.utils.np_utils import to_categorical
y_train= to_categorical(y_train)
num_classes = y_train.shape[1]
num_classes

# Lets plot 10th label.

# In[ ]:


plt.title(y_train[9])
plt.plot(y_train[9])
plt.xticks(range(10));

# Oh its 3 !

# **Designing Neural Network Architecture**
# =========================================

# In[ ]:


# fix random seed for reproducibility
seed = 43
np.random.seed(seed)

# *Linear Model*
# --------------

# In[ ]:


from keras.models import  Sequential
from keras.layers.core import  Lambda , Dense, Flatten, Dropout
from keras.callbacks import EarlyStopping
from keras.layers import BatchNormalization, Convolution2D , MaxPooling2D

# Lets create a simple model from Keras Sequential layer.
# 
# 1. Lambda layer performs simple arithmetic operations like sum, average, exponentiation etc.
# 
#  In 1st layer of the model we have to define input dimensions of our data in (rows,columns,colour channel) format.
#  (In theano colour channel comes first)
# 
# 
# 2. Flatten will transform input into 1D array.
# 
# 
# 3. Dense is fully connected layer that means all neurons in previous layers will be connected to all neurons in fully connected layer.
#  In the last layer we have to specify output dimensions/classes of the model.
#  Here it's 10, since we have to output 10 different digit labels.

# In[ ]:


model= Sequential()
model.add(Lambda(standardize,input_shape=(28,28,1)))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))
print("input shape ",model.input_shape)
print("output shape ",model.output_shape)

# ***Compile network***
# -------------------
# 
# Before making network ready for training we have to make sure to add below things:
# 
#  1.  A loss function: to measure how good the network is
#     
#  2.  An optimizer: to update network as it sees more data and reduce loss
#     value
#     
#  3.  Metrics: to monitor performance of network

# In[ ]:


from keras.optimizers import RMSprop
model.compile(optimizer=RMSprop(lr=0.001),
 loss='categorical_crossentropy',
 metrics=['accuracy'])

# In[ ]:


from keras.preprocessing import image
gen = image.ImageDataGenerator()

# ## Cross Validation 

# In[ ]:


from sklearn.model_selection import train_test_split
X = X_train
y = y_train
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.10, random_state=42)
batches = gen.flow(X_train, y_train, batch_size=64)
val_batches=gen.flow(X_val, y_val, batch_size=64)

# In[ ]:


history=model.fit_generator(generator=batches, steps_per_epoch=batches.n, epochs=3, 
                    validation_data=val_batches, validation_steps=val_batches.n)

# In[ ]:


history_dict = history.history
history_dict.keys()

# In[ ]:


import matplotlib.pyplot as plt
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
epochs = range(1, len(loss_values) + 1)

# "bo" is for "blue dot"
plt.plot(epochs, loss_values, 'bo')
# b+ is for "blue crosses"
plt.plot(epochs, val_loss_values, 'b+')
plt.xlabel('Epochs')
plt.ylabel('Loss')

plt.show()

# In[ ]:


plt.clf()   # clear figure
acc_values = history_dict['acc']
val_acc_values = history_dict['val_acc']

plt.plot(epochs, acc_values, 'bo')
plt.plot(epochs, val_acc_values, 'b+')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')

plt.show()

# ## Fully Connected Model
# 
# Neurons in a fully connected layer have full connections to all activations in the previous layer, as seen in regular Neural Networks. 
# Adding another Dense Layer to model.

# In[ ]:


def get_fc_model():
    model = Sequential([
        Lambda(standardize, input_shape=(28,28,1)),
        Flatten(),
        Dense(512, activation='relu'),
        Dense(10, activation='softmax')
        ])
    model.compile(optimizer='Adam', loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# In[ ]:


fc = get_fc_model()
fc.optimizer.lr=0.01

# In[ ]:


history=fc.fit_generator(generator=batches, steps_per_epoch=batches.n, epochs=1, 
                    validation_data=val_batches, validation_steps=val_batches.n)

# ## Convolutional Neural Network
# CNNs are extremely efficient for images.
# 

# In[ ]:


from keras.layers import Convolution2D, MaxPooling2D

def get_cnn_model():
    model = Sequential([
        Lambda(standardize, input_shape=(28,28,1)),
        Convolution2D(32,(3,3), activation='relu'),
        Convolution2D(32,(3,3), activation='relu'),
        MaxPooling2D(),
        Convolution2D(64,(3,3), activation='relu'),
        Convolution2D(64,(3,3), activation='relu'),
        MaxPooling2D(),
        Flatten(),
        Dense(512, activation='relu'),
        Dense(10, activation='softmax')
        ])
    model.compile(Adam(), loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# In[ ]:


model= get_cnn_model()
model.optimizer.lr=0.01

# In[ ]:


history=model.fit_generator(generator=batches, steps_per_epoch=batches.n, epochs=1, 
                    validation_data=val_batches, validation_steps=val_batches.n)

# ## Data Augmentation
# It is tehnique of showing slighly different or new images to neural network to avoid overfitting. And  to achieve better generalization.
# In case you have very small dataset, you can use different kinds of data augmentation techniques to increase your data size. Neural networks perform better if you provide them more data.
# 
# Different data aumentation techniques are as follows:
# 1. Cropping
# 2. Rotating
# 3. Scaling
# 4. Translating
# 5. Flipping 
# 6. Adding Gaussian noise to input images etc.
# 

# In[ ]:


gen =ImageDataGenerator(rotation_range=8, width_shift_range=0.08, shear_range=0.3,
                               height_shift_range=0.08, zoom_range=0.08)
batches = gen.flow(X_train, y_train, batch_size=64)
val_batches = gen.flow(X_val, y_val, batch_size=64)

# In[ ]:


model.optimizer.lr=0.001
history=model.fit_generator(generator=batches, steps_per_epoch=batches.n, epochs=1, 
                    validation_data=val_batches, validation_steps=val_batches.n)

# ## Adding Batch Normalization
# 
# BN helps to fine tune hyperparameters more better and train really deep neural networks.

# In[ ]:


from keras.layers.normalization import BatchNormalization

def get_bn_model():
    model = Sequential([
        Lambda(standardize, input_shape=(28,28,1)),
        Convolution2D(32,(3,3), activation='relu'),
        BatchNormalization(axis=1),
        Convolution2D(32,(3,3), activation='relu'),
        MaxPooling2D(),
        BatchNormalization(axis=1),
        Convolution2D(64,(3,3), activation='relu'),
        BatchNormalization(axis=1),
        Convolution2D(64,(3,3), activation='relu'),
        MaxPooling2D(),
        Flatten(),
        BatchNormalization(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dense(10, activation='softmax')
        ])
    model.compile(Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# In[ ]:


model= get_bn_model()
model.optimizer.lr=0.01
history=model.fit_generator(generator=batches, steps_per_epoch=batches.n, epochs=1, 
                    validation_data=val_batches, validation_steps=val_batches.n)

# ## Submitting Predictions to Kaggle.
# Make sure you use full train dataset here to train model and predict on test set.
# 

# In[ ]:


model.optimizer.lr=0.01
gen = image.ImageDataGenerator()
batches = gen.flow(X, y, batch_size=64)
history=model.fit_generator(generator=batches, steps_per_epoch=batches.n, epochs=3)

# In[ ]:


predictions = model.predict_classes(X_test, verbose=0)

submissions=pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),
                         "Label": predictions})
submissions.to_csv("DR.csv", index=False, header=True)

# More to come . Please upvote if you find it useful.
# 
# You can increase number of epochs on your GPU enabled machine to get better results.
