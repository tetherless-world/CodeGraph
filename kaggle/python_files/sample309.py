#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import os
print(os.listdir("../input"))

# In[2]:


# load data
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
print(train_df.shape, test_df.shape)

# In[3]:


# convert dataframes to arrays
train = train_df.values
test = test_df.values
print(train.shape, test.shape)

# In[4]:


# extract pixels and labels from train
X_train = train[:, 1:]
print(X_train.shape)
Y_train = train[:,0:1]
print(Y_train.shape)

# In[5]:


# plot the data to check if correct
m = 42000
n_x = 28
train_digit = X_train.reshape((m, n_x, n_x))

import matplotlib.pyplot as plt
plt.imshow(train_digit[678], cmap=plt.cm.binary)
plt.show()
print('The label of this image is ' + str(Y_train[678]))

# In[6]:


# standardize train dataset by dividing by 255
X_train = X_train / 255.
test = test / 255.

# In[7]:


# one-hot encode the labels
from keras.utils.np_utils import to_categorical
onehot_Y_train = to_categorical(Y_train)
print(onehot_Y_train.shape)
print(Y_train[22], onehot_Y_train[22])

# In[8]:


# build the dense neural network model from keras
from keras import models
from keras import layers

model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(784,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.summary()

# In[9]:


# compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy',
             metrics=['accuracy'])

# In[10]:


# set up a dev set (2000 samples) to check the performance of the DNN
X_dev = X_train[:2000]
rem_X_train = X_train[2000:]
print(X_dev.shape, rem_X_train.shape)

Y_dev = onehot_Y_train[:2000]
rem_Y_train = onehot_Y_train[2000:]
print(Y_dev.shape, rem_Y_train.shape)

# In[11]:


# Train and validate the model for 30 epochs
history = model.fit(rem_X_train, rem_Y_train, epochs=30, batch_size=512,
                    validation_data=(X_dev, Y_dev))

# In[12]:


# plot and visualise the training and validation losses
loss = history.history['loss']
dev_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'bo', label='training loss')
plt.plot(epochs, dev_loss, 'b', label='validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# In[13]:


# predict on test set
predictions = model.predict(test)
print(predictions.shape)
print(predictions[7])

# In[14]:


# set the predicted labels to be the one with the highest probability
predicted_labels = []
for i in range(28000):
    predicted_label = np.argmax(predictions[i])
    predicted_labels.append(predicted_label)

# In[15]:


# create submission file
result = pd.DataFrame(predicted_labels, columns=['Label'])
result.insert(0, 'ImageID', value=range(1, len(result)+1))
result.head()

# In[17]:


# generate submission file in csv format
result.to_csv('rhodium_submission_1.csv', index=False)

# In[ ]:



