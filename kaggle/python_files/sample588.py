#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
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


# reshape flattened data into 3D tensor
n_x = 28
X_train_digit = X_train.reshape((-1, n_x, n_x, 1))
print(X_train_digit.shape)

# In[6]:


# similarly for test set
test_digit = test.reshape((-1, n_x, n_x, 1))
print(test_digit.shape)

# In[7]:


# standardize train dataset by dividing by 255
X_train_digit = X_train_digit / 255.
test_digit = test_digit / 255.

# In[8]:


# one-hot encode the labels
from keras.utils.np_utils import to_categorical
onehot_Y_train = to_categorical(Y_train)
print(onehot_Y_train.shape)
print(Y_train[222], onehot_Y_train[222])

# In[14]:


# build the convolutional neural network model from keras
from keras import models
from keras import layers

model = models.Sequential()
model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.summary()

# In[15]:


# compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy',
             metrics=['accuracy'])

# In[16]:


# set up a dev set (2000 samples) to check the performance of the DNN
X_dev = X_train_digit[:2000]
rem_X_train = X_train_digit[2000:]
print(X_dev.shape, rem_X_train.shape)

Y_dev = onehot_Y_train[:2000]
rem_Y_train = onehot_Y_train[2000:]
print(Y_dev.shape, rem_Y_train.shape)

# In[17]:


# Train and validate the model for 10 epochs
history = model.fit(rem_X_train, rem_Y_train, epochs=10, batch_size=256,
                    validation_data=(X_dev, Y_dev))

# In[20]:


# plot and visualise the training and validation losses
loss = history.history['loss']
dev_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)

from matplotlib import pyplot as plt
plt.plot(epochs, loss, 'bo', label='training loss')
plt.plot(epochs, dev_loss, 'b', label='validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# In[21]:


# predict on test set
predictions = model.predict(test_digit)
print(predictions.shape)
print(predictions[7])

# In[22]:


# set the predicted labels to be the one with the highest probability
predicted_labels = []
for i in range(28000):
    predicted_label = np.argmax(predictions[i])
    predicted_labels.append(predicted_label)

# In[23]:


# create submission file
result = pd.DataFrame(predicted_labels, columns=['Label'])
result.insert(0, 'ImageID', value=range(1, len(result)+1))
result.head()

# In[24]:


# generate submission file in csv format
result.to_csv('rhodium_submission_2.csv', index=False)

# In[ ]:



