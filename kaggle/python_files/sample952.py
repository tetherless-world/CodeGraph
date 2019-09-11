#!/usr/bin/env python
# coding: utf-8

# # Indoor Position by RSRP

# In[38]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
import tensorflow as tf


# In[39]:


import os
print(os.listdir("../input"))


# In[40]:


traindf2_1 = pd.read_csv('../input/combined 2.1G.csv').drop(' Max', axis=1)
traindf3_5 = pd.read_csv('../input/combined3.5G.csv').drop(' Max', axis=1)
traindf = pd.merge(traindf2_1, traindf3_5, on=['row', ' col'])
#traindf = traindf2_1
#test = pd.read_csv('../input/test.csv')
print(traindf.describe())
print(traindf.info())
print(traindf.shape)
# check training data
totalRows = traindf['row'].max()
totalCols = traindf[' col'].max()
print(totalRows, ' ',totalCols)
trainData = traindf.iloc[:, 2:]
trainlabels = traindf.iloc[:, 0:2]
print(trainData.info())
print(trainlabels.info())
dimOfdata = trainData.shape[1]
print(dimOfdata)
numberOfSamples = len(trainlabels)

# In[41]:


imageSize = (totalRows+1,totalCols+1)
def getImage():
    image = np.zeros(imageSize)
    for index, lables in trainlabels.iterrows():
        image[lables['row']][lables[' col']] = 255
    return image

image = getImage()
plt.title('images')
plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')

# In[42]:


trainData[trainData.isnull().values==True]

# In[43]:


Y_train = trainlabels.values
#min_max_scaler = preprocessing.MinMaxScaler()
#X_train_minmax = min_max_scaler.fit_transform(trainData)
X_train_minmax = preprocessing.StandardScaler().fit_transform(trainData)
Y_min_max_scaler = preprocessing.MinMaxScaler()
Y_train_minmax = Y_min_max_scaler.fit_transform(Y_train)

print(Y_train.shape) #totalRows+1,totalCols+1
Y_row_train = trainlabels['row'].values
Y_col_train = trainlabels[' col'].values

print(X_train_minmax.shape)
print(Y_row_train.shape)
print(Y_col_train.shape)

# Training:
# 

# In[44]:


dense = 256
batch_size = 128
epochs = 50

inputs = tf.keras.Input(shape=(dimOfdata, 1))

#l1 = tf.keras.layers.GRU(dense, return_sequences=True, input_shape=(None, dimOfdata, 1))(inputs)
#l2 = tf.keras.layers.GRU(dense, activation='relu')(l1)
#l3 = tf.keras.layers.Dense(dense, activation='relu')(l2)

c1 = tf.keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(None, dimOfdata, 1))(inputs)
c2 = tf.keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu')(c1)

c5 = tf.keras.layers.Flatten()(c2)

l4 = tf.keras.layers.Dense(dense, activation='relu')(c5)
l5 = tf.keras.layers.Dense(dense, activation='relu')(l4)
#l6 = tf.keras.layers.Dense(dense, activation='relu')(l5)
#linear
row_outputs = tf.keras.layers.Dense(1, name='row_coordinate', activation="linear")(l5)
col_outputs = tf.keras.layers.Dense(1, name='col_coordinate', activation="linear")(l5)

model = tf.keras.Model(inputs=inputs, outputs=[row_outputs, col_outputs])
model.compile(optimizer='adam',
              loss='mse',
              metrics=['mae'])
 
x_train = X_train_minmax.reshape((-1, dimOfdata, 1)) 

#Y_train_minmax
history = model.fit(x_train, {'row_coordinate' : Y_train_minmax[:, 0], 'col_coordinate' : Y_train_minmax[: , 1]}, epochs=epochs, batch_size=batch_size)
#history = model.fit(x_train, {'row_coordinate' : Y_row_train, 'col_coordinate' : Y_col_train}, epochs=epochs, batch_size=batch_size)


# In[47]:


plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
plt.xlabel('Epoch')
plt.ylabel('Mean Absolute Error Loss')
plt.title('Loss Over Time')
plt.legend(['Train','Valid'])

# Prediction:

# In[49]:


model.summary()
n = 100
randomSamples = np.random.choice(len(X_train_minmax)-1, n)

for sample in randomSamples:
    pred = model.predict(X_train_minmax[sample].reshape(1,dimOfdata,1))
    row_pred, col_pred = pred[0][0], pred[1][0]
    pred_ = np.asarray([row_pred, col_pred]).reshape(1,2)
    predict_ = Y_min_max_scaler.inverse_transform(pred_)
    predict_x, predict_y = predict_[0][0], predict_[0][1]
    print("({}:{}) vs ({}:{})".format(Y_row_train[sample], Y_col_train[sample], predict_x, predict_y))


# In[ ]:



