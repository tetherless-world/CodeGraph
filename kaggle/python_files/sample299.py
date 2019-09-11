#!/usr/bin/env python
# coding: utf-8

# # Indoor Position by RSRP

# In[10]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
import tensorflow as tf
import seaborn as sns


# In[11]:


import os
print(os.listdir("../input"))


# In[12]:


traindf2_1 = pd.read_csv('../input/combined 2.1G.csv').drop(' Max', axis=1)
traindf3_5 = pd.read_csv('../input/combined3.5G.csv').drop(' Max', axis=1)
traindf = pd.merge(traindf2_1, traindf3_5, on=['row', ' col'])
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

# In[13]:


imageSize = (totalRows+1,totalCols+1)

def getImage():
    image = np.zeros(imageSize)
    for index, label in trainlabels.iterrows():
        image[label['row']][label[' col']] = index+1
    return image

image = getImage()
plt.title('images')
plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')

plt.figure(figsize=(7,5)) 
sns.heatmap(traindf.corr(),annot=True,cmap='RdYlGn_r') 
plt.show()

# In[14]:


trainData[trainData.isnull().values==True]

# In[15]:



#min_max_scaler = preprocessing.MinMaxScaler()
#X_train_minmax = min_max_scaler.fit_transform(trainData)
#X_test_minmax = min_max_scaler.transform(X_test)
X_train_minmax = preprocessing.StandardScaler().fit_transform(trainData)

Y_train = trainlabels.values
print(Y_train.shape) #totalRows+1,totalCols+1
Y_row_train = trainlabels['row'].values
Y_col_train = trainlabels[' col'].values

print(X_train_minmax.shape)
print(Y_row_train.shape)
print(Y_col_train.shape)

Y_train_index = trainlabels.index.values
print(Y_train_index.shape)
def fromXY2Index(x, y):
    return image[x][y]-1

def fromIndex2XY(index):
    return (trainlabels[index, 'row'], trainlabels[index, ' col'])

ndim = len(Y_train_index)
print(ndim)

# Training:
# 

# In[16]:


#totalRows+1,totalCols+1

dense = 256
batch_size = 128
epochs = 500
dimOfdata = 12
inputs = tf.keras.Input(shape=(dimOfdata, 1))

#l1 = tf.keras.layers.GRU(dense, return_sequences=True, input_shape=(None, dimOfdata, 1))(inputs)
#l2 = tf.keras.layers.GRU(dense, activation='relu')(l1)
#l3 = tf.keras.layers.Dense(dense, activation='relu')(l2)

c1 = tf.keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(None, dimOfdata, 1))(inputs)
c2 = tf.keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu')(c1)
l3 = tf.keras.layers.Flatten()(c2)
l4 = tf.keras.layers.Dense(dense, activation='relu')(l3)
l5 = tf.keras.layers.Dense(dense, activation='relu')(l4)
l6 = tf.keras.layers.Dense(dense, activation='relu')(l5)

row_output = tf.keras.layers.Dense(totalRows+1, name='row_coordinate', activation="softmax")(l6)
col_output = tf.keras.layers.Dense(totalCols+1, name='col_coordinate', activation="softmax")(l6)

model = tf.keras.Model(inputs=inputs, outputs=[row_output, col_output])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#history = model.fit(X_train_minmax, Y_train, epochs=epochs, batch_size=batch_size)
x_train = X_train_minmax.reshape((-1, 12, 1))
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=20)
history = model.fit(x_train, {'row_coordinate' : Y_row_train, 'col_coordinate' : Y_col_train}, epochs=epochs, batch_size=batch_size, 
                    callbacks=[early_stopping])

# In[17]:


model.summary()
plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
plt.xlabel('Epoch')
plt.ylabel('Mean Absolute Error Loss')
plt.title('Loss Over Time')
plt.legend(['Train','Valid'])

# Prediction:

# In[18]:


def choose(preds):
    # helper function to sample an index from a probability array
    top_k = 3
    top_k_idx = preds.argsort()[::-1][0:top_k]
    return top_k_idx

n = 100
#randomSamples = range(100)
randomSamples = np.random.choice(len(X_train_minmax)-1, n)
for sample in randomSamples:
    pred = model.predict(X_train_minmax[sample].reshape(1,12,1))
    #print(X_train_minmax[sample].reshape(1,shape))
    row_pred, col_pred = pred[0][0], pred[1][0]
    print("({}:{}) vs ({}:{})".format(Y_row_train[sample], Y_col_train[sample], choose(row_pred), choose(col_pred)))


# In[20]:


numberOfSample = len(X_train_minmax)
count = 0
for index in range(numberOfSample):
    pred = model.predict(X_train_minmax[index].reshape(1,12,1))
    row_pred, col_pred = pred[0][0], pred[1][0]
    if Y_row_train[index] in choose(row_pred) and Y_col_train[index] in choose(col_pred):
        count += 1

print("total: {}, accuracy: {}".format(numberOfSample, count))
print("rate: ", count/numberOfSample)

# In[ ]:



