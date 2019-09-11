#!/usr/bin/env python
# coding: utf-8

# # 25/04/2019 Update
# Please check out my  [new embedding model](https://www.kaggle.com/benjibb/entity-embedding-neural-network), which I think is more promising than the existing method.
# 
# # 07/05/2017 Update
# 
# This project is based on my [GitHub link][1] and my research is based on  [this paper][2]. 
# 
# Instead of using Echo state network which was used in the Stanford research paper, we are going to use LSTM which is more advanced in training the neural network.
# 
# More updates will be provided to accommodate the dataset in this Kaggle challenge.  You can simply adjust it to choose your features and window for data.
# 
# Thank you all!
# 
# # Import module first
# 
# 
#   [1]: https://github.com/BenjiKCF/Neural-Network-with-Financial-Time-Series-Data
#   [2]: http://cs229.stanford.edu/proj2012/BernalFokPidaparthi-FinancialMarketTimeSeriesPredictionwithRecurrentNeural.pdf

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import datetime
import math, time
import itertools
from sklearn import preprocessing
import datetime
from operator import itemgetter
from sklearn.metrics import mean_squared_error
from math import sqrt
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM
from keras.models import load_model
import keras
import h5py
import requests
import os

# # Read data and transform them to pandas dataframe

# In[ ]:


df = pd.read_csv("../input/prices-split-adjusted.csv", index_col = 0)
df["adj close"] = df.close # Moving close to the last column
df.drop(['close'], 1, inplace=True) # Moving close to the last column
df.head()

# In[ ]:


df2 = pd.read_csv("../input/fundamentals.csv")
df2.head()

# # Extract all symbols from the list

# In[ ]:


symbols = list(set(df.symbol))
len(symbols)

# In[ ]:


symbols[:11] # Example of what is in symbols

# # Extract a particular price for stock in symbols
# Use GOOG as an example

# In[ ]:


df = df[df.symbol == 'GOOG']
df.drop(['symbol'],1,inplace=True)
df.head()

# # Normalize the data

# In[ ]:


def normalize_data(df):
    min_max_scaler = preprocessing.MinMaxScaler()
    df['open'] = min_max_scaler.fit_transform(df.open.values.reshape(-1,1))
    df['high'] = min_max_scaler.fit_transform(df.high.values.reshape(-1,1))
    df['low'] = min_max_scaler.fit_transform(df.low.values.reshape(-1,1))
    df['volume'] = min_max_scaler.fit_transform(df.volume.values.reshape(-1,1))
    df['adj close'] = min_max_scaler.fit_transform(df['adj close'].values.reshape(-1,1))
    return df
df = normalize_data(df)
df.head()

# # Create training set and testing set

# In[ ]:


def load_data(stock, seq_len):
    amount_of_features = len(stock.columns) # 5
    data = stock.as_matrix() 
    sequence_length = seq_len + 1 # index starting from 0
    result = []
    
    for index in range(len(data) - sequence_length): # maxmimum date = lastest date - sequence length
        result.append(data[index: index + sequence_length]) # index : index + 22days
    
    result = np.array(result)
    row = round(0.9 * result.shape[0]) # 90% split
    train = result[:int(row), :] # 90% date, all features 
    
    x_train = train[:, :-1] 
    y_train = train[:, -1][:,-1]
    
    x_test = result[int(row):, :-1] 
    y_test = result[int(row):, -1][:,-1]

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], amount_of_features))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], amount_of_features))  

    return [x_train, y_train, x_test, y_test]

# # Build the structure of model
# 
# Based on my hyperparameter testing on [here][1]. I found that these parameters are the most suitable for this task.
# 
# ![dropout = 0.3][2]
# ![epochs = 90][3]
# ![LSTM 256 > LSTM 256 > Relu 32 > Linear 1][4]
# 
# 
# 
#   [1]: https://github.com/BenjiKCF/Neural-Network-with-Financial-Time-Series-Data
#   [2]: https://github.com/BenjiKCF/Neural-Network-with-Financial-Time-Series-Data/blob/master/dropout.png?raw=true
#   [3]: https://github.com/BenjiKCF/Neural-Network-with-Financial-Time-Series-Data/blob/master/epochs2.png?raw=true
#   [4]: https://github.com/BenjiKCF/Neural-Network-with-Financial-Time-Series-Data/blob/master/neurons.png?raw=true

# In[ ]:


def build_model(layers):
    d = 0.3
    model = Sequential()
    
    model.add(LSTM(256, input_shape=(layers[1], layers[0]), return_sequences=True))
    model.add(Dropout(d))
        
    model.add(LSTM(256, input_shape=(layers[1], layers[0]), return_sequences=False))
    model.add(Dropout(d))
        
    model.add(Dense(32,kernel_initializer="uniform",activation='relu'))        
    model.add(Dense(1,kernel_initializer="uniform",activation='linear'))
    
    # adam = keras.optimizers.Adam(decay=0.2)
        
    start = time.time()
    model.compile(loss='mse',optimizer='adam', metrics=['accuracy'])
    print("Compilation Time : ", time.time() - start)
    return model

# # Train the model

# In[ ]:


window = 22
X_train, y_train, X_test, y_test = load_data(df, window)
print (X_train[0], y_train[0])

# In[ ]:


model = build_model([5,window,1])

# In[ ]:


model.fit(X_train,y_train,batch_size=512,epochs=90,validation_split=0.1,verbose=1)

# In[ ]:


# print(X_test[-1])
diff=[]
ratio=[]
p = model.predict(X_test)
print (p.shape)
# for each data index in test data
for u in range(len(y_test)):
    # pr = prediction day u
    pr = p[u][0]
    # (y_test day u / pr) - 1
    ratio.append((y_test[u]/pr)-1)
    diff.append(abs(y_test[u]- pr))
    # print(u, y_test[u], pr, (y_test[u]/pr)-1, abs(y_test[u]- pr))
    # Last day prediction
    # print(p[-1]) 

# # Denormalize the data

# In[ ]:


df = pd.read_csv("../input/prices-split-adjusted.csv", index_col = 0)
df["adj close"] = df.close # Moving close to the last column
df.drop(['close'], 1, inplace=True) # Moving close to the last column
df = df[df.symbol == 'GOOG']
df.drop(['symbol'],1,inplace=True)

# Bug fixed at here, please update the denormalize function to this one
def denormalize(df, normalized_value): 
    df = df['adj close'].values.reshape(-1,1)
    normalized_value = normalized_value.reshape(-1,1)
    
    #return df.shape, p.shape
    min_max_scaler = preprocessing.MinMaxScaler()
    a = min_max_scaler.fit_transform(df)
    new = min_max_scaler.inverse_transform(normalized_value)
    return new

newp = denormalize(df, p)
newy_test = denormalize(df, y_test)

# In[ ]:


def model_score(model, X_train, y_train, X_test, y_test):
    trainScore = model.evaluate(X_train, y_train, verbose=0)
    print('Train Score: %.5f MSE (%.2f RMSE)' % (trainScore[0], math.sqrt(trainScore[0])))

    testScore = model.evaluate(X_test, y_test, verbose=0)
    print('Test Score: %.5f MSE (%.2f RMSE)' % (testScore[0], math.sqrt(testScore[0])))
    return trainScore[0], testScore[0]


model_score(model, X_train, y_train, X_test, y_test)

# # Since the Kaggle dataset only contains a few years, the mean square error is not as small as my original model on GitHub.
# 
# With more than 40 years of data, we will get:
# 
# Train Score: 0.00006 MSE (0.01 RMSE)
# 
# Test Score: 0.00029 MSE (0.02 RMSE)

# In[ ]:


import matplotlib.pyplot as plt2

plt2.plot(newp,color='red', label='Prediction')
plt2.plot(newy_test,color='blue', label='Actual')
plt2.legend(loc='best')
plt2.show()

# The result on my original model with more than 40 years of data.
# 
# ![Result][1]
# 
#  Train Score: 0.00006 MSE (0.01 RMSE)
# 
# Test Score: 0.00029 MSE (0.02 RMSE)
# 
#   [1]: https://github.com/BenjiKCF/Neural-Network-with-Financial-Time-Series-Data/raw/master/result2.png

# # Thank you all for reading
#  If you have any question or concern, please leave a comment. Otherwise, see you next time!

# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# 

# In[ ]:




# In[ ]:




# 

# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



