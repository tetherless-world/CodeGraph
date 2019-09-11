#!/usr/bin/env python
# coding: utf-8

# MNIST is a classic toy dataset for image recognition. The dataset consists of handwritten images that have served as the basis for benchmarking classification algorithms
# 
# **Table of Contents**:
# 
# - Imports
# - Preparing Data
# - Explore Data
# - Model
#     - Simple Feed Forward NN (Test Accuracy of 96%)
#     - **CNN based on LeNet5** (Test Accuracy of 98%)
# - Preparing Test Data
# - Looking at Prediction
# - Exporting Data
# - Ensembles (Test Accuracy of 98%)
# - Commit History

# # Imports

# In[ ]:


# Basic Torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import TensorDataset
from torch.optim import Adam, SGD

# Basic Numeric Computation
import numpy as np
import pandas as pd

# Look at data
from matplotlib import pyplot

# Easy way to split train data
from sklearn.model_selection import train_test_split

# Looking at directory
import os
base_dir = "../input"
print(os.listdir(base_dir))

device = torch.device("cpu")# if torch.cuda.is_available() else torch.device("cpu")
device
epochs=10

# # Preparing Data
# ## Extract Transform Load (ETL)

# ### 1. Extract

# In[ ]:


train = pd.read_csv(base_dir + '/train.csv')
test = pd.read_csv(base_dir + '/test.csv')

# In[ ]:


train.head()

# ### 2. Transform

# In[ ]:


# Convert Dataframe into format ready for training
def createImageData(raw: pd.DataFrame):
    y = raw['label'].values
    y.resize(y.shape[0],1)
    x = raw[[i for i in raw.columns if i != 'label']].values
    x = x.reshape([-1,1, 28, 28])
    y = y.astype(int).reshape(-1)
    x = x.astype(float)
    return x, y

## Convert to One Hot Encoding
def one_hot_embedding(labels, num_classes=10):
    y = torch.eye(num_classes) 
    return y[labels] 

# In[ ]:


x_train, y_train = createImageData(train)
#x_train, x_val, y_train, y_val = train_test_split(x,y, test_size=0.02)

#x_train.shape, y_train.shape, x_val.shape, y_val.shape
x_train.shape, y_train.shape

# In[ ]:


# Normalization
mean = x_train.mean()
std = x_train.std()
x_train = (x_train-mean)/std
#x_val = (x_val-mean)/std

# Numpy to Torch Tensor
x_train = torch.from_numpy(np.float32(x_train)).to(device)
y_train = torch.from_numpy(y_train.astype(np.long)).to(device)
y_train = one_hot_embedding(y_train)
#x_val = torch.from_numpy(np.float32(x_val))
#y_val = torch.from_numpy(y_val.astype(np.long))

# ### 3. Load

# In[ ]:


# Convert into Torch Dataset
train_ds = TensorDataset(x_train, y_train)
#val_ds = TensorDataset(x_val,y_val)

# In[ ]:


# Make Data Loader
train_dl = DataLoader(train_ds, batch_size=64)

# > # Explore Data
# 
# It is always benifition to look at your data before building your model. This helps to understand what the model will be dealing with and removes assumptions you might have induce unknowingly into your model

# In[ ]:


index = 1
pyplot.imshow(x_train.cpu()[index].reshape((28, 28)), cmap="gray")
print(y_train[index])

# ## Model
# 
# Below are helper functions. 
# Initially these were written as normal Pytorch functions but latter abstracted to make code clean and easily reuse later.

# In[ ]:


# Helper Functions

## Initialize weight with xavier_uniform
def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)

## Flatten Later
class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

# Train the network and print accuracy and loss overtime
def fit(train_dl, model, loss, optim, epochs=10):
    model = model.to(device)
    print('Epoch\tAccuracy\tLoss')
    accuracy_overtime = []
    loss_overtime = []
    for epoch in range(epochs):
        avg_loss = 0
        correct = 0
        total=0
        for x, y in train_dl: # Iterate over Data Loder
    
            # Forward pass
            yhat = model(x) 
            l = loss(y, yhat)
            
            #Metrics
            avg_loss+=l.item()
            
            # Backward pass
            optim.zero_grad()
            l.backward()
            optim.step()
            
            # Metrics
            _, original =  torch.max(y, 1)
            _, predicted = torch.max(yhat.data, 1)
            total += y.size(0)
            correct = correct + (original == predicted).sum().item()
            
        accuracy_overtime.append(correct/total)
        loss_overtime.append(avg_loss/len(train_dl))
        print(epoch,accuracy_overtime[-1], loss_overtime[-1], sep='\t')
    return accuracy_overtime, loss_overtime

# Plot Accuracy and Loss of Model
def plot_accuracy_loss(accuracy, loss):
    f = pyplot.figure(figsize=(15,5))
    ax1 = f.add_subplot(121)
    ax2 = f.add_subplot(122)
    ax1.title.set_text("Accuracy over epochs")
    ax2.title.set_text("Loss over epochs")
    ax1.plot(accuracy)
    ax2.plot(loss, 'r:')

# Take an array and show what model predicts 
def predict_for_index(array, model, index):
    testing = array[index].view(1,28,28)
    pyplot.imshow(x_train[index].reshape((28, 28)), cmap="gray")
    print(x_train[index].shape)
    a = model(testing.float())
    print('Prediction',torch.argmax(a,1))

# ## Feed Forward Neural Network with 2 hidden layers

# A feed forward neural network is not ideal for Image Classification. The reason is: **Parameters are not shared**.
# 
# Also, if you increase epoch beyond 10, you will find the model quickly overfits.
# 
# Initially, I made a dumb model and focused on just getting through the complete flow. This is because Deep Learning is a iterative process and spending too much time on building the perfect model can lead to a lot of wasted time with no output.
# 
# I will be making Convolutional NN later which will require better hyperparameter setting.
# 
# The input to this feed forward NN is 784 features. The target labels to predict is 10. I choose 100 as number of hidden units in layer intuitively for being a middle ground between 784 and 10.
# 
# (m, 784) -> (m, 100) -> (m,10)
# 
# m: number of items in mini-batch

# In[ ]:


# Define the model

ff_model = nn.Sequential(
    Flatten(),
    nn.Linear(28*28, 100),
    nn.ReLU(),
    nn.Linear(100, 10),
    nn.Softmax(1),
).to(device)

# In[ ]:


# Initialize model with xavier initialization which is recommended for ReLu
ff_model.apply(init_weights)

# Define Hyperparameters and Train

# In[ ]:


optim = Adam(ff_model.parameters())
loss = nn.MSELoss()
output = fit(train_dl, ff_model, loss, optim, epochs)
plot_accuracy_loss(*output)

# # Looking at prediction the model makes

# In[ ]:


index = 4
predict_for_index(x_train, ff_model, index)

# ## Convolutional NN

# A CNN is perfect for images because the parameters used to detect something in one part of image are same as once used in other part of the image.
# 
# Initially, I started with a simple CNN which was not accurate and would often overfit. I had to rewatch videos from Andrew Ng's Convolutional Neural Network Course and re-learned that it is always better to take inspiration from other successful models.
# 
# This model has accuracy of **0.98942**.
# 
# ![image.png](attachment:image.png)
# 
# Sources for creating model:
# - [LeNet-5 – A Classic CNN Architecture](https://engmrk.com/lenet-5-a-classic-cnn-architecture/) (Image Reference)
# - [DeepLearning.ai-Summary / Convolutional Neural Networks](https://github.com/mbadry1/DeepLearning.ai-Summary/tree/master/4-%20Convolutional%20Neural%20Networks#classic-networks)

# In[ ]:


# A too simple NN taken from pytorch.org/tutorials
class Mnist_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(16, 10, kernel_size=3, stride=2, padding=1)

    def forward(self, xb):
        xb = xb.view(-1, 1, 28, 28)
        xb = F.relu(self.conv1(xb))
        xb = F.relu(self.conv2(xb))
        xb = F.relu(self.conv3(xb))
        xb = F.avg_pool2d(xb, 4)
        return xb.view(-1, xb.size(1))

class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1)
        self.average1 = nn.AvgPool2d(2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1)
        self.average2 = nn.AvgPool2d(2, stride=2)
        self.conv3 = nn.Conv2d(16, 120, kernel_size=4, stride=1)
        
        self.flatten = Flatten()
        
        self.fc1 = nn.Linear(120, 82)
        self.fc2 = nn.Linear(82,10)

    def forward(self, xb):
        xb = xb.view(-1, 1, 28, 28)
        xb = F.tanh(self.conv1(xb))
        xb = self.average1(xb)
        xb = F.tanh(self.conv2(xb))
        xb = self.average2(xb)
        xb = F.tanh(self.conv3(xb))
        xb = xb.view(-1, xb.shape[1])
        xb = F.relu(self.fc1(xb))
        xb = F.relu(self.fc2(xb))
        return xb

# In[ ]:


conv_model = LeNet5()
conv_model.apply(init_weights)
loss = nn.MSELoss()
optim = SGD(conv_model.parameters(), lr=0.1, momentum=0.9)
plot_accuracy_loss(*fit(train_dl, conv_model,loss,optim,epochs))

# # Preparing Test Data

# **Normalization**: We need to normalize with the same values used to normalize train data else it will lead to uneven distribution between train and test set.

# In[ ]:


x_test = test.values
x_test = x_test.reshape([-1, 28, 28]).astype(float)
x_test = (x_test-mean)/std
x_test = torch.from_numpy(np.float32(x_test))
x_test.shape

# ## Looking at prediction
# 

# In[ ]:


index = 7
predict_for_index(x_test, ff_model, index)
predict_for_index(x_test, conv_model, index)

# ## Exporting Data for Submission

# In[ ]:


# Export data to CSV in format of submission
def export_csv(model_name, predictions, commit_no):
    df = pd.DataFrame(prediction.tolist(), columns=['Label'])
    df['ImageId'] = df.index + 1
    file_name = f'submission_{model_name}_v{commit_no}.csv'
    print('Saving ',file_name)
    df[['ImageId','Label']].to_csv(file_name, index = False)

# In[ ]:


test.head()

# In[ ]:


# just to make output easier to read
commit_no = 7 

# In[ ]:


ff_test_yhat = ff_model(x_test.float())
prediction = torch.argmax(ff_test_yhat,1)
print('Prediction',prediction)
export_csv('ff_model',prediction, commit_no=commit_no)

# In[ ]:


cn_train_yhat = conv_model(x_test)
prediction = torch.argmax(cn_train_yhat,1)
yo = torch.argmax(y_train,1)
export_csv('lenet_model',prediction, commit_no=commit_no)

# # Ensemble
# 
# Ensemble is a popular technique using usually in competition which allow combining results of different models to product higher accuracy.
# 
# In this particular case it seems like LeNet was much better than feed forward network that is why accuracy of ensemble on test data is same as LeNet5 - **0.98942**

# In[ ]:


ensemble = ff_test_yhat + cn_train_yhat # Add probabilities of individual predictions
ensemble_one_hot = torch.argmax(y_train,1) # Find argmax
export_csv('ensemble',ensemble_one_hot, commit_no=commit_no)

# # Commit History
# 
# **v3:** Corrected Input Shape: There was different shape between y and yhat
# 
# **v4:** Changed output layer to Softmax: It was previously LogSoftmax. I knew it was wrong but for some reason code was not working then with Softmax.
# 
# **v5:** Clean Code using Helper functions and plotting
# 
# **v6:** Bug fixes
# 
# **v7:** Adding Documentation. Verbose saving.
# 
# **v9:** Implementation of LeNet5 architecture.
# 
# **v10:** Updating results from LeNet5. Table of Contents. And LetNet5 architecture and link to sources.[](http://)

# In[ ]:



