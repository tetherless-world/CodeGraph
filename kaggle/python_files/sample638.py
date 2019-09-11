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


import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# In[ ]:


# Load data
df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')
print('Train size: ', df_train.shape)
print('Test size: ', df_test.shape)
df_train.head()

# In[ ]:


# Calculate mean and std of training data - used for normalization later
train_data = df_train.drop('label', axis=1).values
# train_data.shape
# print(train_data.max())
train_mean = train_data.mean()/255.
train_std = train_data.std()/255.
# train_std
print('Mean: ', train_mean)
print('Std: ', train_std)

# In[ ]:


# Train-Val split
mask = np.random.rand(len(df_train)) < 0.8
df_val = df_train[~mask]
df_train = df_train[mask]
print('Train size: ', df_train.shape)
print('Val size: ', df_val.shape)
print('Test size: ', df_test.shape)
df_train.head()

# In[ ]:


# Example Visualization
import matplotlib.pyplot as plt
ind = np.random.randint(0, df_train.shape[0]-1)
plt.imshow(df_train.iloc[ind].values[1:].reshape((28,28)), cmap='gray')
plt.title(str(df_train.iloc[ind][0]))

# In[ ]:


# Create dataset class for PyTorch
class MNISTDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, n):
        data = self.df.iloc[n]
        image = data[1:].values.reshape((28,28)).astype(np.uint8)
        label = data[0]
        if self.transform:
            image = self.transform(image)
        return (image, label)

# In[ ]:


# Initialize transformation, datasets, and loaders
batch_size = 16
classes = range(10)
train_transform = transforms.Compose(
                    [
                    transforms.ToPILImage(),
                    transforms.RandomRotation(15),
#                     transforms.Grayscale(),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[train_mean], std=[train_std]),
                    ])
# don't (really) need the data augmentation in validation
val_transform = transforms.Compose(
                    [
                    transforms.ToPILImage(),
#                     transforms.Grayscale(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[train_mean], std=[train_std]),
                    ])
test_transform = val_transform

train_dataset = MNISTDataset(df_train, transform = train_transform)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                batch_size=batch_size,shuffle = True)
val_dataset = MNISTDataset(df_val, transform = val_transform)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                batch_size=batch_size,shuffle = False)



# In[ ]:


# sanity check for training data
imgs, lbls = next(iter(train_loader))
imgs[7].data.shape
print(imgs.data.min())
print(imgs.data.max())
print(imgs.data.mean())
print(imgs.data.std())
print(classes[lbls[0]])
plt.imshow(imgs[0].data.reshape((28,28)), cmap="gray")

# In[ ]:


# sanity check for validation data
imgs, lbls = next(iter(val_loader))
imgs[0].data.shape
print(imgs.data.min())
print(imgs.data.max())
print(imgs.data.mean())
print(imgs.data.std())
print(classes[lbls[0]])
plt.imshow(imgs[0].data.reshape((28,28)), cmap="gray")

# In[ ]:


# CNN model definition
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Sequential(
                        nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, padding=1),
                        nn.ReLU(inplace=True),
                        nn.MaxPool2d(kernel_size=2, stride=2))
        self.conv2 = nn.Sequential(
                        nn.Conv2d(in_channels=4, out_channels=16, kernel_size=3, padding=1),
                        nn.ReLU(inplace=True),
                        nn.MaxPool2d(kernel_size=2, stride=2))
        self.conv3 = nn.Sequential(
                        nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, padding=1),
                        nn.ReLU(inplace=True),
                        nn.MaxPool2d(kernel_size=3, stride=2))
        self.fc = nn.Sequential(
                        nn.Dropout(p=0.5),
                        nn.Linear(in_features=3*3*64, out_features=128),
                        nn.ReLU(inplace=True),
                        nn.Dropout(p=0.5),
                        nn.Linear(in_features=128, out_features=32),
                        nn.ReLU(inplace=True),
                        nn.Linear(in_features=32, out_features=10))
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view((x.shape[0],-1))
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)
        return x

# In[ ]:


# initialize CNN, cost, and optimizer
model = Model()
model.to(device)
criterion = nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# In[ ]:


# Training Time!
import time

# Some initialization work first...
epochs = 25
train_losses, val_losses = [], []
train_accu, val_accu = [], []
start_time = time.time()

for e in range(epochs):
    epoch_start_time = time.time()
    running_loss = 0
    accuracy=0
    # training step
    model.train()
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        log_ps = model(images)
        
        ps = torch.exp(log_ps)                
        top_p, top_class = ps.topk(1, dim=1)
#         print(labels.view(*top_class.shape))
#         print(top_class.to(torch.uint8))
        equals = top_class == labels.view(*top_class.shape)
        accuracy += torch.mean(equals.type(torch.FloatTensor))
        
        loss = criterion(log_ps, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    # record training loss and error, then evaluate using validation data
    train_losses.append(running_loss/len(train_loader))
    train_accu.append(accuracy/len(train_loader))
    val_loss = 0
    accuracy=0
    model.eval()
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            log_ps = model(images)
            val_loss += criterion(log_ps, labels)

            ps = torch.exp(log_ps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor))
    val_losses.append(val_loss/len(val_loader))
    val_accu.append(accuracy/len(val_loader))

    print("Epoch: {}/{}.. ".format(e+1, epochs),
          "Training Loss: {:.3f}.. ".format(train_losses[-1]),
          "Training Accuracy: {:.3f}.. ".format(train_accu[-1]),
          "Val Loss: {:.3f}.. ".format(val_losses[-1]),
          "Val Accuracy: {:.3f}".format(val_accu[-1]))

    print('Epoch %d / %d took %6.2f seconds' % (e+1, epochs, time.time()-epoch_start_time))
    print('Total training time till this epoch was %8.2f seconds' % (time.time()-start_time))
        

# In[ ]:


# plot training history
plt.figure(figsize=(12,12))
plt.subplot(2,1,1)
ax = plt.gca()
ax.set_xlim([0, epochs + 1])
plt.ylabel('Loss')
plt.plot(range(1, e + 2), train_losses[:e+1], 'r', label='Training Loss')
plt.plot(range(1, e + 2), val_losses[:e+1], 'b', label='Validation Loss')
ax.grid(linestyle='-.')
plt.legend()
plt.subplot(2,1,2)
ax = plt.gca()
ax.set_xlim([0, epochs + 1])
plt.ylabel('Accuracy')
plt.plot(range(1, e + 2), train_accu[:e+1], 'r', label='Training Accuracy')
plt.plot(range(1, e + 2), val_accu[:e+1], 'b', label='Validation Accuracy')
ax.grid(linestyle='-.')
plt.legend()
plt.show()

# In[ ]:


# prepare to predict test data - REMEMBER PRE-PROCESSING!
# I originally forgot to scale and normalize, which caused problems....

# some sanity check to make sure
x_test = df_test.values
x_test = x_test.reshape([-1, 28, 28]).astype(np.float)
x_test = x_test/255.
x_test = (x_test-train_mean)/train_std
print(x_test.min())
print(x_test.max())
print(x_test.mean())
print(x_test.std())

# In[ ]:


# x_test = df_test.values
# x_test = x_test.reshape([-1, 28, 28]).astype(np.float)
# x_test = x_test/255.
# x_test = (x_test-train_mean)/train_std
x_test = np.expand_dims(x_test, axis=1)
x_test = torch.from_numpy(x_test).float().to(device)
# x_test.shape
x_test.type()

# In[ ]:


# prediction time!
model.eval()
with torch.no_grad():
    ps = model(x_test)
    prediction = torch.argmax(ps, 1)
    print('Prediction',prediction)

# In[ ]:


# prepare output file
df_export = pd.DataFrame(prediction.cpu().tolist(), columns = ['Label'])
df_export['ImageId'] = df_export.index +1
df_export = df_export[['ImageId', 'Label']]
df_export.head()

# In[ ]:


df_export.to_csv('output.csv', index=False)

# In[ ]:



