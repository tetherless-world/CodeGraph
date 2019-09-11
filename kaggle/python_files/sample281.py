#!/usr/bin/env python
# coding: utf-8

# # This kernel is U-Net Baseline written by PyTorch
# In this kernel, there are many places that are simplified now.  
# So, you should fix these bad points.  
# 
# [U-Net web site](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/)  
# [U-Net paper](https://arxiv.org/abs/1505.04597)  
# 
# I reference [this blog post](https://lp-tech.net/articles/hzfn7?page=2  ) in U-Net installation.  
# Thank you awesome this blog post.  
# 
# This is [my EDA](https://www.kaggle.com/go1dfish/fgvc6-simple-eda).  
# If you don't know this competition rule and data, this EDA might help you.  

# # Import modules

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook as tqdm

import torch
import torch.nn as nn
from torch import optim
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.autograd import Function, Variable
from pathlib import Path
from itertools import groupby

# In[2]:


input_dir = "../input/"
train_img_dir = "../input/train/"
test_img_dir = "../input/test/"

WIDTH = 512
HEIGHT = 512
category_num = 46 + 1

ratio = 8

epoch_num = 8
batch_size = 4

device = "cuda:0"

# In[3]:


len(os.listdir("../input/train/"))

# In[4]:


len(os.listdir("../input/test/"))

# In[ ]:


train_df = pd.read_csv(input_dir + "train.csv")
train_df.head()

# In[ ]:


train_df.shape

# # Define utils
# For simplicity, It focus only category

# In[ ]:


def make_onehot_vec(x):
    vec = np.zeros(category_num)
    vec[x] = 1
    return vec

# In[ ]:


def make_mask_img(segment_df):
    seg_width = segment_df.at[0, "Width"]
    seg_height = segment_df.at[0, "Height"]
    seg_img = np.full(seg_width*seg_height, category_num-1, dtype=np.int32)
    for encoded_pixels, class_id in zip(segment_df["EncodedPixels"].values, segment_df["ClassId"].values):
        pixel_list = list(map(int, encoded_pixels.split(" ")))
        for i in range(0, len(pixel_list), 2):
            start_index = pixel_list[i] - 1
            index_len = pixel_list[i+1] - 1
            seg_img[start_index:start_index+index_len] = int(class_id.split("_")[0])
    seg_img = seg_img.reshape((seg_height, seg_width), order='F')
    seg_img = cv2.resize(seg_img, (WIDTH, HEIGHT), interpolation=cv2.INTER_NEAREST)
    """
    seg_img_onehot = np.zeros((HEIGHT, WIDTH, category_num), dtype=np.int32)
    #seg_img_onehot = np.zeros((seg_height//ratio, seg_width//ratio, category_num), dtype=np.int32)
    # OPTIMIZE: slow
    for ind in range(HEIGHT):
        for col in range(WIDTH):
            seg_img_onehot[ind, col] = make_onehot_vec(seg_img[ind, col])
    """
    return seg_img

# In[ ]:


def train_generator(df, batch_size):
    img_ind_num = df.groupby("ImageId")["ClassId"].count()
    index = df.index.values[0]
    trn_images = []
    seg_images = []
    for i, (img_name, ind_num) in enumerate(img_ind_num.items()):
        img = cv2.imread(train_img_dir + img_name)
        img = cv2.resize(img, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
        segment_df = (df.loc[index:index+ind_num-1, :]).reset_index(drop=True)
        index += ind_num
        if segment_df["ImageId"].nunique() != 1:
            raise Exception("Index Range Error")
        seg_img = make_mask_img(segment_df)
        
        # HWC -> CHW
        img = img.transpose((2, 0, 1))
        #seg_img = seg_img.transpose((2, 0, 1))
        
        trn_images.append(img)
        seg_images.append(seg_img)
        if((i+1) % batch_size == 0):
            yield np.array(trn_images, dtype=np.float32) / 255, np.array(seg_images, dtype=np.int32)
            trn_images = []
            seg_images = []
    if(len(trn_images) != 0):
        yield np.array(trn_images, dtype=np.float32) / 255, np.array(seg_images, dtype=np.int32)

# In[ ]:


def test_generator(df):
    img_names = df["ImageId"].values
    for img_name in img_names:
        img = cv2.imread(test_img_dir + img_name)
        img = cv2.resize(img, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
        # HWC -> CHW
        img = img.transpose((2, 0, 1))
        yield img_name, np.asarray([img], dtype=np.float32) / 255

# In[ ]:


def encode(input_string):
    return [(len(list(g)), k) for k,g in groupby(input_string)]

def run_length(label_vec):
    encode_list = encode(label_vec)
    index = 1
    class_dict = {}
    for i in encode_list:
        if i[1] != category_num-1:
            if i[1] not in class_dict.keys():
                class_dict[i[1]] = []
            class_dict[i[1]] = class_dict[i[1]] + [index, i[0]]
        index += i[0]
    return class_dict

# # Define Network

# In[ ]:


class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffX = x1.size()[2] - x2.size()[2]
        diffY = x1.size()[3] - x2.size()[3]
        x2 = F.pad(x2, (diffX // 2, int(diffX / 2),
                        diffY // 2, int(diffY / 2)))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x

    
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc = outconv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x

# # Training

# In[ ]:


train_df.shape

# In[ ]:


333415 // 4

# In[ ]:


train_df.iloc[83348:83354, :]

# In[ ]:


train_df.iloc[73350:73354, :]

# For simplicity, use about 25% data.  

# In[ ]:


net = UNet(n_channels=3, n_classes=category_num).to(device)

optimizer = optim.SGD(
    net.parameters(),
    lr=0.1,
    momentum=0.9,
    weight_decay=0.0005
)

criterion = nn.CrossEntropyLoss()

# In[ ]:


val_sta = 73352
val_end = 83351
train_loss = []
valid_loss = []
for epoch in range(epoch_num):
    epoch_trn_loss = 0
    train_len = 0
    net.train()
    for iteration, (X_trn, Y_trn) in enumerate(tqdm(train_generator(train_df.iloc[:val_sta, :], batch_size))):
        X = torch.tensor(X_trn, dtype=torch.float32).to(device)
        Y = torch.tensor(Y_trn, dtype=torch.long).to(device)
        train_len += len(X)
        
        #Y_flat = Y.view(-1)
        mask_pred = net(X)
        #mask_prob = torch.softmax(mask_pred, dim=1)
        #mask_prob_flat = mask_prob.view(-1)
        loss = criterion(mask_pred, Y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_trn_loss += loss.item()
        
        if iteration % 100 == 0:
            print("train loss in {:0>2}epoch  /{:>5}iter:    {:<10.8}".format(epoch+1, iteration, epoch_trn_loss/(iteration+1)))
        
    train_loss.append(epoch_trn_loss/(iteration+1))
    print("train {}epoch loss({}iteration):    {:10.8}".format(epoch+1, iteration, train_loss[-1]))
    
    epoch_val_loss = 0
    val_len = 0
    net.eval()
    for iteration, (X_val, Y_val) in enumerate(tqdm(train_generator(train_df.iloc[val_sta:val_end, :], batch_size))):
        X = torch.tensor(X_val, dtype=torch.float32).to(device)
        Y = torch.tensor(Y_val, dtype=torch.long).to(device)
        val_len += len(X)
        
        #Y_flat = Y.view(-1)
        
        mask_pred = net(X)
        #mask_prob = torch.softmax(mask_pred, dim=1)
        #mask_prob_flat = mask_prob.view(-1)
        loss = criterion(mask_pred, Y)
        epoch_val_loss += loss.item()
        
        if iteration % 100 == 0:
            print("valid loss in {:0>2}epoch  /{:>5}iter:    {:<10.8}".format(epoch+1, iteration, epoch_val_loss/(iteration+1)))
        
    valid_loss.append(epoch_val_loss/(iteration+1))
    print("valid {}epoch loss({}iteration):    {:10.8}".format(epoch+1, iteration, valid_loss[-1]))

# In[ ]:


#plt.plot(list(range(epoch_num)), train_loss, color='green')
#plt.plot(list(range(epoch_num)), valid_loss, color='blue')

# # Test

# In[ ]:


sample_df = pd.read_csv(input_dir + "sample_submission.csv")

# In[ ]:


import torch
import gc
for obj in gc.get_objects():
    try:
        if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
            print(type(obj), obj.size())
    except:
        pass

# In[ ]:


sub_list = []
net.eval()
for img_name, img in test_generator(sample_df):
    X = torch.tensor(img, dtype=torch.float32).to(device)
    mask_pred = net(X)
    mask_pred = mask_pred.cpu().detach().numpy()
    mask_prob = np.argmax(mask_pred, axis=1)
    mask_prob = mask_prob.ravel(order='F')
    class_dict = run_length(mask_prob)
    if len(class_dict) == 0:
        sub_list.append([img_name, "1 1", 1])
    else:
        for key, val in class_dict.items():
            sub_list.append([img_name, " ".join(map(str, val)), key])

# # Make Submission File

# In[ ]:


submission_df = pd.DataFrame(sub_list, columns=sample_df.columns.values)

# In[ ]:


submission_df

# In[ ]:


submission_df.to_csv("submission.csv", index=False)

# # Thank you for watching!
# Please tell me when I make mistakes in program and English.  
# I hope this kernel will help.  
# If you think this kernel is useful, please upvote. If you do, I feel happy and get enough sleep.  

# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



