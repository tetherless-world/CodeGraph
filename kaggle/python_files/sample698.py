#!/usr/bin/env python
# coding: utf-8

# ## General information
# 
# In this kernel I'll train a simple Pytorch model.

# In[ ]:


import torch
import numpy as np
import pandas as pd
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import os
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
import time

# ## Code for modelling

# Converting to tensor

# In[ ]:


class ToTensor:
    def __init__(self, excluded_keys=("id")):
        if not isinstance(excluded_keys, set):
            excluded_keys = set(excluded_keys)
        self.excluded = excluded_keys

    def __call__(self, x):
        result = {k: torch.from_numpy(v) for k, v in x.items() if k not in self.excluded}
        for k in self.excluded:
            if k in x.keys():
                result[k] = x[k]
        return result

# Custom dataset class

# In[ ]:


class IGDataset(Dataset):
    def __init__(self, df, transform=None, test=False):
        self.transform = transform
        data = df
        self.test = test
        if not test:
            self.y = data["target"].astype(int)
        else:
            self.y = pd.Series(np.array([-1] * len(data)))
            
        data = data.drop(['target'], axis=1)
        
        if 'id' in data.columns:
            data.drop('id', axis=1, inplace=True)
        # moving wheezy-copper-turtle-magic to the last position for convenience
        cols = [col for col in data.columns if col != 'wheezy-copper-turtle-magic']
        data = data[cols + ['wheezy-copper-turtle-magic']]
        self.x = data.values

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        x = self.x[idx]
        item = {'inputs': x}
        y = np.array([self.y.iloc[idx]]) 
        item['targets'] = y

        if self.transform:
            item = self.transform(item)
        
        return x, y        

# Neural net class

# In[ ]:


class IGClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.inference = False
        self.activation = nn.ELU(inplace=True)
        # self.sigmoid = nn.Sigmoid()
        # self.relu = nn.ReLU()
        self.emb = nn.Embedding(513, 512)

        self.fc1 = nn.Linear(767, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 16)
        self.bn2 = nn.BatchNorm1d(16)
        self.fc3 = nn.Linear(16, 1)
        self.drop = nn.Dropout(0.2)

    def forward(self, x):
        # notice cast to long tensor here
        # print(x)
        emb = self.emb(x[:, -1].long())
        num = x[:, :-1].float()
        data = torch.cat([emb, num], 1)
        out = self.drop(self.bn1(self.activation(self.fc1(data))))
        out = self.drop(self.bn2(self.activation(self.fc2(out))))
        out = self.fc3(out)

        return out

# CyclicLR

# In[ ]:


class CyclicLR(object):
    def __init__(self, optimizer, base_lr=1e-3, max_lr=6e-3,
                 step_size=2000, factor=0.6, min_lr=1e-4, mode='triangular', gamma=1.,
                 scale_fn=None, scale_mode='cycle', last_batch_iteration=-1):

        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer

        if isinstance(base_lr, list) or isinstance(base_lr, tuple):
            if len(base_lr) != len(optimizer.param_groups):
                raise ValueError("expected {} base_lr, got {}".format(
                    len(optimizer.param_groups), len(base_lr)))
            self.base_lrs = list(base_lr)
        else:
            self.base_lrs = [base_lr] * len(optimizer.param_groups)

        if isinstance(max_lr, list) or isinstance(max_lr, tuple):
            if len(max_lr) != len(optimizer.param_groups):
                raise ValueError("expected {} max_lr, got {}".format(
                    len(optimizer.param_groups), len(max_lr)))
            self.max_lrs = list(max_lr)
        else:
            self.max_lrs = [max_lr] * len(optimizer.param_groups)

        self.step_size = step_size

        if mode not in ['triangular', 'triangular2', 'exp_range'] \
                and scale_fn is None:
            raise ValueError('mode is invalid and scale_fn is None')

        self.mode = mode
        self.gamma = gamma

        if scale_fn is None:
            if self.mode == 'triangular':
                self.scale_fn = self._triangular_scale_fn
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = self._triangular2_scale_fn
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = self._exp_range_scale_fn
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode

        self.batch_step(last_batch_iteration + 1)
        self.last_batch_iteration = last_batch_iteration

        self.last_loss = np.inf
        self.min_lr = min_lr
        self.factor = factor

    def batch_step(self, batch_iteration=None):
        if batch_iteration is None:
            batch_iteration = self.last_batch_iteration + 1
        self.last_batch_iteration = batch_iteration
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

    def step(self, loss):
        if loss > self.last_loss:
            self.base_lrs = [max(lr * self.factor, self.min_lr) for lr in self.base_lrs]
            self.max_lrs = [max(lr * self.factor, self.min_lr) for lr in self.max_lrs]

    def _triangular_scale_fn(self, x):
        return 1.

    def _triangular2_scale_fn(self, x):
        return 1 / (2. ** (x - 1))

    def _exp_range_scale_fn(self, x):
        return self.gamma ** (x)

    def get_lr(self):
        step_size = float(self.step_size)
        cycle = np.floor(1 + self.last_batch_iteration / (2 * step_size))
        x = np.abs(self.last_batch_iteration / step_size - 2 * cycle + 1)

        lrs = []
        param_lrs = zip(self.optimizer.param_groups, self.base_lrs, self.max_lrs)
        for param_group, base_lr, max_lr in param_lrs:
            base_height = (max_lr - base_lr) * np.maximum(0, (1 - x))
            if self.scale_mode == 'cycle':
                lr = base_lr + base_height * self.scale_fn(cycle)
            else:
                lr = base_lr + base_height * self.scale_fn(self.last_batch_iteration)
            lrs.append(lr)
        return lrs

# Model training

# In[ ]:


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def train_model(model, train_loader,valid_loader, test_loader, loss_fn, lr=0.001,
                batch_size=512, n_epochs=4, validate=False):
    param_lrs = [{'params': param, 'lr': lr} for param in model.parameters()]
    optimizer = torch.optim.Adam(param_lrs, lr=lr)

    scheduler = CyclicLR(optimizer, base_lr=0.001, max_lr=0.003, min_lr=0.000001,
                         step_size=300, mode='exp_range', gamma=0.99994)

    valid_loss_min = np.Inf
    patience = 3
    stop = False
    for epoch in range(n_epochs):
        start_time = time.time()

        model.train()
        avg_loss = 0.

        for step, (seq_batch, y_batch) in enumerate(train_loader):
            y_pred = model(seq_batch.cuda()).float()
            scheduler.batch_step()
            loss = loss_fn(y_pred.cpu(), y_batch.float().cpu())

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            avg_loss += loss.item() / len(train_loader)

        model.eval()
        test_preds = np.zeros((len(test_loader.dataset)))

        val_loss = 0
        if validate:

            valid_preds = np.zeros((len(valid_loader.dataset)))
            y_true = []
            y_prediction = []
            for i, (seq_batch, y_batch) in enumerate(valid_loader):
                y_pred = model(seq_batch.cuda()).float()
                val_loss += loss_fn(y_pred.cpu(), y_batch.float()).item() / len(valid_loader)
                valid_preds[i * batch_size:(i+1) * batch_size] = y_pred.detach().cpu().numpy()[:, 0]
                # print(y_batch.detach().cpu().numpy())
                # print(valid_preds[i * batch_size:(i+1) * batch_size])
                y_true.extend(list(y_batch.detach().cpu().numpy()))
                y_prediction.extend(list(y_pred.detach().cpu().numpy()))
                # print('local score', print(metrics.roc_auc_score(y_batch.detach().cpu().numpy(), valid_preds[i * batch_size:(i+1) * batch_size])))

        # all_test_preds.append(test_preds)
        elapsed_time = time.time() - start_time
        if epoch > 0 and epoch % 10 == 0:
            print(f'Epoch {epoch + 1}/{n_epochs} \t loss={avg_loss:.4f} val_loss={val_loss:.4f} \t val_auc={metrics.roc_auc_score(y_true, y_prediction):.4f} time={elapsed_time:.2f}s')
        
        valid_loss = avg_loss
        # print('epoch', epoch, 'valid_loss_min', np.round(valid_loss_min, 4), 'valid_loss', np.round(valid_loss, 4))
        
        if valid_loss <= valid_loss_min:
#             print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
#             valid_loss_min,
#             valid_loss))
#             torch.save(model_conv.state_dict(), 'model.pt')
            valid_loss_min = valid_loss
            p = 0

        # check if validation loss didn't improve
        if valid_loss > valid_loss_min:
            p += 1
            # print(f'{p} epochs of increasing val loss')
            if p > patience:
                print('Stopping training')
                stop = True
                break        

        if stop:
            break
        

    for i, (seq_batch, _) in enumerate(test_loader):
        y_pred = model(seq_batch.long().cuda()).float().detach()

        test_preds[i * batch_size:(i+1) * batch_size] = y_pred.cpu().numpy()[:, 0]
        
        
    results_dict = {}
    results_dict['test_preds'] = test_preds
    if validate:
        results_dict['oof'] = y_prediction

    return results_dict

# Training on folds

# In[ ]:


def train_on_folds(train, test, splits, n_epochs=50, validate=False):
    if validate:
        scores = []
    
    transf = ToTensor()

    test_preds = np.zeros((len(test), len(splits)))
    train_oof = np.zeros((len(train), 1))
    
    test_dataset = IGDataset(test, transform=transf, test=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batchsize, shuffle=False, num_workers=0, pin_memory=False, drop_last=False)

    for i, (train_idx, valid_idx) in enumerate(splits):

        train_dataset = IGDataset(train.iloc[train_idx, :], transform=transf)
        valid_dataset = IGDataset(train.iloc[valid_idx, :], transform=transf)
        train_dataloader = DataLoader(train_dataset, batch_size=batchsize, shuffle=False, num_workers=0, pin_memory=False, drop_last=False)
        valid_dataloader = DataLoader(valid_dataset, batch_size=batchsize, shuffle=False, num_workers=0, pin_memory=False, drop_last=False)
        print(f'Fold {i + 1}')

        set_seed(42 + i)
        model = IGClassifier()
        loss_fn = nn.BCEWithLogitsLoss()
        model.cuda()

        results_dict = train_model(model, train_dataloader, valid_dataloader, test_dataloader, loss_fn=loss_fn, n_epochs=n_epochs, validate=validate, batch_size=batchsize)

        if validate:
            train_oof[valid_idx] = results_dict['oof']#.reshape(-1, 1)
            score = metrics.roc_auc_score(train['target'].iloc[valid_idx], train_oof[valid_idx])
            print('score', score)
            scores.append(score)
            
        test_preds[:, i] = results_dict['test_preds']
    print(f'CV mean score: {np.mean(scores)}. Std: {np.std(scores)}')
    
    return test_preds

# Fixing random state

# In[ ]:


import random
def set_seed(seed: int = 0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

set_seed(42)

# ## Preparing data

# In[ ]:


train = pd.read_csv("../input/train.csv") 
test = pd.read_csv("../input/test.csv")
sub = pd.read_csv('../input/sample_submission.csv')
len_train = train.shape[0]
all_data = pd.concat([train, test], axis=0, sort=False)

scaler = StandardScaler()
cols = [c for c in train.columns if c not in ['id', 'target', 'wheezy-copper-turtle-magic']]
all_data[cols] = scaler.fit_transform(all_data[cols])
train = all_data[:len_train].reset_index(drop=True)
test = all_data[len_train:].reset_index(drop=True)

# In[ ]:


batchsize = 1024
n_fold = 5
splits = list(StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=42).split(train, train['target']))

# ## Model training

# In[ ]:


test_preds = train_on_folds(train, test, splits, n_epochs=200, validate=True)

# In[ ]:


sub['target'] = sigmoid(test_preds.mean(1))
sub.to_csv("submission.csv", index=False)
sub.head()
