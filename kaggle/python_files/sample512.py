#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np, pandas as pd, os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

# In[ ]:


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# In[ ]:



train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

# In[ ]:


cols = [c for c in train.columns if c not in ['id', 'target']]
oof = np.zeros(len(train))
skf = StratifiedKFold(n_splits=5, random_state=42)

# In[ ]:


cols.remove('wheezy-copper-turtle-magic')


# In[ ]:


class IGClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        #self.inference = False
        self.activation = nn.ELU(inplace=True)
        self.fc1 = nn.Linear(255, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, 1)
        self.drop = nn.Dropout(0.2)
    def forward(self, x):
        data = x
        out = self.drop(self.bn1(self.activation(self.fc1(data))))
        out = self.drop(self.bn2(self.activation(self.fc2(out))))
        out = self.fc3(out)
        return out

# In[ ]:


class CyclicLR(object):
    def __init__(self, optimizer, base_lr=1e-3, max_lr=6e-3,
                 step_size=2000, mode='triangular', gamma=1.,
                 scale_fn=None, scale_mode='cycle', last_batch_iteration=-1):

        if not isinstance(optimizer, Optimizer):
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

    def batch_step(self, batch_iteration=None):
        if batch_iteration is None:
            batch_iteration = self.last_batch_iteration + 1
        self.last_batch_iteration = batch_iteration
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

    def _triangular_scale_fn(self, x):
        return 1.

    def _triangular2_scale_fn(self, x):
        return 1 / (2. ** (x - 1))

    def _exp_range_scale_fn(self, x):
        return self.gamma**(x)

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

# In[ ]:


from torch.optim.optimizer import Optimizer


# In[ ]:


from tqdm import tqdm_notebook

# In[ ]:


debug =False


# In[ ]:


oof = np.zeros(len(train))
preds = np.zeros(len(test))
batch_size = 2048
n_epochs = 50
# BUILD 512 SEPARATE MODELS
for i in tqdm_notebook(range(512)):
    train2 = train[train['wheezy-copper-turtle-magic']==i]
    test2 = test[test['wheezy-copper-turtle-magic']==i]
    idx1 = train2.index; idx2 = test2.index
    train2.reset_index(drop=True,inplace=True)
    test2.reset_index(drop=True,inplace=True)
    skf = StratifiedKFold(n_splits=20, random_state=42)
    x_test_cuda = torch.tensor(test2[cols].values, dtype=torch.float).cuda()
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test_cuda), batch_size=batch_size, shuffle=False)
    for train_index, test_index in skf.split(train2.iloc[:,1:-1], train2['target']):
        x_train_fold = torch.tensor(train2.loc[train_index][cols].values, dtype=torch.float).cuda()
        y_train_fold = torch.tensor(train2.loc[train_index]['target'].values[:, np.newaxis], dtype=torch.float32).cuda()
        x_val_fold = torch.tensor(train2.loc[test_index][cols].values, dtype=torch.float).cuda()
        y_val_fold = torch.tensor(train2.loc[test_index]['target'].values[:, np.newaxis], dtype=torch.float32).cuda()
        loss_fn = torch.nn.BCEWithLogitsLoss()
        model = IGClassifier()
        model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr = 0.0005,weight_decay=1e-5) # Using Adam optimizer
        step_size, base_lr, max_lr = 10, 0.0005, 0.0008  
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=max_lr)
        scheduler = CyclicLR(optimizer, base_lr=base_lr, max_lr=max_lr,step_size=step_size, mode='exp_range',gamma=0.99994)
        train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train_fold, y_train_fold), batch_size=batch_size, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_val_fold, y_val_fold), batch_size=batch_size, shuffle=False)
        avg_losses_f = []
        avg_val_losses_f = []
        best_auc = 0
        earlystop = 0
        test_preds_fold = np.zeros((len(test2)))
        for epoch in range(n_epochs):
            model.train()
            avg_loss = 0.
            for i, (x_batch, y_batch) in enumerate(train_loader):
                y_pred = model(x_batch)
                if scheduler:
                    scheduler.batch_step()
                loss = loss_fn(y_pred, y_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                avg_loss += loss.item()/len(train_loader)
            model.eval()
            valid_preds_fold = np.zeros((x_val_fold.size(0)))
            avg_val_loss = 0.
            for i, (x_batch, y_batch) in enumerate(valid_loader):
                y_pred = model(x_batch).detach()
                #avg_val_loss += loss_fn(y_pred, y_batch).item() / len(valid_loader)
                valid_preds_fold[i * batch_size:(i+1) * batch_size] = sigmoid(y_pred.cpu().numpy())[:, 0]
            auc = roc_auc_score(train2.loc[test_index]['target'],valid_preds_fold)
            if auc >= best_auc:
                best_auc = auc
                torch.save(model.state_dict(), 'params.pkl')
            else:
                earlystop += 1
                if debug:
                    print("epoch{}, auc{}".format(epoch,auc))
                if earlystop >=10:
                    break
        avg_losses_f.append(avg_loss)
        avg_val_losses_f.append(avg_val_loss)
        model.load_state_dict(torch.load('params.pkl'))
        model.eval()
        for i, (x_batch,) in enumerate(test_loader):
            y_pred = model(x_batch).detach()
            test_preds_fold[i * batch_size:(i+1) * batch_size] = sigmoid(y_pred.cpu().numpy())[:, 0]
        valid_preds_fold = np.zeros((x_val_fold.size(0)))
        for i, (x_batch, y_batch) in enumerate(valid_loader):
            y_pred = model(x_batch).detach()
            valid_preds_fold[i * batch_size:(i+1) * batch_size] = sigmoid(y_pred.cpu().numpy())[:, 0]
        #print(roc_auc_score(train2.loc[test_index]['target'],valid_preds_fold))
        #print(test_preds_fold)
        oof[idx1[test_index]] += valid_preds_fold#clf.predict_proba(train2.loc[test_index][cols])[:,1]
        preds[idx2] += test_preds_fold / 20 #clf.predict_proba(test2[cols])[:,1] / 25.0        
# PRINT CV AUC
auc = roc_auc_score(train['target'],oof)
print('NN with interactions scores CV =',round(auc,5))

# In[ ]:


auc = roc_auc_score(train['target'],oof)
print('NN with interactions scores CV =',round(auc,5))

# # Submit Predictions

# In[ ]:


sub = pd.read_csv('../input/sample_submission.csv')
sub['target'] = preds
sub.to_csv('submission.csv',index=False)
