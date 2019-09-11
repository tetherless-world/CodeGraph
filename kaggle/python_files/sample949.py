#!/usr/bin/env python
# coding: utf-8

# ### Imports

# In[26]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from keras import layers, Input, Model
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# ### Loading data

# In[27]:


data_dir = '../input/'

# In[28]:


train_raw = pd.read_csv(f'{data_dir}train.csv')
train_raw.head()

# In[29]:


test_raw = pd.read_csv(f'{data_dir}test.csv')
test_raw.head()

# In[30]:


train_raw.shape, test_raw.shape

# In[31]:


train_raw.isnull().sum().sum(), test_raw.isnull().sum().sum()

# So there are no missing values in either training or test set.

# ### Target distribution

# In[32]:


sns.countplot(train_raw.target)
plt.show()

# In[33]:


train_raw.target.value_counts()

# Looks like class labels are uniformly distributed in training data.

# ### Train-validation split
# 
# Let's split our training set such that 15% of data are used for validation -

# In[34]:


trn_x, valid_x, trn_y, valid_y = train_test_split(train_raw.drop(['id', 'target'], axis=1), train_raw.target, random_state=33, test_size=0.15)
trn_x.shape, valid_x.shape, trn_y.shape, valid_y.shape

# ### Categorical Feature

# In[35]:


trn_wheezy = pd.get_dummies(trn_x['wheezy-copper-turtle-magic'])
valid_wheezy = pd.get_dummies(valid_x['wheezy-copper-turtle-magic'])
test_wheezy = pd.get_dummies(test_raw['wheezy-copper-turtle-magic'])

trn_wheezy.shape, valid_wheezy.shape, test_wheezy.shape

# In[36]:


trn_x.drop('wheezy-copper-turtle-magic', axis=1, inplace=True)
valid_x.drop('wheezy-copper-turtle-magic', axis=1, inplace=True)
test_raw.drop('wheezy-copper-turtle-magic', axis=1, inplace=True)

# ### Normalize features

# In[37]:


sc = StandardScaler()
trn_x = sc.fit_transform(trn_x)

valid_x = sc.transform(valid_x)
test_x = sc.transform(test_raw.drop('id', axis=1))

# In[38]:


trn_x = np.concatenate([trn_x, trn_wheezy.values], axis=1)
valid_x = np.concatenate([valid_x, valid_wheezy.values], axis=1)
test_x = np.concatenate([test_x, test_wheezy.values], axis=1)

# ### Model

# In[39]:


def build_model():
    inp = Input(shape=(trn_x.shape[1],), name='input')
    x = layers.Dense(1000, activation='relu')(inp)
    x = layers.Dropout(0.65)(x)
    x = layers.Dense(750, activation='relu')(x)
    x = layers.Dropout(0.65)(x)
    x = layers.Dense(500, activation='relu')(x)
    x = layers.Dropout(0.6)(x)
    x = layers.Dense(1, activation='sigmoid')(x)
    
    model = Model(inp, x)
    model.compile(optimizer='adam',
                 loss='binary_crossentropy', metrics=['acc'])
    
    return model

model = build_model()
model.summary()

# ### Training

# In[40]:


weights_path = f'weights.best.hdf5'
val_loss_checkpoint = ModelCheckpoint(weights_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
reduceLR = ReduceLROnPlateau(monitor='val_loss', patience=5, verbose=1, mode='min', min_lr=1e-6)

# In[41]:


model.fit(trn_x, trn_y, epochs=80, validation_data=(valid_x, valid_y),
         callbacks=[val_loss_checkpoint, reduceLR], batch_size=512, verbose=1)

# In[43]:


model.load_weights(weights_path)

# ### roc_auc_score on validation data

# In[44]:


val_preds = model.predict(valid_x, batch_size=2048, verbose=1)

# In[45]:


roc_auc_score(valid_y.values, val_preds.reshape(-1))

# ### Prediction on test data

# In[46]:


test_preds = model.predict(test_x, batch_size=2048, verbose=1)

# In[47]:


sub_df = pd.read_csv(f'{data_dir}sample_submission.csv')
sub_df.target = test_preds.reshape(-1)
sub_df.head()

# In[48]:


sub_df.to_csv('solution.csv', index=False)
