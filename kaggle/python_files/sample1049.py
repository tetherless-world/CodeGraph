#!/usr/bin/env python
# coding: utf-8

# # Update
# __There was an issue with the random seed from `os.random` and the seed when creating the embedding matrix in version 1 - 3. Version 2 got a lucky seed and scored 0.694 for that reason. The problem is fixed since version 4.__
# 
# If you look at the version history you can see that the validation and training loss for version 4 and 5 are exactly the same so it is 100% reproducible now. But it dropped to 0.690 because the seed is less lucky on the Leaderboard. The CV score is even slightly better though.

# # Preface

# There have been many problems with reproducibility of neural networks in this competition. See for example [this post in the discussion forum](https://www.kaggle.com/c/quora-insincere-questions-classification/discussion/73341).
# 
# There, it is recommended to 
# 
# > Rerun the experiment 10 - 15 times and average the scores to get the progress of your model.
# 
# That's obviously not a good solution because when optimizing the architecture of our NN we would of course like to be as fast as possible while still being sure that our model actually improves. Deviations of up to 0.01 in the F1 score are too large to be even remotely sure of that.
# 
# The problem lies within CuDNN. CuDNN's implementation of GRU and LSTM is [much faster](https://chainer.org/general/2017/03/15/Performance-of-LSTM-Using-CuDNN-v5.html) than the regular implementation but they do not run deterministically in TensorFlow and Keras. In this competition were speed is essential you can not afford to keep determinism  by using the regular implementation of GRU and LSTM.
# 
# ## PyTorch to the rescue!
# 
# In PyTorch, CuDNN determinism is a one-liner: `torch.backends.cudnn.deterministic = True`. This already solves the problem everyone has had so far with Keras. But that's not the only advantage of PyTorch. PyTorch is:
# 
# - significantly faster than Keras and TensorFlow. Again, speed is important in this competition so this is great.
# - has a more pythonic API. I hate working with TensorFlow because there are seemingly tens of thousands of ways to do simple things. PyTorch has (in most cases) one obvious way and is by far not as convoluted as TensorFlow.
# - is executed eagerly. There is no such thing as an execution graph in PyTorch. That makes it much easier to try new things and interact with PyTorch in a notebook.
# 
# Keras solves some of these problems with TensorFlow but it has a high-level API. I think that when doing research, it is often preferable to be able to interact with the model on a low-level. And you will see that the lower level API still doesn't make it complicated to work with PyTorch.

# # Imports

# In[ ]:


# standard imports
import time
import random
import os
from IPython.display import display
import numpy as np
import pandas as pd

# pytorch imports
import torch
import torch.nn as nn
import torch.utils.data

# imports for preprocessing the questions
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# cross validation and metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

# progress bars
from tqdm import tqdm
tqdm.pandas()

# # Loading the data

# In[ ]:


train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")
print('Train data dimension: ', train_df.shape)
display(train_df.head())
print('Test data dimension: ', test_df.shape)
display(test_df.head())

# # Utility functions

# `seed_torch` sets the seed for numpy and torch to make sure functions with a random component behave deterministically. `torch.backends.cudnn.deterministic = true` sets the CuDNN to deterministic mode. 
# 
# This function allows us to run experiments 100% deterministically.

# In[ ]:


def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
seed_everything()

# Function to search for best threshold regarding the F1 score given labels and predictions from the network.

# In[ ]:


def threshold_search(y_true, y_proba):
    best_threshold = 0
    best_score = 0
    for threshold in tqdm([i * 0.01 for i in range(100)]):
        score = f1_score(y_true=y_true, y_pred=y_proba > threshold)
        if score > best_score:
            best_threshold = threshold
            best_score = score
    search_result = {'threshold': best_threshold, 'f1': best_score}
    return search_result

# Sigmoid function in plain numpy.

# In[ ]:


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# # Processing input

# Standard preprocessing procedure. This is not the point of this kernel so I have copied it from [this great kernel](https://www.kaggle.com/gmhost/gru-capsule).

# In[ ]:


embed_size = 300 # how big is each word vector
max_features = 95000 # how many unique words to use (i.e num rows in embedding vector)
maxlen = 70 # max number of words in a question to use

# In[ ]:


puncts = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•',  '~', '@', '£', 
 '·', '_', '{', '}', '©', '^', '®', '`',  '<', '→', '°', '€', '™', '›',  '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…', 
 '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─', 
 '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾', 'Ã', '⋅', '‘', '∞', 
 '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√', ]
def clean_text(x):
    x = str(x)
    for punct in puncts:
        x = x.replace(punct, f' {punct} ')
    return x

# In[ ]:


train_df["question_text"] = train_df["question_text"].str.lower()
test_df["question_text"] = test_df["question_text"].str.lower()

train_df["question_text"] = train_df["question_text"].apply(lambda x: clean_text(x))
test_df["question_text"] = test_df["question_text"].apply(lambda x: clean_text(x))

# fill up the missing values
x_train = train_df["question_text"].fillna("_##_").values
x_test = test_df["question_text"].fillna("_##_").values

# Tokenize the sentences
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(x_train))
x_train = tokenizer.texts_to_sequences(x_train)
x_test = tokenizer.texts_to_sequences(x_test)

# Pad the sentences 
x_train = pad_sequences(x_train, maxlen=maxlen)
x_test = pad_sequences(x_test, maxlen=maxlen)

# Get the target values
y_train = train_df['target'].values

# # Creating the embeddings matrix

# Another step that many others have already done. Again, the same progress as in the kernel from above.

# In[ ]:


def load_glove(word_index):
    EMBEDDING_FILE = '../input/embeddings/glove.840B.300d/glove.840B.300d.txt'
    def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')[:300]
    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE))
    
    all_embs = np.stack(embeddings_index.values())
    emb_mean,emb_std = -0.005838499,0.48782197
    embed_size = all_embs.shape[1]

    # word_index = tokenizer.word_index
    nb_words = min(max_features, len(word_index))
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
    for word, i in word_index.items():
        if i >= max_features: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: embedding_matrix[i] = embedding_vector
            
    return embedding_matrix 
    
def load_fasttext(word_index):    
    EMBEDDING_FILE = '../input/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec'
    def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE) if len(o)>100)

    all_embs = np.stack(embeddings_index.values())
    emb_mean,emb_std = all_embs.mean(), all_embs.std()
    embed_size = all_embs.shape[1]

    # word_index = tokenizer.word_index
    nb_words = min(max_features, len(word_index))
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
    for word, i in word_index.items():
        if i >= max_features: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: embedding_matrix[i] = embedding_vector

    return embedding_matrix

def load_para(word_index):
    EMBEDDING_FILE = '../input/embeddings/paragram_300_sl999/paragram_300_sl999.txt'
    def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE, encoding="utf8", errors='ignore') if len(o)>100)

    all_embs = np.stack(embeddings_index.values())
    emb_mean,emb_std = -0.0053247833,0.49346462
    embed_size = all_embs.shape[1]

    # word_index = tokenizer.word_index
    nb_words = min(max_features, len(word_index))
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
    for word, i in word_index.items():
        if i >= max_features: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: embedding_matrix[i] = embedding_vector
    
    return embedding_matrix

# In[ ]:


# missing entries in the embedding are set using np.random.normal so we have to seed here too
seed_everything()

glove_embeddings = load_glove(tokenizer.word_index)
paragram_embeddings = load_para(tokenizer.word_index)

embedding_matrix = np.mean([glove_embeddings, paragram_embeddings], axis=0)
np.shape(embedding_matrix)

# # Defining the model

# First, define 5-Fold cross-validation. The `random_state` here is important to make sure this is deterministic too.

# In[ ]:


splits = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=10).split(x_train, y_train))

# Now it gets interesting. First, I ported the Attention mechanism many others have used in this competition to PyTorch. I am not sure where the Keras snippet originated from, so I am going to give credit to the [kernel where I have first seen it](https://www.kaggle.com/shujian/single-rnn-with-4-folds-clr).

# In[ ]:


class Attention(nn.Module):
    def __init__(self, feature_dim, step_dim, bias=True, **kwargs):
        super(Attention, self).__init__(**kwargs)
        
        self.supports_masking = True

        self.bias = bias
        self.feature_dim = feature_dim
        self.step_dim = step_dim
        self.features_dim = 0
        
        weight = torch.zeros(feature_dim, 1)
        nn.init.xavier_uniform_(weight)
        self.weight = nn.Parameter(weight)
        
        if bias:
            self.b = nn.Parameter(torch.zeros(step_dim))
        
    def forward(self, x, mask=None):
        feature_dim = self.feature_dim
        step_dim = self.step_dim

        eij = torch.mm(
            x.contiguous().view(-1, feature_dim), 
            self.weight
        ).view(-1, step_dim)
        
        if self.bias:
            eij = eij + self.b
            
        eij = torch.tanh(eij)
        a = torch.exp(eij)
        
        if mask is not None:
            a = a * mask

        a = a / torch.sum(a, 1, keepdim=True) + 1e-10

        weighted_input = x * torch.unsqueeze(a, -1)
        return torch.sum(weighted_input, 1)
    
class SpatialDropout(nn.Dropout2d):
    def forward(self, x):
        x = x.unsqueeze(2)    # (N, T, 1, K)
        x = x.permute(0, 3, 2, 1)  # (N, K, 1, T)
        x = super(SpatialDropout, self).forward(x)  # (N, K, 1, T), some features are masked
        x = x.permute(0, 3, 2, 1)  # (N, T, 1, K)
        x = x.squeeze(2)  # (N, T, K)
        return x

# Now define the neural network. Defining a neural network in PyTorch is done by defining a class. This is almost as intuitive as Keras. The main difference is that you have one function (`__init__`) where it is defined which layers there are in the network and another function (`forward`) which defines the flow of data through the net.
# 
# I replicated the architecture used in [@Shujian Liu's kernel](https://www.kaggle.com/shujian/single-rnn-with-4-folds-clr) in the network.

# In[ ]:


class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        
        hidden_size = 40
        
        self.embedding = nn.Embedding(max_features, embed_size)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False
        
        self.embedding_dropout = SpatialDropout(0.1)
        self.lstm = nn.LSTM(embed_size, hidden_size, bidirectional=True, batch_first=True)
        self.gru = nn.GRU(hidden_size * 2, hidden_size, bidirectional=True, batch_first=True)
        
        self.lstm_attention = Attention(hidden_size * 2, maxlen)
        self.gru_attention = Attention(hidden_size * 2, maxlen)
        
        self.linear = nn.Linear(320, 16)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.out = nn.Linear(16, 1)
    
    def forward(self, x):
        h_embedding = self.embedding(x)
        h_embedding = self.embedding_dropout(h_embedding)
        
        h_lstm, _ = self.lstm(h_embedding)
        h_gru, _ = self.gru(h_lstm)
        
        h_lstm_atten = self.lstm_attention(h_lstm)
        h_gru_atten = self.gru_attention(h_gru)
        
        # global average pooling
        avg_pool = torch.mean(h_gru, 1)
        # global max pooling
        max_pool, _ = torch.max(h_gru, 1)
        
        conc = torch.cat((h_lstm_atten, h_gru_atten, avg_pool, max_pool), 1)
        conc = self.relu(self.linear(conc))
        conc = self.dropout(conc)
        out = self.out(conc)
        
        return out

# # Training

# In[ ]:


batch_size = 512 # how many samples to process at once
n_epochs = 6 # how many times to iterate over all samples

# Now we can already train the network. Unfortunately, we do not have an API as high-level as keras's `.fit` in PyTorch. However, the code is still not too complicated and I have added comments where necessary.

# In[ ]:


# matrix for the out-of-fold predictions
train_preds = np.zeros((len(train_df)))
# matrix for the predictions on the test set
test_preds = np.zeros((len(test_df)))

# always call this before training for deterministic results
seed_everything()

x_test_cuda = torch.tensor(x_test, dtype=torch.long).cuda()
test = torch.utils.data.TensorDataset(x_test_cuda)
test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)

for i, (train_idx, valid_idx) in enumerate(splits):    
    # split data in train / validation according to the KFold indeces
    # also, convert them to a torch tensor and store them on the GPU (done with .cuda())
    x_train_fold = torch.tensor(x_train[train_idx], dtype=torch.long).cuda()
    y_train_fold = torch.tensor(y_train[train_idx, np.newaxis], dtype=torch.float32).cuda()
    x_val_fold = torch.tensor(x_train[valid_idx], dtype=torch.long).cuda()
    y_val_fold = torch.tensor(y_train[valid_idx, np.newaxis], dtype=torch.float32).cuda()
    
    model = NeuralNet()
    # make sure everything in the model is running on the GPU
    model.cuda()

    # define binary cross entropy loss
    # note that the model returns logit to take advantage of the log-sum-exp trick 
    # for numerical stability in the loss
    loss_fn = torch.nn.BCEWithLogitsLoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters())

    train = torch.utils.data.TensorDataset(x_train_fold, y_train_fold)
    valid = torch.utils.data.TensorDataset(x_val_fold, y_val_fold)
    
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid, batch_size=batch_size, shuffle=False)
    
    print(f'Fold {i + 1}')
    
    for epoch in range(n_epochs):
        # set train mode of the model. This enables operations which are only applied during training like dropout
        start_time = time.time()
        model.train()
        avg_loss = 0.  
        for x_batch, y_batch in tqdm(train_loader, disable=True):
            # Forward pass: compute predicted y by passing x to the model.
            y_pred = model(x_batch)

            # Compute and print loss.
            loss = loss_fn(y_pred, y_batch)

            # Before the backward pass, use the optimizer object to zero all of the
            # gradients for the Tensors it will update (which are the learnable weights
            # of the model)
            optimizer.zero_grad()

            # Backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()

            # Calling the step function on an Optimizer makes an update to its parameters
            optimizer.step()
            avg_loss += loss.item() / len(train_loader)
            
        # set evaluation mode of the model. This disabled operations which are only applied during training like dropout
        model.eval()
        
        # predict all the samples in y_val_fold batch per batch
        valid_preds_fold = np.zeros((x_val_fold.size(0)))
        test_preds_fold = np.zeros((len(test_df)))
        
        avg_val_loss = 0.
        for i, (x_batch, y_batch) in enumerate(valid_loader):
            y_pred = model(x_batch).detach()
            
            avg_val_loss += loss_fn(y_pred, y_batch).item() / len(valid_loader)
            valid_preds_fold[i * batch_size:(i+1) * batch_size] = sigmoid(y_pred.cpu().numpy())[:, 0]
        
        elapsed_time = time.time() - start_time 
        print('Epoch {}/{} \t loss={:.4f} \t val_loss={:.4f} \t time={:.2f}s'.format(
            epoch + 1, n_epochs, avg_loss, avg_val_loss, elapsed_time))
        
    # predict all samples in the test set batch per batch
    for i, (x_batch,) in enumerate(test_loader):
        y_pred = model(x_batch).detach()

        test_preds_fold[i * batch_size:(i+1) * batch_size] = sigmoid(y_pred.cpu().numpy())[:, 0]

    train_preds[valid_idx] = valid_preds_fold
    test_preds += test_preds_fold / len(splits)

# # Evaluation

# First, search for the best threshold:

# In[ ]:


search_result = threshold_search(y_train, train_preds)
search_result

# That seems inline with the score from the replicated Keras kernel!

# Finally submit the predictions with the threshold we have just found.

# In[ ]:


submission = test_df[['qid']].copy()
submission['prediction'] = test_preds > search_result['threshold']
submission.to_csv('submission.csv', index=False)

# ## Ways to improve this kernel

# This kernel is intended as a demonstration of PyTorch. I did not spend any time tuning anything, I just ported the models to PyTorch. So, ways to improve this kernel are:
# 
# - Tune the architecture! Now that training is deterministic it should be much less frustrating
# - Increase the number of folds in K-Fold cross-validation. Now that training is faster we can fit more folds into the kernel.
# - Or, keep the folds the same and increase the number of epochs / decrease the learning rate to improve the model at the cost of more time to train.
# - Load the weights with the best validation score after training (implement the equivalent of `ModelCheckpoint` in PyTorch). I am not sure if this will improve the score because you might overfit to the validation data.
# - Use a PyTorch implementation of CLR (cyclic learning rate). That seemed to make the model converge faster in some other kernels.
