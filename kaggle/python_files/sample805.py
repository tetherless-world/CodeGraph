#!/usr/bin/env python
# coding: utf-8

# # Preface

# In this notebook I want to share some knowledge I gained since I wrote the popular preprocessing kernel for the Quora challenge https://www.kaggle.com/christofhenkel/how-to-preprocessing-when-using-embeddings
# 
# Since I am rather lazy, I forked Benjamins https://www.kaggle.com/bminixhofer/speed-up-your-rnn-with-sequence-bucketing to have a solid starting point. In the following I want to share 3 tricks that not only speed up the preprocessing a bit, but also improve a models accuracy. REMARK: Right after I finished I realized I run into memory issues if I do EDA and modelling in one kernel, so I'll have to split into 2 kernels. Sorry for that...
# 
# The 3 main contributions of this kernel are the following:
# 
# - loading embedding from pickles 
# - aimed preprocessing for GloVe and fasttext vectors (the main content of this notebook)
# - fixing some unknown words
# 
# What I will not cover are word-specific preprocessing steps like handling contractions, or mispellings (again, since I am rather lazy and do not want to hardcode dictionaries).
# 
# The neural network architecture is taken from the best scoring public kernel at the time of writing: [Simple LSTM with Identity Parameters - Fast AI](https://www.kaggle.com/kunwar31/simple-lstm-with-identity-parameters-fastai).

# In[1]:


# Put these at the top of every notebook, to get automatic reloading and inline plotting

import fastai
from fastai.train import Learner
from fastai.train import DataBunch
from fastai.callbacks import *
from fastai.basic_data import DatasetType
import fastprogress
from fastprogress import force_console_behavior
import numpy as np
from pprint import pprint
import pandas as pd
import os
import time

import gc
import random
from tqdm._tqdm_notebook import tqdm_notebook as tqdm
from keras.preprocessing import text, sequence
import torch
from torch import nn
from torch.utils import data
from torch.nn import functional as F


# In[2]:


tqdm.pandas()

# In[3]:


# disable progress bars when submitting
def is_interactive():
   return 'SHLVL' not in os.environ

if not is_interactive():
    def nop(it, *a, **k):
        return it

    tqdm = nop

    fastprogress.fastprogress.NO_BAR = True
    master_bar, progress_bar = force_console_behavior()
    fastai.basic_train.master_bar, fastai.basic_train.progress_bar = master_bar, progress_bar

# In[4]:


def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
seed_everything()

# Here, compared to most other public kernels I replace the pretrained embedding files with their pickle corresponds. Loading a pickled version extremly improves timing ;)

# In[5]:


CRAWL_EMBEDDING_PATH = '../input/pickled-crawl300d2m-for-kernel-competitions/crawl-300d-2M.pkl'
GLOVE_EMBEDDING_PATH = '../input/pickled-glove840b300d-for-10sec-loading/glove.840B.300d.pkl'

# Of course we also need to adjust the load_embeddings function, to now handle the pickled dict.

# In[6]:


NUM_MODELS = 2
LSTM_UNITS = 128
DENSE_HIDDEN_UNITS = 4 * LSTM_UNITS
MAX_LEN = 220

def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype='float32')


def load_embeddings(path):
    with open(path,'rb') as f:
        emb_arr = pickle.load(f)
    return emb_arr

def build_matrix(word_index, path):
    embedding_index = load_embeddings(path)
    embedding_matrix = np.zeros((len(word_index) + 1, 300))
    unknown_words = []
    
    for word, i in word_index.items():
        try:
            embedding_matrix[i] = embedding_index[word]
        except KeyError:
            unknown_words.append(word)
    return embedding_matrix, unknown_words

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class SpatialDropout(nn.Dropout2d):
    def forward(self, x):
        x = x.unsqueeze(2)    # (N, T, 1, K)
        x = x.permute(0, 3, 2, 1)  # (N, K, 1, T)
        x = super(SpatialDropout, self).forward(x)  # (N, K, 1, T), some features are masked
        x = x.permute(0, 3, 2, 1)  # (N, T, 1, K)
        x = x.squeeze(2)  # (N, T, K)
        return x

def train_model(learn,test,output_dim,lr=0.001,
                batch_size=512, n_epochs=4,
                enable_checkpoint_ensemble=True):
    
    all_test_preds = []
    checkpoint_weights = [2 ** epoch for epoch in range(n_epochs)]
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)
    n = len(learn.data.train_dl)
    phases = [(TrainingPhase(n).schedule_hp('lr', lr * (0.6**(i)))) for i in range(n_epochs)]
    sched = GeneralScheduler(learn, phases)
    learn.callbacks.append(sched)
    for epoch in range(n_epochs):
        learn.fit(1)
        test_preds = np.zeros((len(test), output_dim))    
        for i, x_batch in enumerate(test_loader):
            X = x_batch[0].cuda()
            y_pred = sigmoid(learn.model(X).detach().cpu().numpy())
            test_preds[i * batch_size:(i+1) * batch_size, :] = y_pred

        all_test_preds.append(test_preds)


    if enable_checkpoint_ensemble:
        test_preds = np.average(all_test_preds, weights=checkpoint_weights, axis=0)    
    else:
        test_preds = all_test_preds[-1]
        
    return test_preds


# In[ ]:




# Let's discuss the function, which is most popular in most public kernels.

# In[7]:


def bad_preprocess(data):
    '''
    Credit goes to https://www.kaggle.com/gpreda/jigsaw-fast-compact-solution
    '''
    punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~`" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'
    def clean_special_chars(text, punct):
        for p in punct:
            text = text.replace(p, ' ')
        return text

    data = data.astype(str).apply(lambda x: clean_special_chars(x, punct))
    return data

# In principle this functions just deletes some special characters. Which is not optimal and I will explain why in a bit. What is additionally inefficient is that later the keras tokenizer with its default parameters is used which has its own with the above function redundant behavior.

# In[8]:


train = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv')
test = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv')

# ## Preprocessing

# ### Important remarks
# Let me start with some remarks, which I also made in the quora notebook:
# 
# 1.  **Don't naively use standard preprocessing steps like stemming, lowercasing or stopword removal when you have pre-trained embeddings** 
# 
# Some of you might used standard preprocessing steps when doing word count based feature extraction (e.g. TFIDF) such as removing stopwords, stemming etc. 
# The reason is simple: You loose valuable information, which would help your NN to figure things out.  
# 
# 2. **Get your vocabulary as close to the embeddings as possible**
# 
# I will focus in this notebook, how to achieve that.
# 
# Getting your vocabulary close to the pretrained embeddings means, that you should aim for your preprocessing to result in tokens that are mostly covered by word vectors. That leads to two conclusions:
# 
# 1. Setting up the preprocessing is some eda and research work
# 
# If a word vector for a token (see remark below for what I mean with token) is available strongly depends on the preprocessing used by the people who trained the embeddings. Unfortunatly most are quite intransparent about this point. (e.g. did they use lower casing, removing contractions, replacement of words, etc. So you need to research their github repositories and/or read the related papers. Did you now the Google pretrained word vectors replace numbers with "##" or the guys training glove twitter embeddings did `text = re.sub("<3", '<HEART>', text)` 
# That all leads to the second conclusion:
# 
# 2. Each pretrained embedding needs its own preprocessing
# 
# If people used different preprocessing for training their embeddings you would also need to do the same, 
# 
# 
# Especially point to can be quite challenging, if you want to concatenate embeddings as in this kernel. Imagine Embedding A preprocesses `"don't"` to a single token`["dont"]` and Embedding B to two tokens`["do","n't"]`. You are basically not able to do both. So you need to find a compromise.
# 
# 
# 
# *(most of the times token and word is the same, but sometimes e.g. "?", "n't" are not words, so I use the term token instead) 

# In[ ]:




# Lets start with two function I mainly use for the EDA. The first one goes through a given vocabulary and tries to find word vectors in your embedding matrix. `build_vocab` builds a ordered dictionary of words and their frequency in your text corpus.

# In[9]:


import operator 

def check_coverage(vocab,embeddings_index):
    a = {}
    oov = {}
    k = 0
    i = 0
    for word in tqdm(vocab):
        try:
            a[word] = embeddings_index[word]
            k += vocab[word]
        except:

            oov[word] = vocab[word]
            i += vocab[word]
            pass

    print('Found embeddings for {:.2%} of vocab'.format(len(a) / len(vocab)))
    print('Found embeddings for  {:.2%} of all text'.format(k / (k + i)))
    sorted_x = sorted(oov.items(), key=operator.itemgetter(1))[::-1]

    return sorted_x

def build_vocab(sentences, verbose =  True):
    """
    :param sentences: list of list of words
    :return: dictionary of words and their count
    """
    vocab = {}
    for sentence in tqdm(sentences, disable = (not verbose)):
        for word in sentence:
            try:
                vocab[word] += 1
            except KeyError:
                vocab[word] = 1
    return vocab

# lets load the two embeddings and time the loading process

# In[10]:


tic = time.time()
glove_embeddings = load_embeddings(GLOVE_EMBEDDING_PATH)
print(f'loaded {len(glove_embeddings)} word vectors in {time.time()-tic}s')

# 10s compared to 2min in the other public kernels ;) So lets build our vocab and check the embeddings coverage without any preprocessing.

# 

# In[11]:


vocab = build_vocab(list(train['comment_text'].apply(lambda x:x.split())))
oov = check_coverage(vocab,glove_embeddings)
oov[:10]

# In[12]:


oov[:10]

# Seems like `'` and other punctuation directly on or in a word is an issue. We could simply delete punctuation to fix that words, but there are better methods. Lets explore the embeddings, in particular symbols a bit. For that we first need to define "what is a symbol" in contrast to a regular letter. I nowadays use the following list for "regular" letters. And symbols are all characters not in that list.

# In[13]:


import string
latin_similar = "’'‘ÆÐƎƏƐƔĲŊŒẞÞǷȜæðǝəɛɣĳŋœĸſßþƿȝĄƁÇĐƊĘĦĮƘŁØƠŞȘŢȚŦŲƯY̨Ƴąɓçđɗęħįƙłøơşșţțŧųưy̨ƴÁÀÂÄǍĂĀÃÅǺĄÆǼǢƁĆĊĈČÇĎḌĐƊÐÉÈĖÊËĚĔĒĘẸƎƏƐĠĜǦĞĢƔáàâäǎăāãåǻąæǽǣɓćċĉčçďḍđɗðéèėêëěĕēęẹǝəɛġĝǧğģɣĤḤĦIÍÌİÎÏǏĬĪĨĮỊĲĴĶƘĹĻŁĽĿʼNŃN̈ŇÑŅŊÓÒÔÖǑŎŌÕŐỌØǾƠŒĥḥħıíìiîïǐĭīĩįịĳĵķƙĸĺļłľŀŉńn̈ňñņŋóòôöǒŏōõőọøǿơœŔŘŖŚŜŠŞȘṢẞŤŢṬŦÞÚÙÛÜǓŬŪŨŰŮŲỤƯẂẀŴẄǷÝỲŶŸȲỸƳŹŻŽẒŕřŗſśŝšşșṣßťţṭŧþúùûüǔŭūũűůųụưẃẁŵẅƿýỳŷÿȳỹƴźżžẓ"
white_list = string.ascii_letters + string.digits + latin_similar + ' '
white_list += "'"

# In[14]:


glove_chars = ''.join([c for c in tqdm(glove_embeddings) if len(c) == 1])
glove_symbols = ''.join([c for c in glove_chars if not c in white_list])
glove_symbols

# So lets have closer look on what we just did. We printed all symbols that we have an embedding vector for. Intrestingly its not only a vast amount of punctuation but also emojis and other symbols. Especially when doing sentiment analysis emojis and other symbols carrying sentiments should not be deleted! What we can delete are symbols we have no embeddings for. So lets check the characters in our texts and find those to delete:

# In[15]:


jigsaw_chars = build_vocab(list(train["comment_text"]))
jigsaw_symbols = ''.join([c for c in jigsaw_chars if not c in white_list])
jigsaw_symbols

# Basically we can delete all symbols we have no embeddings for:

# In[16]:


symbols_to_delete = ''.join([c for c in jigsaw_symbols if not c in glove_symbols])
symbols_to_delete

# The symbols we want to keep we need to isolate from our words. So lets setup a list of those to isolate.

# In[17]:


symbols_to_isolate = ''.join([c for c in jigsaw_symbols if c in glove_symbols])
symbols_to_isolate

# Next comes the next trick. Instead of using an inefficient loop of `replace` we use `translate`. I find the syntax a bit weird, but the improvement in speed is worth the worse readablity.

# In[18]:


isolate_dict = {ord(c):f' {c} ' for c in symbols_to_isolate}
remove_dict = {ord(c):f'' for c in symbols_to_delete}


def handle_punctuation(x):
    x = x.translate(remove_dict)
    x = x.translate(isolate_dict)
    return x

# So lets apply that function to our text and reasses the coverage

# In[19]:


train['comment_text'] = train['comment_text'].progress_apply(lambda x:handle_punctuation(x))
test['comment_text'] = test['comment_text'].progress_apply(lambda x:handle_punctuation(x))

# In[20]:


vocab = build_vocab(list(train['comment_text'].apply(lambda x:x.split())))
oov = check_coverage(vocab,glove_embeddings)
oov[:10]

# In[21]:


from nltk.tokenize.treebank import TreebankWordTokenizer
tokenizer = TreebankWordTokenizer()

# In[22]:


def handle_contractions(x):
    x = tokenizer.tokenize(x)
    x = ' '.join(x)
    return x

# In[23]:


train['comment_text'] = train['comment_text'].progress_apply(lambda x:handle_contractions(x))
test['comment_text'] = test['comment_text'].progress_apply(lambda x:handle_contractions(x))

# In[24]:


vocab = build_vocab(list(train['comment_text'].apply(lambda x:x.split())),verbose=False)
oov = check_coverage(vocab,glove_embeddings)
oov[:10]

# Now the oov words look "normal", apart from those still carrying the `'` token in the beginning of the word. Will need to fix those "per hand"

# In[25]:


def fix_quote(x):
    x = [x_[1:] if x_.startswith("'") else x_ for x_ in x]
    x = ' '.join(x)
    return x

# In[26]:


train['comment_text'] = train['comment_text'].progress_apply(lambda x:fix_quote(x.split()))
test['comment_text'] = test['comment_text'].progress_apply(lambda x:fix_quote(x.split()))

# In[27]:


train['comment_text'].head()

# In[28]:


vocab = build_vocab(list(train['comment_text'].apply(lambda x:x.split())),verbose=False)
oov = check_coverage(vocab,glove_embeddings)
oov[:50]

# Looks good. Now we can implement the preprocessing functions and train a model. See 
