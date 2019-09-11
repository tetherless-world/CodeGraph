#!/usr/bin/env python
# coding: utf-8

# # Import Libraries

# In[1]:


import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
from gensim.models import KeyedVectors
import operator
from tqdm import tqdm
tqdm.pandas()
import gc

# # Define Variables

# In[2]:


epochs=25
batch_size=128
max_words=100000
max_seq_size=256

# # Read Data

# In[3]:


import os
print(os.listdir("../input")) 
print(os.listdir("../input/jigsaw-unintended-bias-in-toxicity-classification"))
print(os.listdir("../input/quoratextemb"))
print(os.listdir("../input/quoratextemb/embeddings"))

# In[4]:


train_df = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv')
test_df  = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv')
sub_df   = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/sample_submission.csv')

# # Explore Data

# In[5]:


train_df.shape

# In[6]:


test_df.shape

# In[7]:


train_df.head()

# In[8]:


test_df.head()

# In[9]:


sub_df.head()

# # Take care of dataframe memory 

# In[10]:


mem_usg = train_df.memory_usage().sum() / 1024**2 
print("Memory usage is: ", mem_usg, " MB")

# I'll select only the columns that we need to reduce some memory usage

# In[11]:


train_df = train_df[["target", "comment_text"]]
mem_usg = train_df.memory_usage().sum() / 1024**2 
print("Memory usage is: ", mem_usg, " MB")

# See? we have more free memory

# # Data Cleaning
# 

# ### Load Embedding

# To increase our covarage we try to combine few embedding together in order for us to more vocab covrage. 
# In term of memory optimize, we will convert our vector to ```float16``` to reduce some memory usage. 

# In[12]:


def combine_embedding(vec_files):
    
    def get_coefs(word, *arr): 
        return word, np.asarray(arr, dtype='float16')

    def optimize_embedding(embedding): 
        optimized_embedding = {}
        for word in embedding.vocab:
            optimized_embedding[word] = np.asarray(embedding[word], dtype='float16')
        return optimized_embedding

    def load_embed(file):
        print("Loading {}".format(file))

        if file == '../input/quoratextemb/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec':
            return dict(get_coefs(*o.strip().split(" ")) for o in open(file) if len(o) > 100)

        elif file == '../input/quoratextemb/embeddings/GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin':
            return optimize_embedding(KeyedVectors.load_word2vec_format(file, binary=True))
        
        elif file == '../input/fasttext-crawl-300d-2m/crawl-300d-2M.vec':
            return optimize_embedding(KeyedVectors.load_word2vec_format(file))

        else:
            return dict(get_coefs(*o.split(" ")) for o in open(file, encoding='latin'))
        
    combined_embedding = {}
    for file in vec_files:
        combined_embedding.update(load_embed(file))
    return combined_embedding

# In[13]:


vec_files = [
    "../input/quoratextemb/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec", 
    "../input/quoratextemb/embeddings/GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin",
    "../input/quoratextemb/embeddings/glove.840B.300d/glove.840B.300d.txt",
    "../input/fasttext-crawl-300d-2m/crawl-300d-2M.vec"
]

# In[14]:


embedding_index = combine_embedding(vec_files)
covered_vocabs = set(list(embedding_index.keys()))
embedding_index.clear()

# In[15]:


gc.collect()

# ### Count occurance of words 

# In[16]:


def count_words_from(series):
    """
    :param sentences: list of list of words
    :return: dictionary of words and their count
    """
    sentences =  series.str.split()
    vocab = {}
    for sentence in tqdm(sentences):
        for word in sentence:
            try:
                vocab[word] += 1
            except KeyError:
                vocab[word] = 1
    return vocab

# ### Check Coverage

# In[17]:


def check_coverage_for(vocab):
    a = 0
    oov = {}
    k = 0
    i = 0
    for word in tqdm(vocab):
        if word in covered_vocabs:
            a += 1
            k += vocab[word]
        else:
            oov[word] = vocab[word]
            i += vocab[word]

    print('Found embeddings for {:.2%} of vocab'.format(a / len(vocab)))
    print('Found embeddings for  {:.2%} of all text'.format(k / (k + i)))
    sorted_x = sorted(oov.items(), key=operator.itemgetter(1))[::-1]

    return sorted_x

# In[18]:


# this methods help to clean up some memory while improve the coverage. As it will release the varaible the send of method
def check_current_coverage(num=50):
    vocab = count_words_from( train_df["comment_text"] )
    coverage = check_coverage_for(vocab)
    return coverage[:num]

# ### Let's check the first coverage 

# In[19]:


check_current_coverage()

# We see only 18% of our vocab has been covered. But 92% of our text has already cover. From the top first uncovered we see we have some problem with contractions. Let's get rid of it. 

# ### Clean contractions

# In[20]:


contraction_mapping = {
    "ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not", 
    "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", 
    "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  
    "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": 
    "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", 
    "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", 
    "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have", 
    "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have",
    "o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", 
    "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", 
    "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not", 
    "shouldn't've": "should not have", "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would", 
    "that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", 
    "here's": "here is","they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", 
    "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", 
    "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", 
    "what'll've": "what will have", "what're": "what are",  "what's": "what is", "what've": "what have", "when's": "when is", 
    "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have",
    "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", 
    "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", 
    "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have",
    "you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have",
    "Trump's": "trump is", "Obama's": "obama is", "Canada's": "canada is", "today's": "today is"
}

# In[21]:


known_contractions = []
for contract in contraction_mapping:
    if contract in covered_vocabs:
        known_contractions.append(contract)
print(known_contractions)

# Our embedding have known some contractions. So we will remove that known contractions from our dictionary and let's our embedding handle it. 

# In[22]:


for cont in known_contractions:
    contraction_mapping.pop(cont)

# In[23]:


def clean_contractions(text):
    specials = ["’", "‘", "´", "`"]
    for s in specials:
        text = text.replace(s, "'")
    words = [contraction_mapping[word] if word in contraction_mapping else word for word in text.split(" ")]
    return ' '.join(words)

# In[24]:


train_df["comment_text"] = train_df["comment_text"].progress_apply(lambda text: clean_contractions(text))

# Let check coverage again

# In[25]:


check_current_coverage()

# Our covarage have increase a little. But at least we are good for next step. From the top uncovered we see that we have problem with some specials charator. 

# #### Clean special characters

# In[26]:


punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'

# In[27]:


specail_signs = { "…": "...", "₂": "2"}

# In[28]:


unknown_puncts = []
for p in punct:
    if p not in covered_vocabs:
        unknown_puncts.append(p)
print(' '.join(unknown_puncts))

# All contraction are known

# In[29]:


def clean_special_chars(text):
    for s in specail_signs: 
        text = text.replace(s, specail_signs[s])
    for p in punct:
        text = text.replace(p, f' {p} ')
    return text

# In[30]:


train_df["comment_text"] = train_df["comment_text"].progress_apply(lambda text: clean_special_chars(text))

# Check coverage again

# In[31]:


check_current_coverage()

# This is the excited part that we have our coverage increase much. Our vocab coverage has increase to 65% and Most of our text (99.6%) has been covered

# # Clean Small Caps

# What is small caps? check it out [here](https://en.wikipedia.org/wiki/Small_caps)
# 
# We see some like like ```ʜᴏᴍᴇ```, ```ᴜᴘ```, ```ᴄʜᴇᴄᴋ``` etc ... We need to convert it to up, home, check ....

# In[32]:


small_caps_mapping = { 
    "ᴀ": "a", "ʙ": "b", "ᴄ": "c", "ᴅ": "d", "ᴇ": "e", "ғ": "f", "ɢ": "g", "ʜ": "h", "ɪ": "i", 
    "ᴊ": "j", "ᴋ": "k", "ʟ": "l", "ᴍ": "m", "ɴ": "n", "ᴏ": "o", "ᴘ": "p", "ǫ": "q", "ʀ": "r", 
    "s": "s", "ᴛ": "t", "ᴜ": "u", "ᴠ": "v", "ᴡ": "w", "x": "x", "ʏ": "y", "ᴢ": "z"
}

# In[33]:


def clean_small_caps(text):
    for char in small_caps_mapping:
        text = text.replace(char, small_caps_mapping[char])
    return text

# In[34]:


train_df["comment_text"] = train_df["comment_text"].progress_apply(lambda text: clean_small_caps(text))

# In[35]:


check_current_coverage()

# # Let's clean up of testing set

# * Convert to lower case
# * Clean contractions
# * Clean special charactor
# * Convert small caps

# In[36]:


def clean_up_text_with_all_process(text):
    text = text.lower()
    text = clean_contractions(text)
    text = clean_special_chars(text)
    text = clean_small_caps(text)
    return text

# In[ ]:


test_df["comment_text"] = test_df["comment_text"].progress_apply(lambda text: clean_up_text_with_all_process(text))

# # Transform Text 

# In[ ]:


tranformer = Tokenizer(lower = True, filters='', num_words=max_words)
tranformer.fit_on_texts( list(train_df["comment_text"].values) + list(test_df["comment_text"].values) )

# Transform training set

# In[ ]:


transformed_x = tranformer.texts_to_sequences(train_df["comment_text"].values)
transformed_x = pad_sequences(transformed_x, maxlen = max_seq_size)

# Transform predicting set

# In[ ]:


x_predict = tranformer.texts_to_sequences(test_df["comment_text"])
x_predict = pad_sequences(x_predict, maxlen = max_seq_size)

# # Build Martix

# In[ ]:


def build_embedding_matrix(word_index, total_vocab, embedding_size):
    embedding_index = combine_embedding(vec_files)
    matrix = np.zeros((total_vocab, embedding_size))
    for word, index in tqdm(word_index.items()):
        try:
            matrix[index] = embedding_index[word]
        except KeyError:
            pass
    return matrix

# In[ ]:


word_index = tranformer.word_index
total_vocab = len(word_index) + 1
embedding_size = 300
embedding_matrix = build_embedding_matrix(tranformer.word_index, total_vocab, embedding_size)

# # Clean up some memory

# Let free up some memory before to other hard job. I'll clean ```vocab``` and ```coverage``` up in order for us to have enough memory to continue

# In[ ]:


del tranformer
del word_index
del embedding_index
gc.collect()

# # Select features and Target

# In[ ]:


y = (train_df['target'].values > 0.5).astype(int)
x_train, x_test, y_train, y_test = train_test_split(transformed_x, y, random_state=10, test_size=0.15)

# Let clean memory again,  I'll clean ```word_index``` and ```embedding_index``` up in order for us to have enough memory for training

# In[ ]:


del train_df
del y
del test_df
del transformed_x
gc.collect()

# # Build Model

# In[ ]:


from tensorflow.nn import relu, sigmoid
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Embedding, SpatialDropout1D, Dropout, add, concatenate
from tensorflow.keras.layers import CuDNNGRU, Bidirectional, GlobalMaxPooling1D, GlobalAveragePooling1D, Conv1D

# In[ ]:


sequence_input = Input(shape=(max_seq_size,), dtype='int32')
embedding_layer = Embedding(total_vocab,
                            embedding_size,
                            weights=[embedding_matrix],
                            input_length=max_seq_size,
                            trainable=False)

x_layer = embedding_layer(sequence_input)
x_layer = SpatialDropout1D(0.2)(x_layer)
x_layer = Bidirectional(CuDNNGRU(64, return_sequences=True))(x_layer)   
x_layer = Conv1D(64, kernel_size = 2, padding = "valid", kernel_initializer = "he_uniform")(x_layer)

avg_pool1 = GlobalAveragePooling1D()(x_layer)
max_pool1 = GlobalMaxPooling1D()(x_layer)     

x_layer = concatenate([avg_pool1, max_pool1])

preds = Dense(1, activation=sigmoid)(x_layer)

model = Model(sequence_input, preds)
model.summary()

# # Compile Model

# In[ ]:


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# # Callbacks

# In[ ]:


from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# In[ ]:


callbacks = [
    EarlyStopping(patience=10, verbose=1),
    ReduceLROnPlateau(factor=0.1, patience=3, min_lr=0.00001, verbose=1),
    ModelCheckpoint('model.h5', verbose=1, save_best_only=True, save_weights_only=True)
]

# # Train Model

# In[ ]:


history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test), callbacks=callbacks)

# # Virtualize Training

# In[ ]:


fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
ax1.plot(history.history['loss'], color='b', label="Training loss")
ax1.plot(history.history['val_loss'], color='r', label="validation loss")
ax1.set_xticks(np.arange(1, epochs, 1))
plt.legend(loc='best', shadow=True)

ax2.plot(history.history['acc'], color='b', label="Training accuracy")
ax2.plot(history.history['val_acc'], color='r',label="Validation accuracy")
ax2.set_xticks(np.arange(1, epochs, 1))
plt.legend(loc='best', shadow=True)
plt.tight_layout()
plt.show()

# # Evaluate Model

# In[ ]:


score = model.evaluate(x_test, y_test, batch_size=batch_size)

# In[ ]:


print('Test loss:', score[0])
print('Test accuracy:', score[1])

# # Prediction

# In[ ]:


y_predict = model.predict(x_predict)

# In[ ]:


print(y_predict)

# # Submission

# In[ ]:


sub_df["prediction"] = y_predict

# In[ ]:


sub_df.head()

# In[ ]:


sub_df.to_csv("submission.csv", index=False)

# Working in progress...
