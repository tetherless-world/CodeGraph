#!/usr/bin/env python
# coding: utf-8

# # Improve your Score with some Text Preprocessing
# 
# 
# This kernel is an improved version of @Dieter's work.
# > https://www.kaggle.com/christofhenkel/how-to-preprocessing-when-using-embeddings
# 
# It is the continuation of this kernel :
# > https://www.kaggle.com/theoviel/improve-your-score-with-some-text-preprocessing
# 
# ### In this version, I check the improvement of my f1-score due to the preprocessing, with a basic model.
# 
# #### Any feedback is appreciated ! 

# In[ ]:


import pandas as pd
import numpy as np
import operator 
import re
import gc
import keras
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style('whitegrid')

# ## Loading data

# In[ ]:


train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
df = pd.concat([train ,test])

# In[ ]:


print("Number of texts: ", df.shape[0])

# ## Loading embeddings
# #### In this v2, I am using GloVe only, for memory usage purposes mostly.

# In[ ]:


def load_embed(file):
    def get_coefs(word,*arr): 
        return word, np.asarray(arr, dtype='float32')
    
    if file == '../input/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec':
        embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(file) if len(o)>100)
    else:
        embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(file, encoding='latin'))
        
    return embeddings_index

# In[ ]:


glove = '../input/embeddings/glove.840B.300d/glove.840B.300d.txt'

# In[ ]:


print("Extracting GloVe embedding")
embed_glove = load_embed(glove)

# ## Vocabulary and Coverage functions
# > Again, check Dieter's work if you haven't, those are his.

# In[ ]:


def build_vocab(texts):
    sentences = texts.apply(lambda x: x.split()).values
    vocab = {}
    for sentence in sentences:
        for word in sentence:
            try:
                vocab[word] += 1
            except KeyError:
                vocab[word] = 1
    return vocab

# In[ ]:


def check_coverage(vocab, embeddings_index):
    known_words = {}
    unknown_words = {}
    nb_known_words = 0
    nb_unknown_words = 0
    for word in vocab.keys():
        try:
            known_words[word] = embeddings_index[word]
            nb_known_words += vocab[word]
        except:
            unknown_words[word] = vocab[word]
            nb_unknown_words += vocab[word]
            pass

    print('Found embeddings for {:.3%} of vocab'.format(len(known_words) / len(vocab)))
    print('Found embeddings for  {:.3%} of all text'.format(nb_known_words / (nb_known_words + nb_unknown_words)))
    unknown_words = sorted(unknown_words.items(), key=operator.itemgetter(1))[::-1]

    return unknown_words

# ## Starting point

# In[ ]:


vocab = build_vocab(df['question_text'])

# In[ ]:


print("Glove : ")
oov_glove = check_coverage(vocab, embed_glove)

# #### Apply lowerization (necessary if using paragram)
# > Fasttext and GloVe understand capitalization, but you still win a bit.

# In[ ]:


df['lowered_question'] = df['question_text'].apply(lambda x: x.lower())

# #### If you apply lowerization, you lost a bit of informations on other embeddings :
# > Therer are words known that are known with upper letters and unknown without. Let us fix that :
# - word.lower() takes the embedding of word if word.lower() doesn't have an embedding

# In[ ]:


def add_lower(embedding, vocab):
    count = 0
    for word in vocab:
        if word in embedding and word.lower() not in embedding:  
            embedding[word.lower()] = embedding[word]
            count += 1
    print(f"Added {count} words to embedding")

# In[ ]:


print("Glove : ")
oov_glove = check_coverage(vocab, embed_glove)
add_lower(embed_glove, vocab)
oov_glove = check_coverage(vocab, embed_glove)

# ### What's wrong ?

# In[ ]:


oov_glove[:10]

# #### First faults appearing are : 
# - Contractions 
# - Words with punctuation in them
# 
# > Let us correct that.

# ## Contractions

# In[ ]:


contraction_mapping = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",  "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have","you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have" }

# In[ ]:


def known_contractions(embed):
    known = []
    for contract in contraction_mapping:
        if contract in embed:
            known.append(contract)
    return known

# In[ ]:


print("- Known Contractions -")
print("   Glove :")
print(known_contractions(embed_glove))

# #### Not a lot of contractions are known. (FastText knows none)
# > We use the map to replace them

# In[ ]:


def clean_contractions(text, mapping):
    specials = ["’", "‘", "´", "`"]
    for s in specials:
        text = text.replace(s, "'")
    text = ' '.join([mapping[t] if t in mapping else t for t in text.split(" ")])
    return text

# In[ ]:


df['treated_question'] = df['lowered_question'].apply(lambda x: clean_contractions(x, contraction_mapping))

# In[ ]:


vocab = build_vocab(df['treated_question'])
print("Glove : ")
oov_glove = check_coverage(vocab, embed_glove)

# ## Now, let us deal with special characters

# In[ ]:


punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'

# In[ ]:


def unknown_punct(embed, punct):
    unknown = ''
    for p in punct:
        if p not in embed:
            unknown += p
            unknown += ' '
    return unknown

# In[ ]:


print("Glove :")
print(unknown_punct(embed_glove, punct))

# #### Some characters are unknown.
# > We use a map to replace unknown characters with known ones.
# 
# > We make sure there are spaces between words and punctuation
# 

# In[ ]:


punct_mapping = {"‘": "'", "₹": "e", "´": "'", "°": "", "€": "e", "™": "tm", "√": " sqrt ", "×": "x", "²": "2", "—": "-", "–": "-", "’": "'", "_": "-", "`": "'", '“': '"', '”': '"', '“': '"', "£": "e", '∞': 'infinity', 'θ': 'theta', '÷': '/', 'α': 'alpha', '•': '.', 'à': 'a', '−': '-', 'β': 'beta', '∅': '', '³': '3', 'π': 'pi', }

# In[ ]:


def clean_special_chars(text, punct, mapping):
    for p in mapping:
        text = text.replace(p, mapping[p])
    
    for p in punct:
        text = text.replace(p, f' {p} ')
    
    specials = {'\u200b': ' ', '…': ' ... ', '\ufeff': '', 'करना': '', 'है': ''}  # Other special characters that I have to deal with in last
    for s in specials:
        text = text.replace(s, specials[s])
    
    return text

# In[ ]:


df['treated_question'] = df['treated_question'].apply(lambda x: clean_special_chars(x, punct, punct_mapping))

# In[ ]:


vocab = build_vocab(df['treated_question'])
print("Glove : ")
oov_glove = check_coverage(vocab, embed_glove)

# In[ ]:


oov_glove[:100]

# ### What's still missing ? 
# - Unknown words
# - Acronyms
# - Spelling mistakes
# 
# ## We can correct manually most frequent mispells
# 
# #### For example, here are some mistakes and their frequency
# - qoura : 85 times
# - mastrubation : 38 times
# - demonitisation : 30 times
# - …

# In[ ]:


mispell_dict = {'colour': 'color', 'centre': 'center', 'favourite': 'favorite', 'travelling': 'traveling', 'counselling': 'counseling', 'theatre': 'theater', 'cancelled': 'canceled', 'labour': 'labor', 'organisation': 'organization', 'wwii': 'world war 2', 'citicise': 'criticize', 'youtu ': 'youtube ', 'Qoura': 'Quora', 'sallary': 'salary', 'Whta': 'What', 'narcisist': 'narcissist', 'howdo': 'how do', 'whatare': 'what are', 'howcan': 'how can', 'howmuch': 'how much', 'howmany': 'how many', 'whydo': 'why do', 'doI': 'do I', 'theBest': 'the best', 'howdoes': 'how does', 'mastrubation': 'masturbation', 'mastrubate': 'masturbate', "mastrubating": 'masturbating', 'pennis': 'penis', 'Etherium': 'Ethereum', 'narcissit': 'narcissist', 'bigdata': 'big data', '2k17': '2017', '2k18': '2018', 'qouta': 'quota', 'exboyfriend': 'ex boyfriend', 'airhostess': 'air hostess', "whst": 'what', 'watsapp': 'whatsapp', 'demonitisation': 'demonetization', 'demonitization': 'demonetization', 'demonetisation': 'demonetization', 'pokémon': 'pokemon'}

# In[ ]:


def correct_spelling(x, dic):
    for word in dic.keys():
        x = x.replace(word, dic[word])
    return x

# In[ ]:


df['treated_question'] = df['treated_question'].apply(lambda x: correct_spelling(x, mispell_dict))

# In[ ]:


vocab = build_vocab(df['treated_question'])
print("Glove : ")
oov_glove = check_coverage(vocab, embed_glove)

# In[ ]:


del vocab, oov_glove, df, test
gc.collect()

# # Part 2 : Checking the improvement
# 
# ## Applying treatment
# 
# I only treat train data here, because I do not intend to submit my results.

# In[ ]:


# Lowering
train['treated_question'] = train['question_text'].apply(lambda x: x.lower())
# Contractions
train['treated_question'] = train['treated_question'].apply(lambda x: clean_contractions(x, contraction_mapping))
# Special characters
train['treated_question'] = train['treated_question'].apply(lambda x: clean_special_chars(x, punct, punct_mapping))
# Spelling mistakes
train['treated_question'] = train['treated_question'].apply(lambda x: correct_spelling(x, mispell_dict))

# ## Data for the network
# ### Texts
# #### Parameters
# I took the same for both models. len_voc can de reduced for the treated model.

# In[ ]:


len_voc = 95000
max_len = 60

# #### Untreated text
# We apply a standard tokenizer and padding.

# In[ ]:


from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# In[ ]:


def make_data(X):
    t = Tokenizer(num_words=len_voc)
    t.fit_on_texts(X)
    X = t.texts_to_sequences(X)
    X = pad_sequences(X, maxlen=max_len)
    return X, t.word_index

# In[ ]:


X, word_index = make_data(train['question_text'])

# #### Treated text
# Same thing, but with no filters.

# In[ ]:


def make_treated_data(X):
    t = Tokenizer(num_words=len_voc, filters='')
    t.fit_on_texts(X)
    X = t.texts_to_sequences(X)
    X = pad_sequences(X, maxlen=max_len)
    return X, t.word_index

# In[ ]:


X_treated, word_index_treated = make_data(train['treated_question'])

# #### Splitting

# In[ ]:


from sklearn.model_selection import train_test_split

# In[ ]:


y = train['target'].values

# In[ ]:


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=420)

# In[ ]:


X_t_train, X_t_val, _, _ = train_test_split(X_treated, y, test_size=0.1, random_state=420)

# In[ ]:


print(f"Training on {X_train.shape[0]} texts")

# ### Embeddings
# I use GloVe here, because I got better results with it than with others. But feel free to change that.

# In[ ]:


def make_embed_matrix(embeddings_index, word_index, len_voc):
    all_embs = np.stack(embeddings_index.values())
    emb_mean,emb_std = all_embs.mean(), all_embs.std()
    embed_size = all_embs.shape[1]
    word_index = word_index
    embedding_matrix = np.random.normal(emb_mean, emb_std, (len_voc, embed_size))
    
    for word, i in word_index.items():
        if i >= len_voc:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: 
            embedding_matrix[i] = embedding_vector
    
    return embedding_matrix

# In[ ]:


embedding = make_embed_matrix(embed_glove, word_index, len_voc)

del word_index
gc.collect()

# In[ ]:


embedding_treated = make_embed_matrix(embed_glove, word_index_treated, len_voc)
del word_index_treated
gc.collect()

#  Note that we have two embedding matrices, one for each pre-treatment.

# ## f1 metric for Keras

# In[ ]:


from keras import backend as K

# In[ ]:


def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


# ## Model

# In[ ]:


from keras.models import Model
from keras.layers import Dense, Embedding, Bidirectional, CuDNNGRU, GlobalAveragePooling1D, GlobalMaxPooling1D, concatenate, Input, Dropout
from keras.optimizers import Adam

# In[ ]:


def make_model(embedding_matrix, embed_size=300, loss='binary_crossentropy'):
    inp    = Input(shape=(max_len,))
    x      = Embedding(len_voc, embed_size, weights=[embedding_matrix], trainable=False)(inp)
    x      = Bidirectional(CuDNNGRU(128, return_sequences=True))(x)
    x      = Bidirectional(CuDNNGRU(64, return_sequences=True))(x)
    avg_pl = GlobalAveragePooling1D()(x)
    max_pl = GlobalMaxPooling1D()(x)
    concat = concatenate([avg_pl, max_pl])
    dense  = Dense(64, activation="relu")(concat)
    drop   = Dropout(0.1)(concat)
    output = Dense(1, activation="sigmoid")(concat)
    
    model  = Model(inputs=inp, outputs=output)
    model.compile(loss=loss, optimizer=Adam(lr=0.0001), metrics=['accuracy', f1])
    return model

# In[ ]:


model = make_model(embedding)

# In[ ]:


model_treated = make_model(embedding_treated)

# In[ ]:


model.summary()

# ###  Callbacks

# In[ ]:


from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

# In[ ]:


checkpoints = ModelCheckpoint('weights.hdf5', monitor="val_f1", mode="max", verbose=True, save_best_only=True)
reduce_lr = ReduceLROnPlateau(monitor='val_f1', factor=0.1, patience=2, verbose=1, min_lr=0.000001)

# In[ ]:


checkpoints_treated = ModelCheckpoint('treated_weights.hdf5', monitor="val_f1", mode="max", verbose=True, save_best_only=True)
reduce_lr_treated = ReduceLROnPlateau(monitor='val_f1', factor=0.1, patience=2, verbose=1, min_lr=0.000001)

# ### Fitting

# In[ ]:


epochs = 8
batch_size = 512

# In[ ]:


history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, 
                    validation_data=[X_val, y_val], callbacks=[checkpoints, reduce_lr])

# In[ ]:


plt.figure(figsize=(12,8))
plt.plot(history.history['acc'], label='Train Accuracy')
plt.plot(history.history['val_acc'], label='Test Accuracy')
plt.show()

# In[ ]:


history = model_treated.fit(X_t_train, y_train, batch_size=batch_size, epochs=epochs, 
                            validation_data=[X_t_val, y_val], callbacks=[checkpoints_treated, reduce_lr_treated])

# In[ ]:


plt.figure(figsize=(12,8))
plt.plot(history.history['acc'], label='Train Accuracy')
plt.plot(history.history['val_acc'], label='Test Accuracy')
plt.show()

# In[ ]:


model.load_weights('weights.hdf5')
model_treated.load_weights('treated_weights.hdf5')

# ## Results
# ### Predictions

# In[ ]:


pred_val = model.predict(X_val, batch_size=512, verbose=1)
pred_t_val = model_treated.predict(X_t_val, batch_size=512, verbose=1)

# #### F1 Scores

# In[ ]:


from sklearn.metrics import f1_score

# In[ ]:


def tweak_threshold(pred, truth):
    thresholds = []
    scores = []
    for thresh in np.arange(0.1, 0.501, 0.01):
        thresh = np.round(thresh, 2)
        thresholds.append(thresh)
        score = f1_score(truth, (pred>thresh).astype(int))
        scores.append(score)
    return np.max(scores), thresholds[np.argmax(scores)]

# In[ ]:


score_val, threshold_val = tweak_threshold(pred_val, y_val)

print(f"Scored {round(score_val, 4)} for threshold {threshold_val} with untreated texts on validation data")

# In[ ]:


score_t_val, threshold_t_val = tweak_threshold(pred_t_val, y_val)

print(f"Scored {round(score_t_val, 4)} for threshold {threshold_t_val} with treated texts on validation data")

# ### Conclusion : 
# 
# Our preprocessing helped improve the results.
# However, results with CuDNN layers are not reproductible, so the gain might vary a bit.
#  
#  ### *Thanks for reading ! *
