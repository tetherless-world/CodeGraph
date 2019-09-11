#!/usr/bin/env python
# coding: utf-8

# In this kernel, i will help you speed up your preprocessing at:
# 
# 1. replace text in `question_text`:
# 
#     replace the text in the dict, e.g.: x = x.replace("?", " ? ")
# 
# 2. load embeddings

# There is a quick report about the kernel. If you want to see the details, follow the codes :)

# | No | Category                 | Type   | DictLength   | Run time (total)  | Run time(s/w) |
# |-----|-----------------------|--------|----------------|---------------------| ----------------- |
# |1      |     replace text         |  slow  |          130      |                  43.4 s  |     0.3338          |
# |2     |     replace text         |   fast   |          130      |                    8.8 s  |     0.0677          |
# |3     |     replace text         |   slow  |          65       |                  23.9 s  |     0.3677          |
# |4     |     replace text         |   fast   |          65       |                  6.05 s  |     0.0931          |
# |5     |     load embedding |   slow  |          ---       |                   51.6 s  |             ---         |
# |6     |     load embedding |   fast   |          ---       |                    19.1 s  |             ---         |
# 

# First, let's import the packages and load the datas.

# In[ ]:


import numpy as np
import pandas as pd

# In[ ]:



# In[ ]:


train_org = pd.read_csv("../input/train.csv")
print("train shape:", train_org.shape)
train_org.head()

# ## 1. replace text in question_text

# In this section, i will use `clean_text_fast` to speed up. The original replace function is `clean_text_slow`.
# 
# The two functions are defined as blow:

# In[ ]:


def clean_text_slow(x, maxlen=None):
    puncts = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•',  '~', '@', '£', 
    '·', '_', '{', '}', '©', '^', '®', '`',  '<', '→', '°', '€', '™', '›',  '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…', 
    '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─', 
    '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾', 'Ã', '⋅', '‘', '∞', 
    '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√', ]
    x = x.lower()
    for punct in puncts[:maxlen]:
        x = x.replace(punct, f' {punct} ')
    return x

def clean_text_fast(x, maxlen=None):
    puncts = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•',  '~', '@', '£', 
    '·', '_', '{', '}', '©', '^', '®', '`',  '<', '→', '°', '€', '™', '›',  '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…', 
    '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─', 
    '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾', 'Ã', '⋅', '‘', '∞', 
    '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√', ]
    x = x.lower()
    for punct in puncts[:maxlen]:
        if punct in x:  # add this line
            x = x.replace(punct, f' {punct} ')
    return x

# the `puncts` contains 130 words or characters
# 
# the `fast` function only add one line: `if punct in x:`
# 
# Let's look at the run time first.

# In[ ]:


_ = train_org.question_text.apply(lambda x: clean_text_slow(x, maxlen=None))

# In[ ]:


_ = train_org.question_text.apply(lambda x: clean_text_fast(x, maxlen=None))

# This is because the `in` operation is more fast than `create a new str object` in python.
# 
# In the `slow` function, we create a new str in every iteration.
# 
# Next, let's use a half `puncts` length to calculate the run time.

# In[ ]:


_ = train_org.question_text.apply(lambda x: clean_text_slow(x, maxlen=65))

# In[ ]:


_ = train_org.question_text.apply(lambda x: clean_text_fast(x, maxlen=65))

# As we can see. The `slow` function runs **double** while the `puncts` become **double**
# 
# The `fast` only use extra **1/3** seconds while the `puncts` become  **double**.

# As our `puncts` grows longer and longer... I think i dont need to say anymore.

# > **do not create a `new string object` if you can use `in operation`** in python.

# ## 2. load embeddings.

# use hard code to speed up.

# In[ ]:


def load_glove_slow(word_index, max_words=200000, embed_size=300):
    EMBEDDING_FILE = '../input/embeddings/glove.840B.300d/glove.840B.300d.txt'
    def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE) if o.split(" ")[0] in word_index)

    all_embs = np.stack(embeddings_index.values())
    emb_mean,emb_std = all_embs.mean(), all_embs.std()
    embed_size = all_embs.shape[1]

    embedding_matrix = np.random.normal(emb_mean, emb_std, (max_words, embed_size))
    for word, i in word_index.items():
        if i >= max_words: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: embedding_matrix[i] = embedding_vector
            
    return embedding_matrix 

def load_glove_fast(word_index, max_words=200000, embed_size=300):
    EMBEDDING_FILE = '../input/embeddings/glove.840B.300d/glove.840B.300d.txt'
    emb_mean, emb_std = -0.005838499, 0.48782197

    embedding_matrix = np.random.normal(emb_mean, emb_std, (max_words, embed_size))
    with open(EMBEDDING_FILE, 'r', encoding="utf8") as f:
        for line in f:
            word, vec = line.split(' ', 1)
            if word not in word_index:
                continue
            i = word_index[word]
            if i >= max_words:
                continue
            embedding_vector = np.asarray(vec.split(' '), dtype='float32')[:300]
            if len(embedding_vector) == 300:
                embedding_matrix[i] = embedding_vector
    return embedding_matrix

# In the `load_glove_slow`, we calculate the `emb_mean` and  `emb_std` on every loading.
# 
# In the `load_glove_fast`, we write the `emb_mean` and  `emb_std` in the code, and avoid create `dict`

# Let create the word_index:

# In[ ]:


from keras.preprocessing.text import Tokenizer

# In[ ]:


tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_org.question_text.values)

# run the functions:

# In[ ]:


_ = load_glove_slow(tokenizer.word_index, len(tokenizer.word_index) + 1)

# In[ ]:


_ = load_glove_fast(tokenizer.word_index, len(tokenizer.word_index) + 1)

# the `fast` function use less **32.5** seconds the `slow`
# 
# **And in my code, the slow code runs totally 5mins while the `fast` runs only 40~50 seconds.**

# > Try use hard code if it is possible**?**

# In[ ]:



