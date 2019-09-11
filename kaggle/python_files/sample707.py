#!/usr/bin/env python
# coding: utf-8

# In this kernel I want to illustrate how I do come up with meaningful preprocessing when building deep learning NLP models. 
# 
# I start with two golden rules:
# 
# 1.  **Don't use standard preprocessing steps like stemming or stopword removal when you have pre-trained embeddings** 
# 
# Some of you might used standard preprocessing steps when doing word count based feature extraction (e.g. TFIDF) such as removing stopwords, stemming etc. 
# The reason is simple: You loose valuable information, which would help your NN to figure things out.  
# 
# 2. **Get your vocabulary as close to the embeddings as possible**
# 
# I will focus in this notebook, how to achieve that. For an example I take the GoogleNews pretrained embeddings, there is no deeper reason for this choice.

# 

# We start with a neat little trick that enables us to see a progressbar when applying functions to a pandas Dataframe

# In[ ]:


import pandas as pd
from tqdm import tqdm
tqdm.pandas()


# Lets load our data

# In[ ]:


train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
print("Train shape : ",train.shape)
print("Test shape : ",test.shape)

# I will use the following function to track our training vocabulary, which goes through all our text and counts the occurance of the contained words. 

# In[ ]:


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

# So lets populate the vocabulary and display the first 5 elements and their count. Note that now we can use progess_apply to see progress bar

# In[ ]:


sentences = train["question_text"].progress_apply(lambda x: x.split()).values
vocab = build_vocab(sentences)
print({k: vocab[k] for k in list(vocab)[:5]})

# Next we import the embeddings we want to use in our model later. For illustration I use GoogleNews here.

# In[ ]:


from gensim.models import KeyedVectors

news_path = '../input/embeddings/GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin'
embeddings_index = KeyedVectors.load_word2vec_format(news_path, binary=True)

# Next I define a function that checks the intersection between our vocabulary and the embeddings. It will output a list of out of vocabulary (oov) words that we can use to improve our preprocessing

# In[ ]:


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

# In[ ]:


oov = check_coverage(vocab,embeddings_index)

# Ouch only 24% of our vocabulary will have embeddings, making 21% of our data more or less useless. So lets have a look and start improving. For this we can easily have a look at the top oov words.

# In[ ]:


oov[:10]

# On first place there is "to". Why? Simply because "to" was removed when the GoogleNews Embeddings were trained. We will fix this later, for now we take care about the splitting of punctuation as this also seems to be a Problem. But what do we do with the punctuation then - Do we want to delete or consider as a token? I would say: It depends. If the token has an embedding, keep it, if it doesn't we don't need it anymore. So lets check:

# In[ ]:


'?' in embeddings_index

# In[ ]:


'&' in embeddings_index

# Interesting. While "&" is in the Google News Embeddings, "?" is not. So we basically define a function that splits off "&" and removes other punctuation.

# In[ ]:


def clean_text(x):

    x = str(x)
    for punct in "/-'":
        x = x.replace(punct, ' ')
    for punct in '&':
        x = x.replace(punct, f' {punct} ')
    for punct in '?!.,"#$%\'()*+-/:;<=>@[\\]^_`{|}~' + '“”’':
        x = x.replace(punct, '')
    return x

# In[ ]:


train["question_text"] = train["question_text"].progress_apply(lambda x: clean_text(x))
sentences = train["question_text"].apply(lambda x: x.split())
vocab = build_vocab(sentences)

# In[ ]:


oov = check_coverage(vocab,embeddings_index)


# In[ ]:




# Nice! We were able to increase our embeddings ratio from 24% to 57% by just handling punctiation. Ok lets check on thos oov words.

# In[ ]:


oov[:10]

# Hmm seems like numbers also are a problem. Lets check the top 10 embeddings to get a clue.

# In[ ]:


for i in range(10):
    print(embeddings_index.index2entity[i])

# hmm why is "##" in there? Simply because as a reprocessing all numbers bigger tha 9 have been replaced by hashs. I.e. 15 becomes ## while 123 becomes ### or 15.80€ becomes ##.##€. So lets mimic this preprocessing step to further improve our embeddings coverage

# In[ ]:


import re

def clean_numbers(x):

    x = re.sub('[0-9]{5,}', '#####', x)
    x = re.sub('[0-9]{4}', '####', x)
    x = re.sub('[0-9]{3}', '###', x)
    x = re.sub('[0-9]{2}', '##', x)
    return x

# In[ ]:


train["question_text"] = train["question_text"].progress_apply(lambda x: clean_numbers(x))
sentences = train["question_text"].progress_apply(lambda x: x.split())
vocab = build_vocab(sentences)

# In[ ]:


oov = check_coverage(vocab,embeddings_index)

# Nice! Another 3% increase. Now as much as with handling the puntuation, but every bit helps. Lets check the oov words

# In[ ]:


oov[:20]

# Ok now we  take care of common misspellings when using american/ british vocab and replacing a few "modern" words with "social media" for this task I use a multi regex script I found some time ago on stack overflow. Additionally we will simply remove the words "a","to","and" and "of" since those have obviously been downsampled when training the GoogleNews Embeddings. 
# 

# In[ ]:


def _get_mispell(mispell_dict):
    mispell_re = re.compile('(%s)' % '|'.join(mispell_dict.keys()))
    return mispell_dict, mispell_re


mispell_dict = {'colour':'color',
                'centre':'center',
                'didnt':'did not',
                'doesnt':'does not',
                'isnt':'is not',
                'shouldnt':'should not',
                'favourite':'favorite',
                'travelling':'traveling',
                'counselling':'counseling',
                'theatre':'theater',
                'cancelled':'canceled',
                'labour':'labor',
                'organisation':'organization',
                'wwii':'world war 2',
                'citicise':'criticize',
                'instagram': 'social medium',
                'whatsapp': 'social medium',
                'snapchat': 'social medium'

                }
mispellings, mispellings_re = _get_mispell(mispell_dict)

def replace_typical_misspell(text):
    def replace(match):
        return mispellings[match.group(0)]

    return mispellings_re.sub(replace, text)

# In[ ]:


train["question_text"] = train["question_text"].progress_apply(lambda x: replace_typical_misspell(x))
sentences = train["question_text"].progress_apply(lambda x: x.split())
to_remove = ['a','to','of','and']
sentences = [[word for word in sentence if not word in to_remove] for sentence in tqdm(sentences)]
vocab = build_vocab(sentences)

# In[ ]:


oov = check_coverage(vocab,embeddings_index)

# We see that although we improved on the amount of embeddings found for all our text from 89% to 99%. Lets check the oov words again 

# In[ ]:


oov[:20]

# Looks good. No obvious oov words there we could quickly fix.
# Thank you for reading and happy kaggling

# In[ ]:



