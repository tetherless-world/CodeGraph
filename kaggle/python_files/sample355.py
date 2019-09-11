#!/usr/bin/env python
# coding: utf-8

# # Text preprocessing steps and universal pipeline
# 
# Before feeding any ML model some kind data, it has to be properly preprocessed. You must have heard the byword: `Garbage in, garbage out` (GIGO). Text is a specific kind of data and can't be directly fed to most ML models, so before feeding it to a model you have to somehow extract numerical features from it, in other word `vectorize`. Vectorization is not the topic of this tutorial, but the main thing you have to understand is that GIGO is also aplicable on vectorization too, you can extract qualitative features only from qualitatively preprocessed text.
# 
# Things we are going to discuss:
# 
# 1. Tokenization
# 1. Cleaning
# 1. Normalization
# 1. Lemmatization
# 1. Steaming
# 
# Finally, we'll create reusable pipeline, which you'll be able to use in your applications.

# In[ ]:


example_text = """
An explosion targeting a tourist bus has injured at least 16 people near the Grand Egyptian Museum, 
next to the pyramids in Giza, security sources say E.U.

South African tourists are among the injured. Most of those hurt suffered minor injuries, 
while three were treated in hospital, N.A.T.O. say.

http://localhost:8888/notebooks/Text%20preprocessing.ipynb

@nickname of twitter user and his email is email@gmail.com . 

A device went off close to the museum fence as the bus was passing on 16/02/2012.
"""

# # Tokenization
# 
# `Tokenization` - text preprocessing step, which assumes splitting text into `tokens`(words, senteces, etc.)
# 
# Seems like you can use somkeind of simple seperator to achieve it, but you don't have to forget that there are a lot of different situations, where separators just don't work. For example, `.` separator for tokenization into sentences will fail if you have abbreviations with dots. So you have to have more complex model to achieve good enough result. Commonly this problem is solved using `nltk` or `spacy` nlp libraries.

# In[ ]:


from nltk.tokenize import sent_tokenize, word_tokenize

nltk_words = word_tokenize(example_text)
display(f"Tokenized words: {nltk_words}")

# In[ ]:


import spacy
import en_core_web_sm

nlp = en_core_web_sm.load()

doc = nlp(example_text)
spacy_words = [token.text for token in doc]
display(f"Tokenized words: {spacy_words}")

# In[ ]:


display(f"In spacy but not in nltk: {set(spacy_words).difference(set(nltk_words))}")
display(f"In nltk but not in spacy: {set(nltk_words).difference(set(spacy_words))}")

# We see that `spacy` tokenized some weird staff like `\n`, `\n\n`, but was able to handle urls, emails and twitter-like mentions. Also we see that `nltk` tokenized abbreviations without the last `.`

# # Cleaning
# 
# `Cleaning` step assumes removing all undesirable content.

# ### Punctuation removal
# `Punctuation removal` might be a good step, when punctuation does not brings additional value for text vectorization. Punctuation removal is better to be done after tokenization step, doing it before might cause undesirable effects. Good choice for `TF-IDF`, `Count`, `Binary` vectorization.

# In[ ]:


import string

display(f"Punctuation symbols: {string.punctuation}")

# In[ ]:


text_with_punct = "@nickname of twitter user, and his email is email@gmail.com ."

# In[ ]:


text_without_punct = text_with_punct.translate(str.maketrans('', '', string.punctuation))
display(f"Text without punctuation: {text_without_punct}")

# Here you can see that important symbols for correct tokenizations were removed. Now email can't be properly detected. As you could mention from the `Tokenization` step, punctuation symbors were parsed as single tokens, so better way would be to tokenize first and then remove punctuation symbols. 

# In[ ]:


doc = nlp(text_with_punct)
tokens = [t.text for t in doc]
# python 
tokens_without_punct_python = [t for t in tokens if t not in string.punctuation]
display(f"Python based removal: {tokens_without_punct_python}")

tokens_without_punct_spacy = [t.text for t in doc if t.pos_ != 'PUNCT']
display(f"Spacy based removal: {tokens_without_punct_spacy}")

# Here you see that `python-based` removal worked even better than spacy, because spacy tagged `@nicname` as `PUNCT` part-of-speech.

# ### Stop words removal
# 
# `Stop words` usually refers to the most common words in a language, which usualy does not bring additional meaning. There is no single universal list of stop words used by all nlp tools, because this term has very fuzzy definition. Although practice has shown, that this step is much have, when preparing text for indexing, but might be tricky for text classification purposes.

# In[ ]:


text = "This movie is just not good enough"

# In[ ]:


spacy_stop_words = spacy.lang.en.stop_words.STOP_WORDS

display(f"Spacy stop words count: {len(spacy_stop_words)}")

# In[ ]:


text_without_stop_words = [t.text for t in nlp(text) if not t.is_stop]
display(f"Spacy text without stop words: {text_without_stop_words}")

# In[ ]:


import nltk

nltk_stop_words = nltk.corpus.stopwords.words('english')
display(f"nltk stop words count: {len(nltk_stop_words)}")

# In[ ]:


text_without_stop_words = [t for t in word_tokenize(text) if t not in nltk_stop_words]
display(f"nltk text without stop words: {text_without_stop_words}")

# Here you see that nltk and spacy has different vocabulary size, so the results of filtering are different. But the main thing I want to underline is that the word `not` was filtered, which in the most cases will be allright, but in the case when you want determine the polarity of this sentence `not` will bring the additional meaning.
# 
# For such cases you are able to set stop words you can ignore in spacy library. In the case of nltk you cat just remove or add custom words to `nltk_stop_words`, it is just a list.

# In[ ]:


import en_core_web_sm

nlp = en_core_web_sm.load()

customize_stop_words = [
    'not'
]

for w in customize_stop_words:
    nlp.vocab[w].is_stop = False

text_without_stop_words = [t.text for t in nlp(text) if not t.is_stop]
display(f"Spacy text without updated stop words: {text_without_stop_words}")

# # Normalization
# 
# Like any data text requires normalization. In case of text it is:
# 
# 1. Converting dates to text
# 2. Numbers to text
# 3. Currency/Percent signs to text
# 4. Expanding of abbreviations (content dependent) NLP - Natural Language Processing, Neuro-linguistic programming, Non-Linear programming
# 5. Spelling mistakes correction
# 
# To summarize, normalization is a convertion of any non-text information into textual equivalent.
# 
# For this purposes exists a great library - [normalize](https://github.com/EFord36/normalise). I'll show you usage of this library from its README. This library is based on `nltk` package, so it expects `nltk` word tokens.

# In[ ]:


from normalise import normalise

text = """
On the 13 Feb. 2007, Theresa May announced on MTV news that the rate of childhod obesity had 
risen from 7.3-9.6% in just 3 years , costing the N.A.T.O £20m
"""

user_abbr = {
    "N.A.T.O": "North Atlantic Treaty Organization"
}

normalized_tokens = normalise(word_tokenize(text), user_abbrevs=user_abbr, verbose=False)
display(f"Normalized text: {' '.join(normalized_tokens)}")

# The worst thing in this library is that for now you can't disable some modules, like abbreviation expanding, and int causes things like `MTV` -> `M T V`. But I have already added an appropriate issue on this repository, maybe it would be fixed in a while.

# # Lematization and Steaming
# 
# `Stemming` is the process of reducing inflection in words to their root forms such as mapping a group of words to the same stem even if the stem itself is not a valid word in the Language. 
# 
# `Lemmatization`, unlike Stemming, reduces the inflected words properly ensuring that the root word belongs to the language. In Lemmatization root word is called Lemma. A lemma (plural lemmas or lemmata) is the canonical form, dictionary form, or citation form of a set of words.

# In[ ]:


from nltk.stem import PorterStemmer
import numpy as np

text = ' '.join(normalized_tokens)
tokens = word_tokenize(text)

# In[ ]:


porter=PorterStemmer()
stem_words = np.vectorize(porter.stem)
stemed_text = ' '.join(stem_words(tokens))
display(f"Stemed text: {stemed_text}")

# In[ ]:


from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()
lemmatize_words = np.vectorize(wordnet_lemmatizer.lemmatize)
lemmatized_text = ' '.join(lemmatize_words(tokens))
display(f"nltk lemmatized text: {lemmatized_text}")

# In[ ]:


lemmas = [t.lemma_ for t in nlp(text)]
display(f"Spacy lemmatized text: {' '.join(lemmas)}")

# We see that `spacy` lemmatized much better than nltk, one of examples `risen` -> `rise`, only `spacy` handeled that.

# # Reusable pipeline
# 
# And now my favourite part! We are going to cretate reusable pipeline, which you could use on any of you projects.

# In[ ]:


import numpy as np
import multiprocessing as mp

import string
import spacy 
import en_core_web_sm
from nltk.tokenize import word_tokenize
from sklearn.base import TransformerMixin, BaseEstimator
from normalise import normalise

nlp = en_core_web_sm.load()


class TextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self,
                 variety="BrE",
                 user_abbrevs={},
                 n_jobs=1):
        """
        Text preprocessing transformer includes steps:
            1. Text normalization
            2. Punctuation removal
            3. Stop words removal
            4. Lemmatization
        
        variety - format of date (AmE - american type, BrE - british format) 
        user_abbrevs - dict of user abbreviations mappings (from normalise package)
        n_jobs - parallel jobs to run
        """
        self.variety = variety
        self.user_abbrevs = user_abbrevs
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        return self

    def transform(self, X, *_):
        X_copy = X.copy()

        partitions = 1
        cores = mp.cpu_count()
        if self.n_jobs <= -1:
            partitions = cores
        elif self.n_jobs <= 0:
            return X_copy.apply(self._preprocess_text)
        else:
            partitions = min(self.n_jobs, cores)

        data_split = np.array_split(X_copy, partitions)
        pool = mp.Pool(cores)
        data = pd.concat(pool.map(self._preprocess_part, data_split))
        pool.close()
        pool.join()

        return data

    def _preprocess_part(self, part):
        return part.apply(self._preprocess_text)

    def _preprocess_text(self, text):
        normalized_text = self._normalize(text)
        doc = nlp(normalized_text)
        removed_punct = self._remove_punct(doc)
        removed_stop_words = self._remove_stop_words(removed_punct)
        return self._lemmatize(removed_stop_words)

    def _normalize(self, text):
        # some issues in normalise package
        try:
            return ' '.join(normalise(text, variety=self.variety, user_abbrevs=self.user_abbrevs, verbose=False))
        except:
            return text

    def _remove_punct(self, doc):
        return [t for t in doc if t.text not in string.punctuation]

    def _remove_stop_words(self, doc):
        return [t for t in doc if not t.is_stop]

    def _lemmatize(self, doc):
        return ' '.join([t.lemma_ for t in doc])

# In[ ]:


import pandas as pd

df_bbc = pd.read_csv('../input/bbc-text.csv')

# In[ ]:


text = TextPreprocessor(n_jobs=-1).transform(df_bbc['text'])

# In[ ]:


print(f"Performance of transformer on {len(df_bbc)} texts and {mp.cpu_count()} processes")

# In[ ]:



