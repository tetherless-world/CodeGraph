#!/usr/bin/env python
# coding: utf-8

# <img src='http://s9.picofile.com/file/8351628176/nlp.png' width=600 height=600 >
# <div style="text-align:center">last update: <b>12/02/2019</b></div>
# 
# 
# >You are reading **10 Steps to Become a Data Scientist** and are now in the 8th step : 
# 
# 1. [Leren Python](https://www.kaggle.com/mjbahmani/the-data-scientist-s-toolbox-tutorial-1)
# 2. [Python Packages](https://www.kaggle.com/mjbahmani/the-data-scientist-s-toolbox-tutorial-2)
# 3. [Mathematics and Linear Algebra](https://www.kaggle.com/mjbahmani/linear-algebra-for-data-scientists)
# 4. <font color="red">You are in the 4th step</font>
# 5. [Big Data](https://www.kaggle.com/mjbahmani/a-data-science-framework-for-quora)
# 6. [Data visualization](https://www.kaggle.com/mjbahmani/top-5-data-visualization-libraries-tutorial)
# 7. [Data Cleaning](https://www.kaggle.com/mjbahmani/machine-learning-workflow-for-house-prices)
# 8. [Tutorial-on-ensemble-learning](https://www.kaggle.com/mjbahmani/tutorial-on-ensemble-learning)
# 9. [A Comprehensive ML  Workflow with Python](https://www.kaggle.com/mjbahmani/a-comprehensive-ml-workflow-with-python)
# 10. [Deep Learning](https://www.kaggle.com/mjbahmani/top-5-deep-learning-frameworks-tutorial)
# 
# 
# 
# ---------------------------------------------------------------------
# You can Fork and Run this kernel on Github:
# > ###### [ GitHub](https://github.com/mjbahmani/10-steps-to-become-a-data-scientist)
# 
# -------------------------------------------------------------------------------------------------------------
# 
#  **I hope you find this kernel helpful and some <font color="red"><b>UPVOTES</b></font> would be very much appreciated**
#  
#  -----------

#  <a id="top"></a> <br>
# ## Notebook  Content
# 1. [Introduction](#1)
#     1. [Import](#11)
#     1. [Version](#12)
#     1. [Setup](#13)
#     1. [Data set](#14)
#     1. [Gendered Pronoun Analysis](#15)
#         1. [Problem Feature](#151)
#         1. [Variables](#152)
# 1. [NLTK](#2)
#     1. [Tokenizing sentences](#21)
#     1. [NLTK and arrays](#22)
#     1. [NLTK stop words](#23)
#     1. [NLTK – stemming](#24)
#     1. [NLTK speech tagging](#25)
#     1. [Natural Language Processing – prediction](#26)
#         1. [nlp prediction example](#261)
#     1. [nlp prediction example](#27)
# 1. [spaCy](#3)
#     1. [Sentence detection](#31)
#     1. [Part Of Speech Tagging](#32)
#     1. [spaCy](#33)
#     1. [displaCy](#34)
# 1. [Gensim](#4)
# 1. [Comparison of Python NLP libraries by Activewizards](#5)
# 1. [References](#6)

# <a id="1"></a> <br>
# # 1-Introduction
# This Kernel is mostly for **beginners**, and of course, all **professionals** who think they need to review  their  knowledge.
# Also, we introduce and teach three known libraries ( NLTK+spaCy+Gensim) for text processing And we will introduce for each of them some examples based on [gendered-pronoun-resolution](https://www.kaggle.com/c/gendered-pronoun-resolution).

# <a id="11"></a> <br>
# ##   1-1 Import

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib
import warnings
import sklearn
import gensim
import scipy
import numpy
import json
import nltk
import sys
import csv
import os

# <a id="12"></a> <br>
# ## 1-2 Version

# In[ ]:


print('matplotlib: {}'.format(matplotlib.__version__))
print('scipy: {}'.format(scipy.__version__))
print('seaborn: {}'.format(sns.__version__))
print('pandas: {}'.format(pd.__version__))
print('numpy: {}'.format(np.__version__))
print('Python: {}'.format(sys.version))

# <a id="13"></a> <br>
# ## 1-3 Setup
# 
# A few tiny adjustments for better **code readability**

# In[ ]:


sns.set(style='white', context='notebook', palette='deep')
warnings.filterwarnings('ignore')
sns.set_style('white')

# <a id="14"></a> <br>
# ## 1-4 Data set

# In[ ]:


print(os.listdir("../input/"))

# In[ ]:


gendered_pronoun_df = pd.read_csv('../input/test_stage_1.tsv', delimiter='\t')

# In[ ]:


submission = pd.read_csv('../input/sample_submission_stage_1.csv')

# In[ ]:


gendered_pronoun_df.shape

# In[ ]:


submission.shape

# <a id="15"></a> <br>
# ## 1-5 Gendered Pronoun Data set Analysis
# <img src='https://storage.googleapis.com/kaggle-media/competitions/GoogleAI-GenderedPronoun/PronounResolution.png' width=600 height=600>
# **Pronoun resolution** is part of coreference resolution, the task of pairing an expression to its referring entity. This is an important task for natural language understanding, and the resolution of ambiguous pronouns is a longstanding challenge. for more information you can check this [link](https://www.kaggle.com/c/gendered-pronoun-resolution)
# <a id="151"></a> <br>
# ### 1-5-1 Problem Feature
# In this competition, you must identify the target of a pronoun within a text passage. The source text is taken from Wikipedia articles. You are provided with the pronoun and two candidate names to which the pronoun could refer. You must create an algorithm capable of deciding whether the pronoun refers to name A, name B, or neither.

# In[ ]:


gendered_pronoun_df.head()

# In[ ]:


gendered_pronoun_df.info()

# <a id="152"></a> <br>
# ### 1-5-2  Variables
# 
# 1. ID - Unique identifier for an example (Matches to Id in output file format)
# 1. Text - Text containing the ambiguous pronoun and two candidate names (about a paragraph in length)
# 1. Pronoun - The target pronoun (text)
# 1. Pronoun-offset The character offset of Pronoun in Text
# 1. A - The first name candidate (text)
# 1. A-offset - The character offset of name A in Text
# 1. B - The second name candidate
# 1. B-offset - The character offset of name B in Text
# 1. URL - The URL of the source Wikipedia page for the example

# In[ ]:


print(gendered_pronoun_df.Text.head())

# <a id="153"></a> <br>
# ### 1-5-3  Evaluation
# Submissions are evaluated using the multi-class logarithmic loss. Each pronoun has been labeled with whether it refers to A, B, or NEITHER. For each pronoun, you must submit a set of predicted probabilities (one for each class). The formula is :
# <img src='http://s8.picofile.com/file/8351608076/1.png'>

# In[ ]:


print("Shape of train set : ",gendered_pronoun_df.shape)

# In[ ]:


gendered_pronoun_df.columns

# ## Check Missing Data

# In[ ]:


def check_missing_data(df):
    flag=df.isna().sum().any()
    if flag==True:
        total = df.isnull().sum()
        percent = (df.isnull().sum())/(df.isnull().count()*100)
        output = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
        data_type = []
        # written by MJ Bahmani
        for col in df.columns:
            dtype = str(df[col].dtype)
            data_type.append(dtype)
        output['Types'] = data_type
        return(np.transpose(output))
    else:
        return(False)

# In[ ]:


check_missing_data(gendered_pronoun_df)

# <a id="154"></a> <br>
# ## 1-5-4 Some New Features
# In this section, I will extract a few new statistical features from the text field

# ### Number of words in the text

# In[ ]:


gendered_pronoun_df["num_words"] = gendered_pronoun_df["Text"].apply(lambda x: len(str(x).split()))

# In[ ]:


#MJ Bahmani
print('maximum of num_words in data_df',gendered_pronoun_df["num_words"].max())
print('min of num_words in data_df',gendered_pronoun_df["num_words"].min())

# ### Number of unique words in the text

# In[ ]:


gendered_pronoun_df["num_unique_words"] = gendered_pronoun_df["Text"].apply(lambda x: len(set(str(x).split())))
print('maximum of num_unique_words in train',gendered_pronoun_df["num_unique_words"].max())
print('mean of num_unique_words in data_df',gendered_pronoun_df["num_unique_words"].mean())

# ### Number of characters in the text

# In[ ]:


gendered_pronoun_df["num_chars"] = gendered_pronoun_df["Text"].apply(lambda x: len(str(x)))
print('maximum of num_chars in data_df',gendered_pronoun_df["num_chars"].max())

# ### Number of stopwords in the text

# In[ ]:


from nltk.corpus import stopwords
eng_stopwords = set(stopwords.words("english"))

# In[ ]:


gendered_pronoun_df["num_stopwords"] = gendered_pronoun_df["Text"].apply(lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords]))

print('maximum of num_stopwords in data_df',gendered_pronoun_df["num_stopwords"].max())

# ### Number of punctuations in the text
# 

# In[ ]:


import string
gendered_pronoun_df["num_punctuations"] =gendered_pronoun_df['Text'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]) )
print('maximum of num_punctuations in data_df',gendered_pronoun_df["num_punctuations"].max())

# ### Number of title case words in the text

# In[ ]:


gendered_pronoun_df["num_words_upper"] = gendered_pronoun_df["Text"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))
print('maximum of num_words_upper in data_df',gendered_pronoun_df["num_words_upper"].max())

# In[ ]:


print(gendered_pronoun_df.columns)
gendered_pronoun_df.head(1)

# In[ ]:


pronoun=gendered_pronoun_df["Pronoun"]

# In[ ]:


np.unique(pronoun)

# In[ ]:


## is suggested by  https://www.kaggle.com/aavella77
binary = {
    "He": 0,
    "he": 0,
    "She": 1,
    "she": 1,
    "His": 2,
    "his": 2,
    "Him": 3,
    "him": 3,
    "Her": 4,
    "her": 4
}
for index in range(len(gendered_pronoun_df)):
    key = gendered_pronoun_df.iloc[index]['Pronoun']
    gendered_pronoun_df.at[index, 'Pronoun_binary'] = binary[key]
gendered_pronoun_df.head(30)

# ## 1-5-4 Visualization

# ### 1-5-4-1 WordCloud

# In[ ]:


from wordcloud import WordCloud as wc
from nltk.corpus import stopwords
def generate_wordcloud(text): 
    wordcloud = wc(relative_scaling = 1.0,stopwords = eng_stopwords).generate(text)
    fig,ax = plt.subplots(1,1,figsize=(10,10))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    ax.margins(x=0, y=0)
    plt.show()

# In[ ]:


from nltk.corpus import stopwords
eng_stopwords = set(stopwords.words("english"))

# In[ ]:


text =" ".join(gendered_pronoun_df.Text)
generate_wordcloud(text)

# In[ ]:


gendered_pronoun_df.hist();

# In[ ]:


pd.plotting.scatter_matrix(gendered_pronoun_df,figsize=(10,10))
plt.figure();

# In[ ]:


sns.jointplot(x='Pronoun-offset',y='A-offset' ,data=gendered_pronoun_df, kind='reg')

# In[ ]:


sns.swarmplot(x='Pronoun-offset',y='B-offset',data=gendered_pronoun_df);

# In[ ]:


sns.distplot(gendered_pronoun_df["Pronoun-offset"])

# In[ ]:


sns.violinplot(data=gendered_pronoun_df,x="Pronoun_binary", y="num_words")

# <a id="14"></a> <br>
# # Top 3 NLP Libraries Tutorial
# 1. NLTK
# 1. spaCy
# 1. Gensim

# <a id="2"></a> <br>
# # 2- NLTK
# The Natural Language Toolkit (NLTK) is one of the leading platforms for working with human language data and Python, the module NLTK is used for natural language processing. NLTK is literally an acronym for Natural Language Toolkit. with it you can tokenizing words and sentences.[https://www.nltk.org/](https://www.nltk.org/)
# <br>
# NLTK is a library of Python that can mine (scrap and upload data) and analyse very large amounts of textual data using computational methods.this tutorial is based on **this great course** [**https://pythonspot.com/category/nltk/**](https://pythonspot.com/category/nltk/)
# <img src='https://arts.unimelb.edu.au/__data/assets/image/0005/2735348/nltk.jpg' width=400 height=400>

# If you are using Windows or Linux or Mac, you can install NLTK using pip:
# >**$ pip install nltk**
# 
# You can use NLTK on Python 2.7, 3.4, and 3.6.

# To get started, we first select a few sentences from the data set.

# In[ ]:


from nltk.tokenize import sent_tokenize, word_tokenize

# In[ ]:


gendered_pronoun_df.Text[0]

# In[ ]:


our_text=gendered_pronoun_df.Text[0]

# In[ ]:


print(word_tokenize(our_text))

# <a id="21"></a> <br>
# ## 2-1 Tokenizing sentences
# **What is Tokenizer?**
# Tokenizing raw text data is an important pre-processing step for many NLP methods. As explained on wikipedia, tokenization is “the process of breaking a stream of text up into words, phrases, symbols, or other meaningful elements called tokens.” In the context of actually working through an NLP analysis, this usually translates to converting a string like "My favorite color is blue" to a list or array like ["My", "favorite", "color", "is", "blue"].[**http://tint.fbk.eu/tokenization.html**](http://tint.fbk.eu/tokenization.html)

# In[ ]:


from nltk.tokenize import sent_tokenize, word_tokenize
print(sent_tokenize(our_text))

# <a id="22"></a> <br>
# ## 2-2 NLTK and Arrays
# If you wish to you can store the words and sentences in arrays.

# In[ ]:


from nltk.tokenize import sent_tokenize, word_tokenize
 
phrases = sent_tokenize(our_text)
words = word_tokenize(our_text)
print(phrases)

# In[ ]:


print(words)

# In[ ]:


type(words)

# <a id="23"></a> <br>
# ## 2-3 NLTK Stop Words
# Natural language processing (nlp) is a research field that presents many challenges such as natural language understanding.
# Text may contain stop words like ‘the’, ‘is’, ‘are’. Stop words can be filtered from the text to be processed. There is no universal list of stop words in nlp research, however the nltk module contains a list of stop words.
# 
# In this article you will learn how to remove stop words with the nltk module.[https://pythonspot.com/nltk-stop-words/](https://pythonspot.com/nltk-stop-words/)

# In[ ]:


from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

# In[ ]:


 

stopWords = set(stopwords.words('english'))
words = word_tokenize(our_text)
wordsFiltered = []
 
for w in words:
    if w not in stopWords:
        wordsFiltered.append(w)
 
print(wordsFiltered)

# A module has been imported:
# 
# 

# In[ ]:


from nltk.corpus import stopwords


# We get a set of English stop words using the line:
# 
# 

# In[ ]:


stopWords = set(stopwords.words('english'))


# The returned list stopWords contains 153 stop words on my computer.
# You can view the length or contents of this array with the lines:

# In[ ]:


print(len(stopWords))
print(stopWords)

# We create a new list called wordsFiltered which contains all words which are not stop words.
# To create it we iterate over the list of words and only add it if its not in the stopWords list.

# In[ ]:


for w in words:
    if w not in stopWords:
        wordsFiltered.append(w)

# <a id="24"></a> <br>
# ## 2-4 NLTK – Stemming
# Stemming is the process of producing morphological variants of a root/base word. Stemming programs are commonly referred to as stemming algorithms or stemmers. A stemming algorithm reduces the words “chocolates”, “chocolatey”, “choco” to the root word, “chocolate” and “retrieval”, “retrieved”, “retrieves” reduce to the stem “retrieve”.[https://www.geeksforgeeks.org/python-stemming-words-with-nltk/](https://www.geeksforgeeks.org/python-stemming-words-with-nltk/).
# <img src='https://pythonspot-9329.kxcdn.com/wp-content/uploads/2016/08/word-stem.png.webp'>
# [Image-credit](https://pythonspot.com/nltk-stemming/)
# 
# Start by defining some words:

# In[ ]:


our_text=gendered_pronoun_df.Text[0]


# In[ ]:


from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize

# And stem the words in the list using:

# In[ ]:


from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize


ps = PorterStemmer()
 
for word in word_tokenize(our_text):
    print(ps.stem(word))

# <a id="25"></a> <br>
# ## 2-5 NLTK speech tagging
# The **module NLTK** can automatically **tag speech**.
# Given a sentence or paragraph, It can label words such as verbs, nouns and so on.
# 
# The example below automatically tags words with a corresponding class.[https://www.nltk.org/book/ch05.html](https://www.nltk.org/book/ch05.html)

# In[ ]:


import nltk
from nltk.tokenize import PunktSentenceTokenizer
 

sentences = nltk.sent_tokenize(our_text)   
for sent in sentences:
    print(nltk.pos_tag(nltk.word_tokenize(sent)))

# We can filter this data based on the type of word:

# In[ ]:


import nltk
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer
 

sentences = nltk.sent_tokenize(our_text)   
 
data = []
for sent in sentences:
    data = data + nltk.pos_tag(nltk.word_tokenize(sent))
 
for word in data: 
    if 'NNP' in word[1]: 
        print(word)

# <a id="26"></a> <br>
# ## 2-6 Natural Language Processing – prediction
# We can use natural language processing to make predictions. Example: Given a product review, a computer can predict if its positive or negative based on the text. In this article you will learn how to make a prediction program based on natural language processing.

# <a id="261"></a> <br>
# ## 2-6-1 NLP Prediction Example Based on pythonspot
# Given a name, the classifier will predict if it’s a male or female.
# 
# To create our analysis program, we have several steps:
# 
# 1. Data preparation
# 1. Feature extraction
# 1. Training
# 1. Prediction
# 1. Data preparation
# The first step is to prepare data. We use the names set included with nltk.[https://pythonspot.com/natural-language-processing-prediction/](https://pythonspot.com/natural-language-processing-prediction/)

# In[ ]:


#https://pythonspot.com/natural-language-processing-prediction/
from nltk.corpus import names
 
# Load data and training 
names = ([(name, 'male') for name in names.words('male.txt')] + [(name, 'female') for name in names.words('female.txt')])

# This dataset is simply a collection of tuples. To give you an idea of what the dataset looks like:

# In[ ]:


[(u'Aaron', 'male'), (u'Abbey', 'male'), (u'Abbie', 'male')]
[(u'Zorana', 'female'), (u'Zorina', 'female'), (u'Zorine', 'female')]

# You can define your own set of tuples if you wish, its simply a list containing many tuples.
# 
# Feature extraction
# Based on the dataset, we prepare our feature. The feature we will use is the last letter of a name:
# We define a featureset using:

# featuresets = [(gender_features(n), g) for (n,g) in names]
# and the features (last letters) are extracted using:

# In[ ]:


def gender_features(word): 
    return {'last_letter': word[-1]}

# Training and prediction
# We train and predict using:

# In[ ]:


#Based on https://pythonspot.com/category/nltk/
import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import names
 
def gender_features(word): 
    return {'last_letter': word[-1]} 
 
# Load data and training 
names = ([(name, 'male') for name in names.words('male.txt')] + 
	 [(name, 'female') for name in names.words('female.txt')])
 
featuresets = [(gender_features(n), g) for (n,g) in names] 
train_set = featuresets
classifier = nltk.NaiveBayesClassifier.train(train_set) 
 
# Predict
print(classifier.classify(gender_features('Frank')))

# If you want to give the name during runtime, change the last line to:

# In[ ]:


# Predict, you can change name
name = 'Sarah'
print(classifier.classify(gender_features(name)))

# <a id="27"></a> <br>
# ## 2-7 Python Sentiment Analysis
# In Natural Language Processing there is a concept known as **Sentiment Analysis**. in this section we use this great [**course**](https://pythonspot.com/category/nltk/) to explain Sentiment Analysis
# 
# <img src='https://s3.amazonaws.com/com.twilio.prod.twilio-docs/images/SentimentAnalysis.width-800.png'>
# [image-credit](https://www.twilio.com/docs/glossary/what-is-sentiment-analysis)
# 
# 1. Given a movie review or a tweet, it can be automatically classified in categories.
# 1. These categories can be user defined (positive, negative) or whichever classes you want.
# 1. Classification is done using several steps: training and prediction.
# 1. The training phase needs to have training data, this is example data in which we define examples. 
# 1. The classifier will use the training data to make predictions.

# We start by defining 3 classes: positive, negative and neutral.
# Each of these is defined by a vocabulary:

# In[ ]:


positive_vocab = [ 'awesome', 'outstanding', 'fantastic', 'terrific', 'good', 'nice', 'great', ':)' ]
negative_vocab = [ 'bad', 'terrible','useless', 'hate', ':(' ]
neutral_vocab = [ 'movie','the','sound','was','is','actors','did','know','words','not' ]

# Every word is converted into a feature using a simplified bag of words model:

# In[ ]:


def word_feats(words):
    return dict([(word, True) for word in words])
 
positive_features = [(word_feats(pos), 'pos') for pos in positive_vocab]
negative_features = [(word_feats(neg), 'neg') for neg in negative_vocab]
neutral_features = [(word_feats(neu), 'neu') for neu in neutral_vocab]

# Our training set is then the sum of these three feature sets:

# In[ ]:


train_set = negative_features + positive_features + neutral_features

# We train the classifier:

# classifier = NaiveBayesClassifier.train(train_set)

# This example classifies sentences according to the training set.

# In[ ]:


#Based on https://pythonspot.com/category/nltk/
import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import names
 
def word_feats(words):
    return dict([(word, True) for word in words])
 
positive_vocab = [ 'awesome', 'outstanding', 'fantastic', 'terrific', 'good', 'nice', 'great', ':)' ]
negative_vocab = [ 'bad', 'terrible','useless', 'hate', ':(' ]
neutral_vocab = [ 'movie','the','sound','was','is','actors','did','know','words','not' ]
 
positive_features = [(word_feats(pos), 'pos') for pos in positive_vocab]
negative_features = [(word_feats(neg), 'neg') for neg in negative_vocab]
neutral_features = [(word_feats(neu), 'neu') for neu in neutral_vocab]
 
train_set = negative_features + positive_features + neutral_features
 
classifier = NaiveBayesClassifier.train(train_set) 
 
# Predict
neg = 0
pos = 0
##sentence = "Awesome movie, I liked it"
our_text = our_text.lower()
words = our_text.split(' ')
for word in words:
    classResult = classifier.classify( word_feats(word))
    if classResult == 'neg':
        neg = neg + 1
    if classResult == 'pos':
        pos = pos + 1
 
print('Positive: ' + str(float(pos)/len(words)))
print('Negative: ' + str(float(neg)/len(words)))

# <a id="3"></a> <br>
# # 3- spaCy
# <img src='https://upload.wikimedia.org/wikipedia/commons/thumb/8/88/SpaCy_logo.svg/1920px-SpaCy_logo.svg.png' width=400 height=400>
# spaCy is an Industrial-Strength Natural Language Processing in python. [**spacy**](https://spacy.io/)

# In[ ]:


import spacy

# In[ ]:


nlp = spacy.load('en')
doc = nlp(our_text)
i=0
for token in doc:
    i=i+1;
    if i<20:
        print('"' + token.text + '"')

# ## 3-1 Sentence detection
# 

# In[ ]:


nlp = spacy.load('en')
doc=nlp(our_text)
i=0
for sent in doc.sents:
    i=i+1
    print(i,' - ',sent)

# ## 3-2 Part Of Speech Tagging

# In[ ]:


doc = nlp( our_text)
print([(token.text, token.tag_) for token in doc])

# ## 3-3 Named Entity Recognition
# 

# In[ ]:


doc = nlp(our_text)
for ent in doc.ents:
    print(ent.text, ent.label_)

# ## 3-4 displaCy 

# In[ ]:


from spacy import displacy
 
doc = nlp(our_text )
displacy.render(doc, style='ent', jupyter=True)

# visualizing the dependency tree!

# In[ ]:


from spacy import displacy
 
doc = nlp(our_text)
displacy.render(doc, style='dep', jupyter=True, options={'distance': 90})

# <a id="4"></a> <br>
# # 4- Gensim
# Gensim is a Python library for topic modelling, document indexing and similarity retrieval with large corpora. Target audience is the natural language processing (NLP) and information retrieval (IR) community.[https://github.com/chirayukong/gensim](https://github.com/chirayukong/gensim)
# 1. Gensim is a FREE Python library
# 1. Scalable statistical semantics
# 1. Analyze plain-text documents for semantic structure
# 1. Retrieve semantically similar documents. [https://radimrehurek.com/gensim/](https://radimrehurek.com/gensim/)

# In[ ]:


import gensim
from gensim import corpora
from pprint import pprint
# How to create a dictionary from a list of sentences?
documents = [" Zoe Telford  played the police officer girlfriend of Simon, Maggie.",
             "Dumped by Simon in the final episode of series 1, after he slept with Jenny, and is not seen again.",
             "Phoebe Thomas played Cheryl Cassidy, Paulines friend and also a year 11 pupil in Simons class.", 
             "Dumped her boyfriend following Simons advice after he wouldnt ",
             "have sex with her but later realised this was due to him catching crabs off her friend Pauline."]

documents_2 = ["One source says the report will likely conclude that", 
                "the operation was carried out without clearance and", 
                "transparency and that those involved will be held", 
                "responsible. One of the sources acknowledged that the", 
                "report is still being prepared and cautioned that", 
                "things could change."]

# ### Tokenize(split) the sentences into words

# In[ ]:



texts = [[text for text in doc.split()] for doc in documents]

# Create dictionary
dictionary = corpora.Dictionary(texts)

# Get information about the dictionary
print(dictionary)

# ### Show the word to id map

# In[ ]:


# 
print(dictionary.token2id)

# <a id="5"></a> <br>
# ## 5- Comparison of Python NLP libraries by Activewizards

# <img src='https://activewizards.com/content/blog/Comparison_of_Python_NLP_libraries/nlp-librares-python-prs-and-cons01.png'>

# >###### you may  be interested have a look at it: [**10-steps-to-become-a-data-scientist**](https://github.com/mjbahmani/10-steps-to-become-a-data-scientist)
# 
# 
# ---------------------------------------------------------------------
# You can Fork and Run this kernel on Github:
# > ###### [ GitHub](https://github.com/mjbahmani/10-steps-to-become-a-data-scientist)
# 
# -------------------------------------------------------------------------------------------------------------
# 
#  **I hope you find this kernel helpful and some <font color="red"><b>UPVOTES</b></font> would be very much appreciated**
#  
#  -----------

# <a id="6"></a> <br>
# # 6- References & Credits
# 1. [https://www.coursera.org/specializations/data-science-python](https://www.coursera.org/specializations/data-science-python)
# 1. [https://github.com/chirayukong/gensim](https://github.com/chirayukong/gensim)
# 1. [https://pythonspot.com/category/nltk/](https://pythonspot.com/category/nltk/)
# 1. [sunscrapers](https://sunscrapers.com/blog/6-best-python-natural-language-processing-nlp-libraries/)
# 1. [spacy](https://spacy.io/)
# 1. [gensim](https://pypi.org/project/gensim/)
# 1. [nlpforhackers](https://nlpforhackers.io/complete-guide-to-spacy/)
# 1. [a-sentiment-analysis-approach-to-predicting-stock-returns](https://medium.com/@tomyuz/a-sentiment-analysis-approach-to-predicting-stock-returns-d5ca8b75a42)
# 1. [machinelearningplus](https://www.machinelearningplus.com/nlp/gensim-tutorial/)
# 1. [https://towardsdatascience.com/topic-modelling-in-python-with-nltk-and-gensim-4ef03213cd21](https://towardsdatascience.com/topic-modelling-in-python-with-nltk-and-gensim-4ef03213cd21)
# ###### [Go to top](#top)

# Go to first step: [**Course Home Page**](https://www.kaggle.com/mjbahmani/10-steps-to-become-a-data-scientist)
# 
# Go to next step : [**Mathematics and Linear Algebra**](https://www.kaggle.com/mjbahmani/linear-algebra-for-data-scientists)
