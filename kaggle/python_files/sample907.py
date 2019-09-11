#!/usr/bin/env python
# coding: utf-8

# # About the dataset
# An existential problem for any major website today is how to handle toxic and divisive content. Quora wants to tackle this problem head-on to keep their platform a place where users can feel safe sharing their knowledge with the world.
# 
# Quora is a platform that empowers people to learn from each other. On Quora, people can ask questions and connect with others who contribute unique insights and quality answers. A key challenge is to weed out insincere questions -- those founded upon false premises, or that intend to make a statement rather than look for helpful answers.
# 
# In this competition, Kagglers will develop models that identify and flag insincere questions. To date, Quora has employed both machine learning and manual review to address this problem. With your help, they can develop more scalable methods to detect toxic and misleading content.
# 
# Here's your chance to combat online trolls at scale. Help Quora uphold their policy of “Be Nice, Be Respectful” and continue to be a place for sharing and growing the world’s knowledge.

# In[ ]:


# Usual Imports
import numpy as np
import pandas as pd
from tqdm import tqdm
import string
import random
import operator
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import NMF, LatentDirichletAllocation, TruncatedSVD
from statistics import *
from sklearn.feature_extraction.text import CountVectorizer
import concurrent.futures
import time
import pyLDAvis.sklearn
from pylab import bone, pcolor, colorbar, plot, show, rcParams, savefig
import textstat
import warnings
import nltk
warnings.filterwarnings('ignore')

import os
print(os.listdir("../input"))

# Plotly based imports for visualization
from plotly import tools
import plotly.plotly as py
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.figure_factory as ff

# spaCy based imports
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English

import warnings
warnings.filterwarnings("ignore")

# In[ ]:


quora_train = pd.read_csv("../input/train.csv")
quora_train.head()

# In[ ]:


# SpaCy Parser for questions
punctuations = string.punctuation
stopwords = list(STOP_WORDS)

parser = English()
def spacy_tokenizer(sentence):
    mytokens = parser(sentence)
    mytokens = [ word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens ]
    mytokens = [ word for word in mytokens if word not in stopwords and word not in punctuations ]
    mytokens = " ".join([i for i in mytokens])
    return mytokens

# In[ ]:


tqdm.pandas()
sincere_questions = quora_train["question_text"][quora_train["target"] == 0].progress_apply(spacy_tokenizer)
insincere_questions = quora_train["question_text"][quora_train["target"] == 1].progress_apply(spacy_tokenizer)

# # Statistics for the given data
# We will use the ```textstat``` package by kaggler Shivam Bansal(@shivamb) for this purpose. 

# In[ ]:


# One function for all plots
def plot_readability(a,b,title,bins=0.1,colors=['#3A4750', '#F64E8B']):
    trace1 = ff.create_distplot([a,b], ["Sincere questions","Insincere questions"], bin_size=bins, colors=colors, show_rug=False)
    trace1['layout'].update(title=title)
    iplot(trace1, filename='Distplot')
    table_data= [["Statistical Measures","Sincere questions","Insincere questions"],
                ["Mean",mean(a),mean(b)],
                ["Standard Deviation",pstdev(a),pstdev(b)],
                ["Variance",pvariance(a),pvariance(b)],
                ["Median",median(a),median(b)],
                ["Maximum value",max(a),max(b)],
                ["Minimum value",min(a),min(b)]]
    trace2 = ff.create_table(table_data)
    iplot(trace2, filename='Table')

# ## 1. Syllable Analysis

# In[ ]:


syllable_sincere = np.array(quora_train["question_text"][quora_train["target"] == 0].progress_apply(textstat.syllable_count))
syllable_insincere = np.array(quora_train["question_text"][quora_train["target"] == 1].progress_apply(textstat.syllable_count))
plot_readability(syllable_sincere,syllable_insincere,"Syllable Analysis",5)
    

# ## 2. Lexicon Analysis

# In[ ]:


lexicon_sincere = np.array(quora_train["question_text"][quora_train["target"] == 0].progress_apply(textstat.lexicon_count))
lexicon_insincere = np.array(quora_train["question_text"][quora_train["target"] == 1].progress_apply(textstat.lexicon_count))
plot_readability(lexicon_sincere,lexicon_insincere,"Lexicon Analysis",4,['#C65D17','#DDB967'])

# ## 3. Question length

# In[ ]:


length_sincere = np.array(quora_train["question_text"][quora_train["target"] == 0].progress_apply(len))
length_insincere = np.array(quora_train["question_text"][quora_train["target"] == 1].progress_apply(len))
plot_readability(length_sincere,length_insincere,"Question Length",40,['#C65D17','#DDB967'])

# ## 4. Average syllables per word in a question 

# In[ ]:


spw_sincere = np.array(quora_train["question_text"][quora_train["target"] == 0].progress_apply(textstat.avg_syllables_per_word))
spw_insincere = np.array(quora_train["question_text"][quora_train["target"] == 1].progress_apply(textstat.avg_syllables_per_word))
plot_readability(spw_sincere,spw_insincere,"Average syllables per word",0.2,['#8D99AE','#EF233C'])

# ## 5. Average letters per word in a question 

# In[ ]:


lpw_sincere = np.array(quora_train["question_text"][quora_train["target"] == 0].progress_apply(textstat.avg_letter_per_word))
lpw_insincere = np.array(quora_train["question_text"][quora_train["target"] == 1].progress_apply(textstat.avg_letter_per_word))
plot_readability(lpw_sincere,lpw_insincere,"Average letters per word",2,['#8491A3','#2B2D42'])

# ## 6. Readability features
# This basically returns the readability statistics for given text. 

# ### 6.1 The Flesch Reading Ease formula
# The following table can be helpful to assess the ease of readability in a document.
# ###    Score	- Difficulty
# * 90-100 - Very Easy
# * 80-89 -	Easy
# * 70-79 -	Fairly Easy
# * 60-69 -	Standard
# * 50-59 -	Fairly Difficult
# * 30-49 -	Difficult
# * 0-29 -	Very Confusing
# 
# Read More: [Wikipedia](https://en.wikipedia.org/wiki/Flesch%E2%80%93Kincaid_readability_tests)

# In[ ]:


fre_sincere = np.array(quora_train["question_text"][quora_train["target"] == 0].progress_apply(textstat.flesch_reading_ease))
fre_insincere = np.array(quora_train["question_text"][quora_train["target"] == 1].progress_apply(textstat.flesch_reading_ease))
plot_readability(fre_sincere,fre_insincere,"Flesch Reading Ease",20)

# ## 6.2 The Flesch-Kincaid Grade Level
# Returns the Flesch-Kincaid Grade of the given text. This is a grade formula in that a score of 9.3 means that a ninth grader would be able to read the document.
# 
# Read More: [Wikipedia](https://en.wikipedia.org/wiki/Flesch%E2%80%93Kincaid_readability_tests)

# In[ ]:


fkg_sincere = np.array(quora_train["question_text"][quora_train["target"] == 0].progress_apply(textstat.flesch_kincaid_grade))
fkg_insincere = np.array(quora_train["question_text"][quora_train["target"] == 1].progress_apply(textstat.flesch_kincaid_grade))
plot_readability(fkg_sincere,fkg_insincere,"Flesch Kincaid Grade",4,['#C1D37F','#491F21'])

# ## 6.3 The Fog Scale (Gunning FOG Formula)
# Returns the FOG index of the given text. This is a grade formula in that a score of 9.3 means that a ninth grader would be able to read the document.
# 
# Read More: [Wikipedia](https://en.wikipedia.org/wiki/Gunning_fog_index)

# In[ ]:


fog_sincere = np.array(quora_train["question_text"][quora_train["target"] == 0].progress_apply(textstat.gunning_fog))
fog_insincere = np.array(quora_train["question_text"][quora_train["target"] == 1].progress_apply(textstat.gunning_fog))
plot_readability(fog_sincere,fog_insincere,"The Fog Scale (Gunning FOG Formula)",4,['#E2D58B','#CDE77F'])

# ## 6.4 Automated Readability Index
# Returns the ARI (Automated Readability Index) which outputs a number that approximates the grade level needed to comprehend the text.For example if the ARI is 6.5, then the grade level to comprehend the text is 6th to 7th grade.
# 
# Read More: [Wikipedia](https://en.wikipedia.org/wiki/Automated_readability_index)

# In[ ]:


ari_sincere = np.array(quora_train["question_text"][quora_train["target"] == 0].progress_apply(textstat.automated_readability_index))
ari_insincere = np.array(quora_train["question_text"][quora_train["target"] == 1].progress_apply(textstat.automated_readability_index))
plot_readability(ari_sincere,ari_insincere,"Automated Readability Index",10,['#488286','#FF934F'])

# ## 6.5 The Coleman-Liau Index
# Returns the grade level of the text using the Coleman-Liau Formula. This is a grade formula in that a score of 9.3 means that a ninth grader would be able to read the document.
# 
# Read More: [Wikipedia](https://en.wikipedia.org/wiki/Coleman%E2%80%93Liau_index)

# In[ ]:


cli_sincere = np.array(quora_train["question_text"][quora_train["target"] == 0].progress_apply(textstat.coleman_liau_index))
cli_insincere = np.array(quora_train["question_text"][quora_train["target"] == 1].progress_apply(textstat.coleman_liau_index))
plot_readability(cli_sincere,cli_insincere,"The Coleman-Liau Index",10,['#8491A3','#2B2D42'])

# ## 6.6 Linsear Write Formula
# Returns the grade level of the text using the Coleman-Liau Formula. This is a grade formula in that a score of 9.3 means that a ninth grader would be able to read the document.
# 
# Read More: [Wikipedia](https://en.wikipedia.org/wiki/Linsear_Write)

# In[ ]:


lwf_sincere = np.array(quora_train["question_text"][quora_train["target"] == 0].progress_apply(textstat.linsear_write_formula))
lwf_insincere = np.array(quora_train["question_text"][quora_train["target"] == 1].progress_apply(textstat.linsear_write_formula))
plot_readability(lwf_sincere,lwf_insincere,"Linsear Write Formula",2,['#8D99AE','#EF233C'])

# ## 6.7 Dale-Chall Readability Score
# Different from other tests, since it uses a lookup table of the most commonly used 3000 English words. Thus it returns the grade level using the New Dale-Chall Formula.
# 
# **Score** - **Understood by**
# * 4.9 or lower - average 4th-grade student or lower
# * 5.0–5.9	- average 5th or 6th-grade student
# * 6.0–6.9	- average 7th or 8th-grade student
# * 7.0–7.9	- average 9th or 10th-grade student
# * 8.0–8.9	- average 11th or 12th-grade student
# * 9.0–9.9	- average 13th to 15th-grade (college) student
# 
# Read More: [Wikipedia](https://en.wikipedia.org/wiki/Dale%E2%80%93Chall_readability_formula)

# In[ ]:


dcr_sincere = np.array(quora_train["question_text"][quora_train["target"] == 0].progress_apply(textstat.dale_chall_readability_score))
dcr_insincere = np.array(quora_train["question_text"][quora_train["target"] == 1].progress_apply(textstat.dale_chall_readability_score))
plot_readability(dcr_sincere,dcr_insincere,"Dale-Chall Readability Score",1,['#C65D17','#DDB967'])

# ## 6.8 Readability Consensus based upon all the above tests
# Based upon all the above tests, returns the estimated school grade level required to understand the text.

# In[ ]:


def consensus_all(text):
    return textstat.text_standard(text,float_output=True)

con_sincere = np.array(quora_train["question_text"][quora_train["target"] == 0].progress_apply(consensus_all))
con_insincere = np.array(quora_train["question_text"][quora_train["target"] == 1].progress_apply(consensus_all))
plot_readability(con_sincere,con_insincere,"Readability Consensus based upon all the above tests",2)

# # N-gram analysis
# In the fields of computational linguistics and probability, an n-gram is a contiguous sequence of n items from a given sample of text or speech. The items can be phonemes, syllables, letters, words or base pairs according to the application. The n-grams typically are collected from a text or speech corpus. When the items are words, n-grams may also be called shingles.

# In[ ]:


def word_generator(text):
    word = list(text.split())
    return word
def bigram_generator(text):
    bgram = list(nltk.bigrams(text.split()))
    bgram = [' '.join((a, b)) for (a, b) in bgram]
    return bgram
def trigram_generator(text):
    tgram = list(nltk.trigrams(text.split()))
    tgram = [' '.join((a, b, c)) for (a, b, c) in tgram]
    return tgram
sincere_words = sincere_questions.progress_apply(word_generator)
insincere_words = insincere_questions.progress_apply(word_generator)
sincere_bigrams = sincere_questions.progress_apply(bigram_generator)
insincere_bigrams = insincere_questions.progress_apply(bigram_generator)
sincere_trigrams = sincere_questions.progress_apply(trigram_generator)
insincere_trigrams = insincere_questions.progress_apply(trigram_generator)


# In[ ]:


color_brewer = ['#57B8FF','#B66D0D','#009FB7','#FBB13C','#FE6847','#4FB5A5','#8C9376','#F29F60','#8E1C4A','#85809B','#515B5D','#9EC2BE','#808080','#9BB58E','#5C0029','#151515','#A63D40','#E9B872','#56AA53','#CE6786','#449339','#2176FF','#348427','#671A31','#106B26','#008DD5','#034213','#BC2F59','#939C44','#ACFCD9','#1D3950','#9C5414','#5DD9C1','#7B6D49','#8120FF','#F224F2','#C16D45','#8A4F3D','#616B82','#443431','#340F09']

def ngram_visualizer(v,t):
    X = v.values
    Y = v.index
    trace = [go.Bar(
                y=Y,
                x=X,
                orientation = 'h',
                marker=dict(color=color_brewer, line=dict(color='rgb(8,48,107)',width=1.5,)),
                opacity = 0.6
    )]
    layout = go.Layout(
        title=t,
        margin = go.Margin(
            l = 200,
            r = 400
        )
    )

    fig = go.Figure(data=trace, layout=layout)
    iplot(fig, filename='horizontal-bar')
    
def ngram_plot(ngrams,title):
    ngram_list = []
    for i in tqdm(ngrams.values, total=ngrams.shape[0]):
        ngram_list.extend(i)
    random.shuffle(color_brewer)
    ngram_visualizer(pd.Series(ngram_list).value_counts()[:20],title)

# In[ ]:


# Top Sincere words
ngram_plot(sincere_words,"Top Sincere Words")

# In[ ]:


# Top Insincere words
ngram_plot(insincere_words,"Top Insincere Words")

# In[ ]:


# Sincere Bigrams
ngram_plot(sincere_bigrams,"Top 20 Sincere Bigrams")

# In[ ]:


# Insincere Bigrams
ngram_plot(insincere_bigrams,"Top 20 Insincere Bigrams")

# In[ ]:


# Sincere Trigrams
ngram_plot(sincere_trigrams,"Top 20 Sincere Trigrams")

# In[ ]:


# Insincere Trigrams
ngram_plot(insincere_trigrams,"Top 20 Insincere Trigrams")

# # What is topic-modelling?
# In machine learning and natural language processing, a topic model is a type of statistical model for discovering the abstract "topics" that occur in a collection of documents. Topic modeling is a frequently used text-mining tool for discovery of hidden semantic structures in a text body. Intuitively, given that a document is about a particular topic, one would expect particular words to appear in the document more or less frequently: "dog" and "bone" will appear more often in documents about dogs, "cat" and "meow" will appear in documents about cats, and "the" and "is" will appear equally in both. A document typically concerns multiple topics in different proportions; thus, in a document that is 10% about cats and 90% about dogs, there would probably be about 9 times more dog words than cat words.
# 
# The "topics" produced by topic modeling techniques are clusters of similar words. A topic model captures this intuition in a mathematical framework, which allows examining a set of documents and discovering, based on the statistics of the words in each, what the topics might be and what each document's balance of topics is. It involves various techniques of dimensionality reduction(mostly non-linear) and unsupervised learning like LDA, SVD, autoencoders etc.

# ## Count Vectorizers for the data

# In[ ]:


vectorizer_sincere = CountVectorizer(min_df=5, max_df=0.9, stop_words='english', lowercase=True, token_pattern='[a-zA-Z\-][a-zA-Z\-]{2,}')
sincere_questions_vectorized = vectorizer_sincere.fit_transform(sincere_questions)
vectorizer_insincere = CountVectorizer(min_df=5, max_df=0.9, stop_words='english', lowercase=True, token_pattern='[a-zA-Z\-][a-zA-Z\-]{2,}')
insincere_questions_vectorized = vectorizer_insincere.fit_transform(insincere_questions)

# ## Applying Latent Dirichlet Allocation(LDA) models

# In[ ]:


# Latent Dirichlet Allocation Model
lda_sincere = LatentDirichletAllocation(n_components=10, max_iter=5, learning_method='online',verbose=True)
sincere_lda = lda_sincere.fit_transform(sincere_questions_vectorized)
lda_insincere = LatentDirichletAllocation(n_components=10, max_iter=5, learning_method='online',verbose=True)
insincere_lda = lda_insincere.fit_transform(insincere_questions_vectorized)

# ## Printing keywords

# In[ ]:


# Functions for printing keywords for each topic
def selected_topics(model, vectorizer, top_n=10):
    for idx, topic in enumerate(model.components_):
        print("Topic %d:" % (idx))
        print([(vectorizer.get_feature_names()[i], topic[i])
                        for i in topic.argsort()[:-top_n - 1:-1]]) 

# In[ ]:


# Keywords for topics clustered by Latent Dirichlet Allocation
print("Sincere questions LDA Model:")
selected_topics(lda_sincere, vectorizer_sincere)

# In[ ]:


print("Insincere questions LDA Model:")
selected_topics(lda_insincere, vectorizer_insincere)

# # Visualizing LDA results of sincere questions with pyLDAvis

# In[ ]:


pyLDAvis.enable_notebook()
dash = pyLDAvis.sklearn.prepare(lda_sincere, sincere_questions_vectorized, vectorizer_sincere, mds='tsne')
dash

# So, the sincere questions mostly deal with topics like education, relationships, work life, product reviews, elements of life etc.

# # Visualizing LDA results of insincere questions

# In[ ]:


pyLDAvis.enable_notebook()
dash = pyLDAvis.sklearn.prepare(lda_insincere, insincere_questions_vectorized, vectorizer_insincere, mds='tsne')
dash

# The insincere questions, however deal with racism(there is a lot of mention of race here), homosexuality, politics, American elections, religion, terrotism and sex etc.
# 
# But words related to India and America appear in both sincere and insincere questions. Kaggle or quora, we are everywhere. 

# ### Show your appreciation by UPVOTES. I welcome suggestions to improve this kernel further.
