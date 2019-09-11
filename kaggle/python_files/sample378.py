#!/usr/bin/env python
# coding: utf-8

# # Text Pre-processing Techniques
# These techniques may or may not be useful for this competition. Given the fact that is a text competition, i thought that it would be a good oportunity to present them. 
# 
# I have used them before in two papers. [A Comparison of Pre-processing Techniques for Twitter Sentiment Analysis](https://link.springer.com/chapter/10.1007/978-3-319-67008-9_31) and [A comparative evaluation of pre-processing techniques and their interactions for twitter sentiment analysis](https://www.sciencedirect.com/science/article/pii/S0957417418303683). 
# 
# The full code is on this [Github repository](https://github.com/Deffro/text-preprocessing-techniques) with some extra techniques.

# In[ ]:


import pandas as pd
import numpy as np
import re

# ## Load Dataset and print some questions

# In[ ]:


train_df = pd.read_csv("../input/train.csv")
X_train = train_df["question_text"].fillna("dieter").values
test_df = pd.read_csv("../input/test.csv")
X_test = test_df["question_text"].fillna("dieter").values
y = train_df["target"]

text = train_df['question_text']

for row in text[:10]:
    print(row)

# ## 1. Remove Numbers
# **Example:** Which is best powerbank for iPhone 7 in India? -> Which is best powerbank for iPhone  in India?

# In[ ]:


def removeNumbers(text):
    """ Removes integers """
    text = ''.join([i for i in text if not i.isdigit()])         
    return text

text_removeNumbers = pd.DataFrame(columns=['TextBefore', 'TextAfter', 'Changed'])
text_removeNumbers['TextBefore'] = text.copy()


# In[ ]:


for index, row in text_removeNumbers.iterrows():
    row['TextAfter'] = removeNumbers(row['TextBefore'])

# In[ ]:


text_removeNumbers['Changed'] = np.where(text_removeNumbers['TextBefore']==text_removeNumbers['TextAfter'], 'no', 'yes')
print("{} of {} ({:.4f}%) questions have been changed.".format(len(text_removeNumbers[text_removeNumbers['Changed']=='yes']), len(text_removeNumbers), 100*len(text_removeNumbers[text_removeNumbers['Changed']=='yes'])/len(text_removeNumbers)))

# In[ ]:


for index, row in text_removeNumbers[text_removeNumbers['Changed']=='yes'].head().iterrows():
    print(row['TextBefore'],'->',row['TextAfter'])

# ## 2. Replace Repetitions of Punctuation
# This technique:
#  - replaces repetitions of exlamation marks with the tag "multiExclamation"
#  - replaces repetitions of question marks with the tag "multiQuestion"
#  - replaces repetitions of stop marks with the tag "multiStop"
#  
#  **Example:** How do I overcome the fear of facing an interview? It's killing me inside..what should I do? -> How do I overcome the fear of facing an interview? It's killing me inside multiStop what should I do?

# In[ ]:


def replaceMultiExclamationMark(text):
    """ Replaces repetitions of exlamation marks """
    text = re.sub(r"(\!)\1+", ' multiExclamation ', text)
    return text

def replaceMultiQuestionMark(text):
    """ Replaces repetitions of question marks """
    text = re.sub(r"(\?)\1+", ' multiQuestion ', text)
    return text

def replaceMultiStopMark(text):
    """ Replaces repetitions of stop marks """
    text = re.sub(r"(\.)\1+", ' multiStop ', text)
    return text

text_replaceRepOfPunct = pd.DataFrame(columns=['TextBefore', 'TextAfter', 'Changed'])
text_replaceRepOfPunct['TextBefore'] = text.copy()

# In[ ]:


for index, row in text_replaceRepOfPunct.iterrows():
    row['TextAfter'] = replaceMultiExclamationMark(row['TextBefore'])
    row['TextAfter'] = replaceMultiQuestionMark(row['TextBefore'])
    row['TextAfter'] = replaceMultiStopMark(row['TextBefore'])

# In[ ]:


text_replaceRepOfPunct['Changed'] = np.where(text_replaceRepOfPunct['TextBefore']==text_replaceRepOfPunct['TextAfter'], 'no', 'yes')
print("{} of {} ({:.4f}%) questions have been changed.".format(len(text_replaceRepOfPunct[text_replaceRepOfPunct['Changed']=='yes']), len(text_replaceRepOfPunct), 100*len(text_replaceRepOfPunct[text_replaceRepOfPunct['Changed']=='yes'])/len(text_replaceRepOfPunct)))

# In[ ]:


for index, row in text_replaceRepOfPunct[text_replaceRepOfPunct['Changed']=='yes'].head().iterrows():
    print(row['TextBefore'],'->',row['TextAfter'])

# ## 3. Remove Punctuation
# **Example:** Why haven't two democracies never ever went for a full fledged war? What stops them? -> Why havent two democracies never ever went for a full fledged war What stops them

# In[ ]:


import string
translator = str.maketrans('', '', string.punctuation)
text_removePunctuation = pd.DataFrame(columns=['TextBefore', 'TextAfter', 'Changed'])
text_removePunctuation['TextBefore'] = text.copy()

# In[ ]:


for index, row in text_removePunctuation.iterrows():
    row['TextAfter'] = row['TextBefore'].translate(translator) 

# In[ ]:


text_removePunctuation['Changed'] = np.where(text_removePunctuation['TextBefore']==text_removePunctuation['TextAfter'], 'no', 'yes')
print("{} of {} ({:.4f}%) questions have been changed.".format(len(text_removePunctuation[text_removePunctuation['Changed']=='yes']), len(text_removePunctuation), 100*len(text_removePunctuation[text_removePunctuation['Changed']=='yes'])/len(text_removePunctuation)))

# In[ ]:


for index, row in text_removePunctuation[text_removePunctuation['Changed']=='yes'].head().iterrows():
    print(row['TextBefore'],'->',row['TextAfter'])

# Hmm, i expected everything to change, because they are question with "?". Let's see the ones that didn't change.

# In[ ]:


for index, row in text_removePunctuation[text_removePunctuation['Changed']=='no'].head().iterrows():
    print(row['TextBefore'],'->',row['TextAfter'])

# ## 4. Replace Contractions
# This techniques replaces contractions to their equivalents.
# 
# **Example:** What's the scariest thing that ever happened to anyone? -> What is the scariest thing that ever happened to anyone?

# In[ ]:


contraction_patterns = [ (r'won\'t', 'will not'), (r'can\'t', 'cannot'), (r'i\'m', 'i am'), (r'ain\'t', 'is not'), (r'(\w+)\'ll', '\g<1> will'), (r'(\w+)n\'t', '\g<1> not'),
                         (r'(\w+)\'ve', '\g<1> have'), (r'(\w+)\'s', '\g<1> is'), (r'(\w+)\'re', '\g<1> are'), (r'(\w+)\'d', '\g<1> would'), (r'&', 'and'), (r'dammit', 'damn it'), (r'dont', 'do not'), (r'wont', 'will not') ]
def replaceContraction(text):
    patterns = [(re.compile(regex), repl) for (regex, repl) in contraction_patterns]
    for (pattern, repl) in patterns:
        (text, count) = re.subn(pattern, repl, text)
    return text

text_replaceContractions = pd.DataFrame(columns=['TextBefore', 'TextAfter', 'Changed'])
text_replaceContractions['TextBefore'] = text.copy()

# In[ ]:


for index, row in text_replaceContractions.iterrows():
    row['TextAfter'] = replaceContraction(row['TextBefore'])

# In[ ]:


text_replaceContractions['Changed'] = np.where(text_replaceContractions['TextBefore']==text_replaceContractions['TextAfter'], 'no', 'yes')
print("{} of {} ({:.4f}%) questions have been changed.".format(len(text_replaceContractions[text_replaceContractions['Changed']=='yes']), len(text_replaceContractions), 100*len(text_replaceContractions[text_replaceContractions['Changed']=='yes'])/len(text_replaceContractions)))

# In[ ]:


for index, row in text_replaceContractions[text_replaceContractions['Changed']=='yes'].head().iterrows():
    print(row['TextBefore'],'->',row['TextAfter'])

# ## 5. Lowercase
# **Example:** What do you know about Bram Fischer and the Rivonia Trial? -> what do you know about bram fischer and the rivonia trial?

# In[ ]:


text_lowercase = pd.DataFrame(columns=['TextBefore', 'TextAfter', 'Changed'])
text_lowercase['TextBefore'] = text.copy()

# In[ ]:


for index, row in text_lowercase.iterrows():
    row['TextAfter'] = row['TextBefore'].lower()

# In[ ]:


text_lowercase['Changed'] = np.where(text_lowercase['TextBefore']==text_lowercase['TextAfter'], 'no', 'yes')
print("{} of {} ({:.4f}%) questions have been changed.".format(len(text_lowercase[text_lowercase['Changed']=='yes']), len(text_lowercase), 100*len(text_lowercase[text_lowercase['Changed']=='yes'])/len(text_lowercase)))

# In[ ]:


for index, row in text_lowercase[text_lowercase['Changed']=='yes'].head().iterrows():
    print(row['TextBefore'],'->',row['TextAfter'])

# Some question are written only in lowercase. This happens when they start with a number.

# In[ ]:


for index, row in text_lowercase[text_lowercase['Changed']=='no'].head().iterrows():
    print(row['TextBefore'],'->',row['TextAfter'])

# ## 6. Replace Negations with Antonyms
# **Example:** Why are humans not able to be evolved developing resistance against diseases? -> Why are humans unable to be evolved developing resistance against diseases ?

# In[ ]:


import nltk
from nltk.corpus import wordnet

def replace(word, pos=None):
    """ Creates a set of all antonyms for the word and if there is only one antonym, it returns it """
    antonyms = set()
    for syn in wordnet.synsets(word, pos=pos):
        for lemma in syn.lemmas():
            for antonym in lemma.antonyms():
                antonyms.add(antonym.name())
    if len(antonyms) == 1:
        return antonyms.pop()
    else:
        return None

def replaceNegations(text):
    """ Finds "not" and antonym for the next word and if found, replaces not and the next word with the antonym """
    i, l = 0, len(text)
    words = []
    while i < l:
        word = text[i]
        if word == 'not' and i+1 < l:
            ant = replace(text[i+1])
            if ant:
                words.append(ant)
                i += 2
                continue
        words.append(word)
        i += 1
    return words

def tokenize1(text):
    tokens = nltk.word_tokenize(text)
    tokens = replaceNegations(tokens)
    text = " ".join(tokens)
    return text

text_replaceNegations = pd.DataFrame(columns=['TextBefore', 'TextAfter', 'Changed'])
text_replaceNegations['TextBefore'] = text.copy()

# In[ ]:


for index, row in text_replaceNegations.iterrows():
    row['TextAfter'] = tokenize1(row['TextBefore'])

# In[ ]:


text_replaceNegations['Changed'] = np.where(text_replaceNegations['TextBefore'].str.replace(" ","")==text_replaceNegations['TextAfter'].str.replace(" ","").str.replace("``",'"').str.replace("''",'"'), 'no', 'yes')
print("{} of {} ({:.4f}%) questions have been changed.".format(len(text_replaceNegations[text_replaceNegations['Changed']=='yes']), len(text_replaceNegations), 100*len(text_replaceNegations[text_replaceNegations['Changed']=='yes'])/len(text_replaceNegations)))

# In[ ]:


for index, row in text_replaceNegations[text_replaceNegations['Changed']=='yes'].head().iterrows():
    print(row['TextBefore'],'->',row['TextAfter'])

# ## 7. Handle Capitalized Words
# **Example:** Which is better to use, Avro or ORC? -> Which is better to use , Avro or ALL_CAPS_ORC ?

# In[ ]:


def addCapTag(word):
    """ Finds a word with at least 3 characters capitalized and adds the tag ALL_CAPS_ """
    if(len(re.findall("[A-Z]{3,}", word))):
        word = word.replace('\\', '' )
        transformed = re.sub("[A-Z]{3,}", "ALL_CAPS_"+word, word)
        return transformed
    else:
        return word

def tokenize2(text):
    finalTokens = []
    tokens = nltk.word_tokenize(text)
    for w in tokens:
        finalTokens.append(addCapTag(w))
    text = " ".join(finalTokens)
    return text

text_handleCapWords = pd.DataFrame(columns=['TextBefore', 'TextAfter', 'Changed'])
text_handleCapWords['TextBefore'] = text.copy()

# In[ ]:


for index, row in text_handleCapWords.iterrows():
    row['TextAfter'] = tokenize2(row['TextBefore'])

# In[ ]:


text_handleCapWords['Changed'] = np.where(text_handleCapWords['TextBefore'].str.replace(" ","")==text_handleCapWords['TextAfter'].str.replace(" ","").str.replace("``",'"').str.replace("''",'"'), 'no', 'yes')
print("{} of {} ({:.4f}%) questions have been changed.".format(len(text_handleCapWords[text_handleCapWords['Changed']=='yes']), len(text_handleCapWords), 100*len(text_handleCapWords[text_handleCapWords['Changed']=='yes'])/len(text_handleCapWords)))

# In[ ]:


for index, row in text_handleCapWords[text_handleCapWords['Changed']=='yes'].head().iterrows():
    print(row['TextBefore'],'->',row['TextAfter'])

# ## 8. Remove Stopwords
# **Example:** How I know whether a girl had done sex before sex with me? -> How I know whether girl done sex sex ?

# In[ ]:


from nltk.corpus import stopwords
stoplist = stopwords.words('english')

def tokenize(text):
    finalTokens = []
    tokens = nltk.word_tokenize(text)
    for w in tokens:
        if (w not in stoplist):
            finalTokens.append(w)
    text = " ".join(finalTokens)
    return text

text_removeStopwords = pd.DataFrame(columns=['TextBefore', 'TextAfter', 'Changed'])
text_removeStopwords['TextBefore'] = text.copy()

# In[ ]:


for index, row in text_removeStopwords.iterrows():
    row['TextAfter'] = tokenize(row['TextBefore'])

# In[ ]:


text_removeStopwords['Changed'] = np.where(text_removeStopwords['TextBefore'].str.replace(" ","")==text_removeStopwords['TextAfter'].str.replace(" ","").str.replace("``",'"').str.replace("''",'"'), 'no', 'yes')
print("{} of {} ({:.4f}%) questions have been changed.".format(len(text_removeStopwords[text_removeStopwords['Changed']=='yes']), len(text_removeStopwords), 100*len(text_removeStopwords[text_removeStopwords['Changed']=='yes'])/len(text_removeStopwords)))

# In[ ]:


for index, row in text_removeStopwords[text_removeStopwords['Changed']=='yes'].head().iterrows():
    print(row['TextBefore'],'->',row['TextAfter'])

# ## 9. Replace Elongated Words
# This technique replaces an elongated word with its basic form, unless the word exists in the lexicon.
# 
# **Example:** Game of Thrones, what does Arya find out about Littlefinger? -> Game of Thrones , what does Arya find out about Litlefinger ?

# In[ ]:


def replaceElongated(word):
    """ Replaces an elongated word with its basic form, unless the word exists in the lexicon """

    repeat_regexp = re.compile(r'(\w*)(\w)\2(\w*)')
    repl = r'\1\2\3'
    if wordnet.synsets(word):
        return word
    repl_word = repeat_regexp.sub(repl, word)
    if repl_word != word:      
        return replaceElongated(repl_word)
    else:       
        return repl_word
    
def tokenize(text):
    finalTokens = []
    tokens = nltk.word_tokenize(text)
    for w in tokens:
        finalTokens.append(replaceElongated(w))
    text = " ".join(finalTokens)
    return text

text_removeElWords = pd.DataFrame(columns=['TextBefore', 'TextAfter', 'Changed'])
text_removeElWords['TextBefore'] = text.copy()

# In[ ]:


for index, row in text_removeElWords.iterrows():
    row['TextAfter'] = tokenize(row['TextBefore'])

# In[ ]:


text_removeElWords['Changed'] = np.where(text_removeElWords['TextBefore'].str.replace(" ","")==text_removeElWords['TextAfter'].str.replace(" ","").str.replace("``",'"').str.replace("''",'"'), 'no', 'yes')
print("{} of {} ({:.4f}%) questions have been changed.".format(len(text_removeElWords[text_removeElWords['Changed']=='yes']), len(text_removeElWords), 100*len(text_removeElWords[text_removeElWords['Changed']=='yes'])/len(text_removeElWords)))

# In[ ]:


for index, row in text_removeElWords[text_removeElWords['Changed']=='yes'].head().iterrows():
    print(row['TextBefore'],'->',row['TextAfter'])

# ## 10. Stemming/Lemmatizing
# **Example:** How do modern military submarines reduce noise to achieve stealth? -> how do modern militari submarin reduc nois to achiev stealth ?

# In[ ]:


from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer() #set stemmer
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer() # set lemmatizer

def tokenize(text):
    finalTokens = []
    tokens = nltk.word_tokenize(text)
    for w in tokens:
        finalTokens.append(stemmer.stem(w)) # change this to lemmatizer.lemmatize(w) for Lemmatizing
    text = " ".join(finalTokens)
    return text

text_stemming = pd.DataFrame(columns=['TextBefore', 'TextAfter', 'Changed'])
text_stemming['TextBefore'] = text.copy()

# In[ ]:


for index, row in text_stemming.iterrows():
    row['TextAfter'] = tokenize(row['TextBefore'])

# In[ ]:


text_stemming['Changed'] = np.where(text_stemming['TextBefore'].str.replace(" ","")==text_stemming['TextAfter'].str.replace(" ","").str.replace("``",'"').str.replace("''",'"'), 'no', 'yes')
print("{} of {} ({:.4f}%) questions have been changed.".format(len(text_stemming[text_stemming['Changed']=='yes']), len(text_stemming), 100*len(text_stemming[text_stemming['Changed']=='yes'])/len(text_stemming)))

# In[ ]:


for index, row in text_stemming[text_stemming['Changed']=='yes'].head().iterrows():
    print(row['TextBefore'],'->',row['TextAfter'])

# ## Combos
# Of course we can use more than one technique at the same time. The order is essential here.
# 
# **Example:** What are the recommended 2D game engines for a beginning Python programmer? -> what recommend d game engin begin python programm

# In[ ]:


def tokenize(text):
    finalTokens = []
    tokens = nltk.word_tokenize(text)
    for w in tokens:
        if (w not in stoplist):
            w = addCapTag(w) # Handle Capitalized Words
            w = w.lower() # Lowercase
            w = replaceElongated(w) # Replace Elongated Words
            w = stemmer.stem(w) # Stemming
            finalTokens.append(w)
    text = " ".join(finalTokens)
    return text

text_combos = pd.DataFrame(columns=['TextBefore', 'TextAfter', 'Changed'])
text_combos['TextBefore'] = text.copy()

# In[ ]:


for index, row in text_combos.iterrows():
    row['TextAfter'] = replaceContraction(row['TextBefore']) # Replace Contractions
    row['TextAfter'] = removeNumbers(row['TextAfter']) # Remove Integers
    row['TextAfter'] = replaceMultiExclamationMark(row['TextAfter']) # Replace Multi Exclamation Marks
    row['TextAfter'] = replaceMultiQuestionMark(row['TextAfter']) # Replace Multi Question Marks
    row['TextAfter'] = replaceMultiStopMark(row['TextAfter']) # Repalce Multi Stop Marks
    row['TextAfter'] = row['TextAfter'].translate(translator) # Remove Punctuation
    row['TextAfter'] = tokenize(row['TextAfter'])

# In[ ]:


text_combos['Changed'] = np.where(text_combos['TextBefore'].str.replace(" ","")==text_combos['TextAfter'].str.replace(" ","").str.replace("``",'"').str.replace("''",'"'), 'no', 'yes')
print("{} of {} ({:.4f}%) questions have been changed.".format(len(text_combos[text_combos['Changed']=='yes']), len(text_combos), 100*len(text_combos[text_combos['Changed']=='yes'])/len(text_combos)))

# In[ ]:


for index, row in text_combos[text_combos['Changed']=='yes'].head().iterrows():
    print(row['TextBefore'],'->',row['TextAfter'])

# Thank you for reaching this point! Hope you enjoyed it! Your upvote will be much appreciated!
