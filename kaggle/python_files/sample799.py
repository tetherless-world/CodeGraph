#!/usr/bin/env python
# coding: utf-8

# **DATA SCIENCE FOR THE CITY OF LOS ANGELES**
# 
# Objectives: 
# * Provide recommendations to make the job bulletins more appealing
# * Provide recommendations to make the job bulletins more inclusive
# * Convert the text format job bulletins into a structured csv file with the given columns, and doing some validation checks (like for ex: validating against a list of valid job titles)
# * Identify any promotion opportunities
# * Include any additional info, as appropriate

# **This notebook does analysis of the job bulletins for the city of Los Angeles, and creates 
# a structured csv file based on the format given, from the content of the job bulletins.
# The analysis of the job bulletins results in number of recommendations to make the bulletins more 
# inclusive and appealing to prospective applicants**

# In[1]:


#Necessary imports. All open source libraries using python 3.6
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re
import nltk
import string
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk import word_tokenize
import random
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import sent_tokenize
from wordcloud import WordCloud, STOPWORDS
from flashtext import KeywordProcessor

import os
print(os.listdir("../input"))


# In[2]:


#Load the data of job bulletins into a List of text, also load the filenames (to be used later), derive
# the job position from the filename for some preliminary analysis 
#(actual position for data dictionary will be derived from job bulletin)
def load_jobopening_dataset():

    data_path = '../input/cityofla/CityofLA/Job Bulletins'

    texts = []
    positions = []
    file_names=[]
    for fname in sorted(os.listdir(data_path)):
        if fname.endswith('.txt'):
            file_names.append(fname)
            with open(os.path.join(data_path, fname),"rb") as f:
                texts.append(str(f.read()))
                positions.append((re.split(' (?=class)', fname))[0])
    
    #print the length of the List of text, length of file_names and positions and make sure they are all equal
    print(len(texts))
    print(len(positions))
    print(len(file_names))

    return (texts,positions,file_names)

# In[3]:


job_data, positions, file_names = load_jobopening_dataset()

# In[ ]:


#Let us examine the first job ad, we print the first 250 chars of the job bulletin
job_data[0].replace("\\r\\n"," ").replace("\\\'s","")[:250]

# In[ ]:


exclude = set(string.punctuation) 
wpt = nltk.WordPunctTokenizer()
stop_words = nltk.corpus.stopwords.words('english')
#Remove unnecessary words by including them in stopword list, for ex: we already know city, los angeles 
#are going to be there and are not going to add any extra meaning, we are adding 'may' as this is showing
#up as most frequent word and this does not carry much information
newStopWords = ['city','los','angele','angeles','may']
stop_words.extend(newStopWords)
table = str.maketrans('', '', string.punctuation)

lemma = WordNetLemmatizer()
porter = PorterStemmer()

def normalize_document(doc):
    #replace newline and tab chars
    doc = doc.replace("\\r\\n"," ").replace("\\\'s","").replace("\t"," ") #.split("b'")[1]
    # tokenize document
    tokens = doc.split()
    # remove punctuation from each word
    tokens = [w.translate(table) for w in tokens]
    # convert to lower case
    lower_tokens = [w.lower() for w in tokens]
    #remove spaces
    stripped = [w.strip() for w in lower_tokens]
    # remove remaining tokens that are not alphabetic
    words = [word for word in stripped if word.isalpha()]
    # filter stopwords out of document
    filtered_tokens = [token for token in words if token not in stop_words]
    #normalized = " ".join(lemma.lemmatize(word) for word in filtered_tokens)
    #join the tokens back to get the original doc
    doc = ' '.join(filtered_tokens)
    return doc

normalize_corpus = np.vectorize(normalize_document)

# In[ ]:


#apply the text normalization to list of job positions
norm_positions=[]
for text_sample in positions:
    norm_positions.append(normalize_document(text_sample))

# In[ ]:


#apply the text normalization to list of job ads
norm_corpus=[]
for text_sample in job_data:
    norm_corpus.append(normalize_document(text_sample))

# In[ ]:


#check the first position (this is for n-gram analysis after removal of numerics, 
#for actual data dictionary numerics will be considered)
norm_positions[0][:250]

# In[ ]:


#check the first normalized job ad, the first 250 chars
norm_corpus[0][:250]

# In[ ]:


#get median number of words per sample from the normalized text
def get_num_words_per_sample(sample_texts):
    num_words = [len(s.split()) for s in sample_texts]
    return np.median(num_words)

# In[ ]:


print("Median value of number of words per sample is")
print(get_num_words_per_sample(norm_corpus))

# In[ ]:


#Plot length distribution of job ads in terms of number of words per sample
def plot_sample_length_distribution(sample_texts):
    plt.hist([len(s.split()) for s in sample_texts], 50)
    plt.xlabel('Length of a sample (No. of words)')
    plt.ylabel('Number of samples')
    plt.title('Sample length distribution')
    plt.show()

# In[ ]:


plot_sample_length_distribution(norm_corpus)

# One obs here is that : **50% of the job ads are above 733 words (median) in length**. Its **too verbose**, most applicants might not read this in full. **There needs to be some way to reduce the length, and make the job ad as compact as possible(maybe max 200-300 words)**

# In[ ]:


#Plot freq distribution of n-grams with single words, bi-grams, tri-grams and four-grams. Plot frequency of
#n-grams with highest frequencies occuring first, input parameters : ngram_range, 
#maximum n_grams to be considered
def plot_frequency_distribution_of_ngrams(sample_texts,
                                          ngram_range=(1, 2),
                                          num_ngrams=30):
    """Plots the frequency distribution of n-grams.

    # Arguments
        samples_texts: list, sample texts.
        ngram_range: tuple (min, mplt), The range of n-gram values to consider.
            Min and mplt are the lower and upper bound values for the range.
        num_ngrams: int, number of n-grams to plot.
            Top `num_ngrams` frequent n-grams will be plotted.
    """
    # Create args required for vectorizing.
    kwargs = {
            'ngram_range': ngram_range,
            'dtype': 'int32',
            'strip_accents': 'unicode',
            'decode_error': 'replace',
            'analyzer': 'word',  # Split text into word tokens.
    }
    vectorizer = CountVectorizer(**kwargs)

    vectorized_texts = vectorizer.fit_transform(sample_texts)

    # This is the list of all n-grams in the index order from the vocabulary.
    all_ngrams = list(vectorizer.get_feature_names())
    num_ngrams = min(num_ngrams, len(all_ngrams))
    ngrams = all_ngrams[:num_ngrams]

    # Add up the counts per n-gram ie. column-wise
    all_counts = vectorized_texts.sum(axis=0).tolist()[0]

    # Sort n-grams and counts by frequency and get top `num_ngrams` ngrams.
    all_counts, all_ngrams = zip(*[(c, n) for c, n in sorted(
        zip(all_counts, all_ngrams), reverse=True)])
    ngrams = list(all_ngrams)[:num_ngrams]
    counts = list(all_counts)[:num_ngrams]

    idx = np.arange(num_ngrams)
    plt.figure(figsize=(30,20)) 
    plt.bar(idx, counts, width=0.6, color='b')
    plt.xlabel('N-grams',fontsize="18")
    plt.ylabel('Frequencies',fontsize="18")
    plt.title('Frequency distribution of n-grams',fontsize="36")
    plt.xticks(idx, ngrams, rotation=45, fontsize=18)
    plt.yticks(fontsize=18)
    plt.show()

# In[ ]:


plot_frequency_distribution_of_ngrams(norm_corpus,ngram_range=(1, 2))

# Observation here is : the words  **'candidates', 'examination' , 'applicants' are occurring too many times.** This probably will make the job ad very very formal, and scare away applicants.
# 
# The good point is that the word **'disability' is making in the top 20 most frequent words**, which means the job ads are inclusive.

# In[ ]:


plot_frequency_distribution_of_ngrams(norm_corpus,ngram_range=(2, 2))

# Some good bi-grams can be seen here, which are **inclusive, ex: 'disability accomodation' , 'equal employment'**
# There is a heavy **emphasis on 'minimum requirements'** as can be seen from the bi-grams **'minimum requirements','meet minimum', 'minimum qualifications'**. While this is customary for a Govt job, there is scope of improvement here to make the job ad more inclusive in nature, by removing such formal words. Suggested format is : This job requires 4 yr full time degree, however with sufficient experience this can be waived off. 
# **It is interesting to note : the word 'waive' does not occur in the n-gram list.**

# In[ ]:


plot_frequency_distribution_of_ngrams(norm_corpus,ngram_range=(3, 3))

# There are excellent **inclusive tri-grams** occuring in the most frequent list, ex: **'equal employment opportunity', 'disability accommodation form','american disabilities act'**
# The bi-grams like **'meet minimum requirements' , 'minimum qualifications met' are a bit formal**, customary of Govt job ads. Can be made more informal to make the job ad more appealing

# In[ ]:


plot_frequency_distribution_of_ngrams(norm_corpus,ngram_range=(4, 4),num_ngrams=15)

# **This analysis again reveals some good and not so good aspects**
# Good aspect is the presence of n-grams like ** 'equal employment opportunity employer',**
# There is a heavy emphasis on **'ensure minimum qualifications met', 'minimum qualifications stated bulletin'**
# **However, there are presence of n-grams like 'marital status sexual orientation' ** and it is not clear why this information is needed, what is the policy of LA Govt on these, does it have a inclusive policy on all sexual orientations. There is scope of improvement here.

# In[ ]:


plot_frequency_distribution_of_ngrams(norm_positions)

# In[ ]:


plot_frequency_distribution_of_ngrams(norm_positions,ngram_range=(2, 2))

# In[ ]:


plot_frequency_distribution_of_ngrams(norm_positions,ngram_range=(3,3))

# ****Shows Supervisor jobs, Chief jobs, Officer, Engineering, Inspector are most prevalent**, followed by electrician, analyst,associate**
# 
# ****Some issues are use of terms like worker, operator** ex: 'wastewater treatment operator', 'electric trouble dispatcher' maybe these can be changed like 'waste water treatment technical specialist' for better appeal**

# In[ ]:


full_norm_corpus=' '.join(norm_corpus)
stopwords = set(STOPWORDS)
stopwords.update(["class", "code"])

wordcloud = WordCloud(stopwords=stopwords,max_words=50).generate(full_norm_corpus)
plt.imshow(wordcloud)
plt.axis("off")
plt.show()

# Some words that might scare off applicants are : **written test, eligible list, minimum qualification**
# 
# Good points : **disability accommodation**

# In[ ]:


full_norm_corpus=' '.join(norm_positions)
stopwords = set(STOPWORDS)
stopwords.update(["rev"])
wordcloud = WordCloud(stopwords=stopwords,max_words=50).generate(full_norm_corpus)
plt.imshow(wordcloud)
plt.axis("off")
plt.show()

# In[ ]:


#Let us check where the keywords like examination are occuring, let us take the first job bulletin as example

text = job_data[0].replace("\\r\\n"," ").replace("\\\'s","")
sentences = sent_tokenize(text)
my_sentence=[sent for sent in sentences if 'examination' in word_tokenize(sent)]
print(my_sentence)

#  This is just for the first job ad for the post of 311 Director, and it shows that the job ad is quite scary to someone looking at it. Instead of a **negative tone**, it could have been written like **'Candidates need to complete a quick Qualifications Questionnaire' instead of 'Applicants who fail to complete the Qualifications Questionnaire** will not be considered further in this examination, and their application will not be processed.'. The other rules could have stated in simple terms, again instead of a negative tone, it should be : 'Applicants who are already in a City position or on a reserve list can apply for this examination' **

# In[ ]:


#Let us choose the sentences where the word 'examination' is occuring in the full corpus 
#and check some random sentences
sentences_ngram=[]
for i in range(len(job_data)):
    text = job_data[i].replace("\\r\\n"," ").replace("\\\'s","")
    sentences = sent_tokenize(text)
    selected_sentence=[sent for sent in sentences if 'examination' in word_tokenize(sent)]
    sentences_ngram.append(selected_sentence)

# In[ ]:


len(sentences_ngram)

# In[ ]:


sentence = random.choice(sentences_ngram)
sentence

# **Some examples:** 'Only applicants that are currently or have previously worked in the Department of Water and Power in DDR Numbers 93-39121, 93-39137, 93-39120, 93-39135, 93-39119, 93-39138, 93-39010, 93-39126, 93-39002, 93-39023, 93-39026, 42-39301, or 93-39130, and meet the above noted requirement qualify to take this examination.' -** this kind of ad is too verbose and might set off applicants. **
# 'This is an extremely competitive examination and a sufficient number of candidates with the highest scores will continue in the selection process.' - **Again a very scary way of posting a job ad. All exams are by nature competitive.**
# 'Candidates in the examination process may file protests as provided in Sec.' - **Not a good way of portraying an exam. Why talk about protests ? Does this happen often?**

# In[ ]:


print(os.listdir("../input/cityofla/CityofLA/Additional data"))

# In[ ]:


print(os.listdir("../input/cityofla/CityofLA/Additional data/PDFs/2018/December/Dec 14"))

# In[ ]:


import docx2txt
my_text = docx2txt.process("../input/cityofla/CityofLA/Additional data/Description of promotions in job bulletins.docx")
print(my_text)

# In[4]:


titles = pd.read_csv("../input/cityofla/CityofLA/Additional data/job_titles.csv", header=None)

# In[45]:


titles.head()
title_text = titles[0]
for title in title_text:
    title=title.strip()
len(title_text)

# In[ ]:


#Let us check the distribution of length (number of words) in the Job Title
plot_sample_length_distribution(title_text)

# In[44]:


#Print job titles greater than 4 words
ctr = 0
for i in range(len(title_text)):
    #print(title)
    title = title_text[i].split()
    if (len(title) >=4):
        title = ' '.join(title)
        print(title)
        ctr=ctr+1
        if (ctr==10):
            break

# The observation here is that some of the titles are too long (>4 words). It is clear the Dept name is part of the job Title, which should not be the case**AIRPORT CHIEF INFORMATION SECURITY OFFICER. Can this be re-organized like Dept : AIRPORT, Title : CHIEF INFORMATION SECURITY OFFICER **

# In[ ]:


sample_job_class = pd.read_csv("../input/cityofla/CityofLA/Additional data/sample job class export template.csv", header=None)

# In[ ]:


sample_job_class

# In[ ]:


data_dict=pd.read_csv("../input/cityofla/CityofLA/Additional data/kaggle_data_dictionary.csv")

# In[ ]:


data_dict

# In[6]:


#for i in range(len(file_names)):
#    fname = file_names[i]
#    if (fname.strip() == "SYSTEMS ANALYST 1596 102717.txt"):
#        print(i)

# In[41]:


#for i in range(500,650):
#    s = job_data[i].replace("\\r\\n"," ").replace("\\t","")
#    if (get_position(s)=='SENIOR SYSTEMS ANALYST'):
#        print(i)

# In[42]:


#Let us print a job bulletin for reference, in its original form (after replacing \\r\\n)
s = job_data[548].replace("\\r\\n"," ").replace("\\t","")
s[:500]

# In[43]:


def get_position(s):
    title_match=False
    pos = re.findall(r'(.*?)Class Code',s)
    pos1 = re.findall(r'(.*?)Class  Code',s)
    if (len(pos1) > 0):
        pos = pos1
    if (len(pos) > 0):
        job_title= pos[0].replace("b'","").replace("b\"","").replace("'","").replace("\\","").strip()
        for title in title_text:
            if (title.replace("'","")==job_title):
                title_match=True
                break
    
    if(title_match==True):
        return job_title
    else:
        return "Invalid job title"
get_position(s)

# In[9]:


def get_JobCode(s):
    job_code = 0
    code = re.findall(r'Class Code:(.*?)Open',s)
    if (len(code)>0):
        job_code= int(code[0].strip())
    return job_code
get_JobCode(s)

# In[11]:



# In[12]:


import dateutil.parser as dparser
from datetime import datetime
import datefinder

def get_OpenDate(s):
    openDateRet=""
    openDate = re.findall(r'Open Date:(.*?)ANNUAL',s)
    openStr=""
    if (len(openDate)>0):
        #print(openDate)
        openDate = openDate[0].strip()
        openStr=re.findall(r'(?:Exam).*',openDate)
        #print(openStr)
    
    matches = list(datefinder.find_dates(openDate))

    if len(matches) > 0:
        for i in range(len(matches)):
            date = matches[i]
            openDateRet=str(date.date())
   
    return openDateRet,openStr
get_OpenDate(s)

# In[13]:


def get_SalaryRange(s):
    salaryRange = re.findall(r'ANNUAL SALARY(.*?)NOTE',s)
    salaryRange_1 = re.findall(r'ANNUAL SALARY(.*?)DUTIES',s)
    salaryRange_2 = re.findall(r'ANNUAL SALARY(.*?)\(flat',s)
    len1=0
    len2=0
    len3=0
    if (len(salaryRange) > 0):
        len1 = len(salaryRange[0])
    if (len(salaryRange_1) > 0):
        len2 = len(salaryRange_1[0])
    if (len(salaryRange_2) > 0):
        len3 = len(salaryRange_2[0])
    if ((len1 > 0) & (len2 > 0)):
        if (len1 < len2):
            salaryRange = salaryRange
        else:
            salaryRange = salaryRange_1
        
    if (len(salaryRange)>0):
        salaryRange = salaryRange[0].strip()
        
    
    return salaryRange
get_SalaryRange(s)

# In[14]:


def get_qualification(s):
    qual = re.findall(r'REQUIREMENTS/MINIMUM QUALIFICATIONS(.*?)WHERE TO APPLY',s)
    if (len(qual)==0):
        qual = re.findall(r'REQUIREMENT/MINIMUM QUALIFICATION(.*?)WHERE TO APPLY',s)
    if (len(qual)==0):
        qual = re.findall(r'REQUIREMENTS(.*?)WHERE TO APPLY',s)
    if (len(qual)==0):
        qual = re.findall(r'REQUIREMENT(.*?)WHERE TO APPLY',s)
    if (len(qual)>0):
        qual = qual[0].replace("\\'s","'s").strip()
    else:
        qual=""
    return qual
qual=get_qualification(s)
qual[:500]

# In[15]:


def get_educationMajor(s):
    educationMajor=""
    sentences = sent_tokenize(s)
    selected_sentences=[sent for sent in sentences if "major" in word_tokenize(sent)]
    for i in range(len(selected_sentences)):
        major = re.findall(r'major in(.*?),',selected_sentences[i])
        if (len(major)>0):
            educationMajor=major[0].strip()

    return educationMajor
major=get_educationMajor(qual)
major

# In[16]:


def get_college(s):
    college=""
    keyword_processor = KeywordProcessor()
    keyword_processor.add_keyword('college or university')
    keyword_processor.add_keyword('college')
    keyword_processor.add_keyword('university')
    keyword_processor.add_keyword('high school')
    sentences = sent_tokenize(s)
    for j in range(len(sentences)):
        sentence = sentences[j]
        keywords_found = keyword_processor.extract_keywords(sentence)
        if (len(keywords_found) > 0):
            for i in range(len(keywords_found) ):
                if (keywords_found[i]=='college or university'):
                    college='college or university'
                    break
                elif (keywords_found[i]=='college'):
                    college='college'
                    break
                elif (keywords_found[i]=='university'):
                    college='university'
                    break
                elif (keywords_found[i]=='high school'):
                    college='high school'
                    break
    

    return college
major=get_college(qual)
major

# In[17]:


def get_eduSemDur(s):
    educationDur=""
    sentences = sent_tokenize(s)
    selected_sentences=[sent for sent in sentences if "semester" in word_tokenize(sent)]
    for i in range(len(selected_sentences)):
        dur = re.findall(r'(.*?)semester',selected_sentences[i])
        #print(dur)
        if (len(dur)>0):
            educationDur=dur[0]+'sememster'

    return educationDur
eduDur=get_eduSemDur(qual)
eduDur

# In[18]:


def get_Duties(s):
    duties = re.findall(r'DUTIES(.*?)REQUIREMENT',s)
    jobDuties=""
    if (len(duties)>0):
        jobDuties= duties[0].strip()
    return jobDuties
duties=get_Duties(s)
duties[:200]

# In[19]:


def get_eduYrs(s):
    keyword_processor = KeywordProcessor()
    education_yrs=0.0
    keyword_processor.add_keyword('four-year')
    keyword_processor.add_keyword('four years')
    sentences = sent_tokenize(s)
    selected_sentences=[sent for sent in sentences if "degree" in word_tokenize(sent)]
    selected_sentences1=[sent for sent in sentences if "Graduation" in word_tokenize(sent)]

    for i in range(len(selected_sentences)):
        keywords_found = keyword_processor.extract_keywords(selected_sentences[i])
        if (len(keywords_found) > 0):
            education_yrs=4.0
    for i in range(len(selected_sentences1)):
        keywords_found = keyword_processor.extract_keywords(selected_sentences1[i])
        if (len(keywords_found) > 0):
            education_yrs=4.0
   
    return education_yrs
get_eduYrs(qual)

# In[20]:


def get_expYrs(s):
    keyword_processor = KeywordProcessor()
    exp_yrs=0.0
    keyword_processor.add_keyword('four-year')
    keyword_processor.add_keyword('four years')
    keyword_processor.add_keyword('three years')
    keyword_processor.add_keyword('one year')
    keyword_processor.add_keyword('two years')
    keyword_processor.add_keyword('six years')
    sentences = sent_tokenize(s)
    selected_sentences=[sent for sent in sentences if "experience" in word_tokenize(sent)]

    for i in range(len(selected_sentences)):
        keywords_found = keyword_processor.extract_keywords(selected_sentences[i])
        for i in range(len(keywords_found)):
            if keywords_found[i]=='two years':
                exp_yrs=2.0
            elif keywords_found[i]=='one year':
                exp_yrs=1.0
            elif keywords_found[i]=='three years':
                exp_yrs=3.0
            elif keywords_found[i]=='six years':
                exp_yrs=6.0
            elif keywords_found[i]=='four years':
                exp_yrs=4.0
            elif keywords_found[i]=='four-year':
                exp_yrs=4.0
                
    return exp_yrs
get_expYrs(s)

# In[21]:


def get_fullTimePartTime(s):
    keyword_processor = KeywordProcessor()
    fullTimePartTime=""
    keyword_processor.add_keyword('full-time')
    keyword_processor.add_keyword('part-time')
    sentences = sent_tokenize(s)
    selected_sentences=[sent for sent in sentences if "experience" in word_tokenize(sent)]

    for i in range(len(selected_sentences)):
        keywords_found = keyword_processor.extract_keywords(selected_sentences[i])
        for i in range(len(keywords_found)):
            if keywords_found[i]=='full-time':
                fullTimePartTime="FULL TIME"
            elif keywords_found[i]=='part-time':
                fullTimePartTime="PART TIME"
           
                
    return fullTimePartTime
get_fullTimePartTime(qual)

# In[22]:


def get_DL(s):
    dl = False
    dl_valid = False
    dl_State = ""
    arr = ['driver', 'license']
    keyword_processor = KeywordProcessor()
    keyword_processor.add_keyword('california')
    if any(re.findall('|'.join(arr), qual)):
        dl = True
    if (dl==True):
        sentences = sent_tokenize(s)
        selected_sentence=[sent for sent in sentences if "driver" in word_tokenize(sent)]
        if (len(selected_sentence)>0):
            words = selected_sentence[0].split()
            selected_word = [word for word in words if "valid" in words]
            if len(selected_word)>0:
                dl_valid=True
        for i in range(len(selected_sentence)):   
            keywords_found = keyword_processor.extract_keywords(selected_sentence[i])
            for i in range(len(keywords_found)):
                if keywords_found[i]=='california':
                    dl_State="CA"
                
    if (dl_valid)==True:
        dl_valid="R"
    else:
        dl_valid="P"
    return dl_valid,dl_State
get_DL(qual)

# In[23]:


from __future__ import unicode_literals, print_function

import plac
import spacy

def filter_spans(spans):
    # Filter a sequence of spans so they don't contain overlaps
    get_sort_key = lambda span: (span.end - span.start, span.start)
    sorted_spans = sorted(spans, key=get_sort_key, reverse=True)
    result = []
    seen_tokens = set()
    for span in sorted_spans:
        if span.start not in seen_tokens and span.end - 1 not in seen_tokens:
            result.append(span)
            seen_tokens.update(range(span.start, span.end))
    return result


def extract_entity_relations(doc,entity):
    # Merge entities and noun chunks into one token
    seen_tokens = set()
    spans = list(doc.ents) + list(doc.noun_chunks)
    spans = filter_spans(spans)
    with doc.retokenize() as retokenizer:
        for span in spans:
            retokenizer.merge(span)

    relations = []
    for money in filter(lambda w: w.ent_type_ == entity, doc):
        if money.dep_ in ("attr", "dobj"):
            subject = [w for w in money.head.lefts if w.dep_ == "nsubj"]
            if subject:
                subject = subject[0]
                relations.append((subject, money))
        elif money.dep_ == "pobj" and money.head.dep_ == "prep":
            relations.append((money.head.head, money))
    return relations

# In[39]:


def get_Relations(TEXTS, nlp, ENTITY_TYPE):
    entities=[]
    for text in TEXTS:
        doc = nlp(text)
        relations = extract_entity_relations(doc,ENTITY_TYPE)
        for r1, r2 in relations:
            relation=r1.text+"-"+r2.text
            entities.append(relation)
    imp_entities='::::'.join(entities)   
    return imp_entities

nlp = spacy.load("en_core_web_sm")
#for i in range(200,250):
#    print(i)
#    s = job_data[i].replace("\\r\\n"," ").replace("\\t","")
#    qual = get_qualification(s)
#    imp_entities=get_Relations([qual],nlp,"QUANTITY") #PERSON #FAC #
#    print(imp_entities)
#print("{:<10}\t{}\t{}".format(r1.text, r2.ent_type_, r2.text))

# In[53]:


keyword_processorTitle = KeywordProcessor()
s = job_data[548].replace("\\r\\n"," ").replace("\\t","")
for title in title_text:
    title=title.strip()
    keyword_processorTitle.add_keyword(title)


# In[58]:


for i in range(500,510):
    s = job_data[i].replace("\\r\\n"," ").replace("\\t","")
    keywords_found = keyword_processorTitle.extract_keywords(s)
    print(i)
    print(get_position(s))
    if (len(keywords_found) > 0):
        for j in range(len(keywords_found) ):
            if (j>0):
                if ((keywords_found[j]!=get_position(s)) & (keywords_found[j]!=keywords_found[j-1])):
                    print("Previous position:")
                    print(keywords_found[j])
            else:
                if (keywords_found[j]!=get_position(s)):
                    print("Previous position:")
                    print(keywords_found[j])

# In[ ]:


job_data_export=pd.DataFrame(columns=["FILE_NAME","JOB_CLASS_TITLE","JOB_CLASS_NO","REQUIREMENT_SET_ID",
                                      "REQUIREMENT_SUBSET_ID","JOB_DUTIES",
                                      "EDUCATION_YEARS","SCHOOL_TYPE","EDUCATION_MAJOR","EXPERIENCE_LENGTH","IMP_ENTITIES_QUAL",
                                     "FULL_TIME_PART_TIME","EXP_JOB_CLASS_TITLE","EXP_JOB_CLASS_ALT_RESP"
                                     ,"EXP_JOB_CLASS_FUNCTION","COURSE_COUNT","COURSE_LENGTH","COURSE_SUBJECT"
                                     ,"MISC_COURSE_DETAILS","DRIVERS_LICENSE_REQ","DRIV_LIC_TYPE",
                                     "ADDTL_LIC","EXAM_TYPE","ENTRY_SALARY_GEN","ENTRY_SALARY_DWP","OPEN_DATE","LEGAL_TERMS"])


# In[ ]:


nlp = spacy.load("en_core_web_sm")

for i in range(100,110):
    s = job_data[i].replace("\\r\\n"," ").replace("\\t","")
    position = get_position(s)
    qual = get_qualification(s)
    DL_valid,DL_state = get_DL(qual)
    education_yrs = get_eduYrs(qual)
    education_major = get_educationMajor(qual)
    job_code = get_JobCode(s)
    openDate, openStr = get_OpenDate(s)
    salaryRange = get_SalaryRange(s)
    expYrs = get_expYrs(s)
    duties = get_Duties(s)
    course_length = get_eduSemDur(qual)
    fullTimePartTime = get_fullTimePartTime(qual)
    imp_qual_entities=get_Relations([qual],nlp,"ORG")
    imp_qual_cardinals=get_Relations([qual],nlp,"CARDINAL")
    imp_legal_terms=get_Relations([s],nlp,"LAW")
    college = get_college(qual)
    job_data_export.loc[i,"JOB_CLASS_TITLE"]=position
    job_data_export.loc[i,"FILE_NAME"]=file_names[i]
    job_data_export.loc[i,"DRIVERS_LICENSE_REQ"]=DL_valid
    job_data_export.loc[i,"EDUCATION_YEARS"]=education_yrs
    job_data_export.loc[i,"JOB_CLASS_NO"]=job_code
    job_data_export.loc[i,"OPEN_DATE"]=openDate
    job_data_export.loc[i,"ENTRY_SALARY_GEN"]=salaryRange
    job_data_export.loc[i,"JOB_DUTIES"]=duties
    job_data_export.loc[i,"EXPERIENCE_LENGTH"]=expYrs
    job_data_export.loc[i,"DRIV_LIC_TYPE"]=DL_state
    job_data_export.loc[i,"EDUCATION_MAJOR"]=education_major
    job_data_export.loc[i,"IMP_ENTITIES_QUAL"]=imp_qual_entities
    job_data_export.loc[i,"COURSE_LENGTH"]=course_length
    job_data_export.loc[i,"FULL_TIME_PART_TIME"]=fullTimePartTime
    job_data_export.loc[i,"SCHOOL_TYPE"]=college
    job_data_export.loc[i,"MISC_COURSE_DETAILS"]=imp_qual_cardinals
    job_data_export.loc[i,"LEGAL_TERMS"]=imp_legal_terms
    job_data_export.loc[i,"EXAM_TYPE"]=openStr
    
job_data_export.head()

# In[ ]:


job_data_export.to_csv("job_dictionary.csv",index=False)

# **OBSERVATIONS**
# * 50% of the job ads are above 733 words (median) in length
# * The bi-grams like 'meet minimum requirements' , 'minimum qualifications met' are a bit formal, and appearing in the most frequent n-gram list. 
# * There are excellent inclusive n-grams occuring in the most frequent list, ex: 'equal employment opportunity', 'disability accommodation form','american disabilities act'
# * There is presence of n-grams like 'marital status sexual orientation'
# * Complex qualification criteria , ex: 'Only applicants that are currently or have previously worked in the Department of Water and Power in DDR Numbers 93-39121, 93-39137, 93-39120, 93-39135, 93-39119, 93-39138, 93-39010, 93-39126, 93-39002, 93-39023, 93-39026, 42-39301, or 93-39130, and meet the above noted requirement qualify to take this examination.'
# * Some overly challenging statements, negative tone, ex: 'Applicants who fail to complete the Qualifications Questionnaire', 'Candidates in the examination process may file protests as provided in Sec.''This is an extremely competitive examination and a sufficient number of candidates with the highest scores will continue in the selection process.'
# * Dept name included in Job Title, ex: AIRPORT CHIEF INFORMATION SECURITY OFFICER
# 
# 
# **RECOMMENDATIONS**
# * Make the job bulletins more compact (max 250 words) to improve readability, attract talents
# * Make the job bulletins less formal with a inviting tone, instead of heavy emphasis on minimum requirements
# * Add inclusive policy on sexual orientation, marital status
# * Simplify the qualification criteria, OR have a expert panel judge a candidate based on basic qualifications
# * Remove negative tone, like 'Applicants who fail to complete.. will not be considered' to 'Applications who complete..will be taken to next round'. Remove words like 'Protests are allowed'.
# * Do not include dept name in Job Title, ex: Dept : Airport Title : CHIEF INFORMATION SECURITY OFFICER
# 
# **DATA DICTIONARY**
# * "JOB_CLASS_TITLE" = Job title 
# * "FILE_NAME" = File name
# * "DRIVERS_LICENSE_REQ" = R/P 
# * "EDUCATION_YEARS" = Education years
# * "JOB_CLASS_NO"= 
# * "OPEN_DATE" = open Date of job application
# * "ENTRY_SALARY_GEN" = salary Range
# * "JOB_DUTIES"=duties
# * "EXPERIENCE_LENGTH"=experience Yrs needed
# * "DRIV_LIC_TYPE"=DL State
# * "EDUCATION_MAJOR"=education major
# * "IMP_ENTITIES_QUAL"=Important ORG (Organization) entities in qualification
# * "COURSE_LENGTH" =course length
# * "FULL_TIME_PART_TIME" =fullTime or PartTime experience needed
# * "SCHOOL_TYPE" =College or University
# * "MISC_COURSE_DETAILS"= Important numbers in Qualification
# * "LEGAL_TERMS"=Important legal terms in qualifications
# * "EXAM_TYPE"=Exam open to All etc

# In[ ]:



