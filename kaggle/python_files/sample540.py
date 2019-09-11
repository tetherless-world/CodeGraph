#!/usr/bin/env python
# coding: utf-8

# ## Table of Content  
# 1. [Introduction](#introduction)
# 2. [Preparation](#preparation)
# 3. [Data Extraction](#data_extraction)   
# 4. [EDA](#eda)

# ## 1. Introduction <a id="introduction"></a> 

# ## 2. Preparations <a id="preparation"></a>

# ### Libraries

# In[ ]:


import os
import re
import math
import random
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import spacy
nlp = spacy.load('en')
nlp.remove_pipe('parser')
nlp.remove_pipe('ner')

from wordcloud import WordCloud
from collections import Counter

# ### Global Parameters

# In[ ]:


warnings.filterwarnings('ignore')
#pd.set_option('display.max_colwidth', -1)

SEED = 13
random.seed(SEED)
np.random.seed(SEED)

# ### Load Data

# #### Job Titles

# In[ ]:


file_job_titles = '../input/cityofla/CityofLA/Additional data/job_titles.csv'
job_titles = pd.read_csv(file_job_titles, header=None, names=['job_title'])

# #### Kaggle Data Dictionary

# In[ ]:


file_data_dic = '../input/cityofla/CityofLA/Additional data/kaggle_data_dictionary.csv'
data_dictionary = pd.read_csv(file_data_dic)

# #### Job Bulletins

# In[ ]:


dir_job_bulletins = '../input/cityofla/CityofLA/Job Bulletins'
data_list = []
for filename in os.listdir(dir_job_bulletins):
    with open(os.path.join(dir_job_bulletins, filename), 'r', errors='ignore') as f:
        data_list.append([filename, ''.join(f.readlines())])
jobs = pd.DataFrame(data_list, columns=['file', 'job_description'])

# For now, we will remove the file `'Vocational Worker  DEPARTMENT OF PUBLIC WORKS.txt'`, because it contains a completely different format.

# In[ ]:


jobs = jobs[jobs['file'] != 'Vocational Worker  DEPARTMENT OF PUBLIC WORKS.txt']

# ## 3. Data Extraction (Job Bulletins) <a id="data_extraction"></a>  
# We will use regular expressions to extract the relevant information.  
# We will first divide the text into general parts (metadata, salary, ...) and then extract details from them (metadata -> job title, class code, open date, ...).

# In[ ]:


def merge_jobs_data(jobs, extracted_data):
    """ Add the extracted_data to the current jobs DataFrame

        param jobs: Current jobs DataFrame
        param extracted_data: Series with DataFrame inside to extract
        return jobs: Merged DataFrame
    """ 
    jobs['temp'] = extracted_data
    for index, row in jobs.iterrows():
        extracted_data = row['temp']
        if isinstance(extracted_data, pd.DataFrame):
            for c in extracted_data.columns:
                jobs.loc[index, c] = extracted_data[c][0]
    jobs = jobs.drop('temp', axis=1) 
    return jobs

def extract_text_by_regex(text, regex_dictionary):
    """ Extract values by regular expressions

        param text: String to extract the values
        param regex_dictionary: Dictionary with the names and regular expressions to extract
        return result: Series with the first extracted values
    """ 
    regex_dictionary = pd.DataFrame(regex_dictionary, columns=['name', 'regexpr'])
    result = regex_dictionary.copy()
    result['text'] = np.NaN
    for index,row in regex_dictionary.iterrows():
        find_reg = re.findall(row['regexpr'], text, re.DOTALL)
        extracted_text = find_reg[0].strip() if find_reg else np.NaN
        result.loc[index, 'text'] = extracted_text
    return result.set_index('name')[['text']].T 

def extract_text_by_regex_index(text, regex_dictionary):
    """ Extract values by regular expressions
    
        Search for the index of the first occurrence of the regular expression 
        and extract the text to the next regular expression.

        param text: String to extract the values
        param regex_dictionary: Dictionary with the names and regular expressions to extract
        return result: Series with the first extracted values
    """ 
    regex_dictionary = pd.DataFrame(regex_dictionary, columns=['name', 'regexpr'])

    result = regex_dictionary.copy()
    result['text'] = np.NaN
    for index,row in regex_dictionary.iterrows():
        find_text = re.search(row['regexpr'], text)
        find_text = find_text.span(0)[0] if find_text else np.nan
        result.loc[index, 'start'] = find_text
    result.dropna(subset=['start'], inplace=True)
    result['end'] = result['start'].apply(lambda x: np.min(result[result['start'] > x]['start'])).fillna(len(text))
    
    for index,row in result.iterrows():
        extracted_text = text[int(row['start']):int(row['end'])]
        find_reg = re.findall(row['regexpr']+'(.*)', extracted_text, re.DOTALL)
        extracted_text = find_reg[0].strip() if find_reg else np.NaN
        result.loc[index, 'text'] = extracted_text
    return result.set_index('name')[['text']].T 

def nlp_transformation(data, token_pos=None):
    """ Use NLP to transform the text corpus to cleaned sentences and word tokens

        param data: List with sentences, which should be processed.
        param token_pos: List with the POS-Tags to filter (Default: None = All POS-Tags)
        return processed_tokens: List with the cleaned and tokenized sentences
    """    
    def token_filter(token):
        """ Keep tokens who are alphapetic, in the pos (part-of-speech) list and not in stop list
            
        """    
        if token_pos:
            return not token.is_stop and token.is_alpha and token.pos_ in token_pos
        else:
            return not token.is_stop and token.is_alpha
    
    data = [re.compile(r'<[^>]+>').sub('', x) for x in data] #Remove HTML-tags
    processed_tokens = []
    data_pipe = nlp.pipe(data)
    for doc in data_pipe:
        filtered_tokens = [token.lemma_.lower() for token in doc if token_filter(token)]
        processed_tokens.append(filtered_tokens)
    return processed_tokens

# ### Upper Sections  
# * Metadata
# * Salary
# * Duties
# * Requirements
# * Where to apply
# * Application deadline
# * Selection Process
# 

# In[ ]:


regex_dictionary = [('metadata', r''), 
                      ('salary', r'(?:ANNUAL SALARY|ANNUALSALARY)'),
                      ('duties', r'(?:DUTIES)'),
                      ('requirements', r'(?:REQUIREMENTS/MINIMUM QUALIFICATIONS|REQUIREMENT/MINIMUM QUALIFICATION|REQUIREMENT|REQUIREMENTS|REQUIREMENT/MIMINUMUM QUALIFICATION)'),
                      ('where_to_apply', r'(?:WHERE TO APPLY|HOW TO APPLY)'),
                      ('application_deadline', r'(?:APPLICATION DEADLINE|APPLICATION PROCESS)'),
                      ('selection_process', r'(?:SELECTION PROCESS|SELELCTION PROCESS)'),
                      ]
extracted_data = jobs['job_description'].dropna().apply(lambda x: extract_text_by_regex_index(x, regex_dictionary))
jobs = merge_jobs_data(jobs, extracted_data)

# ### Metadata  
# * Job title
# * Class code
# * Open Date
# * Revised

# In[ ]:


regex_dictionary = [('job_title', r'(.*?)(?=\n)'), 
                      ('class_code', r'(?:Class Code:|Class  Code:)\s*(\d\d\d\d)'),
                      ('open_date', r'(?:Open Date:|Open date:)\s*(\d\d-\d\d-\d\d)'),
                      ('revised', r'(?:Revised:|Revised|REVISED:)\s*(\d\d-\d\d-\d\d)')
                      ]
extracted_data = jobs['metadata'].dropna().apply(lambda x: extract_text_by_regex(x, regex_dictionary))
jobs = merge_jobs_data(jobs, extracted_data)
jobs['open_date'] = pd.to_datetime(jobs['open_date'], infer_datetime_format=True)
jobs['revised'] = pd.to_datetime(jobs['revised'], infer_datetime_format=True)

# ### Salary  
# * Salary from
# * Salary to
# * Flat-rated
# * Additional informations
# * Notes

# In[ ]:


regex_dictionary = [('salary_from', r'\$((?:\d{1,3})(?:\,\d{3})*(?:\.\d{2})*).*'), 
                      ('salary_to', r'(?:and|to) \$((?:\d{1,3})(?:\,\d{3})*(?:\.\d{2})*).*'),
                      ('salary_flatrated', r'(flat-rated|Flat-Rated)'),
                      ('salary_additional', r'(?:\n)(.*)(?:NOTES)'),
                      ('salary_notes', r'(?:NOTES:)(.*)'),
                      ]
extracted_data = jobs['salary'].dropna().apply(lambda x: extract_text_by_regex(x, regex_dictionary))
jobs = merge_jobs_data(jobs, extracted_data)
jobs['salary_from'] = jobs['salary_from'].dropna().apply(lambda x: float(x.replace(',', '')))
jobs['salary_to'] = jobs['salary_to'].dropna().apply(lambda x: float(x.replace(',', '')))
jobs['salary_flatrated'] = jobs['salary_flatrated'].dropna().apply(lambda x: True)

# ### Duties
# * Text
# * Notes

# In[ ]:


regex_dictionary = [('duties_text', r''), 
                      ('duties_notes', r'(?:NOTE:|NOTES:)'),
                      ]
extracted_data = jobs['duties'].dropna().apply(lambda x: extract_text_by_regex_index(x, regex_dictionary))
jobs = merge_jobs_data(jobs, extracted_data)

# ### Requirements
# * Text
# * Notes
# * Certifications

# In[ ]:


regex_dictionary = [('requirements_text', r''), 
                         ('requirements_notes', r'(?:PROCESS NOTES|NOTES:|NOTE:|PROCESS NOTE)'),
                         ('requirements_certifications', r'(?:SELECTIVE CERTIFICATION|SELECTIVE CERTIFICATION:)'),
                      ]
extracted_data = jobs['requirements'].dropna().apply(lambda x: extract_text_by_regex_index(x, regex_dictionary))
jobs = merge_jobs_data(jobs, extracted_data)

# ### Where to apply
# * Text
# * Notes

# In[ ]:


regex_dictionary = [('where_to_apply_text', r''), 
                         ('where_to_apply_notes', r'(?:NOTE:)'),
                      ]
extracted_data = jobs['where_to_apply'].dropna().apply(lambda x: extract_text_by_regex_index(x, regex_dictionary))
jobs = merge_jobs_data(jobs, extracted_data)

# ### Application deadline
# * Text
# * Notes
# * Review

# In[ ]:


regex_dictionary = [('application_deadline_text', r''), 
                         ('application_deadline_notes', r'(?:NOTE:)'),
                         ('application_deadline_review', r'(?:QUALIFICATIONS REVIEW|EXPERT REVIEW COMMITTEE)'),
                      ]
extracted_data = jobs['application_deadline'].dropna().apply(lambda x: extract_text_by_regex_index(x, regex_dictionary))
jobs = merge_jobs_data(jobs, extracted_data)

# ### Selection process
# * Text
# * Notes
# * Notice

# In[ ]:


regex_dictionary = [('selection_process_text', r''), 
                         ('selection_process_notes', r'(?:NOTES:)'),
                         ('selection_process_notice', r'(?:NOTICE:|Notice:)'),
                      ]
extracted_data = jobs['selection_process'].dropna().apply(lambda x: extract_text_by_regex_index(x, regex_dictionary))
jobs = merge_jobs_data(jobs, extracted_data)

# ### Other Details  
# **Exam Type**

# In[ ]:


def get_exam_type(text):
    """ Extract the exam type from the text
    
        1. Open or Competitive Interdepartmental Promotional (OPEN_INT_PROM)
        2. Interdepartmental Promotional (INT_DEPT_PROM)
        3. Departmental Promotional (DEPT_PROM)
        4. Exam open to anyone (OPEN)
        5. Else Null

        param text: String to extract the values
        return result: String with the exam_type code
    """ 
    regex_OPEN_INT_PROM = 'BOTH.*INTERDEPARTMENTAL.*PROMOTIONAL'
    result_OPEN_INT_PROM = re.findall(regex_OPEN_INT_PROM, text, re.DOTALL) 
    regex_INT_DEPT_PROM = 'INTERDEPARTMENTAL.*PROMOTIONAL'
    result_INT_DEPT_PROM = re.findall(regex_INT_DEPT_PROM, text, re.DOTALL) 
    regex_DEPT_PROM = 'DEPARTMENTAL.*PROMOTIONAL'
    result_DEPT_PROM = re.findall(regex_DEPT_PROM, text, re.DOTALL) 
    regex_OPEN = 'OPEN.*COMPETITIVE.*BASIS'
    result_OPEN = re.findall(regex_OPEN, text, re.DOTALL) 

    if result_OPEN_INT_PROM:
        #result = 'Open or Competitive Interdepartmental Promotional'
        result = 'OPEN_INT_PROM'
    elif result_INT_DEPT_PROM:
        #result = 'Interdepartmental Promotional'
        result = 'INT_DEPT_PROM'
    elif result_DEPT_PROM:
        #result = 'Departmental Promotional'
        result = 'DEPT_PROM'
    elif result_OPEN:
        #result = 'Exam open to anyone'
        result = 'OPEN'
    else:
        result = np.nan
    return result

jobs['exam_type'] = jobs['selection_process'].dropna().apply(get_exam_type)

# **Driver License**

# In[ ]:


def get_driver_license(text):
    """ Extract the driver license from the text
    
        1. Possible (P)
        2. Required (R)
        3. Else Null

        param text: String to extract the values
        return result: String if a driver license is needed
    """ 
    regex_Possible = r'(may[^\.]*requir[^\.]*driver[^\.]*license)'
    result_Possible = re.findall(regex_Possible, text, re.IGNORECASE) 
    regex_Required = r'(requir[^\.]*driver[^\.]*license)|(driver[^\.]*license[^\.]*requir)'
    result_Required = re.findall(regex_Required, text, re.IGNORECASE) 

    if result_Possible:
        #result = 'Possible'
        result = 'P'
    elif result_Required:
        #result = 'Required'
        result = 'R'
    else:
        result = np.nan
    return result

jobs['driver_license'] = jobs['job_description'].dropna().apply(get_driver_license)

# ### Remove unused columns

# In[ ]:


drop_cols = ['job_description', 'metadata', 'salary', 'duties', 
        'requirements', 'where_to_apply', 
        'application_deadline', 'selection_process']
#jobs = jobs.drop(drop_cols, axis=1) 

# ### Print Example  
# Let's have a look on an example to see which information has been extracted.  
# (Expand the `output` to see the result)

# In[ ]:


example = jobs[jobs['file'] == 'SYSTEMS ANALYST 1596 102717.txt'].drop(drop_cols, axis=1).iloc[0,:]
for idx in example.index:
    print('\033[42m'+idx+':'+'\033[0m')
    print(example[idx])

# ## 4. EDA <a id="eda"></a>

# ### Missing Values  
# Here we can check where data is available and which information is less frequently available.  
# Since certain information needs to be checked and the data extraction may need to be adjusted, this plot may still change.

# In[ ]:


temp = jobs.fillna('Missing')
temp = temp.applymap(lambda x: x if x == 'Missing' else 'Available')
figsize_width = 12
figsize_height = len(temp.columns)*0.5
plt_data = pd.DataFrame()
for col in temp.columns:
    temp_col = temp.groupby(col).size()/len(temp.index)
    temp_col = pd.DataFrame({col:temp_col})
    plt_data = pd.concat([plt_data, temp_col], axis=1)
    
ax = plt_data.T.plot(kind='barh', stacked=True, figsize=(figsize_width, figsize_height))

# Annotations
labels = []
for i in plt_data.index:
    for j in plt_data.columns:
        label = '{:.2%}'.format(plt_data.loc[i][j])
        labels.append(label)
patches = ax.patches
for label, rect in zip(labels, patches):
    width = rect.get_width()
    if width > 0:
        x = rect.get_x()
        y = rect.get_y()
        height = rect.get_height()
        ax.text(x + width/2., y + height/2., label, ha='center', va='center')

plt.xlabel('Frequency')
plt.title('Missing values')
plt.xticks(np.arange(0, 1.05, 0.1))
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()

# ### Open Date  
# This plot should show when new jobs should be occupied.

# In[ ]:


plt_data = (jobs.groupby(jobs['open_date'].dropna().dt.strftime('%Y-%m')).size())
plt_data = pd.DataFrame(plt_data)
plt_data.plot(kind='bar', figsize=(15, 5))
plt.xlabel('Month')
plt.ylabel('Quantity')
plt.title('Distribution over time')
plt.legend('')
plt.show()

# ### Salary  
# Here the distribution of the salary can be checked.  
# 50% of jobs will earn at least \$80,000.  
# The maximum annual salary is currently \$280,000.

# In[ ]:


plt_data = jobs[['salary_from', 'salary_to']]
plt_data.plot(kind='box', showfliers=True, vert=False, figsize=(12, 3), grid=True)
plt.xticks(range(0, 300001, 25000))
plt.xlabel('Salary')
plt.title('Salary Distribution')
plt.show()
plt_data.describe()

# In[ ]:


plt_data = jobs[['salary_from', 'salary_to']]
plt_data.plot(kind='hist', bins=1000, density=True, histtype='step', cumulative=True, figsize=(15, 7), lw=2, grid=True)
plt.xlabel('Salary')
plt.ylabel('Cumulative')
plt.title('Cumulative histogram for salary')
plt.legend(loc='upper left')
plt.xlim([25000, 200000])
plt.xticks(range(25000, 200001, 10000))
plt.yticks(np.arange(0, 1.05, 0.05))
plt.show()

# ### Wordcount

# In[ ]:


plt_data_duties = jobs['duties_text'].astype(str).apply(lambda x: len(x.split()))
plt_data_requirements = jobs['requirements_text'].astype(str).apply(lambda x: len(x.split()))
plt_data = pd.DataFrame([plt_data_duties, plt_data_requirements]).T

plt_data.plot(kind='box', showfliers=False, vert=False, figsize=(12, 3), grid=True)
plt.xticks(range(0, 201, 10))
plt.xlabel('Words')
plt.title('Word count')
plt.show()

# ### Exam Type  
# OPEN_INT_PROM = 'Open or Competitive Interdepartmental Promotional'  
# INT_DEPT_PROM = 'Interdepartmental Promotional'  
# DEPT_PROM = 'Departmental Promotional'  
# OPEN = 'Exam open to anyone'  
# None = Not defined

# In[ ]:


plt_data = jobs['exam_type'].fillna('None')
plt_data = plt_data.groupby(plt_data).size()
plt_data.plot(kind='pie', figsize=(10, 5), autopct='%.2f')
plt.title('Exam Type')
plt.ylabel('')
plt.show()

# ### Driver License 
# P = 'Posible'  
# R = 'Required'  
# None = Not defined

# In[ ]:


plt_data = jobs['driver_license'].fillna('None')
plt_data = plt_data.groupby(plt_data).size()
plt_data.plot(kind='pie', figsize=(10, 5), autopct='%.2f')
plt.title('Driver License')
plt.ylabel('')
plt.show()

# ### Wordcloud (Duties)

# In[ ]:


plt_data = nlp_transformation(jobs['duties_text'].dropna(), ['VERB'])
plt_data = [j for i in plt_data for j in i]
plt_data=Counter(plt_data)
plt.figure(figsize=(10, 10))

wordcloud = WordCloud(margin=0, random_state=SEED).generate_from_frequencies(plt_data)
plt.imshow(wordcloud, interpolation='bilinear')
plt.title('Duties (Verbs)')
plt.axis("off")
plt.show() 

# ### Wordcloud (Requirements)

# In[ ]:


plt_data = nlp_transformation(jobs['requirements_text'].dropna(), ['NOUN', 'VERB', 'PROPN', 'ADJ'])
plt_data = [j for i in plt_data for j in i]
plt_data=Counter(plt_data)
plt.figure(figsize=(10, 10))

wordcloud = WordCloud(margin=0, random_state=SEED).generate_from_frequencies(plt_data)
plt.imshow(wordcloud, interpolation='bilinear')
plt.title('Requirements')
plt.axis("off")
plt.show() 
