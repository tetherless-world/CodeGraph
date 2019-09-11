#!/usr/bin/env python
# coding: utf-8

# ## Data Science for Good: City of Los Angeles

# ![](https://media.giphy.com/media/3XAU2dw8fjghZmsRZd/giphy.gif)

# ### Problem statement

# The goal is to convert a folder full of plain-text job postings into a single structured CSV file and then to use this data to:
# 1.  identify language that can negatively bias the pool of applicants; 
# 2.  improve the diversity and quality of the applicant pool; and/or 
# 3.  make it easier to determine which promotions are available to employees in each job class.

# In[ ]:


from wand.image import Image as Img
Img(filename='../input/cityofla/CityofLA/Additional data/PDFs/2017/july 2017/July 21/ARTS ASSOCIATE 2454 072117 REV 072817.pdf', resolution=300)

# ## What is in this kernel?

# 1. As my first step toward solving the problem,i will focus on reading the job bulletin files and extracting 
#     the required data from it.
# 2. Exploratory data analysis
# 3. Suggestions to improve diversity.
# - NOTE : kernel under construction,more to come !

# ## Importing Libraries

# In[ ]:




import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re
import os
import numpy as np
from datetime import datetime
from collections  import Counter
from nltk import word_tokenize
import seaborn as sns
import matplotlib.pyplot as plt
import calendar
from wordcloud import WordCloud ,STOPWORDS
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
print(os.listdir("../input"))
from gensim.models import word2vec
from sklearn.manifold import TSNE
from nltk import pos_tag
from nltk.help import upenn_tagset
import gensim
import matplotlib.colors as mcolors
plt.style.use('ggplot')

# ### checking all subdirectories

# In[ ]:


files=[dir for dir in os.walk('../input/cityofla')]
for file in files:
    print(os.listdir(file[0]))
    print("\n")

# - job bulletins :This directory contains the job bulletins in text format.
# - additional data :This directory contains additional data in pdf and csv format. 

# In[ ]:


bulletins=os.listdir("../input/cityofla/CityofLA/Job Bulletins/")
additional=os.listdir("../input/cityofla/CityofLA/Additional data/")

# We will find and print all the files inside **Additional data** that has csv format.

# In[ ]:


csvfiles=[]
for file in additional:
    if file.endswith('.csv'):
        print(file)
        csvfiles.append("../input/cityofla/CityofLA/Additional data/"+file)
        

# There are 3 comma seperated files inside the folder,
# 1. job titles : contains the title given to different jobs available.
# 2. sample job class export template.csv : contains sample job bulletin to csv export details.
# 3. kaggle_data_dictionary : contains name and description of each column that is in sample job class export template.

# ## Reading the required csv files

# In[ ]:


job_title=pd.read_csv(csvfiles[0])
sample_job=pd.read_csv(csvfiles[1])
kaggle_data=pd.read_csv(csvfiles[2])

# ## Getting  basic ideas

# In the below section we will take a look at the three csv files which was just loaded to get basic understanding of the data.

# In[ ]:


job_title.head()

# In[ ]:


print("The are %d rows and %d cols in job_title file" %(job_title.shape))

# In[ ]:


sample_job.head()

# In[ ]:


print("The are %d rows and %d cols in sample_job file" %(sample_job.shape))


# In[ ]:


kaggle_data.head()

# In[ ]:


print("The are %d rows and %d cols in kaggle_data file" %(kaggle_data.shape))

# In[ ]:


print("There are %d text files in bulletin directory" %len(bulletins))

# #### Extracting the headings from job bulletins

# In[ ]:


def get_headings(bulletin):       
    
    """"function to get the headings from text file
        takes a single argument
        1.takes single argument list of bulletin files"""
    
    with open("../input/cityofla/CityofLA/Job Bulletins/"+bulletins[bulletin]) as f:    ##reading text files 
        data=f.read().replace('\t','').split('\n')
        data=[head for head in data if head.isupper()]
        return data
        
def clean_text(bulletin):      
    
    
    """function to do basic data cleaning
        takes a single argument
        1.takes single argument list of bulletin files"""
                                            
    
    with open("../input/cityofla/CityofLA/Job Bulletins/"+bulletins[bulletin]) as f:
        data=f.read().replace('\t','').replace('\n','')
        return data

# #### Headings

# We will print headings from first two job bulletins file.

# In[ ]:


get_headings(1)


# In[ ]:


get_headings(2)

# - we can observe that there is some specefic pattern or format which is kept while writing job bulletins.
# - The order of the headings almost coincides with each other,which will be beneficial for our task.

# ## Extracting features

# In[ ]:





def to_dataframe(num,df):
    """"function to extract features from job bulletin text files and convert to
    pandas dataframe.
    function take two arguments 
                        1.the number of files to be read
                        2.dataframe object                                      """
    

    
    opendate=re.compile(r'(Open [D,d]ate:)(\s+)(\d\d-\d\d-\d\d)')       #match open date
    
    salary=re.compile(r'\$(\d+,\d+)((\s(to|and)\s)(\$\d+,\d+))?')       #match salary
    
    requirements=re.compile(r'(REQUIREMENTS?/\s?MINIMUM QUALIFICATIONS?)(.*)(PROCESS NOTE)')      #match requirements
    
    for no in range(0,num):
        with open("../input/cityofla/CityofLA/Job Bulletins/"+bulletins[no],encoding="ISO-8859-1") as f:         #reading files 
                try:
                    file=f.read().replace('\t','')
                    data=file.replace('\n','')
                    headings=[heading for heading in file.split('\n') if heading.isupper()]             ##getting heading from job bulletin

                    sal=re.search(salary,data)
                    date=datetime.strptime(re.search(opendate,data).group(3),'%m-%d-%y')
                    try:
                        req=re.search(requirements,data).group(2)
                    except Exception as e:
                        req=re.search('(.*)NOTES?',re.findall(r'(REQUIREMENTS?)(.*)(NOTES?)',
                                                              data)[0][1][:1200]).group(1)
                    
                    duties=re.search(r'(DUTIES)(.*)(REQ[A-Z])',data).group(2)
                    try:
                        enddate=re.search(
                                r'(JANUARY|FEBRUARY|MARCH|APRIL|MAY|JUNE|JULY|AUGUST|SEPTEMBER|OCTOBER|NOVEMBER|DECEMBER)\s(\d{1,2},\s\d{4})'
                                ,data).group()
                    except Exception as e:
                        enddate=np.nan
                    
                    selection= [z[0] for z in re.findall('([A-Z][a-z]+)((\s\.\s)+)',data)]     ##match selection criteria
                    
                    df=df.append({'File Name':bulletins[no],'Position':headings[0].lower(),'salary_start':sal.group(1),
                               'salary_end':sal.group(5),"opendate":date,"requirements":req,'duties':duties,
                                'deadline':enddate,'selection':selection},ignore_index=True)
                    
                except Exception as e:
                    print('umatched sequence')
                    
                    
                
                
        
           
    return df

            
            
            
            

# In[ ]:


df=pd.DataFrame(columns=['File Name','Position','salary_start','salary_end','opendate','requirements','duties','deadline'])
df=to_dataframe(len(bulletins),df)

# In[ ]:


df.shape

# In[ ]:


df.head()

#  Now we have a proper comma seperataed file containing most of the information we need.we will now start exploring it. 

# Here,
# - **Position** implies JOB_CLASS_TITLE
# - **Duties** implies JOB_DUTIES
# - **Requirements** : All of the job requirements posted under requirements section
# - **Salary** has been broken down to **Salary_start** and **Salary_end** which signifies the salary range.
#     for example $ 5000 to $ 6000 
#         salary_start = 5000
#         salary_end = 6000
# - **Opendate** : The job application open date
# - **deadline** : deadline for applying for the job.
# - **selection** : The list of  selection criterias.

# 

# ## which are the common job sectors in LA?

# In[ ]:


print('There are %d different jobs available' %df['Position'].nunique())

# In[ ]:


plt.figure(figsize=(8,5))
text=''.join(job for job in df['Position'])                                ##joining  data to form text
text=word_tokenize(text)
jobs=Counter(text)                                                         ##counting number of occurences
jobs_class=[job for job in jobs.most_common(12) if len(job[0])>3]          ##selecting most common words
#offers=[job[1] for job in jobs.most_common(12) if len(job[0]>3)]
a,b=map(list, zip(*jobs_class))
sns.barplot(b,a,palette='rocket')                                           ##creating barplot
plt.title('Job sectors')
plt.xlabel("count")
plt.ylabel('sector')


# - We can see that **service sector** dominates in creating opputunities.

# - It might be a good idea to have **different display boards** for service and product sectors,so that candidates can easily find job of their preference.

# In[ ]:


""""
    convert salary to proper  form 
    by removing '$' and ',' symbols.
                                    """

df['salary_start']=[int(sal.split(',')[0]+sal.split(',')[1] ) for sal in df['salary_start']]   
df['salary_end']=[sal.replace('$','')  if sal!= None else 0 for sal in df['salary_end']  ]
df['salary_end']=[int(sal.split(',')[0]+sal.split(',')[1] ) if type(sal)!=int else 0 for sal in df['salary_end']]

# ### What about the salary distribution?

# In[ ]:


plt.figure(figsize=(7,5))
sns.distplot(df['salary_start'])
plt.title('salary distribution')
plt.show()

# - we can observe that thet salaries typically varies from $50k to $150k.
# - Most jobs salary start from $80000.

# ## Which are the best paid jobs in LA?

# In[ ]:


'''finding the most paid 10 jobs at LA'''

most_paid=df[['Position','salary_start']].sort_values(by='salary_start',ascending=False)[:10]
plt.figure(figsize=(7,5))
sns.barplot(y=most_paid['Position'],x=most_paid['salary_start'],palette='rocket')
plt.title('Best paid jobs in LA')

# ## Which the jobs with highest salary deviation?

# In[ ]:


''''calculating salary start - salary end '''

df['salary_diff']=abs(df['salary_start']-df['salary_end'])

ranges=df[['Position','salary_diff']].sort_values(by='salary_diff',ascending=False)[:10]

# In[ ]:


plt.figure(figsize=(7,5))
sns.barplot(y=ranges['Position'],x=ranges['salary_diff'],palette='RdBu')   ##plotting



# In[ ]:


ranges

# ## Has job opportunities really increased recently?

# In[ ]:


'''Extracting year out of opendate timestamp object and counting
    the number of each occurence of each year using count_values() '''

df['year_of_open']=[date.year for date in df['opendate']]

count=df['year_of_open'].value_counts(ascending=True)
years=['2020','2019','2018', '2017', '2016', '2015', '2014', '2013', '2012', '2008', '2006',
           '2005', '2002', '1999']
plt.figure(figsize=(7,5))
plt.plot([z for z in reversed(years)],count.values,color='blue')

plt.title('Oppurtunities over years')
plt.xlabel('years')
plt.ylabel('count')
plt.gca().set_xticklabels([z for z in reversed(years)],rotation='45')
plt.show()

# - It is evident from the above graph that job oppurtunities is constantly increasing after 2012 or so. 
# - job oppurtunities has never decreased.

# ## Which month of the year offers most opportunities?

# In[ ]:


'''Extracting month out of opendate timestamp object and counting
    the number of each occurence of each months using count_values() '''


plt.figure(figsize=(7,5))
df['open_month']=[z.month for z in df['opendate']]
count=df['open_month'].value_counts(sort=False)
sns.barplot(y=count.values,x=count.index,palette='rocket')
plt.gca().set_xticklabels([calendar.month_name[x] for x in count.index],rotation='45')
plt.show()

# We can see that there is more job opportunities created in the months of **March,October and December**

# * - It  might be a good idea to declare these months as **hiring months** and this could even attract a lot of potential candidates.

# ## Which day of the week ?

# In[ ]:


'''Extracting weekday out of opendate timestamp object and counting
    the number of each occurence of each weekday using count_values() '''


plt.figure(figsize=(7,5))

df['open_day']=[z.weekday() for z in df['opendate']]
count=df['open_day'].value_counts(sort=False)
sns.barplot(y=count.values,x=count.index,palette='rocket')
plt.gca().set_xticklabels([calendar.day_name[x] for x in count.index],rotation='45')
plt.show()

# Wow ! All the postings are open from **friday** ! This is pretty interesting.
# - Is there any specific reason behind this?
# We will try to find out.
# anyway its good to post all opening in a day  of week so  that candidates can refer the bulletins and apply without missing any openings.

# ### what about deadlines ?

# In[ ]:


print('%d job applications may close without prior notice' %df['deadline'].isna().sum())


# - I think employers should post a deadline for job application in the job bulletins to avoid any unwanted confusions between candidates.

# In[ ]:


#df['dealine']=df['deadline'].fillna(method='backfill',inplace=True)
#df['deadline']=[datetime.strptime(x,'%B %d, %Y')  for x in df['deadline'] ]


# In[ ]:


req=' '.join(text for text in df['requirements'])


# ## Word Cloud of Requirements

# In[ ]:



def show_wordcloud(data, title = None):
    
    
    '''funtion to produce and display wordcloud
        taken 2 arguments
        1.data to produce wordcloud
        2.title of wordcloud'''
    
    
    wordcloud = WordCloud(
        background_color='white',
        stopwords=set(STOPWORDS),
        max_words=250,
        max_font_size=40, 
        scale=3,
        random_state=1 # chosen at random by flipping a coin; it was heads
    ).generate(str(data))

    fig = plt.figure(1, figsize=(12, 12))
    plt.axis('off')
    if title: 
        fig.suptitle(title, fontsize=20)
        fig.subplots_adjust(top=2.3)

    plt.imshow(wordcloud)
    plt.show()
show_wordcloud(text,'REQUIREMENTS')

# ### Most influential words in requirements

# In[ ]:


lem=WordNetLemmatizer()
text=[lem.lemmatize(w) for w in word_tokenize(req)]
vect=TfidfVectorizer(ngram_range=(1,3),max_features=100)
vectorized_data=vect.fit_transform(text)
#id_map=dict((v,k) for k,v in vect.vocabulary_.items())
vect.vocabulary_.keys()

# ### Word 2 Vec

# In[ ]:


def build_corpus(df,col):
    
    '''function to build corpus from dataframe'''
    lem=WordNetLemmatizer()
    corpus= []
    for x in df[col]:
        
        
        words=word_tokenize(x)
        corpus.append([lem.lemmatize(w) for w in words])
    return corpus


# In[ ]:


corpus=build_corpus(df,'requirements')
model = word2vec.Word2Vec(corpus, size=100, window=20, min_count=30, workers=4)


# ## TSNE

# In[ ]:


def tsne_plot(model,title='None'):
    "Creates and TSNE model and plots it"
    labels = []
    tokens = []

    for word in model.wv.vocab:
        tokens.append(model[word])
        labels.append(word)
    
    tsne_model = TSNE(perplexity=80, n_components=2, init='pca', n_iter=2500, random_state=23)
    new_values = tsne_model.fit_transform(tokens)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])
        
    plt.figure(figsize=(12, 12)) 
    plt.title(title)
    for i in range(len(x)):
        plt.scatter(x[i],y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.show()

# In[ ]:


tsne_plot(model,'Requirements')

# - we can see clusters of words used in **Requirement** section.

# ## What are the common requirements for any post?

# In[ ]:


token=word_tokenize(req)
counter=Counter(token)
count=[x[0] for x in counter.most_common(40) if len(x[0])>3]
print("Most common words in Requirement")
print(count)

# - It can be observed that companies prefer  
# - **experienced** 
# - **educated professionals**  having **degree from an accredicted university**
# - also willing to work **full-time**

# - To improve the diversity employers are encourged to give chance to **Freshers** and people willing to work **part time**.
#      This can attract greater pool of applicants.

# ## Duties

# In[ ]:


duties= ' '.join(d for d in df['duties'])
show_wordcloud(duties,'Duties')

# ### Most influential and common words in duties

# In[ ]:


lem=WordNetLemmatizer()
text=[lem.lemmatize(w) for w in word_tokenize(duties)]
vect=TfidfVectorizer(ngram_range=(1,3),max_features=200)
vectorized_data=vect.fit_transform(text)
#id_map=dict((v,k) for k,v in vect.vocabulary_.items())
vect.vocabulary_.keys()

# In[ ]:


token=word_tokenize(duties)
counter=Counter(token)
count=[x[0] for x in counter.most_common(40) if len(x[0])>3]
print("Most common words in Duties")
print(count)

# ## Word 2 Vec

# In[ ]:


corpus=build_corpus(df,'duties')
model = word2vec.Word2Vec(corpus, size=100, window=20, min_count=40, workers=4)


# ## TSNE

# In[ ]:


tsne_plot(model,'Duties')

# ##  Latent Dirichlet Allocation (LDA)

# Topic modeling is a type of statistical modeling for discovering the abstract “topics” that occur in a collection of documents. Latent Dirichlet Allocation (LDA) is an example of topic model and is used to classify text in a document to a particular topic. It builds a topic per document model and words per topic model, modeled as Dirichlet distributions.

# In[ ]:


lem=WordNetLemmatizer()
text=[lem.lemmatize(w) for w in word_tokenize(duties)]
vect=TfidfVectorizer(ngram_range=(1,3),max_features=200)
vectorized_data=vect.fit_transform(text)
id2word=dict((v,k) for k,v in vect.vocabulary_.items())



# In[ ]:


corpus=gensim.matutils.Sparse2Corpus(vectorized_data,documents_columns=False)
ldamodel = gensim.models.ldamodel.LdaModel(corpus,id2word=id2word,num_topics=8,random_state=34,passes=25,per_word_topics=True)


# In[ ]:


ldamodel.show_topic(1)

# ###  What is the Dominant topic and its percentage contribution in each document

# In LDA models, each document is composed of multiple topics. But, typically only one of the topics is dominant. The below code extracts this dominant topic for each sentence and shows the weight of the topic and the keywords in a nicely formatted output.

# In[ ]:


def format_topics_sentences(ldamodel, corpus, texts):
    # Init output
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row_list in enumerate(ldamodel[corpus]):
        row = row_list[0] if ldamodel.per_word_topics else row_list            
        # print(row)
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return(sent_topics_df)


df_topic_sents_keywords = format_topics_sentences(ldamodel=ldamodel, corpus=corpus, texts=build_corpus(df,'duties'))

# Format
df_dominant_topic = df_topic_sents_keywords.reset_index()
df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']
df_dominant_topic.dropna(inplace=True)
df_dominant_topic.head(5)

# ### Topic modeling visualization

# In[ ]:


data=build_corpus(df,'duties')

# ### Word Counts of Topic Keywords

# When it comes to the keywords in the topics, the importance (weights) of the keywords matters. Along with that, how frequently the words have appeared in the documents is also interesting to look.
# 
# Let’s plot the word counts and the weights of each keyword in the same chart.
# 
# You want to keep an eye out on the words that occur in multiple topics and the ones whose relative frequency is more than the weight. Often such words turn out to be less important. The chart I’ve drawn below is a result of adding several such words to the stop words list in the beginning and re-running the training process.

# In[ ]:


topics = ldamodel.show_topics(formatted=False)
data_flat = [w for w_list in build_corpus(df,'duties') for w in w_list]
counter = Counter(data_flat)

out = []
for i, topic in topics:
    for word, weight in topic:
        out.append([word, i , weight, counter[word]])

df_plot= pd.DataFrame(out, columns=['word', 'topic_id', 'importance', 'word_count'])        

# Plot Word Count and Weights of Topic Keywords
fig, axes = plt.subplots(4, 2, figsize=(10,12), sharey=True, dpi=160)
cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]
for i, ax in enumerate(axes.flatten()):
    ax.bar(x='word', height="word_count", data=df_plot.loc[df_plot.topic_id==i, :], color=cols[i], width=0.5, alpha=0.3, label='Word Count')
    ax_twin = ax.twinx()
    ax_twin.bar(x='word', height="importance", data=df_plot.loc[df_plot.topic_id==i, :], color=cols[i], width=0.2, label='Weights')
    ax.set_ylabel('Word Count', color=cols[i])
    #ax_twin.set_ylim(0, 0.040); ax.set_ylim(0, 4000)
    ax.set_title('Topic: ' + str(i), color=cols[i])
    ax.tick_params(axis='y', left=False)
    ax.set_xticklabels(df_plot.loc[df_plot.topic_id==i, 'word'], rotation=30, horizontalalignment= 'right')
    ax.legend(loc='upper left'); ax_twin.legend(loc='upper right')

fig.tight_layout(w_pad=2)    
fig.suptitle('Word Count and Importance of Topic Keywords',y=1)    
plt.show()

# ## What are the most common selection criterias?

# In[ ]:



plt.figure(figsize=(7,7))
count=df['selection'].astype(str).value_counts()[:10]
sns.barplot(y=count.index,x=count,palette='rocket')
plt.gca().set_yticklabels(count.index,rotation='45')
plt.show()


# - It is evident that  **interview** ,**Essay** and **Questionnaire** are the most common selection criterias.

# ### Is there any Gender bias in job bulletins?

# In the follow section i am trying investigate if there is any gender biased terms used in **Requirement** and **Duties** section of the job bulletin.   
# For that i will pos tag all the text data in the requirement field and then,
# - Extract the words having pronoun tag.
# - check if any gender biased terms like he/she is used in the field.
# 

# In[ ]:


def pronoun(data):
    
    '''function to tokenize data and perform pos_tagging.Returns tokens having "PRP" tag'''
    
    prn=[]
    vrb=[]
    token=word_tokenize(data)
    pos=pos_tag(token)
   
    vrb=Counter([x[0] for x in pos if x[1]=='PRP'])
    
    return vrb
    


req_prn=pronoun(req)
duties_prn=pronoun(duties)
print('pronouns used in requirement section are')
print(req_prn.keys())
print('\npronouns used in duties section are')
print(duties_prn.keys())


# 1. Surprisingly, i couldn't find any gender biased or racist pronouns in **Requirement ** or **Duties section**
# 2. you can see all the pronouns used are neutral.

# ##          More to come,stay tuned !

# ### If you like my kernel please consider upvoting.Thank you :)
