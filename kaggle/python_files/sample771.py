#!/usr/bin/env python
# coding: utf-8

# ## City of LA - Job postings should be an invitation, not a barrier
# 

# # Hypothesis : 
# 
# ## Adjusting the readability-level of job postings based on education requirements of that post will increase interest in job postings for those limited written English comprehension skills.
# 
# Upon initial analysis, the very first job text was at the 23rd and 24th grade level!!!!   A Painter.  I don't think that is reasonable to expect your average painter to be able to read at a post-doc level!
# 
# It's no suprise they might have trouble attracting talent
# 
# "The goal is to convert a folder full of plain-text job postings into a single structured CSV file and then to use this data to: (1) identify language that can negatively bias the pool of applicants; (2) improve the diversity and quality of the applicant pool; and/or (3) make it easier to determine which promotions are available to employees in each job class."
# 
# Breaking it down into:  (in all sort of orders in my notebooks as I work in different areas)
#      - Prep
#      - Explore
#      - Goal 1 - create a single structured CSV file
#      - Goal 2 - improve diversity and quality of applicant pool  (focus on readability)
#      - Goal 3 - identify promotional opportunities  (req identification and promotion graphs)
#                 
# ### Consider upvoting if any of this is of interest or value to you. Thanks!
# 
# # Goal 1: Results
# I don't have any yet for this based on the sample that was provided. I struggle with the usefulness of the layout as we don't have insight into the process that will consume it. The files I am creating are inputs to my analysis and recommendations for goals 2 and 3.
# 
# There are some discussion going on it here: https://www.kaggle.com/c/data-science-for-good-city-of-los-angeles/discussion/92339#latest-534057
# 
# # Goal 2 : Results and Recommendations
# I'm reviewing some ideas as I go along.
# 
# 1. grade level too high
#     a. too many high-syllable words
#     b. too many words in a sentence
# 2. the content might be overly-formal, reducing readability and industry tends to label this as 'male' (judge that as you may)
# 3. the length of the postings is generally way too long and exceeds 700 word limit
# 4. the postings appear to be developed for the employer, not the prospects
# 
# # Goal 3 : Results and Recommendations
# Assuming an employee database that stores pertinant data point, it would be advisable to create a table that stores the unique features of each job position (possibly job postings). A compare is done and when a threshold is met, a trigger communication is sent to the current employee of the promotional opportunity. This can also be done for prospects if they enter information and ask to be informed of opportunities. I have extracted some key requirements features. I also did one promotional graph - I have to figure out how to do this at scale based on the provided pdfs.

# # Explore - Getting Smart
# 
# #### White Paper: Gender, Genre, and Writing Style in Formal Written Texts 
#     http://u.cs.biu.ac.il/~koppel/papers/male-female-text-final.pdf
# 
# 
# #### What's a good readability score? 
#     For grade levels, the result of most of the scoring algorithms, the score corresponds roughly to the number of years of education a person has had - based on the USA education system.
#     A grade level of around 10-12 is roughly the reading level on completion of high school.
#     Text to be read by the general public should aim for a grade level of around 8. Written by Steve Linney 
# 
# #### Readability calculations
#     https://www.geeksforgeeks.org/readability-index-pythonnlp/
#     
# #### Comments on length - no more than 700 words
#     https://www.mightyrecruiter.com/blog/6-appalling-job-postings-and-what-you-can-learn-from-them/

# #### Packages to import.   Custom import required
# 
#  PyPDF2    https://pythonhosted.org/PyPDF2/PdfFileReader.html
#  
#  textstat https://pypi.org/project/textstat/

# In[ ]:


# Apologies if some packages are imported down further in the notebook.
#...it's a work in progress and sometimes I forget to bring them back up.
# Also, if there is much code and many packages and I just want to reuse one sections, I have to remember which packages went with it.




import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
import re #search in strings.

import plotly.plotly as py
import cufflinks as cf

import seaborn as sns
import matplotlib.pyplot as plt

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

from wordcloud import WordCloud
import textstat

pd.set_option('max_colwidth', 10000)  # this is important because the requirements are sooooo long 

import warnings
warnings.filterwarnings('ignore')   # get rid of the matplotlib warnings

# ## Prep : Load file titles

# In[ ]:


input_dir = '../input/data-science-for-good-city-of-los-angeles/cityofla/CityofLA/Job Bulletins/'

def getListOfFiles(dirName):
# create a list of file and sub directories 
# names in the given directory 
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
    # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory 
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)
    return allFiles
listOfFiles = getListOfFiles(input_dir)
df_bulletins = pd.DataFrame(listOfFiles, columns = ['job_position'])
df_bulletins.head()

# ### Prep : Clean up the file names

# In[ ]:


# Clean up of the job_position name
df_positions = pd.DataFrame()
df_positions['job_position'] = (df_bulletins['job_position']
                                .str.replace(input_dir, '', regex=False)
                                .str.replace('.txt', '', regex=False)
                                .str.replace('\d+', '')
                                .str.replace(r"\s+\(.*\)","")
                                .str.replace(r"REV",""))

#Remove the numbers
df_positions['class_code'] = (df_bulletins['job_position']
                              .str.replace(input_dir, '', regex=False)
                              .str.replace('.txt', '', regex=False)
                              .str.extract('(\d+)'))

display(df_positions.head())
# Add the Text fields of Salary, Duties and Minimum REQ


# ### Prep : Convert the information in the txt files in a table
# 

# In[ ]:


#Convert the txt files to a table:
import glob
path = input_dir # use your path
all_files = glob.glob(path + "/*.txt")
li = []

for filename in all_files:
    with open (filename, "r",errors='replace') as myfile:
        data=pd.DataFrame(myfile.readlines())
        #df = pd.read_csv(filename, header=0,error_bad_lines=False, encoding='latin-1')
    li.append(data)
frame = pd.concat(li, axis=1, ignore_index=True)
#pd.read_csv(listOfFiles,header = None)
frame = frame.replace('\n','', regex=True)


# Prep : Look for keywords, and append the following strings to the final dataframe

# In[ ]:


# Here the loop should start, for each text file do:
def getString(col_i, frame):
    try:
        filter = frame[col_i] != ""
        bulletin = frame[col_i][filter]
        #display(salary)
        isal = min(bulletin[bulletin.str.contains('SALARY',na=False)].index.values) #take the sum to convert the array to an int...TO CHANGE
        inot = min(bulletin[bulletin.str.contains('NOTES',na=False)].index.values) # NOTES
        idut = min(bulletin[bulletin.str.contains('DUTIES',na=False)].index.values) # DUTIES
        ireq = min(bulletin[bulletin.str.contains('REQUIREMENT',na=False)].index.values) #REQUIREMENTS
        ipro = min(bulletin[bulletin.str.contains('PROCESS',na=False)].index.values) # PROCESS NOTES

        #isal = sum(bulletin.loc[bulletin == 'ANNUAL SALARY'].index.values) #take the sum to convert the array to an int...TO CHANGE
        #inot = sum(bulletin.loc[bulletin == 'NOTES:'].index.values) # NOTES
        #idut = sum(bulletin.loc[bulletin == 'DUTIES'].index.values) # DUTIES
        #ireq = sum(bulletin.loc[bulletin == '(.*)REQUIREMENTS(.*)'].index.values) #REQUIREMENTS
        #ipro = sum(bulletin.loc[bulletin == '(.*)PROCESS(.*)'].index.values) # PROCESS NOTES

        icode = min(bulletin[bulletin.str.contains('Class Code',na=False)].index.values)
        class_code = sum(bulletin.str.extract('(\d+)').iloc[icode].dropna().astype('int'))
        salary = (bulletin.loc[isal+1:inot-1]).to_string()
        duties = (bulletin.loc[idut+1:ireq-1]).to_string()
        requirements = (bulletin.loc[ireq+1:ipro-1]).to_string()
        return (class_code, salary, duties, requirements)
    except:
        return (np.nan,np.nan,np.nan,np.nan)
    
jobsections = pd.DataFrame()
#getString(0,bulletin)
for col_i in range(frame.shape[1]):
    #print(col_i)
    #print(list(getString(col_i,frame)))
    prop = getString(col_i,frame)
    prop = pd.DataFrame(list(prop)).T
    jobsections = jobsections.append(prop)

# In[ ]:


jobsections.head()

# In[ ]:


jobsections.columns = ['class_code','salary','duties','requirements']
jobsections['class_code'] = pd.to_numeric(jobsections['class_code'],downcast='integer')
df_positions['class_code'] = pd.to_numeric(df_positions['class_code'], downcast='integer')
#df_positions['class_code']
df_jobs = df_positions.merge(jobsections, left_on='class_code',right_on='class_code', how='outer')
display(df_jobs.dropna())


# In[ ]:


pd.set_option('max_colwidth', 10000)

jobsections['requirements'].head()

# ## Explore - Duties Wordcloud

# In[ ]:


# Read the whole text.
text = df_jobs['duties'].values

text = str(text)


# Generate a word cloud image
wordcloud = WordCloud().generate(text)

# Display the generated image:
# the matplotlib way:
import matplotlib.pyplot as plt
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")

# lower max_font_size
wordcloud = WordCloud(max_font_size=40).generate(text)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

# # Explore - PDFs
# Let's take a look
# 

# In[ ]:


import PyPDF2   #https://pythonhosted.org/PyPDF2/PdfFileReader.html

# Get the pdf files.
input_dir = '../input/data-science-for-good-city-of-los-angeles/cityofla/CityofLA/Additional data/PDFs/'

listOfFiles = getListOfFiles(input_dir)
listOfFiles


# In[ ]:


df_opening_pdfs = pd.DataFrame(listOfFiles, columns = ['opening_pdf'])

# Clean up the pdf opening names

df_openings = pd.DataFrame()
df_openings['job_position'] = (df_opening_pdfs['opening_pdf']
                                .str.replace(input_dir, '', regex=False)
                                .str.replace('.txt', '', regex=False)
                                .str.replace('\d+', '')
                                .str.replace(r"\s+\(.*\)","")
                                .str.replace(r"REV",""))

#Remove the numbers
df_openings['class_code'] = (df_opening_pdfs['opening_pdf']
                              .str.replace(input_dir, '', regex=False)
                              .str.replace('.txt', '', regex=False)
                              .str.extract('(\d+)'))


df_openings['version'] =       df_opening_pdfs['opening_pdf'].str.slice(start=-10,stop=-4)


pdfFile = PyPDF2.PdfFileReader('../input/data-science-for-good-city-of-los-angeles/cityofla/CityofLA/Additional data/PDFs/2018/December/Dec 7/SENIOR PERSONNEL ANALYST 9167 120718.pdf', 'rb')
#pdfFile = df_opening_pdfs['opening_pdf']
metadata = pdfFile.getDocumentInfo()
metadata
#pdfReader = PyPDF2.PdfFileReader(pdfFileObj) 
#df_openings['info'] = pdfReader.getPage(1) 
                           
df_openings.head()
# Add the Text fields of Salary, Duties and Minimum REQ


#pdfFileObj = open('../input/CityofLA/CityofLA/Additional_data/PDFs/2018/February/Feb_2/SOLID_WASTE_DISPOSAL_SUPERINTENDENT_4108_020218.pdf','rb') #'rb' for read binary mode 
#pdfReader = PyPDF2.PdfFileReader(pdfFileObj) 
#pdfReader.numPages.pageObj = pdfReader.getPage(1) #'1' is the page number pageObj.extractText()

# ## Explore - What is the structure of the PDF?

# In[ ]:


pdfFile = PyPDF2.PdfFileReader('../input/data-science-for-good-city-of-los-angeles/cityofla/CityofLA/Additional data/PDFs/2018/December/Dec 7/SENIOR PERSONNEL ANALYST 9167 120718.pdf', 'rb')
#pdfFile = df_opening_pdfs['opening_pdf']
DocumentInfo = pdfFile.getDocumentInfo()
DocumentInfo

# In[ ]:


pdfFields = pdfFile.getNumPages()
pdfFields

# In[ ]:


pdfPageLayout = pdfFile.getPageLayout()
pdfPageLayout

# In[ ]:


pdfPageMode = pdfFile.getOutlines()
pdfPageMode

# In[ ]:


pdfFile.numPages 
pageObj = pdfFile.getPage(1)
pdftext= pageObj.extractText()

pdftext

# # Goal 2 - Text Reading Levels

# https://www.geeksforgeeks.org/readability-index-pythonnlp/
# 
# To apply the formula:
# 
# Select several 100-word samples throughout the text.
# Compute the average sentence length in words (divide the number of words by the number of sentences).
# Compute the percentage of words NOT on the Dale–Chall word list of 3, 000 easy words.
# Compute this equation
# 
#  Raw score = 0.1579*(PDW) + 0.0496*(ASL) + 3.6365
# Here,
# PDW = Percentage of difficult words not on the Dale–Chall word list.
# ASL = Average sentence length
# The Gunning fog Formula
# 
# Grade level= 0.4 * ( (average sentence length) + (percentage of Hard Words) )
# Here, Hard Words = words with more than two syllables.
# Smog Formula
# 
# SMOG grading = 3 + √(polysyllable count).
# Here, polysyllable count = number of words of more than two syllables in a 
# sample of 30 sentences.
# Flesch Formula
# 
# Reading Ease score = 206.835 - (1.015 × ASL) - (84.6 × ASW)
# Here,
# ASL = average sentence length (number of words divided by number of sentences)
# ASW = average word length in syllables (number of syllables divided by number of words)
# Advantages of Readability Formulae:
# 
# 1. Readability formulas measure the grade-level readers must have to be to read a given text. Thus provides the writer of the text with much needed information to reach his target audience.
# 
# 2. Know Before hand if the target audience can understand your content.
# 
# 3. Easy-to-use.
# 
# 4. A readable text attracts more audience.
# 
# Disadvantages of Readability Formulae:
# 
# 1. Due to many readability formulas, there is an increasing chance of getting wide variations in results of a same text.
# 
# 2. Applies Mathematics to Literature which isn’t always a good idea.
# 
# 3. Cannot measure the complexity of a word or phrase to pinpoint where you need to correct it.

# In[ ]:


#import textstat

# 2018/December/Dec 7/SENIOR PERSONNEL ANALYST 9167 

test_data = (
    "The examination will consist of a qualifying multiple"
    "-choice test, an advisory essay, an advisory oral presentation, and an interview."
    " The qualifying written test will consist of "
    "multiple-choice questions in which emphasis may be placed on the candidate's expertise and knowledge of:"
    " Civil Service selection procedures; Equal Employment Opportunity "
    "(EEO) policies; Americans with Disabilities Act (ADA) regulations; Family and Medical Leave Act (FMLA); "
    "Fair Labor Standards Act (FLSA); and demonstrated proficiency and "
    "familiarity with the City's authoritative documents sufficient to identify the appropriate source, "
    "interpret complex written material, and effectively interpret provisions of the City Charter, "
    "Administrative Code, City Code of Ethics, Memoranda of Understandin (MOUs) provisions,"
    " Mayor's Executive Directives, and Personnel Department rules, policies and procedures, "
    "including Civil Service Commission (CSC) Rules, Personnel Department Policies and Personnel Department Procedures Manual; "
    "interpret complex data such as legislation, technical reports, and graphs; principles and practices of supervision,"
    " including training, counseling, and disciplining subordinate staff; and other necessary knowledge, skills, and abilities."
    "Prior to the multiple-choice test, applicants will be required to prepare some written material related to the work "
    "of a Senior Personnel Analyst employed by the City of Los Angeles. "
    "This essay material will not be separately scored, but will be presented to the interview board "
    "for discussion with the candidate and for consideration in the overall evaluation of the candidate's qualifications."
    "The advisory essay will be administered on-line. Candidates will receive an e-mail from the City of Los Angeles "
    "outlining the specific steps needed to complete the on-line advisory essay. "
    "Candidates will be required to complete the on-line advisory essay between FRIDAY, JANUARY 11, 2019 and "
    "SUNDAY, JANUARY 13, 2019. Additional instructions will be sent via e-mail. "
    "Candidates who fail to complete the advisory essay as instructed may be disqualified."
    "The multiple-choice test will be proctored and administered on-line during a single session. "
    "Candidates invited to participate in the on-line multiple-choice test will be able to take the test "
    "as instructed from a remote location using a computer with a webcam and a reliable internet connection. "
    "Candidates will receive an e-mail from the City of Los Angeles outlining the dates and "
    "specific steps on how to take the multiple-choice test and advisory essay on-line"
)

textstat.flesch_reading_ease(test_data)
textstat.smog_index(test_data)
textstat.flesch_kincaid_grade(test_data)
textstat.coleman_liau_index(test_data)
textstat.automated_readability_index(test_data)
textstat.dale_chall_readability_score(test_data)
textstat.difficult_words(test_data)
textstat.linsear_write_formula(test_data)
textstat.gunning_fog(test_data)
textstat.text_standard(test_data)

# In[ ]:


# Let's take another sample


df_opening_pdfs.head()

# Clean up the pdf opening names

df_openings = pd.DataFrame()
df_openings['job_position'] = (df_opening_pdfs['opening_pdf']
                                .str.replace(input_dir, '', regex=False)
                                .str.replace('.txt', '', regex=False)
                                .str.replace('\d+', '')
                                .str.replace(r"\s+\(.*\)","")
                                .str.replace(r"REV",""))

#Remove the numbers
df_openings['class_code'] = (df_opening_pdfs['opening_pdf']
                              .str.replace(input_dir, '', regex=False)
                              .str.replace('.txt', '', regex=False)
                              .str.extract('(\d+)'))


df_openings['version'] =       df_opening_pdfs['opening_pdf'].str.slice(start=-10,stop=-4)

pdfFile = PyPDF2.PdfFileReader('../input/data-science-for-good-city-of-los-angeles/cityofla/CityofLA/Additional data/PDFs/2018/September/Sept 21/PAINTER 3423 092118.pdf', 'rb')
#pdfFile = df_opening_pdfs['opening_pdf']
metadata = pdfFile.getDocumentInfo()
metadata
#pdfReader = PyPDF2.PdfFileReader(pdfFileObj) 
#df_openings['info'] = pdfReader.getPage(1) 
                           
df_openings.head()

# In[ ]:


pdfFile.numPages 
pageObj = pdfFile.getPage(1)
pdftext= pageObj.extractText()

pdftext

# In[ ]:


# A painter!!!


def replace(value):
  return re.sub(r'\n', r'', value)

pdftext_clean = replace(pdftext)
pdftext_clean
                                
textstat.flesch_reading_ease(pdftext_clean)
textstat.smog_index(pdftext_clean)
textstat.flesch_kincaid_grade(pdftext_clean)
textstat.coleman_liau_index(pdftext_clean)
textstat.automated_readability_index(pdftext_clean)
textstat.dale_chall_readability_score(pdftext_clean)
textstat.difficult_words(pdftext_clean)
textstat.linsear_write_formula(pdftext_clean)
textstat.gunning_fog(pdftext_clean)
textstat.text_standard(pdftext_clean)

# In[ ]:


pdftext_clean

# In[ ]:


basic = "SELECTION PROCESS  The examination will consist entirely of a weighted multiple-choice test administered and proctored on-line. In the on-line multiple-choice test, the following competencies may be evaluated: Mathematics, Teamwork, Equipment Operation, including the operation of hydraulic equipment, such as paint sprayers, scissor lift, and boom lift used to apply paint to surfaces at elevated heights; Safety Focus, including: safety procedures, regulations, and restrictions as required by the California Occupational Safety and Health Administration, South Coast Air Quality Management District, Environmental Protection Act, and California Department of Toxic Substances Control, including procedures necessary when using paints and coatings containing volatile organic compounds, handling and disposing of hazardous or toxic wastes, and working near energized electrical equipment or with toxic and flammable materials; safety procedures and personal protective equipment required when preparing surfaces and applying paint; safety requirements that must be adhered to when using stepladders, extension ladders, and scaffolds; safety procedures required when using high pressure equipment for the preparation of surfaces and application of paint; equipment used to ventilate an area during and/or after painting; and Job Knowledge, including knowledge of: protective and decorative coverings, and the procedures used to mix and apply them; methods, tools, and materials used to prepare a wide variety of surfaces for painting; methods, tools, and equipment used to apply paint or protective coatings to a wide variety of surfaces; other necessary skills, knowledge, and abilities.   Additional information can be obtained by going to http://per.lacity.org/index.cfm?content=jobanalyses and clicking on Competencies under Painter. "

# In[ ]:


# Painter   54th and 55th grade???  Good grief

pdftext_clean = basic                
textstat.flesch_reading_ease(pdftext_clean)
textstat.smog_index(pdftext_clean)
textstat.flesch_kincaid_grade(pdftext_clean)
textstat.coleman_liau_index(pdftext_clean)
textstat.automated_readability_index(pdftext_clean)
textstat.dale_chall_readability_score(pdftext_clean)
textstat.difficult_words(pdftext_clean)
textstat.linsear_write_formula(pdftext_clean)
textstat.gunning_fog(pdftext_clean)
textstat.text_standard(pdftext_clean)

# # Goal 2 - Readability Metrics
# 
# ## I ran the Painter text through readable.com to get some industry measurement standards.
# 
# Interesting to see that syllables, length of words, and length of sentence are key drivers of the scoring. I think I can work with those.
# 
# https://www.kaggle.com/silverfoxdss/painteranalysis

# ![image.png](attachment:image.png)

# # Goal 2 - Bring in the generated scores and take a look

# I generated the readability scores using Readable.com during a 24-hour free demo period. I did not have enough time to process all text and pdf's, but I got a good representation. I dowloaded each output for each file that I processed and merged into a single csv.
# 
# This csv has been published here on Kaggle and is available for your use:  
# 
# https://www.kaggle.com/silverfoxdss/city-of-la-readbility-scores

# In[ ]:


scores = pd.read_csv('../input/city-of-la-readbility-scores/reading_level_samples_combined.csv', header=0)
scores

# In[ ]:


scores.columns

# # Goal 2 - Syllables
# 
# Too many Syllables
# 
# Words that seem common: Identification, Responsibilities
# 
# Words that probably are problematic: Interdepartmental, reconciliation (unless it's an accounting or auditing position)

# In[ ]:


# Read the whole text.
longsyll = scores['Longest Word Syllables Words'].values

text = str(longsyll)


# Generate a word cloud image
wordcloud = WordCloud().generate(text)

# Display the generated image:
# the matplotlib way:
import matplotlib.pyplot as plt
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")

# lower max_font_size
wordcloud = WordCloud(max_font_size=40).generate(text)
plt.figure(figsize = (8, 8), facecolor = None) 
plt.figure()
plt.axis("off")
plt.tight_layout(pad = 0) 
plt.show()



# # Goal 2- Too many words
# 
# 700+ being too many
# 
# The very fact that scale of my distribution plot goes over 8000 is not good.

# In[ ]:


sns.distplot(scores['Word Count'], kde=False, rug=True);

# # Goal 2 - Gender numbers affect ratings
# 
# gender numbers that are lower - or at least on this scale, most likely in the middle, rate higher.

# In[ ]:


# Gender (runs on a scale Female 0 to Male 100). There is documentation that it is noted that there is an inherent issue with this scale.

ax = sns.swarmplot(x="Gender Number", y="Rating", data=scores, order='A,B,C,D,E')

# ### Goal 2 - Time to figure out how to join the document dataframe to the readability dataframe

# In[ ]:


scores.head()

# In[ ]:


# I believe I will have to try to join the job name with the file name
# I need to create a new column on the scores , job_title
# I will need to clean up the Item column
        # Remove underscores
        # Only bring in the words, or the first n spaces          Yes, this code would be better structured for reuse....

scores['job_title'] = (scores['Item']
                                .str.replace('.txt', '', regex=False)
                                .str.replace('.pdf', '', regex=False)
                                .str.replace('_', ' ', regex=False)
                                .str.replace('\d+', '')
                                .str.replace(r"\s+\(.*\)","")
                                .str.replace(r"REV","")
                                .str.upper() )

scores['class_code'] = (scores['Item']
                              .str.replace(input_dir, '', regex=False)
                              .str.replace('.txt', '', regex=False)
                              .str.extract('(\d+)'))
scores.head(25)

# class code isn't perfect...
# combination of all caps, applying  .str.upper() 

# In[ ]:


# if you've seen my other kernels, you'll know I like pandasql. 
# It's not as efficient at joining tables, but I find it more readable (no pun intended!)

import pandasql                    # https://github.com/yhat/pandasql
from pandasql import sqldf
pysqldf = lambda q: sqldf(q, globals())

joined = """
select 
a.job_position as job_position1,a.class_code as class_code1, b.item as item2, b.job_title as job_title2, b.class_code as class_code2
from df_jobs a
left outer join scores b
on a.job_position = b.job_title
"""

# setting up the left outer join to list all of the text and pdfs with score to the list of job titles we created earlier
joined2 = """
select a.*, b.*
from df_jobs b
left outer join scores a
on b.job_position = a.job_title
"""
df_joined2x = pysqldf(joined2)
print(df_joined2x)
# downloading to take a good look
df_joined2x.to_csv('dfjoined2x.csv')

# some manual cleanup for speed purposes



# # Goal 3 - Requirements

#  This is definitely going to be a wander (in circles, for sure)
# 
#  The goal (I believe):
#  
#      Given an employee's current state
#              a. current job title/code
#              b. length of time in current position
#              c. highest level of education
#              d. apprentice stints
#              e. age
#              
#                   Identify current or upcoming promotional activities
#                   
#      Assumption: Ignore lateral moves

# In[ ]:


what = """

SELECT substr(requirements, 1, pos-1) AS req_num1,
       substr(requirements, pos+1) AS req_desc1,
       substr(requirements, pos+2) AS req_num2,
       substr(requirements, pos+3) AS req_desc2
FROM
  (SELECT *,
          instr(requirements,'.') AS pos
   FROM df_jobs)

"""

df_whatx = pysqldf(what)
df_whatx.head(10)

# 1. = 453
# 2. = 449
# 3. = 236
# 4. = 124
# 5. = 69
# 6. = 34
# 7. = 22
# 8. = 10
# 9. = 6
# first byte is not a number, blank, or '.' 2
# second byte not a number 109
# third = 2
# fourth = 2  fifth = 2 6th = 2 7th = 259
# annoying, need to strip the spaces off the front  trim()

# In[ ]:


df_jobs1 = df_jobs
df_jobs1['requirements'] = df_jobs1['requirements'].astype('str')
x = pd.DataFrame(df_jobs1['requirements'].str.split().values.tolist())
#x = pd.DataFrame(df_jobs1.requirements.str.split('.', expand=True).values,
#             columns=['Req1', 'Req2'])
#df_jobs1[['Req1','Req2']] = df_jobs1.requirements.str.split("1.",expand=True,)
x.head()

# this is garbage output but I'll leave it here to show what this code results in.

# In[ ]:


# Warning - will run loooong
#tryit_df = df_jobs.head(10)
tryit = """
with separators as ( values ('1.'), ('2.'), ('3.'), ('4.') ),
  source (s) as ( select requirements from tryit_df ),
  bag (q) as ( -- POSITIONS OF ALL SEPARATORS
    with dim (len) as ( select length(s) from source ),
    ndx (n) as (
      select 1 union all select n+1 from ndx, dim where n < len
    ) select 0 --> PSEUDO SEPARATOR IN FRONT OF SOURCE STRING
      union all select n from ndx, source where substr(s, n, 1) in separators
      union all select len+1 from dim --> PSEUDO SEPARATOR AT BOTTOM
  ),
  pre (p) as ( -- POSITIONS OF SEPARATORS PRECEDING NON SEPARATORS
    select q from bag where not q+1 in bag
  ) select printf("%2d %2d <%s>", p, z, substr(s, p+1, z-p-1)) from (
      select p, (select min(q) from bag where q > p) as z from pre where z
    ), source
    limit 30;
"""

#tryitx = pysqldf(tryit)
#tryitx.head()

#pandasql - again, not the most efficient....
# I tried to Commit entire set but got : PandaSQLException: (sqlite3.OperationalError) database or disk is full
# I am interested to see how this technique works so I will try it on a single row.

# In[ ]:


# Goal 3 - What are the most common requirements words to start looking at?
import collections
import matplotlib.cm as cm
from matplotlib import rcParams
from wordcloud import WordCloud, STOPWORDS

#reqwords = df_jobs['requirements'].values
reqwords = ' '.join(df_jobs['requirements'].str.lower())
text = str(reqwords)
stopwords = ['apply', 'nan', '1.', '2.', '3.', '4.', '5', '6', '7.', 'a.', 'b.', 'c.', 'd.','st...', 'a...',
             '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22',
              '23', '24', '25', '26', '27','28','29','30','31', '32', '33', '34', '35', '36', '37', 
               '38', '39', '50', 'one', 'two', 'three', 'four', 'five', 'six','seven','eight','nine','ten',
            'eleven', 'twelve', 'must', 'may', 'notes:','...', 't...']  + list(STOPWORDS)


# Generate a word cloud image
wordcloud = WordCloud(stopwords=stopwords,width=800, height=400).generate(text)
plt.figure( figsize=(20,10) )
plt.imshow(wordcloud)

# looks like they want full time paid service of a specific time period!

# In[ ]:


filtered_words = [word for word in reqwords.split() if word not in stopwords]
counted_words = collections.Counter(filtered_words)

words = []
counts = []
for letter, count in counted_words.most_common(40):
    words.append(letter)
    counts.append(count)
colors = cm.rainbow(np.linspace(0, 1, 10))
rcParams['figure.figsize'] = 20, 10

plt.title('Top words in the requirements')
plt.xlabel('Count')
plt.ylabel('Words')
plt.barh(words, counts, color=colors)

# rainbow for inclusion lol

# #### Goal #3 - Considering what we want to trigger a behavior nudge for a current employee to apply for a promotion
# 
# Time in prereq position met, 
# Age?, 
# New degree or certification
# 
# Pretend that we have a file of current employee status similar to such: (for simplicity, just their current job counts) :
# Employee ID,  Current Job Code,  Current Job Full-time Months in Service,  Current Job Part-time Months in Service, Paid Position Flag, 
# Accredited PhD Major, Accredited Masters Major, Accredited Bachelors Major, Accredited Associates Major, High School/GED Completed, Some College Completed, Certification1, Certification2, Certification3, Certification4, Certification5, Drivers License Flag, CDL Flag 
# 

# In[ ]:


degreesq = """
select
distinct
class_code,
case when requirements like '%full-time%' or requirements like '%Full-time%' 
        or requirements like '%full time%' or requirements like '%Full time%'
    then 1 else 0 end as full_time
,case when requirements like '%part-time%' or requirements like '%Part-time%' 
        or requirements like '%part time%' or requirements like '%Part time%'
    then 1 else 0 end as part_time
,case when requirements like '%month%' or requirements like '%year%' then 1 else 0 end as time_in_job_required
,case when requirements like '%Degree%' or requirements like '%degree%' then 1 else 0 end as degree_required
,case when requirements like '%Cert%' or requirements like '%cert%' then 1 else 0 end as certification_required
,case when requirements like '%Graduat%' or requirements like '%graduat%' then 1 else 0 end as graduation_required
,case when requirements like '%apprentic%' or requirements like '%Apprentic%' then 1 else 0 end as apprenticeship_required
,case when requirements like '%license%' or requirements like '%License%' then 1 else 0 end as license_required
from df_jobs
order by class_code desc

"""

degrees = pysqldf(degreesq)
degrees.head(200)

# some of the class_codes don't look right. I ran this query up top after I created the df and it still looks that way so it didn't get messed up in the meantime


# In[ ]:


# Goal #3 continue
reqq = """
with list as (
select
distinct
class_code,
case when requirements like '%full-time%' or requirements like '%Full-time%' or requirements like '%full time%' or requirements like '%Full time%' 
then 1 else 0 end as full_time
,case when requirements like '%part-time%' or requirements like '%Part-time%' or requirements like '%part time%' or requirements like '%Part time%' then 1 else 0 end as part_time
,case when requirements like '%month%' or requirements like '%year%'
        or requirements like '%uarter%' or requirements like '%ours%'
then 1 else 0 end as time_in_job_required
,case when requirements like '%GED%' or requirements like '%G.E.D%'
        or requirements like '%igh school%' or requirements like '%HS%' or requirements like '%H.S.%'
then 1 else 0 end as GED_HS_required
,case when requirements like '%Degree%' or requirements like '%degree%' 
then 1 else 0 end as degree_required
,case when requirements like '%Cert%' or requirements like '%cert%' 
then 1 else 0 end as certification_required
,case when requirements like '%school%' or requirements like '%School%' or requirements like '%training%'
   or requirements like '%Training%' or requirements like '%Course%' or requirements like '%course%'
   or requirements like '%Academy%' or requirements like '%academy%' 
   or requirements like '%Instruction%' or requirements like '%instruction%'
then 1 else 0 end as coursework_required
,case when requirements like '%Graduat%' or requirements like '%graduat%' 
then 1 else 0 end as graduation_required
,case when requirements like '%apprentic%' or requirements like '%Apprentic%' 
then 1 else 0 end as apprenticeship_required
,case when requirements like '%license%' or requirements like '%License%' 
then 1 else 0 end as license_required
,case when requirements like '%college%' or requirements like '%College%' or requirements like '%university%' or requirements like '%University%' 
then 1 else 0 end as college_required
,case when requirements like '%urrently employed with the City of L%'  or requirements like '%urrent employment with the City of L%' 
then 1 else 0 end as current_employee
, case when requirements like '%ne year of full-time paid experience%' then 12
       when requirements like '%wo years of full-time paid experience%' then 24
       when requirements like '%hree years of full-time paid experience%' then 36
       when requirements like '%our years of full-time paid experience%'  then 48
       when requirements like '%ive years of full-time paid experience%' then 60
       when requirements like '%ix years of full-time paid experience%' then 72
       when requirements like '%even years of full-time paid experience%' then 84
       when requirements like '%ight years of full-time paid experience%' then 96
       when requirements like '%ine years of full-time paid experience%' then 108
       when requirements like '%ten years of full-time paid experience%' then 120
       when requirements like '%Ten years of full-time paid experience%' then 120
       when requirements like '%wenty four months of%' then 24
       when requirements like '%wenty-four months of%' then 24
       when requirements like '%ne month of%' then 1
       when requirements like '%wo months of%' then 2
       when requirements like '%hree months of%' then 3
       when requirements like '%our months of%' then 4
       when requirements like '%ive months of%' then 5
       when requirements like '%ix months of%' then 6
       when requirements like '%even months of%' then 7
       when requirements like '%ight months of%' then 8
       when requirements like '%ine months of%' then 9
       when requirements like '%ten months of%' then 10
       when requirements like '%Ten months of%' then 10
       when requirements like '%leven months of%' then 11
       when requirements like '%welve months of%' then 12
       when requirements like '%ighteen months of%' then 18
       when requirements like '%wenty four months of%' then 24
       else 0 end as months_ft
,case when requirements like '%14 years of age%' then 14
      when requirements like '%15 years of age%' then 15
      when requirements like '%16 years of age%' then 16
      when requirements like '%17 years of age%' then 17
      when requirements like '%18 years of age%' then 18
      when requirements like '%19 years of age%' then 19
      when requirements like '%20 years of age%' then 20
      when requirements like '%20 1/2 years of age%' then 20.5
      when requirements like '%21 years of age%' then 21
      when requirements like '%25 years of age%' then 25
 else 0 end as min_age
 , case when requirements like '%valid California driver%' then 1 else 0 end as CA_DL_required
 , case when requirements like '%CDL%' or requirements like '%C.D.L.%' then 1 else 0 end as CDL_required
 , case when requirements like '%lass A%' or requirements like '%lass B%' then 1 else 0 end as Class_AorB_DL_required
 , case when requirements like '%ourney%' then 1 else 0 end as Journey_Level_Required
 , case when requirements like '%enior%'  then 1 else 0 end as Senior_Exp_Required
 , case when requirements like '%upervisor%' then 1 else 0 end as Supervisor_Exp_Required
 , case when requirements like '%anager%' then 1 else 0 end as Manager_Exp_Required
,requirements
from df_jobs
order by class_code desc
)
select l.* from df_jobs j
inner join list l
on j.class_code = l.class_code
--where l.one_year_ft = 1
--or    l.two_years_ft = 1
--or    l.three_years_ft = 1
--or    l.four_years_ft = 1


"""

reqs = pysqldf(reqq)
reqs.to_csv('reqs.csv')
# print to csv to take a look

# In[ ]:


reqs.head()

# # Goal 3 - Promotional Graphs

# #### Application Developer - simplified
# Application Developer  1429
# to Programmer Analyst I, II, III, IV 1431a
# Applications 
# to Programmer Analyst V 1431b or Systems Programmer I, II, III 1455
# manager 1409
# 
# Director of Systems 975

# In[ ]:


import networkx as nx
G = nx.DiGraph()
G.add_edges_from(
    [('1429', '1431a'), ('1431a', '1431b'), ('1431a', '1455'), 
     ('1431b', '1409'), ('1455', '1409'),
     ('1409', '975')])

val_map = {'1429': 1.0,
           '1409': 0.5714285714285714,
           '975': 0.0}

values = [val_map.get(node, 0.25) for node in G.nodes()]

# Specify the edges you want here
red_edges = [('1431b', '1409'), ('1409', '975')]
edge_colours = ['black' if not edge in red_edges else 'red'
                for edge in G.edges()]
black_edges = [edge for edge in G.edges() if edge not in red_edges]

# Need to create a layout when doing
# separate calls to draw nodes and edges
pos = nx.spring_layout(G)
nx.draw_networkx_nodes(G, pos, cmap=plt.get_cmap('jet'), 
                       node_color = values, node_size = 500)
nx.draw_networkx_labels(G, pos)
nx.draw_networkx_edges(G, pos, edgelist=red_edges, edge_color='r', arrows=True)
nx.draw_networkx_edges(G, pos, edgelist=black_edges, arrows=False)


# In[ ]:




job_graph = pd.read_csv('../input/city-of-la-job-graph/Job_Graph.csv', header=0,nrows=125)
job_graph['Job_1'] = job_graph['Job_1'].astype(str)
job_graph['Job_2'] = job_graph['Job_2'].astype(str)
jobs =  list(job_graph.Job_1.unique())
promotions = list(job_graph.Job_2.unique())
promotions



# In[ ]:


dict(zip(promotions, promotions))

# In[ ]:


[Job_2 for Job_2 in promotions]

# In[ ]:


plt.figure(figsize=(12, 12))
# Inspiration from  http://jonathansoma.com/lede/algorithms-2017/classes/networks/networkx-graphs-from-source-target-dataframe/
# 1. Create the graph
g = nx.from_pandas_edgelist(job_graph, source = 'Job_1', target = 'Job_2', edge_attr=None, create_using=None)

# 2. Create a layout for our nodes 
layout = nx.spring_layout(g,iterations=20)

# 3. Draw the parts we want
# Edges thin and grey
# Jobs_1 small and grey
# Promotions sized according to their number of connections
# Promotions blue
# Labels for Promotions ONLY
# Promotions that are highly connected are a highlighted color

# Go through every promotion ask the graph how many
# connections it has. Multiply that by 80 to get the circle size
promotion_size = [g.degree(Job_2) * 80 for Job_2 in promotions]
nx.draw_networkx_nodes(g, 
                       layout,
                       nodelist=promotions, 
                       node_size=promotion_size, # a LIST of sizes, based on g.degree
                       node_color='lightblue')

# Draw all jobs
nx.draw_networkx_nodes(g, layout, nodelist=jobs, node_color='#cccccc', node_size=100)

# Draw all jobs with most promotional ops
hot_jobs = [Job_1 for Job_1 in jobs if g.degree(Job_1) > 1]
nx.draw_networkx_nodes(g, layout, nodelist=hot_jobs, node_color='orange', node_size=100)

nx.draw_networkx_edges(g, layout, width=1, edge_color="#cccccc")

node_labels = dict(zip(promotions, promotions))
nx.draw_networkx_labels(g, layout, labels=node_labels)

plt.axis('off')
plt.title("Promotions")
plt.show()

# In[ ]:


from graphviz import Digraph
dot = Digraph(comment='Promotions')

for index, row in job_graph.iterrows():
    dot.edge(str(row["Job_1"]), str(row["Job_2"]), label='')

dot


# # More to come!
