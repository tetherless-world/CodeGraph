#!/usr/bin/env python
# coding: utf-8

# In[ ]:


## import libraries 
from collections import Counter 
import pandas as pd 
import numpy as np 
import string 

from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import plotly.plotly as py
from plotly import tools
import seaborn as sns
init_notebook_mode(connected=True)
from itertools import zip_longest
import string 
import re

from nltk.corpus import stopwords 
from nltk.util import ngrams
import nltk 
stopwords = stopwords.words('english')

## dataset preparation
messages = pd.read_csv("../input/ForumMessages.csv")
messages['CreationDate'] = pd.to_datetime(messages['PostDate'])
messages['CreationYear'] = messages['CreationDate'].dt.year
messages['CreationMonth'] = messages['CreationDate'].dt.month
messages['CreationMonth'] = messages['CreationMonth'].apply(lambda x : "0"+str(x) if len(str(x)) < 2 else x)
messages['CreationDay'] = "29"
messages['KernelDate'] = messages["CreationYear"].astype(str) +"-"+ messages["CreationMonth"].astype(str) +"-"+ messages["CreationDay"].astype(str)
messages['Message'] = messages['Message'].fillna(" ")

## function to remove html entities from text
def striphtml(data):
    p = re.compile(r'<.*?>')
    return p.sub('', data)

## function to clean a text
def clntxt(text):
    text = text.lower()
    text = striphtml(text)
    text = " ".join([c for c in text.split() if c not in stopwords])
    for c in string.punctuation:
        text = text.replace(c, " ")
    text = " ".join([c for c in text.split() if c not in stopwords])
    
    words = []
    ignorewords = ["&nbsp;", "quot", "quote", "www", "http", "com"]
    for wrd in text.split():
        if len(wrd) <= 2: 
            continue
        if wrd in ignorewords:
            continue
        words.append(wrd)
    text = " ".join(words)    
    return text

## function to get top ngrams for a given year
def get_top_ngrams(yr, n, limit):
    # get relevant text
    temp = messages[messages['CreationYear'] == yr]
    text = " ".join(temp['Message']).lower()
    
    # cleaning
    text = striphtml(text)
    text = " ".join([c for c in text.split() if c not in stopwords])
    for c in string.punctuation:
        text = text.replace(c, " ")
    text = " ".join([c for c in text.split() if c not in stopwords])
    
    # ignore 
    words = []
    ignorewords = ["&nbsp;", "quot", "quote", "www", "http", "com"]
    for wrd in text.split():
        if len(wrd) <= 2: 
            continue
        if wrd in ignorewords:
            continue
        words.append(wrd)
    text = " ".join(words)
    
    # tokenize
    token = nltk.word_tokenize(text)
    grams = ngrams(token, n)
    grams = [" ".join(c) for c in grams]
    return dict(Counter(grams).most_common(limit))

def check_presence(txt, wrds):    
    cnt = 0
    txt = " "+txt+" "
    for wrd in wrds.split("|"):
        if " "+wrd+" " in txt:
            cnt += 1 
    return cnt

messages['CMessage'] = messages['Message'].apply(lambda x : clntxt(x))

messages['CreationDay'] = "21"
messages['KernelDate'] = messages["CreationYear"].astype(str) +"-"+ messages["CreationMonth"].astype(str) +"-"+ messages["CreationDay"].astype(str)

# ## Historical Data Science Trends on Kaggle 
# 
# A number of trends have changed over the years in the field of Data Science. Kaggle is the largest and the most popular data science community across the globe. In this kernel, I am using Kaggle Meta Data to explore the Data Science trends over the years. 
# 
# ## 1. Linear Vs Logistic Regression
# 
# Lets look at the comparison of linear regression and logistic regression discussions on forums, kernels and replies on kaggle. 

# In[ ]:


def plotthem(listed, title):    
    traces = []
    for model in listed:
        temp = messages.groupby('KernelDate').agg({model : "sum"}).reset_index()
        trace = go.Scatter(x = temp["KernelDate"], y = temp[model], name=model.split("|")[0].title(), line=dict(shape="spline", width=2), mode = "lines")
        traces.append(trace)

    layout = go.Layout(
        paper_bgcolor='#fff',
        plot_bgcolor="#fff",
        legend=dict(orientation="h", y=1.1),
        title=title,
        xaxis=dict(
            gridcolor='rgb(255,255,255)',
            range = ['2010-01-01','2018-06-01'],
            showgrid=True,
            showline=False,
            showticklabels=True,
            tickcolor='rgb(127,127,127)',
            ticks='outside',
            zeroline=False
        ),
        yaxis=dict(
            title="Number of Kaggle Discussions",
            gridcolor='rgb(255,255,255)',
            showgrid=False,
            showline=False,
            showticklabels=True,
            tickcolor='rgb(127,127,127)',
            ticks='outside',
            zeroline=False
        ),
    )

    fig = go.Figure(data=traces, layout=layout)
    iplot(fig)
    
## linear vs logistic regression
models = ["linear regression", "logistic regression"]
for col in models:
    messages[col] = messages["CMessage"].apply(lambda x : check_presence(x, col))
plotthem(models, "Kaggle Discussions: Linear vs Logistic")    


# > - From the above graph, we can observe that there were always been **more discussions related to logistic regression** than linear regression. The generel trend is that number of discussions are increasing every month. 
# > - One indication is that there are more number of classification problems than regression problems on Kaggle including the most popular **Titanic Survival Prediction competition**. This competition has most number of discussions and is one of the longest running compeition on Kaggle. There is a regression competition as well : House Prices advanced regression, but people more often start it after titanic only.     
# > - The number of logistic regression discussions on forums, kernel comments, and replies boomed to high numbers in October 2017 and March 2018. One of the reason is the the **Toxic Comments Classification Competition"** in which a number of authors shared excellent information related to classification models including logistic regression. 
# 
# ## 2. The dominance of xgboost

# In[ ]:


models = ["decision tree","random forest", "xgboost|xgb", "lightgbm|lgb", "catboost"]
for col in models:
    messages[col] = messages["CMessage"].apply(lambda x : check_presence(x, col))
plotthem(models, "Kaggle Discussions: Tree based models")    


# > - Before 2014, Linear Models, Decision Trees, and Random Forests were very popular. But when XgBoost was open sourced in 2014, it gained popularty quickly and **dominated the kaggle competitions and kernels**. Today, xgboost is still used exhaustively in compeitions and is the part of the winning models of many competitions. Some examples are **Otto Group Classification Competition** in which first place solution made use of xgboost. 
# > - However with the arrival of **Lightgbm in 2016**, the useage of xgboost dipped to some extent and popularity of lightgbm started rising very quickly. Based on the recent increasing trend of lightgbm (shown in red), one can forecast that it will dominate next few years as well, unless any other company opensources a better model.  For example, lightgbm was used in the winning solution of **Porto Seguro’s Safe Driver Prediction** . One of the reason for light gbm popularity is the faster implementation and simple interface as compared to xgboost.  
# > - For instance, Catboost was recently released and is starting gaining popularity.  
# 
# 
# ## 3. Trends of Neural Networks and Deep Learning

# In[ ]:


models = ["neural network", "deep learning"]
for col in models:
    messages[col] = messages["CMessage"].apply(lambda x : check_presence(x, col))
plotthem(models, "Kaggle Discussions: Neural Networks vs Deep Learning")    

# > - Neural networks were present in the industry since the decades but in recent years trends changed because of the access to much larger data and computational power.  
# > - The era of deep learning started in 2014 with the arrival of libraries such as theano, tensorflow in 2015, and keras in 2016. The number of discussions related to deep learning is increasing regularly and are always more than neural networks. Also, many cloud instance providers such as **Amazon AWS, Google cloud** etc showcases their capabilities of training very deep neural networks on clouds.  
# > - The deeplearning models also became popular because of a number of Image Classification competitions on Kaggle such as :  **Data Science Bowl**, competitions from Google etc. Also, deeplearning models became popular for text classification problems for example **Quora Duplicate Questions Classification**.  
# > - Deep learning is also become populary every month because of different variants of models such as RNNs, CNNs have shown great improvements in the kernels. Also, **transfer learning and pre-trained models** have shown great results in competitions.  
# > - Kaggle can launch more competitions / playgrounds related to Image Classification Modelling as people wants to learn from them alot.  Not to forget that Kaggle have added the GPU support in kernels which facilitates the Deep Learning useage on kaggle.  
# 
# ## 4. ML Tools used on Kaggle

# In[ ]:


models = ["scikit", "tensorflow|tensor flow", "keras", "pytorch"]
for col in models:
    messages[col] = messages["CMessage"].apply(lambda x : check_presence(x, col))
plotthem(models, "Kaggle Discussions: ML Tools")    

# > - Scikit Learn was the only library used on kaggle for machine learning tasks, but since 2015 tensorflow gained populartiy. 
# > - Among the ML tools, Keras is the most popular because of the simplistic deep learning implementation.  
# 
# ## 5. XgBoost vs Keras

# In[ ]:


models = ["xgboost|xgb", "keras"]
for col in models:
    messages[col] = messages["CMessage"].apply(lambda x : check_presence(x, col))
plotthem(models, "Kaggle Discussions: Xgboost vs Deep Learning")    

models = ["cnn|convolution", "lstm|rnn|gru"]
for col in models:
    messages[col] = messages["CMessage"].apply(lambda x : check_presence(x, col))
plotthem(models, "Kaggle Discussions: CNN and LSTM")    

# > - Among both the popular techniques on Kaggle - xgboost and deeplearning, xgboost has remained on top because it is faster and requires **less computational infrastructure** than very complex and deeper neural networks. 
# 
# 
# ## 6. What Kagglers are using for Data Visualizations ?

# In[ ]:


models = ["matplotlib", "seaborn", "plotly"]
for col in models:
    messages[col] = messages["CMessage"].apply(lambda x : check_presence(x, col))
plotthem(models, "Kaggle Discussions: Python Data Visualization Libraries")    

models = ["ggplot", "highchart", "leaflet"]
for col in models:
    messages[col] = messages["CMessage"].apply(lambda x : check_presence(x, col))
plotthem(models, "Kaggle Discussions: R Data Visualization Libraries")    

# > - Plotly has gained so much popularity since 2017 and is one of the most used data visualization library among the kernels. The second best is seaborn which is used extensively as well. Some of the high quality visualization kernels by kaggle grandmasters such as SRK and Anistropic are created with plotly. Personally, I am a big fan of plotly as well. :P
# 
# ## 7. Important Data Science Techniques 

# In[ ]:


models = ["exploration|explore|eda" , 'feature engineering', 'parameter tuning|hyperparameter tuning|model tuning|tuning', "ensembling|ensemble"]
for col in models:
    messages[col] = messages["CMessage"].apply(lambda x : check_presence(x, col))
plotthem(models, "Kaggle Discussions: Important Data Science Techniques")    

# > - Among the important data science steps, kagglers focus alot on **Model Ensembling** since many winning solutions on kaggle competitions are ensemble models - the blends and stacked models.  In almost every regression or classification kernels, one can notice the ensemblling kernels. Just for an example - in Toxic Comment Classification Competition, massively large number of ensemling kernels were shared.   
# > - **Data Exploration** is the important technique and people have started stressing on the importance of exploration in the EDA kernels.  
# > - Surprizing to see that discussions related to **Feature Engineering and Model Tuninig are less than Ensembling**. These two tasks have the most important significance in the best and accurate models.  People tend to forget that ensembling is only the last stage of any modelling process but a considerable amount of time should be given to feature engineering and model tuning tasks.  
# 
# ## 8. Kaggle Components : What people talks about the most 

# In[ ]:


models = ["dataset" , 'kernel', 'competition', 'learn']
for col in models:
    messages[col] = messages["CMessage"].apply(lambda x : check_presence(x, col))
plotthem(models, "What is hottest on Kaggle")    

# > - Kaggle communitiy has shared a number of competition related discussions in fourms and are increasing in general.  
# > - With the launch of kernels in 2016, their useage increased to a great extent. Firstly kagglers shared kernels in competitions only, but with a more focus on **kaggle datasets, kernel awards**, the number of discussions related to kernels started rising and have surpassed the discussions related to competitions.  Also, a number of **Data Science for Good Challenges** and **Kernels only competitions** have been launched on kaggle which are one of the reason of kernels popularity. 
# > - Kaggle also launched the awesome **Kaggle Learn** section which is becoming popular and popular but still it is behind than the compeitions, kernels, and discussions. This is because its primarily audience is the novice and begineers, but for sure in coming years and with the more addition of courses, kaggle learn section will reach the similar levels as competitions and kernels. 
