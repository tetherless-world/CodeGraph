#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from IPython.core.display import HTML

path = "../input/"

versions = pd.read_csv(path+"KernelVersions.csv")
kernels = pd.read_csv(path+"Kernels.csv")
users = pd.read_csv(path+"Users.csv")

language_map = {'1' : 'R','5' : 'R', '12' : 'R', '13' : 'R', '15' : 'R', '16' : 'R',
                '2' : 'Python','8' : 'Python', '9' : 'Python', '14' : 'Python'}

def pressence_check(title, tokens, ignore = []):
    present = False
    for token in tokens:
        words = token.split()
        if all(wrd.lower().strip() in title.lower() for wrd in words):
            present = True
    for token in ignore:
        if token in title.lower():
            present = False
    return present 

## check if the latest version of the kernel is about the same topic 
def get_latest(idd):
    latest = versions[versions['KernelId'] == idd].sort_values('VersionNumber', ascending = False).iloc(0)[0]
    return latest['VersionNumber']

def get_kernels(tokens, n, ignore = []):
    versions['isRel'] = versions['Title'].apply(lambda x : pressence_check(x, tokens, ignore))
    relevant = versions[versions['isRel'] == 1]
    results = relevant.groupby('KernelId').agg({'TotalVotes' : 'sum', 
                                                'KernelLanguageId' : 'max', 
                                                'Title' : lambda x : "#".join(x).split("#")[-1],
                                                'VersionNumber' : 'max'})
    results = results.reset_index().sort_values('TotalVotes', ascending = False).head(n)
    results = results.rename(columns={'KernelId' : 'Id', 'TotalVotes': 'Votes'})


    results['latest_version']  = results['Id'].apply(lambda x : get_latest(x))
    results['isLatest'] = results.apply(lambda r : 1 if r['VersionNumber'] == r['latest_version'] else 0, axis=1)
    results = results[results['isLatest'] == 1]

    results = results.merge(kernels, on="Id").sort_values('TotalVotes', ascending = False)
    results = results.merge(users.rename(columns={'Id':"AuthorUserId"}), on='AuthorUserId')
    results['Language'] = results['KernelLanguageId'].apply(lambda x : language_map[str(x)] if str(x) in language_map else "")
    results = results.sort_values("TotalVotes", ascending = False)
    return results[['Title', 'CurrentUrlSlug','Language' ,'TotalViews', 'TotalComments', 'TotalVotes', "DisplayName","UserName"]]


def best_kernels(tokens, n = 10, ignore = [], idd = "one"):
    response = get_kernels(tokens, n, ignore)     
    hs = """<style>
                .rendered_html tr {font-size: 12px; text-align: left}
            </style>
            <h3 id='"""+ idd +"""'><font color="#1768ea">"""+tokens[0].title()+"""</font></h3>
            <table>
            <th>
                <td><b>Kernel</b></td>
                <td><b>Author</b></td>
                <td><b>Language</b></td>
                <td><b>Views</b></td>
                <td><b>Comments</b></td>
                <td><b>Votes</b></td>
            </th>"""
    for i, row in response.iterrows():
        url = "https://www.kaggle.com/"+row['UserName']+"/"+row['CurrentUrlSlug']
        aurl= "https://www.kaggle.com/"+row['UserName']
        hs += """<tr>
                    <td>"""+str(i+1)+"""</td>
                    <td><a href="""+url+""" target="_blank"><b>"""  + row['Title'] + """</b></a></td>
                    <td><a href="""+aurl+""" target="_blank">"""  + row['DisplayName'] + """</a></td>
                    <td>"""+str(row['Language'])+"""</td>
                    <td>"""+str(row['TotalViews'])+"""</td>
                    <td>"""+str(row['TotalComments'])+"""</td>
                    <td>"""+str(row['TotalVotes'])+"""</td>
                    </tr>"""
    hs += "</table>"
    display(HTML(hs))

# # Data Science Glossary on Kaggle 
# 
# Kaggle is the place to do data science projects. There are so many algorithms and concepts to learn. Kaggle Kernels are one of the best resources on internet to understand the practical implementation of algorithms. There are almost 200,000 kernels published on kaggle and sometimes it becomes diffcult to search for the right implementation. I have used the [Meta Kaggle](https://www.kaggle.com/kaggle/meta-kaggle) database to create a glossary of data science models, techniques and tools shared on kaggle kernels. One can use this kernel as the one place to find other great kernels shared by great authors. Hope you like this kernel. 
# 
# ## Contents
# 
# <ul>
# 	<li>1. Regression Algorithms
# 		<ul>
# 			<li><a href="#1.1">1.1 Linear Regression</a></a></li>
# 			<li><a href="#1.2">1.2 Logistic Regression</a></li>
# 		</ul>
# 	</li>
# 	<li>2. Regularization Algorithms
# 		<ul>
# 			<li><a href="#2.1">2.1 Ridge Regression Regression</a></li>
# 			<li><a href="#2.2">2.2 Lasso Regression</a></li>
# 			<li><a href="#2.3">2.3 Elastic Net</a></li>
# 		</ul>
# 	</li>
# 	<li>3. Tree Based Models
# 		<ul>
# 			<li><a href="#3.1">3.1 Decision Tree</a></li>
# 			<li><a href="#3.2">3.2 Random Forests</a></li>
# 			<li><a href="#3.3">3.3 Lightgbm</a></li>
# 			<li><a href="#3.4">3.4 XgBoost</a></li>
# 			<li><a href="#3.5">3.5 Cat Boost</a></li>
# 			<li><a href="#3.6">3.6 Gradient Boosting</a></li>
# 		</ul>
# 	</li>
# 	<li>4. Neural Networks and Deep Learning
# 		<ul>
# 			<li><a href="#4.1">4.1 Neural Networks</a></li>
# 			<li><a href="#4.2">4.2 AutoEncoders</a></li>
# 			<li><a href="$4.3">4.3 DeepLearning</a></li>
# 			<li><a href="#4.4">4.4 Convolutional Neural Networks / CNN</a></li>
# 			<li><a href="#4.5">4.5 Recurrent Neural Networks / RNN</a></li>
# 			<li><a href="#4.6">4.6 LSTMs</a></li>
# 			<li><a href="#4.7">4.7 GRUs</a></li>
# 			<li><a href="#4.8">4.8 MxNet</a></li>
# 			<li><a href="#4.9">4.9 ResNet</a></li>
# 			<li><a href="#4.10">4.10 CapsuleNets</a></li>
# 			<li><a href="#4.11">4.11 Unet</a></li>
# 			<li><a href="#4.12">4.12 VGGs</a></li>
# 			<li><a href="#4.13">4.13 Unet</a></li>
# 			<li><a href="#4.14">4.14 Xception</a></li>
# 			<li><a href="#4.15">4.15 Inception Nets</a></li>
# 			<li><a href="#4.16">4.16 Computer Vision</a></li>
# 			<li><a href="#4.17">4.17 Transfer Learning</a></li>
# 			<li><a href="#4.18">4.18 Object Detection</a></li>
# 			<li><a href="#4.19">4.19 Object Detection</a></li>
# 			<li><a href="#4.20">4.20 RCNN</a></li>
# 			<li><a href="#4.21">4.21 MobileNet</a></li>
# 		</ul>
# 	</li>
# 	<li>5. Clustering Algorithms
# 		<ul>
# 			<li><a href="#5.1">5.1 K Means Clustering</a></li>
# 			<li><a href="#5.2">5.2 Hierarchial Clustering</a></li>
# 			<li><a href="#5.3">5.3 DB Scan</a></li>
# 			<li><a href="#5.4">5.4 Unsupervised Learning</a></li>
# 		</ul>
# 	</li>
# 	<li>6. Misc - Models
# 		<ul>
# 			<li><a href="#6.1">6.1 K Naive Bayes</a></li>
# 			<li><a href="#6.2">6.2 SVMs</a></li>
# 			<li><a href="#6.3">6.3 KNN</a></li>
# 			<li><a href="#6.4">6.4 Recommendation Engine</a></li>
# 		</ul>
# 	</li>
# 	<li>7.1 Data Science Techniques - Preprocessing
# 		<ul>
# 			<li><a href="#7.1.a">a. EDA, Exploration</a></li>
# 			<li><a href="#7.1.b">b. Feature Engineering</a></li>
# 			<li><a href="#7.1.c">c. Feature Selection</a></li>
# 			<li><a href="#7.1.d">d. Outlier Treatment</a></li>
# 			<li><a href="#7.1.e">e. Anomaly Detection</a></li>
# 			<li><a href="#7.1.f">f. SMOTE</a></li>
# 			<li><a href="#7.1.g">g. Pipeline</a></li>
# 			<li><a href="#7.1.h">h. Missing Values</a></li>
# 		</ul>
# 	</li>
# 	<li>7.2 Data Science Techniques - Dimentionality Reduction
# 		<ul>
# 			<li><a href="#7.2.a">a. Dataset Decomposition</a></li>
# 			<li><a href="#7.2.b">b. PCA</a></li>
# 			<li><a href="#7.2.c">c. Tsne</a></li>
# 			<li><a href="#7.2.d">d. SVD</a></li>
# 		</ul>
# 	</li>
# 	<li>7.3 Data Science Techniques - Post Modelling
# 		<ul>
# 			<li><a href="#7.3.a">a. Cross Validation</a></li>
# 			<li><a href="#7.3.b">b. Model Selection</a></li>
# 			<li><a href="#7.3.c">c. Model Tuning</a></li>
# 			<li><a href="#7.3.d">d. Grid Search</a></li>
# 		</ul>
# 	</li>
# 	<li>7.4 Data Science Techniques - Ensemblling
# 		<ul>
# 			<li><a href="#7.4.a">a. Ensembling</a></li>
# 			<li><a href="#7.4.b">b. Stacking</a></li>
# 			<li><a href="#7.4.c">c. Bagging</a></li>
# 			<li><a href="#7.4.d">d. Blending</a></li>
# 		</ul>
# 	</li>
# 	<li>8. Text Data
# 		<ul>
# 			<li><a href="#8.1">8.1. NLP</a></li>
# 			<li><a href="#8.2">8.2. Topic Modelling</a></li>
# 			<li><a href="#8.3">8.3. Word Embeddings</a></li>
# 			<li><a href="#8.4">8.4. Spacy</a></li>
# 			<li><a href="#8.5">8.5. NLTK</a></li>
# 			<li><a href="#8.6">8.6. TextBlob</a></li>
# 		</ul>
# 	</li>
# 	<li>9. Data Science Tools
# 		<ul>
# 			<li><a href="#9.1">9.1 Scikit Learn</a></li>
# 			<li><a href="#9.2">9.2 TensorFlow</a></li>
# 			<li><a href="#9.3">9.3 Theano</a></li>
# 			<li><a href="#9.4">9.4 Kears</a></li>
# 			<li><a href="#9.5">9.5 PyTorch</a></li>
# 			<li><a href="#9.6">9.6 Vopal Wabbit</a></li>
# 			<li><a href="#9.7">9.7 ELI5</a></li>
# 			<li><a href="#9.8">9.8 HyperOpt</a></li>
# 			<li><a href="#9.9">9.9 Pandas</a></li>
# 			<li><a href="#9.10">9.10 Sql</a></li>
# 			<li><a href="#9.11">9.11 BigQuery</a></li>
# 			<li><a href="#9.12">9.12 GPU</a></li>
# 			<li><a href="#9.13">9.12 H2o</a></li>
# 			<li><a href="#9.14">9.13 Fast.AI</a></li>
# 		</ul>
# 	</li>
# 	<li>10. Data Visualizations
# 		<ul>
# 			<li><a href="#10.1">10.1. Visualizations</a></li>
# 			<li><a href="#10.2">10.2. Plotly</a></li>
# 			<li><a href="#10.3">10.3. Seaborn</a></li>
# 			<li><a href="#10.4">10.4. D3.Js</a></li>
# 			<li><a href="#10.5">10.5. Bokeh</a></li>
# 			<li><a href="#10.6">10.6. Highchart</a></li>
# 			<li><a href="#10.7">10.7. Folium</a></li>
# 			<li><a href="#10.8">10.8. ggPlot</a></li>
# 		</ul>
# 	</li>
# 	<li>11. Time Series
# 		<ul>
# 			<li><a href="#11.1">11.1. Time Series Analysis</a></li>
# 			<li><a href="#11.2">11.2. ARIMA</a></li>
# 			<li><a href="#11.3">11.3. Forecasting</a></li>
# 		</ul>
# 	</li>
# 	<li>12. Misc Materials</a></li>
# 		<ul>
# 			<li><a href="#12.1">12.1. Best Tutorials on Kaggle</a></li>
# 			<li><a href="#12.2">12.2. Data Leak</a></li>
# 			<li><a href="#12.3">12.3. Adversarial Validation</a></li>
# 			<li><a href="#12.4">12.4. Generative Adversarial Networks</a></li>
# 		</ul>
# </ul>
# 
# <br>
# <br>
# 
# ## 1. Regression Algorithms

# In[ ]:


tokens = ["linear regression"]
best_kernels(tokens, 10, idd="1.1")

# In[ ]:


tokens = ['logistic regression', "logistic"]
best_kernels(tokens, 10, idd="1.2")

# ## 2. Regularization Algorithms

# In[ ]:


tokens = ['Ridge']
best_kernels(tokens, 10, idd="2.1")

# In[ ]:


tokens = ['Lasso']
best_kernels(tokens, 10, idd="2.2")

# In[ ]:


tokens = ['ElasticNet']
best_kernels(tokens, 4, idd="2.3")

# ## 3. Tree Based Models

# In[ ]:


tokens = ['Decision Tree']
best_kernels(tokens, 10, idd="3.1")

# In[ ]:


tokens = ['random forest']
best_kernels(tokens, 10, idd="3.2")

# In[ ]:


tokens = ['lightgbm', 'light gbm', 'lgb']
best_kernels(tokens, 10, idd="3.3")

# In[ ]:


tokens = ['xgboost', 'xgb']
best_kernels(tokens, 10, idd="3.4")

# In[ ]:


tokens = ['catboost']
best_kernels(tokens, 10, idd="3.5")

# In[ ]:


tokens = ['gradient boosting']
best_kernels(tokens, 10, idd="3.6")

# ## 4. Neural Networks and Deep Learning Models

# In[ ]:


tokens = ['neural network']
best_kernels(tokens, 10, idd="4.1")

# In[ ]:


tokens = ['autoencoder']
best_kernels(tokens, 10, idd="4.2")

# In[ ]:


tokens = ['deep learning']
best_kernels(tokens, 10, idd="4.3")

# In[ ]:


tokens = ['convolutional neural networks', 'cnn']
best_kernels(tokens, 10, idd="4.4")

# In[ ]:


tokens = ['recurrent','rnn']
best_kernels(tokens, 10, idd="4.5")

# In[ ]:


tokens = ['lstm']
best_kernels(tokens, 10, idd="4.6")

# In[ ]:


tokens = ['gru']
ignore = ['grupo']
best_kernels(tokens, 10, ignore, idd="4.7")

# In[ ]:


tokens = ['mxnet']
best_kernels(tokens, 10, idd="4.8")

# In[ ]:


tokens = ['resnet']
best_kernels(tokens, 10, idd="4.9")

# In[ ]:


tokens = ['Capsule network', 'capsulenet']
best_kernels(tokens, 5, idd="4.10")

# In[ ]:


tokens = ['vgg']
best_kernels(tokens, 5, idd="4.11")

# In[ ]:


tokens = ['unet']
best_kernels(tokens, 10, idd="4.12")

# In[ ]:


tokens = ['alexnet']
best_kernels(tokens, 5, idd="4.13")

# In[ ]:


tokens = ['xception']
best_kernels(tokens, 5, idd="4.14")

# In[ ]:


tokens = ['inception']
best_kernels(tokens, 5, idd="4.15")

# In[ ]:


tokens = ['computer vision']
best_kernels(tokens, 5, idd="4.16")

# In[ ]:


tokens = ['transfer']
best_kernels(tokens, 10, idd="4.17")

# In[ ]:


tokens = ['yolo']
best_kernels(tokens, 5, idd="4.18")

# In[ ]:


tokens = ['object detection']
best_kernels(tokens, 5, idd="4.19")

# In[ ]:


tokens = ['rcnn']
best_kernels(tokens, 5, idd="4.20")

# In[ ]:


tokens = ['mobilenet']
best_kernels(tokens, 5, idd="4.21")

# ## 5. Clustering Algorithms 

# In[ ]:


tokens = ['kmeans', 'k means']
best_kernels(tokens, 10, idd="5.1")

# In[ ]:


tokens = ['hierarchical clustering']
best_kernels(tokens, 3, idd="5.2")

# In[ ]:


tokens = ['dbscan']
best_kernels(tokens, 10, idd="5.3")

# In[ ]:


tokens = ['unsupervised']
best_kernels(tokens, 10, idd="5.4")

# ## 6. Misc - Models 

# In[ ]:


tokens = ['naive bayes']
best_kernels(tokens, 10, idd="6.1")

# In[ ]:


tokens = ['svm']
best_kernels(tokens, 10, idd="6.2")

# In[ ]:


tokens = ['knn']
best_kernels(tokens, 10, idd="6.3")

# In[ ]:


tokens = ['recommendation engine']
best_kernels(tokens, 5, idd="6.4")

# ## 7. Important Data Science Techniques

# ### 7.1 Preprocessing

# In[ ]:


tokens = ['EDA', 'exploration', 'exploratory']
best_kernels(tokens, 10, idd="7.1.a")

# In[ ]:


tokens = ['feature engineering']
best_kernels(tokens, 10, idd="7.1.b")

# In[ ]:


tokens = ['feature selection']
best_kernels(tokens, 10, idd="7.1.c")

# In[ ]:


tokens = ['outlier treatment', 'outlier']
best_kernels(tokens, 10, idd="7.1.d")

# In[ ]:


tokens = ['anomaly detection', 'anomaly']
best_kernels(tokens, 8, idd="7.1.e")

# In[ ]:


tokens = ['smote']
best_kernels(tokens, 5, idd="7.1.f")

# In[ ]:


tokens = ['pipeline']
best_kernels(tokens, 10, idd="7.1.g")

# In[ ]:


tokens = ['missing value']
best_kernels(tokens, 10, idd="7.1.h")

# ### 7.2 Dimentionality Reduction

# In[ ]:


tokens = ['dataset decomposition', 'dimentionality reduction']
best_kernels(tokens, 2, idd="7.2.a")

# In[ ]:


tokens = ['PCA']
best_kernels(tokens, 10, idd="7.2.b")

# In[ ]:


tokens = ['Tsne', 't-sne']
best_kernels(tokens, 10, idd="7.2.c")

# In[ ]:


tokens = ['svd']
best_kernels(tokens, 10, idd="7.2.d")

# ### 7.3 Post Modelling Techniques

# In[ ]:


tokens = ['cross validation']
best_kernels(tokens, 10, idd="7.3.a")

# In[ ]:


tokens = ['model selection']
best_kernels(tokens, 10, idd="7.3.b")

# In[ ]:


tokens = ['model tuning', 'tuning']
best_kernels(tokens, 10, idd="7.3.c")

# In[ ]:


tokens = ['gridsearch', 'grid search']
best_kernels(tokens, 10, idd="7.3.d")

# ### 7.4 Ensemblling

# In[ ]:


tokens = ['ensemble']
best_kernels(tokens, 10, idd="7.4.a")

# In[ ]:


tokens = ['stacking', 'stack']
best_kernels(tokens, 10, idd="7.4.b")

# In[ ]:


tokens = ['bagging']
best_kernels(tokens, 10, idd="7.4.c")

# In[ ]:


tokens = ['blend']
best_kernels(tokens, 10, idd="7.4.d")

# ## 8. Text Data

# In[ ]:


tokens = ['NLP', 'Natural Language Processing', 'text mining']
best_kernels(tokens, 10, idd="8.1")

# In[ ]:


tokens = ['topic modelling', 'lda']
best_kernels(tokens, 8, idd="8.2")

# In[ ]:


tokens = ['word embedding','fasttext', 'glove', 'word2vec', 'word vector']
best_kernels(tokens, 8, idd="8.3")

# In[ ]:


tokens = ['spacy']
best_kernels(tokens, 10, idd="8.4")

# In[ ]:


tokens = ['nltk']
best_kernels(tokens, 5, idd="8.5")

# In[ ]:


tokens = ['textblob']
best_kernels(tokens, 5, idd="8.6")

# ## 9. Data Science Tools

# In[ ]:


tokens = ['scikit']
best_kernels(tokens, 10, idd="9.1")

# In[ ]:


tokens = ['tensorflow', 'tensor flow']
best_kernels(tokens, 10, idd="9.2")

# In[ ]:


tokens = ['theano']
best_kernels(tokens, 10, idd="9.3")

# In[ ]:


tokens = ['keras']
best_kernels(tokens, 10, idd="9.4")

# In[ ]:


tokens = ['pytorch']
best_kernels(tokens, 10, idd="9.5")

# In[ ]:


tokens = ['vowpal wabbit','vowpalwabbit']
best_kernels(tokens, 10, idd="9.6")

# In[ ]:


tokens = ['eli5']
best_kernels(tokens, 10, idd="9.7")

# In[ ]:


tokens = ['hyperopt']
best_kernels(tokens, 5, idd="9.8")

# In[ ]:


tokens = ['pandas']
best_kernels(tokens, 10, idd="9.9")

# In[ ]:


tokens = ['SQL']
best_kernels(tokens, 10, idd="9.10")

# In[ ]:


tokens = ['bigquery', 'big query']
best_kernels(tokens, 10, idd="9.11")

# In[ ]:


tokens = ['gpu']
best_kernels(tokens, 10, idd="9.12")

# In[ ]:


tokens = ['h20']
best_kernels(tokens, 5, idd="9.13")

# In[ ]:


tokens = ['fastai', 'fast.ai']
best_kernels(tokens, 10, idd="9.14")

# ## 10. Data Visualization

# In[ ]:


tokens = ['visualization', 'visualisation']
best_kernels(tokens, 10, idd="10.1")

# In[ ]:


tokens = ['plotly', 'plot.ly']
best_kernels(tokens, 10, idd="10.2")

# In[ ]:


tokens = ['seaborn']
best_kernels(tokens, 10, idd="10.3")

# In[ ]:


tokens = ['d3.js']
best_kernels(tokens, 4, idd="10.4")

# In[ ]:


tokens = ['bokeh']
best_kernels(tokens, 10, idd="10.5")

# In[ ]:


tokens = ['highchart']
best_kernels(tokens, 10, idd="10.6")

# In[ ]:


tokens = ['folium']
best_kernels(tokens, 5, idd="10.7")

# In[ ]:


tokens = ['ggplot']
best_kernels(tokens, 10, idd="10.8")

# ## 11. Time Series

# In[ ]:


tokens = ['time series']
best_kernels(tokens, 10, idd="11.1")

# In[ ]:


tokens = ['arima']
best_kernels(tokens, 10, idd="11.2")

# In[ ]:


tokens = ['forecasting']
best_kernels(tokens, 10, idd="11.3")

# ## 12. Misc Materials 
# 
# ### 12.1 Some of the Best Tutorials on Kaggle

# In[ ]:


tokens = ['tutorial']
best_kernels(tokens, 10, idd="12.1")

# ### 12.2 Data Leak

# In[ ]:


tokens = ['data leak', 'leak']
best_kernels(tokens, 10, idd="12.2")

# ### 12.3 Adversarial Validation

# In[ ]:


tokens = ["adversarial validation"]
best_kernels(tokens, 10, idd="12.3")

# ### 12.4 Generative Adversarial Networks

# In[ ]:


tokens = ["generative adversarial networks", "simgan", "-gan"]
best_kernels(tokens, 10, idd="12.4")

# <br>
# Thanks for viewing. Suggest the list of items which can be added to the list. If you liked this kernel, **please upvote**   
# 
