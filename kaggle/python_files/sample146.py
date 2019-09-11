#!/usr/bin/env python
# coding: utf-8

# <h1><b>SEABORN TUTORIAL FOR BEGINNERS</b></h1>

# <h3><b>Content</b></h3>
# <ul>
#     <a href='#1'><li>Introduction</li></a>
#     <a href='#2'><li>Import Library</li></a>
#     <a href='#3'><li>Data Exploratory Analysis</li></a>
#     <a href='#4'><li>Seaborn</li></a>
#         <ul>
#             <a href='#5'><li>Bar Plot</li></a>
#             <a href='#6'><li>Point Plot</li></a>
#             <a href='#7'><li>Joint Plot</li></a>
#             <a href='#8'><li>Pie Chart</li></a>
#             <a href='#9'><li>Lm Plot</li></a>
#             <a href='#10'><li>Kde Plot</li></a>
#             <a href='#11'><li>Violin Plot</li></a>
#             <a href='#12'><li>Heatmap Plot</li></a>
#             <a href='#13'><li>Box Plot</li></a>
#             <a href='#14'><li>Swarm Plot</li></a>
#             <a href='#15'><li>Pair Plot</li></a>
#             <a href='#16'><li>Count Plot</li></a>
#             <a href='#17'><li>FacetGrid</li></a>
#             <a href='#18'><li>Strip Plot</li></a>
#             <a href='#19'><li>Factor Plot</li></a>
#             <a href='#22'><li>DisPlot</li></a>
#              <a href='#23'><li>Line Plot</li></a>
#             <a href='#24'><li>Despine</li></a>
#         </ul>
#    <a href='#20'><li>References</li></a>
#    <a href='#21'><li>Conclusion</li></a>
# </ul>
# 
# <p>Last Updated: <b>12/05/2019</b></p>
# <p><h2>If you like it, please upvote.</h2></p>

# <p id='1'><h3><b>Introduction</b></h3></p>

# ****<p>Hello to everyone,<br>
# In this kernel we will introduce the <b>seaborn</b> library. For this, analyzes and a wide variety of graphs will be generated from a specific data set.</p>
# 
# <p>This data set consists of the marks secured by the students in various subjects.</p>
# <p>Column List</p>
# <ul>
#     <li>gender</li>
#     <li>race/ethnicity</li>
#     <li>parental level of education</li>
#     <li>test preparation course</li>
#     <li>lunch</li>
#     <li>math score</li>
#     <li>reading score</li>
#     <li>writing score</li>
# </ul>
# 
# 

# <p id='2'><h3><b>Import Library</b></h3></p>
# <p>We need to install a wide variety of libraries. For this we will install <b>pandas, numpy, seaborn and matplotlib</b> libraries.</p>

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
print(os.listdir("../input"))
import warnings
warnings.filterwarnings("ignore")

# <p id='3'><h3><b>Data Exploratory Analysis</b></h3></p>
# <p>In the data discovery analysis, we will firstly recognize and analyze our data using a wide variety of functions in the pandas library.</p>

# In[ ]:


data=pd.read_csv('../input/StudentsPerformance.csv')
#read csv for analysis

# In[ ]:


#we'll see the first five lines.
data.head()

# In[ ]:


#we'll see the last five lines.
data.tail()

# In[ ]:


#random data 
data.sample(5)

# In[ ]:


data.sample(frac=0.1)

# In[ ]:


#it is a process that shows the property value in the data set and shows the numbers in the register values.
data.info()

# In[ ]:


data.iloc[:,0:3].dtypes

# **<ul>
#     <li>Count : Shows the total number.</li>
#     <li>Mean  : Shows the average.</li>
#     <li>Std   :  Standard deviation value</li>
#     <li>Min   : Minimum value</li>
#     <li>%25   : First Quantile</li>
#     <li>%50   : Median or Second Quantile</li>
#     <li>%75   : Third Quantile</li>
#     <li>Max   : Maximum value</li>
# </ul>
# 
# <p>What is quantile?</p>
# <ul>
#     <li>1,4,5,6,7,11,12,13,14,15,16,17</li>
#     <li>The median is the number that is in middle of the sequence. In this case It would be 11</li>
#     <li>The lower quartile is the median in between the smallest number and the median etc in between 1 and 11, which is 6</li>
#     <li>The upper quartile you find the median between the median and the largest number etc. betweeb 11 and 17,which will be 14 according to the question above.</li>
# </ul>

# In[ ]:


#It is a function that shows the analysis of numerical values.
data.describe()

# In[ ]:


#It shows the data types in the data set.
data.dtypes

# In[ ]:


#It is a function that shows the analysis of proximity values between data.
data.corr()

# In[ ]:


data.iloc[:,1:].corr()

# In[ ]:


#control data
data.isnull().values.any()

# In[ ]:


#all data control for null values
data.isnull().sum()

# In[ ]:


#show columns
for i,col in enumerate(data.columns):
    print(i+1,". column is ",col)

# In[ ]:


#rename columns
data.rename(columns=({'gender':'Gender','race/ethnicity':'Race/Ethnicity'
                     ,'parental level of education':'Parental_Level_of_Education'
                     ,'lunch':'Lunch','test preparation course':'Test_Preparation_Course'
                      ,'math score':'Math_Score','reading score':'Reading_Score'
                     ,'writing score':'Writing_Score'}),inplace=True)

# In[ ]:


#show columns
for i,col in enumerate(data.columns):
    print(i+1,". column is ",col)

# In[ ]:


#show count Gender
data['Gender'].value_counts()

# In[ ]:


#show Gender's unique
data['Gender'].unique()

# <p id='4'><h3><b>Seaborn</b></h3></p>
# <p>Seaborn is a Python data visualization library based on matplotlib. It provides a high-level interface for drawing attractive and informative statistical graphics.
# 
# For a brief introduction to the ideas behind the library, you can read the introductory notes. Visit the installation page to see how you can download the package. You can browse the example gallery to see what you can do with seaborn, and then check out the tutorial and API reference to find out how.
# 
# To see the code or report a bug, please visit the github repository. General support issues are most at home on stackoverflow, where there is a seaborn tag.</p>

# <p id='5'><h3><b>Bar Plot</b></h3></p>
# 
# <p>seaborn.barplot(x=None, y=None, hue=None, data=None, order=None, hue_order=None, estimator=<function mean>, ci=95, n_boot=1000, units=None, orient=None, color=None, palette=None, saturation=0.75, errcolor='.26', errwidth=None, capsize=None, dodge=True, ax=None, **kwargs)</p>
# 
# <ul>
#     <li>x,y,hue : names of variable in data or vector data</li>
#     <li>data : DataFrame,array or list of array,optional</li>
#     <li>color :matplotlib color,optional</li>
#     <li>palette : palette name,list, or dict,optional</li>
#     <li>ax : matplotlib Axes,optional</li>
# </ul>

# In[ ]:


#Gender show bar plot
sns.set(style='whitegrid')
ax=sns.barplot(x=data['Gender'].value_counts().index,y=data['Gender'].value_counts().values,palette="Blues_d",hue=['female','male'])
plt.legend(loc=8)
plt.xlabel('Gender')
plt.ylabel('Frequency')
plt.title('Show of Gender Bar Plot')
plt.show()

# In[ ]:


sns.barplot(x=data['Gender'].value_counts().index,y=data['Gender'].value_counts().values)
plt.title('Genders other rate')
plt.ylabel('Rates')
plt.show()

# In[ ]:


plt.figure(figsize=(7,7))
sns.barplot(x=data['Race/Ethnicity'].value_counts().index,
              y=data['Race/Ethnicity'].value_counts().values)
plt.xlabel('Race/Ethnicity')
plt.ylabel('Frequency')
plt.title('Show of Race/Ethnicity Bar Plot')
plt.show()

# In[ ]:


data.head()

# In[ ]:


plt.figure(figsize=(10,7))
sns.barplot(x = "Parental_Level_of_Education", y = "Writing_Score", hue = "Gender", data = data)
plt.xticks(rotation=45)
plt.show()

# In[ ]:


sns.barplot(x = "Parental_Level_of_Education", y = "Reading_Score", hue = "Gender", data = data)
plt.xticks(rotation=45)
plt.show()

# In[ ]:


sns.barplot(x = "Parental_Level_of_Education", y = "Math_Score", hue = "Gender", data = data)
plt.xticks(rotation=45)
plt.show()

# In[ ]:


plt.figure(figsize=(12,7))
sns.catplot(y="Gender", x="Math_Score",
                 hue="Parental_Level_of_Education",
                 data=data, kind="bar")
plt.title('for Parental Level Of Education Gender & Math_Score')
plt.show()

# In[ ]:


plt.figure(figsize=(10,10))
sns.catplot(x="Gender", y="Math_Score",
                 hue="Test_Preparation_Course",
                 data=data, kind="bar")
plt.title('for Test Preparation Course Gender & Math_Score')
plt.show()

# In[ ]:


ax = sns.barplot("Parental_Level_of_Education", "Writing_Score", data=data,
                  linewidth=2.5, facecolor=(1, 1, 1, 0),
                  errcolor=".2", edgecolor=".2")
plt.xticks(rotation=90)
plt.show()

# In[ ]:


# Draw a nested barplot to show survival for class and sex
plt.figure(figsize=(20,10))
g = sns.catplot(x="Math_Score", y="Writing_Score", hue="Gender", data=data,kind="bar")
g.set_ylabels("survival probability")
plt.tight_layout()
plt.show()

# In[ ]:


f,ax=plt.subplots(figsize=(9,10))
sns.barplot(x=data['Gender'].value_counts().values,y=data['Gender'].value_counts().index,alpha=0.5,color='red',label='Gender')
sns.barplot(x=data['Race/Ethnicity'].value_counts().values,y=data['Race/Ethnicity'].value_counts().index,color='blue',alpha=0.7,label='Race/Ethnicity')
ax.legend(loc='upper right',frameon=True)
ax.set(xlabel='Gender , Race/Ethnicity',ylabel='Groups',title="Gender vs Race/Ethnicity ")
plt.show()

# <p id='6'><h3><b>Point Plot</b></h3></p>
# <p>seaborn.pointplot(x=None, y=None, hue=None, data=None, order=None, hue_order=None, estimator=<function mean>, ci=95, n_boot=1000, units=None, markers='o', linestyles='-', dodge=False, join=True, scale=1, orient=None, color=None, palette=None, errwidth=None, capsize=None, ax=None, **kwargs)</p>
# 
# <ul>
#     <li>x, y, hue : names of variables in data or vector data, optional</li>
#     <li>data : DataFrame, array, or list of arrays, optional</li>
#     <li>order, hue_order : lists of strings, optional</li>
#     <li>markers : string or list of strings, optional</li>
#     <li>linestyles : string or list of strings, optional</li>
#     <li>color : matplotlib color, optional</li>
#     <li>palette : palette name, list, or dict, optional</li>
#     <li>ax : matplotlib Axes, optional</li>
# </ul>
# 
# 

# In[ ]:


#Gender show point plot
data['Race/Ethnicity'].unique()
len(data[(data['Race/Ethnicity']=='group B')].Math_Score)
f,ax1=plt.subplots(figsize=(25,10))
sns.pointplot(x=np.arange(1,191),y=data[(data['Race/Ethnicity']=='group B')].Math_Score,color='lime',alpha=0.8)
sns.pointplot(x=np.arange(1,191),y=data[(data['Race/Ethnicity']=='group B')].Reading_Score,color='red',alpha=0.5)
#sns.pointplot(x=np.arange(1,191),y=data[(data['Race/Ethnicity']=='group B')].Math_Score,color='lime',alpha=0.8)
plt.xlabel('Group B index State')
plt.ylabel('Frequency')
plt.title('Group B Math Score & Reading_Score')
plt.xticks(rotation=90)
plt.grid()
plt.show()

# In[ ]:


plt.figure(figsi)
ax = sns.pointplot(x="Reading_Score", y="Math_Score", hue="Gender",data=data)
plt.xticks(rotation=90)
plt.show()

# In[ ]:


ax = sns.pointplot(x="Reading_Score", y="Writing_Score", hue="Gender",data=data,markers=["o", "x"],linestyles=["-", "--"])
plt.xticks(rotation=90)
plt.show()

# <p id='7'><h3><b>Joint Plot</b></h3></p>
# <p>seaborn.jointplot(x, y, data=None, kind='scatter', stat_func=None, color=None, height=6, ratio=5, space=0.2, dropna=True, xlim=None, ylim=None, joint_kws=None, marginal_kws=None, annot_kws=None, **kwargs)</p>
# 
# <ul>
#     <li>x, y : strings or vectors</li>
#     <li>data : DataFrame, optional</li>
#     <li>kind : { “scatter” | “reg” | “resid” | “kde” | “hex” }, optional</li>
#     <li>color : matplotlib color, optional</li>
#     <li>dropna : bool, optional</li>
# </ul>

# In[ ]:


plt.figure(figsize=(10,10))
sns.jointplot(x=np.arange(1,191),y=data[(data['Race/Ethnicity']=='group B')].Math_Score,color='lime',alpha=0.8)
plt.xlabel('Group B index State')
plt.ylabel('Frequency')
plt.title('Group B Frequency Race/Ethnicity')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# In[ ]:


plt.figure(figsize=(10,10))
sns.jointplot(x=np.arange(1,191),y=data[(data['Race/Ethnicity']=='group B')].Math_Score,color='lime',kind='hex',alpha=0.8)
plt.xlabel('Group B index State')
plt.ylabel('Frequency')
plt.title('Group B Frequency Race/Ethnicity')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# In[ ]:


plt.figure(figsize=(10,10))
sns.jointplot(x=np.arange(1,191),y=data[(data['Race/Ethnicity']=='group B')].Math_Score,color='lime',space=0,kind='kde')
plt.xlabel('Group B index State')
plt.ylabel('Frequency')
plt.title('Group B Frequency Race/Ethnicity')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# In[ ]:


#Gender show point plot
data['Race/Ethnicity'].unique()
len(data[(data['Race/Ethnicity']=='group B')].Math_Score)
plt.figure(figsize=(10,10))
sns.jointplot(x=np.arange(1,191),y=data[(data['Race/Ethnicity']=='group B')].Reading_Score,color='k').plot_joint(sns.kdeplot, zorder=0, n_levels=6)
plt.xlabel('Group B index State')
plt.ylabel('Frequency')
plt.title('Group B Math Score & Reading_Score')
plt.xticks(rotation=90)
plt.show()

# In[ ]:


plt.figure(figsize=(10,10))
sns.jointplot(x=np.arange(1,191),y=data[(data['Race/Ethnicity']=='group B')].Reading_Score,color='lime',alpha=0.8)
plt.xlabel('Group B index State')
plt.ylabel('Frequency')
plt.title('Group B Frequency Race/Ethnicity')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

#  <p id='8'><h3><b>Pie Chart</b></h3></p>

# In[ ]:


labels=data['Race/Ethnicity'].value_counts().index
colors=['blue','red','yellow','green','brown']
explode=[0,0,0.1,0,0]
values=data['Race/Ethnicity'].value_counts().values

#visualization
plt.figure(figsize=(7,7))
plt.pie(values,explode=explode,labels=labels,colors=colors,autopct='%1.1f%%')
plt.title('Race/Ethnicity According Analysis',color='black',fontsize=10)
plt.show()


# In[ ]:


plt.figure(figsize=(4,4))
labels=['Math Score', 'Reading Score', 'Writing Score']
colors=['blue','red','yellow']
explode=[0,0,0.1]
values=[data.Math_Score.mean(),data.Reading_Score.mean(),data.Writing_Score.mean()]

plt.pie(values,labels=labels,colors=colors,explode=explode,autopct='%1.1f%%',shadow=True)
plt.legend(['Math Score', 'Reading Score', 'Writing Score'] , loc=3)
plt.axis('equal')
plt.tight_layout()
plt.show()

# In[ ]:


data.groupby('Race/Ethnicity')['Reading_Score'].mean()

# In[ ]:


# Data to plot
labels = 'group A', 'group B', 'group C', 'group D','group E'
sizes = data.groupby('Race/Ethnicity')['Reading_Score'].mean().values
colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue']
explode = (0.1, 0, 0, 0,0)  # explode 1st slice
 
# Plot
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
autopct='%1.1f%%', shadow=True, startangle=140)
plt.title('Reading Score for Every Race/Ethnicity Mean')
plt.axis('equal')
plt.show()

#  <p id='9'><h3><b>Lm Plot</b></h3></p>
#  <p>seaborn.lmplot(x, y, data, hue=None, col=None, row=None, palette=None, col_wrap=None, height=5, aspect=1, markers='o', sharex=True, sharey=True, hue_order=None, col_order=None, row_order=None, legend=True, legend_out=True, x_estimator=None, x_bins=None, x_ci='ci', scatter=True, fit_reg=True, ci=95, n_boot=1000, units=None, order=1, logistic=False, lowess=False, robust=False, logx=False, x_partial=None, y_partial=None, truncate=False, x_jitter=None, y_jitter=None, scatter_kws=None, line_kws=None, size=None)</p>
#  
#  <ul>
#      <li>x, y : strings, optional</li>
#      <li>data : DataFrame</li>
#      <li>hue, col, row : strings</li>
#      <li>palette : palette name, list, or dict, optional</li>
#      <li>markers : matplotlib marker code or list of marker codes, optional</li>
#      <li>legend : bool, optional</li>
#      <li>scatter : bool, optional</li>
#  </ul>

# In[ ]:


sns.lmplot(x='Math_Score',y='Reading_Score',data=data)
plt.xlabel('Math Score')
plt.ylabel('Reading Score')
plt.title('Math Score vs Reading Score')
plt.show()

# In[ ]:


sns.lmplot(x='Math_Score',y='Writing_Score',hue='Gender',data=data)
plt.xlabel('Math Score')
plt.ylabel('Writing Score')
plt.title('Math Score vs Writing Score')
plt.show()

# In[ ]:


sns.lmplot(x='Math_Score',y='Writing_Score',hue='Gender',data=data,markers=['x','o'])
plt.xlabel('Math Score')
plt.ylabel('Writing Score')
plt.title('Math Score vs Writing Score')
plt.show()

#  <p id='10'><h3><b>Kde Plot</b></h3></p>
#  <p>seaborn.kdeplot(data, data2=None, shade=False, vertical=False, kernel='gau', bw='scott', gridsize=100, cut=3, clip=None, legend=True, cumulative=False, shade_lowest=True, cbar=False, cbar_ax=None, cbar_kws=None, ax=None, **kwargs)</p>
#  <ul>
#      <li>data : 1d array-like</li>
#      <li>data2: 1d array-like, optional</li>
#      <li>shade : bool, optional</li>
#      <li>vertical : bool, optional</li>
#      <li>kernel : {‘gau’ | ‘cos’ | ‘biw’ | ‘epa’ | ‘tri’ | ‘triw’ }, optional</li>
#      <li>cut : scalar, optional</li>
#      <li>legend : bool, optional</li>
#      <li>ax : matplotlib axes, optional</li>
#  </ul>

# In[ ]:


sns.set(style="dark")
rs = np.random.RandomState(50)

# Set up the matplotlib figure
f, axes = plt.subplots(3, 3, figsize=(9, 9), sharex=True, sharey=True)

# Rotate the starting point around the cubehelix hue circle
for ax, s in zip(axes.flat, np.linspace(0, 3, 10)):

    # Create a cubehelix colormap to use with kdeplot
    cmap = sns.cubehelix_palette(start=s, light=1, as_cmap=True)

    # Generate and plot a random bivariate dataset
    x, y = rs.randn(2, 50)
    sns.kdeplot(x, y, cmap=cmap, shade=True, cut=5, ax=ax)
    ax.set(xlim=(-3, 3), ylim=(-3, 3))

f.tight_layout()
plt.show()

# In[ ]:


sns.kdeplot(data['Math_Score'])
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.title('Math Score Kde Plot System Analysis')
plt.show()

# In[ ]:


sns.kdeplot(data['Reading_Score'],shade=True,color='r')
sns.kdeplot(data['Writing_Score'],shade=True,color='b')
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.title('Reading Score vs Writing Score Kde Plot System Analysis')
plt.show()

# In[ ]:


sns.kdeplot(data['Reading_Score'],data['Writing_Score'])
plt.show()

# In[ ]:


sns.kdeplot(data['Reading_Score'],data['Writing_Score'],shade=True)
plt.show()

# In[ ]:


sns.kdeplot(data['Math_Score'],bw=.15)
plt.xlabel('Math Score')
plt.ylabel('Frequency')
plt.title('Math Score Show Kde Plot')
plt.show()

# In[ ]:


sns.kdeplot(data['Reading_Score'],data['Writing_Score'],cmap='Reds',shade=True,shade_lowest=False)
sns.kdeplot(data['Writing_Score'],data['Reading_Score'],cmap='Blues',shade=True,shade_lowest=False)
plt.show()

# <p id='11'><h3><b>Violin Plot</b></h3></p>
# <p>seaborn.violinplot(x=None, y=None, hue=None, data=None, order=None, hue_order=None, bw='scott', cut=2, scale='area', scale_hue=True, gridsize=100, width=0.8, inner='box', split=False, dodge=True, orient=None, linewidth=None, color=None, palette=None, saturation=0.75, ax=None, **kwargs)</p>
# 
# <ul>
#     <li>x, y, hue : names of variables in data or vector data, optional</li>
#     <li>data : DataFrame, array, or list of arrays, optional</li>
#     <li>scale : {“area”, “count”, “width”}, optional</li>
#     <li>linewidth : float, optional</li>
#     <li>color : matplotlib color, optional</li>
#     <li>palette : palette name, list, or dict, optional</li>
#     <li>ax : matplotlib Axes, optional</li>
#     <li>saturation : float, optional</li>
# </ul>

# In[ ]:


sns.violinplot(data['Math_Score'])
plt.xlabel('Math Score')
plt.ylabel('Frequency')
plt.title('Violin Math Score Show')
plt.show()

# In[ ]:


sns.violinplot(x=data['Race/Ethnicity'],y=data['Math_Score'])
plt.show()

# In[ ]:


sns.violinplot(data['Gender'],y=data['Reading_Score'],hue=data['Race/Ethnicity'],palette='muted')
plt.legend(loc=10)
plt.show()

# In[ ]:


sns.violinplot(data['Race/Ethnicity'],data['Writing_Score'],
               hue=data['Gender'],palette='muted',split=True)
plt.legend(loc=8)
plt.show()

# In[ ]:


sns.violinplot(data['Parental_Level_of_Education'],data['Math_Score'],hue=data['Gender'],dodge=False)
plt.xticks(rotation=90)
plt.show()

# <p id='12'><h3><b>Heatmap Plot</b></h3></p>
# <p>seaborn.heatmap(data, vmin=None, vmax=None, cmap=None, center=None, robust=False, annot=None, fmt='.2g', annot_kws=None, linewidths=0, linecolor='white', cbar=True, cbar_kws=None, cbar_ax=None, square=False, xticklabels='auto', yticklabels='auto', mask=None, ax=None, **kwargs)</p>
# 
# <ul>
#     <li>data : rectangular dataset</li>
#     <li>vmin, vmax : floats, optional</li>
#     <li>cmap : matplotlib colormap name or object, or list of colors, optional</li>
#     <li>annot : bool or rectangular dataset, optional</li>
#     <li>fmt : string, optional</li>
#     <li>linewidths : float, optional</li>
#     <li>ax : matplotlib Axes, optional</li>
# </ul>

# In[ ]:


sns.heatmap(data.corr())
plt.show()

# In[ ]:


sns.heatmap(data.corr(),vmin=0,vmax=1)
plt.show()

# In[ ]:


sns.heatmap(data.corr(),annot=True)
plt.show()

# In[ ]:


# Compute the correlation matrix
corr = data.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})

# In[ ]:


sns.heatmap(data.corr(),cmap='YlGnBu')
plt.show()

# In[ ]:


sns.axes_style("white")
mask = np.zeros_like(data.corr())
mask[np.triu_indices_from(mask)] = True
sns.heatmap(data.corr(),vmax=.3,mask=mask,square=True)
plt.show()

# <p id='13'><h3><b>Box Plot</b></h3></p>
# <p>seaborn.boxplot(x=None, y=None, hue=None, data=None, order=None, hue_order=None, orient=None, color=None, palette=None, saturation=0.75, width=0.8, dodge=True, fliersize=5, linewidth=None, whis=1.5, notch=False, ax=None, **kwargs)</p>
# <ul>
#     <li>x, y, hue : names of variables in data or vector data, optional</li>
#     <li>data : DataFrame, array, or list of arrays, optional</li>
#     <li>color : matplotlib color, optional</li>
#     <li>dodge : bool, optional</li>
#     <li>linewidth : float, optional</li>
#     <li>ax : matplotlib Axes, optional</li>
# </ul>

# In[ ]:


sns.set(style='whitegrid')
sns.boxplot(data['Math_Score'])
plt.show()

# In[ ]:


sns.boxplot(x=data['Gender'],y=data['Math_Score'])
plt.show()

# In[ ]:


sns.boxplot(x=data['Race/Ethnicity'],y=data['Writing_Score'],hue=data['Gender'],palette="Set3")
plt.show()

# In[ ]:


sns.boxplot(data['Math_Score'],orient='h',palette='Set2')
plt.show()

# In[ ]:


sns.boxenplot(x="Race/Ethnicity", y="Writing_Score",
              color="b",
              scale="linear", data=data)
plt.show()

# In[ ]:


sns.boxplot(x=data['Race/Ethnicity'],y=data['Writing_Score'],hue=data['Gender'],dodge=False)
plt.show()

# In[ ]:


sns.boxplot(x=data['Parental_Level_of_Education'],y=data['Math_Score'])
plt.xticks(rotation=90)
sns.swarmplot(x=data['Parental_Level_of_Education'],y=data['Math_Score'],color=".25")
plt.xticks(rotation=90)
plt.show()

# <p id='14'><h3><b>Swarm Plot</b></h3></p>
# <p>seaborn.swarmplot(x=None, y=None, hue=None, data=None, order=None, hue_order=None, dodge=False, orient=None, color=None, palette=None, size=5, edgecolor='gray', linewidth=0, ax=None, **kwargs)</p>
# 
# <ul>
#     <li>x, y, hue : names of variables in data or vector data, optional</li>
#     <li>data : DataFrame, array, or list of arrays, optional</li>
#     <li>dodge : bool, optional</li>
#     <li>palette : palette name, list, or dict, optional</li>
#     <li>size : float, optional</li>
#     <li>ax : matplotlib Axes, optional</li>
#     <li>linewidth : float, optional</li>
#     <li>edgecolor : matplotlib color, “gray” is special-cased, optional</li>
# </ul>

# In[ ]:


sns.set(style='whitegrid')
sns.swarmplot(x=data['Math_Score'])
plt.show()

# In[ ]:


sns.set(style="whitegrid")

sns.swarmplot(y=data["Writing_Score"],color='red')
sns.swarmplot(y=data["Reading_Score"],color='blue')
plt.title('Writing & Reading Scores')
plt.show()

# In[ ]:


sns.swarmplot(x=data['Lunch'],y=data['Reading_Score'])
plt.show()

# In[ ]:


sns.swarmplot(x=data['Test_Preparation_Course'],y=data['Math_Score'],hue=data['Gender'])
plt.show()

# In[ ]:


sns.swarmplot(x=data['Test_Preparation_Course'],y=data['Writing_Score'],hue=data['Race/Ethnicity'],palette='Set2',dodge=True)
plt.show()

# In[ ]:


sns.boxplot(x=data['Lunch'],y=data['Math_Score'],whis=np.inf)
sns.swarmplot(x=data['Lunch'],y=data['Math_Score'],color='.2')
plt.show()

# In[ ]:


sns.violinplot(x=data['Test_Preparation_Course'],y=data['Reading_Score'],inner=None)
sns.swarmplot(x=data['Test_Preparation_Course'],y=data['Reading_Score'],color='white',edgecolor='gray')
plt.show()

# <p id='15'><h3><b>Pair Plot</b></h3></p>
# <p>seaborn.pairplot(data, hue=None, hue_order=None, palette=None, vars=None, x_vars=None, y_vars=None, kind='scatter', diag_kind='auto', markers=None, height=2.5, aspect=1, dropna=True, plot_kws=None, diag_kws=None, grid_kws=None, size=None)</p>
# <ul>
#     <li>data : DataFrame</li>
#     <li>hue : string (variable name), optional</li>
#     <li>hue_order : list of strings</li>
#     <li>palette : dict or seaborn color palette</li>
#     <li>markers : single matplotlib marker code or list, optional</li>
#     <li>dropna : boolean, optional</li>
#     <li>height : scalar, optional</li>
# </ul>

# In[ ]:


sns.pairplot(data)
plt.show()

# In[ ]:


sns.pairplot(data,diag_kind='kde')
plt.show()

# In[ ]:


sns.pairplot(data,kind='reg')
plt.show()

# In[ ]:


sns.pairplot(data, diag_kind="kde", markers="+",
                  plot_kws=dict(s=50, edgecolor="b", linewidth=1),
                  diag_kws=dict(shade=True))
plt.show()

# In[ ]:


sns.pairplot(data, hue="Reading_Score")
plt.show()

# <p id='16'><h3><b>Count Plot</b></h3></p>
# <p>seaborn.countplot(x=None, y=None, hue=None, data=None, order=None, hue_order=None, orient=None, color=None, palette=None, saturation=0.75, dodge=True, ax=None, **kwargs)</p>
# <ul>
#     <li>x, y, hue : names of variables in data or vector data, optional</li>
#     <li>data : DataFrame, array, or list of arrays, optional</li>
#     <li>color : matplotlib color, optional</li>
#     <li>palette : palette name, list, or dict, optional</li>
#     <li>dodge : bool, optional</li>
#     <li>ax : matplotlib Axes, optional</li>
#     <li>ax : matplotlib Axes, optional</li>
# </ul>

# In[ ]:


data.columns

# In[ ]:


sns.countplot(data['Race/Ethnicity'])
plt.show()

# In[ ]:


sns.countplot(data['Gender'])
plt.show()

# In[ ]:


sns.countplot(data['Race/Ethnicity'],hue=data['Gender'])
plt.show()

# In[ ]:


sns.countplot(y=data['Parental_Level_of_Education'],palette="Set3",hue=data['Gender'])
plt.legend(loc=4)
plt.show()

# In[ ]:


sns.countplot(x=data['Lunch'],facecolor=(0,0,0,0),linewidth=5,edgecolor=sns.color_palette('dark',3))
plt.show()

# In[ ]:


sns.countplot(x="Parental_Level_of_Education", hue="Lunch",
                 data=data)
plt.xticks(rotation=45)
plt.show()

# <p id='17'><h3><b>FacetGrid</b></h3></p>
# <p>class seaborn.FacetGrid(data, row=None, col=None, hue=None, col_wrap=None, sharex=True, sharey=True, height=3, aspect=1, palette=None, row_order=None, col_order=None, hue_order=None, hue_kws=None, dropna=True, legend_out=True, despine=True, margin_titles=False, xlim=None, ylim=None, subplot_kws=None, gridspec_kws=None, size=None)</p>
# <ul>
#     <li>data : DataFrame</li>
#     <li>row, col, hue : strings</li>
#     <li>col_wrap : int, optional</li>
#     <li>share{x,y} : bool, ‘col’, or ‘row’ optional</li>
#     <li>palette : palette name, list, or dict, optional</li>
#     <li>legend_out : bool, optional</li>
#     <li>despine : boolean, optional</li>
#     <li>subplot_kws : dict, optional</li>
# </ul>

# In[ ]:


sns.FacetGrid(data,col='Gender',row='Gender')
plt.tight_layout()
plt.show()

# In[ ]:


g=sns.FacetGrid(data,col='Race/Ethnicity',row='Race/Ethnicity')
g=g.map(plt.hist,"Math_Score",bins=np.arange(0,65,3),color='r')
plt.show()

# In[ ]:


g=sns.FacetGrid(data,col='Lunch',row='Lunch')
g=(g.map(plt.scatter,"Reading_Score",'Writing_Score',edgecolor='w').add_legend())
plt.tight_layout()
plt.show()

# In[ ]:


g = sns.FacetGrid(data, col="Parental_Level_of_Education", col_wrap=3)
g = g.map(plt.plot, "Reading_Score", "Writing_Score", marker=".")
plt.show()

# In[ ]:


sns.set()

# Generate an example radial datast
r = np.linspace(0, 10, num=100)
df = pd.DataFrame({'r': r, 'slow': r, 'medium': 2 * r, 'fast': 4 * r})

# Convert the dataframe to long-form or "tidy" format
df = pd.melt(df, id_vars=['r'], var_name='speed', value_name='theta')

# Set up a grid of axes with a polar projection
g = sns.FacetGrid(df, col="speed", hue="speed",
                  subplot_kws=dict(projection='polar'), height=4.5,
                  sharex=False, sharey=False, despine=False)

# Draw a scatterplot onto each axes in the grid
g.map(sns.scatterplot, "theta", "r")
plt.show()

# In[ ]:


pal = sns.cubehelix_palette(10, rot=-.25, light=.7)
g = sns.FacetGrid(data, row="Writing_Score", hue="Reading_Score", aspect=15, height=.5, palette=pal)

# Draw the densities in a few steps
g.map(sns.kdeplot, "Writing_Score", clip_on=False, shade=True, alpha=1, lw=1.5, bw=.2)
g.map(sns.kdeplot, "Reading_Score", clip_on=False, color="w", lw=2, bw=.2)
g.map(plt.axhline, y=0, lw=2, clip_on=False)
plt.show()

# <p id='18'><h3><b>Strip Plot</b></h3></p>
# <p>seaborn.stripplot(x=None, y=None, hue=None, data=None, order=None, hue_order=None, jitter=True, dodge=False, orient=None, color=None, palette=None, size=5, edgecolor='gray', linewidth=0, ax=None, **kwargs)</p>
# <ul>
#     <li>x, y, hue : names of variables in data or vector data, optional</li>
#     <li>data : DataFrame, array, or list of arrays, optional</li>
#     <li>order, hue_order : lists of strings, optional</li>
#     <li>jitter : float, True/1 is special-cased, optional</li>
#     <li>dodge : bool, optional</li>
#     <li>color : matplotlib color, optional</li>
#     <li>edgecolor : matplotlib color, “gray” is special-cased, optional</li>
#     <li>linewidth : float, optional</li>
# </ul>

# In[ ]:


sns.stripplot(x=data['Reading_Score'])
plt.show()

# In[ ]:


sns.stripplot(x="Parental_Level_of_Education",y='Writing_Score',data=data)
plt.xticks(rotation=45)
plt.show()

# In[ ]:


sns.stripplot(x="Gender",y='Writing_Score',jitter=True,data=data)
plt.xticks(rotation=45)
plt.show()

# In[ ]:


sns.stripplot(x="Lunch",y='Reading_Score',jitter=0.05,data=data)
plt.xticks(rotation=45)
plt.show()

# In[ ]:


sns.stripplot(x='Test_Preparation_Course',y='Reading_Score',hue='Gender',jitter=True,data=data)
plt.show()

# In[ ]:


sns.stripplot(x='Race/Ethnicity',y='Math_Score',hue='Lunch',jitter=True,dodge=True,palette="Set2",data=data)
plt.show()

# In[ ]:


sns.stripplot(x='Lunch',y='Math_Score',hue='Lunch',jitter=True,dodge=True,size=20,marker='D',edgecolor='gray',alpha=.25,palette="Set2",data=data)
plt.legend(loc=10)
plt.show()

# <p id='19'><h3><b>Factor Plot</b></h3></p>
# <p>seaborn.catplot(x=None, y=None, hue=None, data=None, row=None, col=None, col_wrap=None, estimator=<function mean>, ci=95, n_boot=1000, units=None, order=None, hue_order=None, row_order=None, col_order=None, kind='strip', height=5, aspect=1, orient=None, color=None, palette=None, legend=True, legend_out=True, sharex=True, sharey=True, margin_titles=False, facet_kws=None, **kwargs)</p>
# <ul>
#     <li>x, y, hue : names of variables in data</li>
#     <li>data : DataFrame</li>
#     <li>row, col : names of variables in data, optional</li>
#     <li>col_wrap : int, optional</li>
#     <li>kind : string, optional</li>
#     <li>orient : “v” | “h”, optional</li>
#     <li>height : scalar, optional</li>
#     <li>color : matplotlib color, optional</li>
#     <li>palette : palette name, list, or dict, optional</li>
#     <li>legend : bool, optional</li>
# </ul>

# In[ ]:


sns.factorplot(x="Lunch", y="Math_Score", hue="Gender", data=data)
plt.show()

# In[ ]:


sns.factorplot(x="Gender", y="Reading_Score", hue="Lunch", kind='violin',data=data)
plt.show()

# In[ ]:


sns.factorplot(x="Race/Ethnicity", y="Math_Score", hue="Gender",col='Lunch',data=data)
plt.show()

# In[ ]:


g=sns.factorplot(x="Parental_Level_of_Education", y="Writing_Score", hue="Lunch",
                col="Gender", data=data)
plt.tight_layout()
plt.show()

# <p id='22'><h3><b>DisPlot</b></h3></p>
# <p>seaborn.distplot(a, bins=None, hist=True, kde=True, rug=False, fit=None, hist_kws=None, kde_kws=None, rug_kws=None, fit_kws=None, color=None, vertical=False, norm_hist=False, axlabel=None, label=None, ax=None)</p>
# 
# <ul>
#     <li>a : Series, 1d-array, or list</li>
#     <li>bins : argument for matplotlib hist(), or None, optional</li>
#     <li>hist : bool, optional</li>
#     <li>kde : bool, optional</li>
#     <li>color : matplotlib color, optional</li>
#     <li>label : string, optional</li>
# </ul>
# 

# 

# In[ ]:


ax = sns.distplot(data['Reading_Score'], rug=True, hist=False)
plt.show()

# In[ ]:


ax = sns.distplot(data['Writing_Score'], vertical=True)
plt.show()

# In[ ]:


ax = sns.distplot(data['Math_Score'])
plt.show()

# In[ ]:


ax = sns.distplot(data['Reading_Score'], color="y")
plt.show()

# In[ ]:


sns.set(style="white", palette="muted", color_codes=True)
rs = np.random.RandomState(10)

f, axes = plt.subplots(2, 2, figsize=(7, 7), sharex=True)
sns.despine(left=True)

# Generate a random univariate dataset
d = rs.normal(size=100)

# Plot a simple histogram with binsize determined automatically
sns.distplot(d, kde=False, color="b", ax=axes[0, 0])

# Plot a kernel density estimate and rug plot
sns.distplot(d, hist=False, rug=True, color="r", ax=axes[0, 1])

# Plot a filled kernel density estimate
sns.distplot(d, hist=False, color="g", kde_kws={"shade": True}, ax=axes[1, 0])

# Plot a historgram and kernel density estimate
sns.distplot(d, color="m", ax=axes[1, 1])

plt.setp(axes, yticks=[])
plt.tight_layout()
plt.show()

# <p id='23'><h3><b>Line Plot</b></h3></p>
# <p>seaborn.lineplot(x=None, y=None, hue=None, size=None, style=None, data=None, palette=None, hue_order=None, hue_norm=None, sizes=None, size_order=None, size_norm=None, dashes=True, markers=None, style_order=None, units=None, estimator='mean', ci=95, n_boot=1000, sort=True, err_style='band', err_kws=None, legend='brief', ax=None, **kwargs)</p>
# <ul>
#     <li>x, y : names of variables in data or vector data, optional</li>
#     <li>hue : name of variables in data or vector data, optional</li>
#     <li>size : name of variables in data or vector data, optional</li>
#     <li>style : name of variables in data or vector data, optional</li>
#     <li>data : DataFrame</li>
#     <li>palette : palette name, list, or dict, optional</li>
# </ul>

# In[ ]:


data.columns

# In[ ]:


data[data['Gender']=='male']['Math_Score'].value_counts().sort_index().plot.line(color='b')
data[data['Gender']=='female']['Math_Score'].value_counts().sort_index().plot.line(color='r')
plt.xlabel('Math_Score')
plt.ylabel('Frequency')
plt.title('Math_Score vs Frequency')
plt.show()

# In[ ]:


sns.lineplot(x='Math_Score',y='Reading_Score',data=data)
plt.show()

# In[ ]:


sns.lineplot(x='Reading_Score',y='Writing_Score',hue='Lunch',data=data)
plt.show()

# In[ ]:


sns.lineplot(x='Writing_Score',y='Reading_Score',data=data,hue='Lunch',
            style='Gender')
plt.show()

# In[ ]:


female_filter=data[data['Gender']=='female']
sns.lineplot(x='Reading_Score',y='Writing_Score',data=female_filter,
            hue='Lunch',style='Test_Preparation_Course',dashes=False)
plt.show()

# In[ ]:


sns.lineplot(x="Math_Score", y="Writing_Score", hue="Lunch",err_style="bars", ci=68, data=data)
plt.show()

# In[ ]:


ax = sns.lineplot(x="Math_Score", y="Reading_Score", hue="Test_Preparation_Course",
                   units="Lunch", estimator=None, lw=1,
                   data=data.query("Gender == 'male'"))

# In[ ]:


ax = sns.lineplot(x="Math_Score", y="Writing_Score",
                   hue="Lunch", style="Gender",
                   data=data)
plt.show()

# In[ ]:


data.groupby('Gender')[['Writing_Score','Reading_Score']].mean()

# In[ ]:


x=data[data.Parental_Level_of_Education=='bachelor\'s degree'].groupby('Race/Ethnicity')['Math_Score'].count()
x

# In[ ]:


sns.lineplot(data=x,color='coral',label='Race/Ethnicity')
plt.show()

# <p id='24'><h3><b>Despine</b></h3></p>
# <p>seaborn.despine(fig=None, ax=None, top=True, right=True, left=False, bottom=False, offset=None, trim=False)</p>
# <ul>
#     <li>fig : matplotlib figure, optional</li>
#     <li>ax : matplotlib axes, optional</li>
#     <li>top, right, left, bottom : boolean, optional</li>
#     <li>offset : int or dict, optional</li>
#     <li>trim : bool, optional</li>
# </ul>

# In[ ]:


f, ax = plt.subplots(figsize=(6.5, 6.5))
sns.despine(f, left=True, bottom=True)
sns.scatterplot(x="Math_Score", y="Reading_Score",
                hue="Gender", size="Gender",data=data)
plt.show()

# In[ ]:


data.head()

# In[ ]:



# Draw the full plot
sns.clustermap(data.corr(), center=0, cmap="vlag",
               linewidths=.75, figsize=(13, 13))

# In[ ]:


sns.set(style="white")
# Plot miles per gallon against horsepower with other semantics
sns.relplot(x="Reading_Score",y="Math_Score",hue="Gender",
            sizes=(40, 400), alpha=.5, palette="muted",
            height=6, data=data)
plt.show()

# In[ ]:


f, ax = plt.subplots(figsize=(6.5, 6.5))
sns.despine(f, left=True, bottom=True)
sns.scatterplot(x="Reading_Score", y="Writing_Score",
                hue="Lunch", size="Gender",data=data)
plt.show()

# In[ ]:


data.head()

# In[ ]:


f, ax = plt.subplots(figsize=(6.5, 6.5))
sns.despine(f, left=True, bottom=True)
sns.scatterplot(x="Reading_Score", y="Writing_Score",
                hue="Lunch", size="Gender",data=data[data['Parental_Level_of_Education']=="some college"])
plt.show()

# In[ ]:


data[np.logical_and(data['Race/Ethnicity']=='group A',data['Parental_Level_of_Education']=='some college')].head()

# In[ ]:


f, ax = plt.subplots(figsize=(6.5, 6.5))
sns.despine(f, left=True, bottom=True)
sns.scatterplot(x="Reading_Score", y="Writing_Score",
                hue="Lunch", size="Gender",data=data[np.logical_and(data['Race/Ethnicity']=='group C',data['Parental_Level_of_Education']=='some college')])
plt.show()

# <p id='20'><h3><b>References</b></h3></p>
# <p>https://www.kaggle.com/spscientist/students-performance-in-exams</p>
# <p>https://seaborn.pydata.org/</p>
# <p>https://www.kaggle.com/kanncaa1/seaborn-tutorial-for-beginners</p>
# <p>https://www.kaggle.com/biphili/seaborn-plot-to-visualize-iris-data</p>

# <p id='21'><h3><b>Conclusion</b></h3></p>
# <p>As a result, we have explained the seaborn library in a very detailed way and created a wide variety of graphs. If you like it, I expect your support. If you like <b>UPVOTED</b> I would be very happy if you do. If you have any questions, I am ready to answer your questions. At the bottom there are the kernel values that I have already done.</p>
# <p>https://www.kaggle.com/kralmachine/analyzing-the-heart-disease</p>
# <p>https://www.kaggle.com/kralmachine/data-visualization-of-suicide-rates</p>
# <p>https://www.kaggle.com/kralmachine/gradient-admission-eda-ml-0-92</p>
# <p>https://www.kaggle.com/kralmachine/football-results-from-1872-to-2018-datavisulation</p>
# <p>https://www.kaggle.com/kralmachine/pandas-tutorial-for-beginner</p>
# <p>https://www.kaggle.com/kralmachine/visual-analysis-of-world-happiness-in-2015</p>
