#!/usr/bin/env python
# coding: utf-8

# ![DonorChoose](data:image/png;base64,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)

# **This notebook will always be a work in progress.**
# <p></p>
#  **INDEX** 
# * Loading Libraries
# * Loading Data
# * Peek into Data
# * Donations - EDA
# * Donors - EDA
# * Schools - EDA
# * Teachers - EDA
# * Projects - EDA
# 
# <h2><a href="#ac">> Andrews Curves</a> - Click Here</h2>

# **Loading Libraries**

# In[ ]:


import pandas as pd # package for high-performance, easy-to-use data structures and data analysis
import numpy as np # fundamental package for scientific computing with Python
import matplotlib
import matplotlib.pyplot as plt # for plotting
import seaborn as sns # for making plots with seaborn
color = sns.color_palette()
import plotly.offline as py
py.init_notebook_mode(connected=True)
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.offline as offline
offline.init_notebook_mode()
from plotly import tools
from mpl_toolkits.basemap import Basemap
from numpy import array
from matplotlib import cm

# import cufflinks and offline mode
import cufflinks as cf
cf.go_offline()

from sklearn import preprocessing
# Supress unnecessary warnings so that presentation looks clean
import warnings
warnings.filterwarnings("ignore")

# Print all rows and columns
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

# **Loading Data**

# In[ ]:


donations = pd.read_csv('../input/Donations.csv')
donors = pd.read_csv('../input/Donors.csv', low_memory=False)
schools = pd.read_csv('../input/Schools.csv', error_bad_lines=False)
teachers = pd.read_csv('../input/Teachers.csv', error_bad_lines=False)
projects = pd.read_csv('../input/Projects.csv', error_bad_lines=False, warn_bad_lines=False, parse_dates=["Project Posted Date","Project Fully Funded Date"])
resources = pd.read_csv('../input/Resources.csv', error_bad_lines=False, warn_bad_lines=False)

# **Peek into Data**

# In[ ]:


donations.head()

# In[ ]:


donations.shape

# In[ ]:


donors.head()

# In[ ]:


donors.shape

# In[ ]:


schools.head()

# In[ ]:


schools.shape

# In[ ]:


teachers.head()

# In[ ]:


teachers.shape

# In[ ]:


projects.head()

# In[ ]:


projects.shape

# **Donations-EDA**

# In[ ]:


donations.groupby('Donation Included Optional Donation')['Donation Included Optional Donation'].count().plot.bar()

# In[ ]:


temp = donations['Donation Included Optional Donation'].value_counts().head()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values')

# In[ ]:


donations['Donation Amount'].value_counts().head()

# In[ ]:


#TOP 5 occuring Donation Amounts
temp = donations['Donation Amount'].value_counts().head()
temp.plot.bar()

# In[ ]:


df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values')

# In[ ]:


donations['Donor Cart Sequence'].value_counts().head()

# In[ ]:


#TOP 5 occuring Donor Cart Sequence
temp = donations['Donor Cart Sequence'].value_counts().head()
temp.plot.bar()

# In[ ]:


df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values')

# **Donors - EDA**

# In[ ]:


donors.groupby('Donor Is Teacher')['Donor Is Teacher'].count().plot.bar()

# In[ ]:


temp = donors['Donor Is Teacher'].value_counts().head()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values')

# In[ ]:


#TOP 10 occuring Cities
temp = donors['Donor City'].value_counts()[:10]
temp.plot.bar()

# In[ ]:


df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values')

# In[ ]:


#TOP 10 occuring States
temp = donors['Donor State'].value_counts()[:10]
temp.plot.bar()

# In[ ]:


df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values')

# In[ ]:


#TOP 10 occuring Zips
temp = donors['Donor Zip'].value_counts()[:10]
temp.plot.bar()

# In[ ]:


df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values')

# **Schools - EDA**

# In[ ]:


schools.groupby('School Metro Type')['School Metro Type'].count().plot.bar()

# In[ ]:


temp = schools['School Metro Type'].value_counts().head()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values')

# In[ ]:


#TOP 5 occuring School Percentages Free Lunch
temp = schools['School Percentage Free Lunch'].value_counts().head()
temp.plot.bar()

# In[ ]:


df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values')

# In[ ]:


#TOP 5 occuring School States
temp = schools['School State'].value_counts().head()
temp.plot.bar()

# In[ ]:


df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values')

# In[ ]:


#TOP 5 occuring School Zips
temp = schools['School Zip'].value_counts().head()
temp.plot.bar()

# In[ ]:


df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values')

# In[ ]:


#TOP 5 occuring School Cities
temp = schools['School City'].value_counts().head()
temp.plot.bar()

# In[ ]:


df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values')

# In[ ]:


#TOP 5 occuring School Counties
temp = schools['School County'].value_counts().head()
temp.plot.bar()

# In[ ]:


df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values')

# In[ ]:


#TOP 5 occuring School Districts
temp = schools['School District'].value_counts().head()
temp.plot.bar()

# In[ ]:


df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values')

# **Teachers - EDA**

# In[ ]:


teachers.groupby('Teacher Prefix')['Teacher Prefix'].count().plot.bar()

# In[ ]:


temp = teachers['Teacher Prefix'].value_counts().head()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values')

# In[ ]:


teachers['Teacher First Project Posted Date'] = pd.to_datetime(teachers['Teacher First Project Posted Date'])
teachers['Year'] = teachers['Teacher First Project Posted Date'].dt.year
teachers['Month'] = teachers['Teacher First Project Posted Date'].dt.month
teachers['Day'] = teachers['Teacher First Project Posted Date'].dt.day
df1 = teachers['Year'].value_counts()
df2 = teachers['Month'].value_counts()
df3 = teachers['Day'].value_counts()

# In[ ]:


#TOP 5 occuring Years
df1.head().plot.bar()

# In[ ]:


temp = teachers['Year'].value_counts().head()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values')

# In[ ]:


#TOP 5 occuring Months
df2.head().plot.bar()

# In[ ]:


temp = teachers['Month'].value_counts().head()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values')

# In[ ]:


#TOP 5 occuring Days
df3.head().plot.bar()

# In[ ]:


temp = teachers['Day'].value_counts().head()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values')

# **Projects - EDA**

# In[ ]:


#TOP 5 occuring Teacher Project Posted Sequences
temp = projects['Teacher Project Posted Sequence'].value_counts().head()
temp.plot.bar()

# In[ ]:


df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values')

# In[ ]:


projects.groupby('Project Type')['Project Type'].count().plot.bar()

# In[ ]:


temp = projects['Project Type'].value_counts().head()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values')

# In[ ]:


from wordcloud import WordCloud, STOPWORDS
wordcloud = WordCloud( max_font_size=50, 
                       stopwords=STOPWORDS,
                       background_color='white',
                     ).generate(str(projects['Project Title'].tolist()))

plt.figure(figsize=(14,7))
plt.title("Wordcloud for Top Keywords in Project Title", fontsize=35)
plt.imshow(wordcloud)
plt.axis('off')
plt.show()

# In[ ]:


from wordcloud import WordCloud, STOPWORDS
wordcloud = WordCloud( max_font_size=50, 
                       stopwords=STOPWORDS,
                       background_color='white',
                     ).generate(str(projects['Project Essay'].sample(2000).tolist()))

plt.figure(figsize=(14,7))
plt.title("Wordcloud for Top Keywords in Project Essay", fontsize=35)
plt.imshow(wordcloud)
plt.axis('off')
plt.show()

# In[ ]:


#TOP 5 occuring Project Subject Category Tree
temp = projects['Project Subject Category Tree'].value_counts().head()
temp.plot.bar()

# In[ ]:


df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values')

# In[ ]:


#TOP 5 occuring Project Subject Subcategory Tree
temp = projects['Project Subject Subcategory Tree'].value_counts().head()
temp.plot.bar()

# In[ ]:


df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values')

# In[ ]:


projects.groupby('Project Grade Level Category')['Project Grade Level Category'].count().plot.bar()

# In[ ]:


temp = projects['Project Grade Level Category'].value_counts()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values')

# In[ ]:


projects.groupby('Project Resource Category')['Project Resource Category'].count().plot.bar()

# In[ ]:


temp = projects['Project Resource Category'].value_counts()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values')

# <h2 id="ac">Andrew Curves</h2>

# **Andrews curves are a method for visualizing multidimensional data by mapping each observation onto a function. 
# This function is defined as**
# 
# ![Formula](https://4.bp.blogspot.com/-0n23UhHOB9w/VEEpDZsoZUI/AAAAAAAAA2k/uQCQAoaLGlM/s400/andrewscurve.png)
# 
# **It has been shown the Andrews curves are able to preserve means, distance (up to a constant) and variances. Which means that Andrews curves that are represented by functions close together suggest that the corresponding data points will also be close together.**

# In[ ]:


# One cool more sophisticated technique pandas has available is called Andrews Curves
# Andrews Curves involve using attributes of samples as coefficients for Fourier series
# and then plotting these
from pandas.tools.plotting import andrews_curves
andrews_curves(donations.sample(1000).drop(["Project ID","Donation ID","Donor ID","Donation Received Date"], axis=1), "Donation Included Optional Donation")

# **In the plots above, the each color used represents a class and we can easily note that the lines that represent samples from the same class have similar curves.**
# Note - Not a perfect place to use it, but you get the basic idea how to use it in on dataset with more numeric values ;)

# ****Stay Tuned and PLEASE, votes up! I will update this Kernel soon.****
