#!/usr/bin/env python
# coding: utf-8

# # Time Series EDA for Microsoft Malware
# 
# Following [Aditya Soni's][1] great [advice][2], I downloaded timestamps from Microsoft for all 18 million data observations in train.csv and test.csv. Timestamps highlight the unique challenge of this competition. The training data are mostly from August and September 2018 while the test data are mostly from October and November 2018. We are challenged to build a time independent model.
# 
# Furthermore, there are 80 mostly categorical variables with an average of 1000 categories each. If we naively one hot encode each variable, we would add 80,000 new variables! Therefore we are also challenged to encode data efficiently.
# 
# In this kernel, we perform EDA, discuss encodings, and build a simple linear logistic model.
# 
# [1]:https://www.kaggle.com/adityaecdrid
# [2]:https://www.kaggle.com/c/microsoft-malware-prediction/discussion/76384#449402

# # Load Malware Train and Test Data

# In[ ]:


# IMPORT LIBRARIES
import pandas as pd, numpy as np, os
import matplotlib.pyplot as plt
from datetime import datetime

# VARIABLES TO IMPORT
vv = ['AVProductStatesIdentifier','AVProductsInstalled','OsBuild',
       'CountryIdentifier','Processor','SmartScreen','Census_OSVersion',
       'Census_ProcessorCoreCount','Census_ProcessorModelIdentifier', 
       'Census_HasOpticalDiskDrive','Census_TotalPhysicalRAM',
       'Census_InternalPrimaryDisplayResolutionVertical','Census_OSInstallTypeName',
       'Census_PowerPlatformRoleName','Census_IsTouchEnabled',
       'Wdft_IsGamer']
dtypes = {}
for x in vv: dtypes[x] = 'category'
dtypes['MachineIdentifier'] = 'str'
dtypes['AvSigVersion'] = 'category'
dtypes['HasDetections'] = 'int8'

# LOAD CSV FILES
df_train = pd.read_csv('../input/microsoft-malware-prediction/train.csv', usecols=dtypes.keys(), dtype=dtypes)
print ('Loaded',len(df_train),'rows of TRAIN.CSV!')
df_test = pd.read_csv('../input/microsoft-malware-prediction/test.csv', usecols=list(dtypes.keys())[0:-1], dtype=dtypes)
print ('Loaded',len(df_test),'rows of TEST.CSV!')

# REDUCE SIZE FOR QUICKER PROTOTYPING
#df_train = df_train.sample(100000)
#df_test = df_test.sample(100000)

# # Append Timestamps to Data

# In[ ]:


# IMPORT TIMESTAMP DICTIONARY
datedict = np.load('../input/malware-timestamps/AvSigVersionTimestamps.npy')
datedict = datedict[()]
# ADD TIMESTAMPS
df_train['Date'] = df_train['AvSigVersion'].map(datedict)
df_test['Date'] = df_test['AvSigVersion'].map(datedict)

# # Define EDA Python Functions
# I wrote two EDA Python functions. To see the code, click the show code button. One function visualizes overall density and detection rate per category value. The other visualizes density and detection rate over time. Feel free to fork my kernel and use my functions to explore the data. I've also made my timestamp database public so you can import timestamps.

# In[ ]:


import calendar, math

# PARAMETERS
# data : pandas.DataFrame : your data to plot
# col  : str : which column to plot histogram for left y-axis
# target : str : which column for mean/rate on right y-axis
# bars : int : how many histogram bars to show (or less if you set show or min)
# show : float : stop displaying bars after 100*show% of data is showing
# minn : float : don't display bars containing under 100*minn% of data
# sortby : str : either 'frequency', 'category', or 'rate'
# verbose : int : display text summary 1=yes, 0=no
# top : int : give this many bars nice color (and matches a subsequent dynamicPlot)
# title : str : title of plot
# asc : boolean : sort ascending (for category and rate)
# dropna : boolean : include missing data as a category or not

def staticPlot(data, col, target='HasDetections', bars=10, show=1.0, sortby='frequency'
               , verbose=1, top=5, title='',asc=False, dropna=False, minn=0.0):
    # calcuate density and detection rate
    cv = data[col].value_counts(dropna=dropna)
    cvd = cv.to_dict()
    nm = cv.index.values; lnn = len(nm); lnn2 = lnn
    th = show * len(data)
    th2 = minn * len(data)
    sum = 0; lnn2 = 0
    for x in nm[0:bars]:
        lnn2 += 1
        try: sum += cvd[x]
        except: sum += cv[x]
        if sum>th:
            break
        try:
            if cvd[x]<th2: break
        except:
            if cv[x]<th2: break
    if lnn2<bars: bars = lnn2
    pct = round(100.0*sum/len(data),2)
    lnn = min(lnn,lnn2)
    ratio = [0.0]*lnn; lnn3 = lnn
    if sortby =='frequency': lnn3 = min(lnn3,bars)
    elif sortby=='category': lnn3 = 0
    for i in range(lnn3):
        if target not in data:
            ratio[i] = np.nan
        elif nan_check(nm[i]):
            ratio[i] = data[target][data[col].isna()].mean()
        else:
            ratio[i] = data[target][data[col]==nm[i]].mean()
    try: all = pd.DataFrame( {'category':nm[0:lnn],'frequency':[cvd[x] for x in nm[0:lnn]],'rate':ratio} )
    except: all = pd.DataFrame( {'category':nm[0:lnn],'frequency':[cv[x] for x in nm[0:lnn]],'rate':ratio} )
    if sortby=='rate': 
        all = all.sort_values(sortby, ascending=asc)
    elif sortby=='category':
        try: 
            all['temp'] = all['category'].astype('float')
            all = all.sort_values('temp', ascending=asc)
        except:
            all = all.sort_values('category', ascending=asc)
    if bars<lnn: all = all[0:bars]
    if verbose==1 and target in data:
        print('TRAIN.CSV variable',col,'has',len(nm),'categories')
        print('The',min(bars,lnn),'bars displayed here contain',pct,'% of data.')
        mlnn = data[col].isna().sum()
        print("The data has %.1f %% NA. The plot is sorted by " % (100.0*mlnn/len(data)) + sortby )
    
    # plot density and detection rate
    fig = plt.figure(1,figsize=(15,3))
    ax1 = fig.add_subplot(1,1,1)
    clrs = ['red', 'green', 'blue', 'yellow', 'magenta']
    barss = ax1.bar([str(x) for x in all['category']],[x/float(len(data)) for x in all['frequency']],color=clrs)
    for i in range(len(all)-top):
        barss[top+i].set_color('cyan')
    if target in data:
        ax2 = ax1.twinx()
        if sortby!='category': infected = all['rate'][0:lnn]
        else:
            infected=[]
            for x in all['category']:
                if nan_check(x): infected.append( data[ data[col].isna() ][target].mean() )
                elif cvd[x]!=0: infected.append( data[ data[col]==x ][target].mean() )
                else: infected.append(-1)
        ax2.plot([str(x) for x in all['category']],infected[0:lnn],'k:o')
        #ax2.set_ylim(a,b)
        ax2.spines['left'].set_color('red')
        ax2.set_ylabel('Detection Rate', color='k')
    ax1.spines['left'].set_color('red')
    ax1.yaxis.label.set_color('red')
    ax1.tick_params(axis='y', colors='red')
    ax1.set_ylabel('Category Proportion', color='r')
    if title!='': plt.title(title)
    plt.show()
    if verbose==1 and target not in data:
        print('TEST.CSV variable',col,'has',len(nm),'categories')
        print('The',min(bars,lnn),'bars displayed here contain',pct,'% of the data.')
        mlnn = data[col].isna().sum()
        print("The data has %.1f %% NA. The plot is sorted by " % (100.0*mlnn/len(data)) + sortby )

# PARAMETERS
# data : pandas.DataFrame : your data to plot
# col  : str : which column for density on left y-axis
# target : str : which column for mean/rate on right y-axis
# start : datetime.datetime : x-axis minimum
# end : datetime.datetime : x-axis maximum
# inc_hr : int : resolution of time sampling = inc_hr + inc_dy*24 + inc_mn*720 hours
# inc_dy : int : resolution of time sampling = inc_hr + inc_dy*24 + inc_mn*720 hours
# inc_mn : int : resolution of time sampling = inc_hr + inc_dy*24 + inc_mn*720 hours
# show : float : only show the most frequent category values that include 100*show% of data
# top : int : plot this many solid lines
# top2 : int : plot this many dotted lines
# title : str : title of plot
# legend : int : include legend or not. 1=yes, 0=no
# dropna : boolean : include missing data as a category or not
        
def dynamicPlot(data,col, target='HasDetections', start=datetime(2018,4,1), end=datetime(2018,12,1)
                ,inc_hr=0,inc_dy=7,inc_mn=0,show=0.99,top=5,top2=4,title='',legend=1, dropna=False):
    # check for timestamps
    if 'Date' not in data:
        print('Error dynamicPlot: DataFrame needs column Date of datetimes')
        return
    
    # remove detection line if category density is too small
    cv = data[col].value_counts(dropna=dropna)
    cvd = cv.to_dict()
    nm = cv.index.values
    th = show * len(data)
    sum = 0; lnn2 = 0
    for x in nm:
        lnn2 += 1
        try: sum += cvd[x]
        except: sum += cv[x]
        if sum>th:
            break
    top = min(top,len(nm))
    top2 = min(top2,len(nm),lnn2,top)

    # calculate rate within each time interval
    diff = (end-start).days*24*3600 + (end-start).seconds
    size = diff//(3600*((inc_mn * 28 + inc_dy) * 24 + inc_hr)) + 5
    data_counts = np.zeros([size,2*top+1],dtype=float)
    idx=0; idx2 = {}
    for i in range(top):
        idx2[nm[i]] = i+1
    low = start
    high = add_time(start,inc_mn,inc_dy,inc_hr)
    data_times = [low+(high-low)/2]
    while low<end:
        slice = data[ (data['Date']<high) & (data['Date']>=low) ]
        data_counts[idx,0] = len(slice)
        for key in idx2:
            if nan_check(key): slice2 = slice[slice[col].isna()]
            else: slice2 = slice[slice[col]==key]
            data_counts[idx,idx2[key]] = len(slice2)
            if target in data:
                data_counts[idx,top+idx2[key]] = slice2['HasDetections'].mean()
        low = high
        high = add_time(high,inc_mn,inc_dy,inc_hr)
        data_times.append(low+(high-low)/2)
        idx += 1

    # plot lines
    fig = plt.figure(1,figsize=(15,3))
    cl = ['r','g','b','y','m']
    ax3 = fig.add_subplot(1,1,1)
    lines = []; labels = []
    for i in range(top):
        tmp, = ax3.plot(data_times,data_counts[0:idx+1,i+1],cl[i%5])
        lines.append(tmp)
        labels.append(str(nm[i]))
    ax3.spines['left'].set_color('red')
    ax3.yaxis.label.set_color('red')
    ax3.tick_params(axis='y', colors='red')
    if col!='ones': ax3.set_ylabel('Category Density', color='r')
    else: ax3.set_ylabel('Data Density', color='r')
    ax3.set_yticklabels([])
    if target in data:
        ax4 = ax3.twinx()
        for i in range(top2):
            ax4.plot(data_times,data_counts[0:idx+1,i+1+top],cl[i%5]+":")
        ax4.spines['left'].set_color('red')
        ax4.set_ylabel('Detection Rate', color='k')
    if title!='': plt.title(title)
    if legend==1: plt.legend(lines,labels)
    plt.show()
        
# INCREMENT A DATETIME
def add_time(sdate,months=0,days=0,hours=0):
    month = sdate.month -1 + months
    year = sdate.year + month // 12
    month = month % 12 + 1
    day = sdate.day + days
    if day>calendar.monthrange(year,month)[1]:
        day -= calendar.monthrange(year,month)[1]
        month += 1
        if month>12:
            month = 1
            year += 1
    hour = sdate.hour + hours
    if hour>23:
        hour = 0
        day += 1
        if day>calendar.monthrange(year,month)[1]:
            day -= calendar.monthrange(year,month)[1]
            month += 1
            if month>12:
                month = 1
                year += 1
    return datetime(year,month,day,hour,sdate.minute)

# CHECK FOR NAN
def nan_check(x):
    if isinstance(x,float):
        if math.isnan(x):
            return True
    return False

# In the plots below, solid lines are density and use the left y-axis. Dotted lines are detection rate and use the right y-axis. You can see that the majority of train data are observations in August and September 2018 while test data is October and November 2018. It is interesting that the malware infection rate is correlated with observation density. Perhaps infected computers send more reports to Microsoft. Or Microsoft chose to give us more infected samples during this time interval of interest.

# In[ ]:


df_train['ones'] = 1; df_test['ones'] = 1
dynamicPlot(df_train,'ones',title='Train Data Density versus Time',legend=0)
dynamicPlot(df_test,'ones',title='Test Data Density versus Time',legend=0)

# # Encoding Categorical Variables
# Nearly all the variables in Microsoft's Malware Detection competition are categorical. Even the ones with numbers like CountryCode are actually categories. Below lists some ways to encode categorical variables. Your choice of encoding depends on both which model you plan to use (Logistic Regression, Boosted Trees, etc) and the properties of the variable.  
# 
# **Category Encode**: We can leave a category variable as a category (and optionally group them into fewer categories). Some machine learning algorithms like DecisionTreeClassifier and LightGMB can accept categorical variables. But for many models, this is not an option.  
# **Binary Encode**: Make a copy of a category's column. Then chose a specific category value and replace each occurence of that value with a 1 and each non occurence with a 0. Repeat this for each category value of interest.  
# **Numeric Encode**: Determine an ordering of the values within a category. Set a specific value as first, one as second, one as third, etc. Then replace each occurance of that value with its ranking number.  
# **Frequency Encode**: Count how many times a specific value occurs within a category. Then replace each occurance of that value with its count number.  
# **Target Encode**: Calculate the proportion of HasDetections=1 for a specific value within a category. Then replace each occurance of that value with this rate of detection.

# # Visualizing Categorical Variables
# Below are 6 variables from the Microsoft Malware dataset and suggested encodings. If you wish to view more variables, fork this kernel and modify the code.
# 
# ## SmartScreen - Binary Encode
# First lets plot the variable `SmartScreen`. From the plot below, we see that 10% of observations have `SmartScreen = ExistsNotSet` and when this is present, the observed rate of malware is 80%! Wow. From the time plot, we see that this variable is time independent. Regardless of time and changing software versions, this variable stays consistently predictive. Let's add a new column which **Binary Encodes** the presence of `ExistsNotSet`.

# In[ ]:


staticPlot(df_train,'SmartScreen',title='SmartScreen')
dynamicPlot(df_train,'SmartScreen',title='SmartScreen')

# ## AVProductsInstalled - Numeric Encode
# Next let's plot `AVProductsInstalled`. Below we see a natural trend. If we order the category values as follows 1 < 2 < 3 < NA < 4 < 5 < 6 and ignore 0 and 7, then we can **Numeric Encode** this variable. We also notice that this variable is time independent. Having more `AVProductsInstalled` reduces malware detection regardless of time. (Don't worry about the weird values of 0 and 7, those categories contain less than 0.1% of the data)

# In[ ]:


staticPlot(df_train,'AVProductsInstalled',title='AVProductsInstalled')
dynamicPlot(df_train,'AVProductsInstalled',title='AVProductsInstalled')

# ## CountryIdentifier - Target Encode
# Let's plot `CountryIdentifier`. This category has 222 categories! And it's still one of the smaller ones. If we one hot encode this with **Binary Encode**, we would add 222 new columns to our data frame. We could **Frequency Encode** this category but there is only a slight downward trend when we plot frequency against detection rate. Let's sort by number and see if the trend between numeric value and detection rate is better. If so, we could **Numeric Encode**.

# In[ ]:


staticPlot(df_train,'CountryIdentifier',title='CountryIdentifier',bars=40,show=0.9)
dynamicPlot(df_train,'CountryIdentifier')

# Each category value is actually a number. Let's order them in their natural ordering and see how they look. Below are the first 40 country codes in sorted order. There is still no linear association between the x-axis and y-axis, so let's avoid **Numeric Encode**. Now let's consider **Target Encode**. The two plots on the bottom are sorted by detection rate, thus forcing a monotonic trend. Now we will **Target Encode** this variable.
# 
# From the middle plot, we see that `CountryCode=104` has the highest detection rate out of 222 countries at 63%. From the bottom plot, we see that `CountryCode=62` has the lowest rate of 38%. I wonder which countries these are!

# In[ ]:


staticPlot(df_train,'CountryIdentifier',title='CountryIdentifier sorted by number',bars=40,show=0.9,sortby='category',verbose=0,asc=True)
staticPlot(df_train,'CountryIdentifier',title='CountryIdentifier sorted by decreasing rate',bars=40,show=0.9,sortby='rate',verbose=0)
staticPlot(df_train,'CountryIdentifier',title='CountryIdentifier sorted by increasing rate',bars=40,show=0.9,sortby='rate',verbose=0,asc=True)

# ## Census_OSVersion - Needs Innovative Encoding!
# Many people have pointed out how the training data and test data are different. [Bojan Tunguz][2] showed recently that a classifier can detect whether an observation came from the training dataset or test dataset with AUC 0.97 [here][1]. And before that, [Olivier][3] showed it [here][4]. [Tito][7] also demonstrated it [here][6] a while back. And another kernel [here][5] shows it. Besides the variable `SMode`, the variables that are most different are versioning numbers which indicate time such as `AvSigVersion`, `EngineVersion`, `AppVersion`, `Census_OSVersion`, etc. Since values for these variables exist in the test data that don't exist in the training data, these variables need an **Innovative Encoding** to become time independent.
# 
# Below is a plot of `Census_OSVersion` over time. You can see that the training dataset is mainly (1) `10.0.17134.165`, (2) `10.0.17134.228`, and (3) `10.0.17134.285` while the test dataset is mainly (3) `10.0.17134.285`, (4) `10.0.17134.345`, and (5) `10.0.17134.407`. I have a feeling that the test dataset will be split into public test containing (3) (4) and private test containing (5). This would mean that the training and public test set have some overlap but the training and private test set may have no overlap!  
# 
# [1]:https://www.kaggle.com/tunguz/ms-malware-adversarial-validation
# [2]:https://www.kaggle.com/tunguz
# [3]:https://www.kaggle.com/ogrellier
# [4]:https://www.kaggle.com/c/microsoft-malware-prediction/discussion/75223
# [5]:https://www.kaggle.com/sasjurse/the-strange-behavior-of-avsigversion/notebook
# [6]:https://www.kaggle.com/its7171/avsigversion-analisys
# [7]:https://www.kaggle.com/its7171
# 

# In[ ]:


dynamicPlot(df_train,'Census_OSVersion',title='TRAIN.CSV Census_OSVersion',show=0.99)
dynamicPlot(df_test,'Census_OSVersion',title='TEST.CSV Census_OSVersion',show=0.99)

# ## "Weeks since January 1, 2018" - Frequency Encode
# Earlier we saw that detection rate correlated with observation rate. So let's engineer a new variable that captures temporal observation rate. We will add a new column called `Week` which enocdes the number of weeks after Jan 1, 2018. (And negative values indicate weeks before Jan 1, 2018). The plot below shows that there is an association between this variable's frequency and detection rate as expected. Therefore, lets **Frequency Encode** this new variable. (Note that we cannot **Binary Encode**, **Numeric Encode**, or **Target Encode** this variable because the test data contains values that are not present in the training data.)

# In[ ]:


# FEATURE ENGINEER - WEEK
first = datetime(2018,1,1); datedict2 = {}
for x in datedict: datedict2[x] = (datedict[x]-first).days//7
df_train['Week'] = df_train['AvSigVersion'].map(datedict2)
df_test['Week'] = df_test['AvSigVersion'].map(datedict2)

staticPlot(df_train,'Week',title='Week',show=0.99,asc=True,bars=25)

# ## Census_OSInstallTypeName - Frequency Encode
# The variable `Census_OSInstallTypeName` shows a negative association between frequency and detection rate when considering all data. However when we look at the time series plot and observe different time intervals, it isn't clear whether this trend continues into October and November 2018 (the test data). We'll **Frequency Encode** this anyway but this may hurt our LB score even though it helps our CV score.

# In[ ]:


staticPlot(df_train,'Census_OSInstallTypeName',show=0.9,bars=50)
dynamicPlot(df_train,'Census_OSInstallTypeName')

# ## Census_TotalPhysicalRAM - Numeric Encode
# You must be careful when doing EDA because many variables are noisy.  Consider `Census_TotalPhysicalRAM` for example. This variable has 3446 unique values but 3437 categories contain very few observations! If you plot the detection rate versus all values sorted, the trend is downward and easy to miss because the plot is noisy. However 98% of the data is contained in 9 categories. If we only plot those 98%, the trend is very clear.
# 

# In[ ]:


staticPlot(df_train,'Census_TotalPhysicalRAM',title='99% of Census_TotalPhysicalRAM sorted by category',
           bars=155,sortby='category',show=0.99,dropna=True)
staticPlot(df_train,'Census_TotalPhysicalRAM',title='98% of Census_TotalPhysicalRAM sorted by category',
           bars=155,sortby='category',show=0.98,dropna=True,verbose=0)

# # Define Python Encoding Functions
# Now we'll write Python functions to encode variables. Click the show code button if you wish to read.

# In[ ]:


# If you call encode_TE, encode_TE_partial, encode_FE_partial, 
# or encode_BE_partial on training data then the function 
# returns a 2 element python list containing [list, dictionary]
# the return[0] = list are the names of new columns added
# the return[1] = dictionary are which category variables got encoded
# When encoding test data after one of 4 calls above, use 'encode_?E_test'
# and pass the dictionary. If you don't use one of 4 above, then you can
# call basic 'encode_?E' on test.

# TARGET ENCODING
def encode_TE(df,col,tar='HasDetections'):
    d = {}
    v = df[col].unique()
    for x in v:
        if nan_check(x):
            m = df[tar][df[col].isna()].mean()
        else:
            m = df[tar][df[col]==x].mean()
        d[x] = m
    n = col+"_TE"
    df[n] = df[col].map(d)
    return [[n],d]

# TARGET ENCODING first ct columns by freq
def encode_TE_partial(df,col,ct,tar='HasDetections',xx=0.5):
    d = {}
    cv = df[col].value_counts(dropna=False)
    nm = cv.index.values[0:ct]
    for x in nm:
        if nan_check(x):
            m = df[tar][df[col].isna()].mean()
        else:
            m = df[tar][df[col]==x].mean()
        d[x] = m
    n = col+"_TE"
    df[n] = df[col].map(d).fillna(xx)
    return [[n],d]

# TARGET ENCODING from dictionary
def encode_TE_test(df,col,mp,xx=0.5):
    n = col+"_TE"
    df[n] = df[col].map(mp).fillna(xx)
    return [[n],0]

# FREQUENCY ENCODING
def encode_FE(df,col):
    d = df[col].value_counts(dropna=False)
    n = col+"_FE"
    df[n] = df[col].map(d)/d.max()
    return [[n],d]

# FREQUENCY ENCODING first ct columns by freq
def encode_FE_partial(df,col,ct):
    cv = df[col].value_counts(dropna=False)
    nm = cv.index.values[0:ct]
    n = col+"_FE"
    df[n] = df[col].map(cv)
    df.loc[~df[col].isin(nm),n] = np.mean(cv.values)
    df[n] = df[n] / max(cv.values)
    d = {}
    for x in nm: d[x] = cv[x]
    return [[n],d]

# FREQUENCY ENCODING from dictionary
def encode_FE_test(df,col,mp,xx=1.0):
    cv = df[col].value_counts(dropna=False)
    n = col+"_FE"
    df[n] = df[col].map(cv)
    df.loc[~df[col].isin(mp),n] = xx*np.mean(cv.values)
    df[n] = df[n] / max(cv.values)
    return [[n],mp]

# BINARY ENCODING
def encode_BE(df,col,val='xyz'):
    if val=='xyz':
        print('BE_encoding all')
        v = df[col].unique()
        n = []
        for x in v: n.append(encode_BE(df,col,x)[0][0])
        return [n,0]
    n = col+"_BE_"+str(val)
    if nan_check(val):
        df[n] = df[col].isna()
    elif isinstance(val, (list,)):
        if not isinstance(val[0], str):
            print('BE_encode Warning: val list not str')
        n = col+"_BE_"+str(val[0])+"_"+str(val[-1])
        d = {}
        for x in val: d[x]=1
        df[n] = df[col].map(d).fillna(0)
    else:
        if not isinstance(val, str):
            print('BE_encode Warning: val is not str')
        df[n] = df[col]==val
    df[n] = df[n].astype('int8')
    return [[n],0]

# BINARY ENCODING first ct columns by freq
def encode_BE_partial(df,col,ct):
    cv = df[col].value_counts(dropna=False)
    nm = cv.index.values[0:ct]
    d = {}
    n = []
    for x in nm: 
        n.append(encode_BE(df,col,x)[0][0])
        d[x] = 1
    return [n,d]

# BINARY ENCODING from dictionary
def encode_BE_test(df,col,mp):
    n = []
    for x in mp: n.append(encode_BE(df,col,x)[0][0])
    return [n,0]

# NUMERIC ENCODING
def encode_NE(df,col):
    n = col+"_NE"
    df[n] = df[col].astype(float)
    mx = np.std(df[n])
    mn = df[n].mean()
    df[n] = (df[n].fillna(mn) - mn) / mx
    return [[n],[mn,mx]]

# NUMERIC ENCODING from mean and std
def encode_NE_test(df,col,mm):
    n = col+"_NE"
    df[n] = df[col].astype(float)
    df[n] = (df[n].fillna(df[n].mean()) - mm[0]) / mm[1]
    return [[n],mm]

# # Encode Train and Test Data
# Let's encode the variables above and a few others that EDA has identified. We will created 24 new columns (variables) from 16 original columns (variables).

# In[ ]:


cols = []

# NUMERIC ENCODE
cols += encode_NE(df_train,'Census_TotalPhysicalRAM')[0]
cols += encode_NE(df_train,'AVProductsInstalled')[0]
cols += encode_NE(df_train,'Census_ProcessorCoreCount')[0]

# CATEGORY ENCODE for logistic regression
tmp = encode_BE_partial(df_train,'SmartScreen',5)
cols += tmp[0]; dict_smartscreen = tmp[1]
tmp = encode_BE_partial(df_train,'AVProductStatesIdentifier',5)
cols += tmp[0]; dict_productstate = tmp[1]

# BINARY ENCODE
cols += encode_BE(df_train,'Processor','x86')[0]
cols += encode_BE(df_train,'Census_IsTouchEnabled','1')[0]
cols += encode_BE(df_train,'Census_HasOpticalDiskDrive','1')[0]
cols += encode_BE(df_train,'Census_InternalPrimaryDisplayResolutionVertical',['800','600'])[0]
cols += encode_BE(df_train,'Census_PowerPlatformRoleName','Slate')[0]
cols += encode_BE(df_train,'Wdft_IsGamer','1')[0]

#FREQUENCY ENCODE
cols += encode_FE(df_train,'Census_ProcessorModelIdentifier')[0]
cols += encode_FE(df_train,'Week')[0]
#FREQUENCY ENCODE remove noise
tmp = encode_FE_partial(df_train,'Census_OSInstallTypeName',7)
cols += tmp[0]; dict_osinstalltype = tmp[1]

#TARGET ENCODE remove noise
tmp = encode_TE_partial(df_train,'CountryIdentifier',150)
cols += tmp[0]; dict_country = tmp[1]
tmp = encode_TE_partial(df_train,'OsBuild',5)
cols += tmp[0]; dict_osbuild = tmp[1]

# # Model with Logistic Regression
# 
# Using our 24 new variables, let's build a simple linear logistic regression model.

# In[ ]:


import statsmodels.api as sm

logr = sm.Logit(df_train['HasDetections'], df_train[cols])
logr = logr.fit(disp=0)
df_train['Prob'] = logr.predict(df_train[cols])
print('Training complete')

# # Training ROC Curve
# This competiton's metric is AUC, area under ROC. Let's plot our training ROC and calculate our training AUC. (We should really calculate validation AUC, but since our model is linear, training AUC should be similar.)

# In[ ]:


#https://stackoverflow.com/questions/25009284/how-to-plot-roc-curve-in-python
from sklearn import metrics
fpr, tpr, threshold = metrics.roc_curve(df_train['HasDetections'].values, df_train['Prob'].values)
roc_auc = metrics.auc(fpr, tpr)

# method I: plt
import matplotlib.pyplot as plt
plt.figure(figsize=(8,8))
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.3f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

# # Predict and Submit to Kaggle

# In[ ]:


cols = []

# NUMERIC ENCODE
cols += encode_NE(df_test,'Census_TotalPhysicalRAM')[0]
cols += encode_NE(df_test,'AVProductsInstalled')[0]
cols += encode_NE(df_test,'Census_ProcessorCoreCount')[0]

# CATEGORY ENCODE for logistic regression
cols += encode_BE_test(df_test,'SmartScreen',dict_smartscreen)[0]
cols += encode_BE_test(df_test,'AVProductStatesIdentifier',dict_productstate)[0] 

# BINARY ENCODE
cols += encode_BE(df_test,'Processor','x86')[0]
cols += encode_BE(df_test,'Census_IsTouchEnabled','1')[0]
cols += encode_BE(df_test,'Census_HasOpticalDiskDrive','1')[0]
cols += encode_BE(df_test,'Census_InternalPrimaryDisplayResolutionVertical',['800','600'])[0]
cols += encode_BE(df_test,'Census_PowerPlatformRoleName','Slate')[0]
cols += encode_BE(df_test,'Wdft_IsGamer','1')[0]

# FREQUENCY ENCODE
cols += encode_FE(df_test,'Census_ProcessorModelIdentifier')[0]
cols += encode_FE(df_test,'Week')[0]
cols += encode_FE_test(df_test,'Census_OSInstallTypeName',dict_osinstalltype)[0]

# TARGET ENCODE
cols += encode_TE_test(df_test,'CountryIdentifier',dict_country)[0]
cols += encode_TE_test(df_test,'OsBuild',dict_osbuild)[0]

# In[ ]:


# PREDICT
df_test['HasDetections'] = logr.predict(df_test[cols])
# SUBMIT
df_test[['MachineIdentifier','HasDetections']].to_csv('submission.csv', index=False)

# ![image](http://playagricola.com/Kaggle/MalwareScore2.png)
# 
# # Conclusion
# In this kernel, we added timestamps to the Microsoft Malware training and test data. Then we performed EDA to observe relationships between category variables, time, detection rate, and density. Based on these observations, we found variables that appear predictive and chose appropriate encoding schemes. Finally we built a simple model from these variables and observed that they are indeed predictive as evidenced by a 0.67 training AUC and 0.64 LB AUC.
# 
# For brevity, I only displayed and discussed 6 variables above. There are many more interesting and predictive variables among the 80 provided by Microsoft. You are welcome to fork my kernel, download the timestamp database, and use my EDA functions to explore the data more. Let me know what you discover.
# 
# # Variable Importance
# 
# Below lists the order of importance of the variables in this model. Logistic regression predicts the probability to be `sigmoid(a_1*x_1+...+a_23*x23)` where a's are learned coefficients and x's are our variables. Importance of `x_k` equals 0 if p-value is greater than 0.05 and equal `a_k*SD(x_k)` otherwise where SD is the standard deviation of `x_k`. If a variable has non-zero importance then it helps the model in a statistically significant way.

# In[ ]:


d = {}; 
for var in cols: d[var] = np.std(df_train[var])
importance = logr.params * pd.Series(d) * logr.pvalues.map(lambda x: 0 if x>=0.05 else 1)
order = importance.abs().sort_values(ascending = False)
importance[order.index]

# # More EDA
# Below are plots of the 9 variables used in this model that weren't displayed above. You willl notice that each variable has independent predictive power. They don't require the presence of other variables and they are independent of time. That makes them good for our linear Logistic Regression model.

# In[ ]:


var = ['Wdft_IsGamer','AVProductStatesIdentifier','Census_PowerPlatformRoleName','OsBuild',
       'Processor','Census_ProcessorCoreCount','Census_InternalPrimaryDisplayResolutionVertical',
       'Census_ProcessorModelIdentifier','Census_IsTouchEnabled','Census_HasOpticalDiskDrive']
bars = [2,20,10,5,10,3,10,20,10,10]
lines = [2,4,3,4,2,3,4,4,2,2]
for i in range(len(var)):
    staticPlot(df_train,var[i],show=0.99,bars=bars[i])
    dynamicPlot(df_train,var[i],top2=lines[i])

# # Correlation Matrix
# Below is the correlation matrix of the 24 variables used in this model. Observe that `AVProductsIdentifier=53447` correlates with `AVProductsInstalled`. Also `Census_PowerPlatformRoleName=Slate`, `Census_InternalPrimaryDisplayResolutionVertical=800 or 600`, `Census_IsTouchEnabled`,  and `Processor=x86` are all correlated. Also `Census_TotalPhysicalRAM` and `Census_ProcessorCoreCount` are correlated.

# In[ ]:


    corr = df_train[cols+['HasDetections']].corr()
    fig, ax = plt.subplots(figsize=(10,10))
    ax.matshow(corr)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90);
    plt.yticks(range(len(corr.columns)), corr.columns);
