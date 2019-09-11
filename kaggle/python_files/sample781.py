#!/usr/bin/env python
# coding: utf-8

# # To discover something new is to explore where it has never been explored.........
# 
# - If the kernel helped you, do share your feedback!
# 
# Thanks

# In[ ]:


from __future__ import division, print_function
## to ignore warnings as we don't like them as such
import warnings
warnings.filterwarnings('ignore')
#plotting lib's
from matplotlib import pyplot as plt
import seaborn as sns
sns.set()
#other's
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_predict, TimeSeriesSplit
from sklearn.model_selection import validation_curve, learning_curve
from sklearn.metrics import confusion_matrix
import pickle
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse import hstack
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV, train_test_split, TimeSeriesSplit
from sklearn.metrics import accuracy_score , roc_curve, auc
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

# In[ ]:


# Read the training and test data sets
train_df = pd.read_csv('../input/train_sessions.csv',
                       index_col='session_id',parse_dates=['time1'])
test_df = pd.read_csv('../input/test_sessions.csv',
                      index_col='session_id',parse_dates=['time1'])

times = ['time%s' % i for i in range(1, 11)]
sites = ['site%s' % i for i in range(1, 11)]

# In[ ]:


train_df.shape, test_df.shape

# In[ ]:


train_df.get_ftype_counts()

# In[ ]:


test_df.get_ftype_counts()

# In[ ]:


sns.countplot(train_df['target']) #Imbalanced Dataset

# In[ ]:


train_df[times] = train_df[times].apply(pd.to_datetime)
test_df[times] = test_df[times].apply(pd.to_datetime)

# Sort the data by time
train_df = train_df.sort_values(by='time1')

train_df[sites] = train_df[sites].fillna(0).astype('int')
test_df[sites] = test_df[sites].fillna(0).astype('int')

# In[ ]:


# Change site1, ..., site10 columns type to integer and fill NA-values with zeros

train_df[sites] = train_df[sites].fillna(0).astype('int')
test_df[sites] = test_df[sites].fillna(0).astype('int')

# Load websites dictionary
with open(r"..//input//site_dic.pkl", "rb") as input_file:
    site_dict = pickle.load(input_file)

# Create dataframe for the dictionary
sites_dict = pd.DataFrame(list(site_dict.keys()), index=list(site_dict.values()), columns=['site'])
print(u'Websites total:', sites_dict.shape[0])

# In[ ]:


# Top websites in the training data set
top_sites = pd.Series(train_df[sites].fillna(0).values.flatten()
                     ).value_counts().sort_values(ascending=False).head(5)
top_sites

# In[ ]:


# Alice's preferences
top_sites_alice = pd.Series(train_df[train_df.target==1][sites].fillna(0).values.flatten()
                           ).value_counts().sort_values(ascending=False).head()
top_sites_alice

# In[ ]:


time_df = pd.DataFrame(index=train_df.index)
time_df['target'] = train_df['target']

# Find sessions' starting and ending
time_df['min'] = train_df[times].min(axis=1)
time_df['max'] = train_df[times].max(axis=1)

# Calculate sessions' duration in seconds
time_df['seconds'] = (time_df['max'] - time_df['min']) / np.timedelta64(1, 's')

# In[ ]:


time_df_test = pd.DataFrame(index=test_df.index)

# Find sessions' starting and ending
time_df_test['min'] = test_df[times].min(axis=1)
time_df_test['max'] = test_df[times].max(axis=1)

# Calculate sessions' duration in seconds
time_df_test['seconds'] = (time_df_test['max'] - time_df_test['min']) / np.timedelta64(1, 's')

# - y_train - our target variable 
# 
# - full_df - the merged table of input data (training and test samples together) 
# 
# * idx_split - the index by which we will separate the training sample from the test
# 

# In[ ]:


# Our target variable
y_train = train_df['target']

# United dataframe of the initial data 
full_df = pd.concat([train_df.drop('target', axis=1), test_df])

#full_df[:].fillna(method='bfill',inplace=True)

# Index to split the training and test data sets
idx_split = train_df.shape[0]

# In[ ]:


full_df.head(5)

# In[ ]:


full_sites = full_df[sites] #only sites
full_sites.head(3)

# ## Creating characteristics
# 
# - Extracting attributes from the site table 
# 
# - Count the number of unique sites in each session

# In[ ]:


#Getting Unique Counts
unique_count = []
for row in full_sites.values:
    unique = np.unique (row)
    if 0 in unique:
        unique_count.append(len(unique) - 1)
    else:
        unique_count.append(len(unique))
unique_count = np.array(unique_count).reshape(-1,1)

# In[ ]:


unique_count[:10] #print few

# In[ ]:


additional_data_df  =  pd.DataFrame(data = unique_count ,columns = ['unique'],index = full_df.index ) #my features dataframe

# Create a dictionary where the keys are sites, and the values ​​- the number of sessions in which this site met.

# In[ ]:


site_dict  =  {} 
for  row  in  full_sites . values : 
    for  site_id  in  np . unique ( row ): 
        if  site_id  in  site_dict : 
            site_dict [ site_id ]  +=  1 
        else : 
            site_dict [ site_id ]  =  1 
site_dict.pop(0)

# Make the presence indicator in the site of the site, which is included in the top 10 popular sites

# In[ ]:


top_10 = sorted (list (site_dict.items ()), key = lambda tup: tup [1], reverse = True) [: 10]
top_10 = [element [0] for element in top_10]
have_top_10 = np.zeros ((full_sites.shape [0], 1), dtype = int)
ind = 0
for row in full_sites.values:
    unique = np.unique (row)
    for site_id in unique:
        if site_id in top_10:
            have_top_10[ind] = 1
    ind += 1

have_top_10

# In[ ]:


additional_data_df [ 'have_top_10' ]  =  have_top_10

# In[ ]:


additional_data_df.head(3)

# Count the number of absolutely unique sites in the session (the sites met only in this session)

# In[ ]:


absolutely_unique_count  =  np.zeros((full_sites.shape[0], 1 ), dtype = int ) 
ind  =  0 
for  row  in  full_sites . values : 
    unique  =  np . unique ( row ) 
    absolutely_unic_row  =  {} 
    for  site_id  in  unique : 
        if (site_id !=0) and (site_dict[site_id] == 1) and (site_id not in absolutely_unic_row ): 
            absolutely_unique_count [ ind ]  +=  1 
            absolutely_unic_row [ site_id ]  =  1 
    ind  +=  1
absolutely_unique_count

# In[ ]:


additional_data_df['absolutely_unique_count'] = absolutely_unique_count

# Make an indicator of the presence of an absolutely unique site in the session

# In[ ]:


absolutely_unique = (absolutely_unique_count > 0 ).astype(int)

# In[ ]:


additional_data_df ['have_absolutely_unique' ] = absolutely_unique

# In[ ]:


sns.countplot(additional_data_df['absolutely_unique_count'])

# ### Work over time 
# 
# - We extract a table in which there will be only columns of time

# In[ ]:


full_time  =  full_df[times] 
full_time.head(3)

# **We calculate the time of staying on each site within one session**

# In[ ]:


def  get_time_diff (row): 
    time_length = row.shape[0] - 1 
    time_diff = [0]*time_length 
    i = 0 
    while (i < time_length)and pd.notnull(row[i+1]): 
        time_diff[i] = (row[i+1] - row[i]) /np.timedelta64(1,'s') 
        i += 1 
    return  time_diff

# In[ ]:


time_diff = []
for row in full_time.values:
    time_diff.append (get_time_diff (row))
time_diff = np.log1p(np.array(time_diff).astype(float))

# In[ ]:


## Appending The Newly Created Features
time_names = ['time_diff'+str(j) for j in range(1,10)] 
for ind,column_name in enumerate(time_names): 
    additional_data_df[column_name] = time_diff[:,ind] 

# **We will calculate the total time spent at each session**

# In[ ]:


def get_total_time(row): 
    time_length = row.shape[0] - 1 
    i = time_length 
    while pd.isnull( row [ i ]): 
        i -= 1 
    return (row[i] - row[0]) / np.timedelta64(1,'s')

# In[ ]:


total_time = []
for row in full_time.values:
    total_time.append(get_total_time(row))
total_time = np.array(total_time).reshape(-1,1).astype(int)

# In[ ]:


additional_data_df['total_time'] =  total_time 
additional_data_df['total_time'] =  np.log1p(additional_data_df['total_time'].values)

# In[ ]:


additional_data_df['start_hour']  =  full_time['time1'].apply(lambda ts: ts.hour) 
additional_data_df['holiday']     = (full_time['time1'].dt.dayofweek >= 5).astype(int) 
additional_data_df['day_of_week'] = (full_time['time1'].dt.dayofweek).astype(int)

# In[ ]:


additional_data_df.head(5)

# In[ ]:


add_data_train = additional_data_df[: idx_split ].copy() 
add_data_test  = additional_data_df[ idx_split :]

# ### Analysis of training and test samples 

# In[ ]:


add_data_train.loc[:, 'target' ]= y_train

# **Let's see what signs we have**

# In[ ]:


dtype_df = add_data_train.dtypes.reset_index() 
dtype_df.columns = ['Column_name','Column_type'] 
dtype_df

# In[ ]:


sns.countplot(add_data_train['target']) ;plt.legend();
print(add_data_train['target'].value_counts())

# **All characteristics are of type int, but they can be divided into two groups-**
# - **categorical **
# 
# - ** quantitative.**

# In[ ]:


plt.figure(figsize=(9,7))
add_data_train['target'].value_counts().plot(kind = 'bar', label = 'Intruder')
plt.legend()
print(add_data_train['target'].value_counts())

# **As you can see, we have two unbalanced classes**

# **Let's look at the correlation table of the dataset**

# In[ ]:


corr = add_data_train.corr('spearman') 
plt.figure(figsize = ( 16 , 11 )) 
sns.heatmap(corr,annot=True,fmt='.2f',cmap="YlGnBu");

# **Let's look at the distribution of some features**

# In[ ]:


feature_list = ['unique','absolutely_unique_count','start_hour','day_of_week'] 
for  column_name in feature_list: 
    fig , (ax1,ax2) = plt.subplots(1,2,figsize = ( 15 , 6 )) 
    fig.suptitle(column_name,fontsize=16) 
    sns.countplot(add_data_train[column_name],ax=ax1) 
    ax1.set_title("Train distribution") 
    for tick in ax1.get_xticklabels(): 
        tick.set_rotation(45)
    sns.countplot(add_data_test[column_name],ax=ax2) 
    ax2.set_title("Test distribution") 
    for tick in ax2.get_xticklabels(): 
        tick.set_rotation(45)

# - As you can see, all the attributes are distributed approximately equally in the training and test sample. 
# 
# - The start_hour attribute is distributed from 7 to 23 hours, and not from 0 to 23 hours, as expected.
# 
# - We will analyze the effect of the characteristics on the target variable

# In[ ]:


def get_target_dist( column_name ): 
    fig , (ax1,ax2) = plt.subplots(1,2,figsize = (15,6)) 
    fig.suptitle(column_name,fontsize=16) 
    sns.countplot(add_data_train[add_data_train['target'] == 1][column_name],ax=ax1) 
    ax1.set_title("Intruder count (target: 1)") 
    for tick in ax1.get_xticklabels(): 
        tick.set_rotation(45)
    sns.barplot(x=column_name,y="target",data=add_data_train,ax=ax2) 
    ax2.set_title("Intruder proportion (target: 1)") 
    for tick in ax2.get_xticklabels(): 
        tick.set_rotation(45)

# In[ ]:


get_target_dist('start_hour')

# Frauds were recorded between 9 and 18 hours, with the period from 16 to 18 hours most saturated with suspicious activity. On the basis of this, we will distinguish 3 features:
# 
# - morning: 7 - 11
# - midday: 12 - 15
# - evening: 16 - 20

# In[ ]:


add_data_train ['morning']  = add_data_train['start_hour'].apply(lambda hour:int(hour >= 7 and hour <= 11 )) 
add_data_train ['midday']   = add_data_train['start_hour'].apply(lambda hour:int(hour >= 12 and hour <= 18)) 
add_data_train ['evening']  = add_data_train['start_hour'].apply(lambda hour:int(hour >= 19 and hour <= 23))
add_data_train ['night']    = add_data_train['start_hour'].apply(lambda hour:int(hour >= 0 and hour <= 6))

add_data_test ['morning']  = add_data_test['start_hour'].apply(lambda hour:int(hour >= 7 and hour <= 12 )) 
add_data_test ['midday']   = add_data_test['start_hour'].apply(lambda hour:int(hour >= 12 and hour <= 18)) 
add_data_test ['evening']  = add_data_test['start_hour'].apply(lambda hour:int(hour >= 19 and hour <= 23))
add_data_test ['night']    = add_data_test['start_hour'].apply(lambda hour:int(hour >= 0 and hour <= 6))

# In[ ]:


get_target_dist('unique')

# In[ ]:


get_target_dist('day_of_week')

# **The least suspicious activity was on Wednesdays, Saturdays and Sundays. But on Mondays(0), on the contrary, more than on other days.**

# In[ ]:


get_target_dist('absolutely_unique_count')

# In[ ]:


feature_list = ['have_top_10', 'have_absolutely_unique']
for column_name in feature_list:
    plt.figure(figsize=(7, 5))
    ax = sns.barplot(x=column_name, y="target", data=add_data_train)
    ax.set(ylabel='Target 1 proportion')
    plt.title(column_name)
    plt.show()

# In[ ]:


ax = sns.boxplot(x = 'target', y = 'total_time', data = add_data_train)

# The duration of sessions with suspicious activity is on average less than that of the usual ones.
# 
# **General observations:**
# 
# - Most often, suspicious activity was seen between 16 and 18 hours
# - Also on Mondays, suspicious activity occurs most often
# - On average, the duration of sessions with suspicious activity is less than that of conventional
# - Suspicious sessions are less common absolutely unique sites (sites that occur only once in the history of visiting sites)

#  Adding Cyclic Co-ordinates

# In[ ]:


pi = np.pi
add_data_train['hour_sin_x'] = add_data_train['start_hour'].apply(lambda ts: np.sin(2*pi*ts/24.))
add_data_train['hour_cos_x'] = add_data_train['start_hour'].apply(lambda ts: np.cos(2*pi*ts/24.))

add_data_test['hour_sin_x'] = add_data_test['start_hour'].apply(lambda ts: np.sin(2*pi*ts/24.))
add_data_test['hour_cos_x'] = add_data_test['start_hour'].apply(lambda ts: np.cos(2*pi*ts/24.))

# In[ ]:


def kmeansshow(k,X):

    from sklearn import cluster
    from matplotlib import pyplot

    kmeans = cluster.KMeans(n_clusters=k)
    kmeans.fit(X)
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    #print centroids

    for i in range(k):
        # select only data observations with cluster label == i
        ds = X[np.where(labels==i)]
        # plot the data observations
        pyplot.plot(ds[:,0],ds[:,1],'o')
        # plot the centroids
        lines = pyplot.plot(centroids[i,0],centroids[i,1],'kx')
        # make the centroid x's bigger
        pyplot.setp(lines,ms=15.0)
        pyplot.setp(lines,mew=2.0)
    pyplot.legend()
    pyplot.show()
    return centroids


# In[ ]:


kmeansshow(3,add_data_train[['hour_sin_x', 'hour_cos_x']].values)
kmeansshow(6,add_data_train[['hour_sin_x', 'hour_cos_x']].values)

# In[ ]:


kmeansshow(3,add_data_test[['hour_sin_x', 'hour_cos_x']].values)
kmeansshow(6,add_data_test[['hour_sin_x', 'hour_cos_x']].values)

# ### Preprocessing data 
# - Convert categorical variables to dummies using get_dummies

# In[ ]:


add_train_dummies  = pd.get_dummies(add_data_train , columns=['day_of_week']) 
add_train_dummies.drop(['target'],axis = 1,inplace=True) 
add_test_dummies  = pd.get_dummies(add_data_test,columns=['day_of_week'])

# In[ ]:


idx_split

# In[ ]:


names_for_scale  =  ['time_diff' + str(x) for x in range (1,10)]  +  ['unique','absolutely_unique_count','total_time'] 
scaler = StandardScaler() 
for column_name in names_for_scale : 
    add_train_dummies[column_name] =scaler.fit_transform(add_train_dummies[column_name].values.reshape(-1,1)) 
    add_test_dummies[column_name]  =scaler.transform(add_test_dummies[column_name].values.reshape(-1,1))

# In[ ]:


# Sequence with indexes 
sites_flatten = full_sites.values.flatten()
# 
full_sites_sparse=csr_matrix(([1]  *  sites_flatten . shape [ 0 ], sites_flatten ,  
                                range(0,sites_flatten.shape[0]+10,10)))[:,1:]

X_train_sparse  =  full_sites_sparse [: idx_split ] 
X_test_sparse  =  full_sites_sparse [ idx_split :] 
full_sites_sparse.shape #(336358, 48371)

# In[ ]:


full_sites = full_df[sites].astype('str')
lst = full_sites[sites].as_matrix().tolist()
flat_list = [' '.join(sublist) for sublist in lst]
vect = TfidfVectorizer(ngram_range=(1,4), max_features=100000,analyzer='char_wb')
tfidf_matrix = vect.fit_transform(flat_list)

X_train_tf = tfidf_matrix[:idx_split]
X_test_tf  = tfidf_matrix[idx_split:]

X_train_tf.shape, X_test_tf.shape

# **Combine the site matrix and additional features**

# In[ ]:


x_train_full  =  csr_matrix(hstack([X_train_tf,add_train_dummies.values])) 
x_test_full   =  csr_matrix(hstack([X_test_tf ,add_test_dummies.values])) 
x_train_full.shape,x_test_full.shape

# ![Image](https://hsto.org/webt/8i/5k/vx/8i5kvxrehatyvf-l3glz_-ymhtw.png)

# In[ ]:


target = train_df['target']
X_train , X_valid , y_train , y_valid = train_test_split(x_train_full,target ,test_size=0.2) 
tscv = TimeSeriesSplit(n_splits = 5);

# In[ ]:


tscv

# In[ ]:


[(el[0].shape, el[1].shape) for el in tscv.split(X_train)]

# **Since our classes are highly unbalanced, the accuracy metric here does not fit.**
# 
# **We will use ROC_AUC as a metric, and also look at the results of confusion_matrix**

# Since we have historical data, interrelated with time, the standard cross-fitting in this case will not work. We will use cross-validation with TimeSeriesSplit - cross-validation techniques for time series. 
# Its essence lies in the following: we divide the sample, for example, into 5 parts: [1, 2, 3, 4, 5], then we train in the following way
# 
# - Fold 1: training [1], Test [2]
# - Fold 2: Training [1, 2], Test [3]
# - Fold 3: teaching [1, 2, 3], Test [4]
# - Fold 4: training [1, 2, 3, 4], Test [5]
# 
# **This approach will help to better evaluate the quality of the algorithm.**

# In[ ]:


clf_lr = LogisticRegression (random_state = 42, n_jobs=1, solver='lbfgs', max_iter=8000)
clf_lr.fit (X_train, y_train)
preds_lr = clf_lr.predict_proba (X_valid)[:, 1]
print ('Train test split LogisticRegression score:% s ROC AUC'% round (roc_auc_score (y_valid, preds_lr), 4))
cross_score_lr = np.mean (cross_val_score (clf_lr, x_train_full, target, scoring = 'roc_auc', cv = tscv))
print ('Cross validation LogisticRegression score:% s ROC AUC'% round (cross_score_lr, 4))

# In[ ]:


clf_rf = RandomForestClassifier (random_state = 42, n_estimators = 100)
clf_rf.fit (X_train, y_train)
preds_rf = clf_rf.predict_proba (X_valid) [:, 1]
print ('Train test split RandomForestClassifier score:% s ROC AUC'% round (roc_auc_score (y_valid, preds_rf), 4))
cross_score_rf = np.mean (cross_val_score (clf_rf, x_train_full, target, scoring = 'roc_auc', cv = tscv))
print ('Cross validation RandomForestClassifier score:% s ROC AUC'% round (cross_score_rf, 4))

# In[ ]:


reg_xgb = xgb.XGBRegressor(10, 0.1, 1000, objective= 'binary:logistic', random_state = 42, booster = 'gblinear',scale_pos_weight = 109)
reg_xgb.fit(X_train, y_train)
preds_xgb_reg = reg_xgb.predict(X_valid)

# In[ ]:


preds_xgb_reg.shape

# In[ ]:


print ('Train test split XGBRegressor score:% s ROC AUC'% round (roc_auc_score (y_valid, preds_xgb_reg), 4))
cross_score_xgb_reg = np.mean (cross_val_score (reg_xgb, x_train_full, target, scoring = 'roc_auc', cv = tscv))
print ('Cross validation XGBRegressor score:% s ROC AUC'% round (cross_score_xgb_reg, 4))

# In[ ]:


clf_xgb = xgb.XGBClassifier (random_state = 42, booster = 'gblinear')
clf_xgb.fit (X_train, y_train)
preds_xgb = clf_xgb.predict_proba (X_valid) [:, 1]
print ('Train test split XGBClassifier score:% s ROC AUC'% round (roc_auc_score (y_valid, preds_xgb), 4))
cross_score_xgb = np.mean (cross_val_score (clf_xgb, x_train_full, target, scoring = 'roc_auc', cv = tscv))
print ('Cross validation XGBClassifier score:% s ROC AUC'% round (cross_score_xgb, 4))

# In[ ]:


fpr_lr,tpr_lr,threshold=roc_curve(y_valid,preds_lr)
roc_auc_lr = auc(fpr_lr,tpr_lr)

fpr_rf,tpr_rf,threshold=roc_curve(y_valid,preds_rf)
roc_auc_rf = auc(fpr_rf,tpr_rf)

fpr_xgb_reg,tpr_xgb_reg,threshold=roc_curve(y_valid,preds_xgb_reg)
roc_auc_xgb_reg = auc(fpr_xgb_reg,tpr_xgb_reg)

fpr_xgb,tpr_xgb,threshold=roc_curve(y_valid,preds_xgb)
roc_auc_xgb = auc(fpr_xgb,tpr_xgb)

plt.figure(figsize=(12,8))
plt.title('Receiver Operating Characteristic Curve')
plt.plot(fpr_lr, tpr_lr,'r', label='AUC LR=%0.4f'%roc_auc_lr)
plt.plot(fpr_rf, tpr_rf,'b', label='AUC RF=%0.4f'%roc_auc_rf)
plt.plot(fpr_xgb, tpr_xgb,'g', label='AUC XGB REG=%0.4f'%roc_auc_xgb_reg)
plt.plot(fpr_xgb_reg, tpr_xgb_reg, 'y', label='AUC XGB=%0.4f'%roc_auc_xgb)
plt.legend(loc='lowerright')
plt.plot([0,1],[0,1],'r--')
plt.xlim([0,1])
plt.ylim([0,1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate');

# In[ ]:


def show_confusion_matrix(y_true,y_pred,title='Confusionmatrix'):
    table=confusion_matrix(y_true.values,y_pred)
    fig,ax=plt.subplots(frameon=False)
    fig.set_size_inches(4,3)
    fig.suptitle(title,fontsize=20)
    ax.axis('off')
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

    the_table=ax.table(cellText=table,
                        colWidths=[0.5]*len([0,1]),
                        rowLabels=['True 0','True 1'],colLabels=['Predicted 0','Predicted 1'],
                        cellLoc='center',rowLoc='center',loc="center")
    the_table.set_fontsize(34)
    the_table.scale(1,4)
    plt.show()

# In[ ]:


show_confusion_matrix(y_valid, clf_lr.predict(X_valid))

# In[ ]:


show_confusion_matrix(y_valid, clf_xgb.predict(X_valid))

# In[ ]:


def get_auc_lr_valid(X, y, C=1.0, seed=17, ratio = 0.9):
    # Split the data into the training and validation sets
    idx = int(round(X.shape[0] * ratio))
    # Classifier training
    lr = LogisticRegression(C=C, random_state=seed,max_iter=8000,n_jobs=1,solver='lbfgs').fit(X[:idx, :], y[:idx])
    # Prediction for validation set
    y_pred = lr.predict_proba(X[idx:, :])[:, 1]
    # Calculate the quality
    score = roc_auc_score(y[idx:], y_pred)
    return score

# In[ ]:


from tqdm import tqdm
# List of possible C-values
Cs = np.logspace(-1, 1, 20)

scores = []

for C in tqdm(Cs):
     scores.append(get_auc_lr_valid(X_train, y_train, C=C))

# In[ ]:


Cs

# In[ ]:


scores, max(scores), Cs[np.argmax(scores)]

# In[ ]:


final_model = LogisticRegression(random_state = 17, C = Cs[np.argmax(scores)],n_jobs=1) 
final_model.fit(X_train,y_train) 
print('ROC_AUC on the test sample: {} '.format(round(roc_auc_score(y_valid,final_model.predict_proba(X_valid)[:,1]),4)))

# In[ ]:


# Function for writing predictions to a file
def write_to_submission_file(predicted_labels, out_file,
                             target='target', index_label="session_id"):
    predicted_df = pd.DataFrame(predicted_labels,
                                index = np.arange(1, predicted_labels.shape[0] + 1),
                                columns=[target])
    predicted_df.to_csv(out_file, index_label=index_label)

# In[ ]:


preds = final_model.predict_proba(x_test_full)[:,1]

# In[ ]:


preds[:10]

# In[ ]:


preds = np.where(preds > 0.945, 0.985, preds)

# In[ ]:


write_to_submission_file(preds,'re-git_v4.csv')

# ### Hope it was Useful (More To Come, till then Enjoy and share your Views Below...****)
# 
# Thanks
# 
# - *Aditya Soni*
