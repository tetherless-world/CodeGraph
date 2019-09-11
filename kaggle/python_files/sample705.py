#!/usr/bin/env python
# coding: utf-8

# 1.  [EDA](#idEDA)
# 2. [Feature Engineering](#idFeature_Engineering)
# 3.  [Prediction](#idPrediction)
# 4.  [Analysis of Prediction](#idAnalysis_of_Prediction)

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.feature_extraction.text import TfidfVectorizer
train_orig = pd.read_csv('../input/train.csv')
test_orig = pd.read_csv('../input/test.csv')
subm = pd.DataFrame()
subm['id'] = test_orig.id.values

# <div id="idEDA">EDA</div>

# In[ ]:


train_orig['bool_belongs_to_collection'] = (train_orig['belongs_to_collection'].notnull()).astype(int)
test_orig['bool_belongs_to_collection'] = (test_orig['belongs_to_collection'].notnull()).astype(int)

# In[ ]:


train_orig['split'] = 'train'
test_orig['split'] = 'test'

# In[ ]:


train_test = pd.concat([train_orig[['popularity','budget','split','bool_belongs_to_collection']], test_orig[['popularity','budget','split','bool_belongs_to_collection']]])

# In[ ]:


train_test.shape

# In[ ]:


fig, ax = plt.subplots()
sns.scatterplot(x="popularity", y="budget", hue="split", data=train_test,ax=ax)


# In[ ]:


fig, ax = plt.subplots()
fig.set_size_inches(18.5, 10.5)
sns.scatterplot(x="popularity", y="budget", hue="split",style='bool_belongs_to_collection', data=train_test,ax=ax, alpha=0.4)
ax.set_xlim([0,100])

# In[ ]:


#https://www.kaggle.com/jlove5/avocados-usa-prices

g = sns.catplot(x='split',y='budget',data=train_test, kind='box' )
g.set_axis_labels("Split", "budget")

# In[ ]:


from bokeh.plotting import figure, output_file, show, output_notebook
from bokeh.models import ColumnDataSource
output_notebook()

x = train_orig.popularity
y = train_orig.revenue

source = ColumnDataSource(data=dict(
    popularity=train_orig.popularity,
    revenue=train_orig.revenue,
    original_language=train_orig.original_language,
))


output_file("popularity_revenue.html", title="Popularity, Revenue", mode="cdn")
TOOLTIPS = [
    ("Popularity", "@popularity"),
    ("Revenue", "@revenue"),
    ("Original Language", "@original_language"),
    
]

p = figure(tooltips=TOOLTIPS,y_axis_type="log")

p.circle('popularity', 'revenue',fill_alpha=0.6, line_color=None, source = source)
p.xaxis.axis_label = "popularity"
p.yaxis.axis_label = "revenue"
show(p)


# In[ ]:


from bokeh.plotting import figure, output_file, show, output_notebook
from bokeh.models import ColumnDataSource
output_notebook()

x = train_orig.budget
y = train_orig.revenue

source = ColumnDataSource(data=dict(
    budget=train_orig.budget,
    revenue=train_orig.revenue,
    original_language=train_orig.original_language,
))


# output to static HTML file (with CDN resources)
output_file("budget_revenue.html", title="Budget, Revenue", mode="cdn")
TOOLTIPS = [
    ("Budget", "@budget"),
    ("Revenue", "@revenue"),
    ("Original Language", "@original_language"),
    
]

p = figure(tooltips=TOOLTIPS,y_axis_type="log")

p.circle('budget', 'revenue',fill_alpha=0.6, line_color=None, source = source)
p.xaxis.axis_label = "budget"
p.yaxis.axis_label = "revenue"
show(p)

# In[ ]:


print(len(train_orig.columns))
print(len(test_orig.columns))
olang = train_orig.original_language.value_counts()[train_orig.original_language.value_counts()>5].index.tolist()
print(olang)
print(len(olang))
train_orig_sample = train_orig[train_orig.original_language.isin(olang)].copy()
print(train_orig_sample.original_language.value_counts())

# In[ ]:


train_orig_sample.loc[:,'revenue'] = np.log(train_orig_sample['revenue'].fillna(0)+1)

# In[ ]:


#https://www.kaggle.com/jlove5/avocados-usa-prices

g = sns.catplot(x='original_language',y='revenue',data=train_orig_sample, kind='box', aspect=2 )
g.set_axis_labels("Original language", "Log of revenue")

# In[ ]:


train_orig.info()

# In[ ]:


train_olang = pd.get_dummies(train_orig.original_language)[olang]
train_orig = pd.concat([train_orig,train_olang], axis=1)
#train_orig.head(2)

# In[ ]:


def extract_id(cell):
    return yaml.load(cell)[0]['id']
#train['belongs_to_collection'].dropna().apply(extract_id).value_counts()


# In[ ]:


train_orig.columns

# <div id="idFeature_Engineering">Feature Engineering</div>

# In[ ]:


test_olang = pd.get_dummies(test_orig.original_language)[olang]
test_orig = pd.concat([test_orig,test_olang], axis=1)
#test_orig.head(2)

# In[ ]:


#https://stackoverflow.com/questions/51643427/how-to-make-tfidfvectorizer-only-learn-alphabetical-characters-as-part-of-the-vo
train_orig['cast_crew'] = train_orig.cast + ' ' + train_orig.crew 
test_orig['cast_crew'] = test_orig.cast + ' ' + test_orig.crew


# In[ ]:


#https://stackoverflow.com/questions/51643427/how-to-make-tfidfvectorizer-only-learn-alphabetical-characters-as-part-of-the-vo
#vec = TfidfVectorizer(stop_words='english',analyzer='word',max_features=50,max_df=0.5,token_pattern=r'(?u)\b[A-Za-z]{3,}\b')
#vec = TfidfVectorizer(stop_words='english',analyzer='word',max_features=50,max_df=0.5,token_pattern=r"(?u)\b[A-Za-z]{3,}\b|\'name': \'\.(.*?)\', \'")
vec = TfidfVectorizer(analyzer='word',max_features=450,token_pattern=r"'name': '(.*?)'")
vec.fit(train_orig.cast_crew.fillna(''))
vocab = vec.get_feature_names()
vec = TfidfVectorizer(analyzer='word',vocabulary=vocab)
train_crew_w = vec.fit_transform(train_orig.cast_crew.fillna(''))
test_crew_w = vec.transform(test_orig.cast_crew.fillna(''))
train_crew_w_cols = vec.get_feature_names()
train_crew_w_cols = ['crew_'+a for a in train_crew_w_cols]
print(train_crew_w.shape)
print(test_crew_w.shape)
print(train_crew_w_cols)
train_crew_w = pd.DataFrame(train_crew_w.toarray(),columns=train_crew_w_cols)
test_crew_w = pd.DataFrame(test_crew_w.toarray(),columns=train_crew_w_cols)


# In[ ]:


#https://stackoverflow.com/questions/51643427/how-to-make-tfidfvectorizer-only-learn-alphabetical-characters-as-part-of-the-vo
#vec = TfidfVectorizer(stop_words='english',analyzer='word',max_features=50,max_df=0.5,token_pattern=r'(?u)\b[A-Za-z]{3,}\b')
#vec = TfidfVectorizer(stop_words='english',analyzer='word',max_features=50,max_df=0.5,token_pattern=r"(?u)\b[A-Za-z]{3,}\b|\'name': \'\.(.*?)\', \'")
vec = TfidfVectorizer(analyzer='word',max_features=100,token_pattern=r"'name': '(.*?)'")
vec.fit(train_orig.production_companies.fillna(''))
vocab = vec.get_feature_names()
vec = TfidfVectorizer(analyzer='word',vocabulary=vocab)
train_production_companies_w = vec.fit_transform(train_orig.production_companies.fillna(''))
test_production_companies_w = vec.transform(test_orig.production_companies.fillna(''))
train_production_companies_w_cols = vec.get_feature_names()
train_production_companies_w_cols = ['prod_comp_'+a for a in train_production_companies_w_cols]
print(train_production_companies_w.shape)
print(test_production_companies_w.shape)
print(train_production_companies_w_cols)
train_production_companies_w = pd.DataFrame(train_production_companies_w.toarray(),columns=train_production_companies_w_cols)
test_production_companies_w = pd.DataFrame(test_production_companies_w.toarray(),columns=train_production_companies_w_cols)


# In[ ]:


#https://stackoverflow.com/questions/51643427/how-to-make-tfidfvectorizer-only-learn-alphabetical-characters-as-part-of-the-vo

vec = TfidfVectorizer(analyzer='word',max_features=20,token_pattern=r"'name': '(.*?)'")
vec.fit(train_orig.production_countries.fillna(''))
vocab = vec.get_feature_names()
vec = TfidfVectorizer(analyzer='word',vocabulary=vocab)
train_production_countries_w = vec.fit_transform(train_orig.production_countries.fillna(''))
test_production_countries_w = vec.transform(test_orig.production_countries.fillna(''))
train_production_countries_w_cols = vec.get_feature_names()
train_production_countries_w_cols = ['prod_country_'+a for a in train_production_countries_w_cols]
print(train_production_countries_w.shape)
print(test_production_countries_w.shape)
print(train_production_countries_w_cols)
train_production_countries_w = pd.DataFrame(train_production_countries_w.toarray(),columns=train_production_countries_w_cols)
test_production_countries_w = pd.DataFrame(test_production_countries_w.toarray(),columns=train_production_countries_w_cols)


# In[ ]:


#https://stackoverflow.com/questions/51643427/how-to-make-tfidfvectorizer-only-learn-alphabetical-characters-as-part-of-the-vo
#vec = TfidfVectorizer(stop_words='english',analyzer='word',max_features=50,max_df=0.5,token_pattern=r'(?u)\b[A-Za-z]{3,}\b')
#vec = TfidfVectorizer(stop_words='english',analyzer='word',max_features=50,max_df=0.5,token_pattern=r"(?u)\b[A-Za-z]{3,}\b|\'name': \'\.(.*?)\', \'")
vec = TfidfVectorizer(analyzer='word',max_features=50,token_pattern=r"'name': '(.*?)'")

train_belongs_to_collection_w = vec.fit_transform(train_orig.belongs_to_collection.fillna(''))
test_belongs_to_collection_w = vec.transform(test_orig.belongs_to_collection.fillna(''))
train_belongs_to_collection_w_cols = vec.get_feature_names()
train_belongs_to_collection_w_cols = ['collection_'+a for a in train_belongs_to_collection_w_cols]
print(train_belongs_to_collection_w.shape)
print(test_belongs_to_collection_w.shape)
print(train_belongs_to_collection_w_cols)
train_belongs_to_collection_w = pd.DataFrame(train_belongs_to_collection_w.toarray(),columns=train_belongs_to_collection_w_cols)
test_belongs_to_collection_w = pd.DataFrame(test_belongs_to_collection_w.toarray(),columns=train_belongs_to_collection_w_cols)


# In[ ]:


#https://stackoverflow.com/questions/51643427/how-to-make-tfidfvectorizer-only-learn-alphabetical-characters-as-part-of-the-vo
vec = TfidfVectorizer(stop_words='english',analyzer='word',max_features=50,token_pattern=r'(?u)\b[A-Za-z]{3,}\b')
train_genres_w = vec.fit_transform(train_orig.genres.fillna(''))
test_genres_w = vec.transform(test_orig.genres.fillna(''))
train_genres_w_cols = vec.get_feature_names()
train_genres_w_cols = ['genre_'+a for a in train_genres_w_cols]
print(train_genres_w.shape)
print(test_genres_w.shape)
print(train_genres_w_cols)
train_genres_w = pd.DataFrame(train_genres_w.toarray(),columns=train_genres_w_cols)
test_genres_w = pd.DataFrame(test_genres_w.toarray(),columns=train_genres_w_cols)
print(train_genres_w.shape)
print(test_genres_w.shape)

# In[ ]:


#https://stackoverflow.com/questions/51643427/how-to-make-tfidfvectorizer-only-learn-alphabetical-characters-as-part-of-the-vo
train_orig['Keywords_tagline_overview'] = train_orig.title + ' ' + train_orig.Keywords +' ' + train_orig.tagline + ' ' + train_orig.overview
test_orig['Keywords_tagline_overview'] = test_orig.title + ' ' + test_orig.Keywords + ' ' + test_orig.tagline + ' ' + test_orig.overview
from sklearn.feature_extraction.text import TfidfVectorizer
vec = TfidfVectorizer(stop_words='english',analyzer='word',max_features=60,token_pattern=r'(?u)\b[A-Za-z]{3,}\b')
train_tagline_keyword_w = vec.fit_transform(train_orig.Keywords_tagline_overview.fillna(''))
train_tagline_keyword_w_cols = vec.get_feature_names()
print(train_tagline_keyword_w.shape)
test_tagline_w = vec.transform(test_orig.Keywords_tagline_overview.fillna(''))
print(test_tagline_w.shape)
train_tagline_keyword_w_cols = ['kw_tg_ow_' + a for a in train_tagline_keyword_w_cols]
train_tagline_keyword_w = pd.DataFrame(train_tagline_keyword_w.toarray(),columns=train_tagline_keyword_w_cols)
test_tagline_keyword_w = pd.DataFrame(test_tagline_w.toarray(),columns=train_tagline_keyword_w_cols)
train = pd.concat([train_orig,train_tagline_keyword_w,train_genres_w,train_belongs_to_collection_w,
                   train_production_companies_w,train_crew_w,train_production_countries_w], axis=1)
test = pd.concat([test_orig,test_tagline_keyword_w,test_genres_w,test_belongs_to_collection_w,
                  test_production_companies_w,test_crew_w,test_production_countries_w], axis=1)
print(train.shape)
print(test.shape)

# In[ ]:


train['bool_belongs_to_collection'] = (train['belongs_to_collection'].notnull()).astype(int)
test['bool_belongs_to_collection'] = (test['belongs_to_collection'].notnull()).astype(int)

# In[ ]:


len(train_tagline_keyword_w_cols)

# In[ ]:


train['release_date'] = pd.to_datetime(train['release_date'] )
test['release_date'] = pd.to_datetime(test['release_date'] )

# In[ ]:


train['release_month'] = train['release_date'].dt.month
#print(train['release_month'].value_counts())
test['release_month'] = test['release_date'].dt.month
#print(test['release_month'].value_counts())

# In[ ]:


train['release_year'] = train['release_date'].dt.year
#print(train['release_year'].value_counts())
test['release_year'] = test['release_date'].dt.year
#print(test['release_year'].value_counts())

# In[ ]:


train['release_dayofyear'] = train['release_date'].dt.dayofyear
test['release_dayofyear'] = test['release_date'].dt.dayofyear


# In[ ]:


train['release_day_of_week'] = train['release_date'].dt.dayofweek
#print(train['release_day_of_week'].value_counts())
test['release_day_of_week'] = test['release_date'].dt.dayofweek
#print(test['release_day_of_week'].value_counts())
train['release_week'] = train['release_date'].dt.week
test['release_week'] = test['release_date'].dt.week

# In[ ]:


test['release_month'].mode()

# In[ ]:


test['release_year'].mode()

# In[ ]:


test['release_week'].mode()

# In[ ]:


test['release_year'].min()

# In[ ]:


test['release_month'] = test['release_month'].fillna(9.0)
test['release_year'] = test['release_year'].fillna(2014.0)
test['release_week'] = test['release_week'].fillna(36.0)

# In[ ]:



train['bool_homepage'] = (train['homepage'].notnull()).astype(int)
test['bool_homepage'] = (test['homepage'].notnull()).astype(int)

# In[ ]:


train['production_companies_len'] = train['production_companies'].str.len()
test['production_companies_len'] = test['production_companies'].str.len()

# In[ ]:


train['production_countries_len'] = train['production_countries'].str.len()
test['production_countries_len'] = test['production_countries'].str.len()

# In[ ]:


train['Keywords_len'] = train['Keywords'].str.len()
test['Keywords_len'] = test['Keywords'].str.len()

# In[ ]:


train['title_len'] = train['title'].str.len()
test['title_len'] = test['title'].str.len()

# In[ ]:


train['genres_len'] = train['genres'].str.len() 
test['genres_len'] = test['genres'].str.len() 

# In[ ]:


train['cast_crew_len'] = train['cast'].str.len() + train['crew'].str.len()
test['cast_crew_len'] = test['cast'].str.len() + test['crew'].str.len()

# In[ ]:


train['cast_crew_len'].fillna(train['cast_crew_len'].median(),inplace=True)
train['runtime'].fillna(train['runtime'].median(),inplace=True)
train['genres_len'].fillna(train['genres_len'].median(),inplace=True)
train['production_companies_len'].fillna(train['production_companies_len'].median(),inplace=True)
train['production_countries_len'].fillna(train['production_countries_len'].median(),inplace=True)
train['Keywords_len'].fillna(train['Keywords_len'].median(),inplace=True)

# In[ ]:


(train['release_year']>2019).sum()

# In[ ]:


train.loc[(train['release_year']>2019),'release_year']=train['release_year'].median()

# In[ ]:


train['month_into_year'] = train['release_month']*train['release_year']

# In[ ]:


(test['release_year']>2019).sum()

# In[ ]:


test.loc[(test['release_year']>2019),'release_year']=test['release_year'].median()

# In[ ]:


vcast_crew_len = test['cast_crew_len'].median()
test['cast_crew_len'].fillna(vcast_crew_len, inplace=True)
test['runtime'].fillna(test['runtime'].median(),inplace=True)
test['release_month'].fillna(test['release_month'].median(),inplace=True)
test['title_len'].fillna(test['title_len'].median(),inplace=True)
test['release_year'].fillna(test['release_year'].median(),inplace=True)
test['release_day_of_week'].fillna(test['release_day_of_week'].median(),inplace=True)
test['release_dayofyear'].fillna(test['release_dayofyear'].median(),inplace=True)
test['genres_len'].fillna(test['genres_len'].median(),inplace=True)
test['production_companies_len'].fillna(test['production_companies_len'].median(),inplace=True)
test['production_countries_len'].fillna(test['production_countries_len'].median(),inplace=True)
test['Keywords_len'].fillna(test['Keywords_len'].median(),inplace=True)

# In[ ]:


test['month_into_year'] = test['release_month']*test['release_year']

# In[ ]:


features = ['bool_homepage', 'release_dayofyear','production_companies_len', 'production_countries_len', 'Keywords_len' , 'cast_crew_len','budget','popularity','runtime','release_month','release_day_of_week','release_week','genres_len','bool_belongs_to_collection', 'title_len','release_year']

# In[ ]:


train['log_revenue'] = np.log(train['revenue'].fillna(0)+1)

# In[ ]:


from bokeh.plotting import figure, output_file, show, output_notebook
from bokeh.models import ColumnDataSource
output_notebook()


source = ColumnDataSource(data=dict(
    cast_crew_len=train.cast_crew_len,
    revenue=train.revenue,
    original_language=train.original_language,
))


output_file("cast_crew_len_revenue.html", title="cast_crew_len, Revenue", mode="cdn")
TOOLTIPS = [
    ("cast_crew_len", "@cast_crew_len"),
    ("Revenue", "@revenue"),
    ("Original Language", "@original_language"),
    
]

p = figure(tooltips=TOOLTIPS,y_axis_type="log")

p.circle('cast_crew_len', 'revenue',fill_alpha=0.6, line_color=None, source = source)
p.xaxis.axis_label = "cast_crew_len"
p.yaxis.axis_label = "revenue"
show(p)


# In[ ]:


#https://www.kaggle.com/jlove5/avocados-usa-prices

g = sns.catplot(x='bool_homepage',y='log_revenue',data=train, kind='box', aspect=1 )
g.set_axis_labels("Is there a homepage", "Log of Revenue")

# In[ ]:


#https://www.kaggle.com/jlove5/avocados-usa-prices

g = sns.catplot(x='release_month',y='log_revenue',data=train, kind='box', aspect=2 )
g.set_axis_labels("Release month", "Log of Revenue")

# In[ ]:


#https://www.kaggle.com/jlove5/avocados-usa-prices

g = sns.catplot(x='release_day_of_week',y='log_revenue',data=train, kind='box', aspect=2 )
g.set_axis_labels("Release day of week", "Log of Revenue")

# In[ ]:


#https://www.kaggle.com/jlove5/avocados-usa-prices

g = sns.catplot(x='release_week',y='log_revenue',data=train, kind='box', aspect=3 )
g.set_axis_labels("Release week", "Log of Revenue")

# In[ ]:


#https://www.kaggle.com/jlove5/avocados-usa-prices

train['is_genre_drama'] = (train['genre_drama']>0).astype(int)
g = sns.catplot(x='is_genre_drama',y='log_revenue', data=train,kind='box' )
g.set_axis_labels("is_genre_drama", "Log of Revenue")



# In[ ]:


#https://www.kaggle.com/jlove5/avocados-usa-prices

train['is_kw_tg_ow_death'] = (train['kw_tg_ow_death']>0).astype(int)
g = sns.catplot(x='is_kw_tg_ow_death',y='log_revenue', data=train,kind='box' )
g.set_axis_labels("is_kw_tg_ow_death", "Log of Revenue")


# In[ ]:


#https://www.kaggle.com/jlove5/avocados-usa-prices

train['is_genre_thriller'] = (train['genre_thriller']>0).astype(int)
g = sns.catplot(x='is_genre_thriller',y='log_revenue', data=train,kind='box' )
g.set_axis_labels("is_genre_thriller", "Log of Revenue")

# In[ ]:


#https://www.kaggle.com/jlove5/avocados-usa-prices

g = sns.catplot(x='release_year',y='log_revenue',data=train, kind='box', aspect=3 )
g.set_axis_labels("Release year", "Log of Revenue")
g.set_xticklabels(rotation=30)

# In[ ]:


len(features)

# In[ ]:


features = features+olang+train_tagline_keyword_w_cols+train_genres_w_cols+train_belongs_to_collection_w_cols \
+train_production_companies_w_cols + train_crew_w_cols+ train_production_countries_w_cols

# In[ ]:


len(features)

# In[ ]:


test.release_year.min()

# In[ ]:


train[features].info()

# In[ ]:


test[features].info()

# In[ ]:


target_column = 'revenue'
columns_for_prediction=features
X = train[columns_for_prediction].copy()
import sklearn.preprocessing as preprocessing
y_scale = preprocessing.MinMaxScaler()
#y = np.log(train[target_column])
#https://stackoverflow.com/questions/26584971/how-to-not-standarize-target-data-in-scikit-learn-regression
y = y_scale.fit_transform(train[target_column].values.reshape(-1, 1) )

X_unseen = test[columns_for_prediction].copy()
# print(X.budget.max())
# budget_q = X['budget'].quantile(0.8)
# print(budget_q)
# print(X.popularity.max())
# popularity_q = X['popularity'].quantile(0.8)
# print(popularity_q)
# print(X.runtime.max())
# runtime_q = X['runtime'].quantile(0.8)
# print(runtime_q)
# print(y.max())
# y_q = y.quantile(0.8)
# print(y_q)

# #Outliers
# X.loc[X.budget > budget_q, 'budget'] = budget_q
# X.loc[X.popularity > popularity_q, 'popularity'] = popularity_q
# X.loc[X.runtime > runtime_q, 'runtime'] = runtime_q
# y[y>y_q] = y_q


# X_unseen.loc[X_unseen.budget > budget_q, 'budget'] = budget_q
# X_unseen.loc[X_unseen.popularity > popularity_q, 'popularity'] = popularity_q
# X_unseen.loc[X_unseen.runtime > runtime_q, 'runtime'] = runtime_q

scale = preprocessing.StandardScaler()
X = pd.DataFrame(scale.fit_transform(X),columns=columns_for_prediction)
X_unseen = pd.DataFrame(scale.transform(test[columns_for_prediction]),columns=columns_for_prediction)

budget_min = X['budget'].quantile(0.28)
X['budget'] = X['budget'].replace(0,budget_min)

X_unseen['budget'] = X_unseen['budget'].replace(0,budget_min)

# X['budget'] = np.log(X['budget'])
# X_unseen['budget'] = np.log(X_unseen['budget'])
# X['runtime'] = np.log(X['runtime']+1)
# X_unseen['runtime'] = np.log(X_unseen['runtime']+1)
# X['popularity'] = np.log(X['popularity']+1)
# X_unseen['popularity'] = np.log(X_unseen['popularity']+1)



# from sklearn.preprocessing import StandardScaler
# scale = StandardScaler()
# X['budget'] = scale.fit(X[['budget']])
# X_unseen = test[columns_for_prediction].copy()
# X_unseen['budget'] = scale.fit(X_unseen[['budget']])


# X[['budget']] = pd.DataFrame(scale.fit_transform(X[['budget']].values))
# X_unseen[['budget']] = pd.DataFrame(scale.fit_transform(X_unseen[['budget']].values))
#X[['budget']].head()


# print(X.release_year.min())
# print(X_unseen.release_year.min())
# release_year_min = X.release_year.min()
# X['release_year'] = X['release_year'] - release_year_min
# X_unseen['release_year'] = X_unseen['release_year'] - release_year_min

#X['month_into_year'] = np.log(X['month_into_year']+1)
#X_unseen['month_into_year'] = np.log(X_unseen['month_into_year']+1)
#X.budget = np.log(X.budget)
#X_unseen.budget

# In[ ]:


columns_for_prediction

# In[ ]:


#features = ['cast_crew_len','budget','popularity','runtime','release_month','release_day_of_week','genres_len','bool_belongs_to_collection', 'title_len','release_year']
#2.67146
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.1, random_state=2019)
# from sklearn.linear_model import BayesianRidge
# from sklearn.metrics import mean_squared_error

# reg = BayesianRidge(normalize=False,verbose=True).fit(X_train, y_train)
# score = reg.score(X_test, y_test)
# print('Test score %d'%score)
# preds = reg.predict(X_test)
# err = mean_squared_error(y_test, preds) ** 0.5
# print('Test rmse %d'%score)
# reg = BayesianRidge(normalize=False).fit(X, y)
# score = reg.score(X, y)
# print('Train score %d'%score)
# preds = reg.predict(test[columns_for_prediction])

# In[ ]:



from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.1, random_state=2019)
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
params = {'n_estimators': 700, 'max_depth': 4, 'min_samples_split': 2,
          'learning_rate': 0.01, 'loss': 'ls'}

reg = GradientBoostingRegressor(**params).fit(X_train, y_train)
score = reg.score(X_test, y_test)
print('Test score %d'%score)
preds = reg.predict(X_test)
err = mean_squared_error(y_test, preds)
print('Test mse %d'%err)
reg = GradientBoostingRegressor(n_estimators=700).fit(X, y)
score = reg.score(X, y)
print('Train score %d'%score)
preds_first = reg.predict(X_unseen)

# Bayes search to find parameters

# In[ ]:


#https://www.kaggle.com/srserves85/boosting-stacking-and-bayes-searching
# uses baysian optimization to find model parameters
# from skopt import BayesSearchCV
# from skopt.space import Real, Categorical, Integer

# model = GradientBoostingRegressor(
#     loss='ls',
#     learning_rate = 0.0035,
#     max_depth=23,
#     n_estimators=30275,
#     #max_features=9,
#     min_samples_leaf=22,
#     min_samples_split=15,
#     min_weight_fraction_leaf=0.0102470171519909,
#     random_state = 0
# )

# search_params = {
#     "n_estimators": Integer(1650, 4000),
#     'max_depth': Integer(3, 5),
#     'min_samples_split': Integer(15, 30),
#     'min_samples_leaf': Integer(15, 50),
#     'learning_rate': Real(0.009,0.2),
#     'subsample': Real(0.9,1),
#     'max_leaf_nodes': Integer(70, 110),
#     'random_state': Integer(2019,2020),
#     #'min_weight_fraction_leaf': Real(0., .5),
#     #'max_features': Integer(8, 88)
# }

# opt = BayesSearchCV(model, search_params, n_iter=50, n_jobs=8, cv=5,random_state=2019)
# opt.fit(X, y)
# opt_best_params = opt.best_params_
# opt_best_params

# {'learning_rate': 0.009445613868676355,
#  'max_depth': 5,
#  'max_leaf_nodes': 72,
#  'min_samples_leaf': 20,
#  'min_samples_split': 5,
#  'n_estimators': 1621,
#  'random_state': 2020,
#  'subsample': 0.9011747167211098}

# params =  {'max_depth': 7,
#  'min_samples_leaf': 33,
#  'min_samples_split': 2,
#  'min_weight_fraction_leaf': 0.0,
#  'n_estimators': 2921}

# {'learning_rate': 0.005205030202575363,
#  'max_depth': 4,
#  'min_samples_leaf': 2,
#  'min_samples_split': 2,
#  'n_estimators': 1617,
#  'subsample': 0.9}

# <div id="idPrediction">Prediction</div>

# Tf regression

# In[ ]:


#https://www.tensorflow.org/tutorials/keras/basic_regression
#https://www.pyimagesearch.com/2019/01/21/regression-with-keras/
#https://www.kaggle.com/hendraherviawan/regression-with-kerasregressor
#https://www.kaggle.com/yusufsatilmis/house-prices-prediction-with-keras
#https://www.kaggle.com/aharless/keras-nn-with-q4-validation
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def norm(x):
    return (x - train_stats.loc['mean']) / train_stats.loc['std']
train_dataset = X
#train_labels = y.values
train_labels = y
test_dataset = X_unseen
train_stats = train_dataset.describe()
normed_train_data = train_dataset
normed_test_data = test_dataset


# def build_model():
#   model = keras.Sequential([
#     layers.Dense(20, activation='relu', 
#                  kernel_initializer=keras.initializers.TruncatedNormal(mean=0.0, stddev=0.05, seed=None),
#                  input_shape=[len(train_dataset.keys())]),
#     layers.Dropout(.5),   
    
# #     layers.Dense(250, activation='relu', kernel_initializer='normal'),
# #     layers.Dropout(.5),  
# #     layers.Dense(100, activation='relu', kernel_initializer='normal'),
# #     layers.Dropout(.4), 
# #     layers.Dense(50, activation='relu', kernel_initializer='normal'),
# #     layers.Dropout(.4),   
# #     layers.Dense(20, activation='relu', kernel_initializer='normal'),
# #     layers.Dropout(.3),   
#     layers.Dense(1, activation='linear', kernel_initializer='normal'),
#     layers.Dropout(.2), 
#   ])
#https://github.com/ZhouYuxuanYX/Kaggle-Titanic-Multilayer-Perceptron-Solution/blob/master/Solution.py
#https://ritwikgupta.me/files/drclassification.html
def build_model():
    model = keras.Sequential([
    layers.Dense(1550, activation=tf.nn.relu, kernel_initializer='normal', input_shape=[len(train_dataset.keys())]),
    layers.BatchNormalization(),
    layers.GaussianDropout(.7),  
    layers.Dense(700, activation=tf.nn.relu, kernel_initializer='normal'),
    layers.BatchNormalization(),
    layers.GaussianDropout(.5), 
    layers.Dense(700, activation=tf.nn.relu, kernel_initializer='normal'),
    layers.BatchNormalization(),        
    layers.GaussianDropout(.4),   
    layers.Dense(350, activation=tf.nn.relu, kernel_initializer='normal'),
    layers.BatchNormalization(),    
    layers.GaussianDropout(.2),   
    layers.Dense(1, activation='linear', kernel_initializer='normal')
    ])

    #optimizer = tf.keras.optimizers.RMSprop(0.0001)
    optimizer = tf.keras.optimizers.Adam(0.0001, decay=1e-5)
    model.compile(loss='mean_squared_error',
                optimizer=optimizer,
                metrics=['mean_absolute_error', 'mean_squared_error'])
    return model



model = build_model()
model.summary()
EPOCHS = 20

history = model.fit(
  normed_train_data, train_labels,batch_size = 200,
  epochs=EPOCHS, validation_split = 0.25, verbose=1)

def plot_history(history):
  hist = pd.DataFrame(history.history)
  hist['epoch'] = history.epoch
  
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error ')
  plt.plot(hist['epoch'], hist['mean_absolute_error'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mean_absolute_error'],
           label = 'Val Error')
  
  plt.legend()
  
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Square Error ')
  plt.plot(hist['epoch'], hist['mean_squared_error'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mean_squared_error'],
           label = 'Val Error')
  
  plt.legend()
  plt.show()


plot_history(history)

model = build_model()
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
history = model.fit(normed_train_data, train_labels, epochs=EPOCHS,batch_size = 200,
                    validation_split = 0.25, verbose=0, callbacks=[early_stop])
plot_history(history)

pred_using_train = model.predict(normed_train_data).flatten()
preds_estop = model.predict(normed_test_data).flatten()

# In[ ]:



ax = sns.scatterplot(x=train[target_column].values, 
                     y= (y_scale.inverse_transform(pred_using_train.reshape(-1, 1))).reshape(1, -1)[0] )

# In[ ]:


 ax = sns.scatterplot(x=train_labels.flatten(), y=pred_using_train.flatten())

# In[ ]:


import seaborn as sns
sns.distplot((y_scale.inverse_transform(pred_using_train.reshape(-1, 1))).reshape(1, -1)[0] )
#train['revenue'].hist(log=True)

# In[ ]:


# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.1, random_state=2019)
# from sklearn.ensemble import GradientBoostingRegressor
# from sklearn.metrics import mean_squared_error
# # params = {'learning_rate': 0.009445613868676355,
# #  'max_depth': 5,
# #  'max_leaf_nodes': 72,
# #  'min_samples_leaf': 20,
# #  'min_samples_split': 5,
# #  'n_estimators': 1621,
# #  'random_state': 2020,
# #  'subsample': 0.9011747167211098}
# # params = {'n_estimators': 700, 'max_depth': 4, 'min_samples_split': 2,
# #           'learning_rate': 0.01, 'loss': 'ls'}


# params = opt_best_params

# # params = {'learning_rate': 0.009,
# #  'max_depth': 3,
# #  'max_leaf_nodes': 73,
# #  'min_samples_leaf': 15,
# #  'min_samples_split': 13,
# #  'n_estimators': 1600,
# #  'random_state': 2019,
# #  'subsample': 0.9732580382105793}

# # params = {'learning_rate': 0.009,
# #  'max_depth': 3,
# #  'max_leaf_nodes': 110,
# #  'min_samples_leaf': 15,
# #  'min_samples_split': 17,
# #  'n_estimators': 1650,
# #  'random_state': 2019,
# #  'subsample': 0.9}


# # params =  {'max_depth': 7,
# #  'min_samples_leaf': 33,
# #  'min_samples_split': 2,
# #  'min_weight_fraction_leaf': 0.0,
# #  'n_estimators': 2921}
# # params =  {'learning_rate': 0.005205030202575363,
# #  'max_depth': 4,
# #  'min_samples_leaf': 2,
# #  'min_samples_split': 2,
# #  'n_estimators': 1617,
# #  'subsample': 0.9}

# # params = {'learning_rate': 0.01,
# #  'max_depth': 7,
# #  'max_leaf_nodes': 25,
# #  'min_samples_leaf': 50,
# #  'min_samples_split': 15,
# #  'n_estimators': 700,
# #  'subsample': 0.9,
# #  'random_state': 2019}

# reg = GradientBoostingRegressor(**params).fit(X_train, y_train)
# score = reg.score(X_test, y_test)
# print('Test score %d'%score)
# preds = reg.predict(X_test)
# err = mean_squared_error(y_test, preds)
# print('Test mse %d'%err)
# #reg = GradientBoostingRegressor(n_estimators=700).fit(X, y)
# score = reg.score(X, y)
# print('Train score %d'%score)
# preds = reg.predict(X_unseen)
# reg.get_params()

# In[ ]:


# feature_importance = reg.feature_importances_
# # make importances relative to max importance
# feature_importance = 100.0 * (feature_importance / feature_importance.max())
# sorted_idx = np.argsort(feature_importance)
# [features[i] for i in sorted_idx][::-1]

# In[ ]:


# from sklearn.linear_model import LassoLarsCV
# reg = LassoLarsCV(cv=10).fit(X, y)
# print(reg.score(X, y)) 
# print(reg.alpha_)
# preds = reg.predict(test[columns_for_prediction])

# In[ ]:


#median_revenue = y.median()

# In[ ]:


preds[:10]

# In[ ]:


#https://www.kaggle.com/tejasrinivas/xgb-baseline-comments-classification
# import xgboost as xgb

# #data_with_imputed_values_wo_Fare = data_with_imputed_values.drop('Fare', axis=1)
# #X_t, X_v, y_t, y_v = train_test_split(data_with_imputed_values,y_train, stratify=y_train, test_size=0.2, random_state=2019)
# #https://stackoverflow.com/questions/48645846/pythons-xgoost-valueerrorfeature-names-may-not-contain-or
# import re
# regex = re.compile(r"\[|\]|<", re.IGNORECASE)
# X_copy = X.copy()
# X_copy.columns = [regex.sub("_", col) if any(x in str(col) for x in set(('[', ']', '<'))) else col for col in X_copy.columns.values]
# X_unseen_copy = X_unseen.copy()
# X_unseen_copy.columns = [regex.sub("_", col) if any(x in str(col) for x in set(('[', ']', '<'))) else col for col in X_unseen_copy.columns.values]

# X_t, X_v, y_t, y_v = train_test_split(X_copy, y, test_size=.2, random_state=2019)
# #X_t, X_v, y_t, y_v = train_test_split(data_with_imputed_values_wo_Fare,y_train, test_size=0.2, random_state=2019)


# def runXGB(X_t, X_v, y_t, y_v, feature_names=None, seed_val=2017, num_rounds=200):
#     param = {}
#     param['objective'] = 'reg:linear'
#     param['eta'] = 0.2
#     param['max_depth'] = 7
#     param['silent'] = 1
#     param['eval_metric'] = 'logloss'
#     param['min_child_weight'] = 1 #3
#     param['subsample'] = 0.9
#    # param['colsample_bytree'] = 0.5
#     param['seed'] = seed_val
#    # param['max_delta_step'] = 8
# #     param['objective'] = 'binary:logistic'
# #     param['eta'] = 0.1
# #     param['max_depth'] = 6
# #     param['silent'] = 1
# #     param['eval_metric'] = 'auc'
# #     param['min_child_weight'] = 1
# #     param['subsample'] = 0.5
# #     param['colsample_bytree'] = 0.5
# #     param['seed'] = seed_val
    
    
#     num_rounds = num_rounds

#     plst = list(param.items())
#     xgtrain = xgb.DMatrix(X_t, label=y_t)


#     xgtest = xgb.DMatrix(X_v, label=y_v)
#     watchlist = [ (xgtrain,'train'), (xgtest, 'test') ]
#     model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=20)
#     #model = xgb.train(plst, xgtrain, num_rounds, watchlist)
#     return model    

# model = runXGB(X_t, X_v, y_t, y_v)
# preds = model.predict(xgb.DMatrix(X_unseen_copy))
# preds[:5]

# In[ ]:



#preds[preds < 0] = median_revenue
#preds=np.exp(preds)
median_revenue = train[target_column].median()
#preds = y_scale.inverse_transform(preds)
preds = preds_estop
preds = y_scale.inverse_transform(preds.reshape(-1, 1))
preds[preds < 0] = median_revenue
#preds=(preds)**2
subm['revenue'] = preds
#median_revenue = y.median()
# preds_first[preds_first < 0] = median_revenue
# preds_first=np.exp(preds_first)
# subm['revenue'] = preds_first
subm.to_csv('submission.csv', index=False)
print(subm.head())

# In[ ]:


len(train)

#  <div id="idAnalysis_of_Prediction">Analysis of Prediction</div>

# In[ ]:


import seaborn as sns
#sns.distplot(train['revenue'] )
train['revenue'].hist(log=True)

# In[ ]:


len(subm)

# In[ ]:


#sns.distplot(subm['revenue'] )
subm['revenue'].hist(log=True)

# In[ ]:


ax = sns.scatterplot(x="popularity", y="revenue",
                     hue="release_year", 
                     data=train)

# In[ ]:


ax = sns.scatterplot(x=test.popularity, y=subm.revenue,
                     hue=test.release_year)

# In[ ]:


ax = sns.scatterplot(x="budget", y="revenue",
                     hue="release_year", 
                     data=train)

# In[ ]:


ax = sns.scatterplot(x=test.budget, y=subm.revenue,
                     hue=test.release_year)

# In[ ]:


train[ ['release_date', 'revenue']].set_index('release_date').resample('A').mean()[:'2019'].plot(style='--')

# In[ ]:


test['revenue'] = subm['revenue']

# In[ ]:


test[ ['release_date', 'revenue']].set_index('release_date').resample('A').mean()[:'2019'].plot(style='--')
