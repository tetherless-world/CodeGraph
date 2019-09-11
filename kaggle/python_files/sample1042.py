#!/usr/bin/env python
# coding: utf-8

# # Within Top 10% with Simple Regression Model.

# # Step By Step Procedure To Predict House Price

# # Importing packages
# We have **numpy** and **pandas** to work with numbers and data, and we have **seaborn** and **matplotlib** to visualize data. We would also like to filter out unnecessary warnings. **Scipy** for normalization and skewing of data.

# In[ ]:


#import some necessary librairies

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt  # Matlab-style plotting
import seaborn as sns
color = sns.color_palette()
sns.set_style('darkgrid')
import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn #ignore annoying warning (from sklearn and seaborn)

from scipy import stats
from scipy.stats import norm, skew #for some statistics

# # Loading and Inspecting data
# With various Pandas functions, we load our training and test data set as well as inspect it to get an idea of the data we're working with. This is a large dataset we will be working on.
# 

# In[ ]:


#Now let's import and put the train and test datasets in  pandas dataframe

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

train.describe()

# In[ ]:


print ("Size of train data : {}" .format(train.shape))

print ("Size of test data : {}" .format(test.shape))

# > That is a very large data set! We are going to have to do a lot of work to clean it up
# 
# **Drop the Id column because we dont need it currently.**

# In[ ]:


#Save the 'Id' column
train_ID = train['Id']
test_ID = test['Id']

#Now drop the  'Id' colum since it's unnecessary for  the prediction process.
train.drop("Id", axis = 1, inplace = True)
test.drop("Id", axis = 1, inplace = True)

# In[ ]:


print ("Size of train data after dropping Id: {}" .format(train.shape))
print ("Size of test data after dropping Id: {}" .format(test.shape))

# ## Dealing with outliers
# 
# Outlinear in the GrLivArea is recommended by the author of the data  to remove it. The author says in documentation “I would recommend removing any houses with more than 4000 square feet from the data set (which eliminates these five unusual observations) before assigning it to students.”
# 

# In[ ]:


fig, ax = plt.subplots()
ax.scatter(x = train['GrLivArea'], y = train['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)
plt.show()

# We can see that there are outlinear with low SalePrice and high GrLivArea. This looks odd.
# We need to remove it.

# In[ ]:


#Deleting outliers
train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)

# ## Correlation Analysis
# 
# Let see the most correlated features.

# In[ ]:


# most correlated features
corrmat = train.corr()
top_corr_features = corrmat.index[abs(corrmat["SalePrice"])>0.5]
plt.figure(figsize=(10,10))
g = sns.heatmap(train[top_corr_features].corr(),annot=True,cmap="RdYlGn")

# - From this we can tell which features **(OverallQual, GrLivArea and TotalBsmtSF )** are highly positively correlated with the SalePrice. 
# - **GarageCars and GarageArea ** also seems correlated with other, Since the no. of car that will fit into the garage will depend on GarageArea. 

# In[ ]:


sns.barplot(train.OverallQual,train.SalePrice)

# **Scatter plots between 'SalePrice' and correlated variables**

# In[ ]:


sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(train[cols], size = 2.5)
plt.show();

# One of the figures we may find interesting is the one between ** 'TotalBsmtSF' and 'GrLiveArea'. **
# 
# We can see the dots drawing a linear line, which almost acts like a border. It totally makes sense that the majority of the dots stay below that line. Basement areas can be equal to the above ground living area, but it is not expected a basement area bigger than the above ground living area

# In[ ]:


sns.scatterplot(train.GrLivArea,train.TotalBsmtSF)

# ## Target Variable Transform
# Different features in the data set may have values in different ranges. For example, in this data set, the range of SalePrice feature may lie from thousands to lakhs but the range of values of YearBlt feature will be in thousands. That means a column is more weighted compared to other.
# 
# **Lets check the skewness of data**
# ![Skew](https://cdn-images-1.medium.com/max/800/1*hxVvqttoCSkUT2_R1zA0Tg.gif)

# In[ ]:


def check_skewness(col):
    sns.distplot(train[col] , fit=norm);
    fig = plt.figure()
    res = stats.probplot(train[col], plot=plt)
    # Get the fitted parameters used by the function
    (mu, sigma) = norm.fit(train[col])
    print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))
    
check_skewness('SalePrice')

# **This distribution is positively skewed.** Notice that the black curve is more deviated towards the right. If you encounter that your predictive (response) variable is skewed, it is **recommended to fix the skewness** to make good decisions by the model.
# 
# ## Okay, So how do I fix the skewness?
# The best way to fix it is to perform a **log transform** of the same data, with the intent to reduce the skewness.

# In[ ]:


#We use the numpy fuction log1p which  applies log(1+x) to all elements of the column
train["SalePrice"] = np.log1p(train["SalePrice"])

check_skewness('SalePrice')

# After taking logarithm of the same data the curve seems to be normally distributed, although not perfectly normal, this is sufficient to fix the issues from a skewed dataset as we saw before.
# 
# **Important : If you log transform the response variable, it is required to also log transform feature variables that are skewed.**

# # Feature Engineering

# Here is the [Documentation](http://ww2.amstat.org/publications/jse/v19n3/Decock/DataDocumentation.txt) you can refer , to know more about the dataset.
# 
# **Concatenate both train and test values.**

# In[ ]:


ntrain = train.shape[0]
ntest = test.shape[0]
y_train = train.SalePrice.values
all_data = pd.concat((train, test)).reset_index(drop=True)
all_data.drop(['SalePrice'], axis=1, inplace=True)
print("all_data size is : {}".format(all_data.shape))

# # Missing Data

# In[ ]:


all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:30]
missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})

# In[ ]:


f, ax = plt.subplots(figsize=(15, 12))
plt.xticks(rotation='90')
sns.barplot(x=all_data_na.index, y=all_data_na)
plt.xlabel('Features', fontsize=15)
plt.ylabel('Percent of missing values', fontsize=15)
plt.title('Percent missing data by feature', fontsize=15)

# In[ ]:


all_data.PoolQC.loc[all_data.PoolQC.notnull()]

# **GarageType,  GarageFinish, GarageQual,  GarageCond, GarageYrBlt,  GarageArea,  GarageCars  these all features have same percentage of null values.**

# # Handle Missing Data

# Since PoolQC has the highest null values according to the data documentation says **null values means 'No Pool.**
# Since majority of houses has no pool.
# So we will replace those null values with 'None'.

# In[ ]:


all_data["PoolQC"] = all_data["PoolQC"].fillna("None")

# * **MiscFeature** : Data documentation says NA means "no misc feature"

# In[ ]:


all_data["MiscFeature"] = all_data["MiscFeature"].fillna("None")

# * **Alley** : data description says NA means "no alley access"
# 

# In[ ]:


all_data["Alley"] = all_data["Alley"].fillna("None")

# * **Fence** : data description says NA means "no fence"
# 

# In[ ]:


all_data["Fence"] = all_data["Fence"].fillna("None")

# * **FireplaceQu** : data description says NA means "no fireplace"

# In[ ]:


all_data["FireplaceQu"] = all_data["FireplaceQu"].fillna("None")

# * **LotFrontage** : Since the area of each street connected to the house property most likely have a similar area to other houses in its neighborhood , we can fill in missing values by the median LotFrontage of the neighborhood.

# In[ ]:


# Grouping by Neighborhood and Check the LotFrontage. Most of the grouping has similar areas
grouped_df = all_data.groupby('Neighborhood')['LotFrontage']

for key, item in grouped_df:
    print(key,"\n")
    print(grouped_df.get_group(key))
    break

# In[ ]:


#Group by neighborhood and fill in missing value by the median LotFrontage of all the neighborhood
all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(
    lambda x: x.fillna(x.median()))

# * **GarageType, GarageFinish, GarageQual and GarageCond** : Replacing missing data with None as per documentation. 

# In[ ]:


for col in ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']:
    all_data[col] = all_data[col].fillna('None')

# In[ ]:


abc = ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond','GarageYrBlt', 'GarageArea', 'GarageCars']
all_data.groupby('GarageType')[abc].count()

# * **GarageYrBlt, GarageArea and GarageCars** : Replacing missing data with 0 (Since No garage = no cars in such garage.)

# In[ ]:


for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
    all_data[col] = all_data[col].fillna(0)

# * **BsmtFinSF1, BsmtFinSF2, BsmtUnfSF, TotalBsmtSF, BsmtFullBath and BsmtHalfBath** : missing values are likely zero for having no basement

# In[ ]:


for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
    all_data[col] = all_data[col].fillna(0)

# * **BsmtQual, BsmtCond, BsmtExposure, BsmtFinType1 and BsmtFinType2** : For all these categorical basement-related features, NaN means that there is no basement.

# In[ ]:


for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    all_data[col] = all_data[col].fillna('None')

# * **MasVnrArea and MasVnrType** : NA most likely means no masonry veneer for these houses. We can fill 0 for the area and None for the type.

# In[ ]:


all_data["MasVnrType"] = all_data["MasVnrType"].fillna("None")
all_data["MasVnrArea"] = all_data["MasVnrArea"].fillna(0)

# * **MSZoning (The general zoning classification)** : 'RL' is by far the most common value. So we can fill in missing values with 'RL'

# In[ ]:


all_data['MSZoning'].value_counts()

# In[ ]:


all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])

# * **Utilities** : Since this is a categorical data and most of the data are of same category, Its not gonna effect on model. So we choose to drop it.

# In[ ]:


all_data['Utilities'].value_counts()

# In[ ]:


all_data = all_data.drop(['Utilities'], axis=1)

# * **Functional** : data description says NA means typical

# In[ ]:


all_data["Functional"] = all_data["Functional"].fillna("Typ")

# * **Electrical,KitchenQual, Exterior1st, Exterior2nd, SaleType** : Since this all are categorical values so its better to replace nan values with the most used keyword.

# In[ ]:


mode_col = ['Electrical','KitchenQual', 'Exterior1st', 'Exterior2nd', 'SaleType']
for col in mode_col:
    all_data[col] = all_data[col].fillna(all_data[col].mode()[0])

# * **MSSubClass** : Na most likely means No building class. We can replace missing values with None
# 

# In[ ]:


all_data['MSSubClass'] = all_data['MSSubClass'].fillna("None")

# ## Lets check for any missing values

# In[ ]:


#Check remaining missing values if any 
all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)
missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})
missing_data.head()

# **Now there any many features that are numerical but categorical.**

# In[ ]:


all_data['OverallCond'].value_counts()

# **Converting some numerical variables that are really categorical type.**
# 
# As you can see the category range from 1 to 9 which are numerical (**not ordinal type**). Since its categorical we need to change it to String type.
# 
# If we do not convert these to categorical, some model may get affect by this as model will compare the value 1<5<10 . We dont need that to happen with our model.

# In[ ]:


#MSSubClass=The building class
all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)


#Changing OverallCond into a categorical variable
all_data['OverallCond'] = all_data['OverallCond'].astype(str)


#Year and month sold are transformed into categorical features.
all_data['YrSold'] = all_data['YrSold'].astype(str)
all_data['MoSold'] = all_data['MoSold'].astype(str)

# ## Label Encoding 
# As you might know by now, we can’t have text in our data if we’re going to run any kind of model on it. So before we can run a model, we need to make this data ready for the model.
# 
# And to convert this kind of categorical text data into model-understandable numerical data, we use the Label Encoder class.
# 
# Suppose, we have a feature State which has 3 category i.e India , France, China . So, Label Encoder will categorize them as 0, 1, 2.

# In[ ]:


from sklearn.preprocessing import LabelEncoder
cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 
        'YrSold', 'MoSold')
# process columns, apply LabelEncoder to categorical features
for c in cols:
    lbl = LabelEncoder() 
    lbl.fit(list(all_data[c].values)) 
    all_data[c] = lbl.transform(list(all_data[c].values))

# shape        
print('Shape all_data: {}'.format(all_data.shape))

# Since area related features are very important to determine house prices, we add one more feature which is the total area of basement, first and second floor areas of each house

# In[ ]:


# Adding total sqfootage feature 
all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']

# **Lets see the highly skewed features we have**

# In[ ]:


numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index

# Check the skew of all numerical features
skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
print("\nSkew in numerical features: \n")
skewness = pd.DataFrame({'Skew' :skewed_feats})
skewness.head(15)

# ## Box Cox Transformation of (highly) skewed features
# 
# When you are dealing with real-world data, you are going to deal with features that are heavily skewed. Transformation technique is useful to **stabilize variance**, make the data **more normal distribution-like**, improve the validity of measures of association.
# 
# The problem with the Box-Cox Transformation is **estimating lambda**. This value will depend on the existing data, and should be considered when performing cross validation on out of sample datasets.

# In[ ]:


skewness = skewness[abs(skewness) > 0.75]
print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))

from scipy.special import boxcox1p
skewed_features = skewness.index
lam = 0.15
for feat in skewed_features:
    #all_data[feat] += 1
    all_data[feat] = boxcox1p(all_data[feat], lam)

# **Getting dummy categorical features**

# In[ ]:


all_data = pd.get_dummies(all_data)
all_data.shape

# Creating train and test data.

# In[ ]:


train = all_data[:ntrain]
test = all_data[ntrain:]
train.shape

# ## Lets apply Modelling
# 
# 1. Importing Libraries
# 
# 2. We will use models
#  - Lasso
#  - Ridge
#  - ElasticNet
#  - Gradient Boosting
#  
# 3. Find the Cross Validation Score.
# 4. Calculate the mean of all model's prediction.
# 5. Submit the CSV file.
#  

# In[ ]:


from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb

# ## Cross Validation
# It's simple way to calculate error for evaluation. 
# 
# **KFold( )** splits the train/test data into k consecutive folds, we also have made shuffle attrib to True.
# 
# **cross_val_score ( )** evaluate a score by cross-validation.

# In[ ]:


#Validation function
n_folds = 5

def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train.values)
    rmse= np.sqrt(-cross_val_score(model, train.values, y_train, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)

# # Modelling
# Since in this dataset we have a large set of features. So to make our model avoid Overfitting and noisy we will use Regularization.
# These model have Regularization parameter.
# 
# Regularization will reduce the magnitude of the coefficients.

# ## Ridge Regression
# - It shrinks the parameters, therefore it is mostly used to prevent multicollinearity.
# - It reduces the model complexity by coefficient shrinkage.
# - It uses L2 regularization technique.

# In[ ]:


KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)
score = rmsle_cv(KRR)
print("Kernel Ridge score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

# ## Lasso Regression
# LASSO (Least Absolute Shrinkage Selector Operator), is quite similar to ridge.
# 
# In case of lasso, even at smaller alpha’s, our coefficients are reducing to absolute zeroes.
#  Therefore, lasso selects the only some feature while reduces the coefficients of others to zero. This property is known as feature selection and which is absent in case of ridge.
#  
# - Lasso uses L1 regularization technique.
# - Lasso is generally used when we have more number of features, because it automatically does feature selection.
# 

# In[ ]:


lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))
score = rmsle_cv(lasso)
print("Lasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

# ## Elastic Net Regression
# 
# Elastic net is basically a combination of both L1 and L2 regularization. So if you know elastic net, you can implement both Ridge and Lasso by tuning the parameters.

# In[ ]:


ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))
score = rmsle_cv(ENet)
print("ElasticNet score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

# ## Gradient Boosting Regression
# Refer [here](https://medium.com/mlreview/gradient-boosting-from-scratch-1e317ae4587d)

# In[ ]:


GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state =5)
score = rmsle_cv(GBoost)
print("Gradient Boosting score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

# **Fit the training dataset on every model**

# In[ ]:


LassoMd = lasso.fit(train.values,y_train)
ENetMd = ENet.fit(train.values,y_train)
KRRMd = KRR.fit(train.values,y_train)
GBoostMd = GBoost.fit(train.values,y_train)

# ## Mean of all model's prediction.
# np.expm1 ( ) is used to calculate exp(x) - 1 for all elements in the array. 

# In[ ]:


finalMd = (np.expm1(LassoMd.predict(test.values)) + np.expm1(ENetMd.predict(test.values)) + np.expm1(KRRMd.predict(test.values)) + np.expm1(GBoostMd.predict(test.values)) ) / 4
finalMd

# ## Submission

# In[ ]:


sub = pd.DataFrame()
sub['Id'] = test_ID
sub['SalePrice'] = finalMd
sub.to_csv('submission.csv',index=False)

# **If you found this notebook helpful or you just liked it , some upvotes would be very much appreciated.**
# 
# **I'll be glad to hear suggestions on improving my models**
