#!/usr/bin/env python
# coding: utf-8

# In this notebook, I have tried to depict the following
# - Little Bit TS Theory from Wiki , Different Blogposts and Online Books
# - EDA
# - Seasonality Demonstration
# - Moving Average, Exponential Average and Smoothen
# - Few Pivot Plots
# - Keras Embedding(Re-added for Testing)
# - Prophet (With and Without Transformations)
# - ARIMA
# - Validating The Forecast via Plots and Different Metrics
# - Modelling
# - **LSTM Modelling( Next Update and Probably the last by me...)**
# 
# Here's the collection of Resources (including all the publicly shared kernels)
# - https://www.kaggle.com/c/demand-forecasting-kernels-only/discussion/63568
# 
# And in particular this discussion (Thanks a Lot, I learnt a lot from it locally)
# - https://www.kaggle.com/c/demand-forecasting-kernels-only/discussion/62592

# We begin with a **simple definition of time series**:
# 
# - Time series is a series of data points indexed (or listed or graphed) in time order. Therefore, the data is organized by relatively deterministic timestamps, and may, compared to random sample data, contain additional information that we can extract.

# ## Necessary Imports

# In[ ]:




# In[ ]:


import time
import lightgbm as lgb
import xgboost as xgb
import seaborn as sns

from fastai.imports import *
from fastai.structured import *
from fbprophet import Prophet

def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn

from sklearn.metrics import r2_score, median_absolute_error, mean_absolute_error
from sklearn.metrics import median_absolute_error, mean_squared_error, mean_squared_log_error
from sklearn.model_selection import KFold
from scipy import stats
from plotly.offline import init_notebook_mode, iplot
from plotly import graph_objs as go

import statsmodels.api as sm
# Initialize plotly
init_notebook_mode(connected=True)
def mean_absolute_percentage_error(y_true, y_pred): 
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

pd.option_context("display.max_rows", 1000);
pd.option_context("display.max_columns", 1000);

# In[ ]:


os.getcwd()

# In[ ]:


PATH = '../input/'

# In[ ]:


print(os.listdir(PATH))

# In[ ]:


df_raw = pd.read_csv(f'{PATH}/train.csv', low_memory=False, parse_dates=['date'], index_col=['date'])
df_test = pd.read_csv(f'{PATH}/test.csv', low_memory=False, parse_dates=['date'], index_col=['date'])
subs = pd.read_csv(f'{PATH}/sample_submission.csv')

# In[ ]:


df_raw.head()

# In[ ]:


print("Train and Test shape are {} and {} respectively".format(df_raw.shape, df_test.shape))

# In[ ]:


df_test.head()

# In[ ]:


#### Seasonality Check
# preparation: input should be float type
df_raw['sales'] = df_raw['sales'] * 1.0

# store types
sales_a = df_raw[df_raw.store == 2]['sales'].sort_index(ascending = True)
sales_b = df_raw[df_raw.store == 3]['sales'].sort_index(ascending = True) # solve the reverse order
sales_c = df_raw[df_raw.store == 1]['sales'].sort_index(ascending = True)
sales_d = df_raw[df_raw.store == 4]['sales'].sort_index(ascending = True)

f, (ax1, ax2, ax3, ax4) = plt.subplots(4, figsize = (12, 13))
c = '#386B7F'

# store types
sales_a.resample('W').sum().plot(color = c, ax = ax1)
sales_b.resample('W').sum().plot(color = c, ax = ax2)
sales_c.resample('W').sum().plot(color = c, ax = ax3)
sales_d.resample('W').sum().plot(color = c, ax = ax4)

#All Stores have same trend... Weird Seems like the dataset is A Synthetic One..;

# In[ ]:


f, (ax1, ax2, ax3, ax4) = plt.subplots(4, figsize = (12, 13))

# Yearly
decomposition_a = sm.tsa.seasonal_decompose(sales_a, model = 'additive', freq = 365)
decomposition_a.trend.plot(color = c, ax = ax1)

decomposition_b = sm.tsa.seasonal_decompose(sales_b, model = 'additive', freq = 365)
decomposition_b.trend.plot(color = c, ax = ax2)

decomposition_c = sm.tsa.seasonal_decompose(sales_c, model = 'additive', freq = 365)
decomposition_c.trend.plot(color = c, ax = ax3)

decomposition_d = sm.tsa.seasonal_decompose(sales_d, model = 'additive', freq = 365)
decomposition_d.trend.plot(color = c, ax = ax4)

# In[ ]:


date_sales = df_raw.drop(['store','item'], axis=1).copy() #it's a temporary DataFrame.. Original is Still intact..

# In[ ]:


date_sales.get_ftype_counts()

# In[ ]:


y = date_sales['sales'].resample('MS').mean() 
y['2017':] #sneak peak

# In[ ]:


y.plot(figsize=(15, 6),);
#The time-series has seasonality pattern, such as sales are always low at the beginning of the year and high at the middle(festive season maybe) of the year
# and again low at the end of the year...
#There is always an upward trend within any single year with a couple of low months in the mid of the year...

# In[ ]:


#We can also visualize our data using a method called time-series decomposition that allows us to decompose our time series into three distinct components: 
#trend, seasonality, and noise.
decomposition = sm.tsa.seasonal_decompose(y, model='additive')
decomposition.plot();
#The plot clearly shows that the sales is unstable, along with its obvious seasonality.;

# In[ ]:


#We can also visualize our data using a method called time-series decomposition that allows us to decompose our time series into three distinct components: 
#trend, seasonality, and noise.
decomposition = sm.tsa.seasonal_decompose(y, model='multiplicative')
decomposition.plot();
#The plot above clearly shows that the sales is unstable, along with its obvious seasonality.;

# ## Moving Average

# Let's start with a naive hypothesis: "tomorrow will be the same as today". However, instead of a model like $\hat{y}_{t} = y_{t-1}$ (which is actually a great baseline for any time series prediction problems and sometimes is impossible to beat), we will assume that the future value of our variable depends on the average of its $k$ previous values. Therefore, we will use the moving average.
# 
# $\hat{y}_{t} = \frac{1}{k} \displaystyle\sum^{k}_{n=1} y_{t-n}$

# In[ ]:


def moving_average(series, n):
    """
        Calculate average of last n observations
    """
    return np.average(series[-n:])

moving_average(date_sales, 24) # prediction for the last observed day (past 24 hours)

# In[ ]:


def plotMovingAverage(series, window, plot_intervals=False, scale=2, plot_anomalies=False):

    """
        series - dataframe with timeseries
        window - rolling window size 
        plot_intervals - show confidence intervals
        plot_anomalies - show anomalies 

    """
    rolling_mean = series.rolling(window=window).mean()

    plt.figure(figsize=(15,5))
    plt.title("Moving average\n window size = {}".format(window))
    plt.plot(rolling_mean, color='Black', label="Rolling mean trend", alpha=0.5)

    # Plot confidence intervals for smoothed values
    if plot_intervals:
        mae = mean_absolute_error(series[window:], rolling_mean[window:])
        deviation = np.std(series[window:] - rolling_mean[window:])
        lower_bond = rolling_mean - (mae + scale * deviation)
        upper_bond = rolling_mean + (mae + scale * deviation)
        plt.plot(upper_bond, color='Black', label="Upper Bond / Lower Bond", alpha=.3)
        plt.plot(lower_bond, color='Black', alpha=.3)
        
        # Having the intervals, find abnormal values
        if plot_anomalies:
            anomalies = pd.DataFrame(index=series.index, columns=series.columns)
            anomalies[series<lower_bond] = series[series<lower_bond]
            anomalies[series>upper_bond] = series[series>upper_bond]
            plt.plot(anomalies, "ro", markersize=10)
        
    plt.plot(series[window:],color='Red', label="Actual values", alpha=.3)
    plt.legend(loc="upper left")
    plt.grid(True)

# ## Exponential smoothing
# 
# Now, let's see what happens if, instead of weighting the last $k$ values of the time series, we start weighting all available observations while exponentially decreasing the weights as we move further back in time. There exists a formula for exponential smoothing that will help us with this:
# 
# $$\hat{y}_{t} = \alpha \cdot y_t + (1-\alpha) \cdot \hat y_{t-1} $$
# 
# Here the model value is a weighted average between the current true value and the previous model values. The $\alpha$ weight is called a smoothing factor. It defines how quickly we will "forget" the last available true observation. The smaller $\alpha$ is, the more influence the previous observations have and the smoother the series is.
# 
# Exponentiality is hidden in the recursiveness of the function -- we multiply by $(1-\alpha)$ each time, which already contains a multiplication by $(1-\alpha)$ of previous model values.

# In[ ]:


def exponential_smoothing(series, alpha):
    """
        series - dataset with timestamps
        alpha - float [0.0, 1.0], smoothing parameter
    """
    result = [series[0]] # first value is same as series
    for n in range(1, len(series)):
        result.append(alpha * series[n] + (1 - alpha) * result[n-1])
    return result

# In[ ]:


def plotExponentialSmoothing(series, alphas):
    """
        Plots exponential smoothing with different alphas
        
        series - dataset with timestamps
        alphas - list of floats, smoothing parameters
        
    """
    with plt.style.context('seaborn-white'):    
        plt.figure(figsize=(15, 7))
        for alpha in alphas:
            plt.plot(exponential_smoothing(series, alpha), label="Alpha {}".format(alpha))
        plt.plot(series.values, "c", label = "Actual", alpha = 0.4)
        plt.legend(loc="best")
        plt.axis('tight')
        plt.title("Exponential Smoothing")
        plt.grid(True);

# In[ ]:


plotExponentialSmoothing(date_sales.sales[:30000], [0.3, 0.05])

# ## Double exponential Smoothing
# 
# Up to now, the methods that we've have seen are for a single future point prediction (with some nice smoothing). That is cool, but it is also not enough. Let's extend exponential smoothing so that we can predict two future points (of course, we will also include more smoothing).
# 
# Series decomposition will help us -- we obtain two components: intercept (i.e. level) $\ell$ and slope (i.e. trend) $b$. We have learnt to predict intercept (or expected series value) with our previous methods; now, we will apply the same exponential smoothing to the trend by assuming that the future direction of the time series changes depends on the previous weighted changes. As a result, we get the following set of functions:
# 
# $$\ell_x = \alpha y_x + (1-\alpha)(\ell_{x-1} + b_{x-1})$$
# 
# $$b_x = \beta(\ell_x - \ell_{x-1}) + (1-\beta)b_{x-1}$$
# 
# $$\hat{y}_{x+1} = \ell_x + b_x$$
# 
# - The first one describes the intercept, which, as before, depends on the current value of the series.
# - The second term is now split into previous values of the level and of the trend. 
#     -  The second function describes the trend, which depends on the level changes at the current step and on the previous value of the trend. In this case, the $\beta$ coefficient is a weight for exponential smoothing. 
# - The final prediction is the sum of the model values of the intercept and trend.

# In[ ]:


def double_exponential_smoothing(series, alpha, beta):
    """
        series - dataset with timeseries
        alpha - float [0.0, 1.0], smoothing parameter for level
        beta - float [0.0, 1.0], smoothing parameter for trend
    """
    # first value is same as series
    result = [series[0]]
    for n in range(1, len(series)+1):
        if n == 1:
            level, trend = series[0], series[1] - series[0]
        if n >= len(series): # forecasting
            value = result[-1]
        else:
            value = series[n]
        last_level, level = level, alpha*value + (1-alpha)*(level+trend)
        trend = beta*(level-last_level) + (1-beta)*trend
        result.append(level+trend)
    return result

def plotDoubleExponentialSmoothing(series, alphas, betas):
    """
        Plots double exponential smoothing with different alphas and betas
        
        series - dataset with timestamps
        alphas - list of floats, smoothing parameters for level
        betas - list of floats, smoothing parameters for trend
    """
    
    with plt.style.context('seaborn-white'):    
        plt.figure(figsize=(20, 8))
        for alpha in alphas:
            for beta in betas:
                plt.plot(double_exponential_smoothing(series, alpha, beta), label="Alpha {}, beta {}".format(alpha, beta))
        plt.plot(series.values, label = "Actual", alpha = 0.1)
        plt.legend(loc="best")
        plt.axis('tight')
        plt.title("Double Exponential Smoothing")
        plt.grid(True)

# In[ ]:


plotDoubleExponentialSmoothing(date_sales.sales[:30000], alphas=[0.9, 0.02], betas=[0.9, 0.02])

# In[ ]:


ts_diff = date_sales - date_sales.shift(7)
plt.figure(figsize=(22,10))
plt.plot(ts_diff[:20000])
plt.title("Differencing method") 
plt.xlabel("Date")
plt.ylabel("Differencing Sales");

# In[ ]:


df_raw = df_raw.reset_index()
df_test = df_test.reset_index()

# In[ ]:


import re
def add_datepart(df, fldname, drop=True):

    """
    Parameters:
    -----------
    df: A pandas data frame. df gain several new columns.
    fldname: A string that is the name of the date column you wish to expand.
        If it is not a datetime64 series, it will be converted to one with pd.to_datetime.
    drop: If true then the original date column will be removed.
    """
    
    fld = df[fldname]
    fld_dtype = fld.dtype
    if isinstance(fld_dtype, pd.core.dtypes.dtypes.DatetimeTZDtype):
        fld_dtype = np.datetime64

    if not np.issubdtype(fld_dtype, np.datetime64):
        df[fldname] = fld = pd.to_datetime(fld, infer_datetime_format=True)
        
    targ_pre = re.sub('[Dd]ate$', '', fldname)
    
    attr = ['Year', 'Month', 'Week', 'Day', 'Dayofweek', 'Dayofyear','weekofyear',
            'Is_month_end', 'Is_month_start', 'Is_quarter_end', 'Is_quarter_start', 'Is_year_end', 'Is_year_start']
    
    for n in attr: 
        df[targ_pre + n] = getattr(fld.dt, n.lower())
        
    if drop: 
        df.drop(fldname, axis=1, inplace=True)

add_datepart(df_raw,'date',False)
add_datepart(df_test,'date',False)

# ##  Few Pivoted Plots

# In[ ]:


pivoted = pd.pivot_table(df_raw, values='sales', columns='Year', index='Month')
pivoted.plot(figsize=(12,12));

# In[ ]:


pivoted = pd.pivot_table(df_raw, values='sales' , columns='Year', index='Week')
pivoted.plot(figsize=(12,12));

# In[ ]:


pivoted = pd.pivot_table(df_raw, values='sales' , columns='Month', index='Day')
pivoted.plot(figsize=(12,12));

# In[ ]:


temp_1 = df_raw.groupby(['Year','Month','item'])['sales'].mean().reset_index()
plt.figure(figsize=(12,8))
sns.swarmplot('item', 'sales', data=temp_1, hue = 'Month');
# Place legend to the right
plt.legend(bbox_to_anchor=(1, 1), loc=2);

# In[ ]:


#In case the above plot is clutterd(which it is), try this, (Will create a grid for Year vs Month)
#sns.factorplot('item', 'sales', data=temp_1, hue = 'Month', col='Year',row='Month', kind='swarm', size = 5);

# In[ ]:


temp_1 = df_raw.groupby(['Year','Month'])['sales'].mean().reset_index()
plt.figure(figsize=(12,8));
sns.lmplot('Month','sales',data = temp_1, hue='Year', fit_reg= False);

# In[ ]:


temp_1 = df_raw.groupby(['Year'])['sales'].mean().reset_index()
plt.figure(figsize=(12,8));
sns.factorplot('Year','sales',data = temp_1, hue='Year', kind='point');

# In[ ]:


def inverse_boxcox(y, lambda_):
    return np.exp(y) if lambda_ == 0 else np.exp(np.log(lambda_ * y + 1) / lambda_)

# In[ ]:


original_target = df_raw.sales.values
target, lambda_prophet = stats.boxcox(df_raw['sales'] + 1)
len_train=target.shape[0]
merged_df = pd.concat([df_raw, df_test])

# ## FE

# **This Code Section rather (Features) comes from this Publicly Shared Kernel and added a bit more to it on basis of this discussion**
# 
#  - https://www.kaggle.com/abhilashawasthi/feature-engineering-lgb-model/comments#362974 (discussion)
# 
#  - https://www.kaggle.com/CVxTz/keras-starter (Kernel)
# 
#  - https://www.kaggle.com/abhilashawasthi/feature-engineering-lgb-model (Kernel)
# 
# **Thanks a lot for the same !!!**
# 
# **Aditya.**

# In[ ]:


merged_df["median-store_item"] = merged_df.groupby(["item", "store"])["sales"].transform("median")
merged_df["mean-store_item"] = merged_df.groupby(["item", "store"])["sales"].transform("mean")
merged_df["mean-Month_item"] = merged_df.groupby(["Month", "item"])["sales"].transform("mean")
merged_df["median-Month_item"] = merged_df.groupby(["Month", "item"])["sales"].transform("median")
merged_df["median-Month_store"] = merged_df.groupby(["Month", "store"])["sales"].transform("median")
merged_df["median-item"] = merged_df.groupby(["item"])["sales"].transform("median")
merged_df["median-store"] = merged_df.groupby(["store"])["sales"].transform("median")
merged_df["mean-item"] = merged_df.groupby(["item"])["sales"].transform("mean")
merged_df["mean-store"] = merged_df.groupby(["store"])["sales"].transform("mean")

merged_df["median-store_item-Month"] = merged_df.groupby(['Month', "item", "store"])["sales"].transform("median")
merged_df["mean-store_item-week"] = merged_df.groupby(["item", "store",'weekofyear'])["sales"].transform("mean")
merged_df["item-Month-mean"] = merged_df.groupby(['Month', "item"])["sales"].transform("mean")# mean sales of that item  for all stores scaled
merged_df["store-Month-mean"] = merged_df.groupby(['Month', "store"])["sales"].transform("mean")# mean sales of that store  for all items scaled

# adding more lags (Check the rationale behind this in the links attached)
lags = [90,91,98,105,112,119,126,182,189,364]
for i in lags:
#     print("Done For Lag {}".format(i))
    merged_df['_'.join(['item-week_shifted-', str(i)])] = merged_df.groupby(['weekofyear',"item"])["sales"].transform(lambda x:x.shift(i).sum()) 
    merged_df['_'.join(['item-week_shifted-', str(i)])] = merged_df.groupby(['weekofyear',"item"])["sales"].transform(lambda x:x.shift(i).mean()) 
    merged_df['_'.join(['item-week_shifted-', str(i)])].fillna(merged_df['_'.join(['item-week_shifted-', str(i)])].mode()[0], inplace=True)
    ##### sales for that item i days in the past
    merged_df['_'.join(['store-week_shifted-', str(i)])] = merged_df.groupby(['weekofyear',"store"])["sales"].transform(lambda x:x.shift(i).sum())
    merged_df['_'.join(['store-week_shifted-', str(i)])] = merged_df.groupby(['weekofyear',"store"])["sales"].transform(lambda x:x.shift(i).mean()) 
    merged_df['_'.join(['store-week_shifted-', str(i)])].fillna(merged_df['_'.join(['store-week_shifted-', str(i)])].mode()[0], inplace=True)

# In[ ]:


df_raw.drop('sales', axis=1, inplace=True)
merged_df.drop(['id','date','sales'], axis=1, inplace=True)

# In[ ]:


merged_df.head(1)

# In[ ]:


# comes from the public kernel
merged_df = merged_df * 1
params = {
    'nthread': 4,
    'categorical_feature' : [0,1,9,10,12,13,14], # Day, DayOfWeek, Month, Week, Item, Store, WeekOfYear
    'max_depth': 8,
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'regression_l1',
    'metric': 'mape', # this is abs(a-e)/max(1,a)
    'num_leaves': 127,
    'learning_rate': 0.25,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 30,
    'lambda_l1': 0.06,
    'lambda_l2': 0.1,
    'verbose': -1
}

# In[ ]:


# do the training
num_folds = 3
test_x = merged_df[len_train:].values
all_x = merged_df[:len_train].values
all_y = target # removing what we did earlier

oof_preds = np.zeros([all_y.shape[0]])
sub_preds = np.zeros([test_x.shape[0]])

feature_importance_df = pd.DataFrame()
folds = KFold(n_splits=num_folds, shuffle=True, random_state=345665)

for n_fold, (train_idx, valid_idx) in enumerate(folds.split(all_x)):
    
    train_x, train_y = all_x[train_idx], all_y[train_idx]
    valid_x, valid_y = all_x[valid_idx], all_y[valid_idx]
    lgb_train = lgb.Dataset(train_x,train_y)
    lgb_valid = lgb.Dataset(valid_x,valid_y)
        
    # train
    gbm = lgb.train(params, lgb_train, 1000, 
        valid_sets=[lgb_train, lgb_valid],
        early_stopping_rounds=100, verbose_eval=100)
    
    oof_preds[valid_idx] = gbm.predict(valid_x, num_iteration=gbm.best_iteration)
    sub_preds[:] += gbm.predict(test_x, num_iteration=gbm.best_iteration) / folds.n_splits
    valid_idx += 1
    importance_df = pd.DataFrame()
    importance_df['feature'] = merged_df.columns
    importance_df['importance'] = gbm.feature_importance()
    importance_df['fold'] = n_fold + 1
    feature_importance_df = pd.concat([feature_importance_df, importance_df], axis=0)
    
e = 2 * abs(all_y - oof_preds) / ( abs(all_y)+abs(oof_preds) )
e = e.mean()
print('Full validation score With Box Cox %.4f' %e)
print('Inverting Box Cox Transformation')
print('Done!!')

sub_preds = inverse_boxcox(sub_preds , lambda_prophet) - 1
oof_preds = inverse_boxcox(oof_preds , lambda_prophet) - 1
e = 2 * abs(all_y - oof_preds) / ( abs(all_y)+abs(oof_preds) )
e = e.mean()
print('Full validation score Re-Box Cox Transformation is %.4f' %e)
#Don't Forget to apply inverse box-cox

# In[ ]:


feature_importance_df.head()

# In[ ]:


importance_df.sort_values(['importance'], ascending=False, inplace=True);

# In[ ]:


def plot_fi(fi): 
    return fi.plot('feature', 'importance', 'barh', figsize=(12,12), legend=False)

# In[ ]:


plot_fi(importance_df[:]);

# In[ ]:


merged_df.get_ftype_counts()

# In[ ]:


# OHE FOR 0,1,9,10,12,13,14  Cols - Day, Dayofweek, Month, Week, item, store, weekofyear
print("Before OHE", merged_df.shape)
merged_df = pd.get_dummies(merged_df, columns=['Day', 'Dayofweek', 'Month', 'Week', 'item', 'store', 'weekofyear'])
print("After OHE", merged_df.shape)
test_x = merged_df[len_train:].values
all_x = merged_df[:len_train].values
all_y = target;

# In[ ]:


def XGB_regressor(train_X, train_y, test_X, test_y= None, feature_names=None, seed_val=2018, num_rounds=500):

    param = {}
    param['objective'] = 'reg:linear'
    param['eta'] = 0.1
    param['max_depth'] = 5
    param['silent'] = 1
    param['eval_metric'] = 'mae'
    param['min_child_weight'] = 4
    param['subsample'] = 0.8
    param['colsample_bytree'] = 0.8
    param['seed'] = seed_val
    num_rounds = num_rounds

    plst = list(param.items())

    xgtrain = xgb.DMatrix(train_X, label=train_y)

    if test_y is not None:
        xgtest = xgb.DMatrix(test_X, label=test_y)
        watchlist = [ (xgtrain,'train'), (xgtest, 'test') ]
        model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=20)
    else:
        xgtest = xgb.DMatrix(test_X)
        model = xgb.train(plst, xgtrain, num_rounds)
        
    return model    

# In[ ]:


model = XGB_regressor(train_X = all_x, train_y = all_y, test_X = test_x)
y_test = model.predict(xgb.DMatrix(test_x), ntree_limit = model.best_ntree_limit)

# In[ ]:


print('Inverting Box Cox Transformation')
y_test = inverse_boxcox(y_test, lambda_prophet) - 1

# ## Prophet
# 
# *(From the Docs Itself)*
# 
# let's take a closer look at how Prophet works. In its essence, this library utilizes the additive regression model $y(t)$ comprising the following components:
# 
# $$y(t) = g(t) + s(t) + h(t) + \epsilon_{t},$$ where:
# 
# - Trend $g(t)$ models non-periodic changes.
# - Seasonality $s(t)$ represents periodic changes.
# - Holidays component $h(t)$ contributes information about holidays and events.
# 
# 
# > ### Trend     $g(t)$
# 
# The Prophet library implements two possible trend models for $g(t)$.
# 
# The first one is called Nonlinear, Saturating Growth. It is represented in the form of the logistic growth model:
# 
# $$g(t) = \frac{C}{1+e^{-k(t - m)}},$$ where:
# 
# - $C$ is the carrying capacity (that is the curve's maximum value).
# - $k$ is the growth rate (which represents "the steepness" of the curve).
# - $m$ is an offset parameter.
# 
# > ### Seasonality    $s(t)$
# 
# The seasonal component $s(t)$ provides a flexible model of periodic changes due to weekly and yearly seasonality. Yearly seasonality model in Prophet relies on Fourier series.
# 
# > ### Holidays and Events      $h(t)$
# 
# The component $h(t)$ represents predictable abnormal days of the year including those on irregular schedules, e.g., Black Fridays.
# 
# To utilize this feature, the analyst needs to provide a custom list of events.
# 
# > ### Error    $\epsilon(t)$
# 
# The error term $\epsilon(t)$ represents information that was not reflected in the model. Usually it is modeled as normally distributed noise.
# 
# In describing these time series, we have used words such as $“trend”$ and $“seasonal”$ which need to be defined more carefully.
# 
# >Trend
# 
#  A trend exists when there is a long-term increase or decrease in the data. It does not have to be linear. Sometimes we will refer to a trend as “changing direction”, when it might go from an increasing trend to a decreasing trend.
#  
# > Seasonal
# 
#  A seasonal pattern occurs when a time series is affected by seasonal factors such as the time of the year or the day of the week. Seasonality is always of a fixed and known frequency.
# 
# > Cycle
# 
# A cycle occurs when the data exhibit rises and falls that are not of a fixed frequency. These fluctuations are usually due to economic conditions, and are often related to the “business cycle”. The duration of these fluctuations is usually at least 2 years.

# In[ ]:


df = date_sales.reset_index()
df.columns = ['ds', 'y']

# In[ ]:


df.head()

# In[ ]:


df['store'] = df_raw['store'].copy()
df['Week'] = df_raw['Week'].copy()
df['item'] = df_raw['item'].copy()

# In[ ]:


df = df.query('item == 1 & store == 1')

# In[ ]:


df.groupby(['Week','store','item'])['y'].mean().reset_index().head(10)

# - The authors of the library generally advise to make predictions based on at least several months, ideally, more than a year of historical data. Luckily, in our case we have more than a couple of years of data to fit the model.
# 
# - To measure the quality of our forecast, we need to split our dataset into the historical part and the prediction part... (We should have done this)

# In[ ]:


prediction_size = 31 
train_df = df[:-prediction_size]
train_df.tail(n=3)

# Now we need to create a new Prophet object. Here we can pass the parameters of the model into the constructor. But currently we will use the defaults as it is.. Then we train our model by invoking its fit method on our training dataset:

# In[ ]:


m = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=True)
m.fit(train_df[['ds','y']]);

# Using the helper method ```Prophet.make_future_dataframe```, we create a dataframe which will contain all dates from the history and also extend into the future for those 92 days that we left out before.

# In[ ]:


future = m.make_future_dataframe(periods=prediction_size)
future.tail(n=3)

# We predict values with Prophet by passing in the dates for which we want to create a forecast. If we also supply the historical dates (as in our case), then in addition to the prediction we will get an in-sample fit for the history. Let's call the model's predict method with our future dataframe as an input:

# In[ ]:


forecast = m.predict(future)
forecast.tail(n=3)

# In the resulting dataframe you can see many columns characterizing the prediction, including trend and seasonality components as well as their confidence intervals. The forecast itself is stored in the yhat column.
# 
# The Prophet library has its own built-in tools for visualization that enable us to quickly evaluate the result.
# 
# - First, there is a method called Prophet.plot that plots all the points from the forecast:
# - The Second function Prophet.plot_components might be much more useful in our case. It allows us to observe different components of the model separately: trend, yearly and weekly seasonality. In addition, if you supply information about holidays and events to your model, they will also be shown in this plot.
# 
# Let's try it out:

# In[ ]:


m.plot(forecast)
m.plot_components(forecast)

# In[ ]:


#Such a bad baseline forecasting on train data.Isn't it!! 

# The last Weekly Plot Says it All.
# 
# - **Seems like People go to Shopping Mostly in July**[](http://)
# - **Have a look at the peak at Sundays and Saturdays** **(3rd Plot)**
# - **So we should add the holidays effect to make Prohet perform better.**

# ## Adding Holidays (Finally)

# Kindly Refer To the [docs](https://facebook.github.io/prophet/docs/seasonality,_holiday_effects,_and_regressors.html) for the nomenclature...
# 
# (In my understanding, It's just denoting that one is more dominant than the other..)

# In[ ]:


playoffs = pd.DataFrame({
  'holiday' : 'playoff',
  'ds' : pd.to_datetime(['2013-01-12','2013-07-12','2013-12-24','2014-01-12', '2014-07-12', '2014-07-19',
                 '2014-07-02','2014-12-24', '2015-07-11','2015-12-24', '2016-07-17',
                 '2016-07-24', '2016-07-07','2016-07-24','2016-12-24','2017-07-17','2017-07-24','2017-07-07','2017-12-24']),
  'lower_window' : 0,
  'upper_window' : 2}
)
superbowls = pd.DataFrame({
  'holiday': 'superbowl',
  'ds': pd.to_datetime(['2013-01-01','2013-01-21','2013-02-14','2013-02-18',
'2013-05-27','2013-07-04','2013-09-02','2013-10-14','2013-11-11','2013-11-28','2013-12-25','2014-01-01','2014-01-20','2014-02-14','2014-02-17',
'2014-05-26','2014-07-04','2014-09-01','2014-10-13','2014-11-11','2014-11-27','2014-12-25','2015-01-01','2015-01-19','2015-02-14','2015-02-16',
'2015-05-25','2015-07-03','2015-09-07','2015-10-12','2015-11-11','2015-11-26','2015-12-25','2016-01-01','2016-01-18','2016-02-14','2016-02-15',
'2016-05-30','2016-07-04','2016-09-05','2016-10-10','2016-11-11','2016-11-24','2016-12-25','2017-01-02','2017-01-16','2017-02-14','2017-02-20',
'2017-05-29','2017-07-04','2017-09-04','2017-10-09','2017-11-10','2017-11-23','2017-12-25','2018-01-01','2018-01-15','2018-02-14','2018-02-19'
                       ]),
  'lower_window': 0,
  'upper_window': 3,
})

holidays = pd.concat((playoffs, superbowls))

# In[ ]:


m_holi = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=True, holidays=holidays)
m_holi.fit(train_df[['ds','y']]);

# In[ ]:


future_holi = m_holi.make_future_dataframe(periods=prediction_size)
future_holi.tail(n=3)

# The holiday effect can be seen in the forecast dataframe:

# In[ ]:


forecast_holi = m_holi.predict(future_holi)
forecast_holi.tail(n=3)

# In[ ]:


#from the docs..
forecast_holi[(forecast_holi['playoff'] + forecast_holi['superbowl']).abs() > 0][
        ['ds', 'playoff', 'superbowl']][-10:]

# The holiday effects will also show up in the components plot, where we see that there is a spike on the days around playoff appearances, with an especially large spike for the superbowl:

# In[ ]:


m_holi.plot(forecast_holi)
m_holi.plot_components(forecast_holi)

# In[ ]:


#remember that we will evaluate the forcast later..

# ## Forecast quality evaluation

# In[ ]:


print(', '.join(forecast.columns))

# We can see that this dataframe contains all the information we need except for the historical values. We need to join the forecast object with the actual values y from the original dataset

# In[ ]:


def make_comparison_dataframe(historical, forecast):
    """Join the history with the forecast
       The resulting dataset will contain columns 'yhat', 'yhat_lower', 'yhat_upper' and 'y'.
    """
    return forecast.set_index('ds')[['yhat', 'yhat_lower', 'yhat_upper']].join(historical.set_index('ds'))

# In[ ]:


cmp_df = make_comparison_dataframe(df, forecast)
cmp_df.tail(3)

# In[ ]:


cmp_df_holi = make_comparison_dataframe(df, forecast_holi)
cmp_df_holi.tail(3)

# We are also going to define a helper function that we will use to gauge the quality of our forecasting with MAPE and MAE error measures:

# In[ ]:


def calculate_forecast_errors(df, prediction_size):
    """Calculate MAPE and MAE of the forecast.
    
       Args:
           df: joined dataset with 'y' and 'yhat' columns.
           prediction_size: number of days at the end to predict.
    """
    
    # Make a copy
    df = df.copy()
    
    # Now we calculate the values of e_i and p_i according to the formulas given in the article above.
    df['e'] = df['y'] - df['yhat']
    df['p'] = 100 * df['e'] / df['y']
    
    # Recall that we held out the values of the last `prediction_size` days
    # in order to predict them and measure the quality of the model. 
    
    # Now cut out the part of the data which we made our prediction for.
    predicted_part = df[-prediction_size:]
    
    # Define the function that averages absolute error values over the predicted part.
    error_mean = lambda error_name: np.mean(np.abs(predicted_part[error_name]))
    
    # Now we can calculate MAPE and MAE and return the resulting dictionary of errors.
    return {'MAPE': error_mean('p'), 'MAE': error_mean('e')}

# In[ ]:


for err_name, err_value in calculate_forecast_errors(cmp_df, prediction_size).items():
    print('Non Holiday', err_name, err_value)

# In[ ]:


for err_name, err_value in calculate_forecast_errors(cmp_df_holi, prediction_size).items():
    print('Including Holiday', err_name, err_value)

# As a result, the relative error of our forecast (MAPE) is about 27.5%, and on average our model is wrong by 3.54 predicts (MAE).

# In[ ]:


from plotly.offline import init_notebook_mode, iplot
from plotly import graph_objs as go

# Initialize plotly
init_notebook_mode(connected=True)

def show_forecast(cmp_df, num_predictions, num_values, title):
    """Visualize the forecast."""
    
    def create_go(name, column, num, **kwargs):
        points = cmp_df.tail(num)
        args = dict(name=name, x=points.index, y=points[column], mode='lines')
        args.update(kwargs)
        return go.Scatter(**args)
    
    lower_bound = create_go('Lower Bound', 'yhat_lower', num_predictions,
                            line=dict(width=0),
                            marker=dict(color="aqua"))
    upper_bound = create_go('Upper Bound', 'yhat_upper', num_predictions,
                            line=dict(width=0),
                            marker=dict(color="aqua"),
                            fillcolor='rgba(68, 68, 68, 0.3)', 
                            fill='tonexty')
    forecast = create_go('Forecast', 'yhat', num_predictions,
                         line=dict(color='rgb(31, 119, 180)'))
    actual = create_go('Actual', 'y', num_values,
                       marker=dict(color="red"))
    
    # In this case the order of the series is important because of the filling
    data = [lower_bound, upper_bound, forecast, actual]

    layout = go.Layout(yaxis=dict(title='sales'), title=title, showlegend = False)
    fig = go.Figure(data=data, layout=layout)
    iplot(fig, show_link=False)

show_forecast(cmp_df, prediction_size, 100, 'Sales on Store $1$ for Item $1$')

# In[ ]:


show_forecast(cmp_df_holi, prediction_size, 100, 'Sales on Store $1$ for Item $1$ Holidays Version')

# ### Prophet With Box Cox Transformation

# In[ ]:


#train_df2 = df.copy().set_index('ds')
train_df2 = df[:-prediction_size]

# In[ ]:


train_df2['y'], lambda_prophet = stats.boxcox(train_df2['y'])
train_df2.reset_index(inplace=True)

# In[ ]:


m2 = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=True)
m2.fit(train_df2[['ds','y']]);

# In[ ]:


future2 = m2.make_future_dataframe(periods=prediction_size)
forecast2 = m2.predict(future2)

# In[ ]:


for column in ['yhat', 'yhat_lower', 'yhat_upper']:
    forecast2[column] = inverse_boxcox(forecast2[column], lambda_prophet)

# In[ ]:


cmp_df2 = make_comparison_dataframe(df, forecast2)
cmp_df2.tail()

# In[ ]:


cmp_df2 = make_comparison_dataframe(df, forecast2)
for err_name, err_value in calculate_forecast_errors(cmp_df2, prediction_size).items():
    print('Including Holidays', err_name, err_value)
# We Get Slight Improvement Over No Transformations !!!

# In[ ]:


show_forecast(cmp_df, prediction_size, 100, 'No transformations')
show_forecast(cmp_df2, prediction_size, 100, 'With Box–Cox transformation')

# ## Basic ARIMA

# **We are going to apply one of the most commonly used method for time-series forecasting, known as ARIMA, which stands for Autoregressive Integrated Moving Average.**
# 
# **ARIMA models are denoted with the notation ARIMA(p, d, q). These three parameters account for seasonality, trend, and noise in data..**
# 
# - AR: Auto-Regressive (p): AR terms are just lags of dependent variable. For example lets say p is 3, we will use x(t-1), x(t-2) and x(t-3) to predict x(t)
# - I: Integrated (d): These are the number of nonseasonal differences. For example, in our case we take the first order difference. So we pass that variable and put d=0
# - MA: Moving Averages (q): MA terms are lagged forecast errors in prediction equation.

# In[ ]:


import itertools
p = d = q = range(0, 3)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
print('Examples of parameter combinations for Seasonal ARIMA...')
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))

# In[ ]:


# figure for subplots
plt.figure(figsize = (12, 8))
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# acf and pacf for A
plt.subplot(421); plot_acf(sales_a, lags = 50, ax = plt.gca(), color = c)
plt.subplot(422); plot_pacf(sales_a, lags = 50, ax = plt.gca(), color = c)

# acf and pacf for B
plt.subplot(423); plot_acf(sales_b, lags = 50, ax = plt.gca(), color = c)
plt.subplot(424); plot_pacf(sales_b, lags = 50, ax = plt.gca(), color = c)

# acf and pacf for C
plt.subplot(425); plot_acf(sales_c, lags = 50, ax = plt.gca(), color = c)
plt.subplot(426); plot_pacf(sales_c, lags = 50, ax = plt.gca(), color = c)

# acf and pacf for D
plt.subplot(427); plot_acf(sales_d, lags = 50, ax = plt.gca(), color = c)
plt.subplot(428); plot_pacf(sales_d, lags = 50, ax = plt.gca(), color = c)
#these plots are showing the correlation of the series with itself, lagged by x time units correlation of the series with itself, lagged by x time units.

# **From the above we cn see that the lags till 50 are having weightage wrt the ACF Plots, but according to the PACF plots they valley out after the 10th lag...**

# In[ ]:


cnt = 0
for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(y,
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)
            results = mod.fit()
            cnt += 1
            if cnt % 50 :
                print('Current Iter - {}, ARIMA{}x{} 12 - AIC:{}'.format(cnt, param, param_seasonal, results.aic))
        except:
            continue

# The above output suggests that SARIMA(2, 0, 1)x(2, 2, 0, 12) yields the lowest AIC value of 17.435499462373613. Therefore we should consider this to be optimal option.
# - ARIMA(0, 0, 0)x(2, 2, 0, 12)12 - AIC:28.152584128715233
# - ARIMA(0, 0, 1)x(2, 2, 0, 12)12 - AIC:21.20352160942468
# - ARIMA(0, 0, 2)x(2, 2, 0, 12)12 - AIC:18.308712222027623
# - ARIMA(1, 0, 1)x(2, 2, 0, 12)12 - AIC:18.039431593093965
# - ARIMA(1, 0, 2)x(2, 2, 0, 12)12 - AIC:17.583895110587616
# - ARIMA(2, 0, 1)x(2, 2, 0, 12)12 - AIC:17.435499462373613
# - ARIMA(2, 0, 2)x(2, 2, 0, 12)12 - AIC:17.473412955915293

# In[ ]:


mod = sm.tsa.statespace.SARIMAX(y,
                                order=(2, 0, 1),
                                seasonal_order=(2, 2, 0, 12),
                                enforce_stationarity=False,
                                enforce_invertibility=False)
results = mod.fit()
print(results.summary().tables[1])

# In[ ]:


## Validating Forecast
pred = results.get_prediction(start=pd.to_datetime('2017-01-01'), dynamic=False)
pred_ci = pred.conf_int()
ax = y['2014':].plot(label='observed')
pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(14, 7))
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.2)
ax.set_xlabel('Date')
ax.set_ylabel('Sales')
plt.legend()

# In[ ]:


y_forecasted = pred.predicted_mean
y_truth = y['2017-01-01':]
mse = ((y_forecasted - y_truth) ** 2).mean()
print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))
#The MSE is a measure of the quality of an estimator — it is always non-negative, 
#and the smaller the MSE, the closer we are to finding the line of best fit.

# In[ ]:


pred_uc = results.get_forecast(steps=100)
pred_ci = pred_uc.conf_int()
ax = y.plot(label='observed', figsize=(14, 7))
pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.25)
ax.set_xlabel('Date')
ax.set_ylabel('Sales')
plt.legend()

# In[ ]:


subs.head()

# In[ ]:


out_df_lgb = pd.DataFrame({'id': subs.id.astype(np.int32), 'sales': sub_preds.astype(np.int32)})
out_df_xgb = pd.DataFrame({'id': subs.id.astype(np.int32), 'sales': y_test.astype(np.int32)})
out_df_combined = pd.DataFrame({'id': subs.id.astype(np.int32), 'sales': ((sub_preds + y_test + 4)/2.).astype(np.int32)})
out_df_combined_25_75 = pd.DataFrame({'id': subs.id.astype(np.int32), 'sales': ((sub_preds *.25 + y_test *.75 + 4)).astype(np.int32)})

out_df_lgb.to_csv('submission_lgbm.csv', index=False)
out_df_xgb.to_csv('submission_xgb.csv', index=False)
out_df_combined.to_csv('submission_combined.csv', index=False)
out_df_combined_25_75.to_csv('submission_combined_25_75.csv', index=False)

# In[ ]:


out_df_combined.head(10)

# ## Keras Embeddings 

# In[ ]:


from keras.models import Sequential
from keras.layers import Dense, Activation, Reshape, merge, Embedding, Input, Concatenate
from keras.models import Model as KerasModel
import keras.backend as K

# In[ ]:


cat_cols = ['Day', 'Dayofweek', 'Month', 'Week', 'item', 'store', 'weekofyear','Year']
for cols in cat_cols:
    df_raw[cols] = df_raw[cols].astype('category')
    df_test[cols] = df_test[cols].astype('category')

# In[ ]:


df_raw_cats = df_raw[cat_cols].copy()
df_test_cats = df_test[cat_cols].copy()

# In[ ]:


cat_cols

# In[ ]:


df_raw_cats.head()

# In[ ]:


def split_features(X):
    
    X_list = []
    
    day = X[..., [0]]
    X_list.append(day)

    day_of_week = X[..., [1]]
    X_list.append(day_of_week)

    month = X[..., [2]]
    X_list.append(month)

    week_of_year = X[..., [6]]
    X_list.append(week_of_year)
    
    item = X[..., [4]]
    X_list.append(item)
    
    store = X[..., [5]]
    X_list.append(store)
    
    year = X[..., [7]]
    X_list.append(year)

    return X_list

# In[ ]:


def custom_smape(x, x_): # From the Public Kernel https://www.kaggle.com/rezas26/simple-keras-starter        
    return K.mean(2*K.abs(x-x_)/(K.abs(x)+K.abs(x_)))
    
class NN_with_EntityEmbedding():

    def __init__(self, X_train, y_train):
        
        super().__init__()
        
        self.epochs = 3
        self.__build_keras_model()
        self.fit(X_train, y_train)
    
    def preprocessing(self, X):
        
        X_list = split_features(X)
        return X_list

    def __build_keras_model(self):
        
        model_day = Input(shape=(1,))
        output_day = Embedding(31, 16, name='day_embedding')(model_day)
        output_day = Reshape(target_shape=(16,))(output_day)

        model_dow = Input(shape=(1,))
        output_dow = Embedding(7, 5, name='dow_embedding')(model_dow)
        output_dow = Reshape(target_shape=(5,))(output_dow)
        
        input_month = Input(shape=(1,))
        output_month = Embedding(12, 6, name='month_embedding')(input_month)
        output_month = Reshape(target_shape=(6,))(output_month)
        
        model_woy = Input(shape=(1,))
        output_woy = Embedding(52, 26, name='week_embedding')(model_woy)
        output_woy = Reshape(target_shape=(26,))(output_woy)
        
        model_item = Input(shape=(1,))
        output_item = Embedding(50, 26, name='item_embedding')(model_item)
        output_item = Reshape(target_shape=(26,))(output_item)
        
        model_store = Input(shape=(1,))
        output_store = Embedding(10, 6, name='store_embedding')(model_store)
        output_store = Reshape(target_shape=(6,))(output_store)
        
        model_year = Input(shape=(1,))
        output_year = Embedding(5, 3, name='year_embedding')(model_year)
        output_year = Reshape(target_shape=(3,))(output_year)

        input_model = [model_day, model_dow, input_month,
                       model_woy, model_item, model_store, model_year]

        output_embeddings = [output_day, output_dow, output_month,
                             output_woy, output_item, output_store, output_year]

        output_model = Concatenate()(output_embeddings)
        output_model = Dense(128, kernel_initializer="glorot_uniform")(output_model)
        output_model = Activation('relu')(output_model)
        output_model = Dense(32, kernel_initializer="glorot_uniform")(output_model)
        output_model = Activation('relu')(output_model)
        output_model = Dense(1)(output_model)

        self.model = KerasModel(inputs=input_model, outputs=output_model)
        self.model.compile(loss= custom_smape, optimizer='sgd')


    def fit(self, X_train, y_train):
        self.model.fit(self.preprocessing(X_train), y_train, epochs=self.epochs, batch_size=64)

    def guess(self, features):
        features = self.preprocessing(features)
        result = self.model.predict(features).flatten()
        return result

# In[ ]:


nn = NN_with_EntityEmbedding(df_raw_cats.values, original_target)

# In[ ]:


nn.model.summary()

# In[ ]:


emd_layers = []
for idx,layer in enumerate(nn.model.layers):
    if 'embedding' in str(layer):
        emd_layers.append(idx)
        print(idx,layer)

# In[ ]:


emd_layers

# In[ ]:


embd_weights = []
for i in emd_layers:
    embd_weights.append(nn.model.layers[i].get_weights()[0])
    #print(nn.model.layers[i].get_weights()[0])
    #print('#'*60, i)

# In[ ]:


nn_preds = nn.guess(df_test_cats.values)
min(nn_preds), max(nn_preds)

# In[ ]:


out_df_nn = pd.DataFrame({'id': subs.id.astype(np.int32), 'sales': nn_preds.astype(np.int32)})
out_df_nn.to_csv('submission_nn.csv', index=False)

out_df_combined = pd.DataFrame({'id': subs.id.astype(np.int32), 'sales': ((sub_preds + nn_preds + 4)/2.).astype(np.int32)})
out_df_combined.to_csv('submission_combined_nn_lgbm.csv', index=False)

# ## Thanks For Making It To The End !!!
