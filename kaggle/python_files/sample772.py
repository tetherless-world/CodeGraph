#!/usr/bin/env python
# coding: utf-8

# # Strategies for Flux Time Series Preprocessing
# ### From simple to advanced methods

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from itertools import chain
sns.set_style('whitegrid')
warnings.simplefilter('ignore', FutureWarning)

# One of the first major challenges in this competition is to properly preprocess the light curve time series data. Unlike many other "well-behaved" time series, the light curves here are not only irregular in terms of observation intervals, but also unsynchronised across different light bands. This makes applying many existing tools and techniques difficult because they often require a regular time series, preferably without missing values. Let us first look at what we have:

# In[ ]:


train_series = pd.read_csv('../input/training_set.csv')
train_metadata = pd.read_csv('../input/training_set_metadata.csv')

# In[ ]:


train_series.head(10)

# There are actually 3 time series (flux, flux_err, detected) per band per object. However, as a first step, here we will only focus on the "flux" time series. Even so, we still have 6 passbands for each object.

# We first look at the distribution of time series lengths.

# In[ ]:


ts_lens = train_series.groupby(['object_id', 'passband']).size()
f, ax = plt.subplots(figsize=(12, 6))
sns.distplot(ts_lens, ax=ax)
ax.set_title('distribution of time series lengths')
plt.show()

# As we see here, the lengths of light curve time series have a large range, so we may face difficulties trying to convert them all into the same length.

# Let us also look at the times at which observations happen, and count the number of observations made at each point in time:

# In[ ]:


f, ax = plt.subplots(figsize=(12, 6))
sns.distplot(train_series['mjd'], ax=ax, bins=200)
ax.set_title('number of observations made at each time point')
plt.show()

# What we see here shows that the observations are not taken evenly through the entire sampling period. There are periods where more samples than usual are taken, but also periods with few observations.
# 
# As for each individual object:

# In[ ]:


f, ax = plt.subplots(figsize=(12, 6))
sns.distplot(train_series[train_series['object_id'] == 615]['mjd'], ax=ax, bins=200)
ax.set_title('number of observations made at each time point')
plt.show()

# What about for a single light band?

# In[ ]:


f, ax = plt.subplots(figsize=(12, 6))
sns.distplot(train_series[(train_series['object_id'] == 615) 
                          & (train_series['passband'] == 2)]['mjd'], ax=ax, bins=200)
ax.set_title('number of observations made at each time point')
plt.show()

# We see that even for individual objects, observations are not taken evenly. There can be large gaps between periods of rich observations.

# Let us also look at the number of observations for each object:

# In[ ]:


obj_obs_count = train_series['object_id'].value_counts().reset_index()
obj_obs_count.columns = ['object_id', 'count']

obj_obs_count_w_ddf = pd.merge(
    obj_obs_count, train_metadata[['object_id', 'ddf']], on='object_id')

selected = obj_obs_count_w_ddf.groupby('ddf')['count'].value_counts()
selected.index.names = ['ddf', 'count_val']
selected = selected.reset_index().pivot('count_val', 'ddf',
                                        'count').rename(columns={
                                            0: 'nonddf',
                                            1: 'ddf'
                                        }).fillna(0)
selected['total'] = selected['nonddf'] + selected['ddf']

f, ax = plt.subplots(figsize=(12, 6))
ax.vlines(x=selected.index, ymin=0, ymax=selected['total'], 
          colors='red', label='ddf')
ax.vlines(x=selected.index, ymin=0, ymax=selected['nonddf'], 
          colors='blue', label='non-ddf')
ax.set_title('number of observations per object')
plt.legend()
plt.show()

# As can be seen above, there is a large gap between the number of observations for an average DDF object and non-DDF object.

# Judging by the observations we made so far, the processing of the light curve time series is indeed a challenging task. We have the following difficulties to overcome:
# 1. The observations, even for one object in one band, are not evenly spaced, so adjacent observations in the time series are not equally "close" to each other.
# 2. The length of the time series vary quite a lot, even among the colour bands of the same object. At no point do we have multiple colour band observations simultaneously, so we cannot naively treat the multiple light bands' time series as a multivariate series.
# 3. There are some large gaps between periods where data is avaiable for an object, so we may miss characteristics of the time series at a certain period if we only look at global features.
# 4. The time series for different objects are not synchronised, making comparisons between objects difficult. We will have to disocover features that are invariant with time shifting or slight stretching.

# #### Ignore time values

# There are multiple strategies to approach this problem, with varying degrees of complexity.
# * We may completely ignore the time values and simply treat the time series as sequences. Many sequence features such as min, max, range, std, monotonicity etc. are invariant with the loss of time interval information, so we can still build many useful features. However, it comes with two major disadvantages:
#     1. We will not able to align observations that occur close together, so we will have to analyse the time series for each passband individually without exploiting their relationships.
#     2. We will lose access to many useful feature extraction techniques, or at least the form of these techniques that are typically implemented in common tools if we throw out time interval information. Features like autocorrelation do not make sense if we do not know the actual time gaps between observations.
#     
# Performing time-less analysis is simple. We do not even need much preprocessing, which makes it a good starting point. In the following example, we extract four very basic features from the time-less time series:

# In[ ]:


simple_features = train_series.groupby(
    ['object_id', 'passband'])['flux'].agg(
    ['mean', 'max', 'min', 'std']).unstack('passband')
simple_features.head().T

# Even if we cannot extract interaction features from multiple bands directly, we may still be able to discover their relationships from the features we have in downstream analysis and learning tasks.
# 
# **What not to do here:** grab cesium or tsfresh, spin up a few cores and extract all the fancy features you can get. Be aware that many of the features in these libraries do not work well with unevenly-spaced time series. I wasted many CPU hours here myself :(

# However, we cannot simply ignore the rich information we just threw out. The time-less approach won't even allow us to discover periodicity! Plus, we know from some astronomy background that it is the composition of light at different frequencies (and by extension, bands) that allow us to identify objects in the first place. We really do not want to ignore the relationship between passbands. So we can come up with a different approach:

# #### Convert to regular TS

# * We may take all **mjd** values in the dataset and declare them 'valid observation timestamps', then construct a time series for each object and each passband that include all these observation timestamps, filling in NA if obseration is not available. Now that all objects and all passband have time series of the same length, they are comparable, and many time series analysis techniques can be applied as long as they can deal with missing values. One immediate drawback of this method is that none of the objects will have multiple passband data at a given timestamp, and there will be a huge number of NAs. The time series will also be really long due to the large number of possible timestamps. Also, we will face more issues when we want to build a model and apply it to new data, as the new timestamps will not match the timestamps in our current dataset. We can make a compromise by binning **mjd** and averaging the observations within a bin, and use all bins within the range of **mjd** as possible time steps in the time series. By doing so, we will be able to obtain a collection of time series that are equally spaced, not overwhemingly NA, and have the same length.

# In[ ]:


print(f'mjd unique values: {train_series["mjd"].nunique()}')

# In[ ]:


print(f'int (1day) mjd unique values: {train_series["mjd"].astype(int).nunique()}')
print(f'''int mjd bins: {train_series["mjd"].astype(int).max()
      - train_series["mjd"].astype(int).min() + 1}''')

# In[ ]:


print(f'5day mjd unique values: {(train_series["mjd"]/5).astype(int).nunique()}')
print(f'''5day mjd bins: {int((train_series["mjd"].astype(int).max()
      - train_series["mjd"].astype(int).min()) / 5 + 1)}''')

# As we see above, while using each unique **mjd** value as time steps is really bad, binning by day or even by 5 days greatly reduces time series length. Here is an example of constructing the resulting time series using binned observations:

# In[ ]:


# binning
ts_mod = train_series[['object_id', 'mjd', 'passband', 'flux']].copy()
ts_mod['mjd_d5'] = (ts_mod['mjd'] / 5).astype(int)
ts_mod = ts_mod.groupby(['object_id', 'mjd_d5', 'passband'])['flux'].mean().reset_index()
ts_mod.head(10)

# In[ ]:


# pivotting
ts_piv = pd.pivot_table(ts_mod, 
                        index='object_id', 
                        columns=['mjd_d5', 'passband'], 
                        values='flux',
                        dropna=False)
ts_piv.head(10)

# In[ ]:


# resetting column index to fill mjd_d5 gaps 
t_min, t_max = ts_piv.columns.levels[0].min(), ts_piv.columns.levels[0].max()
t_range = range(t_min, t_max + 1)
mux = pd.MultiIndex.from_product([list(t_range), list(range(6))], 
                                 names=['mjd_d5', 'passband'])
ts_piv = ts_piv.reindex(columns=mux).stack('passband')
ts_piv.head(10)

# We have now converted the unaligned time series into aligned, equally spaced time series using binning-and-averaging. We can see below that despite our efforts, nearly 90% of the values in our new time series are NaN, so there will be major challenges trying to make use of the converted data with tools that cannot cope well with NAs. Normal procedures like mean imputation or interpolation won't work well here due to the prevalance of missing values. However, some algorithms like LSTM may learn to ignore padded values and data gaps and still manage to extract valuable information from the time series. It is even possible to build multiple time series collections with different bins to look for features at different time scales.

# In[ ]:


np.mean(np.ravel(pd.isna(ts_piv).values))

# One of the drawbacks of the method above is that binning appears to be quite arbitrary. What we are effectively doing is sampling at different time points and averaging flux values around these points. So why not do it properly?

# #### Sample with a distance kernel

# * We may sample the time series data at evenly-spaced time steps, using a time kernel to determine how much weight we put on each observation when taking the moving average.

# In[ ]:


def time_kernel(diff, tau):
    return np.exp(-diff ** 2 / (2 * tau ** 2))

# In[ ]:


t_min, t_max = train_series['mjd'].min(), train_series['mjd'].max()
t_min, t_max

# In[ ]:


sample_points = np.array(np.arange(t_min, t_max, 20))
sample_points

# In[ ]:


weights = time_kernel(np.expand_dims(sample_points, 0) 
                      - np.expand_dims(train_series['mjd'].values, 1), 5)
ts_mod = train_series[['object_id', 'mjd', 'passband', 'flux']].copy()
for i in range(len(sample_points)):
    ts_mod[f'sw_{i}'] = weights[:, i]

# In[ ]:


def group_transform(chunk):
    sample_weights = chunk[[f'sw_{i}' for i in range(len(sample_points))]]
    sample_weights /= np.sum(sample_weights, axis=0)
    weighted_flux = np.expand_dims(chunk['flux'].values, 1) * sample_weights.fillna(0)
    return np.sum(weighted_flux, axis=0)

# In[ ]:


# only using a small sample as this step is slower than the other steps
ts_samp = ts_mod[ts_mod['object_id'].isin([615, 713])].groupby(
    ['object_id', 'passband']).apply(group_transform)

# In[ ]:


ts_samp

# We can see above that we have managed to sample from the time series, using the time difference from the sample point to determine moving average weights.
# 
# There are still many issues with this method. For example, it still does not solve the problem of long gaps between available data, and the use of 0 as a filler value for NA is questionable (however we cannot retain NA, as that will make calculating moving average difficult). Overall, this is still a better approach than simple binning, and still relatively cheap to compute.

# #### Use time / time difference as a feature

# This idea is quite simple. Why not just concatenate the flux values with the timestamp at which they are observed, or the time elapsed since last observation, and pass the whole sequence to a learning algorithm? Indeed, some algorithms have been shown to be able to cope with that. This is not so different from "given (x, y), fit the curve and extract features from the curve". An LSTM or even a regular MLP might be able to deal with that. There is an [example](https://arxiv.org/pdf/1711.10609.pdf) of this being applied to a very similar problem.
# 
# Also, some time series feature extractors accept (t, x) pairs instead of (x) sequences.
# 
# (WIP)

# #### Convert time series to phase series

# If we know that some of the time series will be periodic, we can attempt to find the optimal period and convert the time series to phase domain, so that we can avoid the issues of observation gap, alleviate the problem of sparse observations and make better use of the periodic property. We may also assume that for most objects that exhibit period behaviour, the period is the same for all colour bands and the phases should all be in sync, because they are likely driven by the same physical event.
# To discover the period of the time series, we first normalise all series then extract features about the most prominent frequencies / periods.

# In[ ]:


groups = train_series.groupby(['object_id', 'passband'])

# In[ ]:


def normalise(ts):
    return (ts - ts.mean()) / ts.std()

# In[ ]:


times = groups.apply(
    lambda block: block['mjd'].values).reset_index().rename(columns={0: 'seq'})

# In[ ]:


flux = groups.apply(
    lambda block: normalise(block['flux']).values
).reset_index().rename(columns={0: 'seq'})

# In[ ]:


times_list = times.groupby('object_id').apply(lambda x: x['seq'].tolist()).tolist()
flux_list = flux.groupby('object_id').apply(lambda x: x['seq'].tolist()).tolist()

# In[ ]:


import cesium.featurize as featurize
from scipy import signal
import warnings

# To save some kernel time, we will only extract features from a small subset of objects.
# 
# ([**FIXED - See below**]For some reason, the frequencies features of the cesium package are not working properly for me, so I have to revert to the method used by Michal Haltuf in his notebook [Feature extraction using period analysis](https://www.kaggle.com/rejpalcz/feature-extraction-using-period-analysis). It takes quite some time even for just 20 objects as you can see below:
# 

# In[ ]:


# def extract_freq(t, m, e):
#     fs = np.linspace(2*np.pi/0.1, 2*np.pi/500, 10000)
#     pgram = signal.lombscargle(t, m, fs, normalize=True)
#     return fs[np.argmax(pgram)]

# N = 20
# warnings.simplefilter('ignore', RuntimeWarning)
# feats = featurize.featurize_time_series(times=times_list[:N],
#                                         values=flux_list[:N],
#                                         features_to_use=['freq1'],
#                                         custom_functions={'freq1': extract_freq},
#                                         scheduler=None)

# **EDIT:** I found why the features from the cesium package is acting weird. The frequency features extracted by cesium are oscilation frequencies (cycles per unit time) rather than angular frequencies (rad per unit time), so they differ by $2\pi$. We can now use cesium's 'freqN_freq' feature extractor to extract features slightly faster:

# In[ ]:


warnings.simplefilter('ignore', RuntimeWarning)
N = 100
cfeats = featurize.featurize_time_series(times=times_list[:N],
                                        values=flux_list[:N],
                                        features_to_use=['freq1_freq',
                                                        'freq1_signif',
                                                        'freq1_amplitude1'],
                                        scheduler=None)

# In[ ]:


cfeats.stack('channel').iloc[:24]

# We can see that for some objects, there appears to be a common frequency / period for all its observed passbands, whereas for others the calculated frequencies are all different and seemingly random. We may assume that the objects with completely mismatching frequencies or very low frequencies (very long periods) actually do not exhibit periodic behaviour. Let us look at a few examples where the periods do match up between most bands:

# In[ ]:


def plot_phase(n, fr):
    selected_times = times_list[n]
    selected_flux = flux_list[n]
    colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple']
    f, ax = plt.subplots(figsize=(12, 6))
    for band in range(6):
        ax.scatter(x=(selected_times[band] * fr) % 1, 
                   y=selected_flux[band], 
                   c=colors[band])
    ax.set_xlabel('phase')
    ax.set_ylabel('relative flux')
    ax.set_title(
        f'object {train_metadata["object_id"][n]}, class {train_metadata["target"][n]}')
    plt.show()

# In[ ]:


plot_phase(0, 3.081631)

# In[ ]:


plot_phase(3, 0.001921)

# In[ ]:


plot_phase(6, 1.005547)

# For periodic objects, observations made in the same phase should have similar relative flux values. As we see, Object 615 is clearly periodically changing its magnitude,  whereas object 745 and 1598's 'periodicity' are more like staying stationary most of the time with sudden bursts.
# 
# We kind of found a pattern here. If all colour bands agree in frequency, an object is likely to have periodic behaviours. When the colour bands have wildly different frequencies, the object is likely aperiodic. When most bands have very small frequencies (longer period than the observation window) and a few bands have larger detected frequencies, it is likely a burst event.
# 
# It appears that calculated periods alone cannot tell us the full picture, as there are plenty of objects in the dataset without clear periodic patterns. However, determining whether an object has a periodic pattern can be very important in determining the object class, and for objects that are actually periodic, the time series in phase space can tell us a lot more than in time space. Therefore, for a large portion of the dataset, conversion to phase space is actually a very powerful preprocessing step to use.
# 
# We will plot below a few more examples of where phase conversion works well:

# In[ ]:


plot_phase(46, 1.781711)

# In[ ]:


plot_phase(62, 4.771011)

# In[ ]:


plot_phase(69, 1.597659)

# In[ ]:


plot_phase(91, 2.668811)

# It appears that we are onto something. The classes 16 and 92 appear to be quite periodic where class 90 is what appears to be a group of burst events. So by just calculating a few frequency features and comparing these features between bands, we can already roughly tell them apart. We might still have trouble distinguishing between class 16 and 92, but with frequency divided out through phase-space conversion, we essentially have higher density data for these two classes to work with, and building a model on top of that should be easier.

# One more thing... We have to figure out if the same procedure can work well on non-DDF objects.

# In[ ]:


nonddf_pos = train_metadata[train_metadata['ddf'] == 0].index
nonddf_times_list = [v for i, v in enumerate(times_list) if i in set(nonddf_pos)]
nonddf_flux_list = [v for i, v in enumerate(flux_list) if i in set(nonddf_pos)]

# In[ ]:


warnings.simplefilter('ignore', RuntimeWarning)
N = 50
cfeats = featurize.featurize_time_series(times=nonddf_times_list[:N],
                                        values=nonddf_flux_list[:N],
                                        features_to_use=['freq1_freq',
                                                        'freq1_signif',
                                                        'freq1_amplitude1'],
                                        scheduler=None)

# In[ ]:


cfeats.stack('channel').iloc[:24]

# In[ ]:


plot_phase(nonddf_pos[15], 4.440506)

# In[ ]:


plot_phase(nonddf_pos[20], 0.831637)

# In[ ]:


plot_phase(nonddf_pos[39], 0.817696)

# In[ ]:


plot_phase(nonddf_pos[46], 2.514430)

# Generally, we found that we are still able to use frequency analysis to find the most apparent periodic classes like 16 and 92, but we have to be much more lenient on how different bands agree with each other (sometimes only 3-4 bands will strongly agree in terms of frequency), and we might have to pay much more attention to the significance level of these frequencies. This could be evidence to support the case that DDF and non-DDF objects should be dealt with using different models, or at least treated differently in model.
