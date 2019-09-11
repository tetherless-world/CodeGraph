#!/usr/bin/env python
# coding: utf-8

# **Can you distinguish between the different classes with just your eyes?**
# 
# New: Multiband frequencies are now fitted with the `gatspy` package and the fitted values are plotted. This should give us better period estimation results on periodic classes.

# * [Class 92](#class92)
# * [Class 88](#class88)
# * [Class 42](#class42)
# * [Class 90](#class90)
# * [Class 65](#class65)
# * [Class 16](#class16)
# * [Class 67](#class67)
# * [Class 95](#class95)
# * [Class 62](#class62)
# * [Class 15](#class15)
# * [Class 52](#class52)
# * [Class 6](#class6)
# * [Class 64](#class64)
# * [Class 53](#class53)

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from itertools import chain
sns.set_style('whitegrid')
warnings.simplefilter('ignore', FutureWarning)
warnings.simplefilter('ignore', RuntimeWarning)
import cesium.featurize as featurize
from gatspy.periodic import LombScargleMultiband, LombScargleMultibandFast
import pdb

# In[ ]:


train_series = pd.read_csv('../input/training_set.csv')
train_metadata = pd.read_csv('../input/training_set_metadata.csv')

# In[ ]:


groups = train_series.groupby(['object_id', 'passband'])
times = groups.apply(
    lambda block: block['mjd'].values).reset_index().rename(columns={0: 'seq'})
flux = groups.apply(
    lambda block: block['flux'].values
).reset_index().rename(columns={0: 'seq'})
err = groups.apply(
    lambda block: block['flux_err'].values
).reset_index().rename(columns={0: 'seq'})
det = groups.apply(
    lambda block: block['detected'].astype(bool).values
).reset_index().rename(columns={0: 'seq'})
times_list = times.groupby('object_id').apply(lambda x: x['seq'].tolist()).tolist()
flux_list = flux.groupby('object_id').apply(lambda x: x['seq'].tolist()).tolist()
err_list = err.groupby('object_id').apply(lambda x: x['seq'].tolist()).tolist()
det_list = det.groupby('object_id').apply(lambda x: x['seq'].tolist()).tolist()

# We cannot make much sense of the data with the large observation gap, unsynchronised passband observations and sparsity of data. But We know that some objects have periodic bebaviours, so we can attempt to fold them by period. Here we will examine which classes are more likely to be periodic, and how they typically look like.

# In[ ]:


def fit_multiband_freq(tup):
    idx, group = tup
    t, f, e, b = group['mjd'], group['flux'], group['flux_err'], group['passband']
    model = LombScargleMultiband(fit_period=True)
    model.optimizer.period_range = (0.1, int((group['mjd'].max() - group['mjd'].min()) / 2))
    model.fit(t, f, e, b)
    return model

# In[ ]:


def get_freq_features(N, subsetting_pos=None):
    if subsetting_pos is None:
        subset_times_list = times_list
        subset_flux_list = flux_list
    else:
        subset_times_list = [v for i, v in enumerate(times_list) 
                             if i in set(subsetting_pos)]
        subset_flux_list = [v for i, v in enumerate(flux_list) 
                            if i in set(subsetting_pos)]
    feats = featurize.featurize_time_series(times=subset_times_list[:N],
                                            values=subset_flux_list[:N],
                                            features_to_use=['skew',
                                                            'percent_beyond_1_std',
                                                            'percent_difference_flux_percentile'
                                                            ],
                                            scheduler=None)
    subset = train_series[train_series['object_id'].isin(
        train_metadata['object_id'].iloc[subsetting_pos].iloc[:N])]
    models = list(map(fit_multiband_freq, subset.groupby('object_id')))
    feats['object_pos'] = subsetting_pos[:N]
    feats['freq1_freq'] = [model.best_period for model in models]
    return feats, models

# In[ ]:


unique_classes = train_metadata['target'].unique()
unique_classes

# In[ ]:


def get_class_feats(label, N=10):
    class_pos = train_metadata[train_metadata['target'] == label].index
    class_feats, class_models = get_freq_features(N, class_pos)
    return class_feats, class_models

# In[ ]:


def plot_phase_curves(feats, models, use_median_freq=False, hide_undetected=True, N=10):
    for i in range(N):
        freq = feats.loc[i, 'freq1_freq'].median()
        freq_min = feats.loc[i, 'freq1_freq'].min()
        freq_std = feats.loc[i, 'freq1_freq'].std()
        skew = feats.loc[i, 'skew'].mean()
        object_pos = int(feats.loc[i, 'object_pos'][0])
        f, ax = plt.subplots(1, 2, figsize=(14, 4))
        sample = train_series[train_series['object_id'] ==
                              train_metadata['object_id'].iloc[object_pos]].copy()
        colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple']
        score = models[i].score(models[i].best_period)
        
        ax[0].scatter(x=sample['mjd'], 
                   y=sample['flux'], 
                   c=[colors[b] for b in sample['passband']],
                   s=8, alpha=0.8)
        ax[0].vlines(sample['mjd'], 
                  sample['flux'] - sample['flux_err'],
                  sample['flux'] + sample['flux_err'],
                  colors=[colors[b] for b in sample['passband']],
                  linewidth=1, alpha=0.8)
        
        sample['phase'] = (sample['mjd'] / models[i].best_period) % 1
        ax[1].scatter(x=sample['phase'], 
                   y=sample['flux'], 
                   c=[colors[b] for b in sample['passband']],
                   s=8, alpha=0.8)
        ax[1].vlines(sample['phase'], 
                  sample['flux'] - sample['flux_err'],
                  sample['flux'] + sample['flux_err'],
                  colors=[colors[b] for b in sample['passband']],
                  linewidth=1, alpha=0.8)
        x_range = np.linspace(sample['mjd'].min(), sample['mjd'].max(), 1000)
        for band in range(6):
            y = models[i].predict(x_range, band)
            xs = (x_range / models[i].best_period) % 1
            ords = np.argsort(xs)
            ax[1].plot(xs[ords], y[ords], c=colors[band], alpha=0.4)
        
        title = ax[0].get_title()
        ax[0].set_title('time')
        ax[1].set_title('phase')
        f.suptitle(title + f'object: {sample["object_id"].iloc[0]}, '
                   f'class: {train_metadata["target"].iloc[object_pos]}\n'
                   f'period: {models[i].best_period: .4}, '
                   f'period score: {score: .4}, '
                   f'mean skew: {skew:.4}', y=1.1)
        plt.show()

# In[ ]:


warnings.simplefilter('ignore', UserWarning)

# <a id='class92'></a>

# ### Class 92 ('Regular' Variable Stars?)

# In[ ]:


feats, models = get_class_feats(92)

# In[ ]:


plot_phase_curves(feats, models)

# We can see that class 92 objects are typically periodic with sub-1day periods. They have passbands that are synchronised in period and phase, and the light curves appear to be sinusoidal. They usually have high period scores, which can be used to distiniguish them from other, non-periodic classes.

# <a id='class88'></a>

# ### Class 88 ('Patient' Variable Stars?)

# In[ ]:


feats, models = get_class_feats(88)

# In[ ]:


plot_phase_curves(feats, models)

# Unlike class 99, many class 88 objects do not exhibit periodic behaviour on a small time scale, although their light curves might be periodic on a scale that is difficult to discern with our sample window. The calculated frequencies typically have very high standard deviation, and the longest period detected is usually quite long. The curves could potentially be sinusoidal in nature, but often only fragments of it can be seen in the observed windows. There could also be an overall upwards or downwards trend for the overall curve. As this is an extragalactical class, some faraway objects have higher uncertainties in their observations.
# It is worth noting that some of the class 88 objects appear to have an exactly 1-day period. While in some cases this appear to be spurious, there are also cases where it appears plausible. However, even for those objects where 1-day period is detected, from the raw time-domain plot, we can often find what appears to be a long timescale evolving trend. Further investigation is needed for this class.

# <a id='class42'></a>

# ### Class 42 (Supernovae?)

# In[ ]:


feats, models = get_class_feats(42)

# In[ ]:


plot_phase_curves(feats, models)

# Class 42 appears to be mostly flat. It is extragalactical, so high uncertainty happens in some cases. It most likely is a class of objects with "burst" events, although the actual burst is not always detected in the observation windows. When the burst happens, the object's magnitude increases dramatically across all bands and gradually falls back to normal levels in a few months. If the burst is indeed detected, it will be characterised by relatively low frequency std between bands together with very high detected periods. Other features like skewness can also be used to identify this class of objects, as a burst usually results in high skew in the light curve.

# <a id='class90'></a>

# ### Class 90 (Also Supernovae?)

# In[ ]:


feats, models = get_class_feats(90)

# In[ ]:


plot_phase_curves(feats, models)

# Class 90 is kind of similar to class 42 in that it is very likely a group of objects with "burst" behaviours, or objects with partially unobsered bursts. Like class 42, it is characterised by sudden peaks that usually last for a few months, high skew and often long fitted periods (or 1-day periods) with low score. So far we are still unable to tell the difference between class 90 and class 42.

# <a id='class65'></a>

# ### Class 65 (The galactical mystery)

# In[ ]:


feats, models = get_class_feats(65)

# In[ ]:


plot_phase_curves(feats, models)

# Class 65 is an interesting case. There just does not seem to be a good description for this class. All the fitted periods are relatively short, so likely there are likely little long-term non-periodic behaviour in this class, but all the periods seem to be random, and they all have much smaller significance levels than in other classes. The skew of the data is all over the place. Also, this is a class of objects within the galaxy, but the observation uncertainty can be pretty high. If we look at a sample of original data from this class, it appears that objects in this class are rather faint (in terms of flux difference to the reference picture). Considering the low % of 'detected' observations, it is likely that the high deviation, low error points that occasionally pop up in the light curves are actually the real events, whereas most of the curve is just background.

# In[ ]:


train_series[train_series['object_id'].isin(
    train_metadata[train_metadata['target'] == 65]['object_id'])]['flux'].describe()

# In[ ]:


train_series[train_series['object_id'].isin(
    train_metadata[train_metadata['target'] == 65]['object_id'])]['detected'].mean()

# If we try to hide all 'undetected' events, we can see that the detected events for class 65 is really sparse, but some of these events are at pretty extreme relative flux values. So significant events, but rare and probably irregular.

# <a id='class16'></a>

# ### Class 16 (Binary stars/exoplanet transits?)

# In[ ]:


feats, models = get_class_feats(16)

# In[ ]:


plot_phase_curves(feats, models)

# Class 16 contains both periodic objects and more irregular objects, but a common characteristic of object in this class appears to be that they typically have a negative skew in their light curves, corresponding to the deep dips in the light curve. This implies that rather than the luminosity of the object changing by itself over time, the variability of flux in this group might be caused by occlusion of more than one celestral bodies. The max periods of objects in this class are typically not very long. It is likely that there are more than one period for the light curves in this class.

# <a id='class67'></a>

# ### Class 67

# In[ ]:


feats, models = get_class_feats(67)

# In[ ]:


plot_phase_curves(feats, models)

# <a id='class95'></a>

# ### Class 95

# In[ ]:


feats, models = get_class_feats(95)

# In[ ]:


plot_phase_curves(feats, models)

# Class 95 is another case of non-periodic burst events, although the curvature of the peaks appear to be smaller than the peaks in other burst classes, and some bands visibly 'lag' behind others. We cannot verify with just a few examples, but if we focus just on the peaks, we might be able to find something to distinguish between these burst classes.

# <a id='class62'></a>

# ### Class 62

# In[ ]:


feats, models = get_class_feats(62)

# In[ ]:


plot_phase_curves(feats, models)

# Class 62 is yet another burst event class with seemingly volatile band 0.

# <a id='class15'></a>

# ### Class 15

# In[ ]:


feats, models = get_class_feats(15)

# In[ ]:


plot_phase_curves(feats, models)

# <a id='class52'></a>

# ### Class 52

# In[ ]:


feats, models = get_class_feats(52)

# In[ ]:


plot_phase_curves(feats, models)

# <a id='class6'></a>

# ### Class 6

# In[ ]:


feats, models = get_class_feats(6)

# In[ ]:


plot_phase_curves(feats, models)

# <a id='class64'></a>

# ### Class 64

# In[ ]:


feats, models = get_class_feats(64)

# In[ ]:


plot_phase_curves(feats, models)

# <a id='class53'></a>

# ### Class 53

# In[ ]:


feats, models = get_class_feats(53)

# In[ ]:


plot_phase_curves(feats, models)

# In[ ]:



