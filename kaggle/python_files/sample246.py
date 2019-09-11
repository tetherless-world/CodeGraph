#!/usr/bin/env python
# coding: utf-8

# # Kaggle LANL Earthquake Prediction Modeling
# ### Kevin Maher
# ### Regis University MSDS696 Data Science Practicum II
# ### Associate Professor Dr. Robert Mason
# #### May 2, 2019
# #### Spring, 2019; In partial fullfillment of the Master of Science in Data Science degree, Regis University, Denver, CO

# ### Introduction

# Presented here are a set of models for the Kaggle LANL Earthquake Challenge (Rouet-Leduc, et. al, 2019).  Exploratory data analysis (EDA) is performed in a separate Jupyter notebook, located with this file in the github repository (https://github.com/Vettejeep/MSDS696-Masters-Final-Project).  Please review the EDA for additional perspective on the problem.  The goal of the project is to predict the time that an earthquake will occur in a laboratory test.  The laboratory test applies shear forces to a sample of earth and rock containing a fault line.  Thus we note that these are laboratory earthquakes, not real earthquakes.  The simulated earthquakes tend to occur somewhat periodically because of the test setup, but this periodicity is not guaranteed to the researcher attempting to predict the time until an earthquake.  

# ### Publication

# In an effort to comply with both university and Kaggle requirements, this Jupyter notebook is being published on GitHub and on Kaggle. The notebook was designed for a university course. It has not been tested and probably will not run in the Kaggle environment.  This discloses my code which is being submitted and shared to my professor and class for grading.  The exploratory data analysis notebook for this project will also be published in the same manner.

# ### Problem Approach

# This problem has been approached here by regression modeling. The metric used by Kaggle in this competition is Mean Absolute Error (MAE) and thus a lower value is better with zero representing a perfect fit (Bilogur). This is a common regression metric. The acoustic data provided is used to create statistical features which are fed into supervised learning algorithms which then seek to predict the time until an earthquake from test signals. The training signal is provided by Kaggle in the form of a continuous acoustic signal that is over 629m samples long. This training data is accompanied by a ground truth time-to-failure (time until the next earthquake) for each acoustic sample. The user is left to decide how to extract information from the test signal in order to provide training data for their chosen machine learning algorithms. Given around 629m potential training samples, one challenge is how best to extract effective but still computationally tractable training sets from the given signal. The test signals are all 150k samples in length, thus it seems best to extract 150k sample sets from the training data.

# While there are 2624 test signals provided by Kaggle, only 13% (341) are used for the public leader board (Rouet-Leduc, et. al, 2019).  The remainder are reserved for the final scoring that will be done after the competition concludes and after this course is finished.  While the Kaggle public leader board appears to be the best test set for model ranking currently available, there might be a lot of variance in the results when the remaining 87% of the test data is revealed.  Ensembles of models may therefore perform best where their individual weaknesses and variance tend to somewhat cancel out (Demir).

# Most of the published kernel scripts that this author has reviewed on Kaggle use a data row creation method that slices the 629 million row acoustic input data evenly into 4194 non-overlapping chunks of data that are equivalent in length to the 150k sample size of the Kaggle test samples. An example of this is the Preda (2019) kernel, but there are many other excellent scripts using this approach that the reader might review on Kaggle.  Slicing the data into 4194 chunks avoids overlap and possible information leakage between these slices as they then do not share any signal information.  These scripts appear to underfit the public leader board in the sense that cross validation (CV) scores tend to be higher (worse) than the public leader board score.  When I tried the Preda (2019) script, run from an IDE, this author obtained a public leader board score on Kaggle of 1.556 for the LightGBM model presented in that script.  However, the script CV scores appear to be just above 2.0.  

# The Preda (2019) script references a script by Andrew and one by a Kaggle user named Scirpus. I believe that the Andrew script is the one by Andrew Lukayenko (Lukayenko, 2019). Many of the feature creation ideas here appear to owe their origins to the Lukayenko (2019) script and it's cited predecessors. The script by Scirpus is interesting in being a very effective genetic programming model (Scirpus, 2019). Unfortunately the C++ code that the genetic algorithm has been written in does not appear to be publicly available. Only a result function containing the genetic algorithm's output mathematical functions, relationships and coefficients seem to be given by the author.

# Partly because of the extensive exploration of slicing the data into 4194 non-overlapping slices in the Kaggle kernels by other challenge participants, and partly to set out on an individual exploration of modeling this data, a different approach is tried in the primary models presented here. 24,000 data rows were created in a stratified and then randomized manner. These are obtained from 6 simple slices of the original data, each slice is used to randomly create 4k data rows. This slicing accomplishes several objectives. First, it tends to help spread the random generation of data across the signal without risk of bunching too many slices into a compact region of the original signal. Second, it helped greatly with computational time and memory usage because multiprocessing can then be employed. In order to avoid having to load the whole 629m data set into memory 6 times, only the smaller slices with 1/6 of the data were loaded, one into each process. Multiprocessing allowed the main feature creation to run overnight, instead of possibly requiring days, which might have been required with a single process.  

# Experience below will show both the successes and challenges of this alternate method of feature creation. There are two possible approaches to model cross validation (CV) because of the stratification used by the multiprocessing. While indices for slicing data out of the model were chosen randomly, they were chosen from 6 slices of the original model data. Thus in addition to random selection for cross validation, working with 5 slices for training and 1 slice for validation as a 6 fold CV is also an option. These methods give very different and opposing CV results, but very similar Kaggle public leader board scores. This will be explored when model results are presented below.

# ### Processing Issues

# Many of the processes and functions below are very long running, possibly taking overnight, or days, to complete. It is be best to transfer them to an IDE in order to run them. Also some of the code uses multiprocessing and this can be troublesome if run from Jupyter (Singhal, 2018). Code was tested using an IDE, not this notebook. I have used the Jupyter notebook here only for documentation and presentation purposes. The code is written using Python 3.6.6 and the library dependencies as noted below near the end of this document and in the imports. This code will work best if there is available at least a four core / 8 hyper-thread CPU, it was primarily tested on a Windows 10 operating system with an AMD Ryzen 7 CPU (8 cores, 16 logical threads) and 16GB of RAM. The training data set contains more than 629 million acoustic signal samples and is 10GB in size, so there is a lot of data to process. Then 24,000 data rows are created with 900 features extracted from the signal. Consequently, the individual functions often require many hours or days to run, even when using multiprocessing.

# Another item to note is that the models mentioned herein are averaged ensembles of a cross validation (CV).  This allows usage of all of the training data for creating a Kaggle submission file while still reserving validation holdout sets.  This method also helps create more accurate models by averaging the results of different splits of the training data into training and validation sets.  The idea of creating the models in this way was taken from the Preda (2019) script as well as many others too numerous to cite that are present on the Kaggle website.  The true origin of this modeling approach is unknown to this author.  Accuracy, and Kaggle scoring position, appear to be gained by using this technique at a significant cost in additional model training time and complexity.  As an alternate view, one could argue that the CV is needed anyway, so why not take advantage of it as a direct model.

# ### Code Setup

# Below are the imports needed to run the code.  The code has been written and run in Python 3.6 and 3.7 Anaconda environments.  Many of these libraries request a citation when used in an academic paper.  Note the use of the Scikit-Learn (Pedregosa et al. (2011), XGBoost (Chen & Guestrin, 2016) and LightGBM (Ke, et al., 2017) libraries for machine learning and support.  Numpy is utilized to provide many numerical functions for feature creation (van der Walt, Colbert & Varoquaux, 2011). Pandas is very helpful for its ability to support data manipulation and feature creation (McKinney, 2010).  SciPy is utilized to provide signal processing functions, especially filtering and for Pearson's correlation metrics (Jones E., et al, 2001).  The Jupyter environment in which this project is presented is a descendant of the IPython environment originated by Pérez & Granger (2007).

# In[ ]:


import os
import time
import warnings
import traceback
import numpy as np
import pandas as pd
from scipy import stats
import scipy.signal as sg
import multiprocessing as mp
from scipy.signal import hann
from scipy.signal import hilbert
from scipy.signal import convolve
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

from tqdm import tqdm
warnings.filterwarnings("ignore")

# Define some constants.
# The signal constants define how the signal and Fourier transforms will be filtered to produce bandwidth limited features.

# In[ ]:


OUTPUT_DIR = r'd:\#earthquake\final_model'  # set for local environment
DATA_DIR = r'd:\#earthquake\data'  # set for local environment

SIG_LEN = 150000
NUM_SEG_PER_PROC = 4000
NUM_THREADS = 6

NY_FREQ_IDX = 75000  # the test signals are 150k samples long, Nyquist is thus 75k.
CUTOFF = 18000
MAX_FREQ_IDX = 20000
FREQ_STEP = 2500

# ### Feature Creation

# Function to split the raw data into 6 groups for later multiprocessing.  The feature builder function took so long that it was run as 6 concurrent processes in order to speed it up. This perhaps could have been more easily acomplished with the "skiprows" and "nrows" parameters of the Python Pandas read csv function rather than creating 6 new files. 

# In[ ]:


def split_raw_data():
    df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))

    max_start_index = len(df.index) - SIG_LEN
    slice_len = int(max_start_index / 6)

    for i in range(NUM_THREADS):
        print('working', i)
        df0 = df.iloc[slice_len * i: (slice_len * (i + 1)) + SIG_LEN]
        df0.to_csv(os.path.join(DATA_DIR, 'raw_data_%d.csv' % i), index=False)
        del df0

    del df

# Build six sets of random indices.  Stratified random sampling will be performed on the data.  This is for several reasons.  It ensures relatively even coverage of the width of the input signal and it allows for multiprocessing so that the script runs in a reasonable time.  Also, working on data chunks that represent only a portion of the very large input data set means that the whole data set is not loaded into memory multiple times (once for each process).  This makes the feature building more memory efficient.  All of this helps to avoid crashes and allow the feature building portion of the script to run overnight.

# In[ ]:


def build_rnd_idxs():
    rnd_idxs = np.zeros(shape=(NUM_THREADS, NUM_SEG_PER_PROC), dtype=np.int32)
    max_start_idx = 100000000

    for i in range(NUM_THREADS):
        np.random.seed(5591 + i)
        start_indices = np.random.randint(0, max_start_idx, size=NUM_SEG_PER_PROC, dtype=np.int32)
        rnd_idxs[i, :] = start_indices

    for i in range(NUM_THREADS):
        print(rnd_idxs[i, :8])
        print(rnd_idxs[i, -8:])
        print(min(rnd_idxs[i,:]), max(rnd_idxs[i,:]))

    np.savetxt(fname=os.path.join(OUTPUT_DIR, 'start_indices_4k.csv'), X=np.transpose(rnd_idxs), fmt='%d', delimiter=',')

# Helper functions for feature generation.  These were sourced from a Kaggle kernel script (Preda, 2019).  The "sta_lta" refers to the short term average divided by the long term average.  The trend feature is a linear regression on a portion of the signal.

# In[ ]:


def add_trend_feature(arr, abs_values=False):
    idx = np.array(range(len(arr)))
    if abs_values:
        arr = np.abs(arr)
    lr = LinearRegression()
    lr.fit(idx.reshape(-1, 1), arr)
    return lr.coef_[0]

def classic_sta_lta(x, length_sta, length_lta):
    sta = np.cumsum(x ** 2)
    # Convert to float
    sta = np.require(sta, dtype=np.float)
    # Copy for LTA
    lta = sta.copy()
    # Compute the STA and the LTA
    sta[length_sta:] = sta[length_sta:] - sta[:-length_sta]
    sta /= length_sta
    lta[length_lta:] = lta[length_lta:] - lta[:-length_lta]
    lta /= length_lta
    # Pad zeros
    sta[:length_lta - 1] = 0
    # Avoid division by zero by setting zero values to tiny float
    dtiny = np.finfo(0.0).tiny
    idx = lta < dtiny
    lta[idx] = dtiny
    return sta / lta

# Filter design helper functions.  These were added to allow for obtaining statistics on the signal in a bandwidth limited manner.  Butterworth 4 pole IIR filters are utilized to obtain the signal split into frequency bands.  EDA showed that most, if not all, of the signal above the 20,000 frequency line was likely to be noise, so the frequency bands will concentrate on the region below that.  Note that the signal is 150k lines long, hence by the Nyquist criteria there are 75k valid frequency lines before aliasing.

# In[ ]:


def des_bw_filter_lp(cutoff=CUTOFF):  # low pass filter
    b, a = sg.butter(4, Wn=cutoff/NY_FREQ_IDX)
    return b, a

def des_bw_filter_hp(cutoff=CUTOFF):  # high pass filter
    b, a = sg.butter(4, Wn=cutoff/NY_FREQ_IDX, btype='highpass')
    return b, a

def des_bw_filter_bp(low, high):  # band pass filter
    b, a = sg.butter(4, Wn=(low/NY_FREQ_IDX, high/NY_FREQ_IDX), btype='bandpass')
    return b, a

# The main function to create features. Inspired by script from Preda (2019) and Lukayenko (2019). Added frequency bandwidth limiting to the time domain features. Changes the Fourier transform to evaluate based on magnitude and phase and also to do so in a bandwidth-limited manner as compared to the reference scripts. This is based on the EDA where the magnitude of the Fourier transform looks important, but the phase response seems to be mostly noise.  WIndowed features were not subjected to the digital filters since the windowing is a type of filter.

# In[ ]:


def create_features(seg_id, seg, X, st, end):
    try:
        X.loc[seg_id, 'seg_id'] = np.int32(seg_id)
        X.loc[seg_id, 'seg_start'] = np.int32(st)
        X.loc[seg_id, 'seg_end'] = np.int32(end)
    except:
        pass

    xc = pd.Series(seg['acoustic_data'].values)
    xcdm = xc - np.mean(xc)

    b, a = des_bw_filter_lp(cutoff=18000)
    xcz = sg.lfilter(b, a, xcdm)

    zc = np.fft.fft(xcz)
    zc = zc[:MAX_FREQ_IDX]

    # FFT transform values
    realFFT = np.real(zc)
    imagFFT = np.imag(zc)

    freq_bands = [x for x in range(0, MAX_FREQ_IDX, FREQ_STEP)]
    magFFT = np.sqrt(realFFT ** 2 + imagFFT ** 2)
    phzFFT = np.arctan(imagFFT / realFFT)
    phzFFT[phzFFT == -np.inf] = -np.pi / 2.0
    phzFFT[phzFFT == np.inf] = np.pi / 2.0
    phzFFT = np.nan_to_num(phzFFT)

    for freq in freq_bands:
        X.loc[seg_id, 'FFT_Mag_01q%d' % freq] = np.quantile(magFFT[freq: freq + FREQ_STEP], 0.01)
        X.loc[seg_id, 'FFT_Mag_10q%d' % freq] = np.quantile(magFFT[freq: freq + FREQ_STEP], 0.1)
        X.loc[seg_id, 'FFT_Mag_90q%d' % freq] = np.quantile(magFFT[freq: freq + FREQ_STEP], 0.9)
        X.loc[seg_id, 'FFT_Mag_99q%d' % freq] = np.quantile(magFFT[freq: freq + FREQ_STEP], 0.99)
        X.loc[seg_id, 'FFT_Mag_mean%d' % freq] = np.mean(magFFT[freq: freq + FREQ_STEP])
        X.loc[seg_id, 'FFT_Mag_std%d' % freq] = np.std(magFFT[freq: freq + FREQ_STEP])
        X.loc[seg_id, 'FFT_Mag_max%d' % freq] = np.max(magFFT[freq: freq + FREQ_STEP])

        X.loc[seg_id, 'FFT_Phz_mean%d' % freq] = np.mean(phzFFT[freq: freq + FREQ_STEP])
        X.loc[seg_id, 'FFT_Phz_std%d' % freq] = np.std(phzFFT[freq: freq + FREQ_STEP])

    X.loc[seg_id, 'FFT_Rmean'] = realFFT.mean()
    X.loc[seg_id, 'FFT_Rstd'] = realFFT.std()
    X.loc[seg_id, 'FFT_Rmax'] = realFFT.max()
    X.loc[seg_id, 'FFT_Rmin'] = realFFT.min()
    X.loc[seg_id, 'FFT_Imean'] = imagFFT.mean()
    X.loc[seg_id, 'FFT_Istd'] = imagFFT.std()
    X.loc[seg_id, 'FFT_Imax'] = imagFFT.max()
    X.loc[seg_id, 'FFT_Imin'] = imagFFT.min()

    X.loc[seg_id, 'FFT_Rmean_first_6000'] = realFFT[:6000].mean()
    X.loc[seg_id, 'FFT_Rstd__first_6000'] = realFFT[:6000].std()
    X.loc[seg_id, 'FFT_Rmax_first_6000'] = realFFT[:6000].max()
    X.loc[seg_id, 'FFT_Rmin_first_6000'] = realFFT[:6000].min()
    X.loc[seg_id, 'FFT_Rmean_first_18000'] = realFFT[:18000].mean()
    X.loc[seg_id, 'FFT_Rstd_first_18000'] = realFFT[:18000].std()
    X.loc[seg_id, 'FFT_Rmax_first_18000'] = realFFT[:18000].max()
    X.loc[seg_id, 'FFT_Rmin_first_18000'] = realFFT[:18000].min()

    del xcz
    del zc

    b, a = des_bw_filter_lp(cutoff=2500)
    xc0 = sg.lfilter(b, a, xcdm)

    b, a = des_bw_filter_bp(low=2500, high=5000)
    xc1 = sg.lfilter(b, a, xcdm)

    b, a = des_bw_filter_bp(low=5000, high=7500)
    xc2 = sg.lfilter(b, a, xcdm)

    b, a = des_bw_filter_bp(low=7500, high=10000)
    xc3 = sg.lfilter(b, a, xcdm)

    b, a = des_bw_filter_bp(low=10000, high=12500)
    xc4 = sg.lfilter(b, a, xcdm)

    b, a = des_bw_filter_bp(low=12500, high=15000)
    xc5 = sg.lfilter(b, a, xcdm)

    b, a = des_bw_filter_bp(low=15000, high=17500)
    xc6 = sg.lfilter(b, a, xcdm)

    b, a = des_bw_filter_bp(low=17500, high=20000)
    xc7 = sg.lfilter(b, a, xcdm)

    b, a = des_bw_filter_hp(cutoff=20000)
    xc8 = sg.lfilter(b, a, xcdm)

    sigs = [xc, pd.Series(xc0), pd.Series(xc1), pd.Series(xc2), pd.Series(xc3),
            pd.Series(xc4), pd.Series(xc5), pd.Series(xc6), pd.Series(xc7), pd.Series(xc8)]

    for i, sig in enumerate(sigs):
        X.loc[seg_id, 'mean_%d' % i] = sig.mean()
        X.loc[seg_id, 'std_%d' % i] = sig.std()
        X.loc[seg_id, 'max_%d' % i] = sig.max()
        X.loc[seg_id, 'min_%d' % i] = sig.min()

        X.loc[seg_id, 'mean_change_abs_%d' % i] = np.mean(np.diff(sig))
        X.loc[seg_id, 'mean_change_rate_%d' % i] = np.mean(np.nonzero((np.diff(sig) / sig[:-1]))[0])
        X.loc[seg_id, 'abs_max_%d' % i] = np.abs(sig).max()
        X.loc[seg_id, 'abs_min_%d' % i] = np.abs(sig).min()

        X.loc[seg_id, 'std_first_50000_%d' % i] = sig[:50000].std()
        X.loc[seg_id, 'std_last_50000_%d' % i] = sig[-50000:].std()
        X.loc[seg_id, 'std_first_10000_%d' % i] = sig[:10000].std()
        X.loc[seg_id, 'std_last_10000_%d' % i] = sig[-10000:].std()

        X.loc[seg_id, 'avg_first_50000_%d' % i] = sig[:50000].mean()
        X.loc[seg_id, 'avg_last_50000_%d' % i] = sig[-50000:].mean()
        X.loc[seg_id, 'avg_first_10000_%d' % i] = sig[:10000].mean()
        X.loc[seg_id, 'avg_last_10000_%d' % i] = sig[-10000:].mean()

        X.loc[seg_id, 'min_first_50000_%d' % i] = sig[:50000].min()
        X.loc[seg_id, 'min_last_50000_%d' % i] = sig[-50000:].min()
        X.loc[seg_id, 'min_first_10000_%d' % i] = sig[:10000].min()
        X.loc[seg_id, 'min_last_10000_%d' % i] = sig[-10000:].min()

        X.loc[seg_id, 'max_first_50000_%d' % i] = sig[:50000].max()
        X.loc[seg_id, 'max_last_50000_%d' % i] = sig[-50000:].max()
        X.loc[seg_id, 'max_first_10000_%d' % i] = sig[:10000].max()
        X.loc[seg_id, 'max_last_10000_%d' % i] = sig[-10000:].max()

        X.loc[seg_id, 'max_to_min_%d' % i] = sig.max() / np.abs(sig.min())
        X.loc[seg_id, 'max_to_min_diff_%d' % i] = sig.max() - np.abs(sig.min())
        X.loc[seg_id, 'count_big_%d' % i] = len(sig[np.abs(sig) > 500])
        X.loc[seg_id, 'sum_%d' % i] = sig.sum()

        X.loc[seg_id, 'mean_change_rate_first_50000_%d' % i] = np.mean(np.nonzero((np.diff(sig[:50000]) / sig[:50000][:-1]))[0])
        X.loc[seg_id, 'mean_change_rate_last_50000_%d' % i] = np.mean(np.nonzero((np.diff(sig[-50000:]) / sig[-50000:][:-1]))[0])
        X.loc[seg_id, 'mean_change_rate_first_10000_%d' % i] = np.mean(np.nonzero((np.diff(sig[:10000]) / sig[:10000][:-1]))[0])
        X.loc[seg_id, 'mean_change_rate_last_10000_%d' % i] = np.mean(np.nonzero((np.diff(sig[-10000:]) / sig[-10000:][:-1]))[0])

        X.loc[seg_id, 'q95_%d' % i] = np.quantile(sig, 0.95)
        X.loc[seg_id, 'q99_%d' % i] = np.quantile(sig, 0.99)
        X.loc[seg_id, 'q05_%d' % i] = np.quantile(sig, 0.05)
        X.loc[seg_id, 'q01_%d' % i] = np.quantile(sig, 0.01)

        X.loc[seg_id, 'abs_q95_%d' % i] = np.quantile(np.abs(sig), 0.95)
        X.loc[seg_id, 'abs_q99_%d' % i] = np.quantile(np.abs(sig), 0.99)
        X.loc[seg_id, 'abs_q05_%d' % i] = np.quantile(np.abs(sig), 0.05)
        X.loc[seg_id, 'abs_q01_%d' % i] = np.quantile(np.abs(sig), 0.01)

        X.loc[seg_id, 'trend_%d' % i] = add_trend_feature(sig)
        X.loc[seg_id, 'abs_trend_%d' % i] = add_trend_feature(sig, abs_values=True)
        X.loc[seg_id, 'abs_mean_%d' % i] = np.abs(sig).mean()
        X.loc[seg_id, 'abs_std_%d' % i] = np.abs(sig).std()

        X.loc[seg_id, 'mad_%d' % i] = sig.mad()
        X.loc[seg_id, 'kurt_%d' % i] = sig.kurtosis()
        X.loc[seg_id, 'skew_%d' % i] = sig.skew()
        X.loc[seg_id, 'med_%d' % i] = sig.median()

        X.loc[seg_id, 'Hilbert_mean_%d' % i] = np.abs(hilbert(sig)).mean()
        X.loc[seg_id, 'Hann_window_mean'] = (convolve(xc, hann(150), mode='same') / sum(hann(150))).mean()

        X.loc[seg_id, 'classic_sta_lta1_mean_%d' % i] = classic_sta_lta(sig, 500, 10000).mean()
        X.loc[seg_id, 'classic_sta_lta2_mean_%d' % i] = classic_sta_lta(sig, 5000, 100000).mean()
        X.loc[seg_id, 'classic_sta_lta3_mean_%d' % i] = classic_sta_lta(sig, 3333, 6666).mean()
        X.loc[seg_id, 'classic_sta_lta4_mean_%d' % i] = classic_sta_lta(sig, 10000, 25000).mean()

        X.loc[seg_id, 'Moving_average_700_mean_%d' % i] = sig.rolling(window=700).mean().mean(skipna=True)
        X.loc[seg_id, 'Moving_average_1500_mean_%d' % i] = sig.rolling(window=1500).mean().mean(skipna=True)
        X.loc[seg_id, 'Moving_average_3000_mean_%d' % i] = sig.rolling(window=3000).mean().mean(skipna=True)
        X.loc[seg_id, 'Moving_average_6000_mean_%d' % i] = sig.rolling(window=6000).mean().mean(skipna=True)

        ewma = pd.Series.ewm
        X.loc[seg_id, 'exp_Moving_average_300_mean_%d' % i] = ewma(sig, span=300).mean().mean(skipna=True)
        X.loc[seg_id, 'exp_Moving_average_3000_mean_%d' % i] = ewma(sig, span=3000).mean().mean(skipna=True)
        X.loc[seg_id, 'exp_Moving_average_30000_mean_%d' % i] = ewma(sig, span=6000).mean().mean(skipna=True)

        no_of_std = 2
        X.loc[seg_id, 'MA_700MA_std_mean_%d' % i] = sig.rolling(window=700).std().mean()
        X.loc[seg_id, 'MA_700MA_BB_high_mean_%d' % i] = (
                    X.loc[seg_id, 'Moving_average_700_mean_%d' % i] + no_of_std * X.loc[seg_id, 'MA_700MA_std_mean_%d' % i]).mean()
        X.loc[seg_id, 'MA_700MA_BB_low_mean_%d' % i] = (
                    X.loc[seg_id, 'Moving_average_700_mean_%d' % i] - no_of_std * X.loc[seg_id, 'MA_700MA_std_mean_%d' % i]).mean()
        X.loc[seg_id, 'MA_400MA_std_mean_%d' % i] = sig.rolling(window=400).std().mean()
        X.loc[seg_id, 'MA_400MA_BB_high_mean_%d' % i] = (
                    X.loc[seg_id, 'Moving_average_700_mean_%d' % i] + no_of_std * X.loc[seg_id, 'MA_400MA_std_mean_%d' % i]).mean()
        X.loc[seg_id, 'MA_400MA_BB_low_mean_%d' % i] = (
                    X.loc[seg_id, 'Moving_average_700_mean_%d' % i] - no_of_std * X.loc[seg_id, 'MA_400MA_std_mean_%d' % i]).mean()
        X.loc[seg_id, 'MA_1000MA_std_mean_%d' % i] = sig.rolling(window=1000).std().mean()

        X.loc[seg_id, 'iqr_%d' % i] = np.subtract(*np.percentile(sig, [75, 25]))
        X.loc[seg_id, 'q999_%d' % i] = np.quantile(sig, 0.999)
        X.loc[seg_id, 'q001_%d' % i] = np.quantile(sig, 0.001)
        X.loc[seg_id, 'ave10_%d' % i] = stats.trim_mean(sig, 0.1)

    for windows in [10, 100, 1000]:
        x_roll_std = xc.rolling(windows).std().dropna().values
        x_roll_mean = xc.rolling(windows).mean().dropna().values

        X.loc[seg_id, 'ave_roll_std_' + str(windows)] = x_roll_std.mean()
        X.loc[seg_id, 'std_roll_std_' + str(windows)] = x_roll_std.std()
        X.loc[seg_id, 'max_roll_std_' + str(windows)] = x_roll_std.max()
        X.loc[seg_id, 'min_roll_std_' + str(windows)] = x_roll_std.min()
        X.loc[seg_id, 'q01_roll_std_' + str(windows)] = np.quantile(x_roll_std, 0.01)
        X.loc[seg_id, 'q05_roll_std_' + str(windows)] = np.quantile(x_roll_std, 0.05)
        X.loc[seg_id, 'q95_roll_std_' + str(windows)] = np.quantile(x_roll_std, 0.95)
        X.loc[seg_id, 'q99_roll_std_' + str(windows)] = np.quantile(x_roll_std, 0.99)
        X.loc[seg_id, 'av_change_abs_roll_std_' + str(windows)] = np.mean(np.diff(x_roll_std))
        X.loc[seg_id, 'av_change_rate_roll_std_' + str(windows)] = np.mean(
            np.nonzero((np.diff(x_roll_std) / x_roll_std[:-1]))[0])
        X.loc[seg_id, 'abs_max_roll_std_' + str(windows)] = np.abs(x_roll_std).max()

        X.loc[seg_id, 'ave_roll_mean_' + str(windows)] = x_roll_mean.mean()
        X.loc[seg_id, 'std_roll_mean_' + str(windows)] = x_roll_mean.std()
        X.loc[seg_id, 'max_roll_mean_' + str(windows)] = x_roll_mean.max()
        X.loc[seg_id, 'min_roll_mean_' + str(windows)] = x_roll_mean.min()
        X.loc[seg_id, 'q01_roll_mean_' + str(windows)] = np.quantile(x_roll_mean, 0.01)
        X.loc[seg_id, 'q05_roll_mean_' + str(windows)] = np.quantile(x_roll_mean, 0.05)
        X.loc[seg_id, 'q95_roll_mean_' + str(windows)] = np.quantile(x_roll_mean, 0.95)
        X.loc[seg_id, 'q99_roll_mean_' + str(windows)] = np.quantile(x_roll_mean, 0.99)
        X.loc[seg_id, 'av_change_abs_roll_mean_' + str(windows)] = np.mean(np.diff(x_roll_mean))
        X.loc[seg_id, 'av_change_rate_roll_mean_' + str(windows)] = np.mean(
            np.nonzero((np.diff(x_roll_mean) / x_roll_mean[:-1]))[0])
        X.loc[seg_id, 'abs_max_roll_mean_' + str(windows)] = np.abs(x_roll_mean).max()

    return X

# Manager function to build the feature fields that are extracted from the acoustic signal, for the training set only. The parameter "proc_id" is the multiprocessing identifier passed in by the multiprocessing caller. This allows for selection of the section of the overall data on which to work. Takes overnight to run 6 processes on the input data. If the "create_features_pk_det" function is called to obtain wavelet generated peak detection features, it may take three days to run. 

# In[ ]:


def build_fields(proc_id):
    success = 1
    count = 0
    try:
        seg_st = int(NUM_SEG_PER_PROC * proc_id)
        train_df = pd.read_csv(os.path.join(DATA_DIR, 'raw_data_%d.csv' % proc_id), dtype={'acoustic_data': np.int16, 'time_to_failure': np.float32})
        len_df = len(train_df.index)
        start_indices = (np.loadtxt(fname=os.path.join(OUTPUT_DIR, 'start_indices_4k.csv'), dtype=np.int32, delimiter=','))[:, proc_id]
        train_X = pd.DataFrame(dtype=np.float64)
        train_y = pd.DataFrame(dtype=np.float64, columns=['time_to_failure'])
        t0 = time.time()

        for seg_id, start_idx in zip(range(seg_st, seg_st + NUM_SEG_PER_PROC), start_indices):
            end_idx = np.int32(start_idx + 150000)
            print('working: %d, %d, %d to %d of %d' % (proc_id, seg_id, start_idx, end_idx, len_df))
            seg = train_df.iloc[start_idx: end_idx]
            # train_X = create_features_pk_det(seg_id, seg, train_X, start_idx, end_idx)
            train_X = create_features(seg_id, seg, train_X, start_idx, end_idx)
            train_y.loc[seg_id, 'time_to_failure'] = seg['time_to_failure'].values[-1]

            if count == 10: 
                print('saving: %d, %d to %d' % (seg_id, start_idx, end_idx))
                train_X.to_csv('train_x_%d.csv' % proc_id, index=False)
                train_y.to_csv('train_y_%d.csv' % proc_id, index=False)

            count += 1

        print('final_save, process id: %d, loop time: %.2f for %d iterations' % (proc_id, time.time() - t0, count))
        train_X.to_csv(os.path.join(OUTPUT_DIR, 'train_x_%d.csv' % proc_id), index=False)
        train_y.to_csv(os.path.join(OUTPUT_DIR, 'train_y_%d.csv' % proc_id), index=False)

    except:
        print(traceback.format_exc())
        success = 0

    return success  # 1 on success, 0 if fail

# Manager function to call the create features functions in multiple processes.  

# In[ ]:


def run_mp_build():
    t0 = time.time()
    num_proc = NUM_THREADS
    pool = mp.Pool(processes=num_proc)
    results = [pool.apply_async(build_fields, args=(pid, )) for pid in range(NUM_THREADS)]
    output = [p.get() for p in results]
    num_built = sum(output)
    pool.close()
    pool.join()
    print(num_built)
    print('Run time: %.2f' % (time.time() - t0))

# This function joins the results of the multiprocessing build into one training set for model building.  The output is a usable training set for both features and targets (the earthquake prediction times).

# In[ ]:


def join_mp_build():
    df0 = pd.read_csv(os.path.join(OUTPUT_DIR, 'train_x_%d.csv' % 0))
    df1 = pd.read_csv(os.path.join(OUTPUT_DIR, 'train_y_%d.csv' % 0))

    for i in range(1, NUM_THREADS):
        print('working %d' % i)
        temp = pd.read_csv(os.path.join(OUTPUT_DIR, 'train_x_%d.csv' % i))
        df0 = df0.append(temp)

        temp = pd.read_csv(os.path.join(OUTPUT_DIR, 'train_y_%d.csv' % i))
        df1 = df1.append(temp)

    df0.to_csv(os.path.join(OUTPUT_DIR, 'train_x.csv'), index=False)
    df1.to_csv(os.path.join(OUTPUT_DIR, 'train_y.csv'), index=False)

# Build features from the Kaggle test data files. This produces the test file that will be used for prediction and submission to Kaggle. If the "create_features_pk_det" function is called to obtain wavelet generated peak detection features, it may take two days to run. 

# In[ ]:


def build_test_fields():
    train_X = pd.read_csv(os.path.join(OUTPUT_DIR, 'train_x.csv'))
    try:
        train_X.drop(labels=['seg_id', 'seg_start', 'seg_end'], axis=1, inplace=True)
    except:
        pass

    submission = pd.read_csv(os.path.join(DATA_DIR, 'sample_submission.csv'), index_col='seg_id')
    test_X = pd.DataFrame(columns=train_X.columns, dtype=np.float64, index=submission.index)

    print('start for loop')
    count = 0
    for seg_id in tqdm_notebook(test_X.index):  # just tqdm in IDE
        seg = pd.read_csv(os.path.join(DATA_DIR, 'test', str(seg_id) + '.csv'))
        # train_X = create_features_pk_det(seg_id, seg, train_X, start_idx, end_idx)
        test_X = create_features(seg_id, seg, test_X, 0, 0)

        if count % 100 == 0:
            print('working', seg_id)
        count += 1

    test_X.to_csv(os.path.join(OUTPUT_DIR, 'test_x.csv'), index=False)

# Scale the features. This appeared to help, even with gradient boosted decision tree algorithms and is necessary with many other machine learning algorithms. 

# In[ ]:


def scale_fields(fn_train='train_x.csv', fn_test='test_x.csv', 
                 fn_out_train='scaled_train_X.csv' , fn_out_test='scaled_test_X.csv'):
    train_X = pd.read_csv(os.path.join(OUTPUT_DIR, fn_train))
    try:
        train_X.drop(labels=['seg_id', 'seg_start', 'seg_end'], axis=1, inplace=True)
    except:
        pass
    test_X = pd.read_csv(os.path.join(OUTPUT_DIR, fn_test))

    print('start scaler')
    scaler = StandardScaler()
    scaler.fit(train_X)
    scaled_train_X = pd.DataFrame(scaler.transform(train_X), columns=train_X.columns)
    scaled_test_X = pd.DataFrame(scaler.transform(test_X), columns=test_X.columns)

    scaled_train_X.to_csv(os.path.join(OUTPUT_DIR, fn_out_train), index=False)
    scaled_test_X.to_csv(os.path.join(OUTPUT_DIR, fn_out_test), index=False)

# Put the feature creation functions together and create the features. Some of these functions can take a long time to run, so it is recommended that it be done from an IDE and one function at a time. If it fails part way down due to a path name being wrong then it is not necessary to re-run every function. 

# In[ ]:


split_raw_data()
build_rnd_idxs()
run_mp_build()
join_mp_build()
build_test_fields()
scale_fields()

# do something like this in the IDE, call the functions above in order
# if __name__ == "__main__":
#     function name()
    

# ### Feature Creation using Wavelets

# Feature creation by using wavelets to extract peak value and index information from the signal was also explored.  Due to extremely high computational time, this was only performed for the 24,000 sample models.  This algorithm uses a 'Mexican Hat' wavelet in the SciPy library and by interference with Mexican Hat wavelets finds the peak and peak index locations for the signal.  These features may have has a very small beneficial effect upon the model and a significant number of these features were deemed statistically significant by the Pearson's correlation performed in the feature reduction section of the modeling.  A problem is that this algorithm is very computationally expensive.  Running 6 processes on 24,000 samples required 3 days to complete.  The test set (2624 samples) was run as a single process over two days.  While the features remain in the model, it is arguable that their benefit was not worth 5 days of compute time.  The function below can be called for either the training or test sets.

# In[ ]:


def create_features_pk_det(seg_id, seg, X, st, end):
    X.loc[seg_id, 'seg_id'] = np.int32(seg_id)
    X.loc[seg_id, 'seg_start'] = np.int32(st)
    X.loc[seg_id, 'seg_end'] = np.int32(end)

    sig = pd.Series(seg['acoustic_data'].values)
    b, a = des_bw_filter_lp(cutoff=18000)
    sig = sg.lfilter(b, a, sig)

    peakind = []
    noise_pct = .001
    count = 0

    while len(peakind) < 12 and count < 24:
        peakind = sg.find_peaks_cwt(sig, np.arange(1, 16), noise_perc=noise_pct, min_snr=4.0)
        noise_pct *= 2.0
        count += 1

    if len(peakind) < 12:
        print('Warning: Failed to find 12 peaks for %d' % seg_id)

    while len(peakind) < 12:
        peakind.append(149999)

    df_pk = pd.DataFrame(data={'pk': sig[peakind], 'idx': peakind}, columns=['pk', 'idx'])
    df_pk.sort_values(by='pk', ascending=False, inplace=True)

    for i in range(0, 12):
        X.loc[seg_id, 'pk_idx_%d' % i] = df_pk['idx'].iloc[i]
        X.loc[seg_id, 'pk_val_%d' % i] = df_pk['pk'].iloc[i]

    return X

# Function to restructure wavelet signal peak detection so that the peaks are ordered by index rather than peak value.  This may help the machine learning see the peaks in a more time ordered manner. 

# In[ ]:


import pandas as pd

df = pd.read_csv('test_x_8pk.csv')
df_out = None

for pks in df.itertuples():
    data = {'pk_idxs': [pks.pk_idx_0, pks.pk_idx_1, pks.pk_idx_2, pks.pk_idx_3, pks.pk_idx_4, pks.pk_idx_5, pks.pk_idx_6, pks.pk_idx_7, pks.pk_idx_8, pks.pk_idx_9, pks.pk_idx_10, pks.pk_idx_11],
            'pk_vals': [pks.pk_val_0, pks.pk_val_1, pks.pk_val_2, pks.pk_val_3, pks.pk_val_4, pks.pk_val_5, pks.pk_val_6, pks.pk_val_7, pks.pk_val_8, pks.pk_val_9, pks.pk_val_10, pks.pk_val_11]}
    pdf = pd.DataFrame(data=data)
    pdf.sort_values(by='pk_idxs', axis=0, inplace=True)

    data = {'pk_idx_0': pdf['pk_idxs'].iloc[0], 'pk_val_0': pdf['pk_vals'].iloc[0],
            'pk_idx_1': pdf['pk_idxs'].iloc[1], 'pk_val_1': pdf['pk_vals'].iloc[1],
            'pk_idx_2': pdf['pk_idxs'].iloc[2], 'pk_val_2': pdf['pk_vals'].iloc[2],
            'pk_idx_3': pdf['pk_idxs'].iloc[3], 'pk_val_3': pdf['pk_vals'].iloc[3],
            'pk_idx_4': pdf['pk_idxs'].iloc[4], 'pk_val_4': pdf['pk_vals'].iloc[4],
            'pk_idx_5': pdf['pk_idxs'].iloc[5], 'pk_val_5': pdf['pk_vals'].iloc[5],
            'pk_idx_6': pdf['pk_idxs'].iloc[6], 'pk_val_6': pdf['pk_vals'].iloc[6],
            'pk_idx_7': pdf['pk_idxs'].iloc[7], 'pk_val_7': pdf['pk_vals'].iloc[7],
            'pk_idx_8': pdf['pk_idxs'].iloc[8], 'pk_val_8': pdf['pk_vals'].iloc[8],
            'pk_idx_9': pdf['pk_idxs'].iloc[9], 'pk_val_9': pdf['pk_vals'].iloc[9],
            'pk_idx_10': pdf['pk_idxs'].iloc[10], 'pk_val_10': pdf['pk_vals'].iloc[10],
            'pk_idx_11': pdf['pk_idxs'].iloc[11], 'pk_val_11': pdf['pk_vals'].iloc[11]}

    if df_out is None:
        df_out = pd.DataFrame(data=data, index=[0])
    else:
        temp = pd.DataFrame(data=data, index=[0])
        df_out = df_out.append(temp, ignore_index=True)

df_out = df_out[['pk_idx_0', 'pk_val_0',
                   'pk_idx_1', 'pk_val_1',
                   'pk_idx_2', 'pk_val_2',
                   'pk_idx_3', 'pk_val_3',
                   'pk_idx_4', 'pk_val_4',
                   'pk_idx_5', 'pk_val_5',
                   'pk_idx_6', 'pk_val_6',
                   'pk_idx_7', 'pk_val_7',
                   'pk_idx_8', 'pk_val_8',
                   'pk_idx_9', 'pk_val_9',
                   'pk_idx_10', 'pk_val_10',
                   'pk_idx_11', 'pk_val_11']]
print(df_out.head())
print(df_out.tail())
df_out.to_csv('test_x_8pk_by_idx.csv')

# Function to add a slope value that adds a slope representing the peak vs its distance from ths signal end. When done, this provided 20 features that passed a p-value test with a threshold of at or below 0.05. The indices did not survive the process, but mostly the peak values and some slope values appear to have some merit. 

# In[ ]:


import numpy as np
import pandas as pd

pk_idx_base = 'pk_idx_'
pk_val_base = 'pk_val_'

print('do train')
df = pd.read_csv(r'pk8/train_x_8pk.csv')
slopes = np.zeros((len(df.index), 12))

for i in df.index:
    for j in range(12):
        pk_idx = pk_idx_base + str(j)
        pk_val = pk_val_base + str(j)
        slopes[i, j] = df[pk_val].iloc[i] / (150000 - df[pk_idx].iloc[i])

for j in range(12):
    df['slope_' + str(j)] = slopes[:, j]

print(df.head())
df.to_csv(r'pk8/train_x_8_slope.csv', index=False)

df = pd.read_csv(r'pk8/test_x_8pk.csv')
slopes = np.zeros((len(df.index), 12))

print('do test')
for i in df.index:
    for j in range(12):
        pk_idx = pk_idx_base + str(j)
        pk_val = pk_val_base + str(j)
        slopes[i, j] = df[pk_val].iloc[i] / (150000 - df[pk_idx].iloc[i])

for j in range(12):
    df['slope_' + str(j)] = slopes[:, j]

print(df.head())
df.to_csv(r'pk8/test_x_8_slope.csv', index=False)

print('!DONE!')

# ### Models

# Run a LightGBM model and save for a submission to Kaggle.  This will also output feature importance.  This model scored 1.441 on Kaggle.  For this and the models that follow, remember to adjust the number of jobs(treads or processes) based on the CPU capabilities available.  As noted above, the feature importance from the LightGBM model was abandoned as a feature selection mechanism in favor of Pearson's correlation.

# In[ ]:


params = {'num_leaves': 21,
         'min_data_in_leaf': 20,
         'objective':'regression',
         'learning_rate': 0.001,
         'max_depth': 108,
         "boosting": "gbdt",
         "feature_fraction": 0.91,
         "bagging_freq": 1,
         "bagging_fraction": 0.91,
         "bagging_seed": 42,
         "metric": 'mae',
         "lambda_l1": 0.1,
         "verbosity": -1,
         "random_state": 42}


def lgb_base_model():
    maes = []
    rmses = []
    submission = pd.read_csv(os.path.join(DATA_DIR, 'sample_submission.csv'), index_col='seg_id')
    scaled_train_X = pd.read_csv(r'train_8_and_9\scaled_train_X_8.csv')
    scaled_test_X = pd.read_csv(r'train_8_and_9\scaled_test_X_8.csv')
    train_y = pd.read_csv(r'train_8_and_9\train_y_8.csv')
    predictions = np.zeros(len(scaled_test_X))

    n_fold = 8
    folds = KFold(n_splits=n_fold, shuffle=True, random_state=42)

    fold_importance_df = pd.DataFrame()
    fold_importance_df["Feature"] = scaled_train_X.columns

    for fold_, (trn_idx, val_idx) in enumerate(folds.split(scaled_train_X, train_y.values)):
        print('working fold %d' % fold_)
        strLog = "fold {}".format(fold_)
        print(strLog)

        X_tr, X_val = scaled_train_X.iloc[trn_idx], scaled_train_X.iloc[val_idx]
        y_tr, y_val = train_y.iloc[trn_idx], train_y.iloc[val_idx]

        model = lgb.LGBMRegressor(**params, n_estimators=80000, n_jobs=-1)
        model.fit(X_tr, y_tr,
                  eval_set=[(X_tr, y_tr), (X_val, y_val)], eval_metric='mae',
                  verbose=1000, early_stopping_rounds=200)

        # predictions
        preds = model.predict(scaled_test_X, num_iteration=model.best_iteration_)
        predictions += preds / folds.n_splits
        preds = model.predict(X_val, num_iteration=model.best_iteration_)

        # mean absolute error
        mae = mean_absolute_error(y_val, preds)
        print('MAE: %.6f' % mae)
        maes.append(mae)

        # root mean squared error
        rmse = mean_squared_error(y_val, preds)
        print('RMSE: %.6f' % rmse)
        rmses.append(rmse)

        fold_importance_df['importance_%d' % fold_] = model.feature_importances_[:len(scaled_train_X.columns)]

    print('MAEs', maes)
    print('MAE mean: %.6f' % np.mean(maes))
    print('RMSEs', rmses)
    print('RMSE mean: %.6f' % np.mean(rmses))

    submission.time_to_failure = predictions
    submission.to_csv('submission_lgb_8_80k_108dp.csv', index=False)
    fold_importance_df.to_csv('fold_imp_lgb_8_80k_108dp.csv')  # index needed, it is seg id

# do this in the IDE, call the function
# if __name__ == "__main__":
#     lgb_base_model()

# This is the variant of the model with feature elimination performed by Pearson's correlation.  As noted below, these models usually scored higher individual scores on the Kaggle leader board.

# In[ ]:


params = {'num_leaves': 21,
         'min_data_in_leaf': 20,
         'objective':'regression',
         'max_depth': 108,
         'learning_rate': 0.001,
         "boosting": "gbdt",
         "feature_fraction": 0.91,
         "bagging_freq": 1,
         "bagging_fraction": 0.91,
         "bagging_seed": 42,
         "metric": 'mae',
         "lambda_l1": 0.1,
         "verbosity": -1,
         "random_state": 42}


def lgb_trimmed_model():
    maes = []
    rmses = []
    tr_maes = []
    tr_rmses = []
    submission = pd.read_csv(os.path.join(DATA_DIR, 'sample_submission.csv'), index_col='seg_id')

    scaled_train_X = pd.read_csv(r'pk8/scaled_train_X_8.csv')
    df = pd.read_csv(r'pk8/scaled_train_X_8_slope.csv')
    scaled_train_X = scaled_train_X.join(df)

    scaled_test_X = pd.read_csv(r'pk8/scaled_test_X_8.csv')
    df = pd.read_csv(r'pk8/scaled_test_X_8_slope.csv')
    scaled_test_X = scaled_test_X.join(df)

    pcol = []
    pcor = []
    pval = []
    y = pd.read_csv(r'pk8/train_y_8.csv')['time_to_failure'].values

    for col in scaled_train_X.columns:
        pcol.append(col)
        pcor.append(abs(pearsonr(scaled_train_X[col], y)[0]))
        pval.append(abs(pearsonr(scaled_train_X[col], y)[1]))

    df = pd.DataFrame(data={'col': pcol, 'cor': pcor, 'pval': pval}, index=range(len(pcol)))
    df.sort_values(by=['cor', 'pval'], inplace=True)
    df.dropna(inplace=True)
    df = df.loc[df['pval'] <= 0.05]

    drop_cols = []

    for col in scaled_train_X.columns:
        if col not in df['col'].tolist():
            drop_cols.append(col)

    scaled_train_X.drop(labels=drop_cols, axis=1, inplace=True)
    scaled_test_X.drop(labels=drop_cols, axis=1, inplace=True)

    train_y = pd.read_csv(r'pk8/train_y_8.csv')
    predictions = np.zeros(len(scaled_test_X))
    preds_train = np.zeros(len(scaled_train_X))

    print('shapes of train and test:', scaled_train_X.shape, scaled_test_X.shape)

    n_fold = 6
    folds = KFold(n_splits=n_fold, shuffle=False, random_state=42)

    for fold_, (trn_idx, val_idx) in enumerate(folds.split(scaled_train_X, train_y.values)):
        print('working fold %d' % fold_)
        strLog = "fold {}".format(fold_)
        print(strLog)

        X_tr, X_val = scaled_train_X.iloc[trn_idx], scaled_train_X.iloc[val_idx]
        y_tr, y_val = train_y.iloc[trn_idx], train_y.iloc[val_idx]

        model = lgb.LGBMRegressor(**params, n_estimators=60000, n_jobs=-1)
        model.fit(X_tr, y_tr,
                  eval_set=[(X_tr, y_tr), (X_val, y_val)], eval_metric='mae',
                  verbose=1000, early_stopping_rounds=200)

        # model = xgb.XGBRegressor(n_estimators=1000,
        #                                learning_rate=0.1,
        #                                max_depth=6,
        #                                subsample=0.9,
        #                                colsample_bytree=0.67,
        #                                reg_lambda=1.0, # seems best within 0.5 of 2.0
        #                                # gamma=1,
        #                                random_state=777+fold_,
        #                                n_jobs=12,
        #                                verbosity=2)
        # model.fit(X_tr, y_tr)

        # predictions
        preds = model.predict(scaled_test_X)  #, num_iteration=model.best_iteration_)
        predictions += preds / folds.n_splits
        preds = model.predict(scaled_train_X)  #, num_iteration=model.best_iteration_)
        preds_train += preds / folds.n_splits

        preds = model.predict(X_val)  #, num_iteration=model.best_iteration_)

        # mean absolute error
        mae = mean_absolute_error(y_val, preds)
        print('MAE: %.6f' % mae)
        maes.append(mae)

        # root mean squared error
        rmse = mean_squared_error(y_val, preds)
        print('RMSE: %.6f' % rmse)
        rmses.append(rmse)

        # training for over fit
        preds = model.predict(X_tr)  #, num_iteration=model.best_iteration_)

        mae = mean_absolute_error(y_tr, preds)
        print('Tr MAE: %.6f' % mae)
        tr_maes.append(mae)

        rmse = mean_squared_error(y_tr, preds)
        print('Tr RMSE: %.6f' % rmse)
        tr_rmses.append(rmse)

    print('MAEs', maes)
    print('MAE mean: %.6f' % np.mean(maes))
    print('RMSEs', rmses)
    print('RMSE mean: %.6f' % np.mean(rmses))

    print('Tr MAEs', tr_maes)
    print('Tr MAE mean: %.6f' % np.mean(tr_maes))
    print('Tr RMSEs', rmses)
    print('Tr RMSE mean: %.6f' % np.mean(tr_rmses))

    submission.time_to_failure = predictions
    submission.to_csv('submission_xgb_slope_pearson_6fold.csv')  # index needed, it is seg id

    pr_tr = pd.DataFrame(data=preds_train, columns=['time_to_failure'], index=range(0, preds_train.shape[0]))
    pr_tr.to_csv(r'preds_tr_xgb_slope_pearson_6fold.csv', index=False)
    print('Train shape: {}, Test shape: {}, Y shape: {}'.format(scaled_train_X.shape, scaled_test_X.shape, train_y.shape))
 
# do this in the IDE, call the function above
# if __name__ == "__main__":
#     lgb_trimmed_model()

# ### Feature Selection

# Early on feature selection was performed via feature ranking output from a LightGBM model.  Removing some 150 features by this method provided a very tiny increase (0.001 MAE) in the Kaggle public leader board score.  However, it was difficult to know where to set a threshold for feature removal due to their being few obvious cut points in the feature scores.  More success was achieved by calculating the Pearson's correlation of the features with the target time-to-failure.  The Scipy "pearsonr" function provides a p-value that takes account of the sample size of the model.  Since statisticians generally consider p-values below 0.05 as representing significance, this value was chosen for the model's feature reduction algorithm. Scipy considers this p-value to be reasonably reliable for sample sizes above 500, which clearly is true for the models presented here (Scipy, 2019).

# Feature reduction via Pearson's correlation appears to have had a moderate beneficial effect upon most of the individual models as evidenced by Kaggle public leader board scores for equivalent individual models.  For example, a LightGBM model with 8-fold random cross validation improved from an MAE of 1.439 to 1.434.  An XGBoost model improved from 1.467 to 1.440 under similar conditions.  A 6-fold XGBoost model, where the folds did not overlap in the training signal time domain, improved from 1.472 to 1.437.  In the table below, only one model fell in Kaggle scoring when the feature set was reduced and the divergence was only 0.001 in MAE.

# <a href="https://ibb.co/sqWr8Wv"><img src="https://i.ibb.co/4p1xh1Z/pearson.png" alt="pearson" border="0" /></a>

# ### Individual Model Cross Validation Results

# Individual model cross validation (CV) results are shown below, both for the models presented in this project and two reference models taken from the Kaggle Kernels.  These two outside models are the ones noted by Preda (2019) and Scirpus (2019).  Kaggle leader board values are those obtained by this author in testing.  The Preda (2019) model is his LightGBM single model as run by this author.  The Scirpus (2019) genetic programming model result is also as obtained by this author in testing, and agrees with the Kaggle leader board result reported for this script by its original author at the time the script was run by this author. 

# Both Preda (20190 and Scirpus (2019) scripts report lower Kaggle public leader board scores than their CV scores. Examination of other scripts on the Kaggle kernels section leads this author to believe that this is typical for scripts where the data rows were "bread sliced" from the original acoustic signal and the 4914 resulting data rows do not overlap or leak information.

# Random CV row selection on 24k rows of data where the rows do overlap in the original signal changes the picture significantly from CV results reported for the above.  Instead of the Kaggle public leader board score being better than the CV score, now the CV score is very low for random CV sampling. This is probably caused by information leakage between the samples because they are derived from signals that overlap in time.  In spite of the information leakage, and possible overfitting, this method produced the best individual model Kaggle public leader board score in this report of 1.434 using Light GBM.   

# Using 6 fold cross validation with slices that do not overlap eliminates the leakage because of the splitting of the signal into 6 segments before the random selection was performed. Actually, in this model there is very slight leakage because this author was not fully cognizant of these possible effects and allowed a small 150k sample overlap between the 6 segments. Because this 150k sample overlap is so small compared to the 100 million plus samples in each 1/6th slice of the signal, leakage is effectively negligible. This changes the CV relationship back to that of the 4194 sample models reported, and again the CV score is much worse than the Kaggle leader board score. Please see the table below for examples of CV and public leader board score.

# <a href="https://ibb.co/yRf6H8p"><img src="https://i.ibb.co/W3xkmc0/model-summary.png" alt="model-summary" border="0" /></a>

# The data above may be too small to make many observations regarding the relationship between CV score and Kaggle public leader board results. It seems though that CV score is not a good predictor of the eventual Kaggle public leader board score and this has caused significant challenges throughout the project, especially with hyperparameter tuning. Note, for example, that this author's XGBoost model with essentially non-overlapped cross validation had a CV score of 2.253 average. The Preda (2019) reference script had a CV of 2.082 average using LightGBM. Yet, the XGBoost model in question had a Kaggle public leader board MAE of 1.437, better than obtained for the Preda (2019) LightGBM model. It appears that for this challenge problem the CV score is not a reliable predictor of public leader board score, at least for the small current public leader board test data sample. It is currently unknown whether this will change when the full private leader board is revealed at the end of the competition. 

# ### Hyperparameter Tuning

# Effective hyperparameter tuning proved to be a very large challenge in this project.  For much of the semester the author worked with the random cross validation strategy.  These models required 6 hours to train with LightGBM, and 30 minutes with XGBoost.  Semester time constraints made tuning efforts difficult as it would have required too many days to perform effective grid searches on the problem.  Realization that a sectionalized cross validation was also practical shortened LightGBM training times because the model eventually reached a point where it stopped improving on the validation data.  LightGBM training times then became almost identical to those of XGBoost.  This was actually not true for the randomly sampled CV model with Light GBM.  No final stopping point was ever found for this model.  While 60k estimators was eventually chosen from experience, the model would appear to continue to train up to 100k estimators or beyond.  

# Because of limited time, and observed overfitting, hype parameter tuning was performed by theoretical changes in directions that might reduce overfitting. For example, "colsample_bytree" in XGBoost was set to 0.667 where only 2/3 of the columns are selected for a split at each tree level was chosen because the documentation for XGBoost indicates that this helps with overfitting. Similarly, the number of leaves was decreased and the minimum data in a leaf increased for the same reason in LightGBM. It would have been helpful to try more hyperparameter tuning. Kaggle limits submission models to two per day and this proved to become a limitation for experimentation as the project due date approached. Hyperparamater tuning was particularly difficult because of the disconnect between leader board score and cross validation scores. Hyperparameter tuning was therefore less extensive than would be desirable. 

# ### Model Stacking

# Several issues affect possible model stacking given the state of this project.  First, there are four models with 24,000 data rows that have performed well on the Kaggle leader board.  These are the LightGBM and XGBoost models, both run with substantially different cross validation methods.  Best public leader board scores for LightGBM is 1.434 and 1.437 for the XGBoost models.  The difference does not appear to be significant and might change if further parameter tuning were performed.  Also there is the Scirpus (2019) script to consider, based upon 4194 sample rows.  Because it uses genetic programming rather than the decision trees used by LightGBM and XGBoost, it offers possible diversity to the model.  

# A model stack built by simple averaging was submitted to the Kaggle leader board for scoring using the two best models by this author plus output from the Scirpus (2019) script.  By combining the models a score of 1.392 was achieved.  At the time of submission this was good for the top 1% of 3200 plus competitors.  This Kaggle competition comes with cash prizes and this attracts many fine competitors, so it will not be surprising that this result will fall as more entrants submit models.  Keeping up will probably require new breakthroughs.  The Kaggle submission shown here was made on April 28th, 2019.

# <a href="https://ibb.co/PNcLWK3"><img src="https://i.ibb.co/FK5rY2N/Kaggle-Placing-28-Apr.png" alt="Kaggle-Placing-28-Apr" border="0" /></a>

# ### Lessons Learned

# Several modeling types that this author had no previous experience with were tried on the 24k row data features.  These were CatBoost (Prokhorenkova, Gusev, Vorobev, Dorogush, & Gulin, 2017) and genetic programming via the gplearn library (Stephens, 2016).  Both suffered from long training times in this scenario and CV scores that were not encouraging.  The author's inexperience with both of these algorithms appear to be the primary culprit.  Also tried were Random Forest (Brieman, 2001) decision tree-based algorithm and a model based on the Keras/Tenorflow Deep learning library (Chollet, et al., 2015) (Mart, et al., 2015).  While Keras and Tensorflow work very well on speech and vision applications, it does not to this author apppear fully competitive with the best tree-based gradient boosting models in a regression problem.  The Random Forest also did not perform as well as LightGBM and XGBoost on the feature generation set presented here. 

# Also tried was increasing the number of data rows to 40,000.  This resulted in worse overfitting and a lower CV score.  Because this made a computationally intensive approach to the problem even more computationally difficult, this effort was quickly dropped.

# Long run times for many functions and algorithms made the project more of a challenge than it otherwise might have been.  It was fortunate to have two reasonably powerful computers available for much of the project.  This allowed some parallel development to take place and helped when there were more ideas available than CPU power to investigate them.  It would be easy to keep more computers busy on this project and if it were to be repeated I would try to locate more resources.  When scripts take overnight or even days to run having more computers available is clearly advantageous and allows trying more ideas on the project. 

# ### Future Research Possibilities

# Principal components analysis (PCA) appears to be worth trying, especially if one were to apply the mass feature generation used here on 4194 sample data. This possibly could help with the "large p, small n" issue that might arise if a model with only 4194 training rows was tried and 900 features obtained from splitting the signal up with digital filters. Or, one could continue to use the Pearson's approach to feature generation and experiment with various cutoff values.

# Another area of exploration is that the frequency bands used to create additional features were selected somewhat arbitrarily except for the understanding of the general frequency range desired that was obtained in the EDA by Fourier analysis. Alternate choices for the width and number of frequency bands have not been investigated and might prove worthwhile.

# CatBoost, given its good reputation and being a modern gradient boosting machine, is also worth further study. The author did not have time to fully investigate it and may have been hampered by a lack of experience with the algorithm. It is probably not worthwhile to spend time on Support Vector Machines and Nearest Neighbors algorithms, in other regression models within the author's experience these seem antiquated and appear to under perform newer gradient boosting decision tree based methods such as LightGBM or XGBoost.

# ### Conclusions

# It might be tempting to build and evaluate models based strictly on the training data in projects outside of the Kaggle competition.  One should be cautious in doing so, it has proven all to easy to drastically overfit the training data and trials too numerous to document here have resulted in superb CV scores but lessened Kaggle leader board results. Even the current Kaggle public leader board result obtained is suspect because of the small amount of the test data that it contains, but a better test set does not appear to be available at present.  After the competition is over and the full test set is made available for scoring, this problem would be resolved for non-contest entries.

# Building an effective model for a Kaggle challenge where the data is noisy and the leader board that utilizes only 13% of the potential test data is a significant challenge.  This is a problem where it is difficult to score well.  In terms of obtaining a good leader board score and to hopefully generalize well to the full leader board, 3 models were averaged here and submitted to Kaggle.  It is hoped that the diversity of model types and feature generation will help to stabilize the predictions submitted to Kaggle such that the model does generalize well when the full leader board is revealed at the end of the competition and after this university course has completed.

# ### Acknowledgements

# I feel compelled to note again the contributions of the Preda (2019), Lukayenko (2019) and Scirpus (2019) scripts to this work. Without them and their predecessor kernel scripts on Kaggle, any progress made by this effort would have been far more difficult. The predecessor scripts can be found from citation links in these scripts and full links are referenced below.  

# ### Test Environment

# In[ ]:


import sys
import scipy
import sklearn
print(sys.version)
print('pandas:', pd.__version__)
print('numpy:', np.__version__)
print('scipy:', scipy.__version__)
print('sklearn:', sklearn.__version__)
print('light gbm:', lgb.__version__)
print('xgboost:', xgb.__version__)

# ### Author and License Information

# Kevin Maher  
# Email: Vettejeep365@gmail.com  
# Upvotes and/or github stars appreciated!  
# This code herein has been released under the
# <a href="http://www.apache.org/licenses/LICENSE-2.0"><span style='color:#337AB7;text-decoration:
# none;text-underline:none'>Apache 2.0
# </span></a> open source license.  
# The author please requests a citation for the use or derivation of this work.

# ### References

# Bilogur, A. Model Fit Metrics (undated).  <i>Kaggle</i>.  Retrieved from: https://www.kaggle.com/residentmario/model-fit-metrics 
# 
# Brieman, L. (2001, January).  Random Forests.  <i>Machine Learning</i>, 45(1), 5–32.
# 
# Chen, T., & Guestrin C. (2016).  XGBoost: A scalable tree boosting system. <i>In Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, KDD ’16</i>, 785–794, New York, NY, USA.
# 
# Chollet, F., et al (2015).  Keras: The Python Deep Learning library.  <i>Keras</i>.  Retrieved from: https://keras.io/
# 
# Cook, J. (2016).  Big p, little n.  <i>John D. Cook consulting</i>.  Retrieved from: https://www.johndcook.com/blog/2016/01/07/big-p-little-n/ 
# 
# Demir, N., PhD. (undated).  Ensemble Methods: Elegant Techniques to Produce Improved Machine Learning Results.  <i>Toptal</i>.  Retrieved from: https://www.toptal.com/machine-learning/ensemble-methods-machine-learning 
# 
# Jones E., et al (2001). SciPy: Open Source Scientific Tools for Python, Retrieved from: http://www.scipy.org/ 
# 
# Ke, G., Meng, Q., Findlay, T., Wang, T., Chen, W., Ma, W., … Liu, T. (2017, December).  LightGBM: A Highly Efficient Gradient Boosting Decision Tree.  In Guyon, I. & von Luxburg, U. (General Chairs),  <i>Thirty-first Conference on Neural Information Processing Systems (NIPS 2017)</i>.  Long Beach, CA.  Retrieved from: http://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree 
# 
# Prokhorenkova, L., Gusev, G., Vorobev, A., Dorogush, A., & Gulin, A. (2017).  CatBoost: unbiased boosting with categorical features.  <i>Cornell University</i>.  Retrieved from: https://arxiv.org/abs/1706.09516
# 
# Lukayenko, A. (2019).  Earthquakes FE. More features and samples.  <i>Kaggle</i>.  Retrieved from: https://www.kaggle.com/artgor/earthquakes-fe-more-features-and-samples  
# 
# Mart, A., et al (2015).  Large-Scale Machine Learning on Heterogeneous Systems. <i>tensorflow.org</i>.  Retrieved from: https://chromium.googlesource.com/external/github.com/tensorflow/tensorflow/+/0.6.0/tensorflow/g3doc/index.md  
# 
# McKinney, W., (2010). Data Structures for Statistical Computing in Python.  <i>Proceedings of the 9th Python in Science Conference</i>, 51-56.
# 
# Preda, G (2019).  LANL Earthquake EDA and Prediction.  <i>Kaggle</i>.  Retrieved from: https://www.kaggle.com/gpreda/lanl-earthquake-eda-and-prediction 
# 
# Pedregosa et al. (2011).  Scikit-learn: Machine Learning in Python.  <i>Journal of Machine Learning Research</i>.  Retrieved from: https://scikit-learn.org/stable/ 
# 
# Pérez, F. & Granger, B., (2007). IPython: A System for Interactive Scientific Computing.  <i>Computing in Science & Engineering</i>, 9, 21-29, DOI:10.1109/MCSE.2007.53
# 
# Rouet-Leduc., et al (2019).  LANL Earthquake Prediction.  <i>Kaggle</i>.  Retrieved from: https://www.kaggle.com/c/LANL-Earthquake-Prediction
# 
# Scipy (2019).  scipy.stats.pearsonr.  SciPy.org.  Retrieved from: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html 
# 
# Scirpus (2019).  Andrews Script plus a Genetic Program Model.  <i>Kaggle</i>.  Retrieved from: https://www.kaggle.com/scirpus/andrews-script-plus-a-genetic-program-model/ 
# 
# Singhal, G., (2018).  Multiprocessing in Python on Windows and Jupyter/Ipython — Making it work.  <i>Medium</i>.  Retrieved from: https://medium.com/@grvsinghal/speed-up-your-python-code-using-multiprocessing-on-windows-and-jupyter-or-ipython-2714b49d6fac
# 
# Stevens, T., (2016). Genetic programming in Python, with a scikit-learn inspired API.  <i>gplearn</i>.  Retrieved from: https://gplearn.readthedocs.io/en/stable/ 
# 
# van der Walt, S., Colbert, S. & Varoquaux, G., (2011). The NumPy Array: A Structure for Efficient Numerical Computation.  <i>Computing in Science & Engineering</i>, 13, 22-30, DOI:10.1109/MCSE.2011.37.
# 
