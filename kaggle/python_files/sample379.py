#!/usr/bin/env python
# coding: utf-8

# ### This kernel is forked from [this wonderful kernel](https://www.kaggle.com/braquino/5-fold-lstm-attention-fully-commented-0-694) by Bruno Aquino. I have just stacked the recurrent layers together. Additionally, I have done some feature engineering based on time series entropy and fractal dimension. This achieves an LB score of 0.699.
# 
# Feature Engineering : [Raphael Vallat's entropy repository](https://github.com/raphaelvallat/entropy/blob/master/entropy/utils.py)
# 

# In[ ]:


import pandas as pd
import pyarrow.parquet as pq # Used to read the data
import os 
import numpy as np
from keras.layers import * # Keras is the most friendly Neural Network library, this Kernel use a lot of layers classes
from keras.models import *
from tqdm import tqdm # Processing time measurement
from sklearn.model_selection import train_test_split 
from keras import backend as K # The backend give us access to tensorflow operations and allow us to create the Attention class
from keras import optimizers # Allow us to access the Adam class to modify some parameters
from sklearn.model_selection import GridSearchCV, StratifiedKFold # Used to use Kfold to train our model
from keras.callbacks import * # This object helps the model to train in a smarter way, avoiding overfitting
from keras import activations
from keras import regularizers
from keras import initializers
from keras import constraints
from sklearn.preprocessing import MinMaxScaler
from numba import jit
from math import log, floor
from sklearn.neighbors import KDTree
from scipy.signal import periodogram, welch
from keras.engine import Layer
from keras.engine import InputSpec
from keras.objectives import categorical_crossentropy
from keras.objectives import sparse_categorical_crossentropy
# import gplearn
# from gplearn.genetic import SymbolicTransformer
# from gplearn.functions import make_function

# In[ ]:


# select how many folds will be created
N_SPLITS = 5
# it is just a constant with the measurements data size
sample_size = 800000

# In[ ]:


# It is the official metric used in this competition
# below is the declaration of a function used inside the keras model, calculation with K (keras backend / thensorflow)
def matthews_correlation(y_true, y_pred):
    '''Calculates the Matthews correlation coefficient measure for quality
    of binary classification problems.
    '''
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos

    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos

    tp = K.sum(y_pos * y_pred_pos)
    tn = K.sum(y_neg * y_pred_neg)

    fp = K.sum(y_neg * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg)

    numerator = (tp * tn - fp * fn)
    denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    return numerator / (denominator + K.epsilon())

# In[ ]:


# https://www.kaggle.com/suicaokhoailang/lstm-attention-baseline-0-652-lb

class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),
                        K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        if mask is not None:
            a *= K.cast(mask, K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0],  self.features_dim

# In[ ]:


# just load train data
df_train = pd.read_csv('../input/metadata_train.csv')
# set index, it makes the data access much faster
df_train = df_train.set_index(['id_measurement', 'phase'])
df_train.head()

# In[ ]:


# in other notebook I have extracted the min and max values from the train data, the measurements
max_num = 127
min_num = -128

# In[ ]:


# This function standardize the data from (-128 to 127) to (-1 to 1)
# Theoretically it helps in the NN Model training, but I didn't tested without it
def min_max_transf(ts, min_data, max_data, range_needed=(-1,1)):
    if min_data < 0:
        ts_std = (ts + abs(min_data)) / (max_data + abs(min_data))
    else:
        ts_std = (ts - min_data) / (max_data - min_data)
    if range_needed[0] < 0:    
        return ts_std * (range_needed[1] + abs(range_needed[0])) + range_needed[0]
    else:
        return ts_std * (range_needed[1] - range_needed[0]) + range_needed[0]

# In[ ]:


# This is one of the most important peace of code of this Kernel
# Any power line contain 3 phases of 800000 measurements, or 2.4 millions data 
# It would be praticaly impossible to build a NN with an input of that size
# The ideia here is to reduce it each phase to a matrix of <n_dim> bins by n features
# Each bean is a set of 5000 measurements (800000 / 160), so the features are extracted from this 5000 chunk data.
def transform_ts(ts, n_dim=160, min_max=(-1,1)):
    # convert data into -1 to 1
    ts_std = min_max_transf(ts, min_data=min_num, max_data=max_num)
    # bucket or chunk size, 5000 in this case (800000 / 160)
    bucket_size = int(sample_size / n_dim)
    # new_ts will be the container of the new data
    new_ts = []
    # this for iteract any chunk/bucket until reach the whole sample_size (800000)
    for i in range(0, sample_size, bucket_size):
        # cut each bucket to ts_range
        ts_range = ts_std[i:i + bucket_size]
        # calculate each feature
        mean = ts_range.mean()
        std = ts_range.std() # standard deviation
        std_top = mean + std # I have to test it more, but is is like a band
        std_bot = mean - std
        # I think that the percentiles are very important, it is like a distribuiton analysis from eath chunk
        percentil_calc = np.percentile(ts_range, [0, 1, 25, 50, 75, 99, 100]) 
        max_range = percentil_calc[-1] - percentil_calc[0] # this is the amplitude of the chunk
        relative_percentile = percentil_calc - mean # maybe it could heap to understand the asymmetry
        # now, we just add all the features to new_ts and convert it to np.array
        new_ts.append(np.concatenate([np.asarray([mean, std, std_top, std_bot, max_range]),percentil_calc, relative_percentile]))
    return np.asarray(new_ts)

# In[ ]:


# this function take a piece of data and convert using transform_ts(), but it does to each of the 3 phases
# if we would try to do in one time, could exceed the RAM Memmory
def prep_data(start, end):
    # load a piece of data from file
    praq_train = pq.read_pandas('../input/train.parquet', columns=[str(i) for i in range(start, end)]).to_pandas()
    X = []
    y = []
    # using tdqm to evaluate processing time
    # takes each index from df_train and iteract it from start to end
    # it is divided by 3 because for each id_measurement there are 3 id_signal, and the start/end parameters are id_signal
    for id_measurement in tqdm(df_train.index.levels[0].unique()[int(start/3):int(end/3)]):
        X_signal = []
        # for each phase of the signal
        for phase in [0,1,2]:
            # extract from df_train both signal_id and target to compose the new data sets
            signal_id, target = df_train.loc[id_measurement].loc[phase]
            # but just append the target one time, to not triplicate it
            if phase == 0:
                y.append(target)
            # extract and transform data into sets of features
            X_signal.append(transform_ts(praq_train[str(signal_id)]))
        # concatenate all the 3 phases in one matrix
        X_signal = np.concatenate(X_signal, axis=1)
        # add the data to X
        X.append(X_signal)
    X = np.asarray(X)
    y = np.asarray(y)
    return X, y

# In[ ]:


# this code is very simple, divide the total size of the df_train into two sets and process it
X = []
y = []
def load_all():
    total_size = len(df_train)
    for ini, end in [(0, int(total_size/2)), (int(total_size/2), total_size)]:
        X_temp, y_temp = prep_data(ini, end)
        X.append(X_temp)
        y.append(y_temp)
load_all()
X = np.concatenate(X)
y = np.concatenate(y)

# In[ ]:


def petrosian_fd(x):
    """Petrosian fractal dimension.
    Parameters
    ----------
    x : list or np.array
        One dimensional time series
    Returns
    -------
    pfd : float
        Petrosian fractal dimension
    Notes
    -----
    The Petrosian algorithm can be used to provide a fast computation of
    the FD of a signal by translating the series into a binary sequence.
    The Petrosian fractal dimension of a time series :math:`x` is defined by:
    .. math:: \dfrac{log_{10}(N)}{log_{10}(N) +
       log_{10}(\dfrac{N}{N+0.4N_{\Delta}})}
    where :math:`N` is the length of the time series, and
    :math:`N_{\Delta}` is the number of sign changes in the binary sequence.
    Original code from the pyrem package by Quentin Geissmann.
    References
    ----------
    .. [1] A. Petrosian, Kolmogorov complexity of finite sequences and
       recognition of different preictal EEG patterns, in , Proceedings of the
       Eighth IEEE Symposium on Computer-Based Medical Systems, 1995,
       pp. 212-217.
    .. [2] Goh, Cindy, et al. "Comparison of fractal dimension algorithms for
       the computation of EEG biomarkers for dementia." 2nd International
       Conference on Computational Intelligence in Medicine and Healthcare
       (CIMED2005). 2005.
    Examples
    --------
    Petrosian fractal dimension.
        >>> import numpy as np
        >>> from entropy import petrosian_fd
        >>> np.random.seed(123)
        >>> x = np.random.rand(100)
        >>> print(petrosian_fd(x))
            1.0505
    """
    n = len(x)
    # Number of sign changes in the first derivative of the signal
    diff = np.ediff1d(x)
    N_delta = (diff[1:-1] * diff[0:-2] < 0).sum()
    return np.log10(n) / (np.log10(n) + np.log10(n / (n + 0.4 * N_delta)))


def katz_fd(x):
    """Katz Fractal Dimension.
    Parameters
    ----------
    x : list or np.array
        One dimensional time series
    Returns
    -------
    kfd : float
        Katz fractal dimension
    Notes
    -----
    The Katz Fractal dimension is defined by:
    .. math:: FD_{Katz} = \dfrac{log_{10}(n)}{log_{10}(d/L)+log_{10}(n)}
    where :math:`L` is the total length of the time series and :math:`d`
    is the Euclidean distance between the first point in the
    series and the point that provides the furthest distance
    with respect to the first point.
    Original code from the mne-features package by Jean-Baptiste Schiratti
    and Alexandre Gramfort.
    References
    ----------
    .. [1] Esteller, R. et al. (2001). A comparison of waveform fractal
           dimension algorithms. IEEE Transactions on Circuits and Systems I:
           Fundamental Theory and Applications, 48(2), 177-183.
    .. [2] Goh, Cindy, et al. "Comparison of fractal dimension algorithms for
           the computation of EEG biomarkers for dementia." 2nd International
           Conference on Computational Intelligence in Medicine and Healthcare
           (CIMED2005). 2005.
    Examples
    --------
    Katz fractal dimension.
        >>> import numpy as np
        >>> from entropy import katz_fd
        >>> np.random.seed(123)
        >>> x = np.random.rand(100)
        >>> print(katz_fd(x))
            5.1214
    """
    x = np.array(x)
    dists = np.abs(np.ediff1d(x))
    ll = dists.sum()
    ln = np.log10(np.divide(ll, dists.mean()))
    aux_d = x - x[0]
    d = np.max(np.abs(aux_d[1:]))
    return np.divide(ln, np.add(ln, np.log10(np.divide(d, ll))))

@jit('UniTuple(float64, 2)(float64[:], float64[:])', nopython=True)
def _linear_regression(x, y):
    """Fast linear regression using Numba.
    Parameters
    ----------
    x, y : ndarray, shape (n_times,)
        Variables
    Returns
    -------
    slope : float
        Slope of 1D least-square regression.
    intercept : float
        Intercept
    """
    n_times = x.size
    sx2 = 0
    sx = 0
    sy = 0
    sxy = 0
    for j in range(n_times):
        sx2 += x[j] ** 2
        sx += x[j]
        sxy += x[j] * y[j]
        sy += y[j]
    den = n_times * sx2 - (sx ** 2)
    num = n_times * sxy - sx * sy
    slope = num / den
    intercept = np.mean(y) - slope * np.mean(x)
    return slope, intercept


@jit('i8[:](f8, f8, f8)', nopython=True)
def _log_n(min_n, max_n, factor):
    """
    Creates a list of integer values by successively multiplying a minimum
    value min_n by a factor > 1 until a maximum value max_n is reached.
    Used for detrended fluctuation analysis (DFA).
    Function taken from the nolds python package
    (https://github.com/CSchoel/nolds) by Christopher Scholzel.
    Parameters
    ----------
    min_n (float):
        minimum value (must be < max_n)
    max_n (float):
        maximum value (must be > min_n)
    factor (float):
       factor used to increase min_n (must be > 1)
    Returns
    -------
    list of integers:
        min_n, min_n * factor, min_n * factor^2, ... min_n * factor^i < max_n
        without duplicates
    """
    max_i = int(floor(log(1.0 * max_n / min_n) / log(factor)))
    ns = [min_n]
    for i in range(max_i + 1):
        n = int(floor(min_n * (factor ** i)))
        if n > ns[-1]:
            ns.append(n)
    return np.array(ns, dtype=np.int64)

@jit('float64(float64[:], int32)')
def _higuchi_fd(x, kmax):
    """Utility function for `higuchi_fd`.
    """
    n_times = x.size
    lk = np.empty(kmax)
    x_reg = np.empty(kmax)
    y_reg = np.empty(kmax)
    for k in range(1, kmax + 1):
        lm = np.empty((k,))
        for m in range(k):
            ll = 0
            n_max = floor((n_times - m - 1) / k)
            n_max = int(n_max)
            for j in range(1, n_max):
                ll += abs(x[m + j * k] - x[m + (j - 1) * k])
            ll /= k
            ll *= (n_times - 1) / (k * n_max)
            lm[m] = ll
        # Mean of lm
        m_lm = 0
        for m in range(k):
            m_lm += lm[m]
        m_lm /= k
        lk[k - 1] = m_lm
        x_reg[k - 1] = log(1. / k)
        y_reg[k - 1] = log(m_lm)
    higuchi, _ = _linear_regression(x_reg, y_reg)
    return higuchi


def higuchi_fd(x, kmax=10):
    """Higuchi Fractal Dimension.
    Parameters
    ----------
    x : list or np.array
        One dimensional time series
    kmax : int
        Maximum delay/offset (in number of samples).
    Returns
    -------
    hfd : float
        Higuchi Fractal Dimension
    Notes
    -----
    Original code from the mne-features package by Jean-Baptiste Schiratti
    and Alexandre Gramfort.
    The `higuchi_fd` function uses Numba to speed up the computation.
    References
    ----------
    .. [1] Higuchi, Tomoyuki. "Approach to an irregular time series on the
       basis of the fractal theory." Physica D: Nonlinear Phenomena 31.2
       (1988): 277-283.
    Examples
    --------
    Higuchi Fractal Dimension
        >>> import numpy as np
        >>> from entropy import higuchi_fd
        >>> np.random.seed(123)
        >>> x = np.random.rand(100)
        >>> print(higuchi_fd(x))
            2.051179
    """
    x = np.asarray(x, dtype=np.float64)
    kmax = int(kmax)
    return _higuchi_fd(x, kmax)

# In[ ]:


def _embed(x, order=3, delay=1):
    """Time-delay embedding.
    Parameters
    ----------
    x : 1d-array, shape (n_times)
        Time series
    order : int
        Embedding dimension (order)
    delay : int
        Delay.
    Returns
    -------
    embedded : ndarray, shape (n_times - (order - 1) * delay, order)
        Embedded time-series.
    """
    N = len(x)
    if order * delay > N:
        raise ValueError("Error: order * delay should be lower than x.size")
    if delay < 1:
        raise ValueError("Delay has to be at least 1.")
    if order < 2:
        raise ValueError("Order has to be at least 2.")
    Y = np.zeros((order, N - (order - 1) * delay))
    for i in range(order):
        Y[i] = x[i * delay:i * delay + Y.shape[1]]
    return Y.T

all = ['perm_entropy', 'spectral_entropy', 'svd_entropy', 'app_entropy',
       'sample_entropy']


def perm_entropy(x, order=3, delay=1, normalize=False):
    """Permutation Entropy.
    Parameters
    ----------
    x : list or np.array
        One-dimensional time series of shape (n_times)
    order : int
        Order of permutation entropy
    delay : int
        Time delay
    normalize : bool
        If True, divide by log2(order!) to normalize the entropy between 0
        and 1. Otherwise, return the permutation entropy in bit.
    Returns
    -------
    pe : float
        Permutation Entropy
    Notes
    -----
    The permutation entropy is a complexity measure for time-series first
    introduced by Bandt and Pompe in 2002 [1]_.
    The permutation entropy of a signal :math:`x` is defined as:
    .. math:: H = -\sum p(\pi)log_2(\pi)
    where the sum runs over all :math:`n!` permutations :math:`\pi` of order
    :math:`n`. This is the information contained in comparing :math:`n`
    consecutive values of the time series. It is clear that
    :math:`0 ≤ H (n) ≤ log_2(n!)` where the lower bound is attained for an
    increasing or decreasing sequence of values, and the upper bound for a
    completely random system where all :math:`n!` possible permutations appear
    with the same probability.
    The embedded matrix :math:`Y` is created by:
    .. math:: y(i)=[x_i,x_{i+delay}, ...,x_{i+(order-1) * delay}]
    .. math:: Y=[y(1),y(2),...,y(N-(order-1))*delay)]^T
    References
    ----------
    .. [1] Bandt, Christoph, and Bernd Pompe. "Permutation entropy: a
           natural complexity measure for time series." Physical review letters
           88.17 (2002): 174102.
    Examples
    --------
    1. Permutation entropy with order 2
        >>> from entropy import perm_entropy
        >>> x = [4, 7, 9, 10, 6, 11, 3]
        >>> # Return a value in bit between 0 and log2(factorial(order))
        >>> print(perm_entropy(x, order=2))
            0.918
    2. Normalized permutation entropy with order 3
        >>> from entropy import perm_entropy
        >>> x = [4, 7, 9, 10, 6, 11, 3]
        >>> # Return a value comprised between 0 and 1.
        >>> print(perm_entropy(x, order=3, normalize=True))
            0.589
    """
    x = np.array(x)
    ran_order = range(order)
    hashmult = np.power(order, ran_order)
    # Embed x and sort the order of permutations
    sorted_idx = _embed(x, order=order, delay=delay).argsort(kind='quicksort')
    # Associate unique integer to each permutations
    hashval = (np.multiply(sorted_idx, hashmult)).sum(1)
    # Return the counts
    _, c = np.unique(hashval, return_counts=True)
    # Use np.true_divide for Python 2 compatibility
    p = np.true_divide(c, c.sum())
    pe = -np.multiply(p, np.log2(p)).sum()
    if normalize:
        pe /= np.log2(factorial(order))
    return pe


def spectral_entropy(x, sf, method='fft', nperseg=None, normalize=False):
    """Spectral Entropy.
    Parameters
    ----------
    x : list or np.array
        One-dimensional time series of shape (n_times)
    sf : float
        Sampling frequency
    method : str
        Spectral estimation method ::
        'fft' : Fourier Transform (via scipy.signal.periodogram)
        'welch' : Welch periodogram (via scipy.signal.welch)
    nperseg : str or int
        Length of each FFT segment for Welch method.
        If None, uses scipy default of 256 samples.
    normalize : bool
        If True, divide by log2(psd.size) to normalize the spectral entropy
        between 0 and 1. Otherwise, return the spectral entropy in bit.
    Returns
    -------
    se : float
        Spectral Entropy
    Notes
    -----
    Spectral Entropy is defined to be the Shannon Entropy of the Power
    Spectral Density (PSD) of the data:
    .. math:: H(x, sf) =  -\sum_{f=0}^{f_s/2} PSD(f) log_2[PSD(f)]
    Where :math:`PSD` is the normalised PSD, and :math:`f_s` is the sampling
    frequency.
    References
    ----------
    .. [1] Inouye, T. et al. (1991). Quantification of EEG irregularity by
       use of the entropy of the power spectrum. Electroencephalography
       and clinical neurophysiology, 79(3), 204-210.
    Examples
    --------
    1. Spectral entropy of a pure sine using FFT
        >>> from entropy import spectral_entropy
        >>> import numpy as np
        >>> sf, f, dur = 100, 1, 4
        >>> N = sf * duration # Total number of discrete samples
        >>> t = np.arange(N) / sf # Time vector
        >>> x = np.sin(2 * np.pi * f * t)
        >>> print(np.round(spectral_entropy(x, sf, method='fft'), 2)
            0.0
    2. Spectral entropy of a random signal using Welch's method
        >>> from entropy import spectral_entropy
        >>> import numpy as np
        >>> np.random.seed(42)
        >>> x = np.random.rand(3000)
        >>> print(spectral_entropy(x, sf=100, method='welch'))
            9.939
    3. Normalized spectral entropy
        >>> print(spectral_entropy(x, sf=100, method='welch', normalize=True))
            0.995
    """
    x = np.array(x)
    # Compute and normalize power spectrum
    if method == 'fft':
        _, psd = periodogram(x, sf)
    elif method == 'welch':
        _, psd = welch(x, sf, nperseg=nperseg)
    psd_norm = np.divide(psd, psd.sum())
    se = -np.multiply(psd_norm, np.log2(psd_norm)).sum()
    if normalize:
        se /= np.log2(psd_norm.size)
    return se


def svd_entropy(x, order=3, delay=1, normalize=False):
    """Singular Value Decomposition entropy.
    Parameters
    ----------
    x : list or np.array
        One-dimensional time series of shape (n_times)
    order : int
        Order of permutation entropy
    delay : int
        Time delay
    normalize : bool
        If True, divide by log2(order!) to normalize the entropy between 0
        and 1. Otherwise, return the permutation entropy in bit.
    Returns
    -------
    svd_e : float
        SVD Entropy
    Notes
    -----
    SVD entropy is an indicator of the number of eigenvectors that are needed
    for an adequate explanation of the data set. In other words, it measures
    the dimensionality of the data.
    The SVD entropy of a signal :math:`x` is defined as:
    .. math::
        H = -\sum_{i=1}^{M} \overline{\sigma}_i log_2(\overline{\sigma}_i)
    where :math:`M` is the number of singular values of the embedded matrix
    :math:`Y` and :math:`\sigma_1, \sigma_2, ..., \sigma_M` are the
    normalized singular values of :math:`Y`.
    The embedded matrix :math:`Y` is created by:
    .. math:: y(i)=[x_i,x_{i+delay}, ...,x_{i+(order-1) * delay}]
    .. math:: Y=[y(1),y(2),...,y(N-(order-1))*delay)]^T
    Examples
    --------
    1. SVD entropy with order 2
        >>> from entropy import svd_entropy
        >>> x = [4, 7, 9, 10, 6, 11, 3]
        >>> # Return a value in bit between 0 and log2(factorial(order))
        >>> print(svd_entropy(x, order=2))
            0.762
    2. Normalized SVD entropy with order 3
        >>> from entropy import svd_entropy
        >>> x = [4, 7, 9, 10, 6, 11, 3]
        >>> # Return a value comprised between 0 and 1.
        >>> print(svd_entropy(x, order=3, normalize=True))
            0.687
    """
    x = np.array(x)
    mat = _embed(x, order=order, delay=delay)
    W = np.linalg.svd(mat, compute_uv=False)
    # Normalize the singular values
    W /= sum(W)
    svd_e = -np.multiply(W, np.log2(W)).sum()
    if normalize:
        svd_e /= np.log2(order)
    return svd_e


def _app_samp_entropy(x, order, metric='chebyshev', approximate=True):
    """Utility function for `app_entropy`` and `sample_entropy`.
    """
    _all_metrics = KDTree.valid_metrics
    if metric not in _all_metrics:
        raise ValueError('The given metric (%s) is not valid. The valid '
                         'metric names are: %s' % (metric, _all_metrics))
    phi = np.zeros(2)
    r = 0.2 * np.std(x, axis=-1, ddof=1)

    # compute phi(order, r)
    _emb_data1 = _embed(x, order, 1)
    if approximate:
        emb_data1 = _emb_data1
    else:
        emb_data1 = _emb_data1[:-1]
    count1 = KDTree(emb_data1, metric=metric).query_radius(emb_data1, r,
                                                           count_only=True
                                                           ).astype(np.float64)
    # compute phi(order + 1, r)
    emb_data2 = _embed(x, order + 1, 1)
    count2 = KDTree(emb_data2, metric=metric).query_radius(emb_data2, r,
                                                           count_only=True
                                                           ).astype(np.float64)
    if approximate:
        phi[0] = np.mean(np.log(count1 / emb_data1.shape[0]))
        phi[1] = np.mean(np.log(count2 / emb_data2.shape[0]))
    else:
        phi[0] = np.mean((count1 - 1) / (emb_data1.shape[0] - 1))
        phi[1] = np.mean((count2 - 1) / (emb_data2.shape[0] - 1))
    return phi


@jit('f8(f8[:], i4, f8)', nopython=True)
def _numba_sampen(x, mm=2, r=0.2):
    """
    Fast evaluation of the sample entropy using Numba.
    """
    n = x.size
    n1 = n - 1
    mm += 1
    mm_dbld = 2 * mm

    # Define threshold
    r *= x.std()

    # initialize the lists
    run = [0] * n
    run1 = run[:]
    r1 = [0] * (n * mm_dbld)
    a = [0] * mm
    b = a[:]
    p = a[:]

    for i in range(n1):
        nj = n1 - i

        for jj in range(nj):
            j = jj + i + 1
            if abs(x[j] - x[i]) < r:
                run[jj] = run1[jj] + 1
                m1 = mm if mm < run[jj] else run[jj]
                for m in range(m1):
                    a[m] += 1
                    if j < n1:
                        b[m] += 1
            else:
                run[jj] = 0
        for j in range(mm_dbld):
            run1[j] = run[j]
            r1[i + n * j] = run[j]
        if nj > mm_dbld - 1:
            for j in range(mm_dbld, nj):
                run1[j] = run[j]

    m = mm - 1

    while m > 0:
        b[m] = b[m - 1]
        m -= 1

    b[0] = n * n1 / 2
    a = np.array([float(aa) for aa in a])
    b = np.array([float(bb) for bb in b])
    p = np.true_divide(a, b)
    return -log(p[-1])


def app_entropy(x, order=2, metric='chebyshev'):
    """Approximate Entropy.
    Parameters
    ----------
    x : list or np.array
        One-dimensional time series of shape (n_times)
    order : int (default: 2)
        Embedding dimension.
    metric : str (default: chebyshev)
        Name of the metric function used with
        :class:`~sklearn.neighbors.KDTree`. The list of available
        metric functions is given by: ``KDTree.valid_metrics``.
    Returns
    -------
    ae : float
        Approximate Entropy.
    Notes
    -----
    Original code from the mne-features package.
    Approximate entropy is a technique used to quantify the amount of
    regularity and the unpredictability of fluctuations over time-series data.
    Smaller values indicates that the data is more regular and predictable.
    The value of :math:`r` is set to :math:`0.2 * std(x)`.
    Code adapted from the mne-features package by Jean-Baptiste Schiratti
    and Alexandre Gramfort.
    References
    ----------
    .. [1] Richman, J. S. et al. (2000). Physiological time-series analysis
           using approximate entropy and sample entropy. American Journal of
           Physiology-Heart and Circulatory Physiology, 278(6), H2039-H2049.
    1. Approximate entropy with order 2.
        >>> from entropy import app_entropy
        >>> import numpy as np
        >>> np.random.seed(1234567)
        >>> x = np.random.rand(3000)
        >>> print(app_entropy(x, order=2))
            2.075
    """
    phi = _app_samp_entropy(x, order=order, metric=metric, approximate=True)
    return np.subtract(phi[0], phi[1])


def sample_entropy(x, order=2, metric='chebyshev'):
    """Sample Entropy.
    Parameters
    ----------
    x : list or np.array
        One-dimensional time series of shape (n_times)
    order : int (default: 2)
        Embedding dimension.
    metric : str (default: chebyshev)
        Name of the metric function used with KDTree. The list of available
        metric functions is given by: `KDTree.valid_metrics`.
    Returns
    -------
    se : float
        Sample Entropy.
    Notes
    -----
    Sample entropy is a modification of approximate entropy, used for assessing
    the complexity of physiological time-series signals. It has two advantages
    over approximate entropy: data length independence and a relatively
    trouble-free implementation. Large values indicate high complexity whereas
    smaller values characterize more self-similar and regular signals.
    Sample entropy of a signal :math:`x` is defined as:
    .. math:: H(x, m, r) = -log\dfrac{C(m + 1, r)}{C(m, r)}
    where :math:`m` is the embedding dimension (= order), :math:`r` is
    the radius of the neighbourhood (default = :math:`0.2 * std(x)`),
    :math:`C(m + 1, r)` is the number of embedded vectors of length
    :math:`m + 1` having a Chebyshev distance inferior to :math:`r` and
    :math:`C(m, r)` is the number of embedded vectors of length
    :math:`m` having a Chebyshev distance inferior to :math:`r`.
    Note that if metric == 'chebyshev' and x.size < 5000 points, then the
    sample entropy is computed using a fast custom Numba script. For other
    metric types or longer time-series, the sample entropy is computed using
    a code from the mne-features package by Jean-Baptiste Schiratti
    and Alexandre Gramfort (requires sklearn).
    References
    ----------
    .. [1] Richman, J. S. et al. (2000). Physiological time-series analysis
           using approximate entropy and sample entropy. American Journal of
           Physiology-Heart and Circulatory Physiology, 278(6), H2039-H2049.
    Examples
    --------
    1. Sample entropy with order 2.
        >>> from entropy import sample_entropy
        >>> import numpy as np
        >>> np.random.seed(1234567)
        >>> x = np.random.rand(3000)
        >>> print(sample_entropy(x, order=2))
            2.192
    2. Sample entropy with order 3 using the Euclidean distance.
        >>> from entropy import sample_entropy
        >>> import numpy as np
        >>> np.random.seed(1234567)
        >>> x = np.random.rand(3000)
        >>> print(sample_entropy(x, order=3, metric='euclidean'))
            2.725
    """
    x = np.asarray(x, dtype=np.float64)
    if metric == 'chebyshev' and x.size < 5000:
        return _numba_sampen(x, mm=order, r=0.2)
    else:
        phi = _app_samp_entropy(x, order=order, metric=metric,
                                approximate=False)
        return -np.log(np.divide(phi[1], phi[0]))

# In[ ]:


def entropy_and_fractal_dim(x):
    return [perm_entropy(x), svd_entropy(x), app_entropy(x), sample_entropy(x), petrosian_fd(x), katz_fd(x), higuchi_fd(x)]

# In[ ]:


signals = X.reshape((len(X), X.shape[1]*X.shape[2]))
features = []

for signal in signals:
    features.append(entropy_and_fractal_dim(signal))

# In[ ]:


features = np.array(features).reshape((len(features), 7))
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(features)
features = scaler.transform(features)

# In[ ]:


# model = Sequential([Dense(1, activation='sigmoid', input_shape=(7,))])
# model.compile(loss='binary_crossentropy', optimizer='adam')

# In[ ]:


# model.fit(x, np.array(list(np.zeros((len(features_zero)))) + list(np.ones((len(features_one))))), epochs=1000)

# In[ ]:


# Weights and bias for logistic regression of entropy features
# weights = model.get_weights()[0]
# biases = model.get_weights()[1]

# In[ ]:


# print(weights)
# print(biases)

# In[ ]:


# e = 2.718281

# exp = make_function(function=lambda x: e**x, name='exp', arity=1)
# tanh = make_function(function=lambda x: (e**(2*x) - 1)/(e**(2*x) + 1), name='tanh', arity=1)

# In[ ]:


# function_set = ['add', 'sub', 'mul', 'div',
                # 'sqrt', 'log', 'abs', 'neg', 'inv',
                # 'max', 'min', exp, tanh]

# gp = SymbolicTransformer(generations=10, population_size=2000,
                         # hall_of_fame=100, n_components=50,
                         # function_set=function_set,
                         # parsimony_coefficient=0.0005,
                         # max_samples=0.9, verbose=1,
                         # random_state=0, n_jobs=3)

# gp.fit(features, np.array(list(np.zeros((len(features_zero)))) + list(np.ones((len(features_one))))))

# In[ ]:


# engineered_features = gp._programs

# for i in range(len(engineered_features)):
    # for engineered_feature in engineered_features[i]:
        # if engineered_feature != None:
            # print(engineered_feature)

# In[ ]:


# The X shape here is very important. It is also important undertand a little how a LSTM works
# X.shape[0] is the number of id_measuremts contained in train data
# X.shape[1] is the number of chunks resultant of the transformation, each of this date enters in the LSTM serialized
# This way the LSTM can understand the position of a data relative with other and activate a signal that needs
# a serie of inputs in a specifc order.
# X.shape[3] is the number of features multiplied by the number of phases (3)
print(X.shape, y.shape)
# save data into file, a numpy specific format
np.save("X.npy",X)
np.save("y.npy",y)
np.save("features.npy",features)

# In[ ]:


# This is NN LSTM Model creation
def model_lstm(input_shape, feat_shape):
    inp = Input(shape=(input_shape[1], input_shape[2],))
    feat = Input(shape=(feat_shape[1],))

    bi_lstm = Bidirectional(CuDNNLSTM(128, return_sequences=True), merge_mode='concat')(inp)
    bi_gru = Bidirectional(CuDNNGRU(64, return_sequences=True), merge_mode='concat')(bi_lstm)
    
    attention = Attention(input_shape[1])(bi_gru)
    
    x = concatenate([attention, feat], axis=1)
    x = Dense(64, activation="relu")(x)
    x = Dense(1, activation="sigmoid")(x)
    
    model = Model(inputs=[inp, feat], outputs=x)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[matthews_correlation])
    
    return model

# In[ ]:


# Here is where the training happens

# First, create a set of indexes of the 5 folds
splits = list(StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=2019).split(X, y))
preds_val = []
y_val = []
# Then, iteract with each fold
# If you dont know, enumerate(['a', 'b', 'c']) returns [(0, 'a'), (1, 'b'), (2, 'c')]
for idx, (train_idx, val_idx) in enumerate(splits):
    K.clear_session() # I dont know what it do, but I imagine that it "clear session" :)
    print("Beginning fold {}".format(idx+1))
    # use the indexes to extract the folds in the train and validation data
    train_X, train_feat, train_y, val_X, val_feat, val_y = X[train_idx], features[train_idx], y[train_idx], X[val_idx], features[val_idx], y[val_idx]
    # instantiate the model for this fold
    model = model_lstm(train_X.shape, features.shape)
    # This checkpoint helps to avoid overfitting. It just save the weights of the model if it delivered an
    # validation matthews_correlation greater than the last one.
    ckpt = ModelCheckpoint('weights_{}.h5'.format(idx), save_best_only=True, save_weights_only=True, verbose=1, monitor='val_matthews_correlation', mode='max')
    # Train, train, train
    model.fit([train_X, train_feat], train_y, batch_size=128, epochs=50, validation_data=([val_X, val_feat], val_y), callbacks=[ckpt])
    # loads the best weights saved by the checkpoint
    model.load_weights('weights_{}.h5'.format(idx))
    # Add the predictions of the validation to the list preds_val
    preds_val.append(model.predict([val_X, val_feat], batch_size=512))
    # and the val true y
    y_val.append(val_y)

# concatenates all and prints the shape    
preds_val = np.concatenate(preds_val)[...,0]
y_val = np.concatenate(y_val)
preds_val.shape, y_val.shape


# In[ ]:


# The output of this kernel must be binary (0 or 1), but the output of the NN Model is float (0 to 1).
# So, find the best threshold to convert float to binary is crucial to the result
# this piece of code is a function that evaluates all the possible thresholds from 0 to 1 by 0.01
def threshold_search(y_true, y_proba):
    best_threshold = 0
    best_score = 0
    for threshold in tqdm([i * 0.01 for i in range(100)]):
        score = K.eval(matthews_correlation(y_true.astype(np.float64), (y_proba > threshold).astype(np.float64)))
        if score > best_score:
            best_threshold = threshold
            best_score = score
    search_result = {'threshold': best_threshold, 'matthews_correlation': best_score}
    return search_result

# In[ ]:


optimal_values = threshold_search(y_val, preds_val)
best_threshold = optimal_values['threshold']
best_score = optimal_values['matthews_correlation']

# In[ ]:


print("Optimal Threshold : " + str(best_threshold))
print("Best Matthews Correlation : " + str(best_score))

# In[ ]:


# Now load the test data
# This first part is the meta data, not the main data, the measurements
meta_test = pd.read_csv('../input/metadata_test.csv')

# In[ ]:


meta_test = meta_test.set_index(['signal_id'])
meta_test.head()

# In[ ]:


# First we daclarete a series of parameters to initiate the loading of the main data
# it is too large, it is impossible to load in one time, so we are doing it in dividing in 10 parts
first_sig = meta_test.index[0]
n_parts = 10
max_line = len(meta_test)
part_size = int(max_line / n_parts)
last_part = max_line % n_parts
print(first_sig, n_parts, max_line, part_size, last_part, n_parts * part_size + last_part)
# Here we create a list of lists with start index and end index for each of the 10 parts and one for the last partial part
start_end = [[x, x+part_size] for x in range(first_sig, max_line + first_sig, part_size)]
start_end = start_end[:-1] + [[start_end[-1][0], start_end[-1][0] + last_part]]
print(start_end)
X_test = []
# now, very like we did above with the train data, we convert the test data part by part
# transforming the 3 phases 800000 measurement in matrix (160,57)
for start, end in start_end:
    subset_test = pq.read_pandas('../input/test.parquet', columns=[str(i) for i in range(start, end)]).to_pandas()
    for i in tqdm(subset_test.columns):
        id_measurement, phase = meta_test.loc[int(i)]
        subset_test_col = subset_test[i]
        subset_trans = transform_ts(subset_test_col)
        X_test.append([i, id_measurement, phase, subset_trans])

# In[ ]:


X_test_input = np.asarray([np.concatenate([X_test[i][3],X_test[i+1][3], X_test[i+2][3]], axis=1) for i in range(0,len(X_test), 3)])
np.save("X_test.npy",X_test_input)
X_test_input.shape

# In[ ]:


signals = X_test_input.reshape((len(X_test_input), X_test_input.shape[1]*X_test_input.shape[2]))
features_test = []

for signal in signals:
    features_test.append(entropy_and_fractal_dim(signal))

# In[ ]:


features_test = np.array(features_test).reshape((len(features_test), 7))
features_test = scaler.transform(features_test)
np.save("features_test.npy",features_test)

# In[ ]:


submission = pd.read_csv('../input/sample_submission.csv')
print(len(submission))
submission.head()

# In[ ]:


preds_test = []
for i in range(N_SPLITS):
    model.load_weights('weights_{}.h5'.format(i))
    pred = model.predict([X_test_input, features_test], batch_size=300, verbose=1)
    pred_3 = []
    for pred_scalar in pred:
        for i in range(3):
            pred_3.append(pred_scalar)
    preds_test.append(pred_3)

# In[ ]:


preds_test = (np.squeeze(np.mean(preds_test, axis=0)) > best_threshold).astype(np.int)
preds_test.shape

# In[ ]:


submission['target'] = preds_test
submission.to_csv('submission.csv', index=False)
submission.head()

# In[ ]:



