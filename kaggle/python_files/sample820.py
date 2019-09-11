#!/usr/bin/env python
# coding: utf-8

# This is my first kernel on Kaggle. I am very excited to contribute to this competition and going back to my days of studying biomechatonics where I first met the concept of feature extraction to classify signals from sensors on human body. 
# 
# I made a fast literature review on feature extraction and signal classification and prepared following experiment for my first submissions.
# 
# I will be updating explanatory walktrough alongside model improvements.
# 
# Running and submitting from full notebook on interactive session was problemmatic due to memory troubles with the test set. I will work my way around it meanwhile you can download to use or fork to improve. Open for feedbacks to improve my first kernel.
# 
# Have fun everyone!

# Many thanks to following kernels:
# - For shortening the signals with a simple feature extraction thanks to: https://www.kaggle.com/ashishpatel26/transfer-learning-in-basic-nn
# - For signal denoising and fft: https://www.kaggle.com/theoviel/fast-fourier-transform-denoising

# In[ ]:


import keras
import keras.backend as K
from keras.layers import LSTM,Dropout,Dense,TimeDistributed,Conv1D,MaxPooling1D,Flatten
from keras.models import Sequential
import tensorflow as tf
import gc
from numba import jit
from IPython.display import display, clear_output
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sns.set_style("whitegrid")

# ## 0. Info

# **Signals:**
# 
# - 800.000 measurement points for 8712 signals.
# - The signals are **three-phased** so there are 2904 distinct signaling instances.
# - Three phase signals:
#     - Sums to zero.
#     - When one fails other continue to carry the current.
#     - Can be rectified to be converted to DC current.
#     - Ripples in rectification can be seen on failure.
#    

# ![](https://upload.wikimedia.org/wikipedia/commons/thumb/4/48/3-phase_flow.gif/357px-3-phase_flow.gif)

# ### What is partial discharge?

#  - Typical situation of PD: imagine there is an internal cavity/void or **impurity in insulation**. 
#  - When **High Voltage** is applied on conductor, a field is also induced on the cavity. Further, when the field increases, this **defect breaks down** and **discharges** different forms of energy which result in partial discharge.
#  - This phenomenon is damaging over a long period of time. It is not event that occurs suddenly. 

# ### Classical Modes of Detection
# - Partial Discharges can be detected by **measuring the emissions** they give off: Ultrasonic Sound, Transient Earth Voltages (TEV and UHF energy).
# - Is it possible to enhance the modes of detection by **better feature extraction** for the classifiers?
# - **Intel Mobile ODT** challenge on 2017 was about topping **classical image processing** methods by automatic feature extaction using pre-trained CNN models and **transfer learning**.
# - **Two possible approaches**:
#     - FE on signals and feeding them into NNs for classification.
#     - Using NNs further as feature extractors and then use shallow classifiers (XGBoost) for binary classification
# 

# ### **TASK:** Classify long-term failure of covered conductors based on signal characteristics:
# - Extract features from time series data for classification.
# - Use **CNN** for further FE and **LSTM** to get temporal dependencies and perform time series classification on the top layer.

# ## 1. Load Data

# In[ ]:


import pyarrow.parquet as pq
import pandas as pd
import numpy as np

# In[ ]:


train_set = pq.read_pandas('../input/train.parquet').to_pandas()

# In[ ]:


meta_train = pd.read_csv('../input/metadata_train.csv')

# ## 2. Process and Minimize Data

# In[ ]:


@jit('float32(float32[:,:], int32)')
def feature_extractor(x, n_part=1000):
    lenght = len(x)
    pool = np.int32(np.ceil(lenght/n_part))
    output = np.zeros((n_part,))
    for j, i in enumerate(range(0,lenght, pool)):
        if i+pool < lenght:
            k = x[i:i+pool]
        else:
            k = x[i:]
        output[j] = np.max(k, axis=0) - np.min(k, axis=0)
    return output

# In[ ]:


x_train = []
y_train = []
for i in tqdm(meta_train.signal_id):
    idx = meta_train.loc[meta_train.signal_id==i, 'signal_id'].values.tolist()
    y_train.append(meta_train.loc[meta_train.signal_id==i, 'target'].values)
    x_train.append(abs(feature_extractor(train_set.iloc[:, idx].values, n_part=400)))

# In[ ]:


del train_set; gc.collect()

# In[ ]:


y_train = np.array(y_train).reshape(-1,)
X_train = np.array(x_train).reshape(-1,x_train[0].shape[0])

# ## 3. Build Primitive CNN + LSTM Model

# * CNN is for feature extraction and LSTM is for capturing time dependency.

# In[ ]:


def keras_auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc

# In[ ]:


n_signals = 1 #So far each instance is one signal. We will diversify them in next step
n_outputs = 1 #Binary Classification

# In[ ]:


#Build the model
verbose, epochs, batch_size = True, 15, 16
n_steps, n_length = 40, 10
X_train = X_train.reshape((X_train.shape[0], n_steps, n_length, n_signals))
# define model
model = Sequential()
model.add(TimeDistributed(Conv1D(filters=64, kernel_size=3, activation='relu'), input_shape=(None,n_length,n_signals)))
model.add(TimeDistributed(Conv1D(filters=64, kernel_size=3, activation='relu')))
model.add(TimeDistributed(Dropout(0.5)))
model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
model.add(TimeDistributed(Flatten()))
model.add(LSTM(100))
model.add(Dropout(0.5))
model.add(Dense(100, activation='relu'))
model.add(Dense(n_outputs, activation='sigmoid'))

# In[ ]:


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[keras_auc])

# In[ ]:


model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose)

# In[ ]:


model.save_weights('model1.hdf5')

# In[ ]:


#%%time
#test_set = pq.read_pandas('../input/test.parquet').to_pandas()

# In[ ]:


#%%time
#meta_test = pd.read_csv('../input/metadata_test.csv')

# In[ ]:


#x_test = []
#for i in tqdm(meta_test.signal_id.values):
#    idx=i-8712
#    clear_output(wait=True)
#    x_test.append(abs(feature_extractor(test_set.iloc[:, idx].values, n_part=400)))

# In[ ]:


#del test_set; gc.collect()

# In[ ]:


#X_test = x_test.reshape((x_test.shape[0], n_steps, n_length, n_signals))

# In[ ]:


#preds = model.predict(X_test)

# In[ ]:


#threshpreds = (preds>0.5)*1

# In[ ]:


#sub = pd.read_csv('../input/sample_submission.csv')
#sub.target = threshpreds

# In[ ]:


#sub.to_csv('first_sub.csv',index=False)
#Gave me an LB score of 0.450

# ## 4. Further processing of Signals to Diversify the Model

# We only fed what we were given and only feature engineering was to reduce signal lengths because 800000 time steps is troublesome with LSTM. Signal classification applications takes more than one modalities or channels in real life. So I will try to increase the feature size in terms of channel depth rather than feature count.
# 
# Proposed data structure: **n_instances x n_timesteps x  n_channels**

# ### 4.a - Filter & Transform Signals

# In[ ]:


#Both numpy and scipy has utilities for FFT which is an endlessly useful algorithm
from numpy.fft import *
from scipy import fftpack

# In[ ]:


train_set = pq.read_pandas('../input/train.parquet').to_pandas()

# In[ ]:


#FFT to filter out HF components and get main signal profile
def low_pass(s, threshold=1e4):
    fourier = rfft(s)
    frequencies = rfftfreq(s.size, d=2e-2/s.size)
    fourier[frequencies > threshold] = 0
    return irfft(fourier)

# In[ ]:


def phase_indices(signal_num):
    phase1 = 3*signal_num
    phase2 = 3*signal_num + 1
    phase3 = 3*signal_num + 2
    return phase1,phase2,phase3

# In[ ]:


s_id = 14
p1,p2,p3 = phase_indices(s_id)

# In[ ]:


plt.figure(figsize=(10,5))
plt.title('Signal %d / Target:%d'%(s_id,meta_train[meta_train.id_measurement==s_id].target.unique()[0]))
plt.plot(train_set.iloc[:,p1])
plt.plot(train_set.iloc[:,p2])
plt.plot(train_set.iloc[:,p3])

# In[ ]:


lf_signal_1 = low_pass(train_set.iloc[:,p1])
lf_signal_2 = low_pass(train_set.iloc[:,p2])
lf_signal_3 = low_pass(train_set.iloc[:,p3])

# In[ ]:


plt.figure(figsize=(10,5))
plt.title('De-noised Signal %d / Target:%d'%(s_id,meta_train[meta_train.id_measurement==s_id].target.unique()[0]))
plt.plot(lf_signal_1)
plt.plot(lf_signal_2)
plt.plot(lf_signal_3)

# In[ ]:


plt.figure(figsize=(10,5))
plt.title('Signal %d AbsVal / Target: %d'%(s_id,meta_train[meta_train.id_measurement==s_id].target.unique()[0]))
plt.plot(np.abs(lf_signal_1))
plt.plot(np.abs(lf_signal_2))
plt.plot(np.abs(lf_signal_3))

# In[ ]:


plt.figure(figsize=(10,5))
plt.title('Signal %d / Target: %d'%(s_id,meta_train[meta_train.id_measurement==s_id].target.unique()[0]))
plt.plot(lf_signal_1)
plt.plot(lf_signal_2)
plt.plot(lf_signal_3)
plt.plot((np.abs(lf_signal_1)+np.abs(lf_signal_2)+np.abs(lf_signal_3)))
plt.legend(['phase 1','phase 2','phase 3','DC Component'],loc=1)

# In[ ]:


###Filter out low frequencies from the signal to get HF characteristics
def high_pass(s, threshold=1e7):
    fourier = rfft(s)
    frequencies = rfftfreq(s.size, d=2e-2/s.size)
    fourier[frequencies < threshold] = 0
    return irfft(fourier)

# In[ ]:


hf_signal_1 = high_pass(train_set.iloc[:,p1])
hf_signal_2 = high_pass(train_set.iloc[:,p2])
hf_signal_3 = high_pass(train_set.iloc[:,p3])

# In[ ]:


plt.figure(figsize=(10,5))
plt.title('Signal %d / Target:%d'%(s_id,meta_train[meta_train.id_measurement==s_id].target.unique()[0]))
plt.plot(hf_signal_1)
#plt.plot(hf_signal_2)
#plt.plot(hf_signal_3)

# As seen above we can decouple the signals into their high and low frequency components using FFT. So the number of signal channels that are fed into the model can may very well be diversified using different decouplings in time and frequency domains. 
# 
# The mode of decoupling so far was filtering what we have to get new signals. Following part is about playing around the frequency domain to create features. 

# ### 4.b - Spectogram Features

# In[ ]:


signal = train_set.iloc[:,p1]

# In[ ]:


x = signal
X = fftpack.fft(x,n=400)
freqs = fftpack.fftfreq(n=400,d=2e-2/x.size) 

# In[ ]:


plt.plot(x)

# In[ ]:


fig, ax = plt.subplots()
ax.set_title('Full Spectrum with Scipy')
ax.set_xlabel('Frequency in Hertz [Hz]')
ax.set_ylabel('Frequency Domain (Spectrum) Magnitude')
ax.stem(freqs[1:], np.abs(X)[1:])

# In[ ]:


x = high_pass(train_set.iloc[:,p1])
X = fftpack.fft(x,n=400)
freqs = fftpack.fftfreq(n=400,d=2e-2/x.size) 

# In[ ]:


plt.plot(x)

# In[ ]:


fig, ax = plt.subplots()
ax.set_title('High Frequency Spectrum')
ax.set_xlabel('Frequency in Hertz [Hz]')
ax.set_ylabel('Frequency Domain (Spectrum) Magnitude')
ax.stem(freqs[1:], np.abs(X)[1:])

# In[ ]:


x = low_pass(signal)
X = fftpack.fft(x,n=400)
freqs = fftpack.fftfreq(n=400,d=2e-2/x.size) 

# In[ ]:


plt.plot(x)

# In[ ]:


fig, ax = plt.subplots()
ax.set_title('Low Frequency Spectrum')
ax.set_xlabel('Frequency in Hertz [Hz]')
ax.set_ylabel('Frequency Domain (Spectrum) Magnitude')
ax.stem(freqs[1:], np.abs(X)[1:])

# ### 4.c - Frequencies vs. Time
# 

# In[ ]:


p1,p2,p3 = phase_indices(100)
signal = train_set.iloc[:,p1]

# In[ ]:


from scipy import signal as sgn
M = 1024
rate = 1/(2e-2/signal.size)

freqs, times, Sx = sgn.spectrogram(signal.values, fs=rate, window='hanning',
                                      nperseg=1024, noverlap=M - 100,
                                      detrend='constant', scaling='spectrum')

f, ax = plt.subplots(figsize=(10, 5))
ax.set_ylabel('Frequency [Hz]')
ax.set_xlabel('Time(s)')
ax.set_title('Spectogram')
ax.pcolormesh(times, freqs, np.log10(Sx), cmap='viridis')

# Now we have the temporal behavior of frequencies that compose our signal in terms of their magnitudes. Aggregations can be made on both time and frequency axes by selecting appropriate window sizes.

# ## 5. Add Features

# Apart from the signals themselves we have: 
# - Low Pass Filtered Signal (800000)
# - High Pass Filtered Signal (800000)
# - DC component from adding absolute values of 3-phases. (800000)
# - Frequency Magnitude Spectrum of full signal (400)
# - Frequency Magnitude Spectrum of Low Freq signal (400)
# - Frequency Magnitude Spectrum of High Freq signal (400)
# - Frequency Spectrum 

# For initial improvement of our model we will use diversification of signals from **4.a** and create **4 channels**:
# - **Signal itself**
# - **LF** component
# - **HF** component
# - **DC** component from three-phase merge)
# 
# The justification for DC component is that all three phase signals are rectified to form a DC behavior with a small amount of ripple. I naively assumed that a partial discharce occuring in any of the 3 signals, will result in corruption at the resulting voltage behavior. While composing up the features, each instance that belong to a triple will have the same "DC component."

# In[ ]:


x_train_lp = []
x_train_hp = []
x_train_dc = []
for i in meta_train.signal_id:
    idx = meta_train.loc[meta_train.signal_id==i, 'signal_id'].values.tolist()
    clear_output(wait=True)
    display(idx)
    hp = high_pass(train_set.iloc[:, idx[0]])
    lp = low_pass(train_set.iloc[:, idx[0]])
    meas_id = meta_train.id_measurement[meta_train.signal_id==idx].values[0]
    p1,p2,p3=phase_indices(meas_id)
    lf_signal_1,lf_signal_2,lf_signal_3 = low_pass(train_set.iloc[:,p1]), low_pass(train_set.iloc[:,p2]), low_pass(train_set.iloc[:,p3])
    dc = np.abs(lf_signal_1)+np.abs(lf_signal_2)+np.abs(lf_signal_3)
    x_train_lp.append(abs(feature_extractor(lp, n_part=400)))
    x_train_hp.append(abs(feature_extractor(hp, n_part=400)))
    x_train_dc.append(abs(feature_extractor(dc, n_part=400)))

# In[ ]:


del train_set; gc.collect()

# In[ ]:


#x_test_lp = []
#x_test_hp = []
#x_test_dc = []
#for i in tqdm(meta_test.signal_id):
#    idx = idx=i-8712
#    clear_output(wait=True)
#    #display(idx)
#    hp = high_pass(test_set.iloc[:, idx])
#    lp = low_pass(test_set.iloc[:, idx])
#    meas_id = meta_test.id_measurement[meta_test.signal_id==i].values[0]
#    p1,p2,p3=phase_indices(meas_id)
#    lf_signal_1,lf_signal_2,lf_signal_3 = low_pass(test_set.iloc[:,p1-8712]), low_pass(test_set.iloc[:,p2-8712]), low_pass(test_set.iloc[:,p3-8712])
#    dc = np.abs(lf_signal_1)+np.abs(lf_signal_2)+np.abs(lf_signal_3)
#    x_test_lp.append(abs(feature_extractor(lp, n_part=400)))
#    x_test_hp.append(abs(feature_extractor(hp, n_part=400)))
#    x_test_dc.append(abs(feature_extractor(dc, n_part=400)))

# In[ ]:


x_train = np.array(x_train).reshape(-1,x_train[0].shape[0])
x_train_lp = np.array(x_train).reshape(-1,x_train_lp[0].shape[0])
x_train_hp = np.array(x_train).reshape(-1,x_train_hp[0].shape[0])
x_train_dc = np.array(x_train).reshape(-1,x_train_dc[0].shape[0])

# In[ ]:


#x_test = np.array(x_test).reshape(-1,x_test[0].shape[0])
#x_test_lp = np.array(x_test).reshape(-1,x_test_lp[0].shape[0])
#x_test_hp = np.array(x_test).reshape(-1,x_test_hp[0].shape[0])
#x_test_dc = np.array(x_test).reshape(-1,x_test_dc[0].shape[0])

# In[ ]:


train = np.dstack((x_train,x_train_lp,x_train_hp,x_train_dc))
#test = np.dstack((x_test,x_test_lp,x_test_hp,x_test_dc))

# In[ ]:


y_train = np.array(y_train).reshape(-1,)

# In[ ]:


verbose, epochs, batch_size = True, 15, 16
n_signals,n_steps, n_length = 4,40, 10
train = train.reshape((train.shape[0], n_steps, n_length, n_signals))
# define model
model = Sequential()
model.add(TimeDistributed(Conv1D(filters=64, kernel_size=3, activation='relu'), input_shape=(None,n_length,n_signals)))
model.add(TimeDistributed(Conv1D(filters=64, kernel_size=3, activation='relu')))
model.add(TimeDistributed(Dropout(0.5)))
model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
model.add(TimeDistributed(Flatten()))
model.add(LSTM(100))
model.add(Dropout(0.5))
model.add(Dense(100, activation='relu'))
model.add(Dense(n_outputs, activation='sigmoid'))

# In[ ]:


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[keras_auc])

# In[ ]:


# fit network
model.fit(train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose)

# In[ ]:


model.save_weights('model2.hdf5')

# In[ ]:


#X_test = test.reshape((test.shape[0], n_steps, n_length, n_signals))

# In[ ]:


#preds = model.predict(X_test)

# In[ ]:


#threshpreds = (preds>0.5)*1

# In[ ]:


#sub = pd.read_csv('data/sample_submission.csv')
#sub.target = threshpreds

# In[ ]:


#sub.to_csv('submissions/second_sub.csv',index=False)

# ### Second submission with 4 channels scores higher on LB. Increased number of epochs and playing with threshold may yield better results. I got 0.513 with re-training the model for 3+ times. 

# # TODO#1: Turn Spectral Analysis into Features
# # TODO#2: Enhcance the experiment with cross_val, adaptive learning rate, early stopping, ensembling etc.
