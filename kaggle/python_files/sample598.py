#!/usr/bin/env python
# coding: utf-8

# # Neural Network - Statistical Encoding - Microsoft Malware
# There aren't any examples of using a neural network to model Microsoft Malware, so I thought I'd post one. Also in this kernel, I show statistical one-hot-encoding where only boolean variables that are independently statistically significant are created.
# 
# # Load Train.csv

# In[ ]:


# IMPORT LIBRARIES
import pandas as pd, numpy as np, os, gc

# LOAD AND FREQUENCY-ENCODE
FE = ['EngineVersion','AppVersion','AvSigVersion','Census_OSVersion']
# LOAD AND ONE-HOT-ENCODE
OHE = [ 'RtpStateBitfield','IsSxsPassiveMode','DefaultBrowsersIdentifier',
        'AVProductStatesIdentifier','AVProductsInstalled', 'AVProductsEnabled',
        'CountryIdentifier', 'CityIdentifier', 
        'GeoNameIdentifier', 'LocaleEnglishNameIdentifier',
        'Processor', 'OsBuild', 'OsSuite',
        'SmartScreen','Census_MDC2FormFactor',
        'Census_OEMNameIdentifier', 
        'Census_ProcessorCoreCount',
        'Census_ProcessorModelIdentifier', 
        'Census_PrimaryDiskTotalCapacity', 'Census_PrimaryDiskTypeName',
        'Census_HasOpticalDiskDrive',
        'Census_TotalPhysicalRAM', 'Census_ChassisTypeName',
        'Census_InternalPrimaryDiagonalDisplaySizeInInches',
        'Census_InternalPrimaryDisplayResolutionHorizontal',
        'Census_InternalPrimaryDisplayResolutionVertical',
        'Census_PowerPlatformRoleName', 'Census_InternalBatteryType',
        'Census_InternalBatteryNumberOfCharges',
        'Census_OSEdition', 'Census_OSInstallLanguageIdentifier',
        'Census_GenuineStateName','Census_ActivationChannel',
        'Census_FirmwareManufacturerIdentifier',
        'Census_IsTouchEnabled', 'Census_IsPenCapable',
        'Census_IsAlwaysOnAlwaysConnectedCapable', 'Wdft_IsGamer',
        'Wdft_RegionIdentifier']

# LOAD ALL AS CATEGORIES
dtypes = {}
for x in FE+OHE: dtypes[x] = 'category'
dtypes['MachineIdentifier'] = 'str'
dtypes['HasDetections'] = 'int8'

# LOAD CSV FILE
df_train = pd.read_csv('../input/train.csv', usecols=dtypes.keys(), dtype=dtypes)
print ('Loaded',len(df_train),'rows of TRAIN.CSV!')

# DOWNSAMPLE
sm = 2000000
df_train = df_train.sample(sm)
print ('Only using',sm,'rows to train and validate')
x=gc.collect()

# # Statistically Encode Variables
# All the variables in the Python variable list `FE` will get frequency encoded and all the variables in list `OHE` will get statistically one-hot-encoded. In total, forty-three variables are imported from the training csv while thirty-nine were ignored.
#   
# Among all our category variables, there are a combined 211,562 values! So we can't one-hot-encode all. (Note that this is without Census_OEMModelIdentifier's 175,366 or Census_SystemVolumeTotalCapacity's 536,849) We will use a trick from statistics. First we'll assume we have a random sample. (Which we don't actually have, but let's pretend.) Then for each value, we will test the following hypotheses   
# 
#  $$H_0: \text{Prob(HasDetections=1 given value is present)} = 0.5 $$ 
#  $$H_A: \text{Prob(HasDetections=1 given value is present)} \ne 0.5$$  
#     
# The test statistic z-value equals \\( \hat{p} \\), the observed HasDetections rate given value is present, minus 0.5 divided by the standard deviation of \\( \hat{p} \\). The Central Limit Theorem tells us
# 
# $$\text{z-value} = \frac{\hat{p}-0.5}{SD(\hat{p})} = 2 (\hat{p} - 0.5)\sqrt{n} $$
# 
# where \\(n\\) is the number of occurences of the value. If the absolute value of \\(z\\) is greater than 2.0, we are 95% confident that Prob(HasDetections=1 given value is present) is not equal 0.5 and we will include a boolean for this value in our model. Actually, we'll use a \\(z\\) threshold of 5.0 and require \\( 10^{-7}n>0.005 \\). This adds 350 new boolean variables (instead of naively one-hot-encoding 211,562!).  
#   
#  ## Example - Census_FirmwareManufacturerIdentifier
# In the plots below, the dotted lines use the right y-axis and solid lines/bars use the left. The top plot below shows 20 values of variable `Census_FirmwareManufacturerIdentifier`. Notice that I consider NAN a value. Each of these values contains over 0.5% of the data. And all the variables together contain 97% of the data. Value=93 has a HasDetections rate of 52.5% while value=803 has a HasDetections rate of 35.4%. Their z-values are \\(22.2 = 2\times(0.5253-0.5)\times\sqrt{192481} \text{  }\\)  and \\(-71.3 = 2\times(0.3535-0.5)\times\sqrt{59145}\text{  }\\)   respectively! The probability that value=93 and value=803 have a HasDetections rate of 50% and what we are observing is due to chance is close to nothing. Additionally from the bottom plot, you see that these two values have consistently been high and low throughout all of the year 2018. We can trust that this trend will continue into the test set's October and November computers.  
#    
# ![image](http://playagricola.com/Kaggle/Firm13019.png)
# 
# ## Python Code
# To see the Python encoding functions, click 'see code' to the right.  

# In[ ]:


import math

# CHECK FOR NAN
def nan_check(x):
    if isinstance(x,float):
        if math.isnan(x):
            return True
    return False

# FREQUENCY ENCODING
def encode_FE(df,col,verbose=1):
    d = df[col].value_counts(dropna=False)
    n = col+"_FE"
    df[n] = df[col].map(d)/d.max()
    if verbose==1:
        print('FE encoded',col)
    return [n]

# ONE-HOT-ENCODE ALL CATEGORY VALUES THAT COMPRISE MORE THAN
# "FILTER" PERCENT OF TOTAL DATA AND HAS SIGNIFICANCE GREATER THAN "ZVALUE"
def encode_OHE(df, col, filter, zvalue, tar='HasDetections', m=0.5, verbose=1):
    cv = df[col].value_counts(dropna=False)
    cvd = cv.to_dict()
    vals = len(cv)
    th = filter * len(df)
    sd = zvalue * 0.5/ math.sqrt(th)
    #print(sd)
    n = []; ct = 0; d = {}
    for x in cv.index:
        try:
            if cv[x]<th: break
            sd = zvalue * 0.5/ math.sqrt(cv[x])
        except:
            if cvd[x]<th: break
            sd = zvalue * 0.5/ math.sqrt(cvd[x])
        if nan_check(x): r = df[df[col].isna()][tar].mean()
        else: r = df[df[col]==x][tar].mean()
        if abs(r-m)>sd:
            nm = col+'_BE_'+str(x)
            if nan_check(x): df[nm] = (df[col].isna()).astype('int8')
            else: df[nm] = (df[col]==x).astype('int8')
            n.append(nm)
            d[x] = 1
        ct += 1
        if (ct+1)>=vals: break
    if verbose==1:
        print('OHE encoded',col,'- Created',len(d),'booleans')
    return [n,d]

# ONE-HOT-ENCODING from dictionary
def encode_OHE_test(df,col,dt):
    n = []
    for x in dt: 
        n += encode_BE(df,col,x)
    return n

# BOOLEAN ENCODING
def encode_BE(df,col,val):
    n = col+"_BE_"+str(val)
    if nan_check(val):
        df[n] = df[col].isna()
    else:
        df[n] = df[col]==val
    df[n] = df[n].astype('int8')
    return [n]

# In[ ]:


cols = []; dd = []

# ENCODE NEW
for x in FE:
    cols += encode_FE(df_train,x)
for x in OHE:
    tmp = encode_OHE(df_train,x,0.005,5)
    cols += tmp[0]; dd.append(tmp[1])
print('Encoded',len(cols),'new variables')

# REMOVE OLD
for x in FE+OHE:
    del df_train[x]
print('Removed original',len(FE+OHE),'variables')
x = gc.collect()

# ## Example - Census_OEMModelIdentifier
# Below is variable `Census_OEMModelIdentifier`. Observe how NAN is treated like a category value and that it has consistently had the lowest HasDetections rate all of year 2018. Also notice how value=245824 has consistently been high. Finally note that value=188345 and 248045 are high and low respectively in August and September but earlier in the year their positions were reversed! What will their positions be in the test set's October and November computers??  
#   
# ![image](http://playagricola.com/Kaggle/OEM13019.png)

# # Build and Train Network
# We will a build a 3 layer fully connected network with 100 neurons on each hidden layer. We will use ReLU activation, Batch Normalization, 40% Dropout, Adam Optimizer, and Decaying Learning Rate. Unfortunately we don't have an AUC loss function, so we will use Cross Entrophy instead. After each epoch, we will call a custom Keras callback to display the current AUC and continually save the best model.

# In[ ]:


from keras import callbacks
from sklearn.metrics import roc_auc_score

class printAUC(callbacks.Callback):
    def __init__(self, X_train, y_train):
        super(printAUC, self).__init__()
        self.bestAUC = 0
        self.X_train = X_train
        self.y_train = y_train
        
    def on_epoch_end(self, epoch, logs={}):
        pred = self.model.predict(np.array(self.X_train))
        auc = roc_auc_score(self.y_train, pred)
        print("Train AUC: " + str(auc))
        pred = self.model.predict(self.validation_data[0])
        auc = roc_auc_score(self.validation_data[1], pred)
        print ("Validation AUC: " + str(auc))
        if (self.bestAUC < auc) :
            self.bestAUC = auc
            self.model.save("bestNet.h5", overwrite=True)
        return

# In[ ]:


from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Activation
from keras.callbacks import LearningRateScheduler
from keras.optimizers import Adam

#SPLIT TRAIN AND VALIDATION SET
X_train, X_val, Y_train, Y_val = train_test_split(
    df_train[cols], df_train['HasDetections'], test_size = 0.5)

# BUILD MODEL
model = Sequential()
model.add(Dense(100,input_dim=len(cols)))
model.add(Dropout(0.4))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(100))
model.add(Dropout(0.4))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer=Adam(lr=0.01), loss="binary_crossentropy", metrics=["accuracy"])
annealer = LearningRateScheduler(lambda x: 1e-2 * 0.95 ** x)

# TRAIN MODEL
model.fit(X_train,Y_train, batch_size=32, epochs = 20, callbacks=[annealer,
          printAUC(X_train, Y_train)], validation_data = (X_val,Y_val), verbose=2)

# # Predict Test and Submit to Kaggle
# Even after deleting the training data, our network still needs lot of our available RAM, we'll need to load in test.csv by chunks and predict by chunks. Click 'see code' button to see how this is done.

# In[ ]:


del df_train
del X_train, X_val, Y_train, Y_val
x = gc.collect()

# LOAD BEST SAVED NET
from keras.models import load_model
model = load_model('bestNet.h5')

pred = np.zeros((7853253,1))
id = 1
chunksize = 2000000
for df_test in pd.read_csv('../input/test.csv', 
            chunksize = chunksize, usecols=list(dtypes.keys())[0:-1], dtype=dtypes):
    print ('Loaded',len(df_test),'rows of TEST.CSV!')
    # ENCODE TEST
    cols = []
    for x in FE:
        cols += encode_FE(df_test,x,verbose=0)
    for x in range(len(OHE)):
        cols += encode_OHE_test(df_test,OHE[x],dd[x])
    # PREDICT TEST
    end = (id)*chunksize
    if end>7853253: end = 7853253
    pred[(id-1)*chunksize:end] = model.predict(df_test[cols])
    print('  encoded and predicted part',id)
    id += 1

# In[ ]:


# SUBMIT TO KAGGLE
df_test = pd.read_csv('../input/test.csv', usecols=['MachineIdentifier'])
df_test['HasDetections'] = pred
df_test.to_csv('submission.csv', index=False)

# ![image](http://playagricola.com/Kaggle/NN13019.png)

# # Conclusion
# In this kernel, we saw how to build and train a neural network with Keras. We also saw how to statistically one-hot-encode categorical variables. Our validation AUC was 0.703 and our LB was 0.671. So it appears that we are not time generalizing enough to the test set. Furthermore, other users are getting higher CV scores, so we should be able to improve our AUC by adding more variables and tuning our network more. 
# 
# If anyone forks this kernel and improves it's AUC, let me know. All comments and suggestions are welcome. Thanks for reading.
