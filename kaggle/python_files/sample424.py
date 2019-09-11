#!/usr/bin/env python
# coding: utf-8

# I wrote this kernel to mess around with the sketch data and share my experience so far.  After exploring the website and the data, I'll use a very basic CNN to classify sketches. This model gets 0.60 on the Public LB when run with 6000 recognized images per class.
# 
# ## Quick, Draw
# Ok - I have to say this is kind of a fun thing. You go to the [Quick Draw](https://quickdraw.withgoogle.com/#)  website and agree to sketch several common objects. The host then gives you the object labels one by one, and you have 20 seconds to draw each one. If the AI guesses your sketch (that is, associates it with training set items of the same label) you get a check mark and move on. At the end you get something like this:
# 
# <img src="https://s3.amazonaws.com/nonwebstorage/Screenshot+from+2018-09-26+22-44-21.png" alt="drawing1" width="600"/>
#   
# <p><br></p>
# Notice that it didn't like my bird. Apparently it looked more like a dragon or a diving board(??) or a mosquito.  They're nice enough to show you how other people draw birds so you maybe learn how to draw better.
# <img src="https://s3.amazonaws.com/nonwebstorage/Screenshot+from+2018-09-26+22-45-21.png" alt="drawing1" width="600"/>

# ## Training Images
# We can look at a few sketches and then see what the training data contains overall. I'll use an adaptation of Inversion's 'Getting Started' kernel.

# In[ ]:


import os
import re
from glob import glob
from tqdm import tqdm
import numpy as np
import pandas as pd
import ast
import matplotlib.pyplot as plt

# Here's what the training data looks like. This data frame is actually a concatenation of two rows from each of 6 csv files in the training set.

# In[ ]:


fnames = glob('../input/train_simplified/*.csv')
cnames = ['countrycode', 'drawing', 'key_id', 'recognized', 'timestamp', 'word']
drawlist = []
for f in fnames[0:6]:
    first = pd.read_csv(f, nrows=10) # make sure we get a recognized drawing
    first = first[first.recognized==True].head(2)
    drawlist.append(first)
draw_df = pd.DataFrame(np.concatenate(drawlist), columns=cnames)
draw_df

# And here are some nice sketches...

# In[ ]:


evens = range(0,11,2)
odds = range(1,12, 2)
df1 = draw_df[draw_df.index.isin(evens)]
df2 = draw_df[draw_df.index.isin(odds)]

example1s = [ast.literal_eval(pts) for pts in df1.drawing.values]
example2s = [ast.literal_eval(pts) for pts in df2.drawing.values]

# In[ ]:


labels = df2.word.tolist()
for i, example in enumerate(example1s):
    plt.figure(figsize=(6,3))
    
    for x,y in example:
        plt.subplot(1,2,1)
        plt.plot(x, y, marker='.')
        plt.axis('off')

    for x,y, in example2s[i]:
        plt.subplot(1,2,2)
        plt.plot(x, y, marker='.')
        plt.axis('off')
        label = labels[i]
        plt.title(label, fontsize=10)

    plt.show()  

# ## The Quick, Draw Data Repository
# There is a direct data source outside of Kaggle that seems pretty useful. The main link is a [GitHub Repository](https://github.com/googlecreativelab/quickdraw-dataset) that leads to the data in several formats, including Numpy bitmap files. Each file in the dataset covers a specific type of sketch and is in the shape of a long 1d array. Here are a few.
# 
# Update: The test set for this project doesn't have premade files in the numpy bitmap format. When you apply your model to test arrays converted from matplotlib, there is a loss due to the different conversion process. The best results for an image-based model are had by using the same processing for train and test (no surprise there I guess). So I'd say these are good for exploration but consider the other files for better results.

# In[ ]:


# # commented out to save memory
# import urllib

# LABELS = np.array(['baseball', 'bowtie', 'clock', 'hand', 'hat'])
# for b in LABELS:
#     url = "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/{}.npy".format(b)
#     urllib.request.urlretrieve(url, "{}.npy".format(b))
#     nb = np.load("{}.npy".format(b))
#     print("\n Class '{0}' has {1} examples of {2}x{2} images".format(b, nb.shape[0], int(nb.shape[1]**0.5)))

# ## Convolutional Neural Network (CNN)
# Next we'll build an image classifier. There are some resources in the repository I mentioned earlier showing how people have used the data. One of those resources is a CNN just like what you see here. 
# 
# The biggest usage of resources seems to be converting the drawings to images. You can stick with a stroke-based model or go down the conversion path. Going that way requires watching data usage and managing space limits - deep learning on Kaggle can be like building a ship in a bottle:) 
# 
# UPDATE: I'll be experimenting with a separate kernel to use a fit_generator like in Beluga's kernel and others. This kernel has the most I could squeeze into memory with simple fit and predict.

# In[ ]:



# In[ ]:


#%% import
import os
from glob import glob
import re
import ast
import numpy as np 
import pandas as pd
from PIL import Image, ImageDraw 
from tqdm import tqdm
from dask import bag

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.metrics import top_k_categorical_accuracy
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

# In[ ]:


#%% set label dictionary and params
classfiles = os.listdir('../input/train_simplified/')
numstonames = {i: v[:-4].replace(" ", "_") for i, v in enumerate(classfiles)} #adds underscores

num_classes = 340    #340 max 
imheight, imwidth = 32, 32  
ims_per_class = 2000  #max?

# In[ ]:


# faster conversion function
def draw_it(strokes):
    image = Image.new("P", (256,256), color=255)
    image_draw = ImageDraw.Draw(image)
    for stroke in ast.literal_eval(strokes):
        for i in range(len(stroke[0])-1):
            image_draw.line([stroke[0][i], 
                             stroke[1][i],
                             stroke[0][i+1], 
                             stroke[1][i+1]],
                            fill=0, width=5)
    image = image.resize((imheight, imwidth))
    return np.array(image)/255.

#%% get train arrays
train_grand = []
class_paths = glob('../input/train_simplified/*.csv')
for i,c in enumerate(tqdm(class_paths[0: num_classes])):
    train = pd.read_csv(c, usecols=['drawing', 'recognized'], nrows=ims_per_class*5//4)
    train = train[train.recognized == True].head(ims_per_class)
    imagebag = bag.from_sequence(train.drawing.values).map(draw_it) 
    trainarray = np.array(imagebag.compute())  # PARALLELIZE
    trainarray = np.reshape(trainarray, (ims_per_class, -1))    
    labelarray = np.full((train.shape[0], 1), i)
    trainarray = np.concatenate((labelarray, trainarray), axis=1)
    train_grand.append(trainarray)
    
train_grand = np.array([train_grand.pop() for i in np.arange(num_classes)]) #less memory than np.concatenate
train_grand = train_grand.reshape((-1, (imheight*imwidth+1)))

del trainarray
del train

# In[ ]:


# memory-friendly alternative to train_test_split?
valfrac = 0.1
cutpt = int(valfrac * train_grand.shape[0])

np.random.shuffle(train_grand)
y_train, X_train = train_grand[cutpt: , 0], train_grand[cutpt: , 1:]
y_val, X_val = train_grand[0:cutpt, 0], train_grand[0:cutpt, 1:] #validation set is recognized==True

del train_grand

y_train = keras.utils.to_categorical(y_train, num_classes)
X_train = X_train.reshape(X_train.shape[0], imheight, imwidth, 1)
y_val = keras.utils.to_categorical(y_val, num_classes)
X_val = X_val.reshape(X_val.shape[0], imheight, imwidth, 1)

print(y_train.shape, "\n",
      X_train.shape, "\n",
      y_val.shape, "\n",
      X_val.shape)

# Here's the architecture for the CNN. It's fairly simple compared to what else you can do.

# In[ ]:


model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu', input_shape=(imheight, imwidth, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(680, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
model.summary()

# In[ ]:


def top_3_accuracy(x,y): 
    t3 = top_k_categorical_accuracy(x,y, 3)
    return t3

reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, 
                                   verbose=1, mode='auto', min_delta=0.005, cooldown=5, min_lr=0.0001)
earlystop = EarlyStopping(monitor='val_top_3_accuracy', mode='max', patience=5) 
callbacks = [reduceLROnPlat, earlystop]

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy', top_3_accuracy])

model.fit(x=X_train, y=y_train,
          batch_size = 32,
          epochs = 22,
          validation_data = (X_val, y_val),
          callbacks = callbacks,
          verbose = 1)

# ## Predicting on the Test data
# The CNN does OK on the validation data, even with a basic model and limited training data. Let's generate predictions on the test set and submit.

# In[ ]:


#%% get test set
ttvlist = []
reader = pd.read_csv('../input/test_simplified.csv', index_col=['key_id'],
    chunksize=2048)
for chunk in tqdm(reader, total=55):
    imagebag = bag.from_sequence(chunk.drawing.values).map(draw_it)
    testarray = np.array(imagebag.compute())
    testarray = np.reshape(testarray, (testarray.shape[0], imheight, imwidth, 1))
    testpreds = model.predict(testarray, verbose=0)
    ttvs = np.argsort(-testpreds)[:, 0:3]  # top 3
    ttvlist.append(ttvs)
    
ttvarray = np.concatenate(ttvlist)

# In[ ]:


preds_df = pd.DataFrame({'first': ttvarray[:,0], 'second': ttvarray[:,1], 'third': ttvarray[:,2]})
preds_df = preds_df.replace(numstonames)
preds_df['words'] = preds_df['first'] + " " + preds_df['second'] + " " + preds_df['third']

sub = pd.read_csv('../input/sample_submission.csv', index_col=['key_id'])
sub['word'] = preds_df.words.values
sub.to_csv('subcnn_small.csv')
sub.head()

#  A full version with 6000 images per class at 28x28 gets just under 0.60 on the public LB.  

# In[ ]:


import sys

# These are the usual ipython objects, including this one you are creating
ipython_vars = ['In', 'Out', 'exit', 'quit', 'get_ipython', 'ipython_vars']

# Get a sorted list of the objects and their sizes
sorted([(x, sys.getsizeof(globals().get(x))) for x in dir() if not 
    x.startswith('_') and x not in sys.modules and x 
    not in ipython_vars], key=lambda x: x[1], reverse=True)

# In[ ]:



