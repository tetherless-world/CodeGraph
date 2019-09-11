#!/usr/bin/env python
# coding: utf-8

# **Digit Recognizer**
# 
# This competition involves training a neural network to be able to recognize handwritten digits.
# 
# The MNIST ("Modified National Institute of Standards and Technology") dataset contains a large number of handwritten digits, and is often used to train neural networks for vision-based tasks.
# 
# More information can be found on the competition page.
# https://www.kaggle.com/c/digit-recognizer/overview/description

# In[ ]:


# Let's keep all the imports we need here
# data analysis and wrangling
import pandas as pd
import numpy as np
# preprocessing
from keras.utils.np_utils import to_categorical
# model
from tensorflow.keras.models import Sequential
# layers
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Dropout, MaxPooling2D

# In[ ]:


# Suppress future warnings
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# In[ ]:


# Let's read in the data first
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

# Now that we've loaded in the data, we first want to see what exactly we will be working with.

# In[ ]:


print(train.shape)
train.head()

# From this we see that the first column is the label showing which digit the row represents, followed by 784 pixels.
# We also see that the training dataset contains 42,000 training instances.

# In[ ]:


# We know this is a multiclass classification problem, but let's see how many classes there are
train.label.unique()

# We have training data for the digits 0-9, giving as 10 total classes.
# 
# 

# **Data preprocessing**
# 
# Right now we have a row of pixels, but what they really represent is a square image. We need to do some preprocessing to get it into that shape. The [competition homepage](https://www.kaggle.com/c/digit-recognizer/data) explains how.
# 
# > Each image is 28 pixels in height and 28 pixels in width, for a total of 784 pixels in total. Each pixel has a single pixel-value associated with it, indicating the lightness or darkness of that pixel, with higher numbers meaning darker. This pixel-value is an integer between 0 and 255, inclusive.
# >
# >The training data set, (train.csv), has 785 columns. The first column, called "label", is the digit that was drawn by the user. The rest of the columns contain the pixel-values of the associated image.
# >
# >Each pixel column in the training set has a name like pixelx, where x is an integer between 0 and 783, inclusive. To locate this pixel on the image, suppose that we have decomposed x as x = i * 28 + j, where i and j are integers between 0 and 27, inclusive. Then pixelx is located on row i and column j of a 28 x 28 matrix, (indexing by zero).
# >
# >For example, pixel31 indicates the pixel that is in the fourth column from the left, and the second row from the top, as in the ascii-diagram below.
# >
# >Visually, if we omit the "pixel" prefix, the pixels make up the image like this:
# ```
# 000 001 002 003 ... 026 027
# 028 029 030 031 ... 054 055
# 056 057 058 059 ... 082 083
#  |   |   |   |  ...  |   |
# 728 729 730 731 ... 754 755
# 756 757 758 759 ... 782 783 
# ```
# >The test data set, (test.csv), is the same as the training set, except that it does not contain the "label" column.

# In[ ]:


# Let's separate the pixels from the labels
X_train = (train.iloc[:,1:].values).astype('float32')
y_train = train.iloc[:,0].values.astype('int32')
# The test set has no labels
X_test = test.values.astype('float32')

# In[ ]:


# Normalize the data so the range goes from 0-255, to 0-1
X_train = X_train / 255.0
X_test = X_test / 255.0

# In[ ]:


# Reshape the array so each image is 28 by 28 by 1
# The last dimension is the pixel value
X_train = X_train.reshape(-1, 28, 28,1)
X_test = X_test.reshape(-1,28,28,1)

# **One Hot Encondings**
# 
# Right now our labels are ordinal values. If left as is, our model would think we are trying to score each image from 0 to 9. 
# Since that is not the case, and we want the model to classify each image as one of ten classes, we will convert our labels using one hot enconding. 
# This will create a separate column for each class, 
# and each image will have zeroes in every column, except for one, 
# e.g., an image displaying the digit 0, would have zeroes in every column except the first one.

# In[ ]:


y_train = to_categorical(y_train)

# **Models**
# 
# Before we create any models, we need to split our training data into a training and validation set.
# 
# This is done so we can compare the performance of different models.

# In[ ]:


# Set aside some data for validation
from sklearn.model_selection import train_test_split
X = X_train
y = y_train
# Validation set will be 10% of the training data
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.10, random_state=42)

# We will start by creating a small, simple network to compare our later networks against.

# In[ ]:


model = Sequential()
model.add(Conv2D(28, kernel_size=(5, 5),
                 activation='relu',
                 input_shape=(28, 28, 1)))

model.add(Conv2D(32, kernel_size=(5, 5), activation='relu'))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X_train, y_train,
          batch_size=64,
          epochs=6,
          validation_data=(X_val, y_val))

# This gives us ~98% accuracy accuracy. Taking into account that individual runs can vary by about 0.5%, we can only expect some minimal improvements.
# 
# Ideally we would want to get at least 99% validation accuracy.
# 
# For the next network we will add a pooling layer as well as a dropout layer.

# In[ ]:


model2 = Sequential()
model2.add(Conv2D(28, kernel_size=(5, 5),
                 activation='relu',
                 input_shape=(28, 28, 1)))

model2.add(Conv2D(32, kernel_size=(5, 5), activation='relu'))
model2.add(MaxPooling2D(pool_size=(2, 2)))
model2.add(Dropout(0.25))
model2.add(Flatten())
model2.add(Dense(256, activation='relu'))
model2.add(Dropout(0.5))
model2.add(Dense(10, activation='softmax'))
model2.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model2.fit(X_train, y_train,
          batch_size=64,
          epochs=6,
          validation_data=(X_val, y_val))

# This gives us ~99% accuracy, which is good enough for our purposes.
# This model is still on the simple side, and while a more complex model might give us better accuracy, I don't think what would be at most 1% more accuracy would be worth the drop in performance from such a model.
# 
# The next step is to create the submission file to be submitted to the competition.

# In[ ]:


pred = model2.predict_classes(X_test, verbose=0)
submission = pd.DataFrame({"ImageId": list(range(1,len(pred)+1)),
                         "Label": pred})
submission.to_csv("submission.csv", index=False, header=True)
