#!/usr/bin/env python
# coding: utf-8

# In[77]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical

import os
print(os.listdir("../input"))

# Import the data from BrainStation
mnist = np.genfromtxt('../input/mnist_data.csv', delimiter=",")
print('Shape: ', mnist.shape)

# In[78]:


# Functions 

def PlotBoundaries(model, X, Y) :
    '''
    Helper function that plots the decision boundaries of a model and data (X,Y)
    '''
    
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1,X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.4)

    #Plot
    plt.scatter(X[:, 0], X[:, 1], c=Y,s=20, edgecolor='k')
    plt.show()

# # Visualize

# In[79]:


# feature // target
X = mnist[:, :-1]
y = mnist[:, -1]
print('X shape: {}\ny shape: {}'.format(X.shape,y.shape))


#########################   Comment this line to run the full dataset   ##########################
X, X_holdout, y, y_holdout = train_test_split(X, y, test_size=0.96, stratify = y) ################
##################################################################################################


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, stratify = y)

# the images are in square form, so dim*dim = 784
from math import sqrt
dim = int(sqrt(X_train.shape[1]))
print('The images are {}x{} squares.'.format(dim, dim))
k = sns.countplot(mnist[:, -1], color = 'Cyan')

# Looks like our dataset has somewhat different number of examples of each digit. This could have implications later on...

# ## Example Digits

# In[80]:


plt.figure(figsize=(15,4.5))
for i in range(69):  
    plt.subplot(8, 10, i+1)
    plt.imshow(X_train[i].reshape((28,28)),cmap=plt.cm.binary)
    plt.axis('off')
plt.subplots_adjust(wspace=-0.1, hspace=-0.1)
plt.show()

# ## Average Digits

# In[81]:


m = X_train.shape[0]
n = X_train.shape[1]
labels = np.unique(y_train)
labels_count = labels.shape[0]

# Creating and plotting average digits
average_digits = np.empty((0, n+1))

plt.figure(figsize=(15,11))
plt.gray()

for label in labels:
    digits = X_train[y_train.flatten() == label]
    average_digit = digits.mean(0)   
    average_digits = np.vstack((average_digits, np.append(average_digit, label)))
    image = average_digit.reshape(28, 28)
    plt.subplot(8,10,label+1)
    plt.imshow(image)
    plt.title('Average '+str(label))
plt.show()

average_digits_x = average_digits[:,:-1]
average_digits_y = average_digits[:,-1]

# # Logistic Regression

# In[82]:


from sklearn.linear_model import LogisticRegression

# Create an instance of our model
my_logreg = LogisticRegression(solver = 'lbfgs')

start = time.time()

# Fit the model to data
my_logreg.fit(X_train,y_train)
print(my_logreg.score(X_test, y_test), 'Accuracy Score')
end = time.time()
print(end - start, ' Seconds to fit the logistic regression')
# X_train.shape

# ## Reducing Dimensionality w/ PCA

# PCA dimensionality reduction definitely cuts down on compute time by about 85-90%. Fewer dimensions means fewer calculations from fitting logistic regression, sacrificing performance for compute speed. Fewer datapoints obviously reduces computation time because the number of instances crowding up the CPU is smaller, again sacrificing performance in favor of faster compute time. 

# In[83]:


from sklearn.decomposition import PCA

#Build and fit a PCA model to the data
my_pca = PCA(n_components=2)
my_pca.fit(X)

#Transform the data
X_PCA = my_pca.transform(X)

# Split into training & test sets
PCA_X_train, PCA_X_test, PCA_y_train, PCA_y_test = train_test_split(X_PCA, y, test_size=0.4, stratify = y)

# In[84]:


# Create an instance of our model
my_PCA_logreg = LogisticRegression(solver = 'lbfgs')



start = time.time()

# Fit the model to data
my_PCA_logreg.fit(PCA_X_train,PCA_y_train)

end = time.time()
print(end - start, ' Seconds to fit the PCA logistic regression')
print(my_PCA_logreg.score(PCA_X_test, PCA_y_test), ' PCA score')

# # KNN

# In[85]:


from sklearn.neighbors import KNeighborsClassifier

# Instantiate the model & fit it to our data
start = time.time()

KNN_model_1 = KNeighborsClassifier(n_neighbors=1)
KNN_model_1.fit(X_train,y_train)
print(KNN_model_1.score(X_train,y_train), ' Train Score KNN(n_neighbors=1)')
print(KNN_model_1.score(X_test,y_test), ' Test Score KNN(n_neighbors=1)')
end = time.time()
print(end - start, ' Seconds to fit the KNN(n_neighbors=1)\n')

start = time.time()

KNN_model_n = KNeighborsClassifier(n_neighbors=len(X_train))
KNN_model_n.fit(X_train,y_train)
print(KNN_model_n.score(X_train,y_train), ' Train Score KNN(n_neighbors={})'.format(len(X_train)))
print(KNN_model_n.score(X_test,y_test), ' Test Score KNN(n_neighbors={})'.format(len(X_train)))
end = time.time()
print(end - start, ' Seconds to fit the KNN(n_neighbors={})'.format(len(X_train)))

# It would seem that with 1 neighbors our KNN model will OVERFIT the training set, and still has decent score for the test set. A small value of k means that noise will have a higher influence on the result.
# 
# Conversely, with MANY neighbors the model will seem to underfit, and has very low scores for the training set. A large value makes it computationally expensive. Although the difference between train-score and test-score is low, so it could be argued that this model is more generalizable than with 1 neighbor.

# In[86]:


test_scores = []
train_scores = []
K = []
start = time.time()
for i in range(1,70):
    KNN_model = KNeighborsClassifier(n_neighbors = i)
    KNN_model.fit(X_train, y_train)
    train_scores.append(KNN_model.score(X_train,y_train))
    test_scores.append(KNN_model.score(X_test,y_test))
    K.append(i)

plt.plot(K, train_scores, label= 'Train_scores')
plt.plot(K, test_scores, label = 'Test_scores')

plt.xlabel('n_Neighbors')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

end = time.time()
print(end - start, ' Seconds to fit {}x KNNs'.format(len(K)))
print('Best n-neighbors was: {}'.format(K[np.argmax(test_scores)]))

# From the figure above it would appear that while increasing the n_neighbors paramater, our accuracy scores for both the training and test set decline. This conforms to our previous experiment with k=1 being pretty accurate, and k=n being very inaccurate. Lower number of neighbors is better for predictive performance.
# 
# Any K between 3 and 10 will be good enough for this dataset. We could calculate the K with max accuracy, but it doesn't really matter because this model is not very good. And it will likely be similar to any value between 3 and 10 so might as well just pick one. The gains from picking a different model will be larger than if we try to tune this bad model.

# # Decision Tree

# In[87]:


from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

random_state = np.random.RandomState(69)
n_samples, n_features = X.shape

X_train, X_remainder, y_train, y_remainder = train_test_split(X, y, test_size=0.7, random_state=1, stratify = y)
X_validate, X_test, y_validate, y_test = train_test_split(X_remainder, y_remainder, test_size=0.5, random_state=2, stratify = y_remainder)

#Transform data
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
X_validate = scaler.transform(X_validate)

test_scores = []
validation_scores = []
train_scores = []

# C = [.00000001,.0000001,.000001,.00001,.0001,.001,.1,\
#                 1,10,100,1000,10000,100000,1000000,10000000,100000000,1000000000]

C= [1,2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100,1000]

start = time.time() # start timer

for c in np.array(C) :
#     my_regression = LogisticRegression(penalty='l2',C = c)
    my_regression = DecisionTreeClassifier(max_depth = c, random_state = 69)
    my_regression.fit(X_train,y_train);
    train_scores.append(my_regression.score(X_train,y_train))
    test_scores.append(my_regression.score(X_test,y_test))
    validation_scores.append(my_regression.score(X_validate,y_validate))

plt.plot(np.log10(C), train_scores,label="Train Score")
plt.plot(np.log10(C), test_scores,label="Test Score")
plt.plot(np.log10(C), validation_scores,label="Validation Score")

plt.legend();
plt.show();

end = time.time() # stop timer
print(end - start, ' Seconds')
print('Best Tree depths were: {} and {}'.format(C[np.argmax(validation_scores)], C[np.argmax(train_scores)]))

# Here it seems as we increase tree depth our train set accuracy keeps on rising. As we extend the tree depth our model overfits the training set. This could be a mistake on my part but, it doesn't seem as though increasing the tree depth past the optimal value has a significant effect on test/validation accruacy. Other than the fact that it is overfitting, the model will be less able to fit to the randomness of more such data. 

# # Cross Validation

# In[88]:


# Tune C-value (regularization)
C = []  # Store result to graph
validation_score_list = []
sample_range = [10**i for i in np.arange(-7,7,0.25)]
start = time.time() # start timer

#Do some cross validation
from sklearn.model_selection import cross_val_score
for i in sample_range :
    LR_model = LogisticRegression(penalty=l1, C=i)
    validation_score = np.mean(cross_val_score(LR_model, X, y, cv = 5))
    validation_score_list.append(validation_score)

plt.scatter(np.log10(sample_range), validation_score_list,label="Validation Score")
plt.legend()
plt.xlabel('Regularization Parameter: C')
plt.ylabel('Validation Score')
plt.show();

C_val = validation_score_list.index(np.max(validation_score_list))+1
print('Best C-value was: {}'.format(C_val))

end = time.time() # stop timer
print(end - start, ' Seconds')

# The C-value controls the regularization of the model. Low C-value equates to High regularization. A high C-value equates to Low Regularization. It looks to me like too much, or too little regularization is bad for performance. The sweet spot is somewhere in the middle GoldiLocks-Zone. As calculated above we have the best C-value for regularization

# We could also play with the loss functions. (L2 or L1) I found that L1 works better. 

# I wish I had more time to classify 4 and 9 and visualize the outputs. It would make sense to do something like a PCA or tSNE with something like a SVC kernel/linear. (but the runtime for that would be over 9000!) Also, I really don't enjoy using that PlotBoundaries() function from the notebooks... it's very picky and does not work too well.

# # Confusion Matrix

# In[89]:


from sklearn.metrics import confusion_matrix

logistic_model = LogisticRegression(solver = 'lbfgs')
logistic_model.fit(X_train, y_train)

y_pred = logistic_model.predict(X_test)
confusion = confusion_matrix(y_test, y_pred)
# print(confusion)
sns.heatmap(confusion, annot=True)

# Judging by the confusion matrix here our model seems to misclassify a good portion of the Five's. As we can see below the number of fives in the dataset is smaller. A popular misclassification for the fives was Three, which makes sense, these two digits are similar in shape. 

# In[97]:


k = sns.countplot(mnist[:, -1], color = 'Cyan')

# ### Strengths vs. Weaknesses of this model
# This model is very good at classifying Ones, Zeros and Sevens. I've looked at few of these examples that are INCREDIBLY close, where I thought it was a one but it turned out to be a seven. So, I mean some of these examples are so close between the two that, they might as well be classified as both!  

# # Conv Net

# In[90]:


from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler

# In[91]:


# Reshape for ImageDataGenerator
Y_train = mnist[:, -1]
X_train = mnist[:, :-1]
X_train = X_train / 255.0
X_train = X_train.reshape(-1,28,28,1)
Y_train = to_categorical(Y_train, num_classes = 10)

# In[92]:


datagen = ImageDataGenerator(
        rotation_range=8,  
        zoom_range = 0.1,  
        width_shift_range=0.1, 
        height_shift_range=0.1)

# ## Data Augmentation

# In[93]:


X_train3 = X_train[9,].reshape((1,28,28,1))
Y_train3 = Y_train[9,].reshape((1,10))
plt.figure(figsize=(15,4.5))
for i in range(69):  
    plt.subplot(8, 10, i+1)
    X_train2, Y_train2 = datagen.flow(X_train3,Y_train3).next()
    plt.imshow(X_train2[0].reshape((28,28)),cmap=plt.cm.binary)
    plt.axis('off')
    if i==9: X_train3 = X_train[11,].reshape((1,28,28,1))
    if i==19: X_train3 = X_train[18,].reshape((1,28,28,1))
    if i==29: X_train3 = X_train[24,].reshape((1,28,28,1))
    if i==39: X_train3 = X_train[31,].reshape((1,28,28,1))
    if i==49: X_train3 = X_train[32,].reshape((1,28,28,1))
    if i==59: X_train3 = X_train[45,].reshape((1,28,28,1))
    if i==69: X_train3 = X_train[52,].reshape((1,28,28,1))
plt.subplots_adjust(wspace=-0.1, hspace=-0.1)
plt.show()

# What we're seeing above is our data augmentation, so we're rotating, zooming, and shifting height/width in order to generate a larger train set to feed into the CNN. 

# In[94]:


# Define Keras ConvNet Model 
nets = 21
model = [0] *nets
for j in range(nets):
    model[j] = Sequential()

    model[j].add(Conv2D(32, kernel_size = 3, activation='relu', input_shape = (28, 28, 1)))
    model[j].add(BatchNormalization())
    model[j].add(Conv2D(32, kernel_size = 3, activation='relu'))
    model[j].add(BatchNormalization())
    model[j].add(Conv2D(32, kernel_size = 5, strides=2, padding='same', activation='relu'))
    model[j].add(BatchNormalization())
    model[j].add(Dropout(0.4))

    model[j].add(Conv2D(64, kernel_size = 3, activation='relu'))
    model[j].add(BatchNormalization())
    model[j].add(Conv2D(64, kernel_size = 3, activation='relu'))
    model[j].add(BatchNormalization())
    model[j].add(Conv2D(64, kernel_size = 5, strides=2, padding='same', activation='relu'))
    model[j].add(BatchNormalization())
    model[j].add(Dropout(0.4))

    model[j].add(Conv2D(128, kernel_size = 4, activation='relu'))
    model[j].add(BatchNormalization())
    model[j].add(Flatten())
    model[j].add(Dropout(0.4))
    model[j].add(Dense(10, activation='softmax'))

    model[j].compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
annealer = LearningRateScheduler(lambda x: 1e-3 * 0.95 ** x)

# In[95]:


# Run the model
epochs = 45
history = [0] * nets

for j in range(nets):
    X_train2, X_val2, Y_train2, Y_val2 = train_test_split(X_train, Y_train, test_size = 0.1)
    history[j] = model[j].fit_generator(datagen.flow(X_train2,Y_train2, batch_size=64),
        epochs = epochs, steps_per_epoch = X_train2.shape[0]//64,  
        validation_data = (X_val2,Y_val2), callbacks=[annealer], verbose=0)
    print("CNN {0:d}: Epochs={1:d}, Train accuracy={2:.5f}, Validation accuracy={3:.5f}".format(
        j+1,epochs,max(history[j].history['acc']),max(history[j].history['val_acc']) ))

# In[96]:


# Dump the model into a .pkl jar
import pickle

pickle_out = open("keagan_model.pkl","wb")
pickle.dump(model, pickle_out)
pickle_out.close()
