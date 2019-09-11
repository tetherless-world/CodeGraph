#!/usr/bin/env python
# coding: utf-8

# # FRUIT CLASSIFICATION
# At the moment of this work (05-2019) this dataset contains 103 class of different fruits and 53177 total images.
# My idea is to perform different classification algorithms, in particular SVM, K-NN, Decision Tree. Then I'll apply PCA in order to reduce dimensionaly of the dataset, see the distribution of the data and then try classification having only two dimension. 
# At the end I'll make a comparison between all methods in order to find which of them perform better on this dataset.
# 
# In the future I'd like to implement a CNN that almost surely will be a solution better than others, but for the moment I limit myself to the mentioned algorithms.

# In[ ]:


import numpy as np 
import cv2
import glob
import os
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import svm

print(os.listdir("../input"))
dim = 100

# In[ ]:


def getYourFruits(fruits, data_type):
    images = []
    labels = []
    path = "../input/*/fruits-360/" + data_type + "/"
    
    for i,f in enumerate(fruits):
        p = path + f
        for image_path in glob.glob(os.path.join(p, "*.jpg")):
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            image = cv2.resize(image, (dim, dim))
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            images.append(image)
            labels.append(i)
    images = np.array(images)
    labels = np.array(labels)
    return images, labels

def getAllFruits():
    fruits = []
    for fruit_path in glob.glob("../input/*/fruits-360/Training/*"):
        fruit = fruit_path.split("/")[-1]
        fruits.append(fruit)
    return fruits
    

# # CHOOSE YOUR CLASS
# Cause there are too many classes and each algorithm takes too many time, I've decided to take only N fruits editable in the list above.<br> For those wishing to try the whole dataset, just call *getAllFruits()*

# In[ ]:


#Choose your Fruits

#fruits = getAllFruits()
#fruits = ['Orange', 'Banana' , 'Strawberry', 'Apple Golden 1', 'Kiwi' , 'Lemon', 'Cocos' , 'Pineapple' , 'Peach', 'Cherry 1', 'Cherry 2', 'Mandarine']
#fruits = ['Orange', 'Cocos', 'Banana', 'Strawberry']
fruits = ['Lemon', 'Mandarine' , 'Cocos']
#fruits = ['Mango' , 'Apricot']

#Get Images and Labels
X, y =  getYourFruits(fruits, 'Training')
X_test, y_test = getYourFruits(fruits, 'Test')

#Scale Data Images
scaler = StandardScaler()
X_train = scaler.fit_transform([i.flatten() for i in X])
X_test = scaler.fit_transform([i.flatten() for i in X_test])

# Each image is converted in a 100x100 numpy array for each RGB dimension (x3). Then has been scaled and flatted in one single dimension (100x100x3) 

# # VISUALIZATION OF DATA
# Let's see now how one of our samples appears

# In[ ]:


sample = 10
plt.imshow(X[sample])
print(fruits[y[sample]])

# In[ ]:


def getClassNumber(y):
    v =[]
    i=0
    count = 0
    for index in y:
        if(index == i):
            count +=1
        else:
            v.append(count)
            count = 1
            i +=1
    v.append(count)        
    return v

def plotDataDistribution(X, dim):
    v = getClassNumber(y)
    colors = 'b', 'g', 'r', 'c', 'm', 'y', 'k', 'grey', 'orange', 'purple'
    markers = ['o', 'x' , 'v', 'd']
    tot = len(X)
    start = 0 
    if(dim == 2):
        for i,index in enumerate(v):
            end = start + index
            plt.scatter(X[start:end,0],X[start:end,1] , color=colors[i%len(colors)], marker=markers[i%len(markers)], label = fruits[i])
            start = end
        plt.xlabel('PC1')
        plt.ylabel('PC2')
    
    if(dim == 3):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for i,index in enumerate(v):
            end = start + index
            ax.scatter(X[start:end,0], X[start:end,1], X[start:end,2], color=colors[i%len(colors)], marker=markers[i%len(markers)], label = fruits[i])
            start = end
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')


    plt.legend(loc='lower left')
    plt.xticks()
    plt.yticks()
    plt.show()     

# ## DATA DISTRIBUTION 
# In order to discover how our data appears we need to reduce dimensionality of the dataset in 2 or 3 dimension so that we can plot and visualize them. To do this I've decided to use Principal Component Analysis, explained in the next chapter, but a better solution could be use t-SNE (T-distributed Stochastic Neighbor Embedding) a nonlinear dimensionality reduction technique well-suited for embedding high-dimensional data for visualization in a low-dimensional space of two or three dimensions. 

# ### DATA IN 2D

# In[ ]:


pca = PCA(n_components=2)
dataIn2D = pca.fit_transform(X_train)
plotDataDistribution(dataIn2D, 2)

# ### DATA IN 3D

# In[ ]:


pca = PCA(n_components=3)
dataIn3D = pca.fit_transform(X_train)
plotDataDistribution(dataIn3D, 3)

# # PRINCIPAL COMPONENT ANALYSIS
# Principal Component Analysis is a technique used in order to reduce the dimensionality of a dataset while preserving as mush information as possible. Data is reprojected in a lower dimensional space, in particular we need to find a projection that minimizes squared error in reconstructing the original data. <br>
# There are 3 different technique in order to apply PCA.
# 
# ![Principal Component Analysis](https://www.analyticsvidhya.com/wp-content/uploads/2016/03/2-1-e1458494877196.png)
# 
# 1. **Sequential**  
# 2. **Sample Covariance Matrix**
# 3. **Singular Value Decomposition (SVD)** <br>
# 
# I'll explain the Sample Covariance Matrix technique:
# * The first thing to do is to standardize the data, so for each sample we need to substract the mean of the full dataset and then divide it by the variance, so as having an unitary variance for each istance. This last process is not completly necessary but it is usefull to let the CPU work less.
# $$
#     Z = \frac{X-\mu}{\sigma^2} 
# $$
# <br>
# * Then we need to compute Covariance Matrix, given data { $x_1 ,x_2, ..., x_n$ } with $n$ number of samples, covariance matrix is obtained by:<br><br>
# $$
# \Sigma = \frac {1}{n}\sum_{i=1}^n (x_i - \bar{x})(x - \bar{x})^T $$    $\;\;$  where  $$\bar{x} = \frac {1}{n}\sum_{i=i}^n x_i $$ <br>
# Or simply by multiplying the standardized matrix Z by it self transposed<br>
# $$ COV(X) = Z Z^T $$<br>
# 
# * Principal Components will be the eigenvectors of the Covariance Matrix sorted in order of importance by the respective eigenvalues.<br>**Larger eigenvalues $\Rightarrow$ more important eigenvectors.**<br> They represent the most of the useful information on the entire dataset in a single vector

# In[ ]:


def showPCA(image,X2, X10, X50):
    fig = plt.figure(figsize=(30,10))
    ax1 = fig.add_subplot(1,4,1)
    ax1.axis('off')
    ax1.set_title('Original image')
    plt.imshow(image)
    ax1 = fig.add_subplot(1,4,2)
    ax1.axis('off') 
    ax1.set_title('PCA 50')
    plt.imshow(X50)
    ax1 = fig.add_subplot(1,4,3)
    ax1.axis('off') 
    ax1.set_title('PCA 10')
    plt.imshow(X10)
    ax2 = fig.add_subplot(1,4,4)
    ax2.axis('off') 
    ax2.set_title('PCA 2')
    plt.imshow(X2)
    plt.show()

def computePCA(n, im_scaled, image_id):
    pca = PCA(n)
    principalComponents = pca.fit_transform(im_scaled)
    im_reduced = pca.inverse_transform(principalComponents)
    newImage = scaler.inverse_transform(im_reduced[image_id])
    return newImage

# ## PCA EXAMPLE

# In[ ]:


image_id = 5
image = X[image_id]

#Compute PCA
X_2 = computePCA(2, X_train,image_id)
X_10 = computePCA(10, X_train,image_id)
X_50 = computePCA(50, X_train,image_id)

#Reshape in order to plot images
X2 = np.reshape(X_2, (dim,dim,3)).astype(int)
X10 = np.reshape(X_10, (dim,dim,3)).astype(int)
X50 = np.reshape(X_50, (dim,dim,3)).astype(int)

#Plot
showPCA(image, X2, X10, X50)

# ### COMMENT 
# From those images is possible to understand how considering only a sample and obtaining its principal components from the whole dataset, it's possible for us to classify easly the fruit just considering only 2 dimension instead of all. This means a lot of data less.<br>
# Obviously for a classification algorithm accuracy of the classification will be lower but instead training time will be faster.

# <br>
# # SUPPORT VECTOR MACHINES
# A Support Vector Machine (SVM) is a supervised classification method, that after a training
# phase can identify if a new point belongs to a class or another with the highest
# mathematically accuracy.
# It's a binary classification method, but using an approach called One vs All is possible to using SVM for multi-class classification.
# 
# If a dataset is linearly separable it means that we could use a Hard margin approach, or
# rather find the two parallel hyperplanes that separate the two classes of data, so that the
# distance between them is as large as possible and so we are being able to identify to which class
# belongs each point of the dataset. But mostly of the time datasets are not linearly
# separable and so we can take two ways, one is to continue using a Linear approach using Soft Margin, or simplifying, admitting some misclassification. While the second way is to use a Non-Linear Kernel (that must satisfy Mercer Condition) or rather a mapping of the data on a higher dimensional space, where the data is linearly separable and the classification task can be solved easly,without even need to calculate the points projections.
# 
# 
# General solution | Optimal solution
# - | - 
# ![](https://cdn-images-1.medium.com/max/720/0*9jEWNXTAao7phK-5.png) | ![](https://cdn-images-1.medium.com/max/720/0*0o8xIA4k3gXUDCFU.png)
# 
# <br>
# If we want to use a Linear Soft-Margin approach the optimization task is to find a  margin that should be as big as possible and we need then to add a penalty if a point is misclassified.
# This is made by adding to the optimization problem another term of Loss, regularizated by a slack variable C. This term will say how well we want to fit our data and how many mistakes we grant to do. The mostly used Loss function for this kind of problem is the Hinge Loss.<br>
# 
# $$ L(y, f(x)) = \max ( 0 , 1 - y · f(x) ) $$
# 
# ![Hinge Loss Function](https://i.stack.imgur.com/Ifeze.png)
# 
# <br>
# The optimization problem will be: 
# 
# $$ minimize \frac{1}{2} ||{w}||^2 + C \sum_{i=1}^n \xi_i  
#    \;\;\;,\;subject\;to\;\;\;y_i[x_i·w + b] \geq 1 - \xi_i $$
# 
# <br> Where $\frac{1}{||w||}$ is margin size, C our hyperparameter and $\xi_i$ the distance from the point to the boundarie line.<br>
# 
# At the end we need to find the right trade-off between margin and penalty.
# So, the only parameter that we can choose is C because the optimization problem and the calculus are committed to the calculators.<br> Kernel type could be set to linear if the problem is linearly separable, otherwise a Gaussian kernel (RBF) fit well most of the time. Gamma is a kernel hyperparameter that tries to exactly fit the training data.

# In[ ]:


#SVM 
model = svm.SVC(gamma='auto', kernel='linear')
model = model.fit(X_train, y) 
test_predictions = model.predict(X_test)
precision = metrics.accuracy_score(test_predictions, y_test) * 100
print("Accuracy with SVM: {0:.2f}%".format(precision))

# In[ ]:


#SVM + PCA
pca = PCA(n_components=2)
X_train2D = pca.fit_transform(X_train)
X_test2D = pca.fit_transform(X_test)

model.fit(X_train2D, y) 
test_predictions = model.predict(X_test2D)
precision = metrics.accuracy_score(test_predictions, y_test) * 100
print("Accuracy with SVM considering only first 2PC: {0:.2f}%".format(precision))

#Plotting decision boundaries
plot_decision_regions(X_train2D, y, clf=model, legend=1)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('SVM Decision Boundaries')
plt.show()

# <br> 
# # K-NEAREST NEIGHBOR
# 
# K-NN is a supervised learning method that considers the K closest training examples to the point of interest for predicting its class. The point is assigned to the class that is closest. <br>
# Could be applied different distance metrics sush as: Euclidian, Weighted, Gaussian, or what else. Steps are pretty easy:<br>
# 
# *  Receive an unclassified data
# 
# *  Measure the distance with choosen metrics from the new data to all others data that are already classified.
# 
# *  Gets the K smaller distances
# 
# *  Check the list of classes that had the shortest distance and count the amount of each class that appears
# 
# *  Takes as correct class the class that appeared the most times
# 
# *  Classifies the new data with the class that you took in previous step 
# <br>

# In[ ]:


knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y)
y_pred = knn.predict(X_test)
precision = metrics.accuracy_score(y_pred, y_test) * 100
print("Accuracy with K-NN: {0:.2f}%".format(precision))

# In[ ]:


#K-NN + PCA
knn.fit(X_train2D, y)
y_pred = knn.predict(X_test2D)
precision = metrics.accuracy_score(y_pred, y_test) * 100
print("Accuracy with K-NN considering only first 2PC: {0:.2f}%".format(precision))

#Plotting decision boundaries
plot_decision_regions(X_train2D, y, clf=knn, legend=1)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('K-NN Decision Boundaries')
plt.show()

# # DECISION TREE
# In a decision tree each intermediate node of the tree contains splitting attributes used to build different paths, while leaves contains class labels.
# 
# There are differt algorithm to build a decision tree, but all are made with a greedy approach, optimal locally, because number of partitions has a factorial growth.<br>The most famous is **Hunt's algoritm**. <br>
# 
# ![](https://cdn-images-1.medium.com/max/880/0*QctkHiOX2G2pvfD_.jpg)
# 
# Strarting from an empty tree, we need to find iteratively best attribute on which split the data locally at each step. If a subset contains records thtat belongs to the same class then the leaf containing such class label is created, otherwise if a subset is empty is assigned to default to mayor class.
# 
# Critical points of decision trees are test condition, the selection of the best attribute and the splitting condition. 
# For the selection of the best attribute is generally choosen the attribute that generate homogeneus nodes. 
# There are differt metrics in order to find the best splitting homogenity, the most common are:
# * GINI IMPURITY INDEX: Given **$n$** classes and $p_i$ the fraction of items of class $i$ in a subset p, for $i$∈{1,2,...,n}. Then the GINI index is defined as: $$ GINI = 1 − \sum_{i=1}^n p_i^2 $$
# 
# * INFORMATION GAIN RATIO: The information gain is based on the decrease of entropy after a data-set is split on an attribute. Constructing a decision tree is all about finding attribute that returns the highest information gain (i.e., the most homogeneous branches).<br>Entropy is defined as $H(i) = -\sum_{i=1}^n p_i\log_2 p_i $.<br> So then Information gain is defined as: 
# 
# $$ IG = H(p) - H(p,i)  = H(p) - \sum_{i=1}^n \frac{n_i}{n} H(i) $$
# 
# where p is the parent node.
# Advantages of Decision Trees are velocity, easy to interpretate and good accuracy, but they could be affected by missing data

# In[ ]:


tree = DecisionTreeClassifier()
tree = tree.fit(X_train,y)
y_pred = tree.predict(X_test)
precision = metrics.accuracy_score(y_pred, y_test) * 100
print("Accuracy with Decision Tree: {0:.2f}%".format(precision))

# In[ ]:


#DECISION TREE + PCA
tree = tree.fit(X_train2D,y)
y_pred = tree.predict(X_test2D)
precision = metrics.accuracy_score(y_pred, y_test) * 100
print("Accuracy with Decision Tree considering only first 2PC: {0:.2f}%".format(precision))

#Plotting decision boundaries
plot_decision_regions(X_train2D, y, clf=tree, legend=1)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Decision Tree Decision Boundaries')
plt.show()
