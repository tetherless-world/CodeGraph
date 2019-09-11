#!/usr/bin/env python
# coding: utf-8

#  # <div style="text-align: center">Linear Algebra for Data Scientists 
# <div style="text-align: center">
# Having a basic knowledge of linear algebra is one of the requirements for any data scientist. In this tutorial we will try to cover all the necessary concepts related to linear algebra
# and also this is the third step of the <a href="https://www.kaggle.com/mjbahmani/10-steps-to-become-a-data-scientist">10 Steps to Become a Data Scientist</a>. and you can learn all of the thing you need for being a data scientist with Linear Algabra.</div> 
# 
# <div style="text-align:center">last update: <b>03/01/2019</b></div>
# 
# >###### You may  be interested have a look at 10 Steps to Become a Data Scientist: 
# 
# 1. [Leren Python](https://www.kaggle.com/mjbahmani/the-data-scientist-s-toolbox-tutorial-1)
# 2. [Python Packages](https://www.kaggle.com/mjbahmani/the-data-scientist-s-toolbox-tutorial-2)
# 3. <font color="red">You are in 3th step</font>
# 4. [Programming &amp; Analysis Tools](https://www.kaggle.com/mjbahmani/20-ml-algorithms-15-plot-for-beginners)
# 5. [Big Data](https://www.kaggle.com/mjbahmani/a-data-science-framework-for-quora)
# 6. [Data visualization](https://www.kaggle.com/mjbahmani/top-5-data-visualization-libraries-tutorial)
# 7. [Data Cleaning](https://www.kaggle.com/mjbahmani/machine-learning-workflow-for-house-prices)
# 8. [How to solve a Problem?](https://www.kaggle.com/mjbahmani/the-data-scientist-s-toolbox-tutorial-2)
# 9. [Machine Learning](https://www.kaggle.com/mjbahmani/a-comprehensive-ml-workflow-with-python)
# 10. [Deep Learning](https://www.kaggle.com/mjbahmani/top-5-deep-learning-frameworks-tutorial)
# 
# 
# 
# 
# 
# You can Fork code  and  Follow me on:
# 
# > ###### [ GitHub](https://github.com/mjbahmani/10-steps-to-become-a-data-scientist)
# > ###### [Kaggle](https://www.kaggle.com/mjbahmani/)
# -------------------------------------------------------------------------------------------------------------
#  <b>I hope you find this kernel helpful and some <font color='blue'>UPVOTES</font> would be very much appreciated.</b>
#     
#  -----------

#  <a id="top"></a> <br>
# ## Notebook  Content
# 1. [Introduction](#1)
# 1. [What is Linear Algebra?](#2)
# 1. [Notation ](#2)
# 1. [Matrix Multiplication](#3)
#     1. [Vector-Vector Products](#31)
#     1. [Outer Product of Two Vectors](#32)
#     1. [Matrix-Vector Products](#33)
#     1. [Matrix-Matrix Products](#34)
# 1. [Identity Matrix](#4)
# 1. [Diagonal Matrix](#5)
# 1. [Transpose of a Matrix](#6)
# 1. [Symmetric Metrices](#7)
# 1. [The Trace](#8)
# 1. [Norms](#9)
# 1. [Linear Independence and Rank](#10)
# 1. [Subtraction and Addition of Metrices](#11)
#     1. [Inverse](#111)
# 1. [Orthogonal Matrices](#12)
# 1. [Range and Nullspace of a Matrix](#13)
# 1. [Determinant](#14)
# 1. [Tensors](#16)
# 1. [Hyperplane](#17)
# 1. [Eigenvalues and Eigenvectors](#18)
# 1. [Exercise](#19)
# 1. [Conclusion](#21)
# 1. [References](#22)

# <a id="1"></a> <br>
# #  1-Introduction
# This is the third step of the [10 Steps to Become a Data Scientist](https://www.kaggle.com/mjbahmani/10-steps-to-become-a-data-scientist).
# we will cover following topic:
# 1. notation
# 1. Matrix Multiplication
# 1. Identity Matrix
# 1. Diagonal Matrix
# 1. Transpose of a Matrix
# 1. The Trace
# 1. Norms
# 1. Tensors
# 1. Hyperplane
# 1. Eigenvalues and Eigenvectors
# ## what is linear algebra?
# **Linear algebra** is the branch of mathematics that deals with **vector spaces**. good understanding of Linear Algebra is intrinsic to analyze Machine Learning algorithms, especially for **Deep Learning** where so much happens behind the curtain.you have my word that I will try to keep mathematical formulas & derivations out of this completely mathematical topic and I try to cover all of subject that you need as data scientist.[https://medium.com/@neuralnets](https://medium.com/@neuralnets/linear-algebra-for-data-science-revisiting-high-school-9a6bbeba19c6)
# <img src='https://camo.githubusercontent.com/e42ea0e40062cc1e339a6b90054bfbe62be64402/68747470733a2f2f63646e2e646973636f72646170702e636f6d2f6174746163686d656e74732f3339313937313830393536333530383733382f3434323635393336333534333331383532382f7363616c61722d766563746f722d6d61747269782d74656e736f722e706e67' height=200 width=700>
# 
#  <a id="top"></a> <br>

# <a id="11"></a> <br>
# ## 1-1 Import

# In[ ]:


import matplotlib.patches as patch
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy import linalg
from numpy import poly1d
from sklearn import svm
import tensorflow as tf
import pandas as pd
import numpy as np
import glob
import sys
import os

# <a id="12"></a> <br>
# ##  1-2 Setup

# In[ ]:


plt.style.use('ggplot')
np.set_printoptions(suppress=True)

# <a id="2"></a> <br>
# # 2- What is Linear Algebra?
# Linear algebra is the branch of mathematics concerning linear equations such as
# <img src='https://wikimedia.org/api/rest_v1/media/math/render/svg/f4f0f2986d54c01f3bccf464d266dfac923c80f3'>
# Linear algebra is central to almost all areas of mathematics. [6]
# <img src='https://upload.wikimedia.org/wikipedia/commons/thumb/2/2f/Linear_subspaces_with_shading.svg/800px-Linear_subspaces_with_shading.svg.png' height=400 width=400>
# [wikipedia](https://en.wikipedia.org/wiki/Linear_algebra#/media/File:Linear_subspaces_with_shading.svg)
# 

# In[ ]:


#3-dimensional vector in numpy
a = np.zeros((2, 3, 4))
#l = [[[ 0.,  0.,  0.,  0.],
    #      [ 0.,  0.,  0.,  0.],
     #     [ 0.,  0.,  0.,  0.]],
     #     [[ 0.,  0.,  0.,  0.],
    #      [ 0.,  0.,  0.,  0.],
     #     [ 0.,  0.,  0.,  0.]]]
a

# In[ ]:



# Declaring Vectors

x = [1, 2, 3]
y = [4, 5, 6]

print(type(x))

# This does'nt give the vector addition.
print(x + y)

# Vector addition using Numpy

z = np.add(x, y)
print(z)
print(type(z))

# Vector Cross Product
mul = np.cross(x, y)
print(mul)

# ## 2-1 What is Vectorization?
# In mathematics, especially in linear algebra and matrix theory, the vectorization of a matrix is a linear transformation which converts the matrix into a column vector. Specifically, the vectorization of an m × n matrix A, denoted vec(A), is the mn × 1 column vector obtained by stacking the columns of the matrix A on top of one another:
# <img src='https://wikimedia.org/api/rest_v1/media/math/render/svg/30ca6a8b796fd3a260ba3001d9875e990baad5ab'>
# [wikipedia](https://en.wikipedia.org/wiki/Vectorization_(mathematics) )

# Vectors of the length $n$ could be treated like points in $n$-dimensional space. One can calculate the distance between such points using measures like [Euclidean Distance](https://en.wikipedia.org/wiki/Euclidean_distance). The similarity of vectors could also be calculated using [Cosine Similarity](https://en.wikipedia.org/wiki/Cosine_similarity).
# ###### [Go to top](#top)

# <a id="3"></a> <br>
# ## 3- Notation
# <img src='http://s8.picofile.com/file/8349058626/la.png'>
# [linear.ups.edu](http://linear.ups.edu/html/notation.html)

# <a id="4"></a> <br>
# ## 4- Matrix Multiplication
# <img src='https://www.mathsisfun.com/algebra/images/matrix-multiply-constant.gif'>
# 
# [mathsisfun](https://www.mathsisfun.com/algebra/matrix-multiplying.html)

# The result of the multiplication of two matrixes $A \in \mathbb{R}^{m \times n}$ and $B \in \mathbb{R}^{n \times p}$ is the matrix:

# In[ ]:


# initializing matrices 
x = np.array([[1, 2], [4, 5]]) 
y = np.array([[7, 8], [9, 10]])

# $C = AB \in \mathbb{R}^{m \times n}$

# That is, we are multiplying the columns of $A$ with the rows of $B$:

# $C_{ij}=\sum_{k=1}^n{A_{ij}B_{kj}}$
# <img src='https://cdn.britannica.com/06/77706-004-31EE92F3.jpg'>
# [reference](https://cdn.britannica.com/06/77706-004-31EE92F3.jpg)

# The number of columns in $A$ must be equal to the number of rows in $B$.
# 
# ###### [Go to top](#top)

# In[ ]:


# using add() to add matrices 
print ("The element wise addition of matrix is : ") 
print (np.add(x,y)) 

# In[ ]:


# using subtract() to subtract matrices 
print ("The element wise subtraction of matrix is : ") 
print (np.subtract(x,y)) 

# In[ ]:


# using divide() to divide matrices 
print ("The element wise division of matrix is : ") 
print (np.divide(x,y)) 

# In[ ]:


# using multiply() to multiply matrices element wise 
print ("The element wise multiplication of matrix is : ") 
print (np.multiply(x,y))

# <a id="41"></a> <br>
# ## 4-1 Vector-Vector Products
# 
# numpy.cross(a, b, axisa=-1, axisb=-1, axisc=-1, axis=None)[source]
# Return the cross product of two (arrays of) vectors.[scipy](https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.cross.html)
# <img src='http://gamedevelopertips.com/wp-content/uploads/2017/11/image8.png'>
# [image-gredits](http://gamedevelopertips.com)

# In[ ]:


x = [1, 2, 3]
y = [4, 5, 6]
np.cross(x, y)

# We define the vectors $x$ and $y$ using *numpy*:

# In[ ]:


x = np.array([1, 2, 3, 4])
y = np.array([5, 6, 7, 8])
print("x:", x)
print("y:", y)

# We can now calculate the $dot$ or $inner product$ using the *dot* function of *numpy*:

# In[ ]:


np.dot(x, y)

# The order of the arguments is irrelevant:

# In[ ]:


np.dot(y, x)

# Note that both vectors are actually **row vectors** in the above code. We can transpose them to column vectors by using the *shape* property:

# In[ ]:


print("x:", x)
x.shape = (4, 1)
print("xT:", x)
print("y:", y)
y.shape = (4, 1)
print("yT:", y)

# In fact, in our understanding of Linear Algebra, we take the arrays above to represent **row vectors**. *Numpy* treates them differently.

# We see the issues when we try to transform the array objects. Usually, we can transform a row vector into a column vector in *numpy* by using the *T* method on vector or matrix objects:
# ###### [Go to top](#top)

# In[ ]:


x = np.array([1, 2, 3, 4])
y = np.array([5, 6, 7, 8])
print("x:", x)
print("y:", y)
print("xT:", x.T)
print("yT:", y.T)

# The problem here is that this does not do, what we expect it to do. It only works, if we declare the variables not to be arrays of numbers, but in fact a matrix:

# In[ ]:


x = np.array([[1, 2, 3, 4]])
y = np.array([[5, 6, 7, 8]])
print("x:", x)
print("y:", y)
print("xT:", x.T)
print("yT:", y.T)


# Note that the *numpy* functions *dot* and *outer* are not affected by this distinction. We can compute the dot product using the mathematical equation above in *numpy* using the new $x$ and $y$ row vectors:
# ###### [Go to top](#top)

# In[ ]:


print("x:", x)
print("y:", y.T)
np.dot(x, y.T)

# Or by reverting to:

# In[ ]:


print("x:", x.T)
print("y:", y)
np.dot(y, x.T)

# To read the result from this array of arrays, we would need to access the value this way:

# In[ ]:


np.dot(y, x.T)[0][0]

# <a id="42"></a> <br>
# ## 4-2 Outer Product of Two Vectors
# Compute the outer product of two vectors.

# In[ ]:


x = np.array([[1, 2, 3, 4]])
print("x:", x)
print("xT:", np.reshape(x, (4, 1)))
print("xT:", x.T)
print("xT:", x.transpose())

# Example
# ###### [Go to top](#top)

# We can now compute the **outer product** by multiplying the column vector $x$ with the row vector $y$:

# In[ ]:


x = np.array([[1, 2, 3, 4]])
y = np.array([[5, 6, 7, 8]])
x.T * y

# *Numpy* provides an *outer* function that does all that:

# In[ ]:


np.outer(x, y)

# Note, in this simple case using the simple arrays for the data structures of the vectors does not affect the result of the *outer* function:

# In[ ]:


x = np.array([1, 2, 3, 4])
y = np.array([5, 6, 7, 8])
np.outer(x, y)

# <a id="43"></a> <br>
# ## 4-3 Matrix-Vector Products
# Use numpy.dot or a.dot(b). See the documentation [here](http://docs.scipy.org/doc/numpy/reference/generated/numpy.dot.html).

# In[ ]:


a = np.array([[ 5, 1 ,3], [ 1, 1 ,1], [ 1, 2 ,1]])
b = np.array([1, 2, 3])
print (a.dot(b))

# Using *numpy* we can compute $Ax$:

# In[ ]:


A = np.array([[4, 5, 6],
             [7, 8, 9]])
x = np.array([1, 2, 3])
A.dot(x)

# <a id="44"></a> <br>
# ## 4-4 Matrix-Matrix Products

# In[ ]:


a = [[1, 0], [0, 1]]
b = [[4, 1], [2, 2]]
np.matmul(a, b)

# In[ ]:


matrix1 = np.matrix(a)
matrix2 = np.matrix(b)

# In[ ]:


matrix1 + matrix2

# In[ ]:


matrix1 - matrix2

# <a id="441"></a> <br>
# ### 4-4-1  Multiplication

# In[ ]:


np.dot(matrix1, matrix2)

# In[ ]:



matrix1 * matrix2

# <a id="5"></a> <br>
# ## 5- Identity Matrix

# numpy.identity(n, dtype=None)
# 
# Return the identity array.
# [source](https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.identity.html)

# In[ ]:


np.identity(3)

# How to create *identity matrix* in *numpy*  

# In[ ]:


identy = np.array([[21, 5, 7],[9, 8, 16]])
print("identy:", identy)

# In[ ]:


identy.shape

# In[ ]:


np.identity(identy.shape[1], dtype="int")

# In[ ]:


np.identity(identy.shape[0], dtype="int")

# <a id="51"></a> <br>
# ### 5-1  Inverse Matrices

# In[ ]:


inverse = np.linalg.inv(matrix1)
print(inverse)

# <a id="6"></a> <br>
# ## 6- Diagonal Matrix

# In *numpy* we can create a *diagonal matrix* from any given matrix using the *diag* function:

# In[ ]:


import numpy as np
A = np.array([[0,   1,  2,  3],
              [4,   5,  6,  7],
              [8,   9, 10, 11],
              [12, 13, 14, 15]])
np.diag(A)

# In[ ]:


np.diag(A, k=1)

# In[ ]:


np.diag(A, k=-1)

# <a id="7"></a> <br>
# ## 7- Transpose of a Matrix
# For reading about Transpose of a Matrix, you can visit [this link](https://py.checkio.org/en/mission/matrix-transpose/)

# In[ ]:


a = np.array([[1, 2], [3, 4]])
a

# In[ ]:


a.transpose()

# <a id="8"></a> <br>
# ## 8- Symmetric Matrices
# In linear algebra, a symmetric matrix is a square matrix that is equal to its transpose. Formally,
# <img src='https://wikimedia.org/api/rest_v1/media/math/render/svg/ad8a5a3a4c95de6f7f50b0a6fb592d115fe0e95f'>
# 
# [wikipedia](https://en.wikipedia.org/wiki/Symmetric_matrix)

# In[ ]:


N = 100
b = np.random.random_integers(-2000,2000,size=(N,N))
b_symm = (b + b.T)/2

# <a id="9"></a> <br>
# ## 9-The Trace
# Return the sum along diagonals of the array.

# In[ ]:


np.trace(np.eye(3))

# In[ ]:


print(np.trace(matrix1))

# In[ ]:


det = np.linalg.det(matrix1)
print(det)

# <a id="10"></a> <br>
# # 10- Norms
# numpy.linalg.norm
# This function is able to return one of eight different matrix norms, or one of an infinite number of vector norms (described below), depending on the value of the ord parameter. [scipy](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.linalg.norm.html)
# 
#  <a id="top"></a> <br>

# In[ ]:


v = np.array([1,2,3,4])
norm.median(v)

# <a id="11"></a> <br>
# # 11- Linear Independence and Rank
# How to identify the linearly independent rows from a matrix?

# In[ ]:


#How to find linearly independent rows from a matrix
matrix = np.array(
    [
        [0, 1 ,0 ,0],
        [0, 0, 1, 0],
        [0, 1, 1, 0],
        [1, 0, 0, 1]
    ])

lambdas, V =  np.linalg.eig(matrix.T)
# The linearly dependent row vectors 
print (matrix[lambdas == 0,:])

# <a id="12"></a> <br>
# # 12-  Subtraction and Addition of Metrices

# In[ ]:


import numpy as np
print("np.arange(9):", np.arange(9))
print("np.arange(9, 18):", np.arange(9, 18))
A = np.arange(9, 18).reshape((3, 3))
B = np.arange(9).reshape((3, 3))
print("A:", A)
print("B:", B)

# We can now add and subtract the two matrices $A$ and $B$:

# In[ ]:


A + B

# In[ ]:


A - B

# <a id="121"></a> <br>
# ## 12-1 Inverse
# We use numpy.linalg.inv() function to calculate the inverse of a matrix. The inverse of a matrix is such that if it is multiplied by the original matrix, it results in identity matrix.[tutorialspoint](https://www.tutorialspoint.com/numpy/numpy_inv.htm)

# In[ ]:


x = np.array([[1,2],[3,4]]) 
y = np.linalg.inv(x) 
print (x )
print (y )
print (np.dot(x,y))

# <a id="13"></a> <br>
# ## 13- Orthogonal Matrices
# How to create random orthonormal matrix in python numpy

# In[ ]:


## based on https://stackoverflow.com/questions/38426349/how-to-create-random-orthonormal-matrix-in-python-numpy
def rvs(dim=3):
     random_state = np.random
     H = np.eye(dim)
     D = np.ones((dim,))
     for n in range(1, dim):
         x = random_state.normal(size=(dim-n+1,))
         D[n-1] = np.sign(x[0])
         x[0] -= D[n-1]*np.sqrt((x*x).sum())
         # Householder transformation
         Hx = (np.eye(dim-n+1) - 2.*np.outer(x, x)/(x*x).sum())
         mat = np.eye(dim)
         mat[n-1:, n-1:] = Hx
         H = np.dot(H, mat)
         # Fix the last sign such that the determinant is 1
     D[-1] = (-1)**(1-(dim % 2))*D.prod()
     # Equivalent to np.dot(np.diag(D), H) but faster, apparently
     H = (D*H.T).T
     return H

# <a id="14"></a> <br>
# ## 14- Range and Nullspace of a Matrix

# In[ ]:


from scipy.linalg import null_space
A = np.array([[1, 1], [1, 1]])
ns = null_space(A)
ns * np.sign(ns[0,0])  # Remove the sign ambiguity of the vector

# <a id="15"></a> <br>
# # 15-  Determinant
# Compute the determinant of an array

# In[ ]:


a = np.array([[1, 2], [3, 4]])
np.linalg.det(a)

# <a id="16"></a> <br>
# # 16- Tensors

# A [**tensor**](https://en.wikipedia.org/wiki/Tensor) could be thought of as an organized multidimensional array of numerical values. A vector could be assumed to be a sub-class of a tensor. Rows of tensors extend alone the y-axis, columns along the x-axis. The **rank** of a scalar is 0, the rank of a **vector** is 1, the rank of a **matrix** is 2, the rank of a **tensor** is 3 or higher.
# 
# ###### [Go to top](#top)

# In[ ]:


# credits: https://www.tensorflow.org/api_docs/python/tf/Variable
A = tf.Variable(np.zeros((5, 5), dtype=np.float32), trainable=False)
new_part = tf.ones((2,3))
update_A = A[2:4,2:5].assign(new_part)
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
print(update_A.eval())

# <a id="25"></a> <br>
# # 17- Hyperplane

# The **hyperplane** is a sub-space in the ambient space with one dimension less. In a two-dimensional space the hyperplane is a line, in a three-dimensional space it is a two-dimensional plane, etc.

# Hyperplanes divide an $n$-dimensional space into sub-spaces that might represent clases in a machine learning algorithm.

# In[ ]:


##based on this address: https://stackoverflow.com/questions/46511017/plot-hyperplane-linear-svm-python
np.random.seed(0)
X = np.r_[np.random.randn(20, 2) - [2, 2], np.random.randn(20, 2) + [2, 2]]
Y = [0] * 20 + [1] * 20

fig, ax = plt.subplots()
clf2 = svm.LinearSVC(C=1).fit(X, Y)

# get the separating hyperplane
w = clf2.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(-5, 5)
yy = a * xx - (clf2.intercept_[0]) / w[1]

# create a mesh to plot in
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx2, yy2 = np.meshgrid(np.arange(x_min, x_max, .2),
                     np.arange(y_min, y_max, .2))
Z = clf2.predict(np.c_[xx2.ravel(), yy2.ravel()])

Z = Z.reshape(xx2.shape)
ax.contourf(xx2, yy2, Z, cmap=plt.cm.coolwarm, alpha=0.3)
ax.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.coolwarm, s=25)
ax.plot(xx,yy)

ax.axis([x_min, x_max,y_min, y_max])
plt.show()

# <a id="31"></a> <br>
# ## 20- Exercises
# let's do some exercise.

# ### 20-1 Create a dense meshgrid

# In[ ]:


np.mgrid[0:5,0:5]

# ### 20-2 Permute array dimensions

# In[ ]:


a=np.array([1,2,3])
b=np.array([(1+5j,2j,3j), (4j,5j,6j)])
c=np.array([[(1.5,2,3), (4,5,6)], [(3,2,1), (4,5,6)]])

# In[ ]:


np.transpose(b)

# In[ ]:


b.flatten()

# In[ ]:


np.hsplit(c,2)

# ## 20-3 Polynomials

# In[ ]:


p=poly1d([3,4,5])
p

# ## SciPy Cheat Sheet: Linear Algebra in Python
# This Python cheat sheet is a handy reference with code samples for doing linear algebra with SciPy and interacting with NumPy.
# 
# [DataCamp](https://www.datacamp.com/community/blog/python-scipy-cheat-sheet)

# <a id="21"></a> <br>
# # 21-Conclusion
# If you have made this far – give yourself a pat at the back. We have covered different aspects of **Linear algebra** in this Kernel. You are now finishing the **third step** of the course to continue, return to the [**main page**](https://www.kaggle.com/mjbahmani/10-steps-to-become-a-data-scientist/) of the course.  
# 
# ###### [Go to top](#top)

# you can follow me on:
# > ###### [ GitHub](https://github.com/mjbahmani/)
# > ###### [Kaggle](https://www.kaggle.com/mjbahmani/)
# 
#  <b>I hope you find this kernel helpful and some <font color='red'>UPVOTES</font> would be very much appreciated.<b/>
#  

# <a id="22"></a> <br>
# # 22-References & Credits
# 1. [Linear Algbra1](https://github.com/dcavar/python-tutorial-for-ipython)
# 1. [Linear Algbra2](https://www.oreilly.com/library/view/data-science-from/9781491901410/ch04.html)
# 1. [datacamp](https://www.datacamp.com/community/blog/python-scipy-cheat-sheet)
# 1. [damir.cavar](http://damir.cavar.me/)
# 1. [Linear_algebra](https://en.wikipedia.org/wiki/Linear_algebra)
# 1. [http://linear.ups.edu/html/fcla.html](http://linear.ups.edu/html/fcla.html)
# 1. [mathsisfun](https://www.mathsisfun.com/algebra/matrix-multiplying.html)
# 1. [scipy](https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.outer.html)
# 1. [tutorialspoint](https://www.tutorialspoint.com/numpy/numpy_inv.htm)
# 1. [machinelearningmastery](https://machinelearningmastery.com/introduction-to-tensors-for-machine-learning/)
# 1. [gamedevelopertips](http://gamedevelopertips.com/vector-in-game-development/)

# Go to first step: [Course Home Page](https://www.kaggle.com/mjbahmani/10-steps-to-become-a-data-scientist)
# 
# Go to next step : [Titanic](https://www.kaggle.com/mjbahmani/a-comprehensive-ml-workflow-with-python)
# 
